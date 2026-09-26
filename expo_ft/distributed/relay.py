"""Optional SSH routing adapter; run on a login/relay host, outside the models.

Writes a JSON endpoint list for Transport's `peers` field. Every listener uses
an independent SSH TCP transport, even if the user's SSH config multiplexes.
This command never copies keys or changes persistent SSH configuration.
"""
import argparse
import json
import logging
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time


def forwarding_commands(config):
    """Choose stable endpoints once; every restart reuses the exact command."""
    endpoints, reverse_endpoints, commands = [], [], []
    selected = set()
    count = config.get('connections', 64)
    if not 1 <= count <= 64:
        raise ValueError('connections must be 1..64')
    for i in range(count):
        host = config.get('listen_host', '127.0.0.1')
        requested_port = config['first_port'] + i if 'first_port' in config else 0
        if requested_port and not 1024 <= requested_port <= 65535:
            raise ValueError('local port outside nonprivileged range')
        while True:
            with socket.socket() as probe:
                # A stopped forward may leave TIME_WAIT connections. Match
                # SSH's listener reuse without allowing a second live listener.
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind((host, requested_port))
                port = probe.getsockname()[1]
            if (host, port) not in selected:
                selected.add((host, port))
                break
        command = ['ssh', '-F', config['ssh_config'], '-o', 'BatchMode=yes',
                   '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-o', 'ForwardAgent=no',
                   '-o', 'Compression=no', '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=10',
                   '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
                   '-N', '-L', '{}:{}:{}:{}'.format(host, port, *config['destination'])]
        if config.get('reverse'):
            reverse = config['reverse']
            remote_port = reverse['first_port'] + i
            if not 1024 <= remote_port <= 65535:
                raise ValueError('reverse port outside nonprivileged range')
            command += ['-R', '{}:{}:{}:{}'.format(reverse.get('listen_host', '127.0.0.1'),
                                                   remote_port, *reverse['destination'])]
            reverse_endpoints.append([reverse.get('advertise_host', reverse.get('listen_host', '127.0.0.1')), remote_port])
        command.append(config['ssh_host'])
        commands.append((command, host, port))
        endpoints.append([config.get('advertise_host', host), port])
    return commands, {'peers': endpoints, 'reverse_peers': reverse_endpoints}


def maintain_tunnel(command, host, port, stop):
    """Recover one SSH child; no wait/cleanup here touches another child."""
    delay = 1.0
    while not stop.is_set():
        process = None
        try:
            process = subprocess.Popen(command)
            deadline = time.monotonic() + 20
            while not stop.is_set():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('SSH relay failed to start')
                try:
                    with socket.create_connection((host, port), timeout=.1):
                        break
                except OSError:
                    stop.wait(.1)
            if stop.is_set():
                break
            logging.info('SSH relay listening on %s:%s', host, port)
            ready = time.monotonic()
            while not stop.wait(.2):
                if process.poll() is not None:
                    if time.monotonic() - ready >= 30:
                        delay = 1.0
                    raise RuntimeError('SSH relay disconnected')
        except (OSError, RuntimeError) as exc:
            if not stop.is_set():
                logging.warning('SSH relay %s:%s: %s; retry in %.1fs', host, port, exc, delay)
        finally:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if stop.wait(delay):
            break
        delay = min(30, delay * 2)


def run(config, stop):
    commands, endpoints = forwarding_commands(config)
    workers = []
    try:
        for command, host, port in commands:
            worker = threading.Thread(target=maintain_tunnel, args=(command, host, port, stop))
            worker.start()
            workers.append(worker)
        # Endpoint identities are available even if some links are still starting.
        # This file describes routing, not a guarantee that every link is up.
        output = Path(config['output'])
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(endpoints, indent=2))
        temp.replace(output)
        stop.wait()
    finally:
        stop.set()
        for worker in workers:
            worker.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    run(json.loads(Path(args.config).read_text()), stop)


if __name__ == '__main__':
    main()
