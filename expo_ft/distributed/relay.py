"""Optional SSH routing adapter; run on a login/relay host, outside the models.

Writes a JSON endpoint list for Transport's `peers` field. Every listener uses
an independent SSH TCP transport, even if the user's SSH config multiplexes.
This command never copies keys or changes persistent SSH configuration.
`startup_concurrency` (default 2) and `startup_interval` (default .5 seconds)
bound initial connections and manual recovery attempts. No SSH child is ever
automatically restarted. Use only during active jobs/transfers, then quit.
"""
import argparse
import curses
import json
import logging
import math
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import threading
import time


def forwarding_commands(config):
    """Choose stable endpoints once; every restart reuses the exact command."""
    endpoints, reverse_endpoints, commands = [], [], []
    selected = set()
    count = config.get('max_connections', config.get('connections', 32))
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
                   '-o', 'ConnectionAttempts=1',
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


class StartupGate:
    """Bound SSH handshakes, including retries, without limiting live tunnels."""

    def __init__(self, limit=2, interval=.5):
        if not isinstance(limit, int) or not 1 <= limit <= 64:
            raise ValueError('startup_concurrency must be 1..64')
        if not math.isfinite(interval) or interval < 0:
            raise ValueError('startup_interval must be finite and nonnegative')
        self.slots = threading.BoundedSemaphore(limit)
        self.lock = threading.Lock()
        self.interval = interval
        self.next_start = 0.0
        self.failure_delay = 5.0

    def acquire(self, stop, cancel=None):
        def halted():
            return stop.is_set() or (cancel is not None and cancel.is_set())
        while not halted():
            if not self.slots.acquire(timeout=.1):
                continue
            while not halted():
                with self.lock:
                    remaining = self.next_start - time.monotonic()
                    if remaining <= 0:
                        self.next_start = time.monotonic() + self.interval
                        return True
                stop.wait(min(remaining, .1))
            self.slots.release()
        return False

    def release(self, failed=False):
        with self.lock:
            if failed:
                # A shared cooldown also slows fresh workers during an outage.
                self.next_start = max(self.next_start, time.monotonic() +
                                      random.uniform(self.failure_delay, self.failure_delay * 1.5))
                self.failure_delay = min(30.0, self.failure_delay * 2)
            else:
                self.failure_delay = 5.0
        self.slots.release()


class TunnelPool:
    """One initial attempt per route; subsequent attempts require a user request."""

    def __init__(self, commands, stop, gate, initial=None):
        self.stop, self.gate = stop, gate
        self.initial = len(commands) if initial is None else initial
        if not 0 <= self.initial <= len(commands):
            raise ValueError("initial connections exceed capacity")
        self.disabled = [threading.Event() for _ in commands]
        self.lock = threading.Lock()
        self.rows = [dict(index=i, host=host, port=port, state='queued' if i < self.initial else 'down', attempts=0,
                          error='', pid=None, changed=time.monotonic())
                     for i, (_, host, port) in enumerate(commands)]
        self.requests = [threading.Event() for _ in commands]
        self.workers = [threading.Thread(target=self._worker, args=(i, command, host, port))
                        for i, (command, host, port) in enumerate(commands)]

    def snapshot(self):
        with self.lock:
            return [dict(row) for row in self.rows]

    def _update(self, i, **values):
        with self.lock:
            self.rows[i].update(values, changed=time.monotonic())

    def start(self):
        for i, (request, worker) in enumerate(zip(self.requests, self.workers)):
            if i < self.initial:
                request.set()
            worker.start()

    def add_one(self):
        """Start one unused/down slot; never reconnect automatically."""
        with self.lock:
            failed = [row for row in self.rows if row['state'] == 'down']
            if self.stop.is_set() or not failed:
                return None
            row = min(failed, key=lambda item: item['changed'])
            row.update(state='queued', changed=time.monotonic())
            self.disabled[row['index']].clear()
            self.requests[row['index']].set()
            return row['index']

    def remove_one(self):
        """Remove one live/queued slot, leaving every other tunnel untouched."""
        with self.lock:
            candidates = [r for r in self.rows if r['state'] in ('up', 'connecting', 'queued')]
            if not candidates:
                return None
            row = candidates[-1]
            row.update(state='closing', changed=time.monotonic())
            self.disabled[row['index']].set()
            return row['index']

    def _stderr(self, i, stream):
        # Drain continuously; never retain an unbounded SSH log in RAM.
        for line in stream:
            line = line.strip()
            if line:
                self._update(i, error=line[-240:])
                logging.warning('SSH tunnel %d: %s', i, line)

    @staticmethod
    def _cleanup(process):
        # A ProxyJump child may outlive its failed parent. Own a separate process
        # group and terminate that group only, even if the top-level SSH exited.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    def _worker(self, i, command, host, port):
        request = self.requests[i]
        disabled = self.disabled[i]
        while not self.stop.is_set():
            if not request.wait(.1):
                continue
            request.clear()
            process = reader = None
            starting = False
            error = ''
            try:
                if not self.gate.acquire(self.stop, disabled):
                    if self.stop.is_set():
                        break
                    continue
                starting = True
                if self.stop.is_set() or disabled.is_set():
                    continue
                self._update(i, state='connecting', error='', attempts=self.rows[i]['attempts'] + 1)
                process = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE, universal_newlines=True, start_new_session=True)
                self._update(i, pid=process.pid)
                reader = threading.Thread(target=self._stderr, args=(i, process.stderr), daemon=True)
                reader.start()
                deadline = time.monotonic() + 20
                while not self.stop.is_set() and not disabled.is_set():
                    if process.poll() is not None:
                        raise RuntimeError('SSH exited (%s)' % process.returncode)
                    if time.monotonic() > deadline:
                        raise RuntimeError('SSH listener startup timed out')
                    try:
                        with socket.create_connection((host, port), timeout=.1):
                            break
                    except OSError:
                        self.stop.wait(.1)
                if self.stop.is_set() or disabled.is_set():
                    continue
                self.gate.release()
                starting = False
                self._update(i, state='up')
                logging.info('SSH tunnel %d listening on %s:%s (destination not checked)', i, host, port)
                while not self.stop.wait(.2) and not disabled.is_set():
                    if process.poll() is not None:
                        raise RuntimeError('SSH disconnected (%s)' % process.returncode)
            except (OSError, RuntimeError) as exc:
                error = str(exc)
                logging.warning('SSH tunnel %d: %s; DOWN, press a to add a tunnel', i, exc)
            finally:
                if process is not None:
                    self._cleanup(process)
                if reader is not None:
                    reader.join(timeout=2)
                    if not reader.is_alive():
                        process.stderr.close()
                if starting:
                    self.gate.release(failed=not self.stop.is_set() and not disabled.is_set())
                with self.lock:
                    self.rows[i].update(state='stopped' if self.stop.is_set() else 'down',
                        error=self.rows[i]['error'] or error, pid=None, changed=time.monotonic())
            # No request.set() here: neither failure nor disconnect retries itself.

    def close(self):
        self.stop.set()
        for worker in self.workers:
            if worker.ident is not None:
                worker.join()


def read_status(path):
    try:
        return json.loads(Path(path).read_text()) if path else {}
    except (OSError, ValueError):
        return {}


def dashboard_lines(pool, stats, notice):
    rows = pool.snapshot()
    up = sum(r['state'] == 'up' for r in rows)
    connecting = sum(r['state'] in ('queued', 'connecting') for r in rows)
    lines = [f"SSH relay   Connected {up} | Connecting {connecting} | Capacity {len(rows)}",
             'a: add one   d: remove one   q: quit and close tunnels', '']
    current = stats.get('current')
    stale = time.time() - stats.get('updated', 0) > 3
    if stale:
        lines += ['Transfer telemetry unavailable / stale']
    elif current:
        lines += [f"Current       {current['MBps']:.1f} MB/s (recent acknowledged payload)",
                  f"Progress      {current['done']/1e9:.2f} / {current['size']/1e9:.2f} GB"
                  f" | {current['elapsed']:.1f}s elapsed | {current['phase']}"]
    else:
        lines += ['Current       Idle']
    lines += ['', 'Recent transfers: learner -> inference (payload average; excludes verification)']
    for row in stats.get('history', [])[-5:][::-1]:
        counts = ' -> '.join(str(n) for n in row.get('tunnels', [])) or '?'
        lines.append(f"  {counts:12s} tunnels | {row['MBps']:7.1f} MB/s | {row['elapsed']:6.1f}s | {row['phase']}")
    errors = [r for r in rows if r['error']]
    if errors:
        latest = max(errors, key=lambda r: r['changed'])
        lines += ['', 'Last SSH error: ' + latest['error']]
    lines += ['', notice, 'SSH connection count is not application readiness. Auto-reconnect OFF.']
    return lines


def dashboard(screen, pool, stop, stats_path=None):
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    screen.timeout(200)
    notice = 'One connection attempt per add; failures stay down.'
    while not stop.is_set():
        height, width = screen.getmaxyx()
        screen.erase()
        for y, line in enumerate(dashboard_lines(pool, read_status(stats_path), notice)[:height]):
            try:
                screen.addnstr(y, 0, line, max(0, width - 1))
            except curses.error:
                pass
        screen.refresh()
        key = screen.getch()
        if key in (ord('q'), ord('Q')):
            stop.set()
        elif key in (ord('a'), ord('A')):
            index = pool.add_one()
            notice = 'At capacity / shutting down.' if index is None else f'Adding tunnel {index}.'
        elif key in (ord('d'), ord('D')):
            index = pool.remove_one()
            notice = 'No active tunnel to remove.' if index is None else f'Closing tunnel {index}.'


def publish_routes(path, pool):
    # Tiny volatile status; atomic replacement, deliberately no durable fsync.
    rows = pool.snapshot()
    value = dict(updated=time.time(), active=[r['index'] for r in rows if r['state'] == 'up'])
    output = Path(path)
    temp = output.with_suffix('.tmp')
    temp.write_text(json.dumps(value))
    temp.replace(output)


def run(config, stop, ui=None):
    gate = StartupGate(config.get('startup_concurrency', 2), config.get('startup_interval', .5))
    commands, endpoints = forwarding_commands(dict(config, max_connections=config.get('max_connections', 64)))
    pool = TunnelPool(commands, stop, gate, initial=config.get('connections', 32))
    monitor = None
    def publish():
        while not stop.is_set():
            try:
                publish_routes(config['state_file'], pool)
            except OSError:
                logging.exception('Cannot publish relay state')
            stop.wait(.5)
    try:
        pool.start()
        output = Path(config['output'])
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(endpoints, indent=2))
        temp.replace(output)
        if config.get('state_file'):
            monitor = threading.Thread(target=publish)
            monitor.start()
        if ui is None:
            stop.wait()
        else:
            ui(pool, stop)
    finally:
        pool.close()
        if monitor:
            monitor.join()
            publish_routes(config['state_file'], pool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--no-tui', action='store_true', help='Log only; no automatic recovery')
    parser.add_argument('--log-file', help='Append relay/SSH messages here (TUI stays readable)')
    args = parser.parse_args()
    tui = not args.no_tui and sys.stdin.isatty() and sys.stdout.isatty()
    handlers = [logging.FileHandler(args.log_file)] if args.log_file else []
    if not tui:
        handlers.append(logging.StreamHandler())
    elif not handlers:
        handlers.append(logging.NullHandler())
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=handlers)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    config = json.loads(Path(args.config).read_text())
    ui = (lambda pool, event: curses.wrapper(dashboard, pool, event, config.get('stats_file'))) if tui else None
    run(config, stop, ui)


if __name__ == '__main__':
    main()
