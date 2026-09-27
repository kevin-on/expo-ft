"""Compare the production RAM transport with the earlier host-RAM benchmark.

Driver: two independent sidecars and two application processes on a TEST node.
Sender/receiver: use already provisioned transports on separate TEST machines.
No GPU, robot or camera imports. Payload is read once before the measured trials.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from expo_ft.distributed.buffer import Buffer
from expo_ft.distributed.channel import Channel


def sender(args):
    channel = Channel(args.mailbox, timeout=args.timeout)
    buffer = Buffer.create(args.payload.stat().st_size)
    try:
        with args.payload.open('rb', buffering=0) as source, buffer.view() as view:
            offset = 0
            while offset < len(view):
                n = source.readinto(view[offset:])
                if not n:
                    raise EOFError('incomplete source')
                offset += n
        buffer.seal()
        expected = buffer.digest()
        # Exclude initial module imports, source read/hash and receiver startup.
        channel.send('benchmark-hello', args.session, {'bytes': buffer.size, 'xxh3_128': expected})
        channel.flush()
        channel.receive('benchmark-ready', args.session)
        rows = []
        for trial in range(args.trials):
            key = f'{args.session}/{trial}'
            started = time.monotonic()
            channel.send_buffer('benchmark-payload', key, buffer)
            published = time.monotonic()
            result = channel.receive('benchmark-result', key)
            channel.wait_sent('benchmark-payload', key)
            assert result['xxh3_128'] == expected
            row = dict(trial=trial, bytes=buffer.size, **result,
                       sender_hash_handoff_seconds=published - started,
                       through_application_verification_seconds=time.monotonic() - started)
            row['receive_MBps'] = buffer.size / result['receive_seconds'] / 1e6
            rows.append(row)
            channel.release('benchmark-result', key)
            print(json.dumps(row), flush=True)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'benchmark-results.json').write_text(json.dumps(rows, indent=2))
    finally:
        buffer.close()
        channel.close()


def receiver(args):
    channel = Channel(args.mailbox, timeout=args.timeout)
    try:
        hello = channel.receive('benchmark-hello', args.session)
        channel.send('benchmark-ready', args.session, {'ready': True})
        channel.flush()
        for trial in range(args.trials):
            key = f'{args.session}/{trial}'
            with channel.receive_buffer('benchmark-payload', key) as buffer:
                assert buffer.size == hello['bytes']
                started = time.monotonic()
                digest = buffer.digest()
                result = dict(buffer.timings, xxh3_128=digest,
                              application_digest_seconds=time.monotonic() - started)
                assert digest == hello['xxh3_128']
            channel.release('benchmark-payload', key)
            channel.send('benchmark-result', key, result)
            channel.flush()
            channel.wait_sent('benchmark-result', key)
    finally:
        channel.close()


def driver(args):
    args.output.mkdir(parents=True, exist_ok=False)
    processes, logs = [], []
    with tempfile.TemporaryDirectory(prefix='expo-ram-bench-') as directory:
        local = Path(directory)
        token, cert, private = local / 'token', local / 'cert.pem', local / 'key.pem'
        token.write_text(secrets.token_hex(32))
        token.chmod(0o600)
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'ec', '-pkeyopt', 'ec_paramgen_curve:P-256',
                        '-nodes', '-days', '1', '-subj', '/CN=expo-test', '-keyout', str(private),
                        '-out', str(cert)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        private.chmod(0o600)
        ports = []
        for _ in range(2):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                ports.append(sock.getsockname()[1])
        def launch(name, command):
            log = (args.output / f'{name}.log').open('w')
            logs.append(log)
            p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            processes.append(p)
            return p
        try:
            for i in range(2):
                config = local / f'transport-{i}.json'
                config.write_text(json.dumps(dict(mailbox=str(local / str(i)), token_file=str(token),
                    listen=['127.0.0.1', ports[i]], peers=[['127.0.0.1', ports[1-i]]],
                    parallel_connections=args.connections, record_connections=4,
                    tls=dict(cert_file=str(cert), key_file=str(private), ca_file=str(cert)))))
                launch(f'transport-{i}', [sys.executable, '-u', '-m', 'expo_ft.distributed.transport', '--config', str(config)])
            base = [sys.executable, '-u', __file__, '--payload', str(args.payload), '--output', str(args.output),
                    '--trials', str(args.trials), '--timeout', str(args.timeout), '--session', args.session]
            receive = launch('receiver', base + ['--role', 'receiver', '--mailbox', str(local / '1')])
            send = launch('sender', base + ['--role', 'sender', '--mailbox', str(local / '0')])
            deadline = time.monotonic() + args.timeout
            while send.poll() is None or receive.poll() is None:
                if any(p.poll() not in (None, 0) for p in processes):
                    raise RuntimeError('RAM benchmark failed; inspect role logs')
                if time.monotonic() >= deadline:
                    raise TimeoutError('RAM benchmark')
                time.sleep(.2)
            assert send.returncode == receive.returncode == 0
            assert not list(local.rglob('*.bin')) and not list(local.rglob('*.zip'))
            print((args.output / 'benchmark-results.json').read_text())
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for log in logs:
                log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['driver', 'sender', 'receiver'], default='driver')
    parser.add_argument('--mailbox')
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--connections', type=int, default=64)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--session', default='ram-benchmark')
    args = parser.parse_args()
    globals()[args.role](args)
