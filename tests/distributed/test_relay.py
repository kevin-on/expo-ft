"""Inspect forwarding commands using mocks; no real SSH connection is opened."""
import json
import socket
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from expo_ft.distributed import relay


class Child:
    def __init__(self):
        self.returncode = None
        self.terminate = MagicMock(side_effect=self._terminate)
        self.kill = self.terminate
    def _terminate(self):
        self.returncode = -15
    def poll(self):
        return self.returncode
    def wait(self, timeout=None):
        assert self.returncode is not None
        return self.returncode


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('condition did not become true')
        time.sleep(.01)


class RelayTest(unittest.TestCase):
    def test_port_probe_reuses_time_wait_but_rejects_live_listener(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            port = listener.getsockname()[1]
            config = dict(ssh_config='/private/config', ssh_host='remote',
                          connections=1, first_port=port, destination=['compute', 12345])
            with self.assertRaises(OSError):
                relay.forwarding_commands(config)
            with socket.create_connection(('127.0.0.1', port)) as client:
                accepted, _ = listener.accept()
                accepted.close()  # server closes first, leaving its port in TIME_WAIT
                self.assertEqual(client.recv(1), b'')
        commands, endpoints = relay.forwarding_commands(config)
        self.assertEqual(endpoints['peers'], [['127.0.0.1', port]])
        self.assertEqual(len(commands), 1)

    def test_default_64_ssh_connections_each_carry_both_directions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = dict(ssh_config='/private/config', ssh_host='remote-login',
                startup_interval=0,
                first_port=20000, destination=['remote-compute', 19002],
                reverse=dict(first_port=21000, destination=['local-compute', 19001]), output=str(path / 'endpoints'))
            children = [Child() for _ in range(64)]
            probe = MagicMock()
            probe.__enter__.return_value = probe
            probe.getsockname.side_effect = [('127.0.0.1', 20000+i) for i in range(64)]
            with patch.object(relay.socket, 'socket', return_value=probe), \
                 patch.object(relay.socket, 'create_connection', return_value=MagicMock()), \
                 patch.object(relay.subprocess, 'Popen', side_effect=children) as popen:
                stop = threading.Event()
                worker = threading.Thread(target=relay.run, args=(config, stop))
                worker.start()
                try:
                    wait_until(lambda: (path / 'endpoints').exists() and popen.call_count == 64)
                finally:
                    stop.set()
                    worker.join(timeout=5)
                self.assertFalse(worker.is_alive())
            self.assertEqual(popen.call_count, 64)
            for call in popen.call_args_list:
                command = call.args[0]
                i = int(command[command.index('-L') + 1].split(':')[1]) - 20000
                self.assertIn(f'127.0.0.1:{20000+i}:remote-compute:19002', command)
                self.assertIn(f'127.0.0.1:{21000+i}:local-compute:19001', command)
                self.assertIn('ControlPath=none', command)
                self.assertIn('ForwardAgent=no', command)
                children[i].terminate.assert_called_once()
            result = json.loads((path / 'endpoints').read_text())
            self.assertEqual(len(result['peers']), 64)
            self.assertEqual(len(result['reverse_peers']), 64)

    def test_only_failed_children_restart_with_identical_forwarding(self):
        with tempfile.TemporaryDirectory() as directory:
            config = dict(ssh_config='/private/config', ssh_host='remote-login', connections=3,
                first_port=22000, destination=['remote-compute', 19002],
                reverse=dict(first_port=23000, destination=['local-compute', 19001]),
                output=str(Path(directory) / 'endpoints'))
            history = {}
            lock = threading.Lock()
            def launch(command):
                forward = command[command.index('-L') + 1]
                with lock:
                    attempts = history.setdefault(forward, [])
                    child = Child()
                    # The first link fails at initial startup, the second later.
                    if ':22000:' in forward and not attempts:
                        child.returncode = 255
                    attempts.append((list(command), child))
                    return child
            probe = MagicMock()
            probe.__enter__.return_value = probe
            probe.getsockname.side_effect = [('127.0.0.1', 22000+i) for i in range(3)]
            with patch.object(relay.socket, 'socket', return_value=probe), \
                 patch.object(relay.socket, 'create_connection', return_value=MagicMock()), \
                 patch.object(relay.subprocess, 'Popen', side_effect=launch):
                stop = threading.Event()
                worker = threading.Thread(target=relay.run, args=(config, stop))
                worker.start()
                try:
                    wait_until(lambda: len(history) == 3 and Path(config['output']).exists(), timeout=15)
                    keys = sorted(history)
                    good = history[keys[2]][0][1]
                    history[keys[1]][0][1].returncode = 255
                    wait_until(lambda: all(len(history[k]) == 2 for k in keys[:2]), timeout=15)
                    self.assertEqual(len(history[keys[2]]), 1)
                    good.terminate.assert_not_called()
                    self.assertFalse(stop.is_set())
                    for key in keys[:2]:
                        self.assertEqual(history[key][0][0], history[key][1][0])
                    self.assertEqual(len(json.loads(Path(config['output']).read_text())['peers']), 3)
                finally:
                    stop.set()
                    worker.join(timeout=5)
                self.assertFalse(worker.is_alive())
                for entries in history.values():
                    self.assertIsNotNone(entries[-1][1].poll())

    def test_startup_gate_spaces_attempts_and_shares_outage_cooldown(self):
        stop = MagicMock()
        stop.is_set.return_value = False
        now = [100.0]
        stop.wait.side_effect = lambda seconds: now.__setitem__(0, now[0] + seconds)
        with patch.object(relay.time, 'monotonic', side_effect=lambda: now[0]), \
             patch.object(relay.random, 'uniform', side_effect=lambda low, high: low):
            gate = relay.StartupGate(limit=2, interval=.5)
            self.assertTrue(gate.acquire(stop))
            self.assertEqual(now[0], 100)
            self.assertTrue(gate.acquire(stop))
            self.assertAlmostEqual(now[0], 100.5)
            gate.release(failed=True)
            self.assertTrue(gate.acquire(stop))
            self.assertAlmostEqual(now[0], 105.5)
            gate.release(failed=True)
            self.assertTrue(gate.acquire(stop))
            self.assertAlmostEqual(now[0], 115.5)
            gate.release()
            gate.release()
            self.assertEqual(gate.failure_delay, 5)

    def test_stop_while_startup_slots_are_full(self):
        gate = relay.StartupGate(limit=2, interval=0)
        stop = threading.Event()
        self.assertTrue(gate.acquire(stop))
        self.assertTrue(gate.acquire(stop))
        result = []
        waiter = threading.Thread(target=lambda: result.append(gate.acquire(stop)))
        waiter.start()
        stop.set()
        waiter.join(timeout=1)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(result, [False])
        gate.release()
        gate.release()

    def test_startup_limit_is_released_once_tunnels_are_listening(self):
        stop = threading.Event()
        children = []
        probe_ready = threading.Event()
        def launch(command):
            child = Child()
            children.append(child)
            return child
        def probe(*args, **kwargs):
            if not probe_ready.is_set():
                raise OSError('not listening yet')
            return MagicMock()
        gate = relay.StartupGate(limit=2, interval=0)
        workers = [threading.Thread(target=relay.maintain_tunnel,
                   args=(['fake-ssh'], '127.0.0.1', 22000+i, stop, gate)) for i in range(4)]
        with patch.object(relay.subprocess, 'Popen', side_effect=launch), \
             patch.object(relay.socket, 'create_connection', side_effect=probe):
            try:
                for worker in workers:
                    worker.start()
                wait_until(lambda: len(children) == 2)
                time.sleep(.15)
                self.assertEqual(len(children), 2)
                probe_ready.set()
                wait_until(lambda: len(children) == 4)
                self.assertTrue(all(child.poll() is None for child in children))
            finally:
                stop.set()
                for worker in workers:
                    worker.join(timeout=2)
            self.assertTrue(all(not worker.is_alive() for worker in workers))
            self.assertTrue(all(child.poll() is not None for child in children))

    def test_reject_invalid_startup_settings(self):
        for limit, interval in [(0, .5), (65, .5), (2, -1), (2, float('nan'))]:
            with self.assertRaises(ValueError):
                relay.StartupGate(limit, interval)
