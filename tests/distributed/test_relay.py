"""Inspect forwarding commands using mocks; no real SSH connection is opened."""
import io
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
        self.pid = id(self)
        self.stderr = io.StringIO()
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
    def setUp(self):
        # Exercise pool lifecycle without sending any real signals or SSH traffic.
        self.cleanup = patch.object(relay.TunnelPool, '_cleanup',
            side_effect=lambda child: (child.terminate() if child.poll() is None else None, child.wait()))
        self.cleanup.start()
        self.addCleanup(self.cleanup.stop)

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

    def test_default_32_ssh_connections_each_carry_both_directions(self):
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
                    wait_until(lambda: (path / 'endpoints').exists() and popen.call_count == 32)
                finally:
                    stop.set()
                    worker.join(timeout=5)
                self.assertFalse(worker.is_alive())
            self.assertEqual(popen.call_count, 32)
            for call in popen.call_args_list:
                command = call.args[0]
                i = int(command[command.index('-L') + 1].split(':')[1]) - 20000
                self.assertIn(f'127.0.0.1:{20000+i}:remote-compute:19002', command)
                self.assertIn(f'127.0.0.1:{21000+i}:local-compute:19001', command)
                self.assertIn('ControlPath=none', command)
                self.assertIn('ForwardAgent=no', command)
                self.assertIn('ConnectionAttempts=1', command)
                children[i].terminate.assert_called_once()
            result = json.loads((path / 'endpoints').read_text())
            self.assertEqual(len(result['peers']), 64)
            self.assertEqual(len(result['reverse_peers']), 64)

    def test_failures_wait_for_manual_recovery_and_preserve_healthy_children(self):
        children = []
        def launch(command, **kwargs):
            child = Child()
            if command == ['bad']:
                child.returncode = 255
                child.stderr = io.StringIO('Permission denied\n')
            children.append((command, child))
            return child
        stop = threading.Event()
        pool = relay.TunnelPool([(['bad'], 'localhost', 22000),
                                (['later'], 'localhost', 22001),
                                (['good'], 'localhost', 22002)], stop,
                               relay.StartupGate(limit=2, interval=0))
        with patch.object(relay.subprocess, 'Popen', side_effect=launch), \
             patch.object(relay.socket, 'create_connection', return_value=MagicMock()), \
             patch.object(pool.gate, 'release', side_effect=lambda failed=False: pool.gate.slots.release()):
            pool.start()
            try:
                wait_until(lambda: [r['state'] for r in pool.snapshot()] == ['down', 'up', 'up'])
                by_command = {cmd[0]: child for cmd, child in children}
                by_command['later'].returncode = 255
                wait_until(lambda: pool.snapshot()[1]['state'] == 'down')
                time.sleep(.4)
                self.assertEqual(len(children), 3)  # Neither startup nor later failure respawns.
                self.assertEqual(pool.snapshot()[0]['error'], 'Permission denied')
                self.assertEqual(pool.add_one(), 0)
                wait_until(lambda: pool.snapshot()[0]['attempts'] == 2 and pool.snapshot()[0]['state'] == 'down')
                self.assertEqual(pool.add_one(), 1)  # Oldest failure, not failed #0 again.
                wait_until(lambda: pool.snapshot()[1]['state'] == 'up')
                self.assertEqual([cmd for cmd, _ in children[3:]], [['bad'], ['later']])
                by_command['good'].terminate.assert_not_called()
                time.sleep(.4)
                self.assertEqual(len(children), 5)
                self.assertFalse(stop.is_set())
            finally:
                pool.close()
            self.assertTrue(all(child.poll() is not None for _, child in children))

    def test_recovery_does_not_queue_duplicates_or_restart_healthy_links(self):
        stop = threading.Event()
        pool = relay.TunnelPool([(['ssh'], 'localhost', 22000)], stop, relay.StartupGate())
        self.assertIsNone(pool.add_one())  # Already queued for initial attempt.
        pool._update(0, state='up')
        self.assertIsNone(pool.add_one())
        pool._update(0, state='down')
        self.assertEqual(pool.add_one(), 0)
        self.assertIsNone(pool.add_one())
        stop.set()
        pool._update(0, state='down')
        self.assertIsNone(pool.add_one())

    def test_dashboard_manual_key_and_quit(self):
        pool = MagicMock()
        pool.snapshot.return_value = [dict(index=0, state='down', host='localhost', port=22000,
                                          attempts=1, error='Permission denied', changed=0)]
        pool.add_one.return_value = 0
        screen = MagicMock()
        screen.getmaxyx.return_value = (24, 100)
        screen.getch.side_effect = [ord('a'), ord('d'), ord('q')]
        stop = threading.Event()
        with patch.object(relay.curses, 'curs_set'):
            relay.dashboard(screen, pool, stop)
        pool.add_one.assert_called_once()
        pool.remove_one.assert_called_once()
        self.assertTrue(stop.is_set())

    def test_live_add_remove_and_reuse_slot(self):
        children = []
        def launch(*args, **kwargs):
            child = Child()
            children.append(child)
            return child
        stop = threading.Event()
        pool = relay.TunnelPool([(['ssh'], 'localhost', 22000+i) for i in range(3)],
                               stop, relay.StartupGate(interval=0), initial=1)
        with patch.object(relay.subprocess, 'Popen', side_effect=launch), \
             patch.object(relay.socket, 'create_connection', return_value=MagicMock()):
            pool.start()
            try:
                wait_until(lambda: pool.snapshot()[0]['state'] == 'up')
                self.assertEqual(len(children), 1)
                self.assertEqual(pool.add_one(), 1)
                wait_until(lambda: pool.snapshot()[1]['state'] == 'up')
                self.assertEqual(pool.remove_one(), 1)
                wait_until(lambda: pool.snapshot()[1]['state'] == 'down')
                self.assertIsNotNone(children[1].poll())
                self.assertIsNone(children[0].poll())
                self.assertEqual(pool.remove_one(), 0)
                wait_until(lambda: pool.snapshot()[0]['state'] == 'down')
                self.assertIsNone(pool.remove_one())
                self.assertEqual(pool.add_one(), 2)  # unused oldest slot first
                wait_until(lambda: pool.snapshot()[2]['state'] == 'up')
            finally:
                pool.close()
        self.assertEqual(len(children), 3)

    def test_speed_dashboard_idle_stale_and_history(self):
        pool = MagicMock()
        pool.snapshot.return_value = [dict(state='up', error='', changed=0)]
        stats = dict(updated=time.time(), current=None,
                     history=[dict(tunnels=[32,31], MBps=160, elapsed=12, phase='Complete')])
        lines = '\n'.join(relay.dashboard_lines(pool, stats, ''))
        self.assertIn('Idle', lines)
        self.assertIn('32 -> 31', lines)
        self.assertIn('160.0 MB/s', lines)
        stats['updated'] = 0
        self.assertIn('stale', '\n'.join(relay.dashboard_lines(pool, stats, '')))

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
        def launch(command, **kwargs):
            self.assertTrue(kwargs['start_new_session'])
            child = Child()
            children.append(child)
            return child
        def probe(*args, **kwargs):
            if not probe_ready.is_set():
                raise OSError('not listening yet')
            return MagicMock()
        pool = relay.TunnelPool([(['ssh'], '127.0.0.1', 22000+i) for i in range(4)],
                               stop, relay.StartupGate(limit=2, interval=0))
        with patch.object(relay.subprocess, 'Popen', side_effect=launch), \
             patch.object(relay.socket, 'create_connection', side_effect=probe):
            try:
                pool.start()
                wait_until(lambda: len(children) == 2)
                time.sleep(.15)
                self.assertEqual(len(children), 2)
                probe_ready.set()
                wait_until(lambda: len(children) == 4)
                self.assertTrue(all(child.poll() is None for child in children))
            finally:
                pool.close()
            self.assertTrue(all(not worker.is_alive() for worker in pool.workers))
            self.assertTrue(all(child.poll() is not None for child in children))

    def test_remove_queued_tunnel_cancels_startup_wait(self):
        stop = threading.Event()
        gate = relay.StartupGate(limit=1, interval=0)
        self.assertTrue(gate.acquire(stop))  # Occupy only handshake slot.
        pool = relay.TunnelPool([(['ssh'], 'localhost', 22000)], stop, gate)
        with patch.object(relay.subprocess, 'Popen') as launch:
            pool.start()
            try:
                self.assertEqual(pool.remove_one(), 0)
                wait_until(lambda: pool.snapshot()[0]['state'] == 'down')
                launch.assert_not_called()
            finally:
                pool.close()
                gate.release()

    def test_reject_invalid_startup_settings(self):
        for limit, interval in [(0, .5), (65, .5), (2, -1), (2, float('nan'))]:
            with self.assertRaises(ValueError):
                relay.StartupGate(limit, interval)


class ProcessGroupCleanupTest(unittest.TestCase):
    def test_cleanup_signals_proxy_group_even_when_parent_has_exited(self):
        child = MagicMock(pid=123456)
        child.poll.return_value = 255
        with patch.object(relay.os, 'killpg') as killpg:
            relay.TunnelPool._cleanup(child)
        self.assertEqual(killpg.call_args_list[0].args, (123456, relay.signal.SIGTERM))
        self.assertEqual(killpg.call_args_list[1].args, (123456, relay.signal.SIGKILL))
        self.assertTrue(child.wait.called)

    def test_cleanup_reaps_parent_when_group_is_already_gone(self):
        child = MagicMock(pid=123456)
        with patch.object(relay.os, 'killpg', side_effect=ProcessLookupError):
            relay.TunnelPool._cleanup(child)
        self.assertTrue(child.wait.called)
