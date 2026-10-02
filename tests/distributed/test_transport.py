"""Run on a test machine, never alongside live workstation robot experiments."""
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest

import numpy as np
import xxhash

from expo_ft.distributed.channel import Channel, message_id
from expo_ft.distributed.buffer import Buffer
from expo_ft.distributed.protocol import key, receive_round
from expo_ft.distributed.transport import Store, Transport, PeerEpoch


def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class TransportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'token').write_text('test-only-token-' * 4)
        ports = [port(), port()]
        self.configs = [dict(mailbox=str(self.root / str(i)), token_file=str(self.root / 'token'),
                             listen=['127.0.0.1', ports[i]], peers=[['127.0.0.1', ports[1-i]]],
                             parallel_connections=32, record_connections=4, chunk_bytes=65536,
                             allow_plain_loopback=True, socket_timeout=2) for i in range(2)]
        self.configure_transport()
        self.transports = [Transport(c) for c in self.configs]
        self.threads = [threading.Thread(target=t.run) for t in self.transports]
        for thread in self.threads:
            thread.start()
        self.channels = [Channel(c['mailbox'], timeout=15) for c in self.configs]

    def configure_transport(self):
        pass

    def tearDown(self):
        try:
            for channel in self.channels:
                channel.close()
        finally:
            for t in self.transports:
                t.stop.set()
            for thread in self.threads:
                thread.join(timeout=10)
                self.assertFalse(thread.is_alive())
            self.temp.cleanup()

    def test_simultaneous_records_and_parallel_snapshot(self):
        a, b = self.channels
        payload = np.random.default_rng(1).integers(0, 256, 4 * 1024**2, dtype=np.uint8).tobytes()
        with Buffer.from_bytes(payload) as buffer:
            a.send_buffer('policy', 'v1', buffer)
        source = {'image': np.ones((180, 320, 3), np.uint8), 'actions': np.arange(7, dtype=np.float64)}
        for i in range(20):
            b.send('transition', str(i), source)
        source['actions'][:] = -99  # queued records must be immutable snapshots
        b.flush()
        for i in range(20):
            received = a.receive('transition', str(i))
            np.testing.assert_array_equal(received['actions'], np.arange(7))
        with b.receive_buffer('policy', 'v1') as buffer, buffer.view() as view:
            self.assertEqual(view, payload)
        a.wait_sent('policy', 'v1')
        b.release('policy', 'v1')
        for i in range(20):
            b.wait_sent('transition', str(i))
            a.release('transition', str(i))
        self.assertEqual([t.store.tx_bytes + t.store.rx_bytes for t in self.transports], [0, 0])
        self.assertFalse(list(self.root.rglob('*.bin')))
        self.assertFalse(list(self.root.rglob('*.zip')))

    def test_reordered_duplicate_records_and_episode_barrier(self):
        a, b = self.channels
        for robot in range(2):
            a.send('episode_end', key('s', 0, robot), {'version': 3, 'length': 2, 'success': robot == 1})
            for step in (1, 0):
                record = {'actions': np.zeros(7), 'observations': {'state': np.ones(6)},
                          'rewards': float(step), 'masks': 1 - step, 'dones': step == 1, 'is_hil': False}
                a.send('transition', key('s', 0, robot, step), {'version': 3, 'transition': record})
        a.flush()
        episodes = receive_round(b, 's', 0, 3, 2)
        self.assertEqual([e[1] for e in episodes], [False, True])
        self.assertEqual([r['rewards'] for r in episodes[0][0]], [0., 1.])
        # Delivery duplicates are accepted only if content is identical.
        value = {'version': 3, 'length': 2, 'success': False}
        a.send('episode_end', key('s', 0, 0), value)
        a.flush()
        self.assertEqual(b.receive('episode_end', key('s', 0, 0)), value)

    def test_socket_reconnect_preserves_ram_session(self):
        self.channels[0].send('control', 'before', {'value': 1})
        self.channels[0].flush()
        self.assertEqual(self.channels[1].receive('control', 'before'), {'value': 1})
        self.channels[0].wait_sent('control', 'before')
        for transport in self.transports:
            for connection in transport.bulk + transport.small:
                connection.close()
        self.channels[0].send('control', 'after', {'value': 42})
        self.channels[0].flush()
        self.assertEqual(self.channels[1].receive('control', 'after'), {'value': 42})

    def test_idle_receiver_observes_peer_abort(self):
        a, b = self.channels
        b.start_session('cancel-test')
        b.check_session()
        self.assertEqual(self.transports[1].store.tx_bytes, 0)
        a.send('abort', 'cancel-test', {'aborted': True})
        a.flush()
        a.wait_sent('abort', 'cancel-test')
        # No outgoing transition / background-writer failure triggers this check.
        self.assertIsNone(b.error)
        self.assertTrue(b.queue.empty())
        with self.assertRaisesRegex(RuntimeError, 'peer aborted'):
            b.check_session()


class TLSTransportTest(TransportTest):
    def configure_transport(self):
        import subprocess
        cert, private = self.root / 'cert.pem', self.root / 'private.pem'
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'ec', '-pkeyopt', 'ec_paramgen_curve:P-256',
                        '-nodes', '-days', '1', '-subj', '/CN=expo-test', '-keyout', str(private),
                        '-out', str(cert)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for config in self.configs:
            config['tls'] = dict(cert_file=str(cert), key_file=str(private), ca_file=str(cert))
            config.pop('allow_plain_loopback')


class ValidationTest(unittest.TestCase):
    def test_legacy_or_invalid_checksum_metadata_rejected(self):
        store = Store(4, 1024)
        self.addCleanup(store.close)
        legacy = dict(id=message_id('t', 'k'), topic='t', key='k', size=4, sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'invalid'):
            store.offer(legacy)
        for digest in ('0' * 64, 'g' * 32, ''):
            with self.subTest(digest=digest), self.assertRaisesRegex(ValueError, 'invalid'):
                store.offer(dict(legacy, xxh3_128=digest))
        self.assertEqual(store.rx_bytes, 0)

    def test_small_payload_checksum_and_corruption(self):
        import io
        class Reader:
            def __init__(self, data):
                self.data = io.BytesIO(data)
            def recv_into(self, view):
                return self.data.readinto(view)
        store = Store(16, 1024)
        self.addCleanup(store.close)
        meta = dict(id=message_id('t', 'k'), topic='t', key='k', size=4,
                    xxh3_128=xxhash.xxh3_128(b'good').hexdigest())
        with self.assertRaisesRegex(ValueError, 'checksum'):
            store.small(Reader(b'bad!'), meta)
        self.assertNotIn(meta['id'], store.received)
        store.small(Reader(b'good'), meta)
        self.assertIn(meta['id'], store.received)

    def test_chunk_layout_and_header_bound(self):
        store = Store(4, 100000)
        self.addCleanup(store.close)
        meta = dict(id=message_id('test', 'chunks'), topic='test', key='chunks',
                    size=257, xxh3_128='0' * 32)
        status = store.offer(meta)
        self.assertEqual(status['chunk_bytes'], 4)
        self.assertEqual(status['missing'], list(range(65)))
        oversized = dict(meta, id=message_id('test', 'large'), key='large', size=4 * 8192 + 1)
        with self.assertRaisesRegex(ValueError, 'too many chunks'):
            store.offer(oversized)
        self.assertEqual(store.rx_bytes, 257)

    def test_conflicting_and_incomplete_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(4, 1024)
            self.addCleanup(store.close)
            data = b'12345678'
            meta = {'id': message_id('test', 'key'), 'topic': 'test', 'key': 'key',
                    'size': len(data), 'xxh3_128': xxhash.xxh3_128(data).hexdigest()}
            self.assertEqual(store.offer(meta)['missing'], [0, 1])
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                store.commit(meta)
            with self.assertRaisesRegex(ValueError, 'conflicting'):
                store.offer(dict(meta, xxh3_128='0' * 32))
            with self.assertRaisesRegex(ValueError, 'invalid'):
                store.offer(dict(meta, id='../../escape'))

    def test_partial_range_retry_and_released_dedup(self):
        import io
        class Reader:
            def __init__(self, data):
                self.data = io.BytesIO(data)
            def recv_into(self, view):
                return self.data.readinto(view)
        store = Store(4, 1024)
        self.addCleanup(store.close)
        data = b'abcdefgh'
        meta = dict(id=message_id('test', 'k'), topic='test', key='k', size=len(data),
                    xxh3_128=xxhash.xxh3_128(data).hexdigest())
        store.offer(meta)
        with self.assertRaises(EOFError):
            store.chunk(Reader(b'ab'), {'id': meta['id'], 'index': 0})
        self.assertEqual(store.offer(meta)['missing'], [0, 1])
        store.chunk(Reader(data[:4]), {'id': meta['id'], 'index': 0})
        self.assertEqual(store.offer(meta)['missing'], [1])
        store.chunk(Reader(data[4:]), {'id': meta['id'], 'index': 1})
        store.commit(meta)
        # Duplicate completion is idempotent even after freeing receiver pages.
        store.release(meta['id'])
        self.assertTrue(store.offer(meta)['complete'])
        self.assertEqual(store.rx_bytes, 0)

    def test_corrupt_buffer_not_admitted(self):
        with Buffer.from_bytes(b'abcd') as buffer:
            meta = dict(id=message_id('t', 'k'), topic='t', key='k', size=4,
                        xxh3_128=xxhash.xxh3_128(b'good').hexdigest())
            store = Store(4, 1024)
            self.addCleanup(store.close)
            store.offer(meta)
            entry = store.incoming[meta['id']]
            with entry['buffer'].view() as view:
                view[:] = b'abcd'
            entry['done'].add(0)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                store.commit(meta)
            self.assertNotIn(meta['id'], store.received)
            self.assertEqual(store.offer(meta)['missing'], [0])

    def test_process_restart_changes_epoch(self):
        epoch = PeerEpoch()
        epoch.check('first')
        epoch.check('first')
        with self.assertRaisesRegex(ValueError, 'restarted'):
            epoch.check('second')

    def test_sealed_buffer_is_shared_without_copy(self):
        import os
        with Buffer.from_bytes(b'payload') as original:
            with Buffer.from_fd(os.dup(original.fd)) as duplicate:
                self.assertEqual(os.fstat(original.fd).st_ino, os.fstat(duplicate.fd).st_ino)
                with duplicate.view() as view:
                    self.assertEqual(view, b'payload')
                    self.assertTrue(view.readonly)
                with self.assertRaises(OSError):
                    os.pwrite(original.fd, b'x', 0)

    def test_bad_size_and_quota_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(4, 1024, quota_bytes=4)
            meta = dict(id=message_id('t', 'k'), topic='t', key='k', size=-1, xxh3_128='0'*32)
            with self.assertRaisesRegex(ValueError, 'invalid'):
                store.small(None, meta)
            with self.assertRaisesRegex(ValueError, 'quota'):
                store.offer(dict(meta, size=8))

    def test_missing_transition_and_wrong_version_abort_round(self):
        class Peer:
            def receive(self, topic, key):
                if topic == 'episode_end':
                    return dict(version=4, length=1, success=True)
                raise TimeoutError('missing transition')
        with self.assertRaisesRegex(ValueError, 'version'):
            receive_round(Peer(), 's', 0, 3, 1)
        with self.assertRaises(TimeoutError):
            receive_round(Peer(), 's', 0, 4, 1)

    def test_fresh_application_session_required(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Channel(directory)
            first.start_session('a')
            first.close()
            second = Channel(directory)
            try:
                with self.assertRaisesRegex(ValueError, 'already used'):
                    second.start_session('a')
            finally:
                second.close()

    def test_stream_matches_mirrored_executed_actions(self):
        from expo_ft.env.sft_eval import canonical_observation, physical_action
        from expo_ft.utils.robot_round import collect_round
        original = {k: np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
                    for k in ('exterior_image_1_left', 'exterior_image_2_left', 'wrist_image_left')}
        original.update(cartesian_position=np.arange(6, dtype=np.float32), gripper_position=np.array([.2]))
        class Env:
            def __init__(self, robot):
                self.robot, self.i = robot, 0
            def reset(self):
                return self.get_observation()
            def get_observation(self):
                return canonical_observation(original, self.robot == 1)
            def step(self, action):
                self.i += 1
                return physical_action(np.full(7, self.i / 10), self.robot == 1), 'human' if self.i == 1 else 'policy'
            def get_info_for_step(self):
                return self.i == 2, self.robot == 0, .5, float(self.i != 2)
            def close(self):
                pass
        streamed, ends = {}, {}
        def sample(obs):
            np.testing.assert_array_equal(obs['cartesian_position'], original['cartesian_position'])
            np.testing.assert_array_equal(obs['wrist_image_left'], original['wrist_image_left'])
            return np.ones((2, 7))
        result = collect_round([Env(0), Env(1)], sample, 2, 100000, mirror_robot=1,
            on_transition=lambda r, s, t, timing: streamed.setdefault((r, s), t),
            on_episode_end=lambda r, n, ok: ends.setdefault(r, (n, ok)))
        self.assertEqual(ends, {0: (2, True), 1: (2, False)})
        for robot, (records, _) in enumerate(result):
            for i, record in enumerate(records):
                self.assertIs(record, streamed[robot, i])
                np.testing.assert_allclose(record['actions'], np.full(7, (i+1)/10))
                self.assertEqual(record['is_hil'], i == 0)



class DynamicRelayTest(unittest.TestCase):
    setUp = TransportTest.setUp
    tearDown = TransportTest.tearDown

    def configure_transport(self):
        self.state = self.root / 'relay-state.json'
        self.stats = self.root / 'transfer-stats.json'
        self.set_routes([])
        config = self.configs[0]
        config.update(relay_state_file=str(self.state), stats_file=str(self.stats),
                      parallel_connections=2, chunk_bytes=4 * 1024**2)
        config['peers'] = [['127.0.0.1', port()], config['peers'][0]]
        self.configs[1]['chunk_bytes'] = 4 * 1024**2

    def set_routes(self, active):
        temporary = self.state.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(updated=time.time(), active=active)))
        temporary.replace(self.state)
        if hasattr(self, 'transports'):
            self.transports[0].routes.checked = 0

    def wait_for(self, predicate):
        deadline = time.monotonic() + 8
        while not predicate():
            if time.monotonic() > deadline:
                self.fail('condition timed out')
            time.sleep(.01)

    def test_pause_resume_transfer_and_telemetry(self):
        # Four 4 MiB chunks, deliberately slowed only to exercise live changes.
        from unittest.mock import patch
        payload = b'x' * (16 * 1024**2)
        original = self.transports[1].store.chunk
        def slow_chunk(*args):
            time.sleep(.35)
            return original(*args)
        sender = self.transports[0]
        with patch.object(self.transports[1].store, 'chunk', side_effect=slow_chunk):
            with Buffer.from_bytes(payload) as buffer:
                self.channels[0].send_buffer('policy', 'dynamic', buffer)
            time.sleep(.2)
            self.assertFalse(sender.store.sent)
            self.assertTrue(all(c.sock is None for c in sender.bulk))
            self.set_routes([1])
            self.wait_for(lambda: sender.metrics.current and sender.metrics.current['done'] >= 4*1024**2)
            self.set_routes([])
            time.sleep(.8)
            self.assertFalse(sender.store.sent)
            self.wait_for(lambda: self.stats.exists() and
                          json.loads(self.stats.read_text())['current']['tunnels'][-1] == 0)
            self.set_routes([1])
            with self.channels[1].receive_buffer('policy', 'dynamic') as buffer, buffer.view() as view:
                self.assertEqual(view, payload)
            self.channels[0].wait_sent('policy', 'dynamic')
            self.channels[1].release('policy', 'dynamic')
            self.wait_for(lambda: json.loads(self.stats.read_text())['history'])
            stats = json.loads(self.stats.read_text())
            self.assertIsNone(stats['current'])
            row = stats['history'][-1]
            self.assertEqual(row['size'], len(payload))
            self.assertEqual(row['done'], len(payload))
            self.assertEqual(row['phase'], 'Complete')
            self.assertGreater(row['MBps'], 0)
            self.assertIn(0, row['tunnels'])
            self.assertIn(1, row['tunnels'])
            self.assertIsNone(sender.bulk[0].sock)  # unused slot never dialed


if __name__ == '__main__':
    unittest.main()
