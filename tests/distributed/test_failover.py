"""Fault injection on CPU/loopback; never run beside WS robot experiments."""
from collections import Counter
from contextlib import ExitStack
import socket
import threading
import time
import unittest
from unittest.mock import patch

from expo_ft.distributed.buffer import Buffer
from expo_ft.distributed.transport import TransferPool, exact
import test_transport as fixtures


class FailoverTest(unittest.TestCase):
    setUp = fixtures.TransportTest.setUp
    tearDown = fixtures.TransportTest.tearDown

    def configure_transport(self):
        dead = []
        for _ in range(3):
            sock = socket.socket()
            sock.bind(('127.0.0.1', 0))  # Reserve the address WITHOUT a listener.
            self.addCleanup(sock.close)
            dead.append(list(sock.getsockname()))
        good = self.configs[0]['peers'][0]
        self.configs[0].update(peers=dead + [good] * 29, record_connections=2)
        # 32 bulk workers, first three routes permanently unavailable. The two
        # record workers also start on dead routes and must reach other peers.

    def test_transfer_completes_with_three_routes_permanently_down(self):
        a, b = self.channels
        payload = b'x' * (4 * 1024**2)
        with Buffer.from_bytes(payload) as buffer:
            a.send_buffer('policy', 'v1', buffer)
        for i in range(6):
            a.send('record', str(i), {'value': i})
        a.flush()
        with b.receive_buffer('policy', 'v1', timeout=8) as buffer, buffer.view() as view:
            self.assertEqual(view, payload)
        a.wait_sent('policy', 'v1', timeout=8)
        b.release('policy', 'v1')
        for i in range(6):
            self.assertEqual(b.receive('record', str(i), timeout=8), {'value': i})
            a.wait_sent('record', str(i))
            b.release('record', str(i))
        self.assertFalse(any(t.stop.is_set() for t in self.transports))
        self.assertEqual([t.store.tx_bytes + t.store.rx_bytes for t in self.transports], [0, 0])

    def test_failed_send_traceback_releases_views_before_retry_ack(self):
        a, b = self.channels
        errors, lock = [], threading.Lock()
        def failed_send(data):
            # ssl.sendall retains a derived view like this when a write fails.
            # Keep the exception alive to test prompt cleanup, not GC timing.
            derived = memoryview(data)[1:]
            try:
                raise BrokenPipeError('injected send with a retained memoryview')
            except BrokenPipeError as exc:
                errors.append(exc)
                raise
        def fail_once(original):
            def call(header, data=None):
                with lock:
                    if header['op'] == 'small' and not errors:
                        failed_send(data)
                return original(header, data)
            return call
        try:
            with ExitStack() as patches:
                for conn in self.transports[0].small:
                    patches.enter_context(patch.object(conn, 'call', side_effect=fail_once(conn.call)))
                a.send('record', 'retained-view', {'value': b'x' * 16384})
                a.flush()
                self.assertEqual(b.receive('record', 'retained-view', timeout=8), {'value': b'x' * 16384})
                a.wait_sent('record', 'retained-view', timeout=8)
                b.release('record', 'retained-view')
                self.assertEqual(len(errors), 1)
                self.assertFalse(any(t.stop.is_set() for t in self.transports))
                self.assertEqual(self.transports[0].store.tx_bytes, 0)
        finally:
            errors.clear()

    def test_partial_range_and_lost_commit_ack_retry_without_full_resend(self):
        a, b = self.channels
        receiver = self.transports[1].store
        original_chunk = receiver.chunk
        counts, damaged, commits = Counter(), [], []
        lock = threading.Lock()
        def chunk(sock, header):
            with lock:
                counts[header['index']] += 1
                fail = not damaged
                if fail:
                    damaged.append(header['index'])
            if fail:
                entry = receiver.incoming[header['id']]
                offset = header['index'] * entry['stride']
                with entry['buffer'].view(offset, offset + 64) as view:
                    view[:] = exact(sock, 64)
                raise EOFError('injected disconnect after partial receive')
            return original_chunk(sock, header)
        def lose_first_commit_reply(original):
            def call(header, data=None):
                response = original(header, data)
                if header['op'] == 'commit':
                    commits.append(True)
                    if len(commits) == 1:
                        raise EOFError('injected lost commit ACK')
                return response
            return call
        with ExitStack() as patches:
            patches.enter_context(patch.object(receiver, 'chunk', side_effect=chunk))
            for conn in self.transports[0].bulk:
                patches.enter_context(patch.object(conn, 'call', side_effect=lose_first_commit_reply(conn.call)))
            payload = bytes(range(256)) * 16384
            with Buffer.from_bytes(payload) as buffer:
                a.send_buffer('policy', 'partial', buffer)
            with b.receive_buffer('policy', 'partial', timeout=8) as buffer, buffer.view() as view:
                self.assertEqual(view, payload)
            # Receipt/dedup must still work if the application releases its RAM
            # before the sender successfully receives the retried commit ACK.
            b.release('policy', 'partial')
            a.wait_sent('policy', 'partial', timeout=8)
        self.assertEqual(len(commits), 2)
        # 64 chunks over 32 workers; retries must not resend healthy chunks.
        self.assertEqual(len(counts), 64)
        self.assertEqual(counts[damaged[0]], 2)
        self.assertEqual(sum(counts.values()), 65)

    def test_records_keep_flowing_while_another_rpc_waits(self):
        a, b = self.channels
        receiver = self.transports[1].store
        entered, release = threading.Event(), threading.Event()
        original = receiver.small
        def stall_one(sock, meta):
            if meta['key'] == 'stalled':
                entered.set()
                if not release.wait(8):
                    raise TimeoutError('test stalled RPC')
            return original(sock, meta)
        with patch.object(receiver, 'small', side_effect=stall_one):
            try:
                a.send('record', 'stalled', {'value': -1})
                a.flush()
                self.assertTrue(entered.wait(5))
                for i in range(6):
                    a.send('record', str(i), {'value': i})
                a.flush()
                for i in range(6):
                    self.assertEqual(b.receive('record', str(i), timeout=4), {'value': i})
                self.assertFalse(release.is_set())
            finally:
                release.set()
            self.assertEqual(b.receive('record', 'stalled'), {'value': -1})
            a.wait_sent('record', 'stalled')

    def test_total_outage_keeps_ram_until_links_recover(self):
        a, b = self.channels
        online, attempted = threading.Event(), threading.Event()
        def gate(original):
            def call(*args, **kwargs):
                if not online.is_set():
                    attempted.set()
                    raise ConnectionRefusedError('injected total outage')
                return original(*args, **kwargs)
            return call
        with ExitStack() as patches:
            for conn in self.transports[0].bulk + self.transports[0].small:
                patches.enter_context(patch.object(conn, 'call', side_effect=gate(conn.call)))
            with Buffer.from_bytes(b'x' * (256 * 1024)) as buffer:
                a.send_buffer('policy', 'outage', buffer)
            self.assertTrue(attempted.wait(5))
            self.assertGreater(self.transports[0].store.tx_bytes, 0)
            self.assertFalse(self.transports[0].stop.is_set())
            online.set()
            with b.receive_buffer('policy', 'outage', timeout=8) as buffer, buffer.view() as view:
                self.assertEqual(view, b'x' * (256 * 1024))
            a.wait_sent('policy', 'outage')


class TLSFailoverTest(FailoverTest):
    def configure_transport(self):
        super().configure_transport()
        fixtures.TLSTransportTest.configure_transport(self)


class PoolTest(unittest.TestCase):
    def test_fast_worker_takes_more_chunks_while_slow_worker_waits(self):
        class Link:
            def __init__(self, name):
                self.endpoint = name
            def close(self):
                pass
        entered, release, drained = threading.Event(), threading.Event(), threading.Event()
        counts = Counter()
        def send(conn):
            counts[conn.endpoint] += 1
            if conn.endpoint == 'slow':
                entered.set()
                if not release.wait(5):
                    raise AssertionError('slow link not released')
            else:
                if not entered.wait(5):
                    raise AssertionError('slow link did not start')
                if counts['fast'] == 12:
                    drained.set()
        with TransferPool([Link('slow'), Link('fast')], threading.Event()) as pool:
            jobs = [pool.submit(send) for _ in range(13)]
            try:
                self.assertTrue(entered.wait(5))
                self.assertTrue(drained.wait(5))
                self.assertFalse(all(job.done() for job in jobs))
                self.assertEqual(counts['slow'], 1)
            finally:
                release.set()
            for job in jobs:
                pool.result(job)

    def test_repaired_worker_rejoins_while_other_link_continues(self):
        class Link:
            def __init__(self, name):
                self.endpoint = name
            def close(self):
                pass
        repaired, failed, rejoined = threading.Event(), threading.Event(), threading.Event()
        def send(conn):
            if conn.endpoint == 'broken':
                if not repaired.is_set():
                    failed.set()
                    raise EOFError('broken')
                rejoined.set()
            time.sleep(.01)
            return 1
        with TransferPool([Link('broken'), Link('healthy')], threading.Event()) as pool:
            jobs = [pool.submit(send) for _ in range(16)]
            self.assertEqual(sum(pool.result(job) for job in jobs), 16)
            self.assertTrue(failed.is_set())
            repaired.set()
            deadline = time.monotonic() + 3
            while not rejoined.is_set() and time.monotonic() < deadline:
                jobs = [pool.submit(send) for _ in range(8)]
                self.assertEqual(sum(pool.result(job) for job in jobs), 8)
            self.assertTrue(rejoined.is_set())

    def test_protocol_failure_is_fatal_without_retry(self):
        class Link:
            endpoint = 'test'
            def close(self):
                pass
        calls = []
        def send(conn):
            calls.append(conn)
            raise ValueError('peer RAM transport restarted')
        with TransferPool([Link()], threading.Event()) as pool:
            job = pool.submit(send)
            with self.assertRaisesRegex(ValueError, 'restarted'):
                pool.result(job)
        self.assertEqual(len(calls), 1)
