"""RAM-to-RAM transport sidecar; no model, NumPy, JAX or payload disk I/O.

Bulk buffers are split into N contiguous ranges, streamed concurrently over N
persistent TLS connections directly into disjoint receiver RAM slices. Local
applications share those same pages by Unix descriptor passing.
"""
import argparse
from concurrent.futures import Future, TimeoutError as FutureTimeout
import fcntl
import hashlib
import hmac
import json
import logging
import math
import os
from pathlib import Path
import queue
import re
import signal
import socket
import socketserver
import ssl
import struct
import threading
import time
import traceback
import uuid

from .buffer import Buffer, receive_packet, send_packet
from .channel import atomic_json, message_id


def receive_into(sock, view):
    offset = 0
    while offset < len(view):
        n = sock.recv_into(view[offset:])
        if not n:
            raise EOFError('connection closed')
        offset += n


def exact(sock, count):
    value = bytearray(count)
    with memoryview(value) as view:
        receive_into(sock, view)
    return value


def read(sock):
    size = struct.unpack('!I', exact(sock, 4))[0]
    if size > 65536:
        raise ValueError('oversized header')
    return json.loads(exact(sock, size).decode())


def write(sock, value):
    value = json.dumps(value).encode()
    sock.sendall(struct.pack('!I', len(value)) + value)


def same_message(first, second):
    if any(first[k] != second[k] for k in ('topic', 'key', 'size', 'sha256')):
        raise ValueError('conflicting duplicate message')


class Store:
    def __init__(self, chunk_bytes, max_bytes, quota_bytes=16 * 1024**3,
                 parallel_connections=32, outbox_bytes=8 * 1024**3):
        self.chunk_bytes, self.max_bytes = chunk_bytes, max_bytes
        self.quota_bytes, self.outbox_bytes = quota_bytes, outbox_bytes
        self.parallel_connections = parallel_connections
        self.lock = threading.RLock()
        self.outbox, self.incoming, self.received, self.sent = {}, {}, {}, {}
        self.rx_bytes = self.tx_bytes = 0

    def validate(self, meta):
        ident = meta['id']
        if (not re.fullmatch('[a-f0-9]{64}', ident) or ident != message_id(meta['topic'], meta['key'])
                or not re.fullmatch('[a-f0-9]{64}', meta['sha256'])
                or type(meta['size']) is not int or not 0 < meta['size'] <= self.max_bytes):
            raise ValueError('invalid message metadata')

    def publish(self, meta, buffer):
        """Consume a sealed descriptor; no payload copy or second hashing pass."""
        try:
            self.validate(meta)
            if buffer.size != meta['size']:
                raise ValueError('buffer size mismatch')
            with self.lock:
                ident = meta['id']
                old = self.sent.get(ident) or (self.outbox[ident][0] if ident in self.outbox else None)
                if old is not None:
                    same_message(old, meta)
                    buffer.close()
                    return
                if self.tx_bytes + buffer.size > self.outbox_bytes:
                    raise ValueError('outbox RAM quota exceeded')
                self.tx_bytes += buffer.size
                self.outbox[ident] = meta, buffer
        except BaseException:
            buffer.close()
            raise

    def acknowledged(self, meta):
        with self.lock:
            self.sent[meta['id']] = meta
            _, buffer = self.outbox.pop(meta['id'])
            self.tx_bytes -= buffer.size
            buffer.close()

    def offer(self, meta):
        self.validate(meta)
        ident = meta['id']
        with self.lock:
            if ident in self.received:
                same_message(self.received[ident], meta)
                return {'complete': True}
            if ident not in self.incoming:
                if self.rx_bytes + meta['size'] > self.quota_bytes:
                    raise ValueError('receiver RAM quota exceeded')
                # As in the WAN benchmark: at most N equally sized contiguous
                # stripes, not one WAN round trip for every 4 MiB block.
                stride = max(self.chunk_bytes, math.ceil(meta['size'] / self.parallel_connections))
                count = math.ceil(meta['size'] / stride)
                self.incoming[ident] = dict(meta=meta, buffer=Buffer.create(meta['size']),
                    stride=stride, done=set(), locks=[threading.Lock() for _ in range(count)],
                    started=time.monotonic(), timings={})
                self.rx_bytes += meta['size']
            entry = self.incoming[ident]
            same_message(entry['meta'], meta)
            return {'complete': False, 'chunk_bytes': entry['stride'],
                    'missing': [i for i in range(len(entry['locks'])) if i not in entry['done']]}

    def chunk(self, sock, header):
        ident, index = header['id'], header['index']
        with self.lock:
            entry = self.incoming[ident]
            if type(index) is not int or not 0 <= index < len(entry['locks']):
                raise ValueError('invalid chunk index')
        offset = index * entry['stride']
        size = min(entry['stride'], entry['meta']['size'] - offset)
        # Serialize only duplicate writers of the SAME stripe, never the N
        # independent streams. A failed stream can overwrite its partial stripe.
        with entry['locks'][index]:
            if index in entry['done']:
                # Drain an overlapping retry without modifying committed bytes.
                remaining = size
                while remaining:
                    block = exact(sock, min(remaining, 1024**2))
                    remaining -= len(block)
                return
            with entry['buffer'].view(offset, offset + size) as view:
                receive_into(sock, view)
            with self.lock:
                entry['done'].add(index)
                if len(entry['done']) == len(entry['locks']):
                    entry['timings']['receive_seconds'] = time.monotonic() - entry['started']

    def commit(self, meta):
        # Admission is atomic. No application sees a partially received buffer.
        with self.lock:
            status = self.offer(meta)
            if status['complete']:
                return self.received[meta['id']].get('timings', {})
            if status['missing']:
                raise ValueError('incomplete message')
            entry = self.incoming[meta['id']]
            started = time.monotonic()
            if entry['buffer'].digest() != meta['sha256']:
                entry['done'].clear()
                raise ValueError('message checksum mismatch')
            entry['buffer'].seal()
            entry['timings']['verify_seconds'] = time.monotonic() - started
            self.received[meta['id']] = dict(meta, timings=entry['timings'])
            return entry['timings']

    def small(self, sock, meta):
        self.validate(meta)
        if meta['size'] > self.chunk_bytes:
            raise ValueError('small message too large')
        data = exact(sock, meta['size'])
        if hashlib.sha256(data).hexdigest() != meta['sha256']:
            raise ValueError('message checksum mismatch')
        with self.lock:
            if meta['id'] in self.received:
                same_message(self.received[meta['id']], meta)
                return
            self.offer(meta)
            entry = self.incoming[meta['id']]
            with entry['buffer'].view() as view:
                view[:] = data
            entry['buffer'].seal()
            self.received[meta['id']] = meta

    def release(self, ident):
        with self.lock:
            if ident not in self.received:
                raise ValueError('cannot release an incomplete message')
            entry = self.incoming.pop(ident, None)
            if entry:
                self.rx_bytes -= entry['buffer'].size
                entry['buffer'].close()

    def close(self):
        for _, buffer in self.outbox.values():
            buffer.close()
        for entry in self.incoming.values():
            entry['buffer'].close()
        self.outbox.clear()
        self.incoming.clear()


class PeerEpoch:
    def __init__(self):
        self.value = None
        self.lock = threading.Lock()

    def check(self, value):
        with self.lock:
            if self.value is not None and self.value != value:
                raise ValueError('peer RAM transport restarted; start a fresh session')
            self.value = value


class Connection:
    def __init__(self, endpoint, token, context, timeout, peer_epoch, epoch, connect_timeout=5):
        self.endpoint, self.token, self.context, self.timeout = endpoint, token, context, timeout
        self.connect_timeout = min(connect_timeout, timeout)
        self.peer_epoch, self.epoch = peer_epoch, epoch
        self.sock = None
        self.lock = threading.Lock()

    def close(self):
        sock, self.sock = self.sock, None
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def call(self, header, data=None):
        with self.lock:
            try:
                if self.sock is None:
                    sock = socket.create_connection(tuple(self.endpoint), timeout=self.connect_timeout)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self.sock = sock
                    self.sock = self.context.wrap_socket(sock, server_hostname='expo-link') if self.context else sock
                    self.sock.settimeout(self.timeout)
                write(self.sock, dict(header, token=self.token, epoch=self.epoch))
                if data is not None:
                    self.sock.sendall(data)
                reply = read(self.sock)
                self.peer_epoch.check(reply['epoch'])
                if 'error' in reply:
                    raise ValueError(reply['error'])
                return reply
            except BaseException:
                self.close()
                raise


class TransferPool:
    """Independent link workers sharing retryable, idempotent transfer jobs.

    A failed link returns its job before backing off, so another link can take
    it. Only network failures retry; protocol/auth/epoch failures remain fatal.
    Buffers stay owned by the caller until every submitted job completes.
    """
    def __init__(self, connections, stop, fallback_peers=None):
        self.connections, self.stop = connections, stop
        self.fallback_peers = fallback_peers
        self.closed = threading.Event()
        self.error = None
        self.jobs = queue.Queue()
        self.workers = [threading.Thread(target=self._worker, args=(conn,)) for conn in connections]

    def __enter__(self):
        for worker in self.workers:
            worker.start()
        return self

    def submit(self, operation):
        future = Future()
        self.jobs.put((future, operation))
        return future

    def result(self, future):
        while not self.stop.is_set():
            self.check()
            try:
                return future.result(timeout=.1)
            except FutureTimeout:
                if future.done():
                    raise  # The operation itself raised TimeoutError.
        raise RuntimeError('transport stopped')

    def check(self):
        if self.error is not None:
            raise self.error

    def _worker(self, conn):
        delay = .2
        while not self.stop.is_set() and not self.closed.is_set() and self.error is None:
            try:
                future, operation = self.jobs.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                result = operation(conn)
            except (OSError, EOFError, TimeoutError) as exc:
                # ssl.sendall can leave sliced memoryviews in its exception
                # traceback. Release those frames BEFORE another worker can
                # retry, acknowledge and close this message's mmap.
                reason = str(exc)
                traceback.clear_frames(exc.__traceback__)
                exc.__traceback__ = None
                # Publish the retry BEFORE this worker waits for its broken link.
                self.jobs.put((future, operation))
                logging.warning('Link %s retry: %s', conn.endpoint, reason)
                if self.fallback_peers:
                    # A small pool must reach all relay endpoints, not just 0..3.
                    index = self.fallback_peers.index(conn.endpoint)
                    conn.endpoint = self.fallback_peers[(index + 1) % len(self.fallback_peers)]
                self.closed.wait(delay)
                delay = min(5, delay * 2)
            except BaseException as exc:
                self.error = exc
                future.set_exception(exc)
            else:
                delay = .2
                future.set_result(result)
            finally:
                self.jobs.task_done()

    def __exit__(self, *_):
        self.closed.set()
        for conn in self.connections:
            conn.close()
        for worker in self.workers:
            worker.join()
        while True:
            try:
                future, _ = self.jobs.get_nowait()
            except queue.Empty:
                break
            future.cancel()
            self.jobs.task_done()


class Transport:
    def __init__(self, config):
        self.config = config
        self.root = Path(config['mailbox'])
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.epoch = uuid.uuid4().hex
        self.peer_epoch = PeerEpoch()
        self.token = Path(config['token_file']).read_text().strip()
        if len(self.token) < 32:
            raise ValueError('use a random token of at least 32 characters')
        chunk_bytes = config.get('chunk_bytes', 4 * 1024**2)
        count = config.get('parallel_connections', 32)
        if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= 16 * 1024**2:
            raise ValueError('chunk_bytes must be 1..16 MiB')
        if not 1 <= count <= 64 or not 1 <= config.get('record_connections', 4) <= 64 or not config['peers']:
            raise ValueError('require 1..64 connections and at least one endpoint')
        self.store = Store(chunk_bytes, config.get('max_message_bytes', 8 * 1024**3),
                           config.get('max_pending_bytes', 16 * 1024**3), count,
                           config.get('max_outbox_bytes', 8 * 1024**3))
        self.stop = threading.Event()
        self.active = set()
        self.active_lock = threading.Lock()
        self.server_context = self.client_context = None
        if config.get('tls'):
            tls = config['tls']
            self.server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            self.server_context.load_cert_chain(tls['cert_file'], tls['key_file'])
            self.client_context = ssl.create_default_context(cafile=tls['ca_file'])
            self.client_context.check_hostname = False
        elif not (config.get('allow_plain_loopback') and config['listen'][0] == '127.0.0.1'
                  and all(e[0] == '127.0.0.1' for e in config['peers'])):
            raise ValueError('TLS required outside explicit loopback tests')
        def connection(i):
            return Connection(config['peers'][i % len(config['peers'])], self.token,
                              self.client_context, config.get('socket_timeout', 120), self.peer_epoch, self.epoch,
                              connect_timeout=config.get('connect_timeout', 5))
        self.bulk = [connection(i) for i in range(count)]
        self.small = [connection(i) for i in range(config.get('record_connections', 4))]

    def _send_small(self, meta, connection):
        buffer = self.store.outbox[meta['id']][1]
        with buffer.view() as view:
            connection.call({'op': 'small', 'meta': meta}, view)
        self.store.acknowledged(meta)

    def _send_bulk(self, meta, pool):
        buffer = self.store.outbox[meta['id']][1]
        started = time.monotonic()
        status = pool.result(pool.submit(lambda conn: conn.call({'op': 'offer', 'meta': meta})))
        if not status['complete']:
            def stream(conn, index):
                offset = index * status['chunk_bytes']
                with buffer.view(offset, min(offset + status['chunk_bytes'], buffer.size)) as data:
                    return conn.call({'op': 'chunk', 'id': meta['id'], 'index': index}, data)
            futures = [pool.submit(lambda conn, index=index: stream(conn, index)) for index in status['missing']]
            for future in futures:
                pool.result(future)
            result = pool.result(pool.submit(lambda conn: conn.call({'op': 'commit', 'meta': meta})))
            logging.info('RAM_TRANSFER bytes=%d connections=%d send_through_verify_seconds=%.6f receiver=%s',
                         buffer.size, len(self.bulk), time.monotonic() - started, result.get('timings'))
        self.store.acknowledged(meta)

    def _sender(self, bulk):
        connections = self.bulk if bulk else self.small
        # Bulk workers map to the independent WAN links. Small/control workers
        # can rotate through every endpoint even when their pool is smaller.
        fallback = None if bulk and len(connections) >= len(self.config['peers']) else self.config['peers']
        with TransferPool(connections, self.stop, fallback) as pool:
            pending = {}
            while not self.stop.is_set():
                try:
                    pool.check()
                    for ident, future in list(pending.items()):
                        if future.done():
                            future.result()
                            del pending[ident]
                    with self.store.lock:
                        rows = [m for m, _ in self.store.outbox.values() if (m['size'] > self.store.chunk_bytes) == bulk]
                    rows.sort(key=lambda m: m['created'])
                    if bulk and rows:
                        self._send_bulk(rows[0], pool)
                    else:
                        for meta in rows:
                            if meta['id'] not in pending and len(pending) < len(connections):
                                pending[meta['id']] = pool.submit(lambda conn, meta=meta: self._send_small(meta, conn))
                        self.stop.wait(.005)
                except BaseException as exc:
                    if not self.stop.is_set():
                        logging.exception('Transport sender failed')
                        atomic_json(self.root / 'transport-error.json', {'error': str(exc)})
                    self.stop.set()
                    return

    def _local(self, request, fd, sock):
        buffer = Buffer.from_fd(fd) if fd is not None else None
        try:
            if request.get('epoch') not in (None, self.epoch):
                raise ValueError('RAM transport restarted; start a fresh session')
            abort = request.get('abort_key')
            with self.store.lock:
                if abort and message_id('abort', abort) in self.store.received:
                    raise ValueError('peer aborted this session; restart with a fresh session ID')
                op, ident = request['op'], request.get('id')
                if op == 'publish':
                    if buffer is None:
                        raise ValueError('publish requires a sealed buffer')
                    payload, buffer = buffer, None
                    self.store.publish(request['meta'], payload)
                    send_packet(sock, {'epoch': self.epoch, 'ok': True})
                elif op == 'receive':
                    if ident not in self.store.received:
                        send_packet(sock, {'epoch': self.epoch, 'ready': False})
                    elif ident not in self.store.incoming:
                        raise ValueError('payload already released')
                    else:
                        send_packet(sock, dict(epoch=self.epoch, ready=True, meta=self.store.received[ident]),
                                    self.store.incoming[ident]['buffer'].fd)
                elif op == 'release':
                    self.store.release(ident)
                    send_packet(sock, {'epoch': self.epoch, 'ok': True})
                elif op == 'sent':
                    send_packet(sock, {'epoch': self.epoch, 'ready': ident in self.store.sent})
                elif op == 'check':
                    send_packet(sock, {'epoch': self.epoch, 'ok': True})
                else:
                    raise ValueError('unknown local operation')
        finally:
            if buffer is not None:
                buffer.close()

    def run(self):
        process_lock = (self.root / 'transport.lock').open('a')
        fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (self.root / 'transport-error.json').exists():
            process_lock.close()
            raise RuntimeError('mailbox contains a fatal transport error; start a fresh session')
        owner = self
        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                sock = self.request
                sock.settimeout(owner.config.get('socket_timeout', 120))
                try:
                    if owner.server_context:
                        sock = owner.server_context.wrap_socket(sock, server_side=True)
                    with owner.active_lock:
                        owner.active.add(sock)
                    while not owner.stop.is_set():
                        h = read(sock)
                        if not hmac.compare_digest(h.pop('token', ''), owner.token):
                            raise ValueError('authentication failed')
                        owner.peer_epoch.check(h['epoch'])
                        op = h['op']
                        if op == 'offer':
                            result = owner.store.offer(h['meta'])
                        elif op == 'chunk':
                            owner.store.chunk(sock, h)
                            result = {'ok': True}
                        elif op == 'commit':
                            result = {'ok': True, 'timings': owner.store.commit(h['meta'])}
                        elif op == 'small':
                            owner.store.small(sock, h['meta'])
                            result = {'ok': True}
                        else:
                            raise ValueError('unknown operation')
                        write(sock, dict(result, epoch=owner.epoch))
                except (OSError, EOFError):
                    pass
                except Exception as exc:
                    try:
                        write(sock, {'error': str(exc), 'epoch': owner.epoch})
                    except OSError:
                        pass
                finally:
                    with owner.active_lock:
                        owner.active.discard(sock)
                    sock.close()
        class LocalHandler(socketserver.BaseRequestHandler):
            def handle(self):
                self.request.settimeout(10)
                try:
                    request, fd = receive_packet(self.request)
                    owner._local(request, fd, self.request)
                except Exception as exc:
                    try:
                        send_packet(self.request, {'error': str(exc), 'epoch': owner.epoch})
                    except OSError:
                        pass
        class Server(socketserver.ThreadingTCPServer):
            request_queue_size = 128
            allow_reuse_address = True
        class LocalServer(socketserver.ThreadingUnixStreamServer):
            socket_type = socket.SOCK_SEQPACKET
            request_queue_size = 128
        socket_path = self.root / 'channel.sock'
        socket_path.unlink(missing_ok=True)
        with Server(tuple(self.config['listen']), Handler) as server, LocalServer(str(socket_path), LocalHandler) as local:
            socket_path.chmod(0o600)
            servers = [threading.Thread(target=s.serve_forever, kwargs={'poll_interval': .1}) for s in (server, local)]
            senders = [threading.Thread(target=self._sender, args=(b,)) for b in (False, True)]
            for thread in servers + senders:
                thread.start()
            atomic_json(self.root / 'transport-ready.json', dict(pid=os.getpid(), epoch=self.epoch, address=server.server_address))
            try:
                self.stop.wait()
            finally:
                self.stop.set()
                for conn in self.bulk + self.small:
                    conn.close()
                with self.active_lock:
                    for sock in self.active:
                        try:
                            sock.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                for s in (server, local):
                    s.shutdown()
                for thread in servers + senders:
                    thread.join()
        self.store.close()
        socket_path.unlink(missing_ok=True)
        (self.root / 'transport-ready.json').unlink(missing_ok=True)
        process_lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    transport = Transport(json.loads(Path(args.config).read_text()))
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: transport.stop.set())
    transport.run()


if __name__ == '__main__':
    main()
