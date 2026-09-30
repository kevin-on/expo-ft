"""Application handoff via shared host RAM, independent of WAN deployment.

Only tiny session/diagnostic markers live in the mailbox directory. Payloads are
anonymous sealed RAM buffers passed to the local sidecar over a Unix socket.
"""
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import socket
import threading
import time
import uuid

from .buffer import Buffer, receive_packet, send_packet


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def message_id(topic, key):
    return hashlib.sha256(json.dumps([topic, key], separators=(',', ':')).encode()).hexdigest()


def encode(value):
    import msgpack
    import numpy as np
    def default(obj):
        if isinstance(obj, np.ndarray):
            a = np.ascontiguousarray(obj)
            if a.dtype.hasobject:
                raise TypeError('object arrays are not transferable')
            return {'__array__': True, 'dtype': a.dtype.name, 'shape': a.shape, 'data': a.tobytes()}
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj))
    return msgpack.packb(value, default=default, use_bin_type=True)


def decode(value):
    import msgpack
    import numpy as np
    def hook(obj):
        if obj.get('__array__') is True:
            dtype = np.dtype(obj['dtype'])
            if dtype.hasobject:
                raise ValueError('object array')
            return np.frombuffer(obj['data'], dtype=dtype).reshape(obj['shape']).copy()
        return obj
    return msgpack.unpackb(value, raw=False, object_hook=hook, strict_map_key=False)


class Channel:
    """Bounded background record serialization; immutable buffers for policies.

    No payload is spooled to disk. The sidecar owns pending buffers until remote
    receipt; the receiver owns its buffer until application release. A sidecar
    restart loses volatile state and aborts the session instead of replaying it.
    """
    def __init__(self, directory, capacity=128, timeout=600):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.timeout = timeout
        self.abort_key = None
        self.epoch = None
        self.epoch_lock = threading.Lock()
        self.queue = queue.Queue(maxsize=capacity)
        self.error = None
        self.closed = False
        self.worker = threading.Thread(target=self._writer, daemon=True)
        self.worker.start()

    def _check(self):
        if self.error is not None:
            raise RuntimeError('local RAM publisher failed') from self.error
        failure = self.root / 'transport-error.json'
        if failure.exists():
            raise RuntimeError(failure.read_text())

    def _rpc(self, op, fd=None, rpc_timeout=None, **fields):
        self._check()
        timeout = self.timeout if rpc_timeout is None else min(self.timeout, rpc_timeout)
        deadline = time.monotonic() + timeout
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as sock:
            sock.settimeout(timeout)
            while True:
                try:
                    sock.connect(str(self.root / 'channel.sock'))
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    self._check()
                    if self.epoch is not None:
                        raise RuntimeError('RAM transport stopped; start a fresh session')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('waiting for local RAM transport')
                    time.sleep(.02)
            send_packet(sock, dict(fields, op=op, epoch=self.epoch, abort_key=self.abort_key), fd)
            reply, received_fd = receive_packet(sock)
        try:
            if 'error' in reply:
                raise RuntimeError(reply['error'])
            with self.epoch_lock:
                if self.epoch is not None and self.epoch != reply['epoch']:
                    raise RuntimeError('RAM transport restarted; start a fresh session')
                self.epoch = reply['epoch']
        except BaseException:
            if received_fd is not None:
                os.close(received_fd)
            raise
        return reply, received_fd

    def check_session(self):
        """Observe peer aborts even when no records are being published.

        This queries only the local sidecar, not the remote learner. Bound the
        wait separately from long policy/round transfer timeouts.
        """
        self._rpc('check', rpc_timeout=5)

    def start_session(self, session):
        self.session_lock = (self.root / 'application.lock').open('a')
        fcntl.flock(self.session_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = self.root / 'session.json'
        if marker.exists():
            raise ValueError('application mailbox already used; choose a fresh session and mailbox')
        atomic_json(marker, {'session': session})
        self.abort_key = session

    def _writer(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                topic, key, value = item
                with Buffer.from_bytes(encode(value)) as buffer:
                    self.send_buffer(topic, key, buffer)
            except BaseException as exc:
                self.error = exc
            finally:
                self.queue.task_done()

    def send(self, topic, key, value):
        self._check()
        if self.closed:
            raise RuntimeError('channel closed')
        self.queue.put_nowait((topic, key, deepcopy(value)))

    def send_buffer(self, topic, key, buffer):
        """Share sealed pages with the sidecar, with no full-payload IPC copy."""
        buffer.seal()
        meta = dict(id=message_id(topic, key), topic=topic, key=key, size=buffer.size,
                    xxh3_128=buffer.digest(), created=time.time())
        self._rpc('publish', fd=buffer.fd, meta=meta)

    def flush(self):
        self.queue.join()
        self._check()

    def _wait(self, op, topic, key, timeout, check=None):
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            if check is not None:
                check()
            result, fd = self._rpc(op, id=message_id(topic, key))
            if result['ready']:
                return result, fd
            if time.monotonic() >= deadline:
                raise TimeoutError('waiting for {} {} {}'.format(op, topic, key))
            time.sleep(.02)

    def receive_buffer(self, topic, key, timeout=None, *, check=None):
        # Allow the application to surface background reset failures while waiting
        # for the learner. The callback performs no transport or policy work.
        result, fd = self._wait('receive', topic, key, timeout, check)
        buffer = Buffer.from_fd(fd)
        buffer.timings = result['meta'].get('timings', {})
        if result['meta']['topic'] != topic or result['meta']['key'] != key:
            buffer.close()
            raise ValueError('message identity mismatch')
        return buffer

    def receive(self, topic, key, timeout=None, *, check=None):
        with self.receive_buffer(topic, key, timeout, check=check) as buffer, buffer.view() as view:
            return decode(view)

    def progress(self, topic, key):
        result, _ = self._rpc('progress', id=message_id(topic, key))
        return result

    def poll_buffer(self, topic, key):
        result, fd = self._rpc('receive', id=message_id(topic, key))
        if not result['ready']:
            return None
        buffer = Buffer.from_fd(fd)
        buffer.timings = result['meta'].get('timings', {})
        return buffer

    def poll(self, topic):
        """Take one small control message, without imposing a producer sequence."""
        result, fd = self._rpc('next', topic=topic)
        if not result['ready']:
            return None
        try:
            with Buffer.from_fd(fd) as buffer, buffer.view() as view:
                value = decode(view)
        finally:
            self.release(topic, result['meta']['key'])
        return result['meta']['key'], value

    def release(self, topic, key):
        """Release receiver pages after consumption; retain a small RAM receipt."""
        self._rpc('release', id=message_id(topic, key))

    def wait_sent(self, topic, key, timeout=None):
        self._wait('sent', topic, key, timeout)

    def close(self):
        if not self.closed:
            self.closed = True
            self.queue.put(None)
            self.worker.join()
            if hasattr(self, 'session_lock'):
                self.session_lock.close()
        self._check()
