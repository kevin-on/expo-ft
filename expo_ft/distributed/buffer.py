"""Anonymous host RAM shared with a sidecar by passing a descriptor, not bytes.

Linux memfd has no disk pathname or backing payload file. It lets independently
launched processes map the same pages, including inside a container (no shared
/dev/shm mount required). Sealing makes a published snapshot immutable.
"""
import array
import fcntl
import json
import mmap
import os
import socket
import threading

import xxhash


class Buffer:
    def __init__(self, fd, writable=False):
        self.fd = fd
        self.mapping = None
        self.writable = writable
        self.mapping_lock = threading.Lock()

    @classmethod
    def create(cls, size=0):
        fd = os.memfd_create('expo-payload', os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        os.ftruncate(fd, size)
        return cls(fd, writable=True)

    @classmethod
    def from_bytes(cls, data):
        result = cls.create(len(data))
        try:
            with result.view() as view:
                view[:] = data
            result.seal()
            return result
        except BaseException:
            result.close()
            raise

    @classmethod
    def from_fd(cls, fd):
        """Take ownership only of an immutable, size-sealed shared RAM buffer."""
        required = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK
        try:
            if fcntl.fcntl(fd, fcntl.F_GET_SEALS) & required != required:
                raise ValueError('shared payload is not sealed')
            return cls(fd)
        except BaseException:
            os.close(fd)
            raise

    @property
    def size(self):
        return os.fstat(self.fd).st_size

    def open(self):
        """File-like serialization interface over RAM; never a disk file."""
        f = os.fdopen(os.dup(self.fd), 'r+b' if self.writable else 'rb')
        f.seek(0)
        return f

    def view(self, start=0, end=None):
        with self.mapping_lock:
            if self.mapping is None:
                # Sealed buffers are immutable. A read-only private mapping
                # references the same pages without writable shared-map rights
                # (rejected by F_SEAL_WRITE on the GH200 kernel). PROT_READ also
                # forbids writes/COW, so this does not copy the payload.
                self.mapping = mmap.mmap(self.fd, self.size,
                    flags=mmap.MAP_SHARED if self.writable else mmap.MAP_PRIVATE,
                    prot=mmap.PROT_READ | (mmap.PROT_WRITE if self.writable else 0))
        return memoryview(self.mapping)[start:end]

    def seal(self):
        if self.writable:
            if self.mapping is not None:
                self.mapping.close()
                self.mapping = None
            fcntl.fcntl(self.fd, fcntl.F_ADD_SEALS,
                        fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
            self.writable = False

    def digest(self):
        with self.view() as view:
            return xxhash.xxh3_128(view).hexdigest()

    def close(self):
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def send_packet(sock, value, fd=None):
    data = json.dumps(value).encode()
    if len(data) > 65536:
        raise ValueError('oversized local control packet')
    ancillary = [] if fd is None else [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [fd]))]
    if sock.sendmsg([data], ancillary) != len(data):
        raise EOFError('incomplete local control packet')


def receive_packet(sock):
    data, ancillary, flags, _ = sock.recvmsg(65536, socket.CMSG_SPACE(array.array('i').itemsize),
                                          socket.MSG_CMSG_CLOEXEC)
    fds = []
    for level, kind, value in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            received = array.array('i')
            received.frombytes(value[:len(value) - len(value) % received.itemsize])
            fds.extend(received)
    try:
        if not data or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or len(fds) > 1:
            raise ValueError('invalid local control packet')
        result = json.loads(data)
    except BaseException:
        for fd in fds:
            os.close(fd)
        raise
    return result, fds[0] if fds else None
