"""Optional, best-effort transfer telemetry for the relay terminal (no payload I/O)."""
import json
import logging
from pathlib import Path
import threading
import time


class RouteState:
    def __init__(self, path=None):
        self.path = path
        self.lock = threading.Lock()
        self.checked = 0
        self.active = []

    def indices(self):
        with self.lock:
            if time.monotonic() - self.checked >= .5:
                self.checked = time.monotonic()
                try:
                    value = json.loads(Path(self.path).read_text())
                    self.active = value['active'] if time.time() - value['updated'] < 3 else []
                except (OSError, ValueError, KeyError):
                    self.active = []
            return list(self.active)


class TransferStatus:
    """Update counters on chunk ACK; a separate thread writes at most twice/second."""
    def __init__(self, path, routes, fallback_count):
        self.path, self.routes, self.fallback_count = path, routes, fallback_count
        self.lock = threading.Lock()
        self.current = None
        self.history = []
        self.last_sample = time.monotonic()
        self.last_done = 0
        self.rate = 0

    def begin(self, meta):
        with self.lock:
            self.current = dict(size=meta['size'], done=0, started=time.monotonic(),
                                phase='Sending', tunnels=[])
            self.last_sample = self.current['started']
            self.last_done = self.rate = 0

    def acknowledged(self, size):
        with self.lock:
            if self.current is not None:
                self.current['done'] += size

    def verifying(self):
        with self.lock:
            self.current['phase'] = 'Verifying'

    def finish(self, timings=None, failed=False):
        with self.lock:
            row = self.current
            if row is None:
                return
            elapsed = (timings or {}).get('receive_seconds', time.monotonic() - row['started'])
            self.history.append(dict(size=row['size'], done=row['done'], elapsed=elapsed,
                                     MBps=(row['done'] if failed else row['size']) / max(elapsed, 1e-9) / 1e6,
                                     tunnels=list(row['tunnels']), phase='Failed' if failed else 'Complete'))
            self.history = self.history[-10:]
            self.current = None

    def snapshot(self):
        count = len(self.routes.indices()) if self.routes.path else self.fallback_count
        now = time.monotonic()
        with self.lock:
            row = self.current
            current = None
            if row:
                if not row['tunnels'] or row['tunnels'][-1] != count:
                    row['tunnels'].append(count)
                delta = now - self.last_sample
                if delta >= .4:
                    self.rate = (row['done'] - self.last_done) / delta / 1e6
                    self.last_done, self.last_sample = row['done'], now
                current = dict(row, MBps=self.rate, elapsed=now - row['started'])
            return dict(updated=time.time(), current=current, history=list(self.history))

    def run(self, stop):
        output = Path(self.path)
        while True:
            try:
                # Volatile metrics only: no fsync and never block chunk workers on filesystem I/O.
                temp = output.with_suffix('.tmp')
                temp.write_text(json.dumps(self.snapshot()))
                temp.replace(output)
            except OSError:
                logging.warning('Cannot write relay transfer telemetry', exc_info=True)
            if stop.wait(.5):
                return
