import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from expo_ft.distributed.transfer_status import RouteState, TransferStatus


class StatusTest(unittest.TestCase):
    def test_routes_expire_and_update_without_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'routes.json'
            routes = RouteState(str(path))
            self.assertEqual(routes.indices(), [])
            path.write_text(json.dumps(dict(updated=time.time(), active=[0, 2])))
            routes.checked = 0
            self.assertEqual(routes.indices(), [0, 2])
            path.write_text(json.dumps(dict(updated=0, active=[0])))
            routes.checked = 0
            self.assertEqual(routes.indices(), [])

    def test_chunk_progress_completion_counts_and_failed_history(self):
        routes = RouteState('mock')
        stats = TransferStatus(None, routes, 64)
        with patch.object(routes, 'indices', return_value=list(range(32))):
            stats.begin(dict(size=8_000_000))
            self.assertEqual(stats.snapshot()['current']['tunnels'], [32])
        stats.acknowledged(4_000_000)
        with patch.object(routes, 'indices', return_value=list(range(31))):
            self.assertEqual(stats.snapshot()['current']['done'], 4_000_000)
            self.assertEqual(stats.snapshot()['current']['tunnels'], [32, 31])
        stats.acknowledged(4_000_000)
        stats.verifying()
        stats.finish(dict(receive_seconds=2))
        self.assertIsNone(stats.current)
        self.assertEqual(stats.history[-1]['MBps'], 4)
        self.assertEqual(stats.history[-1]['phase'], 'Complete')
        stats.begin(dict(size=8_000_000))
        stats.finish(failed=True)
        self.assertEqual(stats.history[-1]['phase'], 'Failed')
        stats.acknowledged(100)  # late ACK during failed-pool cleanup is harmless


if __name__ == '__main__':
    unittest.main()
