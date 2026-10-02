"""In-memory real RPC handler/codec; no cameras, robot, sockets or GPU."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from openpi_client import msgpack_numpy
from websockets.exceptions import ConnectionClosed

from client import run_client
from expo_ft.env.env_client import EnvClient


class TimingTests(unittest.TestCase):
    def test_handler_round_trip_and_legacy_reply(self):
        observation = {'image': np.arange(36, dtype=np.uint8).reshape(3, 4, 3)}
        env = SimpleNamespace(get_observation=lambda: observation,
            get_info_for_step=lambda: (False, False, 0., 1.), close=lambda: None,
            observation_timing={'cameras_ms': 0.0})

        class Workstation:
            transport = SimpleNamespace(get_extra_info=lambda _: None)
            remote_address = 'mock'
            def __init__(self, request): self.request, self.reply = request, None
            async def recv(self):
                if self.request is None: raise ConnectionClosed(None, None)
                request, self.request = self.request, None
                return request
            async def send(self, reply): self.reply = reply

        class Connection:
            def send(self, request):
                ws = Workstation(request)
                with patch.dict(run_client._env_storage, {'mock': env}, clear=True):
                    asyncio.run(run_client._handle_environment_request(ws))
                self.reply = ws.reply
            def recv(self): return self.reply
            def close(self): pass

        client = EnvClient(reconnect=False)
        client._conn = Connection()
        obs = client.get_observation('mock')
        np.testing.assert_array_equal(obs['image'], observation['image'])
        self.assertEqual(client.get_info_for_step('mock'), (False, False, 0., 1.))
        timing = client.observation_timing
        self.assertEqual(timing['ws']['environment'], env.observation_timing)
        self.assertGreaterEqual(timing['ws']['processing_ms'], timing['ws']['get_observation_ms'])
        self.assertGreaterEqual(timing['ws']['response_pack_ms'], 0.)
        self.assertEqual(timing['response_bytes'], len(client._conn.reply))
        self.assertAlmostEqual(timing['rpc_ms'], sum(timing[k] for k in
            ('request_pack_ms', 'request_send_ms', 'response_wait_ms', 'response_unpack_ms')))
        self.assertAlmostEqual(timing['transport_and_queue_ms'], timing['rpc_ms']
            - timing['request_pack_ms'] - timing['response_unpack_ms']
            - timing['ws']['processing_ms'] - timing['ws']['response_pack_ms'])
        legacy = msgpack_numpy.packb({'status': 'success', 'observation': observation})
        client._conn = SimpleNamespace(send=lambda _: None, recv=lambda: legacy, close=lambda: None)
        client.get_observation('mock')
        self.assertNotIn('ws', client.observation_timing)
        self.assertNotIn('transport_and_queue_ms', client.observation_timing)
        client.close()

    def test_packing_does_not_serialize_images_twice_or_retain_previous_map(self):
        calls = []
        def pack_array(value):
            calls.append(value)
            return msgpack_numpy.pack_array(value)
        packer = msgpack_numpy.Packer(default=pack_array, autoreset=False)
        image = np.arange(12, dtype=np.uint8)
        for i in range(2):
            reply = run_client._pack_observation_response(
                {'observation': {'image': image}, 'status': 'success'}, {'sequence': i}, packer)
            decoded = msgpack_numpy.unpackb(reply)
            np.testing.assert_array_equal(decoded['observation']['image'], image)
            self.assertEqual(decoded['observation_timing']['sequence'], i)
        self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
