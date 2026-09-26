"""CPU/loopback regressions; run only on the authorized remote test machine."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect

from expo_ft.env.env_client import EnvClient, EnvClientWrapper
from expo_ft.utils.robot_round import collect_round


class CancellationTest(unittest.TestCase):
    def test_abort_before_round_does_not_reset(self):
        env, sample = Mock(), Mock()
        with self.assertRaisesRegex(RuntimeError, 'peer aborted'):
            collect_round([env], sample, 1, 10,
                          check_session=Mock(side_effect=RuntimeError('peer aborted')))
        env.reset.assert_not_called()
        env.close.assert_called_once()
        sample.assert_not_called()

    def test_abort_during_inference_does_not_dispatch_action(self):
        aborted = threading.Event()
        env = Mock()
        env.reset.return_value = {}
        def check():
            if aborted.is_set():
                raise RuntimeError('peer aborted')
        def sample(_):
            aborted.set()
            return np.zeros((1, 7))
        with self.assertRaisesRegex(RuntimeError, 'peer aborted'):
            collect_round([env], sample, 1, 10, check_session=check)
        env.step.assert_not_called()
        env.close.assert_called_once()

    def test_abort_unblocks_ws_accept_and_pending_robot_rpcs(self):
        # Real local WebSockets only. Never construct a workstation RobotEnv.
        for blocked_operation in ('accept', 'create_env', 'reset', 'step'):
            with self.subTest(blocked_operation=blocked_operation):
                env = EnvClientWrapper({}, host='127.0.0.1', port=0, recover=False, lazy=True)
                aborted = threading.Event()
                def check():
                    if aborted.is_set():
                        raise RuntimeError('peer aborted')
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(collect_round, [env], lambda _: np.zeros((1, 7)),
                                             1, 10, check_session=check)
                    ws = None
                    try:
                        deadline = time.monotonic() + 5
                        while env.client._server is None:
                            if future.done():
                                future.result()
                            if time.monotonic() > deadline:
                                self.fail('test listener did not start')
                            time.sleep(.01)
                        port = env.client._server.socket.getsockname()[1]
                        if blocked_operation != 'accept':
                            ws = connect(f'ws://127.0.0.1:{port}', close_timeout=1)
                            for operation in ('create_env', 'reset', 'step'):
                                message = msgpack_numpy.unpackb(ws.recv(timeout=5))
                                self.assertEqual(message['operation'], operation)
                                if operation == blocked_operation:
                                    break  # Deliberately leave this RPC waiting for its reply.
                                reply = (dict(env_id='test', task_description='test')
                                         if operation == 'create_env' else dict(observation={}, done=False))
                                ws.send(msgpack_numpy.Packer().pack(reply))
                        aborted.set()
                        with self.assertRaisesRegex(RuntimeError, 'peer aborted'):
                            future.result(timeout=8)
                        self.assertTrue(env.client._closed)
                        self.assertIsNone(env.client._server)
                        with self.assertRaises(OSError):
                            connect(f'ws://127.0.0.1:{port}', open_timeout=1)
                    finally:
                        aborted.set()
                        if ws is not None:
                            ws.close()
                        env.close()

    def test_closed_client_cannot_start_listener_or_send_after_accept(self):
        client = EnvClient(reconnect=False)
        client.close()
        with patch('websockets.sync.server.serve') as serve:
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                client._ensure_server()
            serve.assert_not_called()
        client = EnvClient(reconnect=False)
        conn = Mock()
        def accept_then_cancel():
            client.close()
            return conn
        with patch.object(client, '_get_connection', side_effect=accept_then_cancel):
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                client.create_env({})
        conn.send.assert_not_called()

