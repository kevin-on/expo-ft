"""Reset pause checks using a private PTY; no robot, camera, or network access."""

import asyncio
import os
from pathlib import Path
import pty
import sys
import termios
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from client import run_client as client


class ResetPauseTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.master, self.slave = pty.openpty()
        self.attrs = termios.tcgetattr(self.slave)
        self.tty_patch = mock.patch.object(client.os, "open", side_effect=lambda *a: os.dup(self.slave))
        self.tty_patch.start()
        self.pause = client._ResetPause()
        self.factory_patch = mock.patch.object(client, "_ResetPause", return_value=self.pause)
        self.factory_patch.start()
        self.socket = SimpleNamespace(state=client.State.OPEN)

    def tearDown(self):
        self.factory_patch.stop()
        self.tty_patch.stop()
        self.assertEqual(termios.tcgetattr(self.slave), self.attrs)
        if self.pause._thread is not None:
            self.assertFalse(self.pause._thread.is_alive())
        os.close(self.master)
        os.close(self.slave)
        client._env_storage.clear()
        client._eval_env_ids.clear()

    def press_space(self, expected_paused):
        os.write(self.master, b" ")
        deadline = time.monotonic() + 2
        while self.pause._paused != expected_paused:
            if time.monotonic() > deadline:
                self.fail("Space was not handled during synchronous reset")
            time.sleep(0.005)

    async def wait_until_paused_after_reset(self):
        async def wait():
            while self.pause._resetting or not self.pause._paused:
                await asyncio.sleep(0.005)
        await asyncio.wait_for(wait(), 2)

    async def test_reset_rpc_waits_for_resume_and_keeps_event_loop_alive(self):
        owner_thread = threading.get_ident()
        reset_finished = False
        replies = []

        def reset():
            nonlocal reset_finished
            self.assertEqual(threading.get_ident(), owner_thread)
            self.press_space(True)
            reset_finished = True  # Pause did not block reset completion.
            return {"frame": 7}

        env = SimpleNamespace(reset=reset, close=mock.Mock())
        client._env_storage["test_eval"] = env
        client._eval_env_ids.add("test_eval")
        request = client.msgpack_numpy.Packer().pack({"operation": "reset", "env_id": "test_eval"})
        socket = SimpleNamespace(
            state=client.State.OPEN,
            remote_address=("test", 0),
            transport=SimpleNamespace(get_extra_info=lambda _: None),
            recv=mock.AsyncMock(side_effect=[request, client.websockets.exceptions.ConnectionClosedOK(None, None)]),
            send=mock.AsyncMock(side_effect=lambda data: replies.append(client.msgpack_numpy.unpackb(data))),
        )
        with self.assertLogs(client.__name__, level="INFO") as logs:
            task = asyncio.create_task(client._handle_environment_request(socket))
            try:
                await self.wait_until_paused_after_reset()
                for _ in range(3):
                    await asyncio.sleep(0.02)  # Keepalive work can run while waiting.
                    self.assertTrue(reset_finished)
                    self.assertEqual(replies, [])
                    self.assertFalse(task.done())
                self.press_space(False)
                await asyncio.wait_for(task, 2)
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(replies, [{"status": "success", "observation": {"frame": 7}, "done": False}])
        text = "\n".join(logs.output)
        for entry in ("[RESET_START]", "[PAUSE] resetting", "[RESET_DONE] paused", "[RESUME] ready"):
            self.assertIn(entry, text)
        env.close.assert_called_once()

    async def test_resume_during_reset_does_not_skip_remaining_reset(self):
        steps = []

        def reset():
            self.press_space(True)
            self.press_space(False)
            steps.append("reset_finished")
            return "obs"

        with self.assertLogs(client.__name__, level="INFO") as logs:
            result = await client._reset_with_pause(SimpleNamespace(reset=reset), self.socket)
        self.assertEqual(result, "obs")
        self.assertEqual(steps, ["reset_finished"])
        self.assertIn("[RESUME] resetting", "\n".join(logs.output))
        self.assertIn("[RESET_DONE] ready", "\n".join(logs.output))

    async def test_queued_rollout_space_is_ignored(self):
        os.write(self.master, b" ")

        def reset():
            time.sleep(0.1)
            self.assertFalse(self.pause._paused)
            return "obs"

        self.assertEqual(await client._reset_with_pause(SimpleNamespace(reset=reset), self.socket), "obs")

    async def test_reset_failure_restores_terminal(self):
        env = SimpleNamespace(reset=mock.Mock(side_effect=ValueError("reset failed")))
        with self.assertRaisesRegex(ValueError, "reset failed"):
            await client._reset_with_pause(env, self.socket)

    async def test_cancellation_while_paused_restores_terminal(self):
        env = SimpleNamespace(reset=lambda: self.press_space(True))
        task = asyncio.create_task(client._reset_with_pause(env, self.socket))
        try:
            await self.wait_until_paused_after_reset()
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_disconnect_while_paused_exits(self):
        env = SimpleNamespace(reset=lambda: self.press_space(True))
        task = asyncio.create_task(client._reset_with_pause(env, self.socket))
        try:
            await self.wait_until_paused_after_reset()
            self.socket.state = client.State.CLOSED
            with self.assertRaisesRegex(ConnectionError, "Connection closed"):
                await asyncio.wait_for(task, 2)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_no_terminal_still_resets(self):
        with mock.patch.object(client.os, "open", side_effect=OSError("no TTY")):
            with self.assertLogs(client.__name__, level="INFO") as logs:
                env = SimpleNamespace(reset=lambda: "obs")
                self.assertEqual(await client._reset_with_pause(env, self.socket), "obs")
        self.assertIn("Space unavailable", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
