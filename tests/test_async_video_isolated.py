"""Stdlib-only tests: no cameras, robot imports, actual encoding or GPU work."""

import ast
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("async_video", ROOT / "client/real_utils/async_video.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
EpisodeVideoWriter = module.EpisodeVideoWriter


def terminal_env(writer):
    """Run the real terminal handler without importing DroidEnv/hardware."""
    tree = ast.parse((ROOT / "client/envs/droid_env.py").read_text())
    klass = next(n for n in tree.body if getattr(n, "name", None) == "DroidEnv")
    klass.bases = []
    klass.body = [n for n in klass.body if getattr(n, "name", None) == "get_info_for_step"]
    saves = []
    ns = {"success_detector_manual": lambda: None,
          "save_episode_video_to_disk": lambda *a, **kw: saves.append((a, kw))}
    exec(compile(ast.Module(body=[klass], type_ignores=[]), "<terminal-handler>", "exec"), ns)
    env = ns["DroidEnv"]()
    env.prev_obs = {}
    env.auto_reset_due = lambda: True
    env.reached_boundary = lambda obs: False
    env.detect = lambda obs: (False, False)
    env.video_dir = "unused"
    env._raw_frame_buffer = [object()]
    env._record_frame_buffer = [object()]
    env._ep_count = 3
    env._video_writer = writer
    return env, saves


class AsyncVideoTests(unittest.TestCase):
    def test_terminal_response_and_reset_buffers_do_not_wait_for_encoding(self):
        encoding = threading.Event()
        release = threading.Event()
        returned = threading.Event()
        saved, result, failures = [], [], []

        def save(frames, directory, episode, **kwargs):
            encoding.set()
            if not release.wait(3):
                raise RuntimeError("test encoder not released")
            saved.append((frames, directory, episode, kwargs["prefix"]))

        writer = EpisodeVideoWriter(save)
        env, _ = terminal_env(writer)
        raw, record = env._raw_frame_buffer, env._record_frame_buffer

        def terminal():
            try:
                with redirect_stdout(io.StringIO()):
                    result.append(env.get_info_for_step())
            except BaseException as exc:
                failures.append(exc)
            finally:
                returned.set()

        thread = threading.Thread(target=terminal)
        try:
            thread.start()
            self.assertTrue(encoding.wait(1))
            self.assertTrue(returned.wait(1), "terminal response waited on video encoding")
            self.assertEqual(failures, [])
            self.assertEqual(result, [(True, False, 0.0, 0.0)])
            self.assertEqual(env._ep_count, 4)
            self.assertEqual(env._raw_frame_buffer, [])
            self.assertEqual(env._record_frame_buffer, [])
            env._raw_frame_buffer.append("next episode frame")
            self.assertEqual(saved, [])
        finally:
            release.set()
            thread.join(2)
            writer.close()
        self.assertIs(saved[0][0], raw)
        self.assertIs(saved[1][0], record)
        self.assertEqual([(x[2], x[3]) for x in saved], [(3, "raw"), (3, "record")])
        self.assertNotIn("next episode frame", raw)

    def test_default_terminal_handler_remains_synchronous(self):
        env, saves = terminal_env(None)
        with redirect_stdout(io.StringIO()):
            env.get_info_for_step()
        self.assertEqual(len(saves), 2)
        self.assertEqual([kwargs["prefix"] for _, kwargs in saves], ["raw", "record"])

    def test_backlog_is_bounded_and_close_drains(self):
        started, release, submitted, closing, closed = (threading.Event() for _ in range(5))
        saved = []

        def save(frames, directory, episode, **kwargs):
            if episode == 0:
                started.set()
                release.wait(3)
            saved.append(episode)

        writer = EpisodeVideoWriter(save)
        writer.submit([("raw", [0])], "unused", 0)
        self.assertTrue(started.wait(1))

        def submit_second():
            writer.submit([("raw", [1])], "unused", 1)
            submitted.set()

        def close():
            closing.set()
            writer.close()
            closed.set()

        second = threading.Thread(target=submit_second)
        closer = threading.Thread(target=close)
        try:
            second.start()
            self.assertFalse(submitted.wait(.05), "unbounded queued episode")
            release.set()
            self.assertTrue(submitted.wait(1))
            closer.start()
            self.assertTrue(closing.wait(1))
            self.assertTrue(closed.wait(1))
        finally:
            release.set()
            second.join(2)
            if closer.ident is not None:
                closer.join(2)
            writer.close()
        self.assertEqual(saved, [0, 1])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            writer.submit([("raw", [])], "unused", 2)

    def test_environment_close_drains_writer_even_on_camera_close_error(self):
        tree = ast.parse((ROOT / "client/envs/droid_env.py").read_text())
        klass = next(n for n in tree.body if getattr(n, "name", None) == "DroidEnv")
        methods = [n for n in klass.body if getattr(n, "name", None) == "close"]
        self.assertEqual(len(methods), 1, "A later close must not override writer cleanup")
        klass.body = methods
        klass.bases = [ast.Name(id="Base", ctx=ast.Load())]
        ast.fix_missing_locations(klass)
        class Base:
            def close(self): pass
        calls = []
        class Camera:
            def disable_cameras(self): raise RuntimeError("disconnected")
        class Writer:
            def close(self): calls.append("drained")
        ns = {"Base": Base}
        exec(compile(ast.Module(body=[klass], type_ignores=[]), "<close>", "exec"), ns)
        env = ns["DroidEnv"]();env.camera_reader=Camera();env._video_writer=Writer()
        env.close();env.close()
        self.assertEqual(calls, ["drained"])

    def test_close_waits_for_active_save(self):
        started, release, closed = (threading.Event() for _ in range(3))
        def save(*args, **kwargs):
            started.set()
            release.wait(3)
        writer = EpisodeVideoWriter(save)
        writer.submit([("raw", [0])], "unused", 0)
        self.assertTrue(started.wait(1))
        thread = threading.Thread(target=lambda: (writer.close(), closed.set()))
        try:
            thread.start()
            self.assertFalse(closed.wait(.05))
        finally:
            release.set()
            thread.join(2)
            writer.close()
        self.assertTrue(closed.is_set())

    def test_failure_is_logged_and_other_views_still_save(self):
        saved = []
        def save(frames, directory, episode, **kwargs):
            if kwargs["prefix"] == "raw":
                raise OSError("test disk failure")
            saved.append(kwargs["prefix"])
        writer = EpisodeVideoWriter(save)
        with self.assertLogs("async_video", level="ERROR") as logs:
            writer.submit([("raw", [0]), ("record", [1])], "unused", 5)
            writer.close()
        self.assertIn("test disk failure", "\n".join(logs.output))
        self.assertEqual(saved, ["record"])


if __name__ == "__main__":
    unittest.main()
