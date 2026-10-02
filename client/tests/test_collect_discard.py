"""Collection discard lifecycle with fake hardware and a private pseudo-terminal."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import pty
import select
import tempfile
import termios
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from client import collect_data as collect


class CollectDiscardTests(unittest.TestCase):
    def test_single_key_escape_sequences_and_terminal_restore(self):
        master, slave = pty.openpty()
        before = termios.tcgetattr(slave)
        handle = os.dup(slave)
        try:
            with patch.object(collect.os, 'open', return_value=handle), redirect_stdout(io.StringIO()):
                with collect.CollectionKeys() as keys:
                    self.assertFalse(termios.tcgetattr(slave)[3] & termios.ICANON)
                    for payload, expected in [(b'\x1b[', None), (b'D', None),
                                              (b'\x1bO', None), (b'D', None),
                                              (b'1D', 'discard'), (b'1', 'success'),
                                              (b'2', 'discard'), (b'\n3', None)]:
                        os.write(master, payload)
                        self.assertTrue(select.select([keys.fd], [], [], .5)[0])
                        self.assertEqual(keys.poll(), expected)
                    os.write(master, b'D')
                    self.assertTrue(select.select([keys.fd], [], [], .5)[0])
                    keys.clear()
                    self.assertIsNone(keys.poll())
            self.assertEqual(termios.tcgetattr(slave), before)
        finally:
            os.close(master)
            os.close(slave)

    def test_discard_closes_then_removes_only_attempt_and_retry_resets(self):
        flags = SimpleNamespace(test_detector=False, keep_vertical=False,
                                save_right_images=True, video_save_width=320,
                                video_save_height=180, num_episodes=1)
        env, controller, keys = Mock(), Mock(), Mock()
        env.control_hz = 10
        env.get_raw_observation.side_effect = lambda: {'timestamp': {}}
        env.get_info_for_step.side_effect = [(False, False, 0, 1), (True, True, 1, 0)]
        controller.get_info.return_value = {'movement_enabled': True}
        controller.forward.return_value = (np.zeros(7), {})
        env.step.return_value = {'executed_action': np.zeros(7)}
        # First attempt: one recorded action then D. Second: normal success.
        keys.poll.side_effect = [None, None, 'discard', None, None]
        events = []

        class Recorder:
            def __init__(self, env, filepath, folder, *args):
                self.path = Path(filepath)
                self.path.write_bytes(b'partial hdf5')
                mp4 = self.path.parent / 'recordings/MP4/wrist.mp4'
                mp4.parent.mkdir(parents=True)
                mp4.write_bytes(b'partial mp4')

            def check(self): pass
            def submit(self, obs, action): events.append('submit')
            def close(self, metadata):
                self.assert_exists = self.path.exists()
                if not self.assert_exists: raise AssertionError('files removed before close')
                events.append('closed')

        with tempfile.TemporaryDirectory(dir='/scr/kevinon/tmp') as tmp:
            keep = Path(tmp) / 'success/0/keep.txt'; keep.parent.mkdir(parents=True); keep.write_text('existing data')
            real_remove = collect.shutil.rmtree

            def remove(path, *args, **kwargs):
                self.assertEqual(events[-1], 'closed')
                events.append('removed')
                return real_remove(path, *args, **kwargs)

            with patch.object(collect, 'FLAGS', flags), patch.object(collect, 'CollectionRecorder', Recorder), \
                 patch.object(collect.shutil, 'rmtree', side_effect=remove), redirect_stdout(io.StringIO()):
                # The production loop must not count the discarded attempt toward num_episodes.
                collect.run_collection(env, controller, tmp, keys)
            self.assertEqual(events, ['submit', 'closed', 'removed'])
            self.assertEqual(env.reset.call_count, 2)
            self.assertEqual(controller.reset_state.call_count, 2)
            self.assertEqual(keys.clear.call_count, 2)
            self.assertEqual(env.step.call_count, 1)
            self.assertEqual(keep.read_text(), 'existing data')
            self.assertEqual(list((Path(tmp) / 'tmp').iterdir()), [])
            self.assertTrue((Path(tmp) / 'success/1').is_dir())  # only the successful retry gets an ID
            self.assertFalse((Path(tmp) / 'success/2').exists())
            for call in env.get_info_for_step.call_args_list:
                self.assertEqual(call.kwargs, {'manual_override': 'keep_going'})

    def test_discard_during_observation_precedes_detector_success_and_action(self):
        env, controller, keys = Mock(), Mock(), Mock()
        keys.poll.side_effect = [None, 'discard']
        env.get_raw_observation.return_value = {'timestamp': {}}
        with redirect_stdout(io.StringIO()):
            result = collect.collect_trajectory(env, controller, keys=keys)
        self.assertTrue(result['discarded'])
        env.step.assert_not_called()
        env.get_info_for_step.assert_not_called()
        controller.forward.assert_not_called()

    def test_collection_override_bypasses_old_terminal_reader_only_when_supplied(self):
        from client.envs import droid_env

        # No constructor: no controllers, cameras or robot connections.
        env = object.__new__(droid_env.PickBlocksEnv)
        env._recorder = None
        env.video_dir = None
        env.auto_reset_due = Mock(return_value=False)
        env.reached_boundary = Mock(return_value=False)
        env.detect = Mock(return_value=(False, False))
        with patch.object(droid_env, 'success_detector_manual', return_value='reset') as reader, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(env.get_info_for_step({}, manual_override='keep_going'), (False, False, 0.0, 1.0))
            reader.assert_not_called()
            self.assertEqual(env.get_info_for_step({}, manual_override='success'), (True, True, 1.0, 0.0))
            reader.assert_not_called()
            # Unmodified online/eval callers still get the existing terminal behavior.
            self.assertEqual(env.get_info_for_step({}), (True, False, 0.0, 0.0))
            reader.assert_called_once()


if __name__ == '__main__':
    unittest.main()
