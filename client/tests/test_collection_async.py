"""Tiny offline checks; no camera, robot, GPU or actual video encoder."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from threading import Event, get_ident
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np

from client import collect_data as collect


class AsyncCollectionTests(unittest.TestCase):
    def test_pairs_are_frozen_ordered_and_drained_with_bounded_pending_work(self):
        started, release, third_submitted = Event(), Event(), Event()
        worker_threads = []
        video = Mock()

        def transform(env, obs, stereo):
            worker_threads.append(get_ident())
            started.set()
            if not release.wait(3):
                raise TimeoutError("test worker not released")
            return {'wrist_image_left': obs['image']['wrist'],
                    'cartesian_position': obs['robot_state']['cartesian_position']}

        with tempfile.TemporaryDirectory(dir='/scr/kevinon/tmp') as tmp, \
             patch.object(collect, 'collection_observation', side_effect=transform), \
             patch.object(collect.imageio, 'get_writer', return_value=video):
            path = str(Path(tmp) / 'traj.hdf5')
            recorder = collect.CollectionRecorder(None, path, str(Path(tmp) / 'images'),
                                                   True, (4, 2), max_pending=2)
            obs = {'image': {'wrist': np.zeros((2, 4, 3), dtype=np.uint8)},
                   'robot_state': {'cartesian_position': np.zeros(6)}}
            action = {'executed_action': np.zeros(7)}
            try:
                recorder.submit(obs, action)
                self.assertTrue(started.wait(1))
                obs['image']['wrist'][:] = 1
                obs['robot_state']['cartesian_position'][:] = 1
                action['executed_action'][:] = 1
                recorder.submit(obs, action)
                obs['image']['wrist'][:] = 2
                obs['robot_state']['cartesian_position'][:] = 2
                action['executed_action'][:] = 2

                def submit_third():
                    third_submitted.set()
                    recorder.submit(obs, action)

                with ThreadPoolExecutor(max_workers=1) as producer:
                    future = producer.submit(submit_third)
                    try:
                        self.assertTrue(third_submitted.wait(1))
                        self.assertFalse(future.done())
                        self.assertEqual(len(recorder.pending), 2)
                    finally:
                        release.set()
                    future.result(timeout=2)
                # Mutations after submit must never alter recorded rows.
                obs['image']['wrist'][:] = 99
                action['executed_action'][:] = 99
            finally:
                release.set()
                recorder.close({'recorded_steps': 3})
            with h5py.File(path, 'r') as f:
                np.testing.assert_array_equal(f['saved_observation/wrist_image_left'][:, 0, 0, 0], [0, 1, 2])
                np.testing.assert_array_equal(f['saved_observation/cartesian_position'][:, 0], [0, 1, 2])
                np.testing.assert_array_equal(f['action/executed_action'][:, 0], [0, 1, 2])
                self.assertEqual(f.attrs['recorded_steps'], 3)
            self.assertEqual(len(set(worker_threads)), 1)
            self.assertNotEqual(worker_threads[0], get_ident())
            self.assertEqual(video.append_data.call_count, 3)
            video.close.assert_called_once()

    def test_write_failure_propagates_and_all_files_close(self):
        hdf5, video = Mock(), Mock()
        hdf5.attrs = {}
        with patch.object(collect, 'collection_observation', return_value={
                'wrist_image_left': np.zeros((2, 4, 3), np.uint8)}), \
             patch.object(collect.h5py, 'File', return_value=hdf5), \
             patch.object(collect, 'write_dict_to_hdf5', side_effect=OSError('disk full')), \
             patch.object(collect.imageio, 'get_writer', return_value=video), \
             patch.object(collect.os, 'makedirs'):
            recorder = collect.CollectionRecorder(None, '/unused/traj.hdf5', '/unused/images', True, (4, 2))
            recorder.submit({}, {})
            try:
                with self.assertRaisesRegex(OSError, 'disk full'):
                    recorder.pending[0].result(timeout=1)
                with self.assertRaisesRegex(OSError, 'disk full'):
                    recorder.check()
            finally:
                with self.assertRaisesRegex(OSError, 'disk full'):
                    recorder.close({})
            hdf5.close.assert_called_once()
            video.close.assert_called_once()

    def test_next_action_does_not_wait_for_image_transform(self):
        second_action = Event()
        env, controller = Mock(), Mock()
        env.control_hz = 10
        env.get_raw_observation.side_effect = [
            {'timestamp': {}, 'value': i} for i in range(3)
        ]
        env.get_info_for_step.side_effect = [(False, False, 0, 1)] * 2 + [(True, True, 1, 0)]
        controller.get_info.return_value = {'movement_enabled': True}
        controller.forward.return_value = (np.zeros(7), {})
        actions = []

        def step(action):
            actions.append(action)
            if len(actions) == 2:
                second_action.set()
            return {'executed_action': action.copy()}

        def transform(env, obs, stereo):
            if not second_action.wait(1):
                raise TimeoutError('transform blocked the next action')
            return {'value': obs['value']}

        env.step.side_effect = step
        flags = SimpleNamespace(save_right_images=True, video_save_width=320, video_save_height=180, video_encoder_threads=2)
        with tempfile.TemporaryDirectory(dir='/scr/kevinon/tmp') as tmp, \
             patch.object(collect, 'FLAGS', flags), \
             patch.object(collect, 'collection_observation', side_effect=transform), \
             redirect_stdout(io.StringIO()):
            result = collect.collect_trajectory(env, controller,
                save_filepath=str(Path(tmp) / 'traj.hdf5'), recording_folderpath=str(Path(tmp) / 'images'))
            self.assertTrue(result['success'])
            self.assertEqual(env.step.call_count, 2)
            with h5py.File(Path(tmp) / 'traj.hdf5', 'r') as f:
                np.testing.assert_array_equal(f['saved_observation/value'][:], [0, 1])
                self.assertEqual(f.attrs['recorded_steps'], 2)


if __name__ == '__main__':
    unittest.main()
