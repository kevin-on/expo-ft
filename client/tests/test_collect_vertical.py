"""Collection checks with fake env/writers only; no hardware, files or encoders."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from absl.testing import flagsaver
import numpy as np
from scipy.spatial.transform import Rotation

from client import collect_data as collect
from client.real_utils.vertical_control import (
    DROID_MAX_ROT_DELTA, VERTICAL_ROT_STEP, rotation_step,
    vertical_orientation, vertical_velocity_action,
)


class CollectVerticalTests(unittest.TestCase):
    def test_velocity_feedback_matches_teleop_rotation_step(self):
        for current in ([2.8, .3, .7], [-3.13, -.2, -3.1], [np.pi, 0, .4]):
            target = vertical_orientation(current)
            action = np.array([.2, -.3, .1, .7, -.6, .2, -1.])
            before = action.copy()
            result = vertical_velocity_action(action, current, target)
            np.testing.assert_array_equal(result[[0, 1, 2, 6]], before[[0, 1, 2, 6]])
            np.testing.assert_array_equal(action, before)
            # DROID applies Euler delta by left-multiplying the current rotation.
            delta = Rotation.from_euler('xyz', result[3:6] * DROID_MAX_ROT_DELTA)
            actual = delta * Rotation.from_euler('xyz', current)
            expected = Rotation.from_euler('xyz', rotation_step(current, target))
            np.testing.assert_allclose(actual.as_matrix(), expected.as_matrix(), atol=1e-12)
            self.assertLessEqual(delta.magnitude(), VERTICAL_ROT_STEP + 1e-12)

    def drive(self, keep_vertical, orientations, terminal_success=True):
        env, controller, writer = Mock(), Mock(), Mock()
        env.control_hz = 10
        raw = [{'robot_state': {'cartesian_position': [.4, -.1, .25, *angles]},
                'timestamp': {}} for angles in orientations]
        env.get_raw_observation.side_effect = raw
        env.get_info_for_step.side_effect = (
            [(False, False, 0, 1)] * (len(raw) - 1) + [(True, terminal_success, 0, 0)]
        )
        controller.get_info.return_value = {'movement_enabled': True}
        input_action = np.array([.2, .1, -.1, .8, -.6, .4, 1.])
        controller.forward.return_value = (input_action, {})
        env.step.side_effect = lambda action: {'cartesian_velocity': action[:6].copy(),
                                               'gripper_velocity': action[6], 'executed_action': action.copy()}
        fake_flags = SimpleNamespace(save_right_images=True, video_save_width=320, video_save_height=180)
        with patch.object(collect, 'FLAGS', fake_flags), \
             patch.object(collect, 'collection_observation', return_value={}), \
             patch.object(collect, 'CollectionRecorder', return_value=writer), \
             patch.object(collect.os, 'makedirs'), \
             patch.object(collect.time, 'sleep'), \
             redirect_stdout(io.StringIO()):
            result = collect.collect_trajectory(env, controller, save_filepath='/not-written/traj.hdf5',
                                                recording_folderpath='/not-written/images',
                                                keep_vertical=keep_vertical)
        env.reset.assert_called_once()
        return env, writer, result, input_action

    def test_corrected_commands_are_recorded_and_yaw_stays_fixed(self):
        angles = [[2.8, .3, .7], [2.9, .2, .6], [3.0, .1, .5]]
        env, writer, result, original = self.drive(True, angles)
        self.assertTrue(result['success'])
        self.assertEqual(env.step.call_count, 2)
        self.assertEqual(writer.submit.call_count, 2)
        target = vertical_orientation(angles[0])
        for i in range(2):
            expected = vertical_velocity_action(original, angles[i], target)
            np.testing.assert_allclose(env.step.call_args_list[i].args[0], expected)
            stored = writer.submit.call_args_list[i].args[1]
            np.testing.assert_allclose(stored['cartesian_velocity'], expected[:6])
            np.testing.assert_allclose(stored['executed_action'], expected)
        self.assertTrue(writer.close.call_args.args[0]['keep_vertical'])

    def test_new_episode_reanchors_yaw_and_disabled_mode_is_unchanged(self):
        for yaw in (.7, -1.2):
            angles = [[2.8, .3, yaw], [2.8, .3, yaw]]
            env, _, _, original = self.drive(True, angles)
            np.testing.assert_allclose(env.step.call_args.args[0],
                                       vertical_velocity_action(original, angles[0], vertical_orientation(angles[0])))
        env, writer, _, original = self.drive(False, angles)
        np.testing.assert_array_equal(env.step.call_args.args[0], original)
        self.assertFalse(writer.close.call_args.args[0]['keep_vertical'])

    def test_terminal_boundary_sends_no_further_action_even_with_vertical_enabled(self):
        env, writer, result, _ = self.drive(True, [[2.8, .3, .7]], terminal_success=False)
        self.assertFalse(result['success'])
        env.step.assert_not_called()
        writer.submit.assert_not_called()

    def test_hyphenated_flag_alias_sets_collection_flag(self):
        with flagsaver.flagsaver():
            collect.FLAGS['keep-vertical'].parse('true')
            self.assertTrue(collect.FLAGS['keep_vertical'].value)

    def test_stereo_names_are_physical_lenses_for_both_default_configs(self):
        for robot in (0, 1):
            cfg = json.loads(Path(f'configs/robots/robot-{robot}.json').read_text())
            env = SimpleNamespace(side_camera_id=cfg['side_camera_id'], wrist_camera_id=cfg['wrist_camera_id'], image_size=None)
            side = env.side_camera_id.rsplit('_', 1)[0]
            wrist = env.wrist_camera_id.rsplit('_', 1)[0]
            images = {key: np.full((2, 4, 3), value, dtype=np.uint8) for key, value in
                      ((side+'_left', 10), (side+'_right', 20), (wrist+'_left', 30), (wrist+'_right', 40))}
            env.transform_observation = lambda raw: {
                'exterior_image_1_left': raw['image'][env.side_camera_id],
                'exterior_image_2_left': raw['image'][env.side_camera_id],
                'wrist_image_left': raw['image'][env.wrist_camera_id],
            }
            saved = collect.collection_observation(env, {'image': images}, True)
            for key, value in (('exterior_image_1_left', 10), ('exterior_image_2_left', 10),
                               ('exterior_image_1_right', 20), ('wrist_image_left', 30), ('wrist_image_right', 40)):
                np.testing.assert_array_equal(saved[key], np.full((2,4,3), value, np.uint8))


if __name__ == '__main__':
    unittest.main()
