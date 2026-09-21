"""Exercise real DROID/EXPO observation code with mocked robot and camera I/O."""
from contextlib import ExitStack
import unittest
from unittest.mock import Mock, patch

import numpy as np

from client.envs.droid_env import DroidEnv
from client.real_utils.vis_utils import raw_frame_from_raw_obs


class BlankCameraTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.rpc = self.stack.enter_context(patch('droid.robot_env.ServerInterface'))
        self.camera = self.stack.enter_context(patch('droid.robot_env.MultiCameraWrapper'))
        self.stack.enter_context(patch('droid.robot_env.load_calibration_info', return_value={}))
        self.reader = self.camera.return_value
        self.reader.camera_dict = {}
        self.reader.read_cameras.return_value = ({}, {})
        self.rpc.return_value.get_robot_state.return_value = (
            {'cartesian_position': np.zeros(6), 'gripper_position': 0.5}, {})
        self.config = dict(camera_serials=['side', 'wrist'], wrist_camera_serial='wrist',
                           side_camera_id='side_left', wrist_camera_id='wrist_left',
                           image_size=(180, 320), language_instruction='test',
                           launch_controller=False)

    def test_real_side_and_blank_wrist_survive_observation_and_video_conversion(self):
        # Keep this fixture independent of the workstation's real wrist mapping.
        config = dict(
            robot_server_ip='172.16.0.1', robot_server_port=4243,
            camera_serials=['29838012', 'TEMP_WRIST_ROBOT_1'],
            blank_camera_serials=['TEMP_WRIST_ROBOT_1'],
            wrist_camera_serial='TEMP_WRIST_ROBOT_1',
            side_camera_id='29838012_left', wrist_camera_id='TEMP_WRIST_ROBOT_1_left',
            image_size=(180, 320), launch_controller=False,
        )
        side = np.full((360, 640, 3), 73, np.uint8)
        self.reader.read_cameras.return_value = ({'image': {config['side_camera_id']: side}}, {})
        env = DroidEnv(**config)
        self.camera.assert_called_once_with({}, ['29838012'], 'TEMP_WRIST_ROBOT_1')
        self.rpc.assert_called_once_with(ip_address='172.16.0.1', port=4243, launch=False)
        raw = env.get_raw_observation()
        self.assertIs(raw['image'][config['side_camera_id']], side)
        wrist = raw['image'][config['wrist_camera_id']]
        self.assertEqual(wrist.shape, (180, 320, 3))
        self.assertEqual(wrist.dtype, np.uint8)
        self.assertFalse(wrist.any())
        obs = env.transform_observation(raw)
        np.testing.assert_array_equal(obs['exterior_image_1_left'], np.full((180, 320, 3), 73, np.uint8))
        self.assertFalse(obs['wrist_image_left'].any())
        frame = raw_frame_from_raw_obs(raw, env.side_camera_id, env.wrist_camera_id)
        self.assertEqual(frame.shape, (360, 1280, 3))
        self.assertFalse(frame[:, 640:].any())
        self.assertEqual(raw['camera_intrinsics'], {})  # No fabricated calibration.

    def test_all_blank_views_work_when_reader_has_no_image_dictionary(self):
        env = DroidEnv(**self.config, blank_camera_serials=['side', 'wrist'])
        self.camera.assert_called_once_with({}, [], 'wrist')
        obs = env.get_observation()
        for key in ('exterior_image_1_left', 'exterior_image_2_left', 'wrist_image_left'):
            self.assertEqual(obs[key].shape, (180, 320, 3))
            self.assertEqual(obs[key].dtype, np.uint8)
            self.assertFalse(obs[key].any())

    def test_unlisted_missing_camera_still_fails(self):
        env = DroidEnv(**self.config, blank_camera_serials=['wrist'])
        with self.assertRaises(KeyError):
            env.get_observation()  # The real side camera is not silently replaced.

    def test_real_camera_startup_failure_is_not_swallowed(self):
        self.camera.side_effect = ValueError('Requested ZED cameras not found: side')
        with self.assertRaisesRegex(ValueError, 'side'):
            DroidEnv(**self.config, blank_camera_serials=['wrist'])

    def test_disabled_mode_preserves_real_views(self):
        side = np.full((180, 320, 3), 30, np.uint8)
        wrist = np.full((180, 320, 3), 90, np.uint8)
        self.reader.read_cameras.return_value = ({'image': {'side_left': side, 'wrist_left': wrist}}, {})
        env = DroidEnv(**self.config)
        self.camera.assert_called_once_with({}, ['side', 'wrist'], 'wrist')
        raw = env.get_raw_observation()
        self.assertIs(raw['image']['side_left'], side)
        self.assertIs(raw['image']['wrist_left'], wrist)

    def test_invalid_blank_config_fails_before_robot_connection(self):
        for overrides in ({'image_size': None}, {'image_size': (0, 320)},
                          {'blank_camera_serials': ['unknown']}, {'record_camera': 'wrist'}):
            config = dict(self.config, blank_camera_serials=['wrist'])
            config.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                DroidEnv(**config)
        self.rpc.assert_not_called()
        self.camera.assert_not_called()


if __name__ == '__main__':
    unittest.main()
