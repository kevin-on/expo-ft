"""Manual joint reset tests with fake NUC and HID; no hardware access."""
from contextlib import redirect_stdout, redirect_stderr
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from client import teleop_spacemouse as teleop
from client.tests.test_teleop_spacemouse import mouse_state

ROOT = Path(teleop.__file__).resolve().parents[1]


class ResetTests(unittest.TestCase):
    def setUp(self):
        self.args = teleop.parse_args(['--robot-config', str(ROOT/'configs/robots/robot-0.json')])
        self.robot = Mock()
        self.robot.get_robot_state.return_value = ({'joint_positions': [0.] * 7, 'gripper_position': .3}, {})

    def drive(self, states, requests):
        device = Mock()
        device.read.side_effect = [*states, KeyboardInterrupt()]
        with patch.object(teleop, 'read_reset_request', side_effect=requests), \
             patch.object(teleop.time, 'monotonic', side_effect=itertools.count(0, .2)), \
             patch.object(teleop.time, 'sleep'), redirect_stdout(io.StringIO()):
            teleop.teleoperate(self.robot, device, self.args)

    def test_both_json_values_match_measured_robot_reset_radians(self):
        expected = [
            [-.17817, .15322, -.08950, -2.40599, .02785, 2.55613, -.37072],
            [.18538, .16620, .08072, -2.41205, -.01882, 2.57992, .36083],
        ]
        for i in (0, 1):
            args = teleop.parse_args(['--robot-config', str(ROOT/f'configs/robots/robot-{i}.json')])
            np.testing.assert_array_equal(args.reset_joints, expected[i])

    def test_r_sends_only_absolute_arm_reset_and_no_automatic_reset(self):
        with self.assertRaises(KeyboardInterrupt):
            self.drive([mouse_state()], [False])
        self.robot.update_joints.assert_not_called()
        with self.assertRaises(KeyboardInterrupt):
            self.drive([mouse_state()], [True])
        self.robot.update_joints.assert_called_once()
        call = self.robot.update_joints.call_args
        np.testing.assert_array_equal(call.args[0], self.args.reset_joints)
        self.assertEqual(call.kwargs, {'velocity': False, 'blocking': True})
        self.robot.update_command.assert_not_called()
        self.robot.update_gripper.assert_not_called()

    def test_configured_joint_values_are_used_without_mirroring(self):
        self.args.reset_joints = np.array([.1, -.2, .3, -1., .4, 2., -.5])
        with self.assertRaises(KeyboardInterrupt):
            self.drive([mouse_state()], [True])
        np.testing.assert_array_equal(self.robot.update_joints.call_args.args[0], self.args.reset_joints)

    def test_no_reset_without_configured_target(self):
        self.args.reset_joints = None
        with self.assertRaises(KeyboardInterrupt):
            self.drive([mouse_state()], [True])
        self.robot.update_joints.assert_not_called()
        self.robot.update_command.assert_not_called()

    def test_reset_request_while_moving_is_rejected(self):
        with self.assertRaises(KeyboardInterrupt):
            self.drive([mouse_state(x=1)], [True])
        self.robot.update_joints.assert_not_called()

    def test_post_reset_input_must_return_to_neutral_before_motion(self):
        with self.assertRaises(KeyboardInterrupt):
            self.drive([mouse_state(), mouse_state(x=1, t=2), mouse_state(t=3), mouse_state(x=1, t=4)],
                       [True, False, False, False])
        self.robot.update_joints.assert_called_once()
        # Only the last mouse sample sends a velocity command; exit sends hold.
        self.assertEqual(len(self.robot.update_command.call_args_list), 2)
        self.assertEqual(self.robot.update_command.call_args_list[0].kwargs['action_space'], 'cartesian_velocity')
        self.assertEqual(self.robot.update_command.call_args_list[1].kwargs['action_space'], 'joint_position')

    def test_reset_failure_or_interrupt_attempts_hold(self):
        for error in (RuntimeError('reset failed'), KeyboardInterrupt()):
            self.robot.reset_mock()
            self.robot.update_joints.side_effect = error
            with self.assertRaises(type(error)):
                self.drive([mouse_state()], [True])
            self.assertEqual(self.robot.update_command.call_args.kwargs['action_space'], 'joint_position')
            np.testing.assert_allclose(self.robot.update_command.call_args.args[0][-1], .7)

    def test_invalid_joint_targets_rejected_before_devices(self):
        base = {'robot_server_ip': '127.0.0.1', 'robot_server_port': 4242, 'spacemouse_device_number': 0}
        for value in ([0] * 6, [[0] * 7], [float('nan')] * 7, ['bad'] * 7):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'robot.json'
                path.write_text(json.dumps(dict(base, reset_joints=value)))
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    teleop.parse_args(['--robot-config', str(path)])

    def test_keyboard_poll_is_nonblocking_and_coalesces_queued_resets(self):
        with patch.object(teleop.sys.stdin, 'isatty', return_value=True), \
             patch.object(teleop.sys.stdin, 'fileno', return_value=0), \
             patch.object(teleop.select, 'select', return_value=([0], [], [])), \
             patch.object(teleop.os, 'read', return_value=b'r\nr\n'):
            self.assertTrue(teleop.read_reset_request())
        with patch.object(teleop.sys.stdin, 'isatty', return_value=False), \
             patch.object(teleop.os, 'read') as read:
            self.assertFalse(teleop.read_reset_request())
            read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
