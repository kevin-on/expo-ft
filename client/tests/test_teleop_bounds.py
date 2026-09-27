"""Offline live XYZ checks; all robot/HID calls are mocked."""
from contextlib import redirect_stdout
import io
import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from client import teleop_spacemouse as teleop


class PositionTests(unittest.TestCase):
    def setUp(self):
        self.robot = Mock()
        self.robot.get_robot_state.return_value = (
            {'cartesian_position': [.28, -.193, .1, 2., 3., 4.],
             'joint_positions': [9.] * 7, 'gripper_position': .25}, {})

    def test_base_xyz_meters_without_axis_flip_or_unit_conversion(self):
        output = io.StringIO()
        with redirect_stdout(output):
            teleop.print_position(self.robot)
        self.assertEqual(output.getvalue(), '\rX= 0.280000  Y=-0.193000  Z= 0.100000  [m, robot base]')
        self.robot.update_command.assert_not_called()

    def test_each_display_reads_current_state(self):
        output = io.StringIO()
        with redirect_stdout(output):
            teleop.print_position(self.robot)
            self.robot.get_robot_state.return_value[0]['cartesian_position'][0] = .6
            teleop.print_position(self.robot)
        self.assertEqual(self.robot.get_robot_state.call_count, 2)
        self.assertIn('X= 0.600000', output.getvalue())

    def test_malformed_state_is_rejected(self):
        for xyz in ([0, float('nan'), 0], [0, 0, float('inf')], [1, 2]):
            self.robot.get_robot_state.return_value[0]['cartesian_position'] = xyz
            with self.assertRaisesRegex(ValueError, 'finite robot'):
                teleop.print_position(self.robot)

    def test_motion_also_displays_xyz_and_failure_still_holds(self):
        args = teleop.parse_args(['--measure-bounds'])
        device = Mock()
        device.read.return_value = SimpleNamespace(t=1., x=1., y=0., z=0., roll=0., pitch=0., yaw=0., buttons=[0, 0])
        with patch.object(teleop.time, 'monotonic', side_effect=itertools.count(0, .2)), \
             patch.object(teleop, 'print_position', side_effect=RuntimeError('display read failed')) as display, \
             self.assertRaisesRegex(RuntimeError, 'display read failed'):
            teleop.teleoperate(self.robot, device, args)
        display.assert_called_once_with(self.robot)
        self.assertEqual(self.robot.update_command.call_args_list[0].kwargs['action_space'], 'cartesian_velocity')
        self.assertEqual(self.robot.update_command.call_args_list[-1].kwargs['action_space'], 'joint_position')


if __name__ == '__main__':
    unittest.main()
