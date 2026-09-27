"""Offline tests: fake robot RPCs, HID and terminal; never open devices."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gevent
from gevent.event import Event
import numpy as np
from scipy.spatial.transform import Rotation

from client import teleop_two_robots as dual
from client.tests.test_teleop_spacemouse import mouse_state


def make_robot(index):
    config = SimpleNamespace(nuc_ip='fake', server_port=4242 + index,
                             max_lin_vel=.5, max_rot_vel=.1, deadzone=.05,
                             reset_joints=np.arange(7, dtype=float) / 10)
    raw = dict(cartesian_position=[.4, .2, .3, 2.8, -.3, .7],
               joint_positions=[.1] * 7, gripper_position=.3)
    command, reader, device = Mock(), Mock(), Mock()
    command.get_robot_state.return_value = (raw, {})
    reader.get_robot_state.return_value = (raw, {})
    device.read.return_value = mouse_state()
    robot = dual.Robot(index, config, device, command, reader)
    robot.read_state()
    robot.tick(mouse_state(), time.monotonic())
    return robot


class MirrorTests(unittest.TestCase):
    def test_reflection_matches_rotation_matrix_and_is_involution(self):
        reflection = np.diag([1, -1, 1])
        for angles in ([.3, -.4, .7], [3.1, 1.57, -2.9], [-2, -1.7, 2]):
            pose = np.r_[[.4, -.2, .3], angles]
            original = pose.copy()
            mirrored = dual.mirror_pose(pose)
            np.testing.assert_allclose(mirrored[:3], reflection @ pose[:3])
            np.testing.assert_allclose(Rotation.from_euler('xyz', mirrored[3:]).as_matrix(),
                                      reflection @ Rotation.from_euler('xyz', angles).as_matrix() @ reflection,
                                      atol=1e-12)
            np.testing.assert_array_equal(dual.mirror_pose(mirrored), pose)
            np.testing.assert_array_equal(original, pose)
        for invalid in ([0] * 7, [float('nan')] * 6):
            with self.assertRaises(ValueError):
                dual.mirror_pose(invalid)

    def test_both_directions_use_one_fixed_destination_pose_not_joint_mirror(self):
        for source in (0, 1):
            robots = [make_robot(i) for i in (0, 1)]
            expected = dual.mirror_pose(robots[source].state['pose'])
            dual.request_move(robots, str(source), time.monotonic())
            robots[source].state['pose'][:] = 0  # Target was captured at key press.
            destination = robots[1-source]
            destination.tick(mouse_state(), time.monotonic())
            destination.command.update_pose.assert_called_once()
            call = destination.command.update_pose.call_args
            np.testing.assert_array_equal(call.args[0], expected)
            self.assertEqual(call.kwargs, dict(velocity=False, blocking=True))
            destination.command.update_joints.assert_not_called()
            destination.command.update_gripper.assert_not_called()
            self.assertEqual(robots[source].command.mock_calls, [])

    def test_stale_busy_or_moving_robot_rejects_mirror(self):
        for field, value in [('received_at', 0), ('input_at', 0), ('busy', True),
                             ('release_required', True), ('action', np.ones(7))]:
            robots = [make_robot(i) for i in (0, 1)]
            setattr(robots[0], field, value)
            dual.request_move(robots, '0', time.monotonic())
            self.assertIsNone(robots[1].pending)

    def test_reset_only_selected_robot_uses_its_own_target(self):
        for index, key in enumerate(('r', 't')):
            robots = [make_robot(i) for i in (0, 1)]
            robot = robots[index]
            dual.request_move(robots, key, time.monotonic())
            robot.tick(mouse_state(), time.monotonic())
            np.testing.assert_array_equal(robot.command.update_joints.call_args.args[0],
                                          robot.config.reset_joints)
            self.assertEqual(robot.command.update_joints.call_args.kwargs,
                             dict(velocity=False, blocking=True))
            robot.command.update_gripper.assert_not_called()
            self.assertEqual(robots[1-index].command.mock_calls, [])
        robot.config.reset_joints = None
        robot.release_required = False
        dual.request_move(robots, key, time.monotonic())
        self.assertIsNone(robot.pending)

    def test_teleop_after_motion_requires_neutral_and_release_holds_once(self):
        robot = make_robot(0)
        robot.perform('joint reset', robot.config.reset_joints)
        robot.tick(mouse_state(x=1, t=2), time.monotonic())
        robot.command.update_command.assert_not_called()
        robot.tick(mouse_state(t=3), time.monotonic())
        robot.tick(mouse_state(x=1, t=4), time.monotonic())
        self.assertEqual(robot.command.update_command.call_args.kwargs['action_space'], 'cartesian_velocity')
        robot.tick(mouse_state(t=5), time.monotonic())
        self.assertEqual(robot.command.update_command.call_args.kwargs['action_space'], 'joint_position')
        robot.tick(mouse_state(t=6), time.monotonic())
        self.assertEqual(robot.command.update_command.call_count, 2)

    def test_no_motion_at_initial_neutral_and_axis_timeout_stops(self):
        robot = make_robot(0)
        self.assertEqual(robot.command.mock_calls, [])
        now = time.monotonic()
        robot.tick(mouse_state(x=1, t=2), now)
        with self.assertRaisesRegex(RuntimeError, 'axis reports stopped'):
            robot.tick(mouse_state(x=1, t=2), now + 2)
        self.assertTrue(robot.needs_hold)

    def test_telemetry_and_other_robot_continue_during_blocking_rpc(self):
        robots = [make_robot(i) for i in (0, 1)]
        entered, release = Event(), Event()
        def blocked(*args, **kwargs):
            entered.set()
            release.wait()
        robots[0].command.update_pose.side_effect = blocked
        job = gevent.spawn(robots[0].perform, 'mirror pose', np.zeros(6))
        try:
            self.assertTrue(entered.wait(timeout=1))
            telemetry = gevent.spawn(robots[0].read_state)
            telemetry.get(timeout=1)
            robots[1].tick(mouse_state(x=1, t=2), time.monotonic())
            robots[1].command.update_command.assert_called_once()
            self.assertTrue(robots[0].busy)
            robots[0].command.update_command.assert_not_called()
        finally:
            release.set()
            job.get(timeout=1)

    def test_dashboard_reports_rotation_error_modulo_euler_wrap(self):
        robot = make_robot(0)
        target = robot.state['pose'].copy()
        target[5] += 2 * np.pi
        robot.target = ('mirror pose', target)
        lines = dual.panel_lines(robot, time.monotonic())
        self.assertTrue(any('rotation=0.00000 rad' in line for line in lines))
        self.assertTrue(any('J5-J7 [rad]' in line for line in lines))

    def test_quit_attempts_hold_for_active_robots(self):
        robots = [make_robot(i) for i in (0, 1)]
        for robot in robots:
            robot.needs_hold = True
        screen = Mock()
        screen.getch.return_value = ord('q')
        with patch.object(dual.curses, 'curs_set'):
            dual.dashboard(screen, robots)
        for robot in robots:
            self.assertEqual(robot.command.update_command.call_args.kwargs['action_space'], 'joint_position')

    def test_failed_exit_hold_is_reported(self):
        robots = [make_robot(i) for i in (0, 1)]
        robots[0].needs_hold = True
        robots[0].command.get_robot_state.side_effect = RuntimeError('disconnected')
        screen = Mock()
        screen.getch.return_value = ord('q')
        with patch.object(dual.curses, 'curs_set'), self.assertRaisesRegex(RuntimeError, 'robot0: exit hold failed'):
            dual.dashboard(screen, robots)

    def test_duplicate_endpoint_rejected_before_any_device_connection(self):
        path = 'configs/robots/robot-0.json'
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            dual.parse_args(['--robot0-config', path, '--robot1-config', path])

    def test_controller_launch_requires_explicit_flag_for_both_robots(self):
        argv = ['--robot0-config', 'configs/robots/robot-0.json',
                '--robot1-config', 'configs/robots/robot-1.json']
        for enabled in (False, True):
            configs = dual.parse_args(argv + (['--launch-controllers'] if enabled else []))
            devices = [Mock(), Mock()]
            for i, device in enumerate(devices):
                device.device.path = f'fake-hid-{i}'
            with patch.object(dual.single, 'open_spacemouse', side_effect=devices), \
                 patch.object(dual.single, 'ServerInterface') as interface, \
                 patch.object(dual.zerorpc, 'Client'), \
                 patch.object(dual.Robot, 'read_state'), \
                 patch.object(dual.curses, 'wrapper'):
                dual.run(configs)
            self.assertEqual(interface.call_count, 2)
            for call, config in zip(interface.call_args_list, configs):
                self.assertEqual(call.kwargs, dict(ip_address=config.nuc_ip,
                                                  port=config.server_port, launch=enabled))


class VerticalTests(unittest.TestCase):
    def make_vertical_robot(self):
        robot = make_robot(0)
        robot.keep_vertical = True
        robot.config.max_rot_vel = 0  # Feedback must work even with manual rotation disabled.
        return robot

    def test_vertical_axis_and_yaw(self):
        for yaw in (-3., -.5, 0., 2.9):
            target = dual.vertical_orientation([2.8, .3, yaw])
            rotation = Rotation.from_euler('xyz', target).as_matrix()
            np.testing.assert_allclose(rotation[:, 2], [0, 0, -1], atol=1e-12)
            np.testing.assert_allclose(rotation[:, 0], [np.cos(yaw), np.sin(yaw), 0], atol=1e-12)

    def test_startup_idle_corrects_gradually_with_fixed_xyz_and_yaw(self):
        robot = self.make_vertical_robot()
        start = robot.state['pose'].copy()
        robot.tick(mouse_state(), time.monotonic())
        call = robot.command.update_command.call_args
        self.assertEqual(call.kwargs, dict(action_space='cartesian_position',
                                          gripper_action_space='velocity', blocking=False))
        np.testing.assert_array_equal(call.args[0][:3], start[:3])
        self.assertEqual(call.args[0][-1], 0)
        old_rotation = Rotation.from_euler('xyz', start[3:])
        new_rotation = Rotation.from_euler('xyz', call.args[0][3:6])
        angle = (new_rotation * old_rotation.inv()).magnitude()
        self.assertGreater(angle, 0)
        self.assertLessEqual(angle, dual.VERTICAL_ROT_STEP + 1e-12)
        desired = Rotation.from_euler('xyz', robot.vertical_rpy)
        self.assertLess((desired * new_rotation.inv()).magnitude(),
                        (desired * old_rotation.inv()).magnitude())
        # Drift must not move the stored hold target or yaw on the next tick.
        raw = robot.command.get_robot_state.return_value[0]
        raw['cartesian_position'] = (start + [.01, 0, 0, 0, 0, .05]).tolist()
        robot.tick(mouse_state(), time.monotonic())
        np.testing.assert_array_equal(robot.vertical_xyz, start[:3])
        self.assertEqual(robot.vertical_rpy[2], start[5])
        self.assertEqual(robot.command.update_command.call_count, 2)

    def test_rotation_wrap_converges_to_vertical(self):
        current = np.array([-np.pi + .001, .3, -3.13])
        target = dual.vertical_orientation(current)
        for _ in range(50):
            current = dual.rotation_step(current, target)
        error = (Rotation.from_euler('xyz', current) * Rotation.from_euler('xyz', target).inv()).magnitude()
        self.assertLess(error, 1e-10)

    def test_translation_matches_existing_scale_and_release_stops_at_actual_position(self):
        robot = self.make_vertical_robot()
        current = robot.state['pose'][:3].copy()
        robot.tick(mouse_state(x=1, y=1, z=1, roll=1, pitch=1, yaw=1, buttons=[1, 0]), time.monotonic())
        target = robot.command.update_command.call_args.args[0]
        np.testing.assert_allclose(target[:3] - current, [-.5*.075, .5*.075, .5*.075])
        self.assertEqual(target[-1], 1)
        np.testing.assert_array_equal(robot.action[3:6], [0, 0, 0])
        actual = current + [.001, .002, .003]
        raw = robot.command.get_robot_state.return_value[0]
        raw['cartesian_position'][:3] = actual.tolist()
        robot.tick(mouse_state(t=2), time.monotonic())
        np.testing.assert_array_equal(robot.command.update_command.call_args.args[0][:3], actual)

    def test_no_commands_when_fresh_state_read_fails(self):
        robot = self.make_vertical_robot()
        robot.command.get_robot_state.side_effect = RuntimeError('no telemetry')
        with self.assertRaisesRegex(RuntimeError, 'no telemetry'):
            robot.tick(mouse_state(), time.monotonic())
        robot.command.update_command.assert_not_called()

    def test_reset_remains_joint_reset_then_reanchors_vertical_at_result(self):
        robot = self.make_vertical_robot()
        robot.tick(mouse_state(), time.monotonic())
        robot.command.update_command.reset_mock()
        robot.pending = ('joint reset', robot.config.reset_joints.copy())
        robot.tick(mouse_state(), time.monotonic())
        robot.command.update_joints.assert_called_once()
        robot.command.update_command.assert_not_called()
        self.assertIsNone(robot.vertical_rpy)
        raw = robot.command.get_robot_state.return_value[0]
        raw['cartesian_position'] = [.5, .1, .4, 2.9, .2, -1.2]
        robot.tick(mouse_state(), time.monotonic())  # Release gate.
        robot.tick(mouse_state(), time.monotonic())
        np.testing.assert_array_equal(robot.vertical_xyz, [.5, .1, .4])
        np.testing.assert_array_equal(robot.vertical_rpy, [np.pi, 0, -1.2])

    def test_mirror_target_respects_vertical_mode(self):
        robots = [make_robot(i) for i in (0, 1)]
        robots[1].keep_vertical = True
        dual.request_move(robots, '0', time.monotonic())
        _, target = robots[1].pending
        np.testing.assert_array_equal(target[:3], [.4, -.2, .3])
        np.testing.assert_array_equal(target[3:], [np.pi, 0, -.7])

    def test_cli_mode_is_opt_in_for_both_robots(self):
        argv = ['--robot0-config', 'configs/robots/robot-0.json',
                '--robot1-config', 'configs/robots/robot-1.json']
        self.assertFalse(any(c.keep_vertical for c in dual.parse_args(argv)))
        self.assertTrue(all(c.keep_vertical for c in dual.parse_args(argv + ['--keep-vertical'])))


class WorkspaceBoundsTests(unittest.TestCase):
    bounds = np.array([[.355, .550], [-.245, -.035], [.100, .450]])

    def bounded_robot(self, position, vertical=False):
        robot = make_robot(0)
        robot.bounds = self.bounds.copy()
        robot.keep_vertical = vertical
        raw = robot.command.get_robot_state.return_value[0]
        raw['cartesian_position'][:3] = list(position)
        return robot

    def test_all_faces_clip_and_outside_positions_do_not_snap(self):
        center = self.bounds.mean(axis=1)
        for axis in range(3):
            for direction, bound in ((-1, self.bounds[axis, 0]), (1, self.bounds[axis, 1])):
                target = center.copy()
                target[axis] = bound + direction * .02
                result = dual.bounded_xyz(center, target, self.bounds)
                self.assertEqual(result[axis], bound)
                outside = target.copy()
                np.testing.assert_array_equal(dual.bounded_xyz(outside, outside, self.bounds), outside)
                outward = outside.copy()
                outward[axis] += direction * .01
                np.testing.assert_array_equal(dual.bounded_xyz(outside, outward, self.bounds), outside)
                inward = outside.copy()
                inward[axis] -= direction * .005
                np.testing.assert_array_equal(dual.bounded_xyz(outside, inward, self.bounds), inward)

    def test_velocity_target_clips_before_crossing_preserves_rotation_and_gripper(self):
        robot = self.bounded_robot([.549, -.13, .25])
        mouse = mouse_state(y=-1, roll=.5, buttons=[1, 0])
        robot.tick(mouse, time.monotonic())
        call = robot.command.update_command.call_args
        action = call.args[0]
        self.assertEqual(call.kwargs['action_space'], 'cartesian_velocity')
        self.assertAlmostEqual(.549 + action[0] * dual.DROID_MAX_LIN_DELTA, .550)
        np.testing.assert_array_equal(action[3:], [-.05, 0., 0., 1.])
        self.assertIn('limited', robot.bounds_note)

    def test_boundary_allows_tangent_motion_and_inward_return(self):
        robot = self.bounded_robot([.550, -.13, .25])
        robot.tick(mouse_state(y=-1, x=1), time.monotonic())
        action = robot.command.update_command.call_args.args[0]
        self.assertEqual(action[0], 0)
        self.assertGreater(action[1], 0)
        robot.tick(mouse_state(y=1, t=2), time.monotonic())
        self.assertLess(robot.command.update_command.call_args.args[0][0], 0)

    def test_vertical_mode_clips_xyz_but_keeps_orientation_feedback(self):
        robot = self.bounded_robot([.549, -.13, .25], vertical=True)
        robot.tick(mouse_state(y=-1), time.monotonic())
        call = robot.command.update_command.call_args
        self.assertEqual(call.kwargs['action_space'], 'cartesian_position')
        self.assertEqual(call.args[0][0], .550)
        self.assertGreater(np.linalg.norm(call.args[0][3:6] - robot.state['pose'][3:]), 0)
        # Release stops at the newly measured position instead of continuing to the boundary.
        robot.command.get_robot_state.return_value[0]['cartesian_position'][0] = .5495
        robot.tick(mouse_state(t=2), time.monotonic())
        self.assertEqual(robot.command.update_command.call_args.args[0][0], .5495)

    def test_vertical_mode_outside_does_not_automatically_pull_xyz_inside(self):
        robot = self.bounded_robot([.6, -.13, .25], vertical=True)
        robot.tick(mouse_state(), time.monotonic())
        self.assertEqual(robot.command.update_command.call_args.args[0][0], .6)
        self.assertIn('Outside bounds', robot.bounds_note)

    def test_mirror_outside_destination_rejected_but_joint_reset_allowed(self):
        robots = [make_robot(i) for i in (0, 1)]
        robots[1].bounds = np.array([[.355,.55],[.035,.245],[.1,.45]])
        robots[0].state['pose'][:3] = [.4,-.13,.25]
        dual.request_move(robots, '0', time.monotonic())
        self.assertIsNotNone(robots[1].pending)
        robots[1].pending = None
        robots[0].state['pose'][0] = .6
        message = dual.request_move(robots, '0', time.monotonic())
        self.assertIn('rejected', message)
        self.assertIsNone(robots[1].pending)
        dual.request_move(robots, 't', time.monotonic())
        self.assertEqual(robots[1].pending[0], 'joint reset')

    def test_bounds_are_opt_in_and_match_each_json(self):
        argv = ['--robot0-config','configs/robots/robot-0.json',
                '--robot1-config','configs/robots/robot-1.json']
        self.assertTrue(all(c.bounds is None for c in dual.parse_args(argv)))
        configs = dual.parse_args(argv + ['--use-bounds'])
        np.testing.assert_array_equal(configs[0].bounds, self.bounds)
        np.testing.assert_array_equal(configs[1].bounds[1], [.035,.245])

    def test_invalid_missing_bounds_fail_before_opening_hardware(self):
        config = json.loads(Path('configs/robots/robot-0.json').read_text())
        for value in (None, [[0,1]]*2, [[1,0]]*3, [[0,float('nan')]]*3):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'robot.json'
                config['bounds'] = value
                path.write_text(json.dumps(config))
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    dual.parse_args(['--robot0-config',str(path), '--robot1-config',
                                     'configs/robots/robot-1.json', '--use-bounds'])


if __name__ == '__main__':
    unittest.main()
