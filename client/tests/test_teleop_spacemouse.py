"""Offline teleoperation checks; all HID and robot access is mocked."""

from contextlib import redirect_stdout, redirect_stderr
import io
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from client import teleop_spacemouse as teleop


def mouse_state(**overrides):
    values = dict(t=1.0, x=0., y=0., z=0., roll=0., pitch=0., yaw=0., buttons=[0, 0])
    values.update(overrides)
    return SimpleNamespace(**values)


class TeleopTests(unittest.TestCase):
    def setUp(self):
        self.args = teleop.parse_args([])
        self.robot = Mock()
        self.joints = [0.1, 0.2, 0.3, -1.5, 0.4, 1.6, 0.7]
        self.robot.get_robot_state.return_value = (
            {"joint_positions": self.joints, "gripper_position": 0.25}, {}
        )

    def run_until_interrupt(self, states):
        device = Mock()
        device.read.side_effect = [*states, KeyboardInterrupt()]
        with patch.object(teleop.time, "monotonic", side_effect=itertools.count(0, 0.2)), \
             patch.object(teleop.time, "sleep"), self.assertRaises(KeyboardInterrupt):
            teleop.teleoperate(self.robot, device, self.args)

    def assert_hold(self, call):
        np.testing.assert_allclose(call.args[0], [*self.joints, 0.75])
        self.assertEqual(call.kwargs["action_space"], "joint_position")
        self.assertEqual(call.kwargs["gripper_action_space"], "position")
        self.assertFalse(call.kwargs["blocking"])

    def test_axis_mapping_and_gripper_buttons(self):
        state = mouse_state(x=1, y=0.4, z=-1, roll=0.5, pitch=-0.5, yaw=1, buttons=[1, 0])
        np.testing.assert_allclose(teleop.action_from_state(state, self.args),
                                   [-0.2, 0.5, -0.5, -0.05, 0.05, -0.1, 1])
        self.assertEqual(teleop.action_from_state(mouse_state(buttons=[0, 1]), self.args)[-1], -1)
        self.assertEqual(teleop.action_from_state(mouse_state(buttons=[1, 1]), self.args)[-1], 0)
        np.testing.assert_array_equal(teleop.action_from_state(mouse_state(x=0.01), self.args),
                                      np.zeros(7))

    def test_invalid_device_does_not_silently_select_zero(self):
        with patch.object(teleop.pyspacemouse, "list_devices", return_value=["SpaceMouse Wireless"]), \
             patch.object(teleop.pyspacemouse, "open") as open_device:
            with self.assertRaises(ValueError):
                teleop.open_spacemouse(1)
            open_device.assert_not_called()

    def test_device_one_is_forwarded(self):
        with patch.object(teleop.pyspacemouse, "list_devices", return_value=["SpaceMouse Wireless"] * 2), \
             patch.object(teleop.pyspacemouse, "open") as open_device:
            self.assertIs(teleop.open_spacemouse(1), open_device.return_value)
            open_device.assert_called_once_with(device="SpaceMouse Wireless", DeviceNumber=1)

    def test_no_device_never_opens_anything(self):
        with patch.object(teleop.pyspacemouse, "list_devices", return_value=[]), \
             patch.object(teleop.pyspacemouse, "open") as open_device:
            with self.assertRaises(RuntimeError):
                teleop.open_spacemouse(0)
            open_device.assert_not_called()

    def test_idle_start_and_exit_send_no_motion_or_gripper_commands(self):
        self.run_until_interrupt([mouse_state(), mouse_state()])
        self.robot.update_command.assert_not_called()

    def test_interrupt_after_motion_holds_pose_and_gripper(self):
        self.run_until_interrupt([mouse_state(x=1)])
        calls = self.robot.update_command.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs["action_space"], "cartesian_velocity")
        self.assert_hold(calls[1])

    def test_releasing_controls_holds_once(self):
        self.run_until_interrupt([mouse_state(x=1), mouse_state(t=2), mouse_state(t=3)])
        calls = self.robot.update_command.call_args_list
        self.assertEqual(len(calls), 2)
        self.assert_hold(calls[1])

    def test_stale_axis_reports_end_control_and_hold(self):
        device = Mock()
        device.read.return_value = mouse_state(x=1)
        with patch.object(teleop.time, "monotonic", side_effect=itertools.count(0, 0.6)), \
             patch.object(teleop.time, "sleep"), self.assertRaisesRegex(RuntimeError, "reports stopped"):
            teleop.teleoperate(self.robot, device, self.args)
        self.assert_hold(self.robot.update_command.call_args_list[-1])

    def test_hid_read_failure_ends_control_and_holds(self):
        device = Mock()
        device.read.side_effect = [mouse_state(x=1), RuntimeError("HID disconnected")]
        with patch.object(teleop.time, "monotonic", side_effect=itertools.count(0, 0.2)), \
             patch.object(teleop.time, "sleep"), self.assertRaisesRegex(RuntimeError, "HID disconnected"):
            teleop.teleoperate(self.robot, device, self.args)
        self.assert_hold(self.robot.update_command.call_args_list[-1])

    def test_connection_is_reused_without_launch_or_reset(self):
        device = Mock()
        with patch.object(teleop, "open_spacemouse", return_value=device), \
             patch.object(teleop, "ServerInterface", return_value=self.robot) as interface, \
             patch.object(teleop, "teleoperate") as loop, redirect_stdout(io.StringIO()):
            teleop.run(self.args)
        interface.assert_called_once_with(ip_address=self.args.nuc_ip, port=4242, launch=False)
        self.robot.launch_controller.assert_not_called()
        self.robot.launch_robot.assert_not_called()
        self.robot.update_command.assert_not_called()
        loop.assert_called_once_with(self.robot, device, self.args)
        device.close.assert_called_once()
        self.robot.server.close.assert_called_once()

    def test_unready_controller_reports_error_without_automatic_launch(self):
        device = Mock()
        self.robot.get_robot_state.side_effect = RuntimeError("not initialized")
        with patch.object(teleop, "open_spacemouse", return_value=device), \
             patch.object(teleop, "ServerInterface", return_value=self.robot), \
             patch.object(teleop, "teleoperate") as loop, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Cannot read robot state"):
                teleop.run(self.args)
        loop.assert_not_called()
        self.robot.launch_controller.assert_not_called()
        self.robot.update_command.assert_not_called()
        device.close.assert_called_once()
        self.robot.server.close.assert_called_once()

    def test_controller_start_requires_explicit_flag(self):
        self.args.launch_controllers = True
        with patch.object(teleop, "open_spacemouse"), \
             patch.object(teleop, "ServerInterface", return_value=self.robot) as interface, \
             patch.object(teleop, "teleoperate"), redirect_stdout(io.StringIO()):
            teleop.run(self.args)
        interface.assert_called_once_with(ip_address=self.args.nuc_ip, port=4242, launch=True)
        self.robot.launch_controller.assert_not_called()
        self.robot.launch_robot.assert_not_called()

    def test_robot_configs_route_matching_mouse_and_server(self):
        root = Path(teleop.__file__).resolve().parents[1]
        for index, port, path in ((0, 4242, "/dev/hidraw2"), (1, 4243, "/dev/hidraw0")):
            with self.subTest(robot=index):
                args = teleop.parse_args(["--robot-config", str(root / f"configs/robots/robot-{index}.json")])
                with patch.object(teleop, "open_spacemouse") as mouse, \
                     patch.object(teleop, "ServerInterface", return_value=self.robot) as interface, \
                     patch.object(teleop, "teleoperate"), redirect_stdout(io.StringIO()):
                    teleop.run(args)
                mouse.assert_called_once_with(0, path)
                interface.assert_called_once_with(ip_address="172.16.0.1", port=port, launch=False)

    def test_cli_selection_overrides_config_path(self):
        root = Path(teleop.__file__).resolve().parents[1]
        args = teleop.parse_args(["--robot-config", str(root / "configs/robots/robot-1.json"),
                                 "--device-number", "1", "--server-port", "4249"])
        self.assertIsNone(args.device_path)
        self.assertEqual(args.device_number, 1)
        self.assertEqual(args.server_port, 4249)

    def test_invalid_config_rejected_before_hardware_access(self):
        for config in ([], {}, {"robot_server_ip": "172.16.0.1"},
                       {"robot_server_ip": "172.16.0.1", "robot_server_port": 4243},
                       {"robot_server_ip": "172.16.0.1", "robot_server_port": "4243",
                        "spacemouse_device_path": "/dev/hidraw0"}):
            with self.subTest(config=config), tempfile.TemporaryDirectory() as tmp:
                file = Path(tmp) / "robot.json"
                file.write_text(json.dumps(config))
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    teleop.parse_args(["--robot-config", str(file)])

    def test_exact_path_selects_matching_model(self):
        hid = SimpleNamespace(path="/dev/hidraw0", vendor_id=0x256f, product_id=0xc62e)
        with patch.object(teleop.pyspacemouse, "Enumeration") as enum, \
             patch.object(teleop.pyspacemouse, "open") as open_device:
            enum.return_value.find.return_value = [hid]
            self.assertIs(teleop.open_spacemouse(0, hid.path), open_device.return_value)
            open_device.assert_called_once_with(device="SpaceMouse Wireless", path=hid.path)

    def test_bad_path_never_falls_back_to_another_mouse(self):
        for devices in ([], [SimpleNamespace(path="/dev/hidraw0", vendor_id=0xffff, product_id=0xffff)]):
            with patch.object(teleop.pyspacemouse, "Enumeration") as enum, \
                 patch.object(teleop.pyspacemouse, "open") as open_device:
                enum.return_value.find.return_value = devices
                with self.assertRaises(ValueError):
                    teleop.open_spacemouse(0, "/dev/hidraw0")
                open_device.assert_not_called()

    def test_attach_failure_closes_mouse_without_entering_loop(self):
        with patch.object(teleop, "open_spacemouse") as mouse, \
             patch.object(teleop, "ServerInterface", side_effect=RuntimeError("not ready")), \
             patch.object(teleop, "teleoperate") as loop, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Cannot read robot state"):
                teleop.run(self.args)
            mouse.return_value.close.assert_called_once()
            loop.assert_not_called()

    def test_invalid_numeric_options_are_rejected(self):
        for argv in (["--device-number", "-1"], ["--max-lin-vel", "nan"],
                     ["--max-rot-vel", "inf"], ["--deadzone", "1"],
                     ["--max-lin-vel", "-0.1"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                teleop.parse_args(argv)


if __name__ == "__main__":
    unittest.main()
