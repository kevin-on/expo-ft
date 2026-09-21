"""SpaceMouse teleoperation without cameras, recording, detection, or reset."""

import argparse
from contextlib import ExitStack
import math
import json
import sys
import time

import numpy as np
import pyspacemouse

from droid.misc.parameters import nuc_ip
from droid.misc.server_interface import ServerInterface


CONTROL_PERIOD = 0.1  # Match the pick collection loop's 10 Hz command rate.
AXIS_INPUT_TIMEOUT = 1.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", help="Robot JSON with server address/port and SpaceMouse selection; cameras are ignored.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--device-number", type=int, help="SpaceMouse index (default: 0 without a robot config).")
    selection.add_argument("--device-path", help="Exact HID path, e.g. /dev/hidraw2; overrides JSON selection.")
    parser.add_argument("--nuc-ip", help="Override the NUC address.")
    parser.add_argument("--server-port", type=int, help="Override the DROID RPC port (default: 4242 without a robot config).")
    parser.add_argument("--max-lin-vel", type=float, default=0.5,
                        help="Normalized translation scale, not m/s (default: 0.5).")
    parser.add_argument("--max-rot-vel", type=float, default=0.1,
                        help="Normalized rotation scale, not rad/s (default: 0.1).")
    parser.add_argument("--deadzone", type=float, default=0.05,
                        help="Ignore small axis inputs (default: 0.05).")
    parser.add_argument("--launch-controllers", action="store_true",
                        help="Explicitly start/restart this server's arm and gripper "
                             "controllers; normally reuse the running controllers.")
    args = parser.parse_args(argv)
    config = {}
    if args.robot_config:
        try:
            with open(args.robot_config) as file:
                config = json.load(file)
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot read robot config: {exc}")
        if not isinstance(config, dict):
            parser.error("Robot config must be a JSON object")
        # A partial robot config must not silently route to robot 0.
        if args.nuc_ip is None and not config.get("robot_server_ip"):
            parser.error("Robot config requires robot_server_ip")
        if args.server_port is None and "robot_server_port" not in config:
            parser.error("Robot config requires robot_server_port")
        if (args.device_path is None and args.device_number is None
                and not config.get("spacemouse_device_path")
                and "spacemouse_device_number" not in config):
            parser.error("Robot config requires a SpaceMouse path or number")
    if args.nuc_ip is None:
        args.nuc_ip = config.get("robot_server_ip", nuc_ip)
    if args.server_port is None:
        args.server_port = config.get("robot_server_port", 4242)
    if args.device_path is None and args.device_number is None:
        args.device_path = config.get("spacemouse_device_path")
        args.device_number = config.get("spacemouse_device_number", 0)
    if args.device_number is None:
        args.device_number = 0
    if type(args.server_port) is not int or not 1 <= args.server_port <= 65535:
        parser.error("Server port must be an integer between 1 and 65535")
    if args.device_path is not None and (not isinstance(args.device_path, str) or not args.device_path):
        parser.error("SpaceMouse path must be a nonempty string")
    if type(args.device_number) is not int or args.device_number < 0:
        parser.error("--device-number must be non-negative")
    if not isinstance(args.nuc_ip, str) or not args.nuc_ip.strip():
        parser.error("Set nuc_ip in DROID parameters or pass --nuc-ip")
    for name in ("max_lin_vel", "max_rot_vel", "deadzone"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            parser.error("--" + name.replace("_", "-") + " must be finite and between 0 and 1")
    if args.deadzone == 1:
        parser.error("--deadzone must be less than 1")
    return args


def open_spacemouse(device_number, device_path=None):
    if device_path is not None:
        hid = next((dev for dev in pyspacemouse.Enumeration().find()
                    if dev.path == device_path), None)
        if hid is None:
            raise ValueError(f"SpaceMouse HID path not found: {device_path}")
        name = next((name for name, spec in pyspacemouse.device_specs.items()
                     if (hid.vendor_id, hid.product_id) == tuple(spec.hid_id)), None)
        if name is None:
            raise ValueError(f"Unsupported SpaceMouse at HID path: {device_path}")
        device = pyspacemouse.open(device=name, path=device_path)
        if device is None:
            raise RuntimeError(f"Failed to open SpaceMouse at {device_path}")
        return device
    names = pyspacemouse.list_devices()
    if not names:
        raise RuntimeError("No supported SpaceMouse found")
    name = names[0]
    count = names.count(name)
    # pyspacemouse 1.x silently falls back to device 0 for an invalid index.
    if not 0 <= device_number < count:
        raise ValueError(f"Device {device_number} is unavailable; {count} {name} device(s) found")
    device = pyspacemouse.open(device=name, DeviceNumber=device_number)
    if device is None:
        raise RuntimeError("Failed to open SpaceMouse")
    return device


def action_from_state(state, args):
    # Same axis mapping as client.real_utils.spacemouse.SpaceMouseExpert.
    axes = np.array([-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw],
                    dtype=np.float64)
    if not np.isfinite(axes).all():
        raise RuntimeError("SpaceMouse returned non-finite axis values")
    axes[np.abs(axes) < args.deadzone] = 0.0
    axes[:3] *= args.max_lin_vel
    axes[3:] *= args.max_rot_vel
    buttons = state.buttons
    close = bool(buttons[0]) if len(buttons) > 0 else False
    open_ = bool(buttons[1]) if len(buttons) > 1 else False
    # Neither/both buttons: hold. Unlike collection, release does not open the gripper.
    return np.clip(np.r_[axes, int(close) - int(open_)], -1.0, 1.0)


def hold_current_pose(robot):
    state, _ = robot.get_robot_state()
    # State stores normalized opening width; absolute commands store closure.
    command = np.r_[state["joint_positions"], 1.0 - state["gripper_position"]]
    robot.update_command(command, action_space="joint_position",
                         gripper_action_space="position", blocking=False)


def teleoperate(robot, device, args):
    active = False
    next_send = 0.0
    last_report = None
    last_report_at = time.monotonic()
    try:
        while True:
            state = device.read()
            if state is None:
                raise RuntimeError("SpaceMouse stopped returning state")
            now = time.monotonic()
            if state.t != last_report:
                last_report, last_report_at = state.t, now
            if now >= next_send:
                action = action_from_state(state, args)
                # Do not repeat a cached nonzero axis command after input loss.
                # Button-only reports may occur only on press/release, so their
                # held state is not subject to the axis-report timeout.
                if np.any(action[:6]) and now - last_report_at > AXIS_INPUT_TIMEOUT:
                    raise RuntimeError("SpaceMouse axis reports stopped; ending teleoperation")
                if np.any(action):
                    # Mark active before the RPC: a failed reply may still mean execution.
                    active = True
                    robot.update_command(action, action_space="cartesian_velocity",
                                         gripper_action_space="velocity", blocking=False)
                elif active:
                    hold_current_pose(robot)
                    active = False
                next_send = now + CONTROL_PERIOD
            # Drain HID events promptly without a background thread or busy loop.
            time.sleep(0.001)
    finally:
        if active:
            try:
                hold_current_pose(robot)
            except Exception as exc:
                print(f"Could not confirm the final hold command: {exc}", file=sys.stderr)


def run(args):
    with ExitStack() as cleanup:
        device = open_spacemouse(args.device_number, args.device_path)
        cleanup.callback(device.close)
        print(f"SpaceMouse: {device.device.path} -> DROID {args.nuc_ip}:{args.server_port}", flush=True)
        if args.launch_controllers:
            print("Starting this NUC server's arm/gripper controllers...", flush=True)
        else:
            print("Attaching to running arm/gripper controllers...", flush=True)
        try:
            # In the fork, launch=False already attaches the robot interfaces.
            # launch=True must start controllers BEFORE attempting that attachment.
            robot = ServerInterface(ip_address=args.nuc_ip, port=args.server_port,
                                    launch=args.launch_controllers)
            cleanup.callback(robot.server.close)
            state, _ = robot.get_robot_state()
        except Exception as exc:
            raise RuntimeError(
                "Cannot read robot state. Check the NUC server/controllers. "
                "Use --launch-controllers only if you intend to start/restart them. "
                f"Server error: {exc}"
            ) from exc
        print("Current joints:", np.round(state["joint_positions"], 4), flush=True)
        print("Starting at the current pose; no task workspace bounds are applied.", flush=True)
        print("Move/twist: arm | button 0: close | button 1: open | release: hold", flush=True)
        print("Ctrl+C: hold current pose and exit.", flush=True)
        teleoperate(robot, device, args)


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("Teleoperation ended.")
        return 0
    except Exception as exc:
        print(f"Teleoperation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
