"""Two independent SpaceMouse robots with one-shot mirrored pose moves and resets.

0: robot0 pose -> mirrored robot1 pose; 1: robot1 -> mirrored robot0.
r: robot0 joint reset; t: robot1 joint reset; q/Ctrl+C: exit and attempt hold.
No cameras or automatic initial reset. --use-bounds limits teleop XYZ targets
using each JSON's bounds; joint resets and motion trajectories are not constrained.
Controllers are reused unless --launch-controllers is explicitly passed.
--keep-vertical actively aligns tool +Z with base -Z, including at startup.
"""
import argparse
from contextlib import ExitStack
import curses
import json
import time

import gevent
import numpy as np
from scipy.spatial.transform import Rotation
import zerorpc

from client import teleop_spacemouse as single
from client.real_utils.vertical_control import VERTICAL_ROT_STEP, rotation_step, vertical_orientation


# DROID RobotIKSolver's normalized translation scale (also checked on the NUC).
DROID_MAX_LIN_DELTA = .075


def bounded_xyz(current, target, bounds):
    """Clip commanded XYZ; outside the box allow only hold or inward movement.

    Extending each interval to the measured position avoids automatically snapping
    an out-of-bounds arm back to a boundary when the operator gives no XYZ input.
    """
    return np.clip(target, np.minimum(current, bounds[:, 0]),
                   np.maximum(current, bounds[:, 1]))


def mirror_pose(pose):
    """Match dataset XYZ + extrinsic xyz Euler reflection about base y=0."""
    pose = np.array(pose, dtype=np.float64, copy=True)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("Expected finite [x,y,z,roll,pitch,yaw], meters/radians")
    pose[[1, 3, 5]] *= -1
    return pose


def checked_state(state):
    pose = np.array(state['cartesian_position'], dtype=np.float64, copy=True)
    joints = np.array(state['joint_positions'], dtype=np.float64, copy=True)
    gripper = float(state['gripper_position'])
    if (pose.shape != (6,) or joints.shape != (7,) or not np.isfinite(pose).all()
            or not np.isfinite(joints).all() or not np.isfinite(gripper)):
        raise ValueError('Invalid pose/joint/gripper telemetry')
    return dict(pose=pose, joints=joints, gripper=gripper)


class Robot:
    def __init__(self, index, config, device, command, reader):
        self.index, self.config, self.device = index, config, device
        # Separate sockets/greenlets keep telemetry flowing during blocking motion RPCs.
        self.command, self.reader = command, reader
        self.state, self.received_at = None, None
        self.action = np.zeros(7)
        self.input_at = None
        self.last_report, self.last_report_at = None, None
        self.pending = None
        self.busy = False
        self.release_required = False
        self.needs_hold = False
        self.mode = 'connecting'
        self.note = ''
        self.target = None
        self.keep_vertical = getattr(config, 'keep_vertical', False)
        self.vertical_rpy = None
        self.vertical_xyz = None
        self.vertical_translating = False
        self.bounds = getattr(config, 'bounds', None)
        self.bounds_note = ''

    def ready(self, now):
        return (self.state is not None and self.received_at is not None
                and 0 <= now - self.received_at < 1.0
                and self.input_at is not None and 0 <= now - self.input_at < 1.0
                and not self.busy and self.pending is None and not self.release_required
                and not np.any(self.action))

    def read_state(self):
        state, _ = self.reader.get_robot_state()
        self.state = checked_state(state)
        self.received_at = time.monotonic()

    def perform(self, operation, target):
        self.busy = True
        self.needs_hold = True  # RPC failure does not imply the command never ran.
        self.mode = operation
        self.target = (operation, target.copy())
        try:
            if operation == 'mirror pose':
                self.command.update_pose(target, velocity=False, blocking=True)
            else:
                self.command.update_joints(target, velocity=False, blocking=True)
            # Server-side movement errors can be swallowed: show error to target, not "success".
            self.note = 'Motion RPC returned; inspect actual pose / target error.'
            self.needs_hold = False
            self.release_required = True
            self.mode = 'release SpaceMouse'
            # Reset/mirror motions are separate. Resume with their resulting XYZ/yaw.
            self.vertical_rpy = self.vertical_xyz = None
            self.vertical_translating = False
        finally:
            self.busy = False

    def control_state(self):
        """Fresh state on the command socket for constructing the next target."""
        raw, _ = self.command.get_robot_state()
        self.state = checked_state(raw)
        self.received_at = time.monotonic()
        return self.state['pose']

    def limit_xyz(self, current, target):
        if self.bounds is None:
            return target
        limited = bounded_xyz(current, target, self.bounds)
        if np.any(current < self.bounds[:, 0]) or np.any(current > self.bounds[:, 1]):
            self.bounds_note = 'Outside bounds: only hold/inward XYZ commands allowed.'
        elif not np.allclose(limited, target, rtol=0, atol=1e-12):
            self.bounds_note = 'XYZ target limited at boundary; inward/tangent motion still allowed.'
        else:
            self.bounds_note = ''
        return limited

    def vertical_tick(self):
        # Read on the command socket: the 5 Hz dashboard cache is too old for
        # constructing absolute position commands in this 10 Hz control loop.
        pose = self.control_state()
        if self.vertical_rpy is None:
            self.vertical_rpy = vertical_orientation(pose[3:])
        translating = bool(np.any(self.action[:3]))
        if translating:
            linear = self.action[:3] / max(1., np.linalg.norm(self.action[:3]))
            # Same XYZ step as the existing normalized cartesian_velocity path.
            self.vertical_xyz = pose[:3] + linear * DROID_MAX_LIN_DELTA
        elif self.vertical_xyz is None or self.vertical_translating:
            # Release at the measured position, not the previous forward target.
            self.vertical_xyz = pose[:3].copy()
        self.vertical_translating = translating
        self.vertical_xyz = self.limit_xyz(pose[:3], self.vertical_xyz)
        target = np.r_[self.vertical_xyz, rotation_step(pose[3:], self.vertical_rpy), self.action[6]]
        self.target = ('vertical pose', np.r_[self.vertical_xyz, self.vertical_rpy])
        self.needs_hold = True
        self.mode = 'vertical teleop' if translating else 'vertical hold'
        self.command.update_command(target, action_space='cartesian_position',
                                    gripper_action_space='velocity', blocking=False)

    def tick(self, mouse, now):
        self.action = single.action_from_state(mouse, self.config)
        if self.keep_vertical:
            self.action[3:6] = 0.  # Ignore manual rotation; feedback supplies correction.
        self.input_at = now
        if mouse.t != self.last_report:
            self.last_report, self.last_report_at = mouse.t, now
        if self.release_required:
            if not np.any(self.action):
                self.release_required = False
                self.mode = 'hold'
            return
        if np.any(self.action[:6]) and now - self.last_report_at > single.AXIS_INPUT_TIMEOUT:
            raise RuntimeError(f'robot{self.index}: SpaceMouse axis reports stopped')
        if self.pending is not None:
            operation, target = self.pending
            self.pending = None
            if np.any(self.action):
                self.note = 'Command cancelled: release SpaceMouse and retry.'
            else:
                self.perform(operation, target)
                return
        if self.keep_vertical:
            self.vertical_tick()
            return
        if np.any(self.action):
            action = self.action.copy()
            if self.bounds is not None:
                current = self.control_state()[:3]
                linear = action[:3] / max(1., np.linalg.norm(action[:3]))
                target = self.limit_xyz(current, current + linear * DROID_MAX_LIN_DELTA)
                action[:3] = (target - current) / DROID_MAX_LIN_DELTA
            self.needs_hold = True
            self.target = None
            self.mode = 'teleop'
            self.command.update_command(action, action_space='cartesian_velocity',
                                        gripper_action_space='velocity', blocking=False)
        elif self.needs_hold:
            single.hold_current_pose(self.command)
            self.needs_hold = False
            self.mode = 'hold'
        else:
            self.mode = 'hold'

    def control_loop(self):
        next_send = 0.0
        while True:
            mouse = self.device.read()
            if mouse is None:
                raise RuntimeError(f'robot{self.index}: SpaceMouse disconnected')
            now = time.monotonic()
            # Drain HID reports between command ticks without sending extra commands.
            if mouse.t != self.last_report:
                self.last_report, self.last_report_at = mouse.t, now
            if now >= next_send:
                self.tick(mouse, now)
                next_send = time.monotonic() + single.CONTROL_PERIOD
            gevent.sleep(.001)

    def telemetry_loop(self):
        while True:
            self.read_state()
            gevent.sleep(.2)


def request_move(robots, key, now):
    """Only the destination gets a fixed pose target; no continuous following."""
    if key in ('0', '1'):
        source = int(key)
        destination = 1 - source
        if not all(robot.ready(now) for robot in robots):
            return 'Mirror: release BOTH SpaceMice; both robots must be idle with fresh telemetry.'
        target = mirror_pose(robots[source].state['pose'])
        bounds = robots[destination].bounds
        if bounds is not None and (np.any(target[:3] < bounds[:, 0]) or np.any(target[:3] > bounds[:, 1])):
            return f'Mirror rejected: target XYZ is outside robot{destination} bounds.'
        if robots[destination].keep_vertical:
            target[3:] = vertical_orientation(target[3:])
        robots[destination].pending = ('mirror pose', target)
        return f'robot{source} -> robot{destination}: one-shot mirrored pose; gripper unchanged.'
    if key in ('r', 't'):
        index = 0 if key == 'r' else 1
        robot = robots[index]
        if not robot.ready(now):
            return f'robot{index}: release SpaceMouse and wait for fresh telemetry.'
        if robot.config.reset_joints is None:
            return f'robot{index}: reset_joints missing from its config.'
        robot.pending = ('joint reset', robot.config.reset_joints.copy())
        return f'robot{index}: joint reset requested; gripper unchanged.'
    return None


def panel_lines(robot, now):
    lines = [f'ROBOT {robot.index} | {robot.config.nuc_ip}:{robot.config.server_port} | {robot.mode}']
    if robot.state is None:
        return lines + ['Waiting for robot state...']
    age = now - robot.received_at
    pose, joints = robot.state['pose'], robot.state['joints']
    values = lambda a: ' '.join(f'{v: .5f}' for v in a)
    lines += [f'State age: {age:.2f}s' + ('  STALE' if age >= 1 else ''),
              'XYZ [m]       ' + values(pose[:3]),
              'RPY [rad]     ' + values(pose[3:]),
              'J1-J4 [rad]   ' + values(joints[:4]),
              'J5-J7 [rad]   ' + values(joints[4:]),
              f'Gripper opening [0..1]: {robot.state["gripper"]:.4f}']
    if robot.target is not None:
        kind, target = robot.target
        if kind in ('mirror pose', 'vertical pose'):
            angle = (Rotation.from_euler('xyz', target[3:]) *
                     Rotation.from_euler('xyz', pose[3:]).inv()).magnitude()
            lines += [f'Target error: position={np.linalg.norm(target[:3]-pose[:3]):.5f} m  rotation={angle:.5f} rad']
        else:
            lines += [f'Target error: max joint={np.max(np.abs(target-joints)):.5f} rad']
    else:
        lines += ['Target error: --']
    if robot.bounds is not None:
        limits = ' '.join(f'{axis}[{lo:.3f},{hi:.3f}]' for axis, (lo, hi) in zip('XYZ', robot.bounds))
        lines[1] += ' | ' + limits
    return lines + [robot.bounds_note or robot.note]


def draw(screen, robots, message):
    screen.erase()
    height, width = screen.getmaxyx()
    bounds_label = ('JSON bounds ON (teleop targets; reset exempt)' if any(r.bounds is not None for r in robots)
                    else 'no task XYZ bounds')
    rows = [f'Two-robot teleop | physical base coordinates | {bounds_label}',
            '0: mirror 0->1  1: mirror 1->0  r: reset 0  t: reset 1  q: quit',
            message]
    for robot in robots:
        rows.extend(panel_lines(robot, time.monotonic()))
        rows.append('')
    for row, line in enumerate(rows[:height]):
        try:
            screen.addnstr(row, 0, line, max(0, width - 1))
        except curses.error:
            pass
    screen.refresh()


def dashboard(screen, robots):
    screen.nodelay(True)
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    jobs = [gevent.spawn(method) for robot in robots
            for method in (robot.control_loop, robot.telemetry_loop)]
    message = 'Ready. No automatic reset. Single-key commands; no Enter needed.'
    if any(robot.keep_vertical for robot in robots):
        message = 'VERTICAL ACTIVE: aligns tool +Z toward base -Z; XYZ / gripper teleop. Reset is an exception.'
    try:
        while True:
            for job in jobs:
                if job.ready():
                    job.get()
                    raise RuntimeError('Robot worker unexpectedly stopped')
            key = screen.getch()
            if key in (ord('q'), 3):
                break
            if 0 <= key < 256:
                message = request_move(robots, chr(key).lower(), time.monotonic()) or message
            draw(screen, robots, message)
            gevent.sleep(.05)
    finally:
        for job in jobs:
            job.kill(block=False)
        gevent.joinall(jobs, timeout=1)
        # Only our two clients. A local RPC cancellation cannot guarantee cancelling
        # a trajectory already running inside the NUC; hardware stop remains separate.
        def hold(robot):
            if robot.needs_hold or robot.busy:
                try:
                    with gevent.Timeout(3):
                        single.hold_current_pose(robot.command)
                except (Exception, gevent.Timeout) as exc:
                    return f'robot{robot.index}: exit hold failed ({type(exc).__name__}: {exc})'
        holds = [gevent.spawn(hold, robot) for robot in robots]
        gevent.joinall(holds, timeout=4)
        failures = []
        for job in holds:
            if not job.ready():
                job.kill()
                failures.append('Exit hold timed out')
            elif job.value:
                failures.append(job.value)
        if failures:
            raise RuntimeError('; '.join(failures))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot0-config', required=True)
    parser.add_argument('--robot1-config', required=True)
    parser.add_argument('--launch-controllers', action='store_true',
                        help="Start/restart both configured robots' arm and gripper controllers.")
    parser.add_argument('--keep-vertical', action='store_true',
                        help='Actively keep tool +Z toward base -Z; ignore mouse rotation. '
                             'Starts correcting immediately; reset is an exception, then alignment resumes.')
    parser.add_argument('--use-bounds', action='store_true',
                        help='Limit teleop XYZ targets using each robot JSON bounds (meters, base frame). '
                             'Outside mirror targets are rejected; joint reset/trajectories are not constrained.')
    parser.add_argument('--max-lin-vel', type=float, default=.5)
    parser.add_argument('--max-rot-vel', type=float, default=.1)
    parser.add_argument('--deadzone', type=float, default=.05)
    args = parser.parse_args(argv)
    configs = [single.parse_args(['--robot-config', path, '--max-lin-vel', str(args.max_lin_vel),
               '--max-rot-vel', str(args.max_rot_vel), '--deadzone', str(args.deadzone)]
               + (['--launch-controllers'] if args.launch_controllers else []))
               for path in (args.robot0_config, args.robot1_config)]
    for config in configs:
        config.keep_vertical = args.keep_vertical
        config.bounds = None
        if args.use_bounds:
            try:
                with open(config.robot_config) as file:
                    bounds = np.asarray(json.load(file)['bounds'], dtype=np.float64)
                if (bounds.shape != (3, 2) or not np.isfinite(bounds).all()
                        or not np.all(bounds[:, 0] < bounds[:, 1])):
                    raise ValueError('expected finite [[xmin,xmax],[ymin,ymax],[zmin,zmax]], min < max')
            except (OSError, KeyError, TypeError, ValueError) as exc:
                parser.error(f'{config.robot_config}: invalid/missing bounds: {exc}')
            config.bounds = bounds
    if (configs[0].nuc_ip, configs[0].server_port) == (configs[1].nuc_ip, configs[1].server_port):
        parser.error('robot0 and robot1 must have distinct NUC server endpoints')
    if configs[0].device_path and configs[0].device_path == configs[1].device_path:
        parser.error('robot0 and robot1 must have distinct SpaceMouse devices')
    return configs


def run(configs):
    with ExitStack() as cleanup:
        devices = []
        for config in configs:
            device = single.open_spacemouse(config.device_number, config.device_path)
            cleanup.callback(device.close)
            devices.append(device)
        if devices[0].device.path == devices[1].device.path:
            raise ValueError('Both configs resolved to the same SpaceMouse')
        robots = []
        for i, (config, device) in enumerate(zip(configs, devices)):
            command = single.ServerInterface(ip_address=config.nuc_ip, port=config.server_port,
                                             launch=config.launch_controllers)
            cleanup.callback(command.server.close)
            # NUC may serialize state reads behind a blocking trajectory. Keep the
            # UI responsive and mark old state STALE, rather than aborting a reset.
            reader = zerorpc.Client(heartbeat=30, timeout=30)
            cleanup.callback(reader.close)
            reader.connect(f'tcp://{config.nuc_ip}:{config.server_port}')
            robot = Robot(i, config, device, command, reader)
            robot.read_state()
            robots.append(robot)
        # Neither robot sends teleop/motion commands until BOTH connections are ready.
        curses.wrapper(dashboard, robots)


def main(argv=None):
    configs = parse_args(argv)
    try:
        run(configs)
    except KeyboardInterrupt:
        print('Teleop ended; hold attempted.')
    except Exception as exc:
        print(f'Teleop stopped: {exc}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
