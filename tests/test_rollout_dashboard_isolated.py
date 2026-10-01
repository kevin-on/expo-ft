"""Small start-gate/control tests with in-memory robots; no SDK/JAX imports."""
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
import queue
import runpy
import threading
import time
from types import SimpleNamespace as NS
import unittest
from contextlib import redirect_stdout
import io

from test_split_reset_overlap_isolated import ROOT, load_definitions

Dashboard = runpy.run_path(str(ROOT / 'expo_ft/utils/rollout_dashboard.py'))['RolloutDashboard']
Progress = runpy.run_path(str(ROOT / 'expo_ft/utils/rollout_dashboard.py'))['RolloutProgress']
namespace = dict(deque=deque, Future=Future, ThreadPoolExecutor=ThreadPoolExecutor,
                 deepcopy=deepcopy, queue=queue, threading=threading, time=time,
                 np=NS(asarray=lambda x: x))
load_definitions('expo_ft/utils/robot_round.py', ['collect_round'], namespace)
collect_round = namespace['collect_round']


class DashboardTests(unittest.TestCase):
    def test_resume_totals_continue_without_double_counting_or_counting_handoff(self):
        for robots in (1, 2):
            progress = Progress(robots)
            for robot in range(robots):
                for success in (True, False, True):
                    progress.record(robot, dict(dones=False))
                    progress.record(robot, dict(dones=True, is_success=success))
            restored = Progress(robots, progress.snapshot())
            ui = Dashboard(robots, 80, mode='auto')
            ui.restore_progress(restored.snapshot(), 6 * robots)
            ui.ready(3, 6 * robots)
            for robot in range(robots):
                self.assertTrue(ui.wait_for_start(robot, threading.Event()))
                ui.step(robot, 1, True)
                ui.step(robot, 2, False, trainable=False)
                ui.step(robot, 3, False)
                ui.episode_done(robot, True)
                self.assertEqual(ui.states[robot]['completed'], 4)
                self.assertEqual(ui.states[robot]['successes'], 3)
                self.assertEqual(ui.states[robot]['transitions'], 8)
                self.assertEqual(ui.states[robot]['steps'], 3)
                restored.record(robot, dict(dones=False))
                restored.record(robot, dict(dones=True, is_success=True))
            self.assertEqual(ui.total_transitions, 8 * robots)
            ui.restore_progress(restored.snapshot(), 8 * robots)
            ui.ready(4, 8 * robots)
            self.assertEqual(ui.total_transitions, 8 * robots)
            self.assertEqual(ui.round, 5)
            text = io.StringIO()
            with redirect_stdout(text):
                ui.draw()
            self.assertIn('Total transitions: {}'.format(8 * robots), text.getvalue())
            self.assertTrue(all(s['completed'] == 4 and s['successes'] == 3 for s in ui.states))

    def test_single_robot_controls_ignore_robot1(self):
        ui = Dashboard(1, 80)
        ui.ready(0, 0)
        ui.handle_key(b'1'); ui.handle_key(b't')
        self.assertEqual(ui.states[0]['status'], 'ready')
        ui.handle_key(b'r')
        self.assertEqual(ui.states[0]['status'], 'resetting')
        calls = []
        def reset():
            calls.append(0)
            ui.handle_key(b' ')  # ignored while reset is in progress
        with ThreadPoolExecutor(max_workers=1) as pool:
            stopped = threading.Event()
            future = pool.submit(ui.wait_for_start, 0, stopped, reset)
            try:
                deadline = time.monotonic() + 2
                with ui.condition:
                    while ui.states[0]['status'] != 'ready':
                        self.assertLess(time.monotonic(), deadline)
                        ui.condition.wait(.02)
                self.assertEqual(calls, [0])
                self.assertFalse(future.done())
                ui.handle_key(b' ')
                self.assertTrue(future.result(timeout=1))
            finally:
                stopped.set()

    def test_reset_keys_require_manual_ready_and_do_not_queue(self):
        ui = Dashboard(2, 80, mode='auto')
        ui.handle_key(b'r');ui.ready(0, 0)
        ui.handle_key(b'r')
        self.assertEqual(ui.states[0]['status'], 'ready')
        ui.handle_key(b'm');ui.handle_key(b't')
        self.assertEqual(ui.states[1]['status'], 'resetting')
        ui.handle_key(b'1');ui.handle_key(b't')
        self.assertEqual(ui.states[1]['status'], 'resetting')
        ui.handle_key(b'0');ui.handle_key(b'r')
        self.assertEqual(ui.states[0]['status'], 'starting')

    def test_manual_reset_blocks_only_its_robot_and_waits_for_start(self):
        ui = Dashboard(2, 80)
        ui.ready(0, 0)
        entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
        calls = []
        def reset():
            calls.append(0);entered.set();release.wait(2)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(ui.wait_for_start, 0, stopped, reset)
            try:
                ui.handle_key(b'r');self.assertTrue(entered.wait(1))
                ui.handle_key(b'r');ui.handle_key(b'0')
                self.assertFalse(future.done())
                ui.handle_key(b'1');self.assertTrue(ui.wait_for_start(1, stopped))
                release.set()
                deadline = time.monotonic()+2
                with ui.condition:
                    while ui.states[0]['status'] != 'ready':
                        self.assertLess(time.monotonic(),deadline)
                        ui.condition.wait(.02)
                self.assertFalse(future.done());self.assertEqual(calls,[0])
                self.assertEqual(ui.states[0]['completed'],0)
                ui.handle_key(b'0');self.assertTrue(future.result(timeout=1))
            finally:release.set();stopped.set()

    def test_early_keys_ignored_and_mode_changes_do_not_restart_running_robot(self):
        ui = Dashboard(2, 80)
        ui.handle_key(b'0')
        ui.ready(0, 0)
        self.assertEqual([s['status'] for s in ui.states], ['ready', 'ready'])
        ui.handle_key(b'0')
        self.assertTrue(ui.wait_for_start(0, threading.Event()))
        ui.step(0, 5, True)
        ui.handle_key(b'm')  # auto releases the other READY robot
        self.assertTrue(ui.wait_for_start(1, threading.Event()))
        ui.handle_key(b'm')  # manual does not stop either started episode
        self.assertEqual(ui.states[0]['steps'], 5)
        self.assertEqual(ui.states[0]['status'], 'human')
        ui.episode_done(0, True)
        ui.episode_done(1, False)
        self.assertEqual(ui.states[0]['last'], 'success')
        self.assertEqual(ui.states[1]['successes'], 0)
        ui.handle_key(b'0')  # DONE cannot run twice in this round
        self.assertEqual(ui.states[0]['status'], 'done')
        ui.ready(1, 160)
        self.assertEqual([s['status'] for s in ui.states], ['ready', 'ready'])
        self.assertEqual(ui.states[0]['completed'], 1)

    def test_space_releases_both_ready_gates_and_repeated_keys_do_not_queue(self):
        ui = Dashboard(2, 80)
        ui.ready(0, 0)
        ui.handle_key(b' ')
        self.assertEqual([s['status'] for s in ui.states], ['starting', 'starting'])
        ui.handle_key(b' ')
        for robot in (0, 1):
            self.assertTrue(ui.wait_for_start(robot, threading.Event()))
            ui.episode_done(robot, True)
        ui.ready(1, 160)
        self.assertEqual([s['status'] for s in ui.states], ['ready', 'ready'])

    def run_collector(self, abort=False, reset_first=False, reset_failure=False):
        ui = Dashboard(2, 80)
        ui.ready(0, 0)
        frames = [threading.Event(), threading.Event()]
        done = [threading.Event(), threading.Event()]
        closed = []
        resets = []

        class Env:
            def __init__(self, robot):
                self.robot = robot
            def start_episode(self):
                frames[self.robot].set()
                return {'robot': self.robot}
            def reset_only(self):
                resets.append(self.robot)
                if reset_failure: raise RuntimeError('reset RPC failed')
            def step(self, action):
                return action, 'policy'
            def get_observation(self):
                return {}
            def get_info_for_step(self):
                return True, True, 1., 0.
            def close(self):
                closed.append(self.robot)

        def end(robot, length, success):
            ui.episode_done(robot, success)
            done[robot].set()

        with ThreadPoolExecutor(max_workers=1) as pool:
            envs = [Env(0), Env(1)]
            result = pool.submit(collect_round, envs, lambda obs: [[1.]], 1, 10000,
                reset_done=True, wait_for_start=lambda r, stop: ui.wait_for_start(r, stop, envs[r].reset_only), check_session=ui.check,
                on_transition=lambda r, s, record: ui.step(r, s + 1, record['is_hil']), on_episode_end=end)
            try:
                self.assertFalse(frames[0].wait(.03))
                self.assertFalse(frames[1].is_set())
                if reset_first:
                    ui.handle_key(b'r')
                    if reset_failure:
                        with self.assertRaisesRegex(RuntimeError, 'reset RPC failed'):result.result(timeout=2)
                        self.assertEqual(sorted(closed),[0,1])
                        self.assertFalse(any(event.is_set() for event in frames))
                        return
                    deadline=time.monotonic()+2
                    with ui.condition:
                        while ui.states[0]['status']!='ready':
                            self.assertLess(time.monotonic(),deadline);ui.condition.wait(.02)
                    self.assertEqual(resets,[0]);self.assertFalse(frames[0].is_set())
                if abort:
                    ui.handle_key(b'q')
                    with self.assertRaisesRegex(RuntimeError, 'stopped'):
                        result.result(timeout=2)
                    self.assertEqual(sorted(closed), [0, 1])
                    self.assertFalse(any(event.is_set() for event in frames))
                else:
                    ui.handle_key(b'1')
                    self.assertTrue(done[1].wait(1))
                    self.assertFalse(frames[0].is_set())
                    self.assertFalse(result.done())
                    ui.handle_key(b' ')
                    self.assertEqual(len(result.result(timeout=2)), 2)
                    self.assertTrue(done[0].is_set())
            finally:
                ui.handle_key(b'q')

    def test_one_robot_completes_while_other_waits_without_reading_a_frame(self):
        self.run_collector()

    def test_abort_unblocks_waiting_gates_without_motion(self):
        self.run_collector(abort=True)

    def test_reset_before_rollout_does_not_add_transitions_or_episodes(self):
        self.run_collector(reset_first=True)

    def test_reset_failure_aborts_round_without_starting_episodes(self):
        self.run_collector(reset_first=True,reset_failure=True)


if __name__ == '__main__':
    unittest.main()
