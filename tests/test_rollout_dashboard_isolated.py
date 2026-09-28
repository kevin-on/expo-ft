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

from test_split_reset_overlap_isolated import ROOT, load_definitions

Dashboard = runpy.run_path(str(ROOT / 'expo_ft/utils/rollout_dashboard.py'))['RolloutDashboard']
namespace = dict(deque=deque, Future=Future, ThreadPoolExecutor=ThreadPoolExecutor,
                 deepcopy=deepcopy, queue=queue, threading=threading, time=time,
                 np=NS(asarray=lambda x: x))
load_definitions('expo_ft/utils/robot_round.py', ['collect_round'], namespace)
collect_round = namespace['collect_round']


class DashboardTests(unittest.TestCase):
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

    def run_collector(self, abort=False):
        ui = Dashboard(2, 80)
        ui.ready(0, 0)
        frames = [threading.Event(), threading.Event()]
        done = [threading.Event(), threading.Event()]
        closed = []

        class Env:
            def __init__(self, robot):
                self.robot = robot
            def start_episode(self):
                frames[self.robot].set()
                return {'robot': self.robot}
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
            result = pool.submit(collect_round, [Env(0), Env(1)], lambda obs: [[1.]], 1, 10000,
                reset_done=True, wait_for_start=ui.wait_for_start, check_session=ui.check,
                on_transition=lambda r, s, record: ui.step(r, s + 1, record['is_hil']), on_episode_end=end)
            try:
                self.assertFalse(frames[0].wait(.03))
                self.assertFalse(frames[1].is_set())
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


if __name__ == '__main__':
    unittest.main()
