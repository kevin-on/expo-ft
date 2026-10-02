"""No hardware: gates, mirror transforms, policy serialization and worker cleanup."""
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
from expo_ft.env.robot_eval import RobotEvaluation
from expo_ft.env.rollout_rate import RolloutRate
from expo_ft.env.sft_eval import validate_eval_task


def observation():
    image=np.arange(18,dtype=np.uint8).reshape(2,3,3)
    return dict(cartesian_position=np.arange(6,dtype=np.float32),gripper_position=np.array([.4]),
        exterior_image_1_left=image,exterior_image_2_left=image,wrist_image_left=image)


class Env:
    def __init__(self, done_at=2):
        self.resets=0;self.starts=0;self.steps=[];self.done_at=done_at;self.n=0;self.closed=False
    def reset_only(self): self.resets+=1;self.n=0
    def start_episode(self): self.starts+=1;return observation()
    def step(self, action): self.steps.append(np.array(action));self.n+=1;return action, 'human' if self.n==1 else 'policy'
    def get_observation(self): return observation()
    def get_info_for_step(self): return self.n>=self.done_at, True, 1., 0.
    def close(self): self.closed=True


def wait_for(test):
    deadline=time.monotonic()+3
    while not test():
        if time.monotonic()>deadline: raise TimeoutError('worker test timed out')
        time.sleep(.002)


class EvalTest(unittest.TestCase):
    def test_rollout_hz_tracks_recent_intervals_stalls_and_episode_reset(self):
        rate = RolloutRate()
        self.assertIsNone(rate.hz(0))
        rate.step(0)
        self.assertIsNone(rate.hz(0))
        for tick in range(1, 11):
            rate.step(tick / 10)
        self.assertAlmostEqual(rate.hz(1), 10)
        self.assertAlmostEqual(rate.hz(2), 5)  # A stalled step lowers the live rate.
        for tick in range(1, 11):
            rate.step(1 + tick / 5)
        self.assertAlmostEqual(rate.hz(3), 5)  # Old 10Hz steps fall out of window.
        rate.reset()
        rate.step(100)
        self.assertIsNone(rate.hz(100))  # Manual waiting isn't part of the next episode.

    def test_eval_hz_is_per_robot_and_hidden_when_not_running(self):
        session = RobotEvaluation({0:Env(), 1:Env()}, lambda _:np.zeros((2,7)),
            replan_steps=2,control_hz=10,max_steps=2)
        try:
            with patch('expo_ft.env.rollout_rate.time.monotonic', return_value=0):
                session.state(0, status='starting')
                session.state(0, status='running', steps=1)
            with patch('expo_ft.env.rollout_rate.time.monotonic', return_value=.2):
                session.state(0, status='running', steps=2)
                self.assertAlmostEqual(session.snapshot()[0]['rollout_hz'], 5)
                self.assertIsNone(session.snapshot()[1]['rollout_hz'])
                session.state(0, status='resetting')
                self.assertIsNone(session.snapshot()[0]['rollout_hz'])
                session.state(0, status='starting')
                session.state(0, status='running', steps=1)
                self.assertIsNone(session.snapshot()[0]['rollout_hz'])
        finally:
            session.close()

    def test_observation_breakdown_is_attached_to_the_matching_step(self):
        class TimedEnv(Env):
            def get_observation_timing(self):
                return {'rpc_ms': float(self.n), 'ws': {'processing_ms': self.n / 2}}
        env = TimedEnv()
        session = RobotEvaluation({0:env}, lambda obs:np.zeros((2,7)),
            replan_steps=2,control_hz=1000,max_steps=2)
        try:
            session.prepare(); wait_for(session.poll)
            session.start(1); wait_for(session.poll)
            row = session.results.get_nowait()
            self.assertEqual([t['observation_breakdown']['rpc_ms'] for t in row['timings']], [1., 2.])
        finally:
            session.close()

    def test_manual_reset_only_ready_robot_and_no_extra_episode(self):
        entered, release = threading.Event(), threading.Event()
        class SlowReset(Env):
            def reset_only(self):
                super().reset_only()
                if self.resets == 2:
                    entered.set();release.wait(3)
        envs={0:SlowReset(1),1:Env(1)}
        session=RobotEvaluation(envs,lambda obs:np.zeros((1,7)),replan_steps=1,control_hz=1000,max_steps=1)
        try:
            self.assertFalse(session.reset_ready(0))
            session.prepare();wait_for(session.poll)
            self.assertFalse(session.reset_ready(2))
            self.assertTrue(session.reset_ready(0));self.assertTrue(entered.wait(2))
            self.assertFalse(session.reset_ready(0));self.assertFalse(session.start(1,[0]))
            self.assertEqual(session.snapshot()[0]['status'],'resetting')
            # Resetting robot 0 must not block robot 1's rollout.
            self.assertTrue(session.start(1,[1]));wait_for(lambda:session.by_robot[1].done())
            self.assertEqual(session.results.get_nowait()['robot'],1)
            release.set();wait_for(session.poll)
            self.assertEqual(envs[0].resets,2);self.assertEqual(envs[0].starts,0)
            self.assertFalse(envs[0].steps);self.assertTrue(session.results.empty())
            self.assertEqual(session.snapshot()[0]['episodes'],0)
            self.assertTrue(session.start(1,[0]));wait_for(session.poll)
            self.assertEqual(session.results.get_nowait()['episode'],1)
        finally:release.set();session.close()

    def test_manual_reset_error_is_reported_without_rollout(self):
        class FailedReset(Env):
            def reset_only(self):
                super().reset_only()
                if self.resets > 1: raise RuntimeError('reset RPC failed')
        env=FailedReset()
        session=RobotEvaluation({1:env},lambda obs:np.zeros((1,7)),replan_steps=1,control_hz=1000,max_steps=1)
        try:
            session.prepare();wait_for(session.poll)
            self.assertTrue(session.reset_ready(1))
            wait_for(lambda:session.by_robot[1].done())
            with self.assertRaisesRegex(RuntimeError,'reset RPC failed'):session.poll()
            self.assertEqual(session.snapshot()[1]['status'],'error')
            self.assertTrue(session.results.empty());self.assertEqual(env.starts,0)
        finally:session.close()

    def test_start_gate_mirror_shared_policy_and_results(self):
        envs={0:Env(),1:Env()};observations=[];active=0;peak=0
        def predict(obs):
            nonlocal active,peak
            active+=1;peak=max(peak,active);observations.append(obs)
            time.sleep(.003);active-=1
            return np.ones((2,7))*.25
        session=RobotEvaluation(envs,predict,replan_steps=2,control_hz=1000,max_steps=2)
        try:
            session.prepare();wait_for(session.poll)
            self.assertTrue(all(e.starts==0 and not e.steps for e in envs.values()))
            self.assertTrue(session.start(1));self.assertFalse(session.start(2))
            wait_for(session.poll)
            self.assertEqual(peak,1)
            self.assertTrue(all(e.resets==2 and e.starts==1 for e in envs.values()))
            np.testing.assert_array_equal(envs[0].steps[0],np.ones(7)*.25)
            np.testing.assert_array_equal(envs[1].steps[0],np.array([1,-1,1,-1,1,-1,1])*.25)
            mirrored=[o for o in observations if o['cartesian_position'][1]<0][0]
            np.testing.assert_array_equal(mirrored['wrist_image_left'],observation()['wrist_image_left'][:,::-1])
            rows=[session.results.get_nowait() for _ in envs]
            for row in rows:
                self.assertEqual(row['human_steps'],1);self.assertEqual(row['intervention_rate'],.5)
                self.assertTrue(row['had_intervention']);self.assertTrue(row['success']);self.assertFalse(row['success_without_intervention'])
        finally:session.close()
        self.assertTrue(all(e.closed for e in envs.values()))

    def test_start_one_robot_leaves_other_ready(self):
        envs={0:Env(1),1:Env(1)}
        session=RobotEvaluation(envs,lambda obs:np.zeros((1,7)),replan_steps=1,control_hz=1000,max_steps=1)
        try:
            session.prepare();wait_for(session.poll)
            self.assertTrue(session.start({0:1},[0]));wait_for(session.poll)
            self.assertEqual(envs[0].starts,1);self.assertEqual(envs[1].starts,0)
            self.assertTrue(session.start({1:1},[1]));wait_for(session.poll)
            self.assertEqual(envs[0].starts,1);self.assertEqual(envs[1].starts,1)
        finally:session.close()

    def test_first_actions_wait_for_both_plans(self):
        envs={0:Env(1),1:Env(1)};release=threading.Event();second=threading.Event();calls=0
        def predict(obs):
            nonlocal calls
            calls+=1
            if calls==2:second.set();release.wait(2)
            return np.zeros((1,7))
        session=RobotEvaluation(envs,predict,replan_steps=1,control_hz=1000,max_steps=1)
        try:
            session.prepare();wait_for(session.poll);session.start(1)
            self.assertTrue(second.wait(2));self.assertTrue(all(not e.steps for e in envs.values()))
            release.set();wait_for(session.poll)
        finally:release.set();session.close()

    def test_independent_reset_next_round_gate(self):
        entered=threading.Event();release=threading.Event()
        class Slow(Env):
            def step(self,action):entered.set();release.wait(2);return super().step(action)
        envs={0:Env(1),1:Slow(1)}
        session=RobotEvaluation(envs,lambda obs:np.zeros((1,7)),replan_steps=1,control_hz=1000,max_steps=1)
        try:
            session.prepare();wait_for(session.poll);session.start(1);self.assertTrue(entered.wait(2))
            wait_for(lambda:envs[0].resets==2)
            self.assertFalse(session.start(2));release.set();wait_for(session.poll)
        finally:release.set();session.close()

    def test_failed_plan_aborts_first_action_barrier(self):
        envs={0:Env(),1:Env()};calls=0
        def predict(obs):
            nonlocal calls
            calls+=1
            if calls==2:raise ValueError('broken policy')
            return np.zeros((2,7))
        session=RobotEvaluation(envs,predict,replan_steps=2,control_hz=10,max_steps=2)
        try:
            session.prepare();wait_for(session.poll);session.start(1)
            wait_for(lambda:all(f.done() for f in session.pending))
            with self.assertRaises(Exception):session.poll()
            self.assertTrue(all(not e.steps for e in envs.values()))
        finally:session.close()

    def test_client_contract_and_terminal_limit(self):
        with self.assertRaisesRegex(ValueError,'control_hz'):
            validate_eval_task({'control_hz':5},{'control_hz':10})
        session=RobotEvaluation({0:Env(3)},lambda o:np.zeros((1,7)),replan_steps=1,control_hz=1000,max_steps=1)
        try:
            session.prepare();wait_for(session.poll);session.start(1)
            wait_for(lambda:session.pending[0].done())
            with self.assertRaisesRegex(RuntimeError,'did not terminate'):session.poll()
            self.assertTrue(session.results.empty())
        finally:session.close()


if __name__=='__main__':unittest.main()
