"""No hardware: gates, mirror transforms, policy serialization and worker cleanup."""
import threading
import time
import unittest
import numpy as np
from expo_ft.env.robot_eval import RobotEvaluation
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
