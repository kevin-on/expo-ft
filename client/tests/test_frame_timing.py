import unittest
import numpy as np
from client.real_utils.frame_timing import action_frame_timing
import runpy
from pathlib import Path
collect_round = runpy.run_path(str(Path(__file__).resolve().parents[2]/'expo_ft/utils/robot_round.py'))['collect_round']
from expo_ft.env.robot_eval import RobotEvaluation


def obs():
    a=np.zeros((2,2,3),np.uint8)
    return dict(exterior_image_1_left=a,exterior_image_2_left=a,wrist_image_left=a,
                cartesian_position=np.zeros(6),gripper_position=np.zeros(1))

class Env:
    def __init__(self):self.n=0;self.metadata=[]
    def start_episode(self):return obs()
    def reset_only(self):pass
    def get_observation_metadata(self):return {'frame_received_ms':{'side':1000+self.n,'wrist':1001+self.n}}
    def set_action_observation(self,meta):self.metadata.append(meta)
    def step(self,a):self.n+=1;return a,'policy'
    def get_observation(self):return obs()
    def get_info_for_step(self):return self.n>=5,True,1.,0.
    def close(self):pass

class TimingTests(unittest.TestCase):
    def test_host_timestamp_difference_and_human_not_policy_latency(self):
        meta={'frame_received_ms':{'side':1000.,'wrist':1005.},'buffer_sequence':42}
        self.assertEqual(action_frame_timing(meta,1100,'policy')['frame_age_ms'],{'side':100.,'wrist':95.})
        self.assertEqual(action_frame_timing(meta,1100,'human')['frame_age_ms'],{})
        self.assertEqual(action_frame_timing(None,1100,'policy')['frame_age_ms'],{})
        self.assertEqual(action_frame_timing(meta,None,'policy')['frame_age_ms'],{})

    def test_online_chunk_keeps_original_observation_for_both_robots(self):
        envs=[Env(),Env()]
        collect_round(envs,lambda o:np.zeros((3,7)),3,100000,reset_done=True,canonical_frame=True)
        for env in envs:
            self.assertEqual([m['frame_received_ms']['side'] for m in env.metadata],[1000,1000,1000,1003,1003])

    def test_eval_chunk_keeps_original_observation(self):
        import threading
        env=Env();session=RobotEvaluation({1:env},lambda o:np.zeros((3,7)),replan_steps=3,control_hz=100000,max_steps=5)
        try: session.episode(1,1,threading.Barrier(1))
        finally:session.close()
        self.assertEqual([m['frame_received_ms']['side'] for m in env.metadata],[1000,1000,1000,1003,1003])
