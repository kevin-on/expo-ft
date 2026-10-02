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
    def get_action_timing(self):
        return action_frame_timing(self.metadata[-1],1100+100*self.n,'policy')
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
        rows={0:[],1:[]}
        def completed(r,s,record,t):
            self.assertEqual(t['step'],s+1)
            self.assertEqual(record['dones'],s==4)
            self.assertNotIn('action_frame_timing',record)
            rows[r].append(t)
        with self.assertLogs(level='INFO') as logs:
            result=collect_round(envs,lambda o:np.zeros((3,7)),3,100000,reset_done=True,canonical_frame=True,
                on_transition=completed, round_id=7)
        for r,env in enumerate(envs):
            self.assertEqual([m['frame_received_ms']['side'] for m in env.metadata],[1000,1000,1000,1003,1003])
            self.assertEqual([t['action_frame_timing']['frame_age_ms']['side'] for t in rows[r]],
                             [200,300,400,497,597])
            self.assertFalse(any('action_frame_timing' in record for record in result[r][0]))
        self.assertEqual(len(logs.records),10)
        self.assertTrue(all('"round":7' in row.getMessage() for row in logs.records))

    def test_eval_chunk_keeps_original_observation(self):
        import threading
        env=Env();session=RobotEvaluation({1:env},lambda o:np.zeros((3,7)),replan_steps=3,control_hz=100000,max_steps=5)
        try:
            with self.assertLogs(level='INFO') as logs:
                row=session.episode(1,1,threading.Barrier(1))
        finally:session.close()
        self.assertEqual([m['frame_received_ms']['side'] for m in env.metadata],[1000,1000,1000,1003,1003])
        self.assertEqual([t['action_frame_timing']['frame_age_ms']['side'] for t in row['timings']],
                         [200,300,400,497,597])
        self.assertEqual(len(logs.records),5)
        self.assertTrue(all('"episode":1' in entry.getMessage() for entry in logs.records))

    def test_online_human_and_handoff_do_not_claim_policy_frame_age(self):
        class HumanEnv(Env):
            def step(self,a):
                self.n+=1
                self.source='human' if self.n in (2,3) else 'policy'
                return a,self.source
            def get_action_timing(self):
                return action_frame_timing(self.metadata[-1],1100+100*self.n,self.source)
        rows=[]
        episodes=collect_round([HumanEnv()],lambda o:np.zeros((3,7)),3,100000,
            reset_done=True,mark_handoff=True,on_transition=lambda r,s,record,t:rows.append(t))
        self.assertTrue(episodes[0][0][3]['is_handoff'])
        for t in rows[1:4]:
            self.assertEqual(t['action_frame_timing']['frame_age_ms'],{})
            self.assertIsNone(t['policy_observation_age_ms'])
        self.assertEqual(rows[4]['action_frame_timing']['frame_received_ms']['side'],1004)
