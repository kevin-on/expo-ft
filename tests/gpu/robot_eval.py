"""One two-robot round using a real policy and recorded data; no robot/WS sockets."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--kind',choices=['sft','online'],required=True)
    p.add_argument('--demo',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import jax
    from configs.task.pick import get_config
    from expo_ft.env.checkpoint_policy import SFTPolicy,OnlinePolicy
    from expo_ft.env.droid_utils import process_droid_dataset
    from expo_ft.env.robot_eval import RobotEvaluation
    from expo_ft.env.sft_eval import canonical_observation
    assert jax.default_backend()=='gpu'
    task=get_config();obs=process_droid_dataset(a.demo,task,num_data=1)[0]['observations']
    policy=SFTPolicy(a.checkpoint,prompt=task.language_instruction) if a.kind=='sft' else OnlinePolicy(a.checkpoint,task=task)
    class RecordedEnv:
        def __init__(self,robot):self.robot=robot;self.steps=0;self.resets=0;self.closed=False
        def reset_only(self):self.steps=0;self.resets+=1
        def start_episode(self):return self.get_observation()
        def get_observation(self):return canonical_observation(copy.deepcopy(obs),mirror=self.robot==1)
        def step(self,action):
            assert action.shape==(7,) and np.isfinite(action).all()
            self.steps+=1
            return action,'policy'
        def get_info_for_step(self):return self.steps==2,True,1.,0.
        def close(self):self.closed=True
    envs={r:RecordedEnv(r) for r in (0,1)}
    session=RobotEvaluation(envs,lambda obs:policy.sample_actions(obs,only_base_actions=a.kind=='sft')[0],
        replan_steps=8,control_hz=10,max_steps=2)
    deadline=time.monotonic()+300
    def ready():
        while not session.poll():
            if time.monotonic()>deadline:raise TimeoutError('coordinator')
            time.sleep(.01)
    try:
        session.prepare();ready()
        assert all(e.steps==0 for e in envs.values())
        assert session.start(1);ready()
        rows=[session.results.get_nowait() for _ in envs]
        assert all(row['steps']==2 and row['success_without_intervention'] for row in rows)
        assert all(e.resets==2 for e in envs.values())
        if a.kind=='online':assert int(policy.agent.actor_train_state.step)==1
        a.output.write_text(json.dumps(dict(passed=True,kind=a.kind,rows=rows),indent=2))
        print('REAL_MODEL_TWO_ROBOT_EVAL_OK',a.kind,flush=True)
    finally:session.close()
    assert all(e.closed for e in envs.values())


if __name__=='__main__':main()
