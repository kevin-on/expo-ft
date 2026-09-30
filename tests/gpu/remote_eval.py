"""Actual checkpoint -> eval envelope -> inference parity, with no robot devices.

Run export on CPU, reference and compact inference in separate GPU invocations.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['export','reference','compact'])
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--kind',required=True,choices=['sft','online'])
    p.add_argument('--base-params',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--demo',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=='export':
        from expo_ft.eval.checkpoint import export,save
        with export(a.checkpoint,a.kind,training_run_id='remote-validation',
                    checkpoint_path=f'{a.kind}/remote-validation/checkpoints/{a.checkpoint.name}') as payload:
            saved=save(payload,a.output)
            (a.output/'weights-path.txt').write_text(str(saved))
            print('EVAL_EXPORT',payload.size,str(saved),flush=True)
        return
    from configs.task.pick import get_config
    from expo_ft.env.droid_utils import process_droid_dataset
    task=get_config();obs=process_droid_dataset(a.demo,task,num_data=1)[0]['observations']
    if a.mode=='reference':
        from expo_ft.env.checkpoint_policy import SFTPolicy,OnlinePolicy
        policy=SFTPolicy(a.checkpoint,seed=42,prompt=task.language_instruction) if a.kind=='sft' else OnlinePolicy(a.checkpoint,task=task,seed=42)
    else:
        from expo_ft.eval.checkpoint import load_base,read_file
        from expo_ft.eval.model import build
        base=load_base(a.base_params)
        payload=read_file((a.output/'weights-path.txt').read_text())
        policy=build(base,payload,task,seed=42)
    if a.kind=='online':
        import jax
        policy.agent=policy.agent.replace(rng=jax.random.PRNGKey(42))
    action=policy.sample_actions(obs,only_base_actions=a.kind=='sft')[0]
    action=np.asarray(action)
    if a.mode=='reference':np.save(a.output/'reference.npy',action)
    else:
        expected=np.load(a.output/'reference.npy')
        np.testing.assert_allclose(action,expected,rtol=1e-4,atol=1e-5)
        print('REMOTE_EVAL_PARITY_OK',a.kind,'max_abs',float(np.max(np.abs(action-expected))),flush=True)
        # No hardware: exercise independent then simultaneous starts with actual policy.
        import copy,time
        from expo_ft.env.robot_eval import RobotEvaluation
        from expo_ft.env.sft_eval import canonical_observation
        class Env:
            def __init__(self,r):self.r=r;self.steps=0
            def reset_only(self):self.steps=0
            def start_episode(self):return self.get_observation()
            def get_observation(self):return canonical_observation(copy.deepcopy(obs),self.r==1)
            def step(self,a):self.steps+=1;return a,'policy'
            def get_info_for_step(self):return self.steps>=1,True,1.,0.
            def close(self):pass
        session=RobotEvaluation({r:Env(r) for r in (0,1)},lambda o:policy.sample_actions(o,only_base_actions=a.kind=='sft')[0],
            replan_steps=8,control_hz=10,max_steps=1)
        try:
            def ready():
                while not session.poll():time.sleep(.01)
            session.prepare();ready();assert session.start(1,[0]);ready();assert session.start(1,[1]);ready()
            assert session.start(2);ready()
            print('REMOTE_EVAL_ROBOT_MOCK_OK',flush=True)
        finally:session.close()


if __name__=='__main__':main()
