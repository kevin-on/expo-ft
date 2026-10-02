"""Interactive one/two-robot evaluation with one checkpoint-owned SFT/EXPO model."""
import argparse
from contextlib import contextmanager
import importlib.util
import json
import logging
from pathlib import Path
import queue
import select
import sys
import termios
import tty

from expo_ft.env.robot_eval import RobotEvaluation


@contextmanager
def keyboard():
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield fd
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def display(session, episodes):
    lines = ['Policy evaluation | Space: start when ALL ready | r/t: reset READY robot0/1 | q: quit']
    for robot, state in session.snapshot().items():
        last = '-' if state['last'] is None else ('SUCCESS' if state['last'] else 'FAIL')
        lines.append(f"robot{robot}: {state['status']:<18} step {state['steps']:3}/{session.max_steps} "
                     f"episodes {state['episodes']}/{episodes} success {state['successes']}/{state['episodes']} last {last}")
    sys.stdout.write('\033[H\033[J'+'\n'.join(lines)+'\n');sys.stdout.flush()


def drive(session, episodes, stream, fd):
    session.prepare()
    number = 0
    def drain():
        while True:
            try: row = session.results.get_nowait()
            except queue.Empty: break
            stream.write(json.dumps(row)+'\n');stream.flush()
    try:
        while True:
            ready = session.poll()
            drain()
            display(session, episodes)
            if ready and number >= episodes:
                return
            readable,_,_ = select.select([fd],[],[],0.1)
            if not readable:
                continue
            import os
            key = os.read(fd,4096)
            if not key or any(k in key for k in (b'q',b'Q',b'\x03')):
                return
            if b'r' in key or b't' in key:
                session.reset_ready(0 if b'r' in key else 1)
            elif b' ' in key and ready and number < episodes and session.start(number+1):
                number += 1
    finally:
        session.close()
        drain()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--policy-kind',choices=['sft','online'],default='sft')
    p.add_argument('--initial-sft-checkpoint',type=Path)
    p.add_argument('--initial-sft-base',type=Path)
    p.add_argument('--robots',type=int,nargs='+',choices=[0,1],default=[0,1])
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--replan-steps',type=int)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--host',default='0.0.0.0')
    p.add_argument('--base-port',type=int,default=8202)
    p.add_argument('--task',default='configs/task/pick.py')
    p.add_argument('--robot-config-dir',type=Path,default=Path('configs/robots'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--client-video-dir',type=Path,required=True)
    a=p.parse_args()
    if not sys.stdin.isatty(): p.error('Use an interactive terminal (SSH: -tt)')
    if len(set(a.robots)) != len(a.robots) or a.episodes < 1: p.error('Distinct robots and positive episode count required')
    if not a.client_video_dir.is_absolute(): p.error('client-video-dir is an absolute WS path')
    spec=importlib.util.spec_from_file_location('eval_task',a.task)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    task=module.get_config()
    if (task.action_space,task.gripper_action_space) != ('cartesian_velocity','velocity'):
        p.error('Requires Cartesian velocity and gripper velocity')
    from expo_ft.env.checkpoint_policy import SFTPolicy, OnlinePolicy
    print('Loading checkpoint configuration and shared model...',flush=True)
    if a.policy_kind=='online':
        policy=OnlinePolicy(a.checkpoint_dir,task=task,initial_sft_checkpoint=a.initial_sft_checkpoint,
                            initial_sft_base=a.initial_sft_base,seed=a.seed)
        replan=policy.record['replan_steps']
        if a.replan_steps is not None and a.replan_steps != replan: p.error('replan-steps differs from checkpoint')
    else:
        policy=SFTPolicy(a.checkpoint_dir,seed=a.seed,prompt=task.language_instruction)
        replan=8 if a.replan_steps is None else a.replan_steps
    if not 0 < replan <= policy.config.model.action_horizon: p.error('Invalid replan length')
    expected = {k:getattr(task,k) for k in ('auto_reset_steps','control_hz','action_space','gripper_action_space','language_instruction')}
    requests={}
    for robot in a.robots:
        config=json.loads((a.robot_config_dir/f'robot-{robot}.json').read_text())
        for k,v in expected.items():
            if config.get(k,v) != v: p.error(f'Robot {robot} overrides task {k}; use matching task on both machines')
        requests[robot]=dict(env_usage='eval',coordinated_eval=True,async_video=True,
            video_dir=str(a.client_video_dir/f'robot{robot}'),expected_task_settings=expected,
            expected_camera_views={k:config[k] for k in ('side_camera_id','wrist_camera_id')})
    a.output_dir.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(filename=a.output_dir/'eval.log',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    (a.output_dir/'eval_config.json').write_text(json.dumps(dict(checkpoint=str(a.checkpoint_dir.resolve()),
        policy_kind=a.policy_kind,robots=a.robots,episodes=a.episodes,replan_steps=replan,seed=a.seed,
        task=expected,requests=requests,omit_image_keys=list(policy.config.model.omit_image_keys)),indent=2)+'\n')
    from expo_ft.env.env_client import EnvClientWrapper
    envs={}
    try:
        for robot,request in requests.items():
            envs[robot]=EnvClientWrapper(request,host=a.host,port=a.base_port+robot,recover=False,lazy=True)
        session=RobotEvaluation(envs,lambda obs:policy.sample_actions(obs,only_base_actions=a.policy_kind=='sft')[0],
            replan_steps=replan,control_hz=task.control_hz,max_steps=task.auto_reset_steps)
        with keyboard() as fd,(a.output_dir/'episodes.jsonl').open('x') as stream:
            drive(session,a.episodes,stream,fd)
    except KeyboardInterrupt:
        print('\nEvaluation interrupted.')
    finally:
        for env in envs.values():env.close()


if __name__=='__main__':main()
