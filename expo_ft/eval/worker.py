"""One GPU lifetime per eval. Input weights arrive via inherited sealed RAM fds."""
import argparse
import importlib.util
import json
import logging
import os
from pathlib import Path
import select
import socket
import time

from expo_ft.distributed.buffer import Buffer, send_packet, receive_packet


def task_config(path):
    spec = importlib.util.spec_from_file_location('remote_eval_task', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    task = module.get_config()
    if (task.action_space, task.gripper_action_space) != ('cartesian_velocity', 'velocity'):
        raise ValueError('Expected Cartesian/gripper velocity task')
    return task


def run(sock, base, payload, options):
    from .checkpoint import manifest
    from .model import build
    from expo_ft.env.robot_eval import RobotEvaluation
    from expo_ft.env.env_client import EnvClientWrapper
    import numpy as np
    import queue
    task = task_config(options['task'])
    meta = manifest(payload)['metadata']
    output = Path(options['output'])
    # stdout/stderr are already redirected to this eval's logs/eval.log.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    expected = {k:getattr(task,k) for k in ('auto_reset_steps','control_hz','action_space','gripper_action_space','language_instruction')}
    requests = {}
    config_dir=output/'config'
    (config_dir/'robots').mkdir(parents=True,exist_ok=True)
    for robot in options['robots']:
        cfg_text = (Path(options['robot_config_dir'])/f'robot-{robot}.json').read_text()
        cfg = json.loads(cfg_text)
        if any(cfg.get(k,v)!=v for k,v in expected.items()): raise ValueError('Robot task differs')
        (config_dir/'robots'/f'robot-{robot}.json').write_text(cfg_text)
        requests[robot] = dict(env_usage='eval', coordinated_eval=True, async_video=True,
            video_dir=str(Path(options['client_video_dir'])/f'robot{robot}'),expected_task_settings=expected,
            expected_camera_views={k:cfg[k] for k in ('side_camera_id','wrist_camera_id')})
    (config_dir/'eval_config.json').write_text(json.dumps(dict(training_run_id=meta['training_run_id'],checkpoint_path=meta['checkpoint_path'],
        kind=meta['kind'],base_hash=meta['base_hash'],options=options,requests=requests),indent=2)+'\n')
    send_packet(sock, {'phase': 'Loading model'})
    policy = build(base, payload, task, options['seed'], base_verified=True)
    send_packet(sock, {'phase': 'Warming up'})
    obs = {k: np.full((180, 320, 3), 127, np.uint8) for k in
           ('exterior_image_1_left','exterior_image_2_left','wrist_image_left')}
    obs.update(cartesian_position=np.zeros(6,np.float32), gripper_position=np.zeros(1,np.float32))
    policy.sample_actions(obs, only_base_actions=meta['kind']=='sft')
    # Warmup must not consume evaluation RNG.
    import jax
    if meta['kind']=='sft': policy.policy._rng = jax.random.key(options['seed'])
    else: policy.agent = policy.agent.replace(rng=jax.random.PRNGKey(options['seed']))
    envs = {}
    session = None
    rows = []
    try:
        for r,request in requests.items():
            envs[r] = EnvClientWrapper(request,host=options['host'],port=options['base_port']+r,recover=False,lazy=True)
        session = RobotEvaluation(envs,lambda obs: policy.sample_actions(obs,only_base_actions=meta['kind']=='sft')[0],
            replan_steps=meta['replan_steps'],control_hz=task.control_hz,max_steps=task.auto_reset_steps)
        with (output/'episodes.jsonl').open('x') as stream:
            def drain():
                while True:
                    try: row = session.results.get_nowait()
                    except queue.Empty: break
                    rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            session.prepare()
            try:
                while True:
                    ready = session.poll(); drain()
                    states = session.snapshot()
                    send_packet(sock,dict(phase='Eval',robots=states,max_steps=session.max_steps))
                    if ready and all(s['episodes']>=options['episodes'] for s in states.values()): break
                    if not select.select([sock],[],[],.1)[0]: continue
                    message,_ = receive_packet(sock)
                    if message['command']=='stop': break
                    if message['command']=='reset':
                        session.reset_ready(message['robot'])
                    if message['command']=='start':
                        selected = message.get('robots',list(states))
                        if all(r in states and states[r]['episodes']<options['episodes'] for r in selected):
                            session.start({r:states[r]['episodes']+1 for r in selected}, selected)
            finally:
                session.close();drain()
    finally:
        if session is not None: session.close()
        for env in envs.values(): env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('control-fd','base-fd','weights-fd'): p.add_argument('--'+k,type=int,required=True)
    a=p.parse_args()
    with socket.socket(fileno=a.control_fd) as sock, Buffer.from_fd(a.base_fd) as base, Buffer.from_fd(a.weights_fd) as payload:
        options,_=receive_packet(sock)
        try: run(sock,base,payload,options)
        except BaseException as exc:
            send_packet(sock,dict(error=f'{type(exc).__name__}: {exc}'))
            raise


if __name__=='__main__': main()
