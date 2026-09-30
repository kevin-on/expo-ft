"""Real payload/transport and disposable GPU worker with injected fake robots."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import threading
import time


def main():
    os.environ['JAX_PLATFORMS']='cpu'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',required=True)
    p.add_argument('--base-params',required=True)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    from expo_ft.eval.checkpoint import load_base,read_file
    from expo_ft.eval.server import Session,receiver,prepare_output
    from expo_ft.distributed.buffer import send_packet,receive_packet
    spec=importlib.util.spec_from_file_location('transport_fixture',Path(__file__).parents[1]/'distributed/test_transport.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class LargePayloadTransport(module.TransportTest):
        def configure_transport(self):
            for config in self.configs:
                config.update(chunk_bytes=4*1024**2, socket_timeout=120)
    fixture=LargePayloadTransport();fixture.setUp()
    base=load_base(a.base_params);session=Session(base,a.output/'runs')
    stop=threading.Event();thread=threading.Thread(target=receiver,args=(fixture.channels[1],session,stop));thread.start()
    sender=fixture.channels[0];process=None
    try:
        session.toggle_receive()
        sender.send('eval-offer','real',{'timeout':600});sender.flush()
        assert sender.receive('eval-admission','real')['accepted'];sender.release('eval-admission','real')
        t=time.monotonic()
        with read_file(a.weights) as payload:
            sender.send_buffer('eval-weights','real',payload)
            assert sender.receive('eval-result','real',timeout=600)['accepted'];sender.release('eval-result','real')
        print('REAL_RAM_TRANSFER_AND_BASE_VERIFY_SECONDS',time.monotonic()-t,flush=True)
        assert session.persist()
        assert session.begin_eval()
        parent,child=socket.socketpair(socket.AF_UNIX,socket.SOCK_SEQPACKET)
        # Fake wrapper is injected before the worker imports it. No robot sockets,
        # camera/HID imports, controller calls or WS processes are involved.
        code='''
import numpy as np
import expo_ft.env.env_client as rpc
class Fake:
 def __init__(self,*a,**kw):self.n=0
 def reset_only(self):self.n=0
 def start_episode(self):return self.get_observation()
 def get_observation(self):
  d={k:np.full((180,320,3),127,np.uint8) for k in ('exterior_image_1_left','exterior_image_2_left','wrist_image_left')}
  d.update(cartesian_position=np.zeros(6,np.float32),gripper_position=np.zeros(1,np.float32));return d
 def step(self,a):self.n+=1;return a,'policy'
 def get_info_for_step(self):return self.n==1,True,1.,0.
 def close(self):pass
rpc.EnvClientWrapper=Fake
from expo_ft.eval.worker import main
main()
'''
        with (a.output/'worker.log').open('w') as log:
            process=subprocess.Popen([sys.executable,'-c',code,'--control-fd',str(child.fileno()),
                '--base-fd',str(base.fd),'--weights-fd',str(session.payload.fd)],
                env=dict(os.environ,JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false'),
                pass_fds=(child.fileno(),base.fd,session.payload.fd),stdout=log,stderr=log)
            child.close()
            out,options=prepare_output(session,dict(task='configs/task/pick.py',seed=42,robots=[0,1],episodes=1,
                robot_config_dir='configs/robots',client_video_dir='/unused/mock/videos',host='127.0.0.1',base_port=8202))
            send_packet(parent,options)
            sender.send('eval-offer','busy',{'timeout':600});sender.flush()
            assert not sender.receive('eval-admission','busy')['accepted'];sender.release('eval-admission','busy')
            started=set();deadline=time.monotonic()+600
            while process.poll() is None:
                if time.monotonic()>deadline:raise TimeoutError('Worker lifecycle')
                if not select.select([parent],[],[],.2)[0]:continue
                try:state,_=receive_packet(parent)
                except ValueError:break
                if 'error' in state:raise RuntimeError(state['error'])
                robots=state.get('robots',{})
                for r in (0,1):
                    if str(r) in robots and robots[str(r)]['status']=='ready' and r not in started:
                        if r==1 and robots['0']['episodes']<1:continue
                        send_packet(parent,dict(command='start',robots=[r]));started.add(r)
            assert process.wait(timeout=30)==0
        parent.close();session.end_eval()
        assert session.persist() and session.saved=='Saved'
        assert session.offer('next',10);session.cancel('next','test completed')
        episodes=[json.loads(line) for line in (out/'episodes.jsonl').read_text().splitlines()]
        assert len(episodes)==2 and {row['robot'] for row in episodes}=={0,1}
        assert not (out/'summary.json').exists()
        print('GPU_WORKER_EXIT_SAVE_AND_RECEIVE_MODE_OK',episodes,flush=True)
    finally:
        if process is not None and process.poll() is None:process.terminate();process.wait(timeout=30)
        stop.set();thread.join();session.close();base.close();fixture.tearDown()


if __name__=='__main__':main()
