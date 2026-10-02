"""Persistent receive/save TUI; each eval owns a separate, disposable GPU process."""
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import threading
import time
import uuid

from expo_ft.distributed.buffer import send_packet, receive_packet
from expo_ft.distributed.channel import Channel
from .checkpoint import load_base, manifest, location, read_file, save, BaseIdentity, within, weights_path, WEIGHTS_NAME, LEGACY_WEIGHTS_NAME


class Session:
    def __init__(self, base, root, validator=None):
        self.base, self.root, self.validator = base, Path(root), validator or BaseIdentity()
        self.lock = threading.RLock()
        self.mode, self.payload, self.meta = 'receive', None, None
        self.saved, self.notice, self.request = 'RAM only', 'Waiting for checkpoint', None
        self.saved_path = None
        self.error = None
        self.progress = None
        self.receive_armed = False

    def load_saved(self, path):
        """Load the startup checkpoint; trust an existing canonical disk copy."""
        path = weights_path(path)
        with self.lock:
            if self.mode != 'receive' or self.payload is not None:
                raise RuntimeError('Local checkpoint loading requires an empty idle session')
            self.receive_armed = True
            request = uuid.uuid4().hex
            self.offer(request, 600)
        try:
            self.accept(request, read_file(path))
        except BaseException:
            self.cancel(request, 'Local checkpoint load failed')
            raise
        with self.lock:
            step = location(self.root, self.meta)
            targets = (step / WEIGHTS_NAME, step / LEGACY_WEIGHTS_NAME)
            self.saved = 'Saved' if any(Path(path).resolve() == target.resolve() for target in targets) else 'Loaded from disk'
            self.saved_path = str(path) if self.saved == 'Saved' else None
            self.notice = str(path)

    def toggle_receive(self):
        with self.lock:
            if self.mode != 'receive': return False
            self.receive_armed = not self.receive_armed
            self.notice = ('One transfer allowed; incoming checkpoint replaces current RAM checkpoint'
                           if self.receive_armed else 'Reception locked')
            return True

    def offer(self, request, timeout):
        with self.lock:
            if self.mode != 'receive' or not self.receive_armed: return False
            self.receive_armed = False
            self.mode, self.request = 'receiving', request
            self.deadline = time.monotonic()+min(max(float(timeout),1),3600)
            self.notice = 'Sender exporting / transferring checkpoint'
            self.progress = None
            return True

    def cancel(self, request, reason):
        with self.lock:
            if request == self.request:
                self.mode, self.request, self.notice = 'receive', None, reason

    def accept(self, request, payload):
        try:
            meta = manifest(payload)['metadata']
            self.validator(self.base, payload)
            with self.lock:
                if self.mode != 'receiving' or self.request != request: raise ValueError('Expired transfer')
                old = self.payload
                self.payload, self.meta = payload, meta
                self.saved, self.mode, self.request = 'RAM only', 'receive', None
                self.saved_path = None
                self.notice = 'Checkpoint ready'
            if old is not None: old.close()
        except BaseException:
            payload.close()
            raise

    def begin_eval(self):
        with self.lock:
            if self.mode != 'receive' or self.payload is None: return False
            if self.saved != 'Saved':
                self.notice = 'Save checkpoint with S before starting eval'
                return False
            self.receive_armed = False
            self.mode, self.notice = 'eval', 'Loading model'
            return True

    def end_eval(self):
        with self.lock:
            self.mode, self.notice = 'receive', 'GPU process exited; checkpoint retained in RAM'

    def persist(self):
        with self.lock:
            if self.mode != 'receive' or self.payload is None: return False
            self.receive_armed = False
            if self.saved == 'Saved':
                # The immutable RAM checkpoint was already persisted in this
                # session. accept() invalidates this state on every replacement.
                self.notice = self.saved_path
                return True
            self.mode, self.saved = 'saving', 'Saving'
        try:
            path = save(self.payload,self.root)
        except Exception as exc:
            with self.lock: self.saved, self.notice, self.mode = 'Save failed', str(exc), 'receive'
            return False
        with self.lock:
            self.saved, self.notice, self.mode = 'Saved', str(path), 'receive'
            self.saved_path = str(path)
        return True

    def snapshot(self):
        with self.lock:
            return dict(mode=self.mode,meta=self.meta,saved=self.saved,notice=self.notice,progress=self.progress,
                        receive_armed=self.receive_armed, saved_path=self.saved_path)

    def close(self):
        if self.payload is not None: self.payload.close();self.payload=None


def receiver(channel, session, stop):
    discarded = set()
    try:
        while not stop.wait(.05):
            offer = channel.poll('eval-offer')
            if offer:
                request, data = offer
                admitted = session.offer(request,data.get('timeout',600))
                reason = ('Reception locked: press R on receiver to allow one transfer'
                          if session.mode == 'receive' else 'Eval server busy: '+session.mode)
                channel.send('eval-admission',request,dict(accepted=admitted,reason='' if admitted else reason))
                channel.flush()
            cancel = channel.poll('eval-cancel')
            if cancel:
                discarded.add(cancel[0])
                session.cancel(cancel[0],'Sender cancelled transfer')
            # Discard any late completed transfer from an expired/cancelled offer.
            for stale in tuple(discarded):
                abandoned = channel.poll_buffer('eval-weights',stale)
                if abandoned is not None:
                    abandoned.close();channel.release('eval-weights',stale);discarded.remove(stale)
            with session.lock:
                request = session.request
                expired = request is not None and time.monotonic()>session.deadline
            if request is None: continue
            if expired:
                discarded.add(request)
                session.cancel(request,'Transfer timed out; previous checkpoint retained')
                channel.send('eval-result',request,dict(accepted=False,reason='Transfer timed out'));continue
            buffer = channel.poll_buffer('eval-weights',request)
            if buffer is None:
                with session.lock: session.progress = channel.progress('eval-weights',request)
                continue
            try:
                session.accept(request,buffer)
                result=dict(accepted=True)
            except Exception as exc:
                session.cancel(request,str(exc));result=dict(accepted=False,reason=str(exc))
            finally: channel.release('eval-weights',request)
            channel.send('eval-result',request,result);channel.flush()
    except Exception as exc:
        session.error = str(exc)
        session.notice = 'Transport error: '+str(exc)


def prepare_output(session, options):
    stamp = datetime.now().astimezone().strftime('%Y%m%d-%H%M%S-%f')
    eval_id = session.meta['training_run_id'][:100]+'-'+stamp
    relative = 'eval/'+eval_id
    output = within(session.root,relative)
    output.mkdir(parents=True,exist_ok=False)
    for folder in ('config','logs','cube-records'):
        (output/folder).mkdir()
    record = dict(run_id=eval_id,training_run_id=session.meta['training_run_id'],
        checkpoint_path=session.meta['checkpoint_path'],config_path=relative+'/config/eval_config.json')
    (output/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    options = dict(options,output=str(output),client_video_dir=str(Path(options['client_video_dir'])/eval_id))
    return output, options


class EvalProcess:
    def __init__(self, session, options):
        if not session.begin_eval(): raise RuntimeError(session.notice if session.saved!='Saved' else 'Eval is not available')
        self.session, self.process, self.status, self.log, self.sock = session, None, {}, None, None
        child = None
        try:
            output, options = prepare_output(session,options)
            self.log=(output/'logs/eval.log').open('w')
            self.sock,child=socket.socketpair(socket.AF_UNIX,socket.SOCK_SEQPACKET)
            env=dict(os.environ,JAX_PLATFORMS=options.pop('platform'),XLA_PYTHON_CLIENT_PREALLOCATE='false')
            self.process=subprocess.Popen([sys.executable,'-m','expo_ft.eval.worker','--control-fd',str(child.fileno()),
                '--base-fd',str(session.base.fd),'--weights-fd',str(session.payload.fd)],
                env=env,pass_fds=(child.fileno(),session.base.fd,session.payload.fd),stdin=subprocess.DEVNULL,
                stdout=self.log,stderr=subprocess.STDOUT)
            send_packet(self.sock,options)
        except BaseException:
            if self.process is not None:
                self.process.terminate();self.process.wait()
            self.close();raise
        finally:
            if child is not None: child.close()

    def command(self, command, **kwargs):
        if self.process.poll() is None:
            try: send_packet(self.sock,dict(command=command,**kwargs))
            except (BrokenPipeError,OSError): pass

    def poll(self):
        while select.select([self.sock],[],[],0)[0]:
            try: self.status,_=receive_packet(self.sock)
            except (ValueError,OSError): break
        code=self.process.poll()
        if code is None: return False
        error=self.status.get('error') or (f'Eval process exited with code {code}; see eval.log' if code else None)
        self.close()
        if error: self.session.notice=error
        return True

    def close(self):
        if self.log: self.log.close();self.log=None
        if self.sock: self.sock.close();self.sock=None
        self.session.end_eval()


def lines(session, worker, episodes):
    s=session.snapshot();m=s['meta']
    result=[f"Remote Checkpoint Eval   |   {'EVAL' if worker else 'RECEIVE / SAVE'}",'']
    if m:
        result += [f"Run     {m['training_run_id']}",f"Checkpoint {m['checkpoint_path']}    Policy {m['kind']}",
                   f"Storage {s['saved']}",f"Path    {s['saved_path'] or location(session.root,m)/WEIGHTS_NAME}"]
    else: result += ['Checkpoint   None']
    reception = ('Transferring; further offers blocked' if s['mode']=='receiving' else
                 'Blocked during '+s['mode'] if s['mode']!='receive' else
                 'Ready for ONE transfer' if s['receive_armed'] else 'Locked')
    result += [f'Reception {reception}']
    result += ['',s['notice']]
    progress=s['progress']
    if s['mode']=='receiving' and progress and progress['size']:
        result += [f"Received {progress['done']/1e9:.2f} / {progress['size']/1e9:.2f} GB"]
    if worker:
        status=worker.status
        result += [status.get('phase','Loading model'), 'New checkpoint reception disabled',
                   f'Target: {episodes} episodes per robot | Success / completed (%)', '']
        for r,state in status.get('robots',{}).items():
            last='—' if state['last'] is None else 'SUCCESS' if state['last'] else 'FAIL'
            rate=f"{100*state['successes']/state['episodes']:.1f}%" if state['episodes'] else '—'
            hz = '—' if state.get('rollout_hz') is None else f"{state['rollout_hz']:.1f}"
            result.append(f"Robot {r}  {state['status']:<18}  {state['steps']:3}/{status['max_steps']} steps"
                          f"   {hz} Hz   {state['successes']}/{state['episodes']} ({rate})   last {last}")
        result += ['','[0/1] Start robot  [Space] Start both  [r/t] Reset READY robot 0/1  [Esc] End eval']
    else:
        gpu = 'Starting eval' if s['mode']=='eval' else 'GPU released'
        action = 'Eval' if s['mode']=='receive' and s['saved']=='Saved' else 'Eval (disabled)'
        receive_action = ('Receive (disabled)' if s['mode']!='receive' else
                          'Lock reception' if s['receive_armed'] else 'Allow one receive')
        result += ['',f"State   {s['mode']}    {gpu}",f'[E] {action}  [S] Save  [R] {receive_action}  [Q] Quit']
    return result


def main():
    os.environ['JAX_PLATFORMS']='cpu'  # Parent never creates a GPU context.
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mailbox',required=True)
    p.add_argument('--base-params',type=Path,required=True)
    p.add_argument('--experiments-root',type=Path,default=Path('/iliad/u/kevinon/experiments/expo-ft'))
    p.add_argument('--weights',type=Path,help='Saved trainable_weights.bin or checkpoint directory (legacy eval/weights.bin accepted)')
    p.add_argument('--robots',type=int,nargs='+',choices=[0,1],default=[0,1])
    p.add_argument('--episodes',type=int,default=30,help='Default for the episode-count prompt shown before each eval')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--host',default='0.0.0.0')
    p.add_argument('--base-port',type=int,default=8202)
    p.add_argument('--task',default='configs/task/pick.py')
    p.add_argument('--robot-config-dir',default='configs/robots')
    p.add_argument('--client-video-dir',required=True)
    p.add_argument('--platform',default='cuda',choices=['cuda','cpu'])
    a=p.parse_args()
    if not sys.stdin.isatty(): p.error('Interactive terminal required (SSH -tt)')
    if a.episodes<1 or len(set(a.robots))!=len(a.robots): p.error('Positive episodes and distinct robots required')
    if not Path(a.client_video_dir).is_absolute(): p.error('client-video-dir must be an absolute WS path')
    options={k:getattr(a,k) for k in ('robots','episodes','seed','host','base_port','task','robot_config_dir','client_video_dir','platform')}
    print('Loading local base into CPU RAM...',flush=True)
    base=load_base(a.base_params)
    session=Session(base,a.experiments_root)
    if a.weights:
        session.load_saved(a.weights)
    channel=Channel(a.mailbox)
    stop=threading.Event();thread=threading.Thread(target=receiver,args=(channel,session,stop),daemon=True)
    thread.start();writer=None;worker=None;launch=None;episode_input=None
    launcher=ThreadPoolExecutor(max_workers=1,thread_name_prefix='eval-start')
    from eval_sft_robots import keyboard
    try:
        with keyboard() as fd:
            while True:
                if launch is not None and launch.done():
                    try: worker=launch.result()
                    except Exception as exc: session.notice=str(exc)
                    launch=None
                if worker and worker.poll(): worker=None
                display=lines(session,worker,options['episodes'])
                if episode_input is not None:
                    display += ['',f'Episodes per robot [{options["episodes"]}]: {episode_input}',
                                '[Enter] Confirm  [Backspace] Edit  [Esc] Cancel']
                sys.stdout.write('\033[H\033[J'+'\n'.join(display)+'\n');sys.stdout.flush()
                if not select.select([fd],[],[],.1)[0]: continue
                key=os.read(fd,1)
                if episode_input is not None:
                    if not key or key==b'\x03': break
                    if key==b'\x1b':
                        episode_input=None
                        session.notice='Eval cancelled'
                    elif key in (b'\r',b'\n'):
                        episodes=int(episode_input) if episode_input else options['episodes']
                        if episodes<1:
                            session.notice='Enter a positive episode count'
                        else:
                            options=dict(options,episodes=episodes)
                            episode_input=None
                            launch=launcher.submit(EvalProcess,session,dict(options))
                    elif key in (b'\x7f',b'\x08'):
                        episode_input=episode_input[:-1]
                    elif b'0'<=key<=b'9' and len(episode_input)<9:
                        episode_input+=key.decode('ascii')
                    continue
                if worker:
                    if b'\x1b' in key or b'q' in key or b'\x03' in key: worker.command('stop')
                    elif b'r' in key or b't' in key:
                        robot = 0 if b'r' in key else 1
                        if robot in a.robots: worker.command('reset',robot=robot)
                    elif b' ' in key: worker.command('start',robots=a.robots)
                    else:
                        for r in a.robots:
                            if str(r).encode() in key: worker.command('start',robots=[r])
                elif b'q' in key or b'\x03' in key or not key: break
                elif b'r' in key or b'R' in key:
                    session.toggle_receive()
                elif b's' in key or b'S' in key:
                    if writer is None or not writer.is_alive():
                        writer=threading.Thread(target=session.persist);writer.start()
                elif b'e' in key or b'E' in key:
                    with session.lock:
                        if launch is None and session.mode=='receive' and session.saved=='Saved':
                            session.receive_armed=False
                            session.notice='Choose episodes per robot'
                            episode_input=''
                        elif session.mode=='receive':
                            session.notice='Save checkpoint with S before starting eval'
    finally:
        if launch is not None:
            try: worker=launch.result()
            except Exception: pass
        launcher.shutdown(wait=True)
        if worker:
            worker.command('stop')
            while not worker.poll(): time.sleep(.1)
        stop.set();thread.join()
        if writer: writer.join()
        channel.close();session.close();base.close()


if __name__=='__main__':main()
