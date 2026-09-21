"""Camera-only multi-process throughput test; never imports/constructs RobotEnv.

Run with the client venv from the repository root. Each --pair owns two cameras,
read concurrently as in rollout. Images are copied by the existing DROID reader
and discarded; only timings and resource samples are saved. Shared standard
camera defaults are changed only in each spawned test process, never on disk.
"""
import argparse
import csv
import json
import multiprocessing as mp
from pathlib import Path
import queue
import statistics
import subprocess
import time
import traceback


def distribution(values):
    if not values:
        return None
    values = sorted(values)
    return {'mean': statistics.mean(values), 'p95': values[int(.95*(len(values)-1))],
            'p99': values[int(.99*(len(values)-1))], 'max': max(values)}


def worker(index, serials, args, start, stop, messages):
    from concurrent.futures import ThreadPoolExecutor
    from droid.camera_utils.camera_readers import zed_camera as zed
    import pyzed.sl as sl
    cameras = []
    output = Path(args.output)
    result = {'pair': index, 'serials': serials, 'passed': False}
    try:
        if not args.use_camera_defaults:
            zed.standard_params['camera_resolution'] = sl.RESOLUTION.HD1080
            zed.standard_params['camera_fps'] = args.fps
        cameras = zed.gather_zed_cameras(serials, wrist_camera_serial=serials[1])
        configurations = []
        for camera in cameras:
            camera.set_reading_parameters(image=True, depth=False, pointcloud=False, left_only=True)
            camera.set_trajectory_mode()
            cfg = camera._cam.get_camera_information().camera_configuration
            depth_mode = camera._cam.get_init_parameters().depth_mode
            if args.expect_depth_none and depth_mode != sl.DEPTH_MODE.NONE:
                raise RuntimeError(f'Expected depth NONE, got {depth_mode}')
            configurations.append({'serial': camera.serial_number, 'fps': cfg.fps,
                                   'width': cfg.resolution.width, 'height': cfg.resolution.height, 'depth_mode': str(depth_mode)})
            if (cfg.resolution.width, cfg.resolution.height) != (1920, 1080) or abs(cfg.fps-args.fps)>.1:
                raise RuntimeError(f'Applied camera mode differs from request: {configurations[-1]}')
        result['configurations'] = configurations
        messages.put({'event': 'ready', 'pair': index, 'configurations': configurations})
        while not start.wait(.2):
            if stop.is_set():
                raise RuntimeError('Test stopped before all cameras were ready')
        stats = {c.serial_number: {'frames': 0, 'errors': 0, 'duplicates': 0, 'backward_timestamps': 0,
                 'estimated_missing': 0, 'gaps_ms': [], 'read_ms': [], 'last_timestamp': None,
                 'sdk_dropped_start': None, 'sdk_dropped_end': None} for c in cameras}
        def read(camera):
            began = time.monotonic()
            item = camera.read_camera()
            ended = time.monotonic()
            if item is None:
                return camera.serial_number, ended, (ended-began)*1000, None, 'grab_failed', camera._cam.get_frame_dropped_count()
            data, stamps = item
            frame = data['image'][camera.serial_number+'_left']
            if frame.shape != (1080,1920,3):
                raise RuntimeError(f'Unexpected image shape {camera.serial_number}: {frame.shape}')
            return camera.serial_number, ended, (ended-began)*1000, stamps[camera.serial_number+'_frame_received'], '', camera._cam.get_frame_dropped_count()
        measured_start = time.monotonic()+args.warmup
        deadline = measured_start+args.duration
        measured_cpu = None
        last_report = measured_start
        end = measured_start
        with (output/f'pair-{index}-frames.csv').open('w') as stream, ThreadPoolExecutor(max_workers=2) as pool:
            writer = csv.writer(stream)
            writer.writerow(['serial','elapsed_seconds','read_ms','image_timestamp_ms','error','sdk_dropped'])
            while not stop.is_set() and time.monotonic()<deadline:
                rows = list(pool.map(read, cameras))
                now = time.monotonic()
                if now < measured_start:
                    continue
                if measured_cpu is None:
                    measured_cpu = time.process_time()
                end = now
                for serial, ended, latency, timestamp, error, dropped in rows:
                    # Exclude calls completing before the common measurement boundary.
                    if ended < measured_start:
                        continue
                    s = stats[serial]
                    writer.writerow([serial, ended-measured_start, latency, timestamp, error, dropped])
                    if s['sdk_dropped_start'] is None:
                        s['sdk_dropped_start'] = dropped
                    s['sdk_dropped_end'] = dropped
                    s['read_ms'].append(latency)
                    if error:
                        s['errors'] += 1
                        continue
                    previous = s['last_timestamp']
                    if previous is not None:
                        gap = timestamp-previous
                        if gap == 0:
                            s['duplicates'] += 1
                            continue
                        if gap < 0:
                            s['backward_timestamps'] += 1
                            continue
                        s['gaps_ms'].append(gap)
                        s['estimated_missing'] += max(0, round(gap/(1000/args.fps))-1)
                    s['last_timestamp'] = timestamp
                    s['frames'] += 1
                if now-last_report>=30:
                    messages.put({'event':'progress','pair':index,'seconds':round(now-measured_start,1),
                                  'fps':{serial:round(s['frames']/(now-measured_start),2) for serial,s in stats.items()},
                                  'errors':{serial:s['errors'] for serial,s in stats.items()}})
                    stream.flush()
                    last_report=now
        elapsed=end-measured_start
        result['elapsed_seconds']=elapsed
        result['mean_cpu_cores']=(time.process_time()-measured_cpu)/elapsed if measured_cpu is not None and elapsed>0 else None
        result['cameras']={}
        for serial,s in stats.items():
            frames=s['frames']; missing=s['estimated_missing']
            summary={k:v for k,v in s.items() if k not in ('gaps_ms','read_ms','last_timestamp')}
            summary.update(received_fps=frames/elapsed if elapsed>0 else 0,
                           estimated_missing_fraction=missing/(frames+missing) if frames+missing else 1,
                           frame_interval_ms=distribution(s['gaps_ms']),read_latency_ms=distribution(s['read_ms']),
                           sdk_dropped_delta=(s['sdk_dropped_end']-s['sdk_dropped_start']) if s['sdk_dropped_start'] is not None else None)
            summary['passed']=(elapsed>=args.duration and summary['received_fps']>=args.fps*.98
                               and summary['estimated_missing_fraction']<=.01 and s['errors']==0
                               and s['duplicates']==0 and s['backward_timestamps']==0
                               and bool(s['gaps_ms']) and max(s['gaps_ms'])<=250)
            result['cameras'][serial]=summary
        result['passed']=all(s['passed'] for s in result['cameras'].values()) and not stop.is_set()
    except BaseException as exc:
        result['error']=str(exc)
        result['traceback']=traceback.format_exc()
        stop.set()
    finally:
        for camera in cameras:
            camera.disable_camera()
        (output/f'pair-{index}-summary.json').write_text(json.dumps(result,indent=2)+'\n')
        messages.put({'event':'finished','pair':index,'passed':result['passed'],'error':result.get('error')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair',nargs=2,action='append',required=True,metavar=('SIDE_SERIAL','WRIST_SERIAL'))
    parser.add_argument('--fps',type=int,choices=[15,30],default=15)
    parser.add_argument('--use-camera-defaults',action='store_true',help='Validate production resolution/FPS without overriding them')
    parser.add_argument('--expect-depth-none',action='store_true',help='Fail if SDK depth computation is enabled')
    parser.add_argument('--duration',type=float,default=300)
    parser.add_argument('--warmup',type=float,default=10)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    serials=[s for pair in args.pair for s in pair]
    if len(set(serials))!=len(serials) or args.duration<=0 or args.warmup<0:
        parser.error('Use disjoint camera serials, positive duration and nonnegative warmup')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'request.json').write_text(json.dumps({**vars(args),'output':str(args.output),
        'capture':'DROID standard parameters, overridden HD1080/fps; left RGB retrieval and copy; no RobotEnv',
        'criteria':'full duration, >=98% requested FPS, <=1% timestamp-inferred loss, no grab errors/duplicate/backward timestamps, max gap <=250ms'},indent=2)+'\n')
    context=mp.get_context('spawn')
    start,stop=context.Event(),context.Event()
    messages=context.Queue()
    workers=[context.Process(target=worker,args=(i,pair,args,start,stop,messages)) for i,pair in enumerate(args.pair)]
    ready=set(); finished=set(); began=time.monotonic(); measuring=None; resource_at=0
    def cpu_counters():
        fields=[int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
        return sum(fields),fields[3]+fields[4]
    last_cpu=cpu_counters()
    try:
        for process in workers: process.start()
        with (args.output/'resources.jsonl').open('w') as resources:
            while len(finished)<len(workers):
                now=time.monotonic()
                if now-resource_at>=10:
                    total,idle=cpu_counters(); previous_total,previous_idle=last_cpu
                    usage=100*(1-(idle-previous_idle)/max(1,total-previous_total)); last_cpu=(total,idle)
                    sample=subprocess.run(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu','--format=csv,noheader'],text=True,capture_output=True,timeout=8)
                    resources.write(json.dumps({'seconds':now-began,'system_cpu_percent':usage,'gpu':sample.stdout.strip()})+'\n');resources.flush(); resource_at=now
                try: message=messages.get(timeout=.5)
                except queue.Empty: message=None
                if message:
                    print(json.dumps(message),flush=True)
                    if message['event']=='ready': ready.add(message['pair'])
                    if message['event']=='finished': finished.add(message['pair'])
                if len(ready)==len(workers) and measuring is None:
                    measuring=time.monotonic(); start.set()
                    print('All cameras open; synchronized warmup then measurement',flush=True)
                if (measuring is None and now-began>120) or (measuring is not None and now-measuring>args.warmup+args.duration+45):
                    raise TimeoutError('Camera worker did not finish within bounded timeout')
                for i,p in enumerate(workers):
                    if p.exitcode is not None and i not in finished and (args.output/f'pair-{i}-summary.json').exists():
                        finished.add(i)
                    elif p.exitcode not in (None,0) and i not in finished:
                        raise RuntimeError(f'Worker {i} exited {p.exitcode}')
    finally:
        stop.set(); start.set()
        for process in workers:
            if process.pid is None: continue
            process.join(timeout=10)
            if process.is_alive(): process.terminate();process.join(timeout=5)
            if process.is_alive(): process.kill();process.join()
    results=[json.loads((args.output/f'pair-{i}-summary.json').read_text()) for i in range(len(workers))]
    summary={'passed':all(r['passed'] for r in results),'pairs':results}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Result:',args.output/'summary.json','PASS' if summary['passed'] else 'FAIL',flush=True)
    return 0 if summary['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
