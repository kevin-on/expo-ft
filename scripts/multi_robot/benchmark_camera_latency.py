#!/usr/bin/env python3
"""Camera-only latency breakdown, using the real DROID camera wrappers.

No RobotEnv, RPC, HID, inference, video encoding or image files. Close camera
clients first. Run with client/.venv/bin/python from the repo root. Child
processes use JSON lens settings and depth NONE; --resolution and --fps change
capture settings only in the benchmark process. --selected-eyes reads only the
selected lens instead of both eyes for right-lens cameras. Use --serials to
select cameras explicitly, or --camera-count for a prefix of the configured
side/wrist pairs. By default each robot retains its own worker process.
--all-parallel puts all selected cameras in one read pool; --independent-cameras
gives each camera its own process/read loop. --dry-run prints
the plan without importing the SDK or accessing any devices.
Instrumentation is local to these child processes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import json
import multiprocessing as mp
from pathlib import Path
import statistics
import time
import traceback


def stats(values):
    values = sorted(values)
    return dict(mean=statistics.mean(values), p50=values[len(values)//2],
                p95=values[int(.95*(len(values)-1))], maximum=max(values))


class TimedSDK:
    def __init__(self, camera, row):
        self.camera, self.row = camera, row

    def __getattr__(self, name):
        return getattr(self.camera, name)

    def grab(self, *args, **kwargs):
        start = time.perf_counter()
        result = self.camera.grab(*args, **kwargs)
        self.row['grab_ms'] = (time.perf_counter()-start)*1000
        return result

    def retrieve_image(self, mat, view, **kwargs):
        start = time.perf_counter()
        result = self.camera.retrieve_image(mat, view, **kwargs)
        self.row['retrieve_'+str(view)+'_ms'] = (time.perf_counter()-start)*1000
        return result


def instrument(camera, expected_shape):
    import cv2
    row = {}
    original = camera.read_camera
    camera._cam = TimedSDK(camera._cam, row)

    def process(frame):
        start = time.perf_counter()
        array = frame.get_data()
        after_get = time.perf_counter()
        array = deepcopy(array)
        after_copy = time.perf_counter()
        if len(array.shape) == 3 and array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
        after_color = time.perf_counter()
        for key, val in [('get_data_ms', after_get-start), ('copy_ms', after_copy-after_get),
                         ('color_ms', after_color-after_copy)]:
            row[key] = row.get(key, 0)+val*1000
        if camera.resizer_resolution != (0, 0):
            raise RuntimeError('Unexpected resize in raw camera reader')
        return array

    def read():
        row.clear()
        start = time.perf_counter()
        result = original()
        row['total_ms'] = (time.perf_counter()-start)*1000
        if result is None:
            raise RuntimeError('Camera grab failed: '+camera.serial_number)
        data, stamps = result
        for frame in data['image'].values():
            if frame.shape != expected_shape:
                raise RuntimeError('Unexpected raw frame shape: '+str(frame.shape))
        row['image_timestamp_ms'] = stamps[camera.serial_number+'_frame_received']
        row['sdk_dropped'] = camera._cam.get_frame_dropped_count()
        return result

    camera._process_frame = process
    camera.read_camera = read
    return row


@contextmanager
def camera_group(config_paths, serials, args, wrapper_type):
    """Normal per-camera wrappers, with one read pool for the requested group.

    Each wrapper receives its owner's settings and selected eye. Shutdown joins
    every read before closing cameras, including when one read fails.
    """
    paths = config_paths if isinstance(config_paths, list) else [config_paths]
    with ExitStack() as resources:
        wrappers, cameras = [], {}
        for path in paths:
            config = json.loads(Path(path).read_text())
            eyes = dict(config[key].rsplit('_', 1) for key in ('side_camera_id', 'wrist_camera_id'))
            for role in ('hand_camera', 'varied_camera', 'static_camera'):
                settings = config.setdefault('camera_kwargs', {}).setdefault(role, {})
                settings.update(capture_resolution=args.resolution, camera_fps=args.fps)
            for serial in serials:
                if serial not in eyes:
                    continue
                if serial in cameras:
                    raise ValueError('Duplicate camera: '+serial)
                views = {serial: [eyes[serial]]} if args.selected_eyes else None
                wrapper = wrapper_type(config['camera_kwargs'], [serial],
                                       config['wrist_camera_serial'], camera_views=views)
                resources.callback(wrapper.disable_cameras)
                wrappers.append(wrapper)
                cameras.update(wrapper.camera_dict)
        if set(cameras) != set(serials):
            raise RuntimeError('Opened cameras differ from requested serials')
        if len(wrappers) == 1:
            read = wrappers[0].read_cameras
        else:
            pool = resources.enter_context(ThreadPoolExecutor(max_workers=len(wrappers)))
            def read():
                return list(pool.map(lambda wrapper: wrapper.read_cameras(), wrappers))
        yield cameras, read


def worker(index, config_path, serials, args, barrier, messages):
    resources = ExitStack()
    output = Path(args.output)
    try:
        import cv2
        import pyzed.sl as sl
        from droid.camera_utils.wrappers.multi_camera_wrapper import MultiCameraWrapper
        width, height = (1280, 720) if args.resolution == '720p' else (1920, 1080)
        cameras, read_cameras = resources.enter_context(
            camera_group(config_path, serials, args, MultiCameraWrapper))
        modes = []
        for camera in cameras.values():
            cfg = camera._cam.get_camera_information().camera_configuration
            depth = camera._cam.get_init_parameters().depth_mode
            modes.append(dict(serial=camera.serial_number, width=cfg.resolution.width,
                              height=cfg.resolution.height, fps=cfg.fps,
                              depth=str(depth), left_only=camera.left_only))
            assert (cfg.resolution.width, cfg.resolution.height, cfg.fps) == (width, height, args.fps)
            assert depth == sl.DEPTH_MODE.NONE
            modes[-1]['views'] = list(camera.views)
        rows = {serial: instrument(cam, (height, width, 3)) for serial, cam in cameras.items()}
        messages.put(dict(event='ready', robot=index, modes=modes, opencv_threads=cv2.getNumThreads()))
        summaries = []
        with (output/f'robot{index}-samples.jsonl').open('w') as stream:
            for requested_hz in args.request_hz:
                barrier.wait(timeout=120)
                warm_end = time.perf_counter()+args.warmup
                while time.perf_counter() < warm_end:
                    read_cameras()
                    if requested_hz:
                        time.sleep(1/requested_hz)
                barrier.wait(timeout=120)
                start = time.perf_counter()
                cpu_start = time.process_time()
                samples = []
                while time.perf_counter()-start < args.duration:
                    began = time.perf_counter()
                    obs = read_cameras()
                    ended = time.perf_counter()
                    sample = dict(request_hz=requested_hz, wall=time.time(),
                                  pair_ms=(ended-began)*1000,
                                  cameras={k:dict(v) for k,v in rows.items()})
                    samples.append(sample)
                    del obs
                    if requested_hz:
                        time.sleep(max(0, 1/requested_hz-(time.perf_counter()-began)))
                elapsed = time.perf_counter()-start
                summary = dict(request_hz=requested_hz, samples=len(samples), seconds=elapsed,
                               achieved_hz=len(samples)/elapsed,
                               cpu_cores=(time.process_time()-cpu_start)/elapsed,
                               pair_ms=stats([s['pair_ms'] for s in samples]), cameras={})
                for serial in rows:
                    readings = [s['cameras'][serial] for s in samples]
                    metrics = {k:stats([r[k] for r in readings]) for k in readings[0]
                               if k.endswith('_ms') and k != 'image_timestamp_ms'}
                    stamps = [r['image_timestamp_ms'] for r in readings]
                    metrics['duplicates'] = sum(a==b for a,b in zip(stamps,stamps[1:]))
                    metrics['image_interval_ms'] = stats([b-a for a,b in zip(stamps,stamps[1:])])
                    metrics['sdk_dropped_delta'] = readings[-1]['sdk_dropped']-readings[0]['sdk_dropped']
                    summary['cameras'][serial] = metrics
                summaries.append(summary)
                for sample in samples:
                    stream.write(json.dumps(sample)+'\n')
                stream.flush()
                messages.put(dict(event='phase_finished', robot=index, **summary))
        (output/f'robot{index}-summary.json').write_text(json.dumps(dict(config=str(config_path),
            modes=modes, opencv_threads=cv2.getNumThreads(), phases=summaries), indent=2)+'\n')
    except BaseException:
        messages.put(dict(event='error', robot=index, traceback=traceback.format_exc()))
        barrier.abort()
        raise
    finally:
        resources.close()


def camera_plan(config_paths, serials=None, count=None, *, scheduling="per-robot"):
    """Resolve selection from files only, preserving per-robot worker grouping."""
    entries = []
    known = []
    for index, path in enumerate(config_paths):
        config = json.loads(Path(path).read_text())
        cameras = []
        for key in ('side_camera_id', 'wrist_camera_id'):
            serial, eye = config[key].rsplit('_', 1)
            if eye not in ('left', 'right') or serial not in config['camera_serials']:
                raise ValueError(f'Invalid {key} in {path}')
            cameras.append(dict(serial=serial, eye=eye))
            known.append(serial)
        entries.append(dict(robot=index, config=str(path), cameras=cameras))
    if len(set(known)) != len(known):
        raise ValueError('Config files must identify distinct side/wrist cameras')
    if serials is None:
        count = len(known) if count is None else count
        if count > len(known):
            raise ValueError('Requested count exceeds configured cameras')
        serials = known[:count]
    if not serials or len(set(serials)) != len(serials):
        raise ValueError('Select at least one camera; duplicate serials are not allowed')
    unknown = set(serials)-set(known)
    if unknown:
        raise ValueError('Serials absent from configs: '+', '.join(sorted(unknown)))
    plans = []
    for entry in entries:
        entry['cameras'] = [c for c in entry['cameras'] if c['serial'] in serials]
        if entry['cameras']:
            plans.append(entry)
    if scheduling == 'independent':
        return [dict(robot=cam['serial'], config=entry['config'], cameras=[cam])
                for entry in plans for cam in entry['cameras']]
    if scheduling == 'all-parallel':
        return [dict(robot='all', config=[entry['config'] for entry in plans],
                     cameras=[cam for entry in plans for cam in entry['cameras']])]
    return plans


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs', nargs=2, default=['configs/robots/robot-0.json', 'configs/robots/robot-1.json'])
    p.add_argument('--duration', type=float, default=30, help='Seconds per phase')
    p.add_argument('--warmup', type=float, default=3)
    p.add_argument('--request-hz', nargs='+', type=float, default=[0, 10], help='0: read continuously; otherwise paced reads')
    p.add_argument('--output', type=Path)
    p.add_argument('--resolution', choices=['1080p', '720p'], default='1080p')
    p.add_argument('--fps', type=int, choices=[15, 30], default=15, help='Camera capture FPS, independent of --request-hz')
    selection = p.add_mutually_exclusive_group()
    selection.add_argument('--camera-count', type=int, choices=[1, 2, 3, 4],
                          help='Prefix: robot0 side, robot0 wrist, robot1 side, robot1 wrist; default all')
    selection.add_argument('--serials', nargs='+', help='Only open these configured camera serials')
    p.add_argument('--dry-run', action='store_true', help='Print selection/settings without accessing cameras')
    p.add_argument('--selected-eyes', action='store_true',
                   help='Read only each configured side/wrist lens, including right-only')
    scheduling = p.add_mutually_exclusive_group()
    scheduling.add_argument('--all-parallel', dest='scheduling', action='store_const', const='all-parallel',
                            help='One process/pool; wait for all selected cameras per read')
    scheduling.add_argument('--independent-cameras', dest='scheduling', action='store_const', const='independent',
                            help='One process/read loop per camera; synchronize phase starts only')
    p.set_defaults(scheduling='per-robot')
    args = p.parse_args()
    if args.duration < 2 or args.warmup < 0 or any(x < 0 for x in args.request_hz):
        p.error('Invalid duration/warmup/request-hz')
    try:
        plans = camera_plan(args.configs, args.serials, args.camera_count, scheduling=args.scheduling)
    except (ValueError, KeyError, OSError) as exc:
        p.error(str(exc))
    if args.dry_run:
        print(json.dumps(dict(resolution=args.resolution, fps=args.fps,
                              selected_eyes=args.selected_eyes, request_hz=args.request_hz, scheduling=args.scheduling,
                              workers=plans), indent=2))
        return
    if args.output is None:
        p.error('--output is required unless --dry-run is used')
    args.output.mkdir(parents=True, exist_ok=False)
    ctx = mp.get_context('spawn')
    barrier, messages = ctx.Barrier(len(plans)), ctx.Queue()
    children = [ctx.Process(target=worker, args=(plan['robot'], plan['config'],
                [c['serial'] for c in plan['cameras']], args, barrier, messages))
                for plan in plans]
    import queue
    try:
        for child in children:
            child.start()
        while any(child.is_alive() for child in children):
            try:
                print(json.dumps(messages.get(timeout=1)), flush=True)
            except queue.Empty:
                pass
        for child in children:
            child.join()
        while True:
            try:
                print(json.dumps(messages.get_nowait()), flush=True)
            except queue.Empty:
                break
        if any(child.exitcode != 0 for child in children):
            raise SystemExit('Camera measurement failed; see worker error')
    except KeyboardInterrupt:
        barrier.abort()
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            child.join()
        raise


if __name__ == '__main__':
    main()
