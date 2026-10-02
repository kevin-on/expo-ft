"""Benchmark scheduling/settings/lifecycle without importing the camera SDK."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import runpy
import tempfile
import threading
from types import SimpleNamespace
import unittest

benchmark = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                              'scripts/multi_robot/benchmark_camera_latency.py'))


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.paths = []
        for robot in range(2):
            side, wrist = f's{robot}', f'w{robot}'
            path = Path(folder.name)/f'{robot}.json'
            path.write_text(json.dumps(dict(camera_serials=[side,wrist],wrist_camera_serial=wrist,
                side_camera_id=side+'_right',wrist_camera_id=wrist+'_left',
                camera_kwargs={'hand_camera':{'left_only':False}})))
            self.paths.append(str(path))
        self.args = SimpleNamespace(resolution='720p',fps=30,selected_eyes=True)

    def test_grouping_selection_and_native_eye_settings(self):
        for mode, sizes in [('per-robot',[2,2]),('all-parallel',[4]),('independent',[1,1,1,1])]:
            plans = benchmark['camera_plan'](self.paths,scheduling=mode)
            self.assertEqual([len(p['cameras']) for p in plans],sizes)
        plans = benchmark['camera_plan'](self.paths,serials=['w0','s1'],scheduling='all-parallel')
        self.assertEqual([c['serial'] for c in plans[0]['cameras']],['w0','s1'])
        opened, closed = [], []
        class Wrapper:
            def __init__(self,settings,serials,wrist,camera_views):
                self.serial = serials[0]
                self.camera_dict = {self.serial:object()}
                opened.append((settings,serials,wrist,camera_views))
            def read_cameras(self):return self.serial
            def disable_cameras(self):closed.append(self.serial)
        before = [Path(p).read_text() for p in self.paths]
        with benchmark['camera_group'](self.paths,['w0','s1'],self.args,Wrapper) as (cameras,read):
            self.assertEqual(set(cameras),{'w0','s1'})
            self.assertEqual(set(read()),{'w0','s1'})
        self.assertEqual(set(closed),{'w0','s1'})
        self.assertEqual(opened[0][2:],('w0',{'w0':['left']}))
        self.assertEqual(opened[1][2:],('w1',{'s1':['right']}))
        self.assertEqual(opened[0][0]['hand_camera']['camera_fps'],30)
        self.assertEqual(opened[1][0]['varied_camera']['capture_resolution'],'720p')
        self.assertEqual(before,[Path(p).read_text() for p in self.paths])

    def test_read_failure_waits_for_other_read_before_closing(self):
        entered, failed, release = threading.Event(), threading.Event(), threading.Event()
        closed = []
        class Wrapper:
            def __init__(self,settings,serials,wrist,camera_views):
                self.serial = serials[0]
                self.camera_dict = {self.serial:object()}
            def read_cameras(self):
                if self.serial == 's0':
                    entered.wait(2); failed.set(); raise RuntimeError('read failed')
                entered.set(); release.wait(2)
            def disable_cameras(self):closed.append(self.serial)
        def read():
            with benchmark['camera_group'](self.paths,['s0','w0'],self.args,Wrapper) as (_,read):read()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(read)
            try:
                self.assertTrue(failed.wait(1))
                self.assertFalse(future.done())
                self.assertEqual(closed,[])
            finally:release.set()
            with self.assertRaisesRegex(RuntimeError,'read failed'):future.result(timeout=2)
        self.assertEqual(set(closed),{'s0','w0'})
