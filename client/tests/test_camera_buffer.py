"""Bounded latest-frame lifecycle tests, synthetic producers only."""
import threading
import time
import unittest
import numpy as np
from droid.camera_utils.wrappers.latest_frame import LatestFrame
from client.envs.camera_config import prepare_images

class BufferTests(unittest.TestCase):
    def test_reads_dont_grab_and_reset_waits_for_new_timestamp(self):
        calls=[];release=threading.Event()
        def read():
            calls.append(1)
            if len(calls)>1: release.wait(1)
            return {'image':np.array([len(calls)])},{'s_frame_received':time.time_ns()/1e6}
        b=LatestFrame(read,fps=100,timeout=1)
        try:
            first,stamps=b.get()
            for _ in range(4):
                self.assertIs(b.get()[0],first)
            self.assertEqual(len(calls),1)
            b.require_fresh(time.time_ns()/1e6)
            result=[];t=threading.Thread(target=lambda:result.append(b.get()));t.start()
            time.sleep(.025);self.assertEqual(result,[])
            release.set();t.join(1)
            self.assertGreaterEqual(result[0][0]['image'][0],2)
            self.assertEqual(first['image'][0],1) # Retained video frames unaffected.
        finally:release.set();b.close()

    def test_error_and_stale_images_do_not_return_old_success(self):
        def fail():raise ValueError('unplugged')
        b=LatestFrame(fail,timeout=.1)
        try:
            with self.assertRaisesRegex(RuntimeError,'Background'):b.get()
        finally:b.close()
        b=LatestFrame(lambda:({}, {'s_frame_received':0}),timeout=.03)
        try:
            with self.assertRaises(TimeoutError):b.get()
        finally:b.close()

    def test_two_camera_wrapper_publishes_complete_pair_and_prepares_once(self):
        from concurrent.futures import ThreadPoolExecutor
        from droid.camera_utils.wrappers.multi_camera_wrapper import MultiCameraWrapper
        from unittest.mock import Mock
        w=MultiCameraWrapper.__new__(MultiCameraWrapper)
        w._latest=w._executor=w._buffer_settings=w._processor=None
        w._recording=w._calibrating=False
        w.camera_dict={}
        for name in ('s','w'):
            camera=Mock();camera.is_running.return_value=True
            camera.read_camera.side_effect=lambda n=name:({'image':{n:np.zeros((3,4,3),np.uint8)}},{n+'_frame_received':time.time_ns()/1e6})
            w.camera_dict[name]=camera
        prepared=[]
        def process(data):prepared.append(1);return data
        w.configure_buffer(process,fps=1)
        try:
            data,ts=w.read_cameras()
            self.assertEqual(set(data['image']),{'s','w'})
            self.assertIs(w.read_cameras()[0],data)
            self.assertEqual(len(prepared),1)
            for cam in w.camera_dict.values():cam.read_camera.assert_called_once()
        finally:w.disable_cameras()
        for cam in w.camera_dict.values():cam.disable_camera.assert_called_once()
