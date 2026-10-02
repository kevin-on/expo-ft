"""Synthetic pixels/calibration only; never instantiate a camera."""
import unittest
import numpy as np
from client.envs.camera_config import selected_views, prepare_images, crop_for

class CameraConfigTests(unittest.TestCase):
    def test_only_selected_eyes(self):
        self.assertEqual(selected_views(('side_right','wrist_left')), {'side':['right'],'wrist':['left']})
    def test_crop_only_for_matching_mode_and_adjust_intrinsics(self):
        frame = np.arange(8*12*3,dtype=np.uint8).reshape(8,12,3)
        crop={'s_right':{'720p':[2,1,8,6]}}
        raw={'image':{'s_right':frame}, 'camera_intrinsics':{'s_right':np.array([[10.,0,6],[0,12,4],[0,0,1]])}}
        images,K=prepare_images(raw, ['s_right'],crop,{'s_right':'720p'},(3,4))
        self.assertEqual(images['s_right'].shape,(3,4,3))
        np.testing.assert_allclose(K['s_right'],[[5,0,2],[0,6,1.5],[0,0,1]])
        self.assertIs(crop_for('s_right',frame,crop,'1080p')[0],frame)
        with self.assertRaises(ValueError):crop_for('s_right',frame,{'s_right':{'720p':[0,0,13,2]}},'720p')

if __name__ == '__main__': unittest.main()

class ReaderTests(unittest.TestCase):
    def test_right_only_uses_requested_resolution_and_fps(self):
        from unittest.mock import Mock
        from droid.camera_utils.camera_readers.zed_camera import ZedCamera, sl
        # No __init__/open/device enumeration: inject a fake SDK instance.
        camera = ZedCamera.__new__(ZedCamera)
        camera.serial_number = 's'
        camera.set_reading_parameters(views=['right'], capture_resolution='720p', camera_fps=30)
        self.assertEqual(camera.trajectory_params['camera_fps'], 30)
        self.assertEqual(camera.trajectory_params['camera_resolution'], sl.RESOLUTION.HD720)
        camera.skip_reading=False; camera.image=True; camera.concatenate_images=False
        camera._cam=Mock();camera._runtime=object();camera._right_img=object();camera._left_img=object()
        camera.zed_resolution=object();camera.latency=83
        camera._cam.grab.return_value=sl.ERROR_CODE.SUCCESS
        camera._cam.get_timestamp.return_value.get_milliseconds.return_value=1234
        camera._process_frame=lambda _:np.zeros((2,2,3),np.uint8)
        data, stamps=camera.read_camera()
        self.assertEqual(list(data['image']), ['s_right'])
        self.assertEqual(stamps['s_frame_received'],1234)
        camera._cam.retrieve_image.assert_called_once_with(camera._right_img,sl.VIEW.RIGHT,resolution=camera.zed_resolution)
