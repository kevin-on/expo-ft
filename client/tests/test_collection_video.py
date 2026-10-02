"""Low-resolution MP4 frame checks without cameras, encoders, or files."""
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from client import collect_data as collect


class CollectionVideoTests(unittest.TestCase):
    def test_recording_reuses_resized_obs_for_mp4_and_hdf5(self):
        env, controller, hdf5, video = Mock(), Mock(), Mock(), Mock()
        hdf5.attrs = {}
        env.side_camera_id = 'side_right'
        env.control_hz = 10
        # Deliberately different pixels from the saved observation to detect upscaling.
        full = np.full((1080,1920,3), [10,20,30], np.uint8)
        raw = {'image': {'side_left': full}, 'timestamp': {}}
        small = np.full((180,320,3), 99, np.uint8)
        env.get_raw_observation.return_value = raw
        env.get_info_for_step.side_effect = [(False,False,0,1),(True,True,1,0)]
        controller.get_info.return_value = {'movement_enabled': True}
        controller.forward.return_value = (np.zeros(7), {})
        env.step.return_value = {'cartesian_velocity': np.zeros(6), 'gripper_velocity':0}
        fake_flags = SimpleNamespace(save_right_images=True, video_save_width=320, video_save_height=180, video_encoder_threads=2)
        with patch.object(collect,'FLAGS',fake_flags), \
             patch.object(collect,'collection_observation',return_value={'exterior_image_1_left':small}), \
             patch.object(collect.h5py,'File',return_value=hdf5), \
             patch.object(collect,'write_dict_to_hdf5') as write_hdf5, \
             patch.object(collect.imageio,'get_writer',return_value=video) as get_writer, \
             patch.object(collect.os,'makedirs'), redirect_stdout(io.StringIO()):
            collect.collect_trajectory(env,controller,save_filepath='/not-written/traj.hdf5',
                                       recording_folderpath='/not-written/images')
        stored = write_hdf5.call_args.args[1]['saved_observation']['exterior_image_1_left']
        self.assertIs(stored, small)
        frame = video.append_data.call_args.args[0]
        self.assertEqual(frame.shape, (180,320,3))
        self.assertIs(frame, small)
        self.assertEqual(collect.FLAGS["video_save_width"].default, 320)
        self.assertEqual(collect.FLAGS["video_save_height"].default, 180)
        self.assertEqual(get_writer.call_args.kwargs['macro_block_size'],1)
        video.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
