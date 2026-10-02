"""Actual WS request handler with a fake robot and in-memory websocket."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from websockets.exceptions import ConnectionClosed
from openpi_client import msgpack_numpy
from client import run_client
from expo_ft.env.model_frame import ModelFrame
from expo_ft.env.env_client import EnvClient

class FrameTests(unittest.TestCase):
    def test_observation_and_action_match_previous_canonical_convention(self):
        image=np.arange(18).reshape(2,3,3)
        obs=dict(cartesian_position=np.arange(6,dtype=float),gripper_position=[1],
                 exterior_image_1_left=image,exterior_image_2_left=image,wrist_image_left=image)
        frame=ModelFrame({'mirror_images':{'side':True,'wrist':True},'mirror_robot_coordinates':True})
        result=frame.observation(obs)
        np.testing.assert_array_equal(result['cartesian_position'],[0,-1,2,-3,4,-5])
        np.testing.assert_array_equal(result['wrist_image_left'],image[:,::-1])
        np.testing.assert_array_equal(obs['cartesian_position'],np.arange(6))
        a=np.arange(7,dtype=float)
        np.testing.assert_array_equal(frame.action(frame.action(a)),a)
        independent=ModelFrame({'mirror_images':{'side':True}}).observation(obs)
        np.testing.assert_array_equal(independent['wrist_image_left'],image)
        np.testing.assert_array_equal(independent['cartesian_position'],obs['cartesian_position'])

    def test_policy_human_and_clipped_actions_cross_boundary_once(self):
        frame=ModelFrame({'mirror_robot_coordinates':True})
        for human in (False,True):
            calls=[]
            def step(a):
                calls.append(a.copy());a=a.copy();a[0]=0 # Physical workspace clipping.
                return {'executed_action':a}
            env=SimpleNamespace(_model_frame=frame,step=step,close=lambda:None,last_action_send_ms=1200.)
            class WS:
                remote_address='fake';transport=SimpleNamespace(get_extra_info=lambda _:None)
                request=msgpack_numpy.packb({'operation':'step','env_id':'r','action':np.ones(7)*.2,'observation_metadata':{'frame_received_ms':{'side':1000.,'wrist':1010.}}})
                async def recv(self):
                    if self.request is None:raise ConnectionClosed(None,None)
                    r,self.request=self.request,None;return r
                async def send(self,r):self.reply=msgpack_numpy.unpackb(r)
            ws=WS(); human_action=np.arange(7,dtype=float)/10
            with patch.dict(run_client._env_storage,{'r':env},clear=True), patch.object(run_client,'_task_config',SimpleNamespace(env_type='droid')), patch.object(run_client,'_get_human_override_action',return_value=(human_action,human)):
                asyncio.run(run_client._handle_environment_request(ws))
            expected=human_action if human else frame.action(np.ones(7)*.2)
            np.testing.assert_array_equal(calls[0],expected)
            clipped=expected.copy();clipped[0]=0
            np.testing.assert_array_equal(ws.reply['action'],frame.action(clipped))
            self.assertEqual(ws.reply['action_type'],'human' if human else 'policy')
            self.assertEqual(ws.reply['action_timing']['frame_age_ms'],{} if human else {'side':200.,'wrist':190.})

    def test_old_server_rejected(self):
        c=EnvClient()
        with patch.object(c,'_call_operation',return_value={'env_id':'r','task_description':'task'}):
            with self.assertRaisesRegex(RuntimeError,'model-frame'):c.create_env({})
        c.close()
