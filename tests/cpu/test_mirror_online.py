"""Mirror online collection tests: CPU memory/files only, no devices or sockets."""
import ast
import copy
import json
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from expo_ft.env.sft_eval import validate_camera_views
# Load this NumPy-only module directly, avoiding utils/__init__'s unrelated JAX imports.
spec = importlib.util.spec_from_file_location("mirror_test_robot_round", ROOT / "expo_ft/utils/robot_round.py")
robot_round = importlib.util.module_from_spec(spec)
spec.loader.exec_module(robot_round)
collect_round = robot_round.collect_round
from scripts.convert_two_robot_data_to_lerobot import prepare_step_for_sft
from test_two_robot_conversion import example_step

class MirrorOnlineTests(unittest.TestCase):
    def test_online_observations_executed_actions_and_human_actions_match_converter(self):
        class Env:
            def __init__(self, robot):
                self.robot = robot
                self.config = json.loads((ROOT / f"configs/robots/robot-{robot}-sft-eval.json").read_text())
                self.samples, self.executed, self.commands = [], [], []
            def reset(self):
                self.index = 0
                return self.get_observation()
            def get_observation(self):
                sample = example_step()
                sample['saved_observation']['cartesian_position'] += self.index + self.robot
                self.samples.append(copy.deepcopy(sample))
                obs = copy.deepcopy(sample['saved_observation'])
                side = obs['exterior_image_1_' + self.config['side_camera_id'].rsplit('_', 1)[1]]
                obs.update(exterior_image_1_left=side, exterior_image_2_left=side,
                           wrist_image_left=obs['wrist_image_' + self.config['wrist_camera_id'].rsplit('_', 1)[1]])
                return obs
            def step(self, command):
                self.commands.append(np.array(command))
                executed = np.array(command)
                human = self.robot == 1 and self.index == 1
                if human:
                    executed = np.array([.1, .2, .3, .4, .5, .6, -1.])
                else:
                    executed[1] = 0  # simulate physical workspace clipping
                self.executed.append(executed.copy())
                self.index += 1
                return executed, 'human' if human else 'policy'
            def get_info_for_step(self):
                done = self.index == 4
                return done, done, float(done), float(not done)
            def close(self):
                pass
        envs = [Env(0), Env(1)]
        policy_action = np.array([.3, -.2, .1, -.4, .5, -.6, 1.])
        inference_observations = []
        def sample(obs):
            inference_observations.append(copy.deepcopy(obs))
            obs['cartesian_position'][:] = 999  # must not corrupt stored observations
            return np.tile(policy_action, (2, 1))
        episodes = collect_round(envs, sample, 2, 10000, mirror_robot=1)
        for robot, (rows, success) in enumerate(episodes):
            assert success and len(rows) == 4
            env = envs[robot]
            for index, row in enumerate(rows):
                sample = env.samples[index]
                sample['action'] = dict(cartesian_velocity=env.executed[index][:6], gripper_velocity=env.executed[index][6])
                expected = prepare_step_for_sft(sample, mirror=robot == 1)
                for key, value in expected['saved_observation'].items():
                    np.testing.assert_array_equal(row['observations'][key], value)
                np.testing.assert_allclose(row['actions'], np.r_[expected['action']['cartesian_velocity'], expected['action']['gripper_velocity']])
                assert row['is_hil'] == (robot == 1 and index == 1)
            np.testing.assert_allclose(env.commands[0], policy_action * np.array([1, -1, 1, -1, 1, -1, 1]) if robot else policy_action)
            assert rows[-1]['dones'] and rows[-1]['rewards'] == 1
        np.testing.assert_array_equal(envs[1].commands[2], np.zeros(7))  # unchanged human-to-policy handoff
        np.testing.assert_allclose(envs[1].commands[3], envs[1].commands[0])
        # Every policy input is an actual canonical observation, never a physical robot1 pose/image.
        expected_observations = [row['observations'] for rows, _ in episodes for row in rows]
        for observation in inference_observations:
            assert any(all(np.array_equal(observation[k], candidate[k]) for k in observation) for candidate in expected_observations)


    def test_camera_mapping_check_runs_before_hardware_construction(self):
        for robot in (0, 1):
            correct = json.loads((ROOT / f'configs/robots/robot-{robot}-sft-eval.json').read_text())
            expected = {k: correct[k] for k in ('side_camera_id', 'wrist_camera_id')}
            validate_camera_views(correct, expected)
            old = json.loads((ROOT / f'configs/robots/robot-{robot}.json').read_text())
            with self.assertRaisesRegex(ValueError, 'Mirror training requires'):
                validate_camera_views(old, expected)
        tree = ast.parse((ROOT / 'client/run_client.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        check = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'validate_camera_views')
        create = next(n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'env')
        assert check.lineno < create.lineno


if __name__ == "__main__":
    unittest.main()
