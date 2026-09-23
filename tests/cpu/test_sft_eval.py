"""Lightweight SFT eval checks: no JAX, model, camera, socket, or robot access."""

import ast
import copy
import os
from collections import deque
from types import SimpleNamespace
from unittest import mock
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from expo_ft.env.sft_eval import canonical_observation, physical_action
from scripts.convert_two_robot_data_to_lerobot import prepare_step_for_sft
from test_two_robot_conversion import example_step


class SFTEvalTests(unittest.TestCase):
    def test_both_live_mappings_match_training_converter(self):
        for robot in (0, 1):
            with self.subTest(robot=robot):
                raw = example_step()
                before = copy.deepcopy(raw)
                cfg = json.loads((ROOT / f"configs/robots/robot-{robot}-sft-eval.json").read_text())
                side_eye = cfg["side_camera_id"].rsplit("_", 1)[1]
                wrist_eye = cfg["wrist_camera_id"].rsplit("_", 1)[1]
                side = raw["saved_observation"][f"exterior_image_1_{side_eye}"]
                wrist = raw["saved_observation"][f"wrist_image_{wrist_eye}"]
                obs = dict(raw["saved_observation"], exterior_image_1_left=side,
                           exterior_image_2_left=side, wrist_image_left=wrist, prompt="pick up the cube")
                actual = canonical_observation(obs, mirror=robot == 1)
                expected = prepare_step_for_sft(raw, mirror=robot == 1)
                for key, value in expected["saved_observation"].items():
                    np.testing.assert_array_equal(actual[key], value)
                self.assertEqual(actual["prompt"], "pick up the cube")
                raw_policy_output = np.r_[expected["action"]["cartesian_velocity"],
                                          expected["action"]["gripper_velocity"]]
                # Here the policy output has already been unnormalized.
                np.testing.assert_allclose(physical_action(raw_policy_output, mirror=robot == 1),
                                           np.r_[raw["action"]["cartesian_velocity"], -1])
                for group in raw:
                    for key in raw[group]:
                        np.testing.assert_array_equal(raw[group][key], before[group][key])
                for eye, role in ((side_eye, "varied_camera"), (wrist_eye, "hand_camera")):
                    if eye == "right":
                        self.assertFalse(cfg["camera_kwargs"][role]["left_only"])
                training_cfg = json.loads((ROOT / f"configs/robots/robot-{robot}.json").read_text())
                for key in ("robot_server_ip", "robot_server_port", "camera_serials", "spacemouse_device_path"):
                    self.assertEqual(cfg[key], training_cfg[key])

    def test_inverse_action_transform_and_gripper(self):
        action = np.array([1., 2., 3., 4., 5., 6., -1.])
        mirrored = physical_action(action, True)
        np.testing.assert_array_equal(mirrored, [1, -2, 3, -4, 5, -6, -1])
        np.testing.assert_array_equal(physical_action(mirrored, True), action)
        self.assertFalse(np.shares_memory(mirrored, action))
        for invalid in (np.zeros(6), np.zeros((1, 7)), np.full(7, np.nan)):
            with self.assertRaises(ValueError):
                physical_action(invalid, True)

    def test_camera_calibration_not_passed_as_mirrored_model_input(self):
        obs = example_step()["saved_observation"]
        obs["camera_intrinsics"] = {"physical_camera": np.eye(3)}
        canonical = canonical_observation(obs, True)
        self.assertNotIn("camera_intrinsics", canonical)
        with self.assertRaises(ValueError):
            canonical_observation(dict(obs, cartesian_position=np.zeros(7)), True)

    def test_actual_eval_loop_mirrors_actions_and_logs_interventions(self):
        # Execute the real rollout loop AST with in-memory env/policy doubles.
        # Omit imports/model initialization so no device, network or GPU is opened.
        tree = ast.parse((ROOT / "eval_droid_policy.py").read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        start = next(i for i, node in enumerate(main.body)
                     if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id == "successes")
        main.name = "run_eval_loop"
        main.args.args[0].arg = "agent"
        main.body = main.body[start:]
        loop_code = compile(ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[])),
                            "eval_droid_policy.py:rollout", "exec")
        for mirror in (False, True):
            max_steps, end_after = 4, 2
            with self.subTest(mirror=mirror):
                sample = example_step()
                obs = dict(sample["saved_observation"], wrist_image_left=sample["saved_observation"]["wrist_image_right"])
                expected = prepare_step_for_sft(sample, True)
                policy_action = np.r_[expected["action"]["cartesian_velocity"], -1]
                env = mock.Mock()
                sent = []
                env.reset.return_value = obs
                env.get_observation.return_value = obs
                env.get_info_for_step.side_effect = lambda: (len(sent) >= end_after, len(sent) >= end_after,
                                                             float(len(sent) >= end_after), 1.)
                def step(action):
                    sent.append(action)
                    return np.asarray(action), "human" if len(sent) == 1 else "policy"
                env.step.side_effect = step
                agent = mock.Mock(spec=["sample_actions"])
                received = []
                def sample_actions(observation, only_base_actions):
                    self.assertTrue(only_base_actions)
                    received.append(copy.deepcopy(observation))
                    return policy_action[None], agent, {}
                agent.sample_actions.side_effect = sample_actions
                logger = mock.Mock()
                namespace = dict(
                    FLAGS=SimpleNamespace(num_episodes=1, delay=0, sim_latency=0, replan_steps=1,
                                          only_base_actions=True, mirror_y=mirror),
                    np=np, json=json, os=os, time=SimpleNamespace(time=lambda: 0., sleep=lambda _: None),
                    deque=deque, logger=logger,
                    jax=SimpleNamespace(device_get=lambda value: value),
                    agent=agent, env=env, model_cls="EXPOLearner",
                    max_traj_len=max_steps, dt=0., video_dir="/workstation/videos",
                    canonical_observation=canonical_observation, physical_action=physical_action,
                )
                exec(loop_code, namespace)
                namespace["run_eval_loop"](agent)
                self.assertEqual(env.reset.call_count, 1)
                self.assertEqual(len(sent), 2)
                self.assertEqual(len(received), 3)  # preserve the original loop ordering
                for prediction_input in received:
                    expected_obs = expected["saved_observation"] if mirror else obs
                    for key, value in expected_obs.items():
                        np.testing.assert_array_equal(prediction_input[key], value)
                for action in sent:
                    expected_action = np.r_[sample["action"]["cartesian_velocity"], -1] if mirror else policy_action
                    np.testing.assert_allclose(action, expected_action)
                logger.info.assert_any_call("  success=%s return=%.1f len=%d human_override_steps=%d",
                                            True, 1., 3, 1)

    def test_no_heavy_or_hardware_imports(self):
        for name in ("jax", "torch", "pyzed", "droid", "pyspacemouse", "lerobot"):
            self.assertFalse(any(key == name or key.startswith(name + ".") for key in sys.modules), name)


if __name__ == "__main__":
    unittest.main()
