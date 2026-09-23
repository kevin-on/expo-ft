"""Tiny CPU-only checks; no LeRobot/Torch/JAX/robot imports or dataset conversion.

Run directly to avoid loading the rest of the test suite:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        client/.venv/bin/python tests/cpu/test_two_robot_conversion.py
"""

import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import convert_two_robot_data_to_lerobot as conversion


def example_step():
    # Distinct eyes and columns expose wrong-eye selection or vertical flipping.
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    return {
        "saved_observation": {
            "exterior_image_1_left": image.copy(),
            "exterior_image_2_left": image.copy(),
            "exterior_image_1_right": image + 30,
            "wrist_image_left": image + 60,
            "wrist_image_right": image + 90,
            "cartesian_position": np.array([0.4, 0.2, 0.3, 0.6, -0.4, 0.8], dtype=np.float32),
            "gripper_position": np.float32(0.25),
        },
        "action": {
            "cartesian_velocity": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
            "gripper_velocity": np.float32(-1),
        },
    }


def xyz_matrix(angles):
    x, y, z = map(float, angles)
    cx, cy, cz = math.cos(x), math.cos(y), math.cos(z)
    sx, sy, sz = math.sin(x), math.sin(y), math.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


class TwoRobotConversionTests(unittest.TestCase):
    def test_balanced_nested_reproducible_selection(self):
        inputs = [[Path(f"robot{robot}/{i}/traj.hdf5") for i in range(50)] for robot in (0, 1)]
        before = copy.deepcopy(inputs)
        pools = conversion.select_episode_pools(*inputs, seed=42)
        self.assertEqual(inputs, before)
        self.assertEqual(pools, conversion.select_episode_pools(*inputs, seed=42))
        previous = []
        for count in conversion.EPISODES_PER_ROBOT:
            selected = conversion.mixed_episodes(pools, count)
            self.assertEqual(len(selected), count * 2)
            self.assertEqual(selected[:len(previous)], previous)
            for robot in (0, 1):
                paths = [path for robot_id, path in selected if robot_id == robot]
                self.assertEqual(len(paths), count)
                self.assertEqual(len(set(paths)), count)
            previous = selected

    def test_shortage_and_overlap_rejected(self):
        paths = [Path(f"robot0/{i}/traj.hdf5") for i in range(50)]
        with self.assertRaisesRegex(ValueError, "need 50"):
            conversion.select_episode_pools(paths[:49], paths, seed=42)
        with self.assertRaisesRegex(ValueError, "same source"):
            conversion.select_episode_pools(paths, paths, seed=42)

    def test_mirror_uses_side_left_wrist_right_and_preserves_source(self):
        step = example_step()
        before = copy.deepcopy(step)
        mirrored = conversion.prepare_step_for_sft(step, mirror=True)
        obs = mirrored["saved_observation"]
        np.testing.assert_array_equal(obs["exterior_image_1_left"], np.array([
            [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            [[15, 16, 17], [12, 13, 14], [9, 10, 11]],
        ], dtype=np.uint8))
        np.testing.assert_array_equal(obs["exterior_image_2_left"], obs["exterior_image_1_left"])
        np.testing.assert_array_equal(obs["wrist_image_left"], obs["exterior_image_1_left"] + 90)
        np.testing.assert_allclose(obs["cartesian_position"], [0.4, -0.2, 0.3, -0.6, -0.4, -0.8])
        np.testing.assert_allclose(mirrored["action"]["cartesian_velocity"], [0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
        self.assertEqual(obs["gripper_position"], before["saved_observation"]["gripper_position"])
        self.assertEqual(mirrored["action"]["gripper_velocity"], -1)
        for group in before:
            for key in before[group]:
                np.testing.assert_array_equal(step[group][key], before[group][key])
        self.assertFalse(np.shares_memory(obs["exterior_image_1_left"], step["saved_observation"]["exterior_image_1_left"]))

    def test_reference_uses_side_right_wrist_left_without_flipping(self):
        step = example_step()
        before = copy.deepcopy(step)
        prepared = conversion.prepare_step_for_sft(step, mirror=False)
        obs = prepared["saved_observation"]
        np.testing.assert_array_equal(obs["exterior_image_1_left"], before["saved_observation"]["exterior_image_1_right"])
        np.testing.assert_array_equal(obs["exterior_image_2_left"], obs["exterior_image_1_left"])
        np.testing.assert_array_equal(obs["wrist_image_left"], before["saved_observation"]["wrist_image_left"])
        np.testing.assert_array_equal(obs["cartesian_position"], before["saved_observation"]["cartesian_position"])
        np.testing.assert_array_equal(prepared["action"]["cartesian_velocity"], before["action"]["cartesian_velocity"])
        for group in before:
            for key in before[group]:
                np.testing.assert_array_equal(step[group][key], before[group][key])

    def test_pose_and_rotation_action_match_geometric_reflection(self):
        step = example_step()
        mirrored = conversion.prepare_step_for_sft(step, mirror=True)
        reflection = np.diag([1, -1, 1])
        before = step["saved_observation"]["cartesian_position"]
        after = mirrored["saved_observation"]["cartesian_position"]
        np.testing.assert_allclose(after[:3], reflection @ before[:3])
        np.testing.assert_allclose(xyz_matrix(after[3:]), reflection @ xyz_matrix(before[3:]) @ reflection, atol=1e-7)
        before_action = step["action"]["cartesian_velocity"]
        after_action = mirrored["action"]["cartesian_velocity"]
        np.testing.assert_allclose(after_action[3:], -reflection @ before_action[3:])

    def test_tiny_hdf5_discovery_and_missing_stereo_validation(self):
        with tempfile.TemporaryDirectory(prefix="two-robot-conversion-") as directory:
            path = Path(directory) / "0" / "traj.hdf5"
            path.parent.mkdir()
            step = example_step()
            with h5py.File(path, "w") as file:
                for group, fields in step.items():
                    target = file.create_group(group)
                    for key, value in fields.items():
                        target.create_dataset(key, data=np.asarray(value)[None])
            self.assertEqual(conversion.find_episodes(directory), [path])
            self.assertEqual(conversion.validate_episode(path, mirror=False), 1)
            self.assertEqual(conversion.validate_episode(path, mirror=True), 1)
            with h5py.File(path, "a") as file:
                del file["saved_observation/wrist_image_right"]
            with self.assertRaisesRegex(ValueError, "wrist_image_right"):
                conversion.validate_episode(path, mirror=True)
            self.assertEqual(conversion.validate_episode(path, mirror=False), 1)

    def test_no_heavy_or_hardware_modules_imported(self):
        for name in ("torch", "jax", "lerobot", "pyzed", "droid", "pyspacemouse"):
            self.assertFalse(any(key == name or key.startswith(name + ".") for key in sys.modules), name)


if __name__ == "__main__":
    unittest.main()
