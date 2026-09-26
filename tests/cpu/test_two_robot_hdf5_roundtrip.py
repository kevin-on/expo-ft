"""Tiny integration test using real LeRobot/Parquet writers; no hardware or GPU.

Run separately in the SFT environment, with BLAS/OMP threads limited to one:
    python tests/cpu/test_two_robot_hdf5_roundtrip.py
The test exercises asymmetric counts and saved permutation order.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import convert_two_robot_data_to_lerobot as conversion
from scripts.convert_lerobot_to_hdf5 import convert
from test_two_robot_conversion import example_step


class TwoRobotHdf5RoundtripTests(unittest.TestCase):
    def test_direct_hdf5_matches_lerobot_export(self):
        import torch
        from scripts import convert_droid_data_to_lerobot as original

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="two-robot-roundtrip-") as directory:
            root = Path(directory)
            for robot in (0, 1):
                for episode in range(2):
                    path = root / f"robot{robot}/{episode}/traj.hdf5"
                    path.parent.mkdir(parents=True)
                    steps = []
                    for index in range(2 + episode + robot):
                        step = example_step()
                        for key, value in step["saved_observation"].items():
                            if "image" in key:
                                step["saved_observation"][key] = value + robot * 20 + episode * 5 + index * 3
                        step["saved_observation"]["cartesian_position"] += robot + index * .125
                        step["saved_observation"]["gripper_position"] = np.float32(.2 + index * .3)
                        step["action"]["cartesian_velocity"] += robot + index * .25
                        step["action"]["gripper_velocity"] = np.float32(-1 if index == 0 else 1)
                        steps.append(step)
                    with h5py.File(path, "w") as file:
                        for group, fields in steps[0].items():
                            for key in fields:
                                file.create_dataset(f"{group}/{key}", data=np.stack([s[group][key] for s in steps]))

            task = root / "task.py"
            task.write_text('from types import SimpleNamespace\ndef get_config():\n'
                            '    return SimpleNamespace(action_space="cartesian_velocity", '
                            'gripper_action_space="velocity", control_hz=10, language_instruction="test pick")\n')
            lerobot_root = root / "lerobot"
            create = original.LeRobotDataset.create

            def create_small(**kwargs):
                kwargs.update(root=lerobot_root / kwargs["repo_id"], image_writer_processes=0,
                              image_writer_threads=1)
                return create(**kwargs)

            selection = root / "selection.json"
            selection.write_text(json.dumps({"mirror_robot": 1, "shuffle_seeds": {"robot0": 3, "robot1": 4},
                "ordered_top50": {f"robot{r}": [f"/old/robot{r}/{i}/traj.hdf5" for i in (1, 0)] for r in (0, 1)}}))
            for robot1_count in (0, 1, 2):
                count = 2 + robot1_count
                repo_id = f"test/tiny_{count}"
                with patch.object(original, "HF_LEROBOT_HOME", lerobot_root), \
                     patch.object(original.LeRobotDataset, "create", side_effect=create_small):
                    conversion.main(root / "robot0", root / "robot1", 1, str(task), repo_id,
                                    selection, 2, robot1_count, hdf5_root=root / "direct")
                dataset = lerobot_root / repo_id
                manifest = json.loads((dataset / "meta/source_episodes.json").read_text())
                report = convert(dataset, root / f"exported_{count}")
                self.assertEqual(report["episodes"], count)
                self.assertEqual(report["frames"], sum(e["frames"] for e in manifest["episodes"]))
                expected_robots = {0: [0, 0], 1: [0, 1, 0], 2: [0, 1, 0, 1]}[robot1_count]
                self.assertEqual([e["robot_id"] for e in manifest["episodes"]], expected_robots)
                self.assertEqual([e["source_episode"] for e in manifest["episodes"] if e["robot_id"] == 0], [1, 0])
                self.assertEqual([e["mirrored"] for e in manifest["episodes"]], [r == 1 for r in expected_robots])
                expected_keys = {
                    "saved_observation/exterior_image_1_left", "saved_observation/exterior_image_2_left",
                    "saved_observation/wrist_image_left", "saved_observation/cartesian_position",
                    "saved_observation/gripper_position", "action/cartesian_velocity", "action/gripper_velocity",
                }
                for index in range(count):
                    with h5py.File(root / f"direct/test/tiny_{count}/{index}/traj.hdf5") as direct, \
                         h5py.File(root / f"exported_{count}/{index}/traj.hdf5") as exported:
                        self.assertEqual(dict(direct.attrs), dict(exported.attrs))
                        keys = set()
                        direct.visititems(lambda key, obj: keys.add(key) if isinstance(obj, h5py.Dataset) else None)
                        self.assertEqual(keys, expected_keys)
                        export_keys = set()
                        exported.visititems(lambda key, obj: export_keys.add(key) if isinstance(obj, h5py.Dataset) else None)
                        self.assertEqual(export_keys, keys)
                        for key in sorted(keys):
                            self.assertEqual(direct[key].dtype, exported[key].dtype)
                            np.testing.assert_array_equal(direct[key][:], exported[key][:], err_msg=f"episode {index}: {key}")

            print("Verified single-robot, asymmetric and balanced datasets: direct and LeRobot-exported HDF5 agree exactly.")


if __name__ == "__main__":
    unittest.main()
