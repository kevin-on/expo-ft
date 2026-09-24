"""Tiny integration test using real LeRobot/Parquet writers; no hardware or GPU.

Run separately in the SFT environment, with BLAS/OMP threads limited to one:
    python tests/cpu/test_two_robot_hdf5_roundtrip.py
Only the test patches the episode counts and LeRobot image-writer concurrency.
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

            # Exercise the real conversion loop without changing production dataset sizes.
            with patch.object(conversion, "EPISODES_PER_ROBOT", (1, 2)), \
                 patch.object(original, "HF_LEROBOT_HOME", lerobot_root), \
                 patch.object(original.LeRobotDataset, "create", side_effect=create_small):
                conversion.main(root / "robot0", root / "robot1", 1, str(task), "test/tiny", 3,
                                hdf5_root=root / "direct")

            for count in (2, 4):
                dataset = lerobot_root / f"test/tiny_{count}"
                manifest = json.loads((dataset / "meta/source_episodes.json").read_text())
                report = convert(dataset, root / f"exported_{count}")
                self.assertEqual(report["episodes"], count)
                self.assertEqual(report["frames"], sum(e["frames"] for e in manifest["episodes"]))
                self.assertEqual([e["robot_id"] for e in manifest["episodes"]], [0, 1] * (count // 2))
                self.assertEqual([e["mirrored"] for e in manifest["episodes"]], [False, True] * (count // 2))
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

            print("Verified both nested datasets: direct and LeRobot-exported HDF5 agree exactly.")


if __name__ == "__main__":
    unittest.main()
