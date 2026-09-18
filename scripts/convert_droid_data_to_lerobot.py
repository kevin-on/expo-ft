"""
Minimal example script for converting a dataset collected on the DROID platform to LeRobot format.

Language instruction is read from the task config (config.language_instruction), not passed as a CLI argument.

Usage:
uv run scripts/expo_pi/real_utils/convert_droid_data_to_lerobot.py \\
    --data_dir /path/to/your/data --repo_name my/repo --task_config configs/task/pick.py

The resulting dataset will get saved to the $LEROBOT_HOME directory.
"""

from collections import defaultdict
import copy
import glob
import json
import random
import sys
from pathlib import Path
import shutil

# Add project root so configs.task.* can be imported when run as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # expo_ft -> scripts -> repo root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro

def resize_image(image, size):
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def _action_from_step(action_dict: dict, action_key: str, gripper_key: str) -> np.ndarray:
    """Build concatenated action array from step['action'] using config keys."""
    arm = np.asarray(action_dict[action_key], dtype=np.float32).flatten()
    grip = np.asarray(action_dict[gripper_key], dtype=np.float32)
    grip = np.atleast_1d(grip)
    return np.concatenate([arm, grip], axis=-1).astype(np.float32)


def load_task_config(config_path: str):
    """Load task config from a FILE PATH (or a dotted module path).

    Returns the full config (action_space, gripper_action_space,
    language_instruction).

    A .py path is loaded by location, not turned into a dotted module name: task config
    file names carry speed factors ("hang0.5speed.py") and a dot in the stem reads as a
    package separator to __import__, which cannot import them at all.
    """
    import importlib.util
    import os

    if config_path.endswith(".py"):
        path = os.path.abspath(config_path)
        if not os.path.exists(path):
            raise ImportError(f"Task config '{config_path}' does not exist (looked at {path})")
        # A name, not a module path: the stem may hold dots, and nothing imports it back.
        spec = importlib.util.spec_from_file_location(
            f"_task_config_{os.path.splitext(os.path.basename(path))[0]}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = __import__(config_path.replace("/", "."), fromlist=["get_config"])
    return module.get_config()


def main(
    data_dir: str,
    *,
    repo_name: str,
    task_config: str,
    max_episodes: int | None = None,
    push_to_hub: bool = False,
    use_cartesian_state: bool = False,
):
    """
    Convert DROID-style trajectories to LeRobot.

    The dataset is saved under HF_LEROBOT_HOME / repo_name.
    Language instruction is read from task_config.language_instruction.
    use_cartesian_state: if True, save cartesian_position (6D) as state; else save joint_position (7D).
    """
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    data_dir = Path(data_dir)
    task_cfg = load_task_config(task_config)
    language_instruction = task_cfg.language_instruction
    action_key = task_cfg.action_space
    gripper_key = f"gripper_{task_cfg.gripper_action_space}"
    action_dim = 7 if action_key == "cartesian_velocity" else 8  # 6+1 or 7+1

    # Create LeRobot dataset, define features to store
    # We will follow the DROID data naming conventions here.
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=15,  # DROID data is typically recorded at 15fps
        features={
            # We call this "left" since we will only use the left stereo camera (following DROID RLDS convention)
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (180, 320, 3),  # This is the resolution used in the DROID RLDS dataset
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            **(
                {
                    "cartesian_position": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["cartesian_position"],
                    },
                }
                if use_cartesian_state
                else {
                    "joint_position": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["joint_position"],
                    },
                }
            ),
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw DROID fine-tuning datasets and write episodes to the LeRobot dataset
    # We assume the following directory structure:
    # RAW_DROID_PATH/
    #   - <...>/
    #     - recordings/
    #        - MP4/
    #          - <camera_id>.mp4  # single-view video of left stereo pair camera
    #     - trajectory.hdf5
    #   - <...>/
    def _episode_sort_key(p):
        name = p.parent.name
        return (int(name),) if name.isdigit() else (float("inf"), name)

    def _has_trajectory(p):
        try:
            with h5py.File(p, "r") as f:
                return "saved_observation" in f and "action" in f
        except OSError:
            return False

    episode_paths = sorted(data_dir.glob("**/traj.hdf5"), key=_episode_sort_key)
    # Aborted recordings leave an empty traj.hdf5; drop them before max_episodes slicing.
    skipped = [p for p in episode_paths if not _has_trajectory(p)]
    if skipped:
        print(f"Skipping {len(skipped)} empty episodes: {[str(p.parent) for p in skipped]}")
        episode_paths = [p for p in episode_paths if p not in set(skipped)]
    # random.shuffle(episode_paths)

    if max_episodes is not None:
        episode_paths = episode_paths[:max_episodes]
        print(f"Using {len(episode_paths)} episodes (max_episodes={max_episodes})")
    else:
        print(f"Found {len(episode_paths)} episodes for conversion")

    # We will loop over each dataset_name and write episodes to the LeRobot dataset
    for episode_path in tqdm(episode_paths, desc="Converting episodes"):
        # Load from HDF5 only; images come from saved_observation, not from MP4 recordings
        print(f"processing traj: {episode_path}")
        trajectory = load_trajectory(str(episode_path))

        # Language instruction comes from task config
        # print(f"Converting episode with language instruction: {language_instruction}")

        # Write to LeRobot dataset
        if len(trajectory) == 0:
            print(f"Skipping empty trajectory: {episode_path}")
            continue

        for step in trajectory:
            dataset.add_frame(
                {
                    "exterior_image_1_left": resize_image(
                        step["saved_observation"]["exterior_image_1_left"], (320, 180)
                    ),
                    "exterior_image_2_left": resize_image(
                        step["saved_observation"]["exterior_image_2_left"], (320, 180)
                    ),
                    "wrist_image_left": resize_image(step["saved_observation"]["wrist_image_left"], (320, 180)),
                    **(
                        {
                            "cartesian_position": np.asarray(
                                step["saved_observation"]["cartesian_position"], dtype=np.float32
                            ),
                        }
                        if use_cartesian_state
                        else {
                            "joint_position": np.asarray(
                                step["saved_observation"]["joint_position"], dtype=np.float32
                            ),
                        }
                    ),
                    "gripper_position": np.asarray(
                        step["saved_observation"]["gripper_position"][None], dtype=np.float32
                    ),
                    "actions": _action_from_step(step["action"], action_key, gripper_key),
                    "task": language_instruction,
                }
            )
            
        dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

##########################################################################################################
################ The rest of this file are functions to parse the raw DROID data #########################
################ You don't need to worry about understanding this part           #########################
################ It was copied from here: https://github.com/JonathanYang0127/r2d2_rlds_dataset_builder/blob/parallel_convert/r2_d2/r2_d2.py
##########################################################################################################


camera_type_dict = {
    "hand_camera_id": 0,
    "varied_camera_1_id": 1,
    "varied_camera_2_id": 1,
}

camera_type_to_string_dict = {
    0: "hand_camera",
    1: "varied_camera",
    2: "fixed_camera",
}


def get_camera_type(cam_id):
    if cam_id not in camera_type_dict:
        return None
    type_int = camera_type_dict[cam_id]
    return camera_type_to_string_dict[type_int]


class MP4Reader:
    def __init__(self, filepath, serial_number):
        # Save Parameters #
        self.serial_number = serial_number
        self._index = 0

        # Open Video Reader #
        self._mp4_reader = cv2.VideoCapture(filepath)
        if not self._mp4_reader.isOpened():
            raise RuntimeError("Corrupted MP4 File")

    def set_reading_parameters(
        self,
        image=True,  # noqa: FBT002
        concatenate_images=False,  # noqa: FBT002
        resolution=(0, 0),
        resize_func=None,
    ):
        # Save Parameters #
        self.image = image
        self.concatenate_images = concatenate_images
        self.resolution = resolution
        self.resize_func = cv2.resize
        self.skip_reading = not image
        if self.skip_reading:
            return

    def set_frame_index(self, index):
        if self.skip_reading:
            return

        if index < self._index:
            self._mp4_reader.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
            self._index = index

        while self._index < index:
            self.read_camera(ignore_data=True)

    def _process_frame(self, frame):
        frame = copy.deepcopy(frame)
        if self.resolution == (0, 0):
            return frame
        return self.resize_func(frame, self.resolution)

    def read_camera(self, ignore_data=False, correct_timestamp=None):  # noqa: FBT002
        # Skip if Read Unnecesary #
        if self.skip_reading:
            return {}

        # Read Camera #
        success, frame = self._mp4_reader.read()

        self._index += 1
        if not success:
            return None
        if ignore_data:
            return None

        # Return Data #
        data_dict = {}

        if self.concatenate_images or "stereo" not in self.serial_number:
            data_dict["image"] = {self.serial_number: self._process_frame(frame)}
        else:
            single_width = frame.shape[1] // 2
            data_dict["image"] = {
                self.serial_number + "_left": self._process_frame(frame[:, :single_width, :]),
                self.serial_number + "_right": self._process_frame(frame[:, single_width:, :]),
            }

        return data_dict

    def disable_camera(self):
        if hasattr(self, "_mp4_reader"):
            self._mp4_reader.release()


class RecordedMultiCameraWrapper:
    def __init__(self, recording_folderpath, camera_kwargs={}):  # noqa: B006
        # Save Camera Info #
        self.camera_kwargs = camera_kwargs

        # Open Camera Readers #
        mp4_filepaths = glob.glob(recording_folderpath + "/*.mp4")
        all_filepaths = mp4_filepaths

        self.camera_dict = {}
        for f in all_filepaths:
            serial_number = f.split("/")[-1][:-4]
            cam_type = get_camera_type(serial_number)
            camera_kwargs.get(cam_type, {})

            if f.endswith(".mp4"):
                Reader = MP4Reader  # noqa: N806
            else:
                raise ValueError

            self.camera_dict[serial_number] = Reader(f, serial_number)

    def read_cameras(self, index=None, camera_type_dict={}, timestamp_dict={}):  # noqa: B006
        full_obs_dict = defaultdict(dict)

        # Read Cameras In Randomized Order #
        all_cam_ids = list(self.camera_dict.keys())
        # random.shuffle(all_cam_ids)

        for cam_id in all_cam_ids:
            if "stereo" in cam_id:
                continue
            try:
                cam_type = camera_type_dict[cam_id]
            except KeyError:
                print(f"{self.camera_dict} -- {camera_type_dict}")
                raise ValueError(f"Camera type {cam_id} not found in camera_type_dict")  # noqa: B904
            curr_cam_kwargs = self.camera_kwargs.get(cam_type, {})
            self.camera_dict[cam_id].set_reading_parameters(**curr_cam_kwargs)

            timestamp = timestamp_dict.get(cam_id + "_frame_received", None)
            if index is not None:
                self.camera_dict[cam_id].set_frame_index(index)

            data_dict = self.camera_dict[cam_id].read_camera(correct_timestamp=timestamp)

            # Process Returned Data #
            if data_dict is None:
                return None
            for key in data_dict:
                full_obs_dict[key].update(data_dict[key])

        return full_obs_dict


def get_hdf5_length(hdf5_file, keys_to_ignore=[]):  # noqa: B006
    length = None

    for key in hdf5_file:
        if key in keys_to_ignore:
            continue

        curr_data = hdf5_file[key]
        if isinstance(curr_data, h5py.Group):
            curr_length = get_hdf5_length(curr_data, keys_to_ignore=keys_to_ignore)
        elif isinstance(curr_data, h5py.Dataset):
            curr_length = len(curr_data)
        else:
            raise ValueError

        if length is None:
            length = curr_length
        assert curr_length == length

    return length


def load_hdf5_to_dict(hdf5_file, index, keys_to_ignore=[]):  # noqa: B006
    data_dict = {}

    for key in hdf5_file:
        if key in keys_to_ignore:
            continue

        curr_data = hdf5_file[key]
        if isinstance(curr_data, h5py.Group):
            data_dict[key] = load_hdf5_to_dict(curr_data, index, keys_to_ignore=keys_to_ignore)
        elif isinstance(curr_data, h5py.Dataset):
            data_dict[key] = curr_data[index]
        else:
            raise ValueError

    return data_dict


def load_hdf5_bulk(hdf5_file, keys_to_ignore=[]):  # noqa: B006
    """Read all datasets from an HDF5 file/group at once (single I/O per dataset)."""
    data_dict = {}
    for key in hdf5_file:
        if key in keys_to_ignore:
            continue
        curr_data = hdf5_file[key]
        if isinstance(curr_data, h5py.Group):
            data_dict[key] = load_hdf5_bulk(curr_data, keys_to_ignore=keys_to_ignore)
        elif isinstance(curr_data, h5py.Dataset):
            data_dict[key] = curr_data[:]
        else:
            raise ValueError
    return data_dict


def bulk_index(bulk_data, index):
    """Index into a bulk-loaded nested dict to get a single timestep."""
    out = {}
    for key, val in bulk_data.items():
        if isinstance(val, dict):
            out[key] = bulk_index(val, index)
        else:
            out[key] = val[index]
    return out

def load_trajectory(
    filepath=None,
    read_cameras=True,  # noqa: FBT002
    recording_folderpath=None,
    camera_kwargs={},  # noqa: B006
    remove_skipped_steps=False,  # noqa: FBT002
    num_samples_per_traj=None,
    num_samples_per_traj_coeff=1.5,
):
    read_recording_folderpath = read_cameras and (recording_folderpath is not None)

    hdf5_file = h5py.File(filepath, "r")
    horizon = get_hdf5_length(hdf5_file)

    if read_recording_folderpath:
        camera_reader = RecordedMultiCameraWrapper(recording_folderpath, camera_kwargs)

    # Choose Timesteps To Save #
    if num_samples_per_traj:
        num_to_save = num_samples_per_traj
        if remove_skipped_steps:
            num_to_save = int(num_to_save * num_samples_per_traj_coeff)
        max_size = min(num_to_save, horizon)
        indices_to_save = np.sort(np.random.choice(horizon, size=max_size, replace=False))
    else:
        indices_to_save = np.arange(horizon)

    # Only read the keys we actually use — skip observation/image/ (full-res, ~4 GB)
    bulk_data = load_hdf5_bulk(hdf5_file, keys_to_ignore=["videos", "image"])
    hdf5_file.close()

    timestep_list = []

    # Iterate Over Trajectory #
    for i in indices_to_save:
        # Get HDF5 Data #
        timestep = bulk_index(bulk_data, i)

        # If Applicable, Get Recorded Data #
        if read_recording_folderpath:
            timestamp_dict = timestep["observation"]["timestamp"]["cameras"]
            camera_type_dict = {
                k: camera_type_to_string_dict[v] for k, v in timestep["observation"]["camera_type"].items()
            }
            camera_obs = camera_reader.read_cameras(
                index=i, camera_type_dict=camera_type_dict, timestamp_dict=timestamp_dict
            )
            camera_failed = camera_obs is None

            # Add Data To Timestep If Successful #
            if camera_failed:
                break
            timestep["observation"].update(camera_obs)

        # Filter Steps #
        step_skipped = not timestep.get("observation", {}).get("controller_info", {}).get("movement_enabled", True)
        delete_skipped_step = step_skipped and remove_skipped_steps

        # Save Filtered Timesteps #
        if delete_skipped_step:
            del timestep
        else:
            timestep_list.append(timestep)

    # Remove Extra Transitions #
    timestep_list = np.array(timestep_list)
    if (num_samples_per_traj is not None) and (len(timestep_list) > num_samples_per_traj):
        ind_to_keep = np.random.choice(len(timestep_list), size=num_samples_per_traj, replace=False)
        timestep_list = timestep_list[ind_to_keep]

    # Return Data #
    return timestep_list


if __name__ == "__main__":
    tyro.cli(main)
