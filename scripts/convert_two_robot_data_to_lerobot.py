r"""Build five balanced LeRobot datasets from two robots' successful demonstrations.

Keep the original converter unchanged; reuse its HDF5 reader, image resizing,
action assembly, task configuration loader, and LeRobot feature conventions.
Only the selected robot is mirrored across base-frame y=0. This assumes matching
mirrored tool frames and horizontally mirrored side/wrist camera geometry.
Reference views: side right, wrist left. Mirrored views: side left, wrist right.
Output image names retain the original SFT schema, regardless of the input eye.

Example (in the existing SFT/LeRobot environment, from the repository root):
    python scripts/convert_two_robot_data_to_lerobot.py \
        --robot0-dir data/pick_cube_balance/robot0/success \
        --robot1-dir data/pick_cube_balance/robot1/success \
        --mirror-robot 1 --task-config configs/task/pick.py \
        --repo-prefix expo_ft/pick_mixed --seed 42

Outputs <repo-prefix>_20, _40, _60, _80, _100 under HF_LEROBOT_HOME.
The physical right-hand robot is not inferred from its ID: set --mirror-robot.
Existing output directories are refused. Raw HDF5 files are never modified.
Normalization and SFT are separate steps; neither is launched here.
"""

import argparse
import json
from pathlib import Path
import random
import sys

import h5py
import numpy as np

EPISODES_PER_ROBOT = (10, 20, 30, 40, 50)


def find_episodes(data_dir):
    """Use the original converter's recursive discovery and numeric episode order."""
    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        raise ValueError(f"Successful-demonstration directory not found: {data_dir}")

    def sort_key(path):
        name = path.parent.name
        return (int(name), str(path)) if name.isdigit() else (float("inf"), str(path))

    paths = []
    for path in sorted(data_dir.glob("**/traj.hdf5"), key=sort_key):
        try:
            with h5py.File(path, "r") as file:
                valid = "saved_observation" in file and "action" in file
                if valid:
                    valid = len(file["saved_observation/cartesian_position"]) > 0
        except (OSError, KeyError):
            valid = False
        if valid:
            paths.append(path.resolve())
        else:
            print(f"Skipping empty or unreadable episode: {path}")
    return paths


def select_episode_pools(robot0_paths, robot1_paths, seed):
    """Shuffle once per robot; all five datasets use prefixes of these same pools."""
    pools = []
    for robot_id, paths in enumerate((robot0_paths, robot1_paths)):
        if len(paths) < max(EPISODES_PER_ROBOT):
            raise ValueError(f"robot{robot_id}: need 50 usable episodes, found {len(paths)}")
        shuffled = list(paths)
        random.Random(seed + robot_id).shuffle(shuffled)
        pools.append(shuffled[:max(EPISODES_PER_ROBOT)])
    if set(pools[0]) & set(pools[1]):
        raise ValueError("The two robot inputs contain the same source episodes")
    return pools


def mixed_episodes(pools, count):
    """Alternate robots while retaining each original episode as a separate episode."""
    return [(robot_id, pools[robot_id][i]) for i in range(count) for robot_id in (0, 1)]


def validate_episode(path, mirror):
    """Check required HDF5 fields/shapes without loading image arrays."""
    image_keys = (
        ("exterior_image_1_left", "wrist_image_right") if mirror else
        ("exterior_image_1_right", "wrist_image_left")
    )
    with h5py.File(path, "r") as file:
        fields = [
            "saved_observation/cartesian_position", "saved_observation/gripper_position",
            "action/cartesian_velocity", "action/gripper_velocity",
            *(f"saved_observation/{key}" for key in image_keys),
        ]
        for key in fields:
            if key not in file:
                raise ValueError(f"{path}: missing required source view or field {key}")
        count = len(file[fields[0]])
        if count == 0 or any(len(file[key]) != count for key in fields):
            raise ValueError(f"{path}: empty or inconsistent step counts")
        for key in (fields[0], fields[2]):
            if file[key].shape != (count, 6):
                raise ValueError(f"{path}: {key} must contain six values per step")
        for key in image_keys:
            shape = file[f"saved_observation/{key}"].shape
            if len(shape) != 4 or shape[-1] != 3:
                raise ValueError(f"{path}: {key} must contain HWC RGB images")
    return count


def prepare_step_for_sft(step, mirror):
    """Return the SFT fields in the reference robot's frame; do not mutate input.

    Reference uses side right/wrist left; mirrored uses flipped side left/wrist right.
    Rotation uses the repository's xyz Euler convention with R' = S R S,
    S = diag(1, -1, 1). The gripper is unchanged. Transform before normalization.
    Only fields consumed by the original converter's Cartesian SFT path are returned.
    """
    obs = step["saved_observation"]
    action = step["action"]
    pose = np.array(obs["cartesian_position"], dtype=np.float32, copy=True)
    velocity = np.array(action["cartesian_velocity"], dtype=np.float32, copy=True)
    side = obs["exterior_image_1_left" if mirror else "exterior_image_1_right"]
    wrist = obs["wrist_image_right" if mirror else "wrist_image_left"]
    if mirror:
        pose[[1, 3, 5]] *= -1
        velocity[[1, 3, 5]] *= -1
        side = np.ascontiguousarray(side[:, ::-1, :])
        wrist = np.ascontiguousarray(wrist[:, ::-1, :])
    return {
        "saved_observation": {
            "exterior_image_1_left": side,
            "exterior_image_2_left": side,
            "wrist_image_left": wrist,
            "cartesian_position": pose,
            "gripper_position": obs["gripper_position"],
        },
        "action": {
            "cartesian_velocity": velocity,
            "gripper_velocity": action["gripper_velocity"],
        },
    }


def main(robot0_dir, robot1_dir, mirror_robot, task_config, repo_prefix, seed):
    if mirror_robot not in (0, 1):
        raise ValueError("mirror_robot must be 0 or 1")
    prefix = Path(repo_prefix)
    if prefix.is_absolute() or ".." in prefix.parts or len(prefix.parts) != 2:
        raise ValueError("repo_prefix must be a dataset ID such as expo_ft/pick_mixed")
    pools = select_episode_pools(find_episodes(robot0_dir), find_episodes(robot1_dir), seed)
    lengths = {
        path: validate_episode(path, mirror=(robot_id == mirror_robot))
        for robot_id, paths in enumerate(pools) for path in paths
    }

    # Load the existing conversion dependencies only for a real conversion.
    # --help and the small helper tests do not import LeRobot, Torch, or robot SDKs.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import convert_droid_data_to_lerobot as original

    task_cfg = original.load_task_config(task_config)
    if task_cfg.action_space != "cartesian_velocity" or task_cfg.gripper_action_space != "velocity":
        raise ValueError("This mirror conversion requires Cartesian velocity + gripper velocity actions")
    fps = task_cfg.control_hz
    if fps <= 0 or int(fps) != fps:
        raise ValueError("task control_hz must be a positive integer for LeRobot fps")
    fps = int(fps)
    language_instruction = task_cfg.language_instruction
    action_key = task_cfg.action_space
    gripper_key = f"gripper_{task_cfg.gripper_action_space}"
    action_dim = 7
    repo_names = [f"{repo_prefix}_{2 * count}" for count in EPISODES_PER_ROBOT]
    for repo_name in repo_names:
        output_path = original.HF_LEROBOT_HOME / repo_name
        if output_path.exists():
            raise FileExistsError(f"Output already exists; choose a new --repo-prefix: {output_path}")

    for count, repo_name in zip(EPISODES_PER_ROBOT, repo_names):
        episode_paths = mixed_episodes(pools, count)
        print(f"Creating {repo_name}: {count} episodes per robot, fps={fps}")
        # Same Cartesian-state feature schema and writer options as the original converter.
        dataset = original.LeRobotDataset.create(
            repo_id=repo_name,
            robot_type="panda",
            fps=fps,
            features={
                "exterior_image_1_left": {
                    "dtype": "image", "shape": (180, 320, 3),
                    "names": ["height", "width", "channel"],
                },
                "exterior_image_2_left": {
                    "dtype": "image", "shape": (180, 320, 3),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image_left": {
                    "dtype": "image", "shape": (180, 320, 3),
                    "names": ["height", "width", "channel"],
                },
                "cartesian_position": {
                    "dtype": "float32", "shape": (6,), "names": ["cartesian_position"],
                },
                "gripper_position": {
                    "dtype": "float32", "shape": (1,), "names": ["gripper_position"],
                },
                "actions": {
                    "dtype": "float32", "shape": (action_dim,), "names": ["actions"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )
        try:
            manifest = {
                "repo_id": repo_name,
                "episodes_per_robot": count,
                "shuffle_seeds": {"robot0": seed, "robot1": seed + 1},
                "mirror_robot": mirror_robot,
                "mirror": "y reflection; xyz Euler; horizontal flip of selected side and wrist views",
                "image_sources": {
                    f"robot{robot_id}": {
                        "side": "exterior_image_1_left" if robot_id == mirror_robot else "exterior_image_1_right",
                        "wrist": "wrist_image_right" if robot_id == mirror_robot else "wrist_image_left",
                        "horizontal_flip": robot_id == mirror_robot,
                    }
                    for robot_id in (0, 1)
                },
                "fps": fps,
                "task_config": str(Path(task_config).resolve()),
                "episodes": [],
            }
            for episode_index, (robot_id, episode_path) in enumerate(episode_paths):
                print(f"processing robot{robot_id} traj: {episode_path}")
                trajectory = original.load_trajectory(str(episode_path), read_cameras=False)
                if len(trajectory) != lengths[episode_path]:
                    raise ValueError(f"Episode changed since validation: {episode_path}")
                for step in trajectory:
                    step = prepare_step_for_sft(step, mirror=(robot_id == mirror_robot))
                    # Same image/state/action assembly as the original Cartesian converter.
                    dataset.add_frame({
                        "exterior_image_1_left": original.resize_image(
                            step["saved_observation"]["exterior_image_1_left"], (320, 180)),
                        "exterior_image_2_left": original.resize_image(
                            step["saved_observation"]["exterior_image_2_left"], (320, 180)),
                        "wrist_image_left": original.resize_image(
                            step["saved_observation"]["wrist_image_left"], (320, 180)),
                        "cartesian_position": np.asarray(
                            step["saved_observation"]["cartesian_position"], dtype=np.float32),
                        "gripper_position": np.asarray(
                            step["saved_observation"]["gripper_position"][None], dtype=np.float32),
                        "actions": original._action_from_step(step["action"], action_key, gripper_key),
                        "task": language_instruction,
                    })
                dataset.save_episode()
                manifest["episodes"].append({
                    "episode_index": episode_index,
                    "robot_id": robot_id,
                    "source_hdf5": str(episode_path),
                    "mirrored": robot_id == mirror_robot,
                    "frames": len(trajectory),
                })
            manifest_path = original.HF_LEROBOT_HOME / repo_name / "meta" / "source_episodes.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Saved {repo_name}; source episode manifest: {manifest_path}")
        finally:
            dataset.stop_image_writer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot0-dir", required=True, help="robot0 successful-episode directory")
    parser.add_argument("--robot1-dir", required=True, help="robot1 successful-episode directory")
    parser.add_argument("--mirror-robot", type=int, choices=(0, 1), required=True,
                        help="Physical robot whose data should be mirrored into the other robot's frame")
    parser.add_argument("--task-config", default="configs/task/pick.py")
    parser.add_argument("--repo-prefix", default="expo_ft/pick_mixed")
    parser.add_argument("--seed", type=int, default=42)
    main(**vars(parser.parse_args()))
