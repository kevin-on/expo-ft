import json
import os
import shutil
import select
import termios
import time
import tty
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Dict

import cv2
import h5py
import imageio
import numpy as np
from absl import app, flags
from ml_collections import config_flags
from droid.misc.time import time_ms

from client.real_utils.spacemouse import SpaceMousePolicy
from client.real_utils.vertical_control import vertical_orientation, vertical_velocity_action
from client.envs.utils import process_image_for_obs
from droid.trajectory_utils.trajectory_writer import write_dict_to_hdf5

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "save_root",
    "data/pick_and_place_cube",
    "Root directory where collected trajectories will be stored.",
)
flags.DEFINE_integer(
    "num_episodes",
    0,
    "Number of successful trajectories to collect. Set to 0 to run indefinitely.",
)
config_flags.DEFINE_config_file(
    "task_config",
    "configs/task/pick.py",
    "File path to the task configuration.",
    lock_config=False,
)
flags.DEFINE_string(
    "robot_config",
    None,
    "Robot JSON overriding the task's server, camera, and SpaceMouse settings.",
)
flags.DEFINE_bool(
    "save_right_images",
    True,
    "Also save the side and wrist right views in HDF5 and MP4 (collection only).",
)
flags.DEFINE_bool(
    "keep_vertical",
    False,
    "After each reset, actively keep tool +Z toward base -Z using the post-reset yaw; ignore mouse rotation.",
)
flags.DEFINE_alias("keep-vertical", "keep_vertical")
flags.DEFINE_bool(
    "test_detector",
    False,
    "If True, never return when done; reset env and keep the collection loop running (for testing the detector).",
)
# Saved MP4 encoder CPU budget (per writer).
flags.DEFINE_integer("video_encoder_threads", 2, "CPU encoding threads per MP4 writer.")
flags.DEFINE_alias("video-encoder-threads", "video_encoder_threads")
# Saved MP4 resolution (width, height); reuse resized HDF5 images.
flags.DEFINE_integer("video_save_width", 320, "Width of saved MP4 frames.")
flags.DEFINE_integer("video_save_height", 180, "Height of saved MP4 frames.")


class CollectionKeys:
    """Single owner of collection terminal input; preserve Ctrl+C and restore tty."""

    def __enter__(self):
        self.fd = None
        self.saved = None
        self.escape = None
        try:
            self.fd = os.open('/dev/tty', os.O_RDONLY | os.O_NONBLOCK)
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd, termios.TCSANOW)
        except (OSError, termios.error):
            self.__exit__(None, None, None)
            print('[collection] No controlling terminal; discard hotkey unavailable.')
        else:
            print('[collection] D: discard current rollout and reset | 1: success | 2: discard/reset | Ctrl+C: exit (no Enter needed)')
        return self

    def __exit__(self, *_):
        if self.fd is not None:
            try:
                if self.saved is not None:
                    termios.tcsetattr(self.fd, termios.TCSANOW, self.saved)
            finally:
                os.close(self.fd)
                self.fd = None

    def clear(self):
        self.escape = None
        if self.fd is not None:
            termios.tcflush(self.fd, termios.TCIFLUSH)

    def poll(self):
        if self.fd is None or not select.select([self.fd], [], [], 0)[0]:
            return None
        try:
            keys = os.read(self.fd, 4096).lower()
        except BlockingIOError:
            return None
        plain = bytearray()
        for key in keys:
            if self.escape == 'start':
                self.escape = 'sequence' if key in (ord('['), ord('o')) else None
            elif self.escape == 'sequence':
                if 0x40 <= key <= 0x7e:
                    self.escape = None
            elif key == 27:
                self.escape = 'start'
            else:
                plain.append(key)
        # Ignore arrow/function escape sequences (Left ends in D!). Discard
        # takes precedence over success, even when both arrive in one read.
        if b'd' in plain or b'2' in plain:
            return 'discard'
        if b'1' in plain:
            return 'success'
        return None  # Enter / 3 / unrelated keys never enter a blocking prompt.


def collection_observation(env, raw_obs, save_right_images):
    """Save physical left/right lenses even when the policy selects a right view."""
    saved_obs = env.transform_observation(raw_obs)
    if save_right_images:
        for camera_id, output_prefix in (
            (env.side_camera_id, "exterior_image_1"),
            (env.wrist_camera_id, "wrist_image"),
        ):
            serial = camera_id.rsplit("_", 1)[0]
            selected_image = saved_obs[f"{output_prefix}_left"]
            for eye in ("left", "right"):
                saved_obs[f"{output_prefix}_{eye}"] = (
                    selected_image if camera_id == f"{serial}_{eye}" else
                    process_image_for_obs(raw_obs["image"][f"{serial}_{eye}"],
                                          bgr_to_rgb=True, image_size=env.image_size)
                )
        saved_obs["exterior_image_2_left"] = saved_obs["exterior_image_1_left"]
    return saved_obs


class CollectionRecorder:
    """One ordered worker owns image transforms, HDF5 and MP4 for an episode."""

    def __init__(self, env, filepath, recording_folderpath, save_right_images,
                 video_size, max_pending=8, video_encoder_threads=2):
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.env = env
        self.filepath = filepath
        self.video_dir = os.path.join(os.path.dirname(recording_folderpath), "recordings", "MP4")
        self.save_right_images = save_right_images
        if video_encoder_threads < 1:
            raise ValueError("video_encoder_threads must be positive")
        self.video_encoder_threads = video_encoder_threads
        self.video_size = video_size
        self.max_pending = max_pending
        self.pending = deque()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="collect-recorder")
        self.hdf5 = None
        self.videos = {}
        self.error = None

    def check(self):
        while self.pending and self.pending[0].done():
            self.pending.popleft().result()

    def submit(self, obs, action):
        self.check()
        if len(self.pending) >= self.max_pending:
            # Backpressure instead of dropping paired samples or growing RAM forever.
            print("[recording] queue full; waiting for storage worker")
            self.pending[0].result()
            self.pending.popleft()
        # Freeze this step before camera/controller buffers can be reused.
        obs, action = deepcopy((obs, action))
        self.pending.append(self.executor.submit(self._write, obs, action))

    def _write(self, obs, action):
        if self.error is not None:
            raise self.error
        try:
            saved_obs = collection_observation(self.env, obs, self.save_right_images)
            if self.hdf5 is None:
                self.hdf5 = h5py.File(self.filepath, "x")
                os.makedirs(self.video_dir, exist_ok=True)
                for key, val in saved_obs.items():
                    if isinstance(val, np.ndarray) and val.ndim == 3 and val.shape[-1] == 3:
                        self.videos[key] = imageio.get_writer(
                            os.path.join(self.video_dir, f"{key}.mp4"),
                            fps=30, format="ffmpeg", codec="libx264", macro_block_size=1,
                            output_params=["-preset", "ultrafast", "-crf", "28",
                                           "-threads", str(self.video_encoder_threads)],
                        )
            # Use the existing HDF5 schema/serializer, synchronously in this worker.
            # No second unbounded writer queue; write failures reach the control loop.
            write_dict_to_hdf5(self.hdf5, {"saved_observation": saved_obs, "action": action})
            for key, writer in self.videos.items():
                frame = np.asarray(saved_obs[key], dtype=np.uint8)
                if (frame.shape[1], frame.shape[0]) != self.video_size:
                    frame = cv2.resize(frame, self.video_size, interpolation=cv2.INTER_LINEAR)
                writer.append_data(frame)
        except BaseException as exc:
            self.error = exc
            raise

    def _close(self, metadata):
        error = self.error
        try:
            if self.hdf5 is not None:
                for key, value in metadata.items():
                    self.hdf5.attrs[key] = value
        except BaseException as exc:
            error = error or exc
        finally:
            for writer in self.videos.values():
                try:
                    writer.close()
                except BaseException as exc:
                    error = error or exc
            if self.hdf5 is not None:
                try:
                    self.hdf5.close()
                except BaseException as exc:
                    error = error or exc
        if error is not None:
            raise error

    def close(self, metadata):
        try:
            # FIFO executor: every accepted sample finishes before files close/reset.
            self.executor.submit(self._close, deepcopy(metadata)).result()
        finally:
            self.executor.shutdown(wait=True)
            self.pending.clear()


def smallest_missing_id(dir_path: str) -> int:
    os.makedirs(dir_path, exist_ok=True)
    ids = set()
    for name in os.listdir(dir_path):
        p = os.path.join(dir_path, name)
        if os.path.isdir(p) and name.isdigit():
            ids.add(int(name))
    i = 0
    while i in ids:
        i += 1
    return i


def collect_trajectory(
    env,
    controller,
    save_filepath=None,
    recording_folderpath=False,
    test_detector=False,
    keep_vertical=False,
    keys=None,
):
    controller.reset_state()
    env.camera_reader.set_trajectory_mode()

    recorder = None

    t_reset0 = time.perf_counter()
    env.reset()
    if keys is not None:
        keys.clear()  # Ignore commands typed during reset / previous file finalization.
    print("[between-episode] env.reset()={:.2f}s (start of episode)".format(time.perf_counter() - t_reset0))
    vertical_target = None  # Re-anchor yaw from this episode's post-reset observation.

    start_recording = False
    last_control_start = None
    recorded_steps = 0
    first_recorded_step_time = None
    last_recorded_step_time = None

    try:
        while True:
            manual = keys.poll() if keys is not None else None
            if manual == 'discard':
                return {'success': False, 'failure': False, 'discarded': True}
            if recorder is not None:
                recorder.check()
            time_start = time_ms()
            controller_info = controller.get_info()
            control_timestamps = {"step_start": time_ms()}
            t_after_controller = time_ms()

            read_camera_start = time_ms()
            obs = env.get_raw_observation()
            read_camera_end = time_ms()
            if keys is not None:
                latest = keys.poll()
                if latest == 'discard':
                    return {'success': False, 'failure': False, 'discarded': True}
                manual = latest or manual
                # Collection owns tty input. Do not let the normal manual detector
                # race this reader or block on readline while tty is in cbreak mode.
                done, success, _, _ = env.get_info_for_step(obs, manual_override=manual or 'keep_going')
            else:
                done, success, _, _ = env.get_info_for_step(obs)
            t_after_obs = time_ms()

            # Return as soon as done is detected -- don't compute/send one more action
            # (e.g. a gripper release) to the robot after the episode is already over.
            if done:
                if test_detector:
                    print(f"Done (success={success}); test_detector=True")
                    continue
                result = {
                    "success": success,
                    "failure": not success,
                    "info": controller_info,
                }
                return result

            obs["controller_info"] = controller_info

            control_timestamps["policy_start"] = time_ms()
            action, controller_action_info = controller.forward(obs, include_info=True)
            if keep_vertical:
                orientation = obs["robot_state"]["cartesian_position"][3:6]
                if vertical_target is None:
                    vertical_target = vertical_orientation(orientation)
                    print(f"[keep-vertical] target RPY [rad]: {vertical_target.tolist()}")
                action = vertical_velocity_action(action, orientation, vertical_target)

            control_timestamps["sleep_start"] = time_ms()
            if last_control_start is None:
                sleep_left = 0.0
            else:
                comp_time = time_ms() - last_control_start
                sleep_left = (1 / env.control_hz) - (comp_time / 1000)

            if sleep_left > 0:
                time.sleep(sleep_left)

            control_timestamps["control_start"] = time_ms()
            # Monotonic action-start time for the observation/action row we may save.
            recorded_step_time = time.perf_counter()
            action_info = env.step(action)
            last_control_start = control_timestamps["control_start"]

            control_timestamps["step_end"] = time_ms()
            action_info.update(controller_action_info)

            obs["timestamp"]["control"] = control_timestamps

            # Before the write below, so the step that opens the writers is itself saved.
            if (not start_recording) and controller_info.get("movement_enabled", False) and recording_folderpath:
                # SVO recording intentionally disabled: the MP4 writers below already
                # capture everything we need, and enabling SVO makes each control-loop
                # grab() also H.265-encode+write the full frame inline (~10-14ms added
                # to read_camera). We keep only the MP4 stream.
                start_recording = True
                if save_filepath:
                    recorder = CollectionRecorder(
                        env, save_filepath, recording_folderpath, FLAGS.save_right_images,
                        (FLAGS.video_save_width, FLAGS.video_save_height),
                        video_encoder_threads=FLAGS.video_encoder_threads,
                    )
                print("start recording (async image transform + HDF5 + MP4)")

            if recorder is not None and start_recording:
                recorder.submit(obs, action_info)
                recorded_steps += 1
                if first_recorded_step_time is None:
                    first_recorded_step_time = recorded_step_time
                last_recorded_step_time = recorded_step_time

            t_after_timestep = time_ms()

            time_end = time_ms()
            # Block timings (ms)
            ts = control_timestamps
            print(
                "timing ms:",
                "controller=", t_after_controller - time_start,
                "read_camera=", read_camera_end - read_camera_start,
                "get_info=", t_after_obs - read_camera_end,
                "policy=", ts["sleep_start"] - ts["policy_start"],
                "sleep=", ts["control_start"] - ts["sleep_start"],
                "env_step=", ts["step_end"] - ts["control_start"],
                "record_enqueue=", t_after_timestep - ts["step_end"],
                "timestep_append=", time_end - t_after_timestep,
                "| total=", time_end - time_start,
            )
    finally:
        # N recorded steps span N-1 intervals; exclude reset, idle before recording,
        # the terminal unsaved step, and file finalization from this measurement.
        recorded_duration_s = (
            last_recorded_step_time - first_recorded_step_time if recorded_steps > 1 else 0.0
        )
        recorded_fps = (
            (recorded_steps - 1) / recorded_duration_s if recorded_duration_s > 0 else float("nan")
        )
        timing_metadata = {
            "keep_vertical": keep_vertical,
            "recorded_steps": recorded_steps,
            "recorded_duration_s": recorded_duration_s,
            "recorded_fps": recorded_fps,
        }
        print(
            f"[episode-recording] steps={recorded_steps} "
            f"first_to_last_s={recorded_duration_s:.3f} avg_hz={recorded_fps:.3f}"
        )
        t0 = time.perf_counter()
        if recorder is not None:
            metadata = dict(controller_info) if "controller_info" in locals() else {}
            metadata.update(timing_metadata)
            recorder.close(metadata)
        print("[between-episode] drain_and_close_recording={:.2f}s".format(time.perf_counter() - t0))

def run_and_route_one(env, controller, base_dir, keys=None) -> Dict[str, object]:
    tmp_root = os.path.join(base_dir, "tmp")
    os.makedirs(tmp_root, exist_ok=True)
    # A fast discard/retry must never reuse another attempt's partial directory.
    tmp_dir = tempfile.mkdtemp(prefix='session_', dir=tmp_root)
    images_dir = os.path.join(tmp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    save_filepath = os.path.join(tmp_dir, "traj.hdf5")

    print("Temp session dir:", tmp_dir)
    print("Temp traj file:", save_filepath)
    print("Temp recording path:", images_dir)

    print("Start collecting")
    result = collect_trajectory(
        env,
        controller=controller,
        save_filepath=save_filepath,
        recording_folderpath=images_dir,
        test_detector=FLAGS.test_detector,
        keep_vertical=FLAGS.keep_vertical,
        keys=keys,
    )

    if result.get('discarded', False):
        # collect_trajectory's finally has drained and closed every writer already.
        # Fail visibly if removal fails; never route a discarded attempt to success.
        shutil.rmtree(tmp_dir)
        print('Discarded current rollout (HDF5 + MP4). Resetting for a new attempt.')
        return {'dest_dir': None, 'id': None, 'result': result}

    success = result.get("success", False)
    print(f"Outcome: {'success' if success else 'failure'}")

    if not success:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("Failure — discarded temp data.")
        return {"dest_dir": None, "id": None, "result": result}

    outcome_root = os.path.join(base_dir, "success")
    new_id = smallest_missing_id(outcome_root)
    dest_dir = os.path.join(outcome_root, str(new_id))
    os.makedirs(outcome_root, exist_ok=True)

    print("Assigning id:", new_id)
    print("Destination:", dest_dir)

    t_move0 = time.perf_counter()
    shutil.move(tmp_dir, dest_dir)
    print("[between-episode] move={:.2f}s".format(time.perf_counter() - t_move0))
    print(
        "Result saved:",
        os.path.join(dest_dir, "traj.hdf5"),
        os.path.join(dest_dir, "images"),
        os.path.join(dest_dir, "recordings", "MP4"),
    )
    return {"dest_dir": dest_dir, "id": new_id, "result": result}


def main(_):
    if FLAGS.video_encoder_threads < 1:
        raise ValueError("video_encoder_threads must be positive")
    width, height = FLAGS.video_save_width, FLAGS.video_save_height
    if width <= 0 or height <= 0:
        raise ValueError("video_save_width/height must both be positive")
    task_config = FLAGS.task_config
    if FLAGS.robot_config:
        with open(FLAGS.robot_config) as file:
            robot_config = json.load(file)
        for key, value in robot_config.items():
            current = task_config.get(key)
            if isinstance(current, np.ndarray):
                value = np.asarray(value, dtype=current.dtype)
            task_config[key] = value

    base_dir = FLAGS.save_root
    os.makedirs(base_dir, exist_ok=True)

    # use cartesian velocity and velocity for collecting data
    task_config.action_space = "cartesian_velocity"
    task_config.gripper_action_space = "velocity"
    
    env_kwargs = dict(task_config)
    env_kwargs["video_encoder_threads"] = FLAGS.video_encoder_threads
    camera_kwargs = {
        key: dict(value)
        for key, value in task_config.camera_kwargs.items()
    }
    if FLAGS.save_right_images:
        for settings in camera_kwargs.values():
            settings["left_only"] = False
    env_kwargs["camera_kwargs"] = camera_kwargs
    # Only the HQ record camera writes during teleop collection (upstream passes no
    # video_dir here, so the per-episode raw clips stay an RL train/eval feature).
    if task_config.get("record_camera"):
        env_kwargs["video_dir"] = os.path.join(base_dir, "recordings")
    # Keep configured cameras open during teleop (eval may release unused cameras).
    env_kwargs.setdefault("release_unused_cameras", False)
    env = task_config.env(**env_kwargs)
    # Teleop collection ends episodes on success/detector/manual/bounds only, not the step budget.
    env.ignore_auto_reset = True
    controller = SpaceMousePolicy(
        max_lin_vel=task_config.collect_max_lin_vel,
        max_rot_vel=task_config.collect_max_rot_vel,
        device_number=task_config.get("spacemouse_device_number", 0),
        device_path=task_config.get("spacemouse_device_path"),
    )

    with CollectionKeys() as keys:
        run_collection(env, controller, base_dir, keys)


def run_collection(env, controller, base_dir, keys):
    if FLAGS.test_detector:
        print('test_detector=True: no saving. D discards/resets; Ctrl+C stops.')
        while True:
            collect_trajectory(env, controller=controller, test_detector=True,
                               keep_vertical=FLAGS.keep_vertical, keys=keys)

    episode = 0
    successful_episodes = 0
    while True:
        episode += 1

        print(f"\n{'=' * 60}")
        print(f"Starting trajectory collection #{episode} (Successful: {successful_episodes}/{FLAGS.num_episodes if FLAGS.num_episodes > 0 else '∞'})")
        print(f"{'=' * 60}\n")

        result = run_and_route_one(env, controller, base_dir, keys=keys)
        
        if result["result"]["success"]:
            successful_episodes += 1
            print(f"\n{'=' * 60}")
            print(f"Completed successful trajectory #{successful_episodes} (Total attempts: {episode})")
            print(f"{'=' * 60}\n")

        if FLAGS.num_episodes > 0 and successful_episodes >= FLAGS.num_episodes:
            print(f"\n{'=' * 60}")
            print(f"Reached target of {FLAGS.num_episodes} successful episodes after {episode} total attempts")
            print(f"{'=' * 60}\n")
            break


if __name__ == "__main__":
    app.run(main)
