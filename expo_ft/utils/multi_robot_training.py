"""Synchronous robot rounds for the existing EXPO learner."""
import json
import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import jax
import numpy as np
import wandb

from expo_ft.data.replay_buffer import (
    prepare_robot_replay_resume, restore_replay_buffer, save_replay_buffer_batch,
)
from expo_ft.env.env_client import EnvClientWrapper
from expo_ft.utils.robot_round import collect_round, updates_for_round


def train_multi_robot(flags, agent, buffers, batch_processor, checkpoint_manager,
                      checkpoint_dir, video_dir, save_checkpoint, start_step, resuming, replicated_sharding,
                      mirror_robot=None):
    checkpoint_dir = Path(checkpoint_dir)
    camera_views = [{} for _ in buffers]
    if mirror_robot is not None:
        for index in range(len(buffers)):
            path = Path(__file__).resolve().parents[2] / f"configs/robots/robot-{index}.json"
            config = json.loads(path.read_text())
            camera_views[index] = {key: config[key] for key in ("side_camera_id", "wrist_camera_id")}
    step, episode_count, pending_steps = start_step, 0, 0
    combine_rng = jax.random.PRNGKey(flags.seed + 100)
    if resuming:
        state = json.loads((checkpoint_dir / f"round-{step}.json").read_text())
        if state["num_robot"] != len(buffers):
            raise ValueError("Resume requires the same ordered robot configuration")
        if state.get("mirror_robot") != mirror_robot:
            raise ValueError("Resume requires the same live robot mirror convention")
        episode_count, pending_steps = state["episode_count"], state["pending_steps"]
        combine_rng = np.asarray(state["combine_rng"], dtype=np.uint32)
        prepare_robot_replay_resume(checkpoint_dir, up_to_step=step, num_robot=len(buffers))
        for index, buffer in enumerate(buffers):
            restore_replay_buffer(checkpoint_dir / f"robot-{index}", buffer, up_to_step=step)
            buffer.restore_success_marks()

    envs = [EnvClientWrapper(
        env_creation_request={
            "example_action": flags.config_task.example_action,
            "env_usage": "train", "video_dir": str(Path(video_dir) / f"robot-{index}"),
            "expected_camera_views": camera_views[index],
            "async_video": True,
        }, host=flags.client_host, port=flags.client_port + index,
        recover=False, lazy=True,
    ) for index in range(len(buffers))]

    def sample_actions(observation):
        nonlocal agent
        actions, agent, _ = agent.sample_actions(observation)
        return np.asarray(jax.device_get(actions))

    def checkpoint():
        # The round ledger and all replay records are written before publishing
        # the model checkpoint, so a restored policy always has a full barrier.
        state = dict(num_robot=len(buffers), episode_count=episode_count,
                     pending_steps=pending_steps, combine_rng=np.asarray(combine_rng).tolist(),
                     mirror_robot=mirror_robot)
        (checkpoint_dir / f"round-{step}.json").write_text(json.dumps(state))
        save_checkpoint(checkpoint_manager, agent, step)

    reset_workers = ThreadPoolExecutor(max_workers=len(envs), thread_name_prefix="robot-reset")
    resets = []

    def begin_resets():
        logging.info("Local: resetting %d robots before the next round", len(envs))
        return [reset_workers.submit(env.reset_only) for env in envs]

    def check_reset_errors():
        for reset in resets:
            if reset.done():
                reset.result()

    def finish_resets():
        pending = set(resets)
        while pending:
            done, pending = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
            for reset in done:
                reset.result()

    last_checkpoint = start_step
    save_request = checkpoint_dir / "save.request"
    try:
        if step < flags.max_steps:
            resets = begin_resets()
        while step < flags.max_steps:
            finish_resets()
            resets = []
            episodes = collect_round(envs, sample_actions, flags.replan_steps, flags.config_task.control_hz,
                                     mirror_robot=mirror_robot, reset_done=True)
            round_steps = sum(len(transitions) for transitions, _ in episodes)
            metrics = {}
            next_step = step + 1
            for index, (transitions, success) in enumerate(episodes):
                for transition in transitions:
                    transition["is_success"] = bool(success)
                if flags.checkpoint_buffer:
                    save_replay_buffer_batch(checkpoint_dir / f"robot-{index}", transitions,
                                             start_step=next_step)
                next_step += len(transitions)
            for index, (buffer, (transitions, success)) in enumerate(zip(buffers, episodes)):
                for transition in transitions:
                    buffer.insert(transition)
                    step += 1
                metrics[f"robot-{index}/success"] = float(success)
                metrics[f"robot-{index}/episode_length"] = len(transitions)
                metrics[f"robot-{index}/return"] = sum(t["rewards"] for t in transitions)

            # All robots' records are saved/inserted before allowing their next reset.
            # Reset workers only perform RPCs; the caller exclusively owns the model.
            if step < flags.max_steps:
                resets = begin_resets()

            count, pending_steps = updates_for_round(
                pending_steps, round_steps, can_update=episode_count >= 10 * len(buffers) and step >= flags.batch_size,
                num_updates=flags.num_updates, step_interval=flags.step_interval,
            )
            for _ in range(count):
                check_reset_errors()
                batch, actor_batch, combine_rng = batch_processor.next_batch(combine_rng)
                agent = agent.replace(rng=jax.device_put(agent.rng, replicated_sharding))
                agent, update_info = agent.update(agent, batch, flags.utd_ratio, actor_batch)
                metrics.update({f"training/{key}": value for key, value in update_info.items()})
            if count:
                # JAX dispatch is asynchronous: finish updates before the next observation.
                jax.block_until_ready(agent)
            check_reset_errors()
            episode_count += len(episodes)
            metrics.update(episodes=episode_count, updates=count, round_steps=round_steps)
            wandb.log(metrics, step=step)
            logging.info("Round complete: %d episodes, %d transitions, %d updates", episode_count, step, count)
            manual_save = save_request.is_file()
            if manual_save or (flags.checkpoint_model and flags.checkpoint_interval > 0
                               and step - last_checkpoint >= flags.checkpoint_interval):
                checkpoint()
                if manual_save:
                    checkpoint_manager.wait_until_finished()
                    save_request.unlink(missing_ok=True)
                    logging.info("Manual checkpoint saved at step %d in %s", step, checkpoint_dir)
                last_checkpoint = step
        if flags.checkpoint_model and step != last_checkpoint:
            checkpoint()
    finally:
        for reset in resets:
            reset.cancel()
        # Close RPCs before joining: an unfinished reset may be waiting for a client.
        for env in envs:
            try:
                env.close()
            except Exception:
                logging.exception("Could not close robot client")
        try:
            reset_workers.shutdown(wait=True)
        finally:
            checkpoint_manager.wait_until_finished()
