"""Synchronous robot rounds for the existing EXPO learner."""
import json
import logging
from pathlib import Path

import jax
import numpy as np
import wandb

from expo_ft.data.replay_buffer import restore_replay_buffer, save_replay_buffer_transition
from expo_ft.env.env_client import EnvClientWrapper
from expo_ft.utils.robot_round import collect_round, updates_for_round


def train_multi_robot(flags, agent, buffers, batch_processor, checkpoint_manager,
                      checkpoint_dir, video_dir, save_checkpoint, start_step, resuming, replicated_sharding):
    checkpoint_dir = Path(checkpoint_dir)
    step, episode_count, pending_steps = start_step, 0, 0
    combine_rng = jax.random.PRNGKey(flags.seed + 100)
    if resuming:
        state = json.loads((checkpoint_dir / f"round-{step}.json").read_text())
        if state["num_robot"] != len(buffers):
            raise ValueError("Resume requires the same ordered robot configuration")
        episode_count, pending_steps = state["episode_count"], state["pending_steps"]
        combine_rng = np.asarray(state["combine_rng"], dtype=np.uint32)
        records = list(checkpoint_dir.glob("robot-*/buffers/*.pkl"))
        if sum(int(path.stem) <= step for path in records) != step:
            raise ValueError("Cannot resume: incomplete robot replay records for this checkpoint")
        # Discard the abandoned suffix before assigning new global step numbers;
        # robot episode lengths (and therefore their step ranges) may change.
        for path in records:
            if int(path.stem) > step:
                path.unlink()
        for index, buffer in enumerate(buffers):
            restore_replay_buffer(checkpoint_dir / f"robot-{index}", buffer, up_to_step=step)
            buffer.restore_success_marks()

    envs = [EnvClientWrapper(
        env_creation_request={
            "example_action": flags.config_task.example_action,
            "env_usage": "train", "video_dir": str(Path(video_dir) / f"robot-{index}"),
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
                     pending_steps=pending_steps, combine_rng=np.asarray(combine_rng).tolist())
        (checkpoint_dir / f"round-{step}.json").write_text(json.dumps(state))
        save_checkpoint(checkpoint_manager, agent, step)

    last_checkpoint = start_step
    try:
        while step < flags.max_steps:
            episodes = collect_round(envs, sample_actions, flags.replan_steps, flags.config_task.control_hz)
            round_steps = sum(len(transitions) for transitions, _ in episodes)
            metrics = {}
            for index, (buffer, (transitions, success)) in enumerate(zip(buffers, episodes)):
                for transition in transitions:
                    # Mark before inserting/saving; no episode can mark another robot's rows.
                    transition["is_success"] = bool(success)
                    buffer.insert(transition)
                    step += 1
                    if flags.checkpoint_buffer:
                        save_replay_buffer_transition(checkpoint_dir / f"robot-{index}", transition, step=step)
                metrics[f"robot-{index}/success"] = float(success)
                metrics[f"robot-{index}/episode_length"] = len(transitions)
                metrics[f"robot-{index}/return"] = sum(t["rewards"] for t in transitions)

            count, pending_steps = updates_for_round(
                pending_steps, round_steps, can_update=episode_count >= 10 and step >= flags.batch_size,
                num_updates=flags.num_updates, step_interval=flags.step_interval,
            )
            for _ in range(count):
                batch, actor_batch, combine_rng = batch_processor.next_batch(combine_rng)
                agent = agent.replace(rng=jax.device_put(agent.rng, replicated_sharding))
                agent, update_info = agent.update(agent, batch, flags.utd_ratio, actor_batch)
                metrics.update({f"training/{key}": value for key, value in update_info.items()})
            episode_count += len(episodes)
            metrics.update(episodes=episode_count, updates=count, round_steps=round_steps)
            wandb.log(metrics, step=step)
            logging.info("Round complete: %d episodes, %d transitions, %d updates", episode_count, step, count)
            if flags.checkpoint_model and flags.checkpoint_interval > 0 and step - last_checkpoint >= flags.checkpoint_interval:
                checkpoint()
                last_checkpoint = step
        if flags.checkpoint_model and step != last_checkpoint:
            checkpoint()
    finally:
        for env in envs:
            env.close()
        checkpoint_manager.wait_until_finished()
