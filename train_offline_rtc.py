#!/usr/bin/env python
"""Offline RTC-SFT: prefix-conditioned BC of pi0.5 on real-robot demos."""

import os
import logging
import time

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

import jax
import etils.epath as epath

import wandb
from expo_ft.agents import initialize_checkpoint_dir
from expo_ft.data.replay_buffer import create_replay_buffer
from expo_ft.data.batch_processor import BatchProcessor
from expo_ft.env.droid_utils import process_droid_dataset
from expo_ft.utils.train_utils import (
    get_batch_info, init_logging, init_wandb, set_compilation_cache_dir,
)
from expo_ft.agents.alg.rtc import load_agent, restore_checkpoint, save_checkpoint
from expo_ft.agents.vla.pi05 import build_pi05

import openpi.training.sharding as openpi_sharding

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "expo-ft", "wandb project name.")
flags.DEFINE_string("run_name", None, "wandb run name (also used as the log subdir).")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("batch_size", 64, "Mini batch size.")
flags.DEFINE_integer("max_steps", 100_000, "Number of gradient updates to run.")
flags.DEFINE_integer("num_data", 0, "Max number of offline demo episodes to load (0 = all).")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("checkpoint_model", True, "Save agent checkpoint during training.")
flags.DEFINE_integer("checkpoint_interval", 2000, "Save agent checkpoint every N gradient steps.")
flags.DEFINE_integer("keep_period", None, "Keep checkpoints every N steps.")
flags.DEFINE_boolean("overwrite", False, "Overwrite existing checkpoint directory.")
flags.DEFINE_boolean("resume", False, "Resume training from checkpoint.")
flags.DEFINE_string("output_dir", "./logs", "Directory for logs and checkpoints.")
flags.DEFINE_integer("fsdp_devices", 1, "Number of FSDP devices for sharding.")

flags.DEFINE_integer("replan_steps", 8, "Number of replan steps (action-chunk execution horizon).")

flags.DEFINE_string("dataset_path", "", "Path to the dataset.")
config_flags.DEFINE_config_file(
    "config",
    "configs/model/rtc_pi_config.py",
    "File path to the training hyperparameter configuration (model_cls must be RTCLearner).",
    lock_config=False,
)
config_flags.DEFINE_config_file(
    "config_task",
    "configs/task/pick.py",
    "File path to the task configuration.",
    lock_config=False,
)


def main(_):
    init_logging()
    logger = logging.getLogger(__name__)

    if FLAGS.config.model_cls != "RTCLearner":
        raise ValueError(
            f"train_offline_rtc.py supports model_cls=RTCLearner only; got {FLAGS.config.model_cls!r}. "
            "Use configs/model/rtc_pi_config.py."
        )
    if FLAGS.config_task.env_type != "droid":
        raise ValueError(f"Unsupported env_type: {FLAGS.config_task.env_type}")
    if FLAGS.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {FLAGS.batch_size} must be divisible by num devices {jax.device_count()}."
        )
    if not FLAGS.run_name:
        raise ValueError("--run_name is required.")

    set_compilation_cache_dir(f"offline-{FLAGS.config.model_cls}")

    mesh = openpi_sharding.make_mesh(FLAGS.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    log_dir = os.path.join(FLAGS.output_dir, FLAGS.run_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_dir_path = epath.Path(checkpoint_dir)
    checkpoint_manager, resuming = initialize_checkpoint_dir(
        checkpoint_dir_path,
        keep_period=FLAGS.keep_period,
        overwrite=FLAGS.overwrite,
        resume=FLAGS.resume,
    )

    init_wandb(checkpoint_dir_path, resuming, FLAGS.project_name, FLAGS.run_name)
    wandb.config.update(FLAGS.flag_values_dict(), allow_val_change=resuming)

    dataset = process_droid_dataset(
        FLAGS.dataset_path,
        FLAGS.config_task,
        num_data=FLAGS.num_data,
    )
    if not dataset:
        raise ValueError(f"No demos loaded from {FLAGS.dataset_path}.")
    example_action = dataset[0]["actions"][np.newaxis]

    task_description = getattr(FLAGS.config_task, "language_instruction", "")
    if not task_description:
        raise ValueError("config_task.language_instruction is required for offline training.")
    logger.info("Offline training with task prompt %r.", task_description)

    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        FLAGS.config, FLAGS.seed, mesh, data_sharding, replicated_sharding,
        resuming, task_description,
    )
    # The demos are the whole dataset: nothing is inserted after them.
    replay_buffer = create_replay_buffer(
        config=FLAGS.config,
        example_action=example_action,
        capacity=len(dataset),
        task_description=task_description,
        replan_steps=FLAGS.replan_steps,
        seed=FLAGS.seed,
    )
    # DAgger-style HIL sampling draws the actor batch from the demo chunks, so no
    # offline buffer or success-only actor batch is involved.
    batch_processor = BatchProcessor(
        replay_buffer=replay_buffer,
        offline_replay_buffer=None,
        data_sharding=data_sharding,
        batch_size=FLAGS.batch_size,
        utd_ratio=1,
        offline_ratio=0.0,
        actor_success_only=False,
        use_dagger_hil_sampling=True,
        dataset=dataset,
    )

    agent_example_observation, agent_example_state, agent_example_action = replay_buffer.convert_to_critic_format({
        "base_image": replay_buffer.dataset_dict["base_image"][0][np.newaxis],
        "left_wrist_image": replay_buffer.dataset_dict["left_wrist_image"][0][np.newaxis],
        "state": replay_buffer.dataset_dict["state"][0][np.newaxis],
        "actions": replay_buffer.dataset_dict["actions"][0][np.newaxis],
    })
    actor.action_dim = agent_example_action.squeeze().shape[-1]
    actor.state_dim = agent_example_state.squeeze().shape[-1]
    agent = load_agent(
        seed=FLAGS.seed,
        example_observation=agent_example_observation.squeeze(),
        example_action=agent_example_action.squeeze(),
        example_state=agent_example_state.squeeze(),
        actor=actor,
        actor_train_state=actor_train_state,
        target_actor_params=target_actor_params,
        agent_kwargs=agent_kwargs,
        metadata=vla_metadata,
        mesh=mesh,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        resume=resuming,
        replan_steps=FLAGS.replan_steps,
        default_prompt=task_description,
        edit_action_xyzg=FLAGS.config_task.edit_action_xyzg,
    )

    start_step = 0
    if resuming:
        agent = restore_checkpoint(checkpoint_manager, agent)
        agent = agent.cache_infer_params()
        steps = tuple(checkpoint_manager.all_steps())
        if steps:
            start_step = max(steps)
            logger.info("Resuming from step %d", start_step)

    combine_rng = jax.random.PRNGKey(FLAGS.seed + 100)

    for step in tqdm.tqdm(range(start_step, FLAGS.max_steps),
                          smoothing=0.1, disable=not FLAGS.tqdm):
        loop_start = time.time()
        step_metrics = {}

        batch, actor_batch, combine_rng = batch_processor.next_batch(combine_rng)
        step_metrics["batch_info"] = get_batch_info(batch)
        agent = agent.replace(rng=jax.device_put(agent.rng, replicated_sharding))
        agent, update_info = agent.update(agent, batch, 1, actor_batch)
        for k, v in update_info.items():
            step_metrics[f"training/{k}"] = v

        if (FLAGS.checkpoint_model and FLAGS.checkpoint_interval > 0
                and step > 0 and step % FLAGS.checkpoint_interval == 0):
            try:
                save_checkpoint(checkpoint_manager, agent, step)
                logger.info("Saved checkpoint at step %d", step)
            except Exception as exc:
                logger.error("Could not save checkpoint at step %d: %s", step, exc)

        step_metrics["training/loop_time_ms"] = (time.time() - loop_start) * 1000.0
        wandb.log(step_metrics, step=step)

    if FLAGS.checkpoint_model:
        try:
            save_checkpoint(checkpoint_manager, agent, FLAGS.max_steps)
            logger.info("Saved final checkpoint at step %d", FLAGS.max_steps)
        except Exception as exc:
            logger.error("Could not save final checkpoint: %s", exc)
        checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    app.run(main)
