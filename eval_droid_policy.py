#!/usr/bin/env python

from __future__ import annotations

import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import etils.epath as epath
import jax
import numpy as np
from absl import app, flags
from ml_collections import config_flags

from expo_ft.agents import initialize_checkpoint_dir
from expo_ft.data.replay_buffer import create_replay_buffer, _critic_key_to_storage
from expo_ft.env.env_client import EnvClientWrapper
from expo_ft.env.droid_utils import process_droid_dataset
from expo_ft.env.sft_eval import canonical_observation, physical_action

import openpi.training.sharding as openpi_sharding

config_flags.DEFINE_config_file(
    "config",
    "configs/model/expo_ft_pi_config.py",
    "Training config (must match the checkpoint).",
    lock_config=False,
)
config_flags.DEFINE_config_file(
    "config_task",
    "configs/task/pick.py",
    "Task config (must match training).",
    lock_config=False,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_path", "", "Path to DROID dataset (for example_action).")
flags.DEFINE_integer("num_data", 1, "Number of episodes to load from dataset (only need 1 for example_action).")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory (e.g. .../checkpoints/<run_name>/checkpoints).")
flags.DEFINE_integer("checkpoint_step", None, "Checkpoint step to load; default is latest.")
flags.DEFINE_string("client_host", "0.0.0.0", "Bind host to listen on; the rollout client dials in.")
flags.DEFINE_integer("client_port", 8102, "Bind port to listen on.")
flags.DEFINE_integer("num_episodes", 10, "Number of evaluation episodes.")
flags.DEFINE_integer("replan_steps", 8, "Replan every N steps (match training).")
flags.DEFINE_integer(
    "delay", 0,
    "Simulated inference latency in env steps. RealTimeEXPOFTLearner launches "
    "sample_pre_cache asynchronously; RTCLearner launches sample_actions "
    "asynchronously because it has no pre-cache phase.",
)
flags.DEFINE_boolean("only_base_actions", False, "Use only base (OpenPI) actions, no edit, sample 1.")
flags.DEFINE_float("sim_latency", 0.0, "Simulated extra inference latency in ms added to each sample_actions call; 0 disables.")
flags.DEFINE_boolean("save_video", True, "Save evaluation videos.")
flags.DEFINE_integer("fsdp_devices", 1, "Number of FSDP devices (match training).")
flags.DEFINE_boolean("mirror_y", False, "Mirror RGB/Cartesian pose and invert actions for the mirrored robot.")
flags.DEFINE_string("client_video_dir", "", "Video directory on the workstation; overrides the derived path.")


def main(_):
    config = FLAGS.config
    config_task = FLAGS.config_task

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    if config_task.env_type != "droid":
        raise ValueError(
            "This script is for DROID evaluation only; config_task.env_type must be 'droid'."
        )

    if not FLAGS.dataset_path:
        raise ValueError("--dataset_path is required.")

    checkpoint_manager = None
    checkpoint_steps = ()
    if FLAGS.checkpoint_dir:
        checkpoint_dir_path = epath.Path(FLAGS.checkpoint_dir)
        if not checkpoint_dir_path.exists():
            raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir_path}")
        checkpoint_manager, _ = initialize_checkpoint_dir(
            checkpoint_dir_path,
            keep_period=None,
            overwrite=False,
            resume=True,
        )
        checkpoint_steps = tuple(checkpoint_manager.all_steps())
    step = FLAGS.checkpoint_step
    if step is None:
        step = max(checkpoint_steps) if checkpoint_steps else 0
        logger.info("Using latest checkpoint step %s", step)
    if step != 0 and checkpoint_manager is None:
        raise ValueError("--checkpoint_dir is required when --checkpoint_step is nonzero.")
    if step != 0:
        if step not in checkpoint_steps:
            raise ValueError(f"Step {step} not in checkpoint steps {checkpoint_steps}")
        logger.info("Will load checkpoint at step %s", step)

    example_action = config_task.example_action
    # Dataset only for agent observation/action shapes (run one sample through transform)
    dataset = process_droid_dataset(
        FLAGS.dataset_path,
        config_task,
        num_data=FLAGS.num_data,
    )

    task_description = config_task.language_instruction
    max_traj_len = config_task.auto_reset_steps
    dt = 1.0 / config_task.control_hz
    if FLAGS.delay < 0 or FLAGS.delay > FLAGS.replan_steps:
        raise ValueError(
            f"--delay must be in [0, replan_steps]; got delay={FLAGS.delay}, "
            f"replan_steps={FLAGS.replan_steps}."
        )

    # Agent config (match train_pi_robo)
    mesh = openpi_sharding.make_mesh(FLAGS.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    model_cls = config.model_cls
    if model_cls == "BCLearner":
        from expo_ft.agents.alg.bc import load_agent, restore_checkpoint
    elif model_cls == "EXPOLearner":
        from expo_ft.agents.alg.expo_ft import load_agent, restore_checkpoint
    elif model_cls == "RealTimeEXPOFTLearner":
        from expo_ft.agents.alg.realtime_expo_ft import load_agent, restore_checkpoint
    elif model_cls == "RTCLearner":
        from expo_ft.agents.alg.rtc import load_agent, restore_checkpoint
    else:
        raise ValueError(f"Unsupported model class: {model_cls}")

    from expo_ft.agents.vla.pi05 import build_pi05
    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        config, FLAGS.seed, mesh, data_sharding, replicated_sharding,
        resume=(step != 0), default_prompt=task_description,
    )

    # Must match training (train_pi_robo.py), or the critic stem conv is built with the
    # wrong in_channels (3 per camera) and the checkpoint restore shape-mismatches.
    from expo_ft.agents.alg.batch_utils import CRITIC_CAMERA_KEYS
    critic_camera_keys = tuple(getattr(config_task, "critic_camera_keys", CRITIC_CAMERA_KEYS))

    replay_buffer = create_replay_buffer(
        config=config,
        example_action=example_action,
        capacity=max_traj_len * 2,
        task_description=task_description,
        replan_steps=FLAGS.replan_steps,
        seed=FLAGS.seed,
        critic_camera_keys=critic_camera_keys,
    )
    replay_buffer.insert_dataset(dataset[:1])

    critic_example = {
        _critic_key_to_storage(k): replay_buffer.dataset_dict[_critic_key_to_storage(k)][0][np.newaxis]
        for k in critic_camera_keys
    }
    critic_example["state"] = replay_buffer.dataset_dict["state"][0][np.newaxis]
    critic_example["actions"] = replay_buffer.dataset_dict["actions"][0][np.newaxis]
    agent_example_observation, agent_example_state, agent_example_action = replay_buffer.convert_to_critic_format(
        critic_example)
    actor.action_dim = agent_example_action.squeeze().shape[-1]
    actor.state_dim = agent_example_state.squeeze().shape[-1]
    agent_kwargs["critic_camera_keys"] = critic_camera_keys
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
        resume=(step != 0),
        replan_steps=FLAGS.replan_steps,
        default_prompt=task_description,
        edit_action_xyzg=config_task.edit_action_xyzg,
    )

    if step != 0:
        agent = restore_checkpoint(checkpoint_manager, agent, step=step)
        logger.info("Loaded checkpoint at step %s", step)

    if hasattr(agent, 'cache_infer_params'):
        agent = agent.cache_infer_params()

    video_dir = None
    if FLAGS.save_video and FLAGS.client_video_dir:
        # This path is interpreted and created by the WS client only.
        video_dir = FLAGS.client_video_dir
        logger.info("Saving evaluation videos on workstation: %s", video_dir)
    elif FLAGS.save_video:
        weight_loader_path = getattr(config, "pi05_weight_loader_path", "") or ""
        if FLAGS.checkpoint_dir:
            eval_root = os.path.join(os.path.dirname(FLAGS.checkpoint_dir), "eval", f"step_{step}")
        elif weight_loader_path:
            # No expo checkpoint: the evaluated weights are the loader path itself, so
            # results belong under that run's <run_dir>/evals/<weight_step>. Loader path
            # is <run_dir>/checkpoints/<step>/params.
            params_dir = os.path.dirname(weight_loader_path)
            weight_step = os.path.basename(params_dir)
            run_dir = os.path.dirname(os.path.dirname(params_dir))
            eval_root = os.path.join(run_dir, "evals", weight_step)
        else:
            eval_root = "./eval_droid_policy/evals"
        parts = ["only_base" if FLAGS.only_base_actions else "full", f"delay{FLAGS.delay}"]
        video_dir = os.path.join(eval_root, "_".join(parts))
        os.makedirs(video_dir, exist_ok=True)
        logger.info("Saving evaluation videos to %s", video_dir)

    eval_env_creation_request = {
        "example_action": example_action,
        "env_usage": "eval",
        # Client-local path: arms the env-side recorders
        # (HQ record_camera MP4s, and DroidEnv's raw episode videos).
        "video_dir": video_dir or "",
    }
    logger.info("Listening for rollout client on %s:%s ...", FLAGS.client_host, FLAGS.client_port)
    env = EnvClientWrapper(
        env_creation_request=eval_env_creation_request,
        host=FLAGS.client_host,
        port=FLAGS.client_port,
    )
    print("resetting environment...")
    env.reset()
    print("environment reset")

    time.sleep(10)

    successes = []
    episode_returns = []
    episode_lengths = []
    supports_pre_cache_delay = hasattr(agent, "sample_pre_cache") and FLAGS.delay > 0
    supports_rtc_delay = model_cls == "RTCLearner" and FLAGS.delay > 0
    supports_async_delay = supports_pre_cache_delay or supports_rtc_delay
    inference_executor = ThreadPoolExecutor(max_workers=1) if supports_async_delay else None
    if supports_pre_cache_delay:
        logger.info("RealTimeEXPOFTLearner eval: using async main-actor inference delay=%d env-steps", FLAGS.delay)
    if supports_rtc_delay:
        logger.info("RTCLearner eval: using async sample_actions delay=%d env-steps", FLAGS.delay)

    def clear_pending(pending):
        if pending is not None:
            pending.cancel()
        return None

    def simulate_latency():
        if FLAGS.sim_latency > 0:
            time.sleep(FLAGS.sim_latency / 1000.0)

    for ep in range(FLAGS.num_episodes):
        logger.info("Episode %d / %d", ep + 1, FLAGS.num_episodes)
        observation = env.reset()
        prefix_cache = None
        pending_inference = None
        last_control_start = None
        action_plan = deque()
        sample_info_history = []
        ep_return = 0.0
        ep_len = 0
        ep_human_steps = 0

        def run_pre_cache(agent_snapshot, obs, prefix):
            compute_start = time.time()
            precached, new_agent = agent_snapshot.sample_pre_cache(obs, prefix_padded=prefix)
            precached = jax.block_until_ready(precached)
            return precached, new_agent, (time.time() - compute_start) * 1000.0

        def run_rtc_sample(agent_snapshot, obs, prefix):
            compute_start = time.time()
            action_chunk, new_agent, sample_info = agent_snapshot.sample_actions(
                obs,
                only_base_actions=FLAGS.only_base_actions,
                delay=FLAGS.delay,
                p1_observations=obs,
                prefix_padded=prefix,
            )
            action_chunk = jax.block_until_ready(action_chunk)
            simulate_latency()
            return action_chunk, new_agent, sample_info, (time.time() - compute_start) * 1000.0

        def launch_async_inference(obs):
            nonlocal pending_inference
            if pending_inference is not None or len(action_plan) != FLAGS.delay:
                return
            if supports_pre_cache_delay:
                if prefix_cache is None:
                    return
                pending_inference = inference_executor.submit(
                    run_pre_cache,
                    agent,
                    obs,
                    prefix_cache,
                )
            elif supports_rtc_delay:
                if prefix_cache is None:
                    return
                pending_inference = inference_executor.submit(
                    run_rtc_sample,
                    agent,
                    obs,
                    prefix_cache,
                )

        for step in range(max_traj_len):
            step_t0 = time.time()
            timing = {
                "wait_ms": 0.0,
                "obs_ms": 0.0,
                "info_ms": 0.0,
                "plan_ms": 0.0,
                "act_ms": 0.0,
            }

            t_obs0 = time.time()
            observation = env.get_observation()
            if FLAGS.mirror_y:
                observation = canonical_observation(observation, mirror=True)
            timing["obs_ms"] = (time.time() - t_obs0) * 1000.0
            t_info0 = time.time()
            done, success, reward, _ = env.get_info_for_step()
            timing["info_ms"] = (time.time() - t_info0) * 1000.0

            launch_async_inference(observation)

            t_plan0 = time.time()
            if not action_plan:
                compute_ms = None
                if supports_pre_cache_delay:
                    if prefix_cache is None:
                        precached, agent = agent.sample_pre_cache(observation, prefix_padded=None)
                        action_chunk, agent, new_si = agent.sample_actions(
                            observation,
                            precached=precached,
                            delay=0,
                            only_base_actions=FLAGS.only_base_actions,
                        )
                        simulate_latency()
                    else:
                        if pending_inference is None:
                            logger.warning(
                                "No pending async pre-cache at replan boundary; running it synchronously."
                            )
                            precached, precache_agent, compute_ms = run_pre_cache(
                                agent, observation, prefix_cache,
                            )
                        else:
                            wait_start = time.time()
                            precached, precache_agent, compute_ms = pending_inference.result()
                            timing["async_wait_ms"] = (time.time() - wait_start) * 1000.0
                            pending_inference = None
                        timing["async_compute_ms"] = compute_ms
                        agent = agent.replace(rng=precache_agent.rng)
                        action_chunk, agent, new_si = agent.sample_actions(
                            observation,
                            precached=precached,
                            delay=FLAGS.delay,
                            only_base_actions=FLAGS.only_base_actions,
                        )
                        simulate_latency()
                    prefix_cache = new_si["executed_padded"][FLAGS.replan_steps - FLAGS.delay:FLAGS.replan_steps]
                elif supports_rtc_delay:
                    if prefix_cache is None:
                        action_chunk, agent, new_si, compute_ms = run_rtc_sample(agent, observation, None)
                    elif pending_inference is None:
                        logger.warning(
                            "No pending async RTC sample at replan boundary; running it synchronously."
                        )
                        action_chunk, agent, new_si, compute_ms = run_rtc_sample(
                            agent, observation, prefix_cache,
                        )
                    else:
                        wait_start = time.time()
                        action_chunk, agent, new_si, compute_ms = pending_inference.result()
                        timing["async_wait_ms"] = (time.time() - wait_start) * 1000.0
                        pending_inference = None
                    if compute_ms is not None:
                        timing["async_compute_ms"] = compute_ms
                    prefix_cache = new_si["executed_padded"][FLAGS.replan_steps - FLAGS.delay:FLAGS.replan_steps]
                else:
                    action_chunk, agent, new_si = agent.sample_actions(
                        observation,
                        only_base_actions=FLAGS.only_base_actions,
                    )
                    simulate_latency()
                action_chunk = np.asarray(jax.device_get(action_chunk))
                if action_chunk.ndim == 1:
                    action_chunk = action_chunk[None, :]
                action_plan.extend(list(action_chunk[: FLAGS.replan_steps]))
                sample_info_history.append(new_si)
            else:
                sample_info_history.append(sample_info_history[-1] if sample_info_history else None)
            timing["plan_ms"] = (time.time() - t_plan0) * 1000.0
            action = action_plan.popleft()

            ep_return += reward
            ep_len += 1

            if done:
                timing_total_ms = (time.time() - step_t0) * 1000.0
                logger.info(
                    "[timing][ep %d step %d] total=%.1fms wait=%.1f obs=%.1f info=%.1f plan=%.1f act=%.1f done=%s",
                    ep + 1,
                    step,
                    timing_total_ms,
                    timing["wait_ms"],
                    timing["obs_ms"],
                    timing["info_ms"],
                    timing["plan_ms"],
                    timing["act_ms"],
                    done,
                )
                break

            # Pace command dispatches at control_hz. Anchor to the previous
            # env.step dispatch time, not the previous env.step return time.
            if last_control_start is None:
                sleep_left = 0.0
            else:
                sleep_left = dt - (time.time() - last_control_start)
            if sleep_left > 0:
                t_wait0 = time.time()
                time.sleep(sleep_left)
                timing["wait_ms"] = (time.time() - t_wait0) * 1000.0

            last_control_start = time.time()
            t_act0 = time.time()
            if FLAGS.mirror_y:
                action = physical_action(action, mirror=True)
            _, action_type = env.step(np.asarray(action).tolist())
            ep_human_steps += int(action_type == "human")
            timing["act_ms"] = (time.time() - t_act0) * 1000.0

            timing_total_ms = (time.time() - step_t0) * 1000.0
            logger.info(
                "[timing][ep %d step %d] total=%.1fms wait=%.1f obs=%.1f info=%.1f plan=%.1f act=%.1f done=%s",
                ep + 1,
                step,
                timing_total_ms,
                timing["wait_ms"],
                timing["obs_ms"],
                timing["info_ms"],
                timing["plan_ms"],
                timing["act_ms"],
                done,
            )

        pending_inference = clear_pending(pending_inference)

        successes.append(success)
        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        logger.info("  success=%s return=%.1f len=%d human_override_steps=%d",
                    success, ep_return, ep_len, ep_human_steps)

    n = len(successes)
    success_rate = float(np.mean(successes))
    mean_return = float(np.mean(episode_returns))
    mean_len = float(np.mean(episode_lengths))
    logger.info("Evaluation complete: success_rate=%.2f (%d/%d) mean_return=%.2f mean_len=%.1f",
                success_rate, int(np.sum(successes)), n, mean_return, mean_len)
    print(f"success_rate={success_rate:.2f} mean_return={mean_return:.2f} mean_len={mean_len:.1f}")
    if inference_executor is not None:
        inference_executor.shutdown(wait=True)


if __name__ == "__main__":
    app.run(main)
