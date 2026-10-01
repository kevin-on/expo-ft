"""Synchronous split roles. Only Channel sees a local mailbox; no network config.

Rollout still owns the existing WS RPC loop. Learner receives complete rounds;
individual canonical transitions are sent immediately in the background.
"""
import json
import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
import re
import sys
import time

import jax
import numpy as np

from .channel import Channel, atomic_json
from .policy import export_policy, import_policy, identity
from .protocol import key, receive_round, task_contract, training_round
from .learner_group import LearnerGroup, local_value, replicate
from expo_ft.utils.log_utils import log_round_interventions
from expo_ft.utils.rollout_dashboard import RolloutDashboard, RolloutProgress


def _channel(flags):
    if not flags.split_session or not flags.split_mailbox:
        raise ValueError('split roles require a fresh shared --split_session and local --split_mailbox')
    if not re.fullmatch('[A-Za-z0-9_-]+', flags.split_session):
        raise ValueError('split_session must contain only letters, numbers, dash and underscore')
    channel = Channel(flags.split_mailbox, timeout=flags.split_timeout)
    try:
        channel.start_session(flags.split_session)
    except BaseException:
        channel.close()
        raise
    return channel


def _abort(channel, session):
    try:
        channel.send('abort', session, {'aborted': True})
        channel.flush()
    except Exception:
        logging.exception('Could not publish peer abort; peer timeout remains active')


def run_learner(flags, agent, buffers, batch_processor, checkpoint_manager, checkpoint_dir,
                save_checkpoint, start_step, resuming, replicated_sharding, mirror_robot,
                *, channel=None, policy_exchange=None, contract=None):
    import wandb
    from expo_ft.data.replay_buffer import (
        prepare_robot_replay_resume, restore_replay_buffer, save_replay_buffer_batch,
    )
    from expo_ft.utils.robot_round import updates_for_round

    directory = Path(checkpoint_dir)
    group = LearnerGroup()
    def connect():
        nonlocal channel
        if channel is None:
            channel = _channel(flags)

    group.call(connect)
    session = flags.split_session
    if contract is None:
        contract = identity(agent, task_contract(flags, mirror_robot))
    if group.call(lambda: contract) != contract:
        raise ValueError('learner nodes have different policy identities')
    episode_count, pending_steps, step, round_id = 0, 0, start_step, 0
    progress = RolloutProgress(len(buffers))
    combine_rng = jax.random.PRNGKey(flags.seed + 100)
    inference_rng = None
    if resuming:
        state = group.call(lambda: json.loads((directory / f'split-{start_step}.json').read_text()))
        if state['identity'] != contract:
            raise ValueError('split resume contract changed')
        if state['last_session'] == session:
            raise ValueError('resume requires a fresh session ID')
        episode_count, pending_steps = state['episode_count'], state['pending_steps']
        saved_progress = state.get('rollout_progress')
        progress = RolloutProgress(len(buffers), saved_progress)
        combine_rng = replicate(np.asarray(state['combine_rng'], dtype=np.uint32), replicated_sharding)
        inference_rng = state['inference_rng']
        group.call(lambda: prepare_robot_replay_resume(
            directory, up_to_step=step, num_robot=len(buffers),
            abandoned_dir=directory / 'abandoned-replay' / session))
        for index, buffer in enumerate(buffers):
            # Old checkpoints have no TUI totals. Rebuild them during the existing
            # replay read, rather than reading all saved images a second time.
            restore_replay_buffer(directory / f'robot-{index}', buffer, up_to_step=start_step,
                on_restore=(lambda record, robot=index: progress.record(robot, record))
                           if saved_progress is None and group.leader else None)
            buffer.restore_success_marks()
        progress = RolloutProgress(len(buffers), group.call(progress.snapshot))
    version = step
    policy_started = time.monotonic()
    policy_metrics = {}

    def publish_policy():
        if policy_exchange is not None:
            policy_metrics.update(policy_exchange.publish(agent, contract, version))
            return
        started = time.monotonic()
        with export_policy(agent, contract, version) as snapshot:
            serialized = time.monotonic()
            channel.send_buffer('policy', key(session, version), snapshot)
            policy_metrics.update(policy_bytes=snapshot.size, export_seconds=serialized - started,
                                  hash_handoff_seconds=time.monotonic() - serialized)

    group.call(publish_policy)
    last_checkpoint = start_step
    save_request = directory / 'save.request'

    def checkpoint():
        # The checkpoint cursor is independent from transport receipt IDs.
        group.call(lambda: atomic_json(directory / f'split-{step}.json', {
            'identity': contract, 'episode_count': episode_count, 'pending_steps': pending_steps,
            'rollout_progress': progress.snapshot(),
            'combine_rng': local_value(combine_rng).tolist(),
            'inference_rng': inference_rng, 'last_session': session, 'last_round': round_id,
        }))
        save_checkpoint(checkpoint_manager, agent, step)

    try:
        while step < flags.max_steps:
            def admit():
                channel.send('admit', key(session, round_id), {
                    'version': version, 'identity': contract, 'inference_rng': inference_rng if round_id == 0 else None,
                    'rollout_progress': progress.snapshot(), 'total_transitions': step,
                    'completed_rounds': episode_count // len(buffers),
                })
                channel.flush()
                return channel.receive('installed', key(session, round_id))

            ready = group.call(admit)
            if ready['version'] != version:
                raise ValueError('inference has not installed the required version')
            if policy_started is not None:
                policy_metrics.update(ready.get('timings', {}))
                policy_metrics['through_inference_ready_seconds'] = time.monotonic() - policy_started
                logging.info('POLICY_READY version=%d timing=%s', version, policy_metrics)
                if group.leader:
                    wandb.log({f'split/{k}': v for k, v in policy_metrics.items()}, step=step)
                policy_started = None
            def receive():
                episodes = receive_round(channel, session, round_id, version, flags.num_robot)
                finished = channel.receive('round_finished', key(session, round_id))
                return episodes, finished

            received_episodes, finished = group.call(receive)
            if finished['version'] != version:
                raise ValueError('round completion version mismatch')
            inference_rng = finished['inference_rng']
            episodes = training_round(received_episodes)
            # Persist both episodes before releasing transport buffers or admitting
            # the round to replay. A failed save cannot permit reset/update.
            for records, success in episodes:
                for record in records:
                    record['is_success'] = success

            def persist_and_release():
                next_step = step + 1
                for robot, (records, success) in enumerate(episodes):
                    if flags.checkpoint_buffer:
                        save_replay_buffer_batch(directory / f'robot-{robot}', records, start_step=next_step)
                    next_step += len(records)
                # Small transport receipts remain for deduplication. With replay
                # checkpointing disabled, persistence is deliberately skipped.
                for robot, (records, _) in enumerate(received_episodes):
                    for index in range(len(records)):
                        channel.release('transition', key(session, round_id, robot, index))
                    channel.release('episode_end', key(session, round_id, robot))
                channel.release('installed', key(session, round_id))
                channel.release('round_finished', key(session, round_id))
            group.call(persist_and_release)
            metrics = {}
            round_steps = sum(len(records) for records, _ in episodes)
            for robot, (buffer, (records, success)) in enumerate(zip(buffers, episodes)):
                for record in records:
                    buffer.insert(record)
                    progress.record(robot, record)
                    step += 1
                metrics[f'robot-{robot}/success'] = float(success)
                control_records = received_episodes[robot][0]
                metrics[f'robot-{robot}/episode_length'] = len(control_records)
                metrics[f'robot-{robot}/return'] = sum(float(r['rewards']) for r in control_records)
            episode_count += len(episodes)
            count, pending_steps = updates_for_round(
                pending_steps, round_steps, can_update=episode_count >= flags.split_warmup_episodes * len(buffers) and step >= flags.batch_size,
                num_updates=flags.num_updates, step_interval=flags.step_interval,
            )
            if step < flags.max_steps:
                # Permit only the next reset. The normal admit/policy messages
                # still gate rollout until updates and checkpointing finish.
                group.barrier('replay-ready')
                def prepare_reset():
                    channel.send('prepare_reset', key(session, round_id + 1), {'identity': contract})
                    channel.flush()
                group.call(prepare_reset)
            update_started = time.monotonic()
            for _ in range(count):
                sample_args = ({'update_step': int(local_value(agent.actor_train_state.step))}
                               if group.size > 1 else {})
                batch, actor_batch, combine_rng = batch_processor.next_batch(combine_rng, **sample_args)
                agent = agent.replace(rng=jax.device_put(agent.rng, replicated_sharding))
                agent, info = agent.update(agent, batch, flags.utd_ratio, actor_batch)
                jax.block_until_ready(info)
                info = jax.tree.map(local_value, info)
                if not all(np.isfinite(v).all() for v in jax.tree.leaves(info)):
                    raise FloatingPointError('nonfinite learner update; no new policy admitted')
                metrics.update({f'training/{name}': value for name, value in info.items()})
            update_finished = time.monotonic()
            if count:
                metrics['split/update_seconds'] = update_finished - update_started
            log_round_interventions(received_episodes, metrics)
            metrics.update(episodes=episode_count, updates=count, round_steps=round_steps, policy_version=version)
            # Leave this step open for the next installed-policy timing ACK.
            # Otherwise W&B discards that second log at the already-committed step.
            if group.leader:
                wandb.log(metrics, step=step, commit=not (count and step < flags.max_steps))
            logging.info('Split round %d complete: %d episodes, %d transitions, %d updates', round_id, episode_count, step, count)
            manual_save = group.call(save_request.is_file)
            if manual_save or (flags.checkpoint_model and flags.checkpoint_interval > 0
                               and step - last_checkpoint >= flags.checkpoint_interval):
                checkpoint()
                if manual_save:
                    checkpoint_manager.wait_until_finished()
                    group.barrier('manual-checkpoint-saved')
                    group.call(lambda: save_request.unlink(missing_ok=True))
                    if group.leader:
                        logging.info('Manual checkpoint saved at step %d in %s', step, directory)
                last_checkpoint = step
            if count and step < flags.max_steps:
                version = step
                policy_started = update_finished
                policy_metrics = {}
                group.call(publish_policy)
            round_id += 1
        if flags.checkpoint_model and step != last_checkpoint:
            checkpoint()
        # After any completed round inference waits for reset permission first.
        # A resume already at max_steps has no previous round and waits on admit.
        stop_topic = 'prepare_reset' if round_id else 'admit'
        def stop_inference():
            channel.send(stop_topic, key(session, round_id), {'stop': True})
            channel.flush()
            return channel.receive('stopped', session)
        group.call(stop_inference)
    except BaseException:
        if group.leader:
            _abort(channel, session)
        raise
    finally:
        if channel is not None:
            channel.close()
        checkpoint_manager.wait_until_finished()
    return agent


def build_inference(flags):
    from expo_ft.agents.alg.expo_ft import load_agent
    from expo_ft.agents.alg.batch_utils import CRITIC_CAMERA_KEYS
    from expo_ft.agents.vla.pi05 import build_pi05
    from expo_ft.data.replay_buffer import create_replay_buffer, _critic_key_to_storage
    from openpi.training import sharding

    mesh = sharding.make_mesh(1)
    if jax.device_count() != 1:
        raise ValueError('Expose exactly one GPU to the inference process')
    data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    actor, actor_state, _, kwargs, metadata = build_pi05(
        flags.config, flags.seed, mesh, data, replicated, False,
        flags.config_task.language_instruction, inference_only=True,
    )
    cameras = tuple(getattr(flags.config_task, 'critic_camera_keys', CRITIC_CAMERA_KEYS))
    # Reuse the current replay schema to determine graph input shapes, capacity ONE.
    buffer = create_replay_buffer(config=flags.config, example_action=flags.config_task.example_action,
        capacity=1, task_description=flags.config_task.language_instruction, replan_steps=flags.replan_steps,
        seed=flags.seed, delay=0, critic_camera_keys=cameras)
    example = { _critic_key_to_storage(k): np.zeros_like(buffer.dataset_dict[_critic_key_to_storage(k)][:1]) for k in cameras }
    example['state'] = np.zeros_like(buffer.dataset_dict['state'][:1])
    example['actions'] = np.zeros_like(buffer.dataset_dict['actions'][:1])
    observation, state, action = buffer.convert_to_critic_format(example)
    actor.action_dim, actor.state_dim = action.squeeze().shape[-1], state.squeeze().shape[-1]
    kwargs.update(inference_only=True, critic_camera_keys=cameras, rollout_cache=False)
    agent = load_agent(seed=flags.seed, example_observation=observation.squeeze(),
        example_action=action.squeeze(), example_state=state.squeeze(), actor=actor,
        actor_train_state=actor_state, target_actor_params=None, agent_kwargs=kwargs, metadata=metadata,
        mesh=mesh, data_sharding=data, replicated_sharding=replicated, resume=False,
        replan_steps=flags.replan_steps, default_prompt=flags.config_task.language_instruction,
        edit_action_xyzg=flags.config_task.edit_action_xyzg)
    from expo_ft.utils.model_config import make_record
    actor.checkpoint_record = make_record(flags.config, flags.config_task, flags.replan_steps, flags.num_robot)
    return agent.replace(rng=jax.random.fold_in(jax.random.PRNGKey(flags.seed), 7351))


def run_inference(flags, agent=None, env_factory=None, *, channel=None,
                  policy_exchange=None, contract=None):
    from expo_ft.env.env_client import EnvClientWrapper
    from expo_ft.utils.robot_round import collect_round

    if flags.resume and policy_exchange is None:
        raise ValueError('Only learner uses --resume; inference receives RNG/policy from learner')
    dashboard = None
    if getattr(flags, 'rollout_dashboard', False):
        if not sys.stdin.isatty() or not sys.stdout.isatty() or not flags.rollout_log:
            raise ValueError('Rollout dashboard requires ssh -tt / srun --pty and --rollout_log')
        dashboard = RolloutDashboard(flags.num_robot, flags.config_task.auto_reset_steps, flags.rollout_mode)
    elif getattr(flags, 'rollout_mode', 'auto') == 'manual':
        raise ValueError('Manual rollout requires --rollout_dashboard')
    if agent is None and policy_exchange is None:
        agent = build_inference(flags)
    mirror_robot = 1 if flags.num_robot == 2 else None
    if contract is None:
        contract = identity(agent, task_contract(flags, mirror_robot))
    if channel is None:
        channel = _channel(flags)
    session, round_id, installed = flags.split_session, 0, None
    envs = []
    reset_workers = ThreadPoolExecutor(max_workers=flags.num_robot, thread_name_prefix='robot-reset')
    resets = []

    def check_session():
        if dashboard is not None:
            dashboard.check()
        channel.check_session()

    def reset_robot(robot, env):
        if dashboard is not None:
            dashboard.resetting(robot)
        env.reset_only()
        if dashboard is not None:
            dashboard.reset_done(robot)

    def begin_resets():
        check_session()
        logging.info('Split round %d: resetting %d robots while waiting for policy', round_id, len(envs))
        if dashboard is not None:
            dashboard.set_phase('Reset + learner update / policy transfer')
        return [reset_workers.submit(reset_robot, robot, env) for robot, env in enumerate(envs)]

    def check_reset_errors():
        if dashboard is not None:
            dashboard.check()
        for reset in resets:
            if reset.done():
                reset.result()

    def finish_resets():
        pending = set(resets)
        while pending:
            check_session()
            done, pending = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
            for reset in done:
                reset.result()
        check_session()

    def acknowledge_stop():
        channel.send('stopped', session, {'stopped': True})
        channel.flush()
        channel.wait_sent('stopped', session)

    try:
        if dashboard is not None:
            dashboard.start(flags.rollout_log)
        while True:
            if round_id:
                prepare = channel.receive('prepare_reset', key(session, round_id), check=check_reset_errors)
                channel.release('prepare_reset', key(session, round_id))
                if prepare.get('stop'):
                    acknowledge_stop()
                    break
                if prepare['identity'] != contract:
                    raise ValueError('learner/inference reset configuration mismatch')
                resets = begin_resets()
            admit = channel.receive('admit', key(session, round_id), check=check_reset_errors)
            channel.release('admit', key(session, round_id))
            if admit.get('stop'):
                acknowledge_stop()
                break
            if admit['identity'] != contract:
                raise ValueError('learner/inference configuration mismatch')
            version = admit['version']
            if dashboard is not None and 'rollout_progress' in admit:
                dashboard.restore_progress(admit['rollout_progress'], admit['total_transitions'])
            install_metrics = {}
            if installed != version:
                if dashboard is not None:
                    dashboard.set_phase('Receiving / installing policy')
                if policy_exchange is not None:
                    agent, install_metrics = policy_exchange.install(
                        agent, contract, version, check=check_reset_errors)
                else:
                    with channel.receive_buffer('policy', key(session, version), check=check_reset_errors) as snapshot:
                        started = time.monotonic()
                        agent = import_policy(agent, snapshot, contract, version)
                        install_metrics = dict(snapshot.timings, install_seconds=time.monotonic() - started)
                channel.release('policy', key(session, version))
                installed = version
            if admit.get('inference_rng') is not None:
                agent = agent.replace(rng=jax.device_put(np.asarray(admit['inference_rng'], dtype=np.uint32), agent.actor.infer_sharding))
            # Only after compatible policy/config verification may robot RPCs be created.
            if not envs:
                for robot in range(flags.num_robot):
                    path = Path(__file__).resolve().parents[2] / f'configs/robots/robot-{robot}.json'
                    config = json.loads(path.read_text())
                    views = {k: config[k] for k in ('side_camera_id', 'wrist_camera_id')}
                    request = {'example_action': flags.config_task.example_action, 'env_usage': 'train',
                               'async_video': True,
                               'video_dir': str(Path(flags.output_dir) / flags.run_name / 'train_videos' / f'robot-{robot}'),
                               'expected_camera_views': views}
                    envs.append((env_factory or EnvClientWrapper)(env_creation_request=request,
                        host=flags.client_host, port=flags.client_port + robot, recover=False, lazy=True))
                resets = begin_resets()
            channel.send('installed', key(session, round_id), {'version': version, 'timings': install_metrics})
            finish_resets()
            resets = []
            if dashboard is not None:
                dashboard.ready(admit.get('completed_rounds', round_id), version)
            def sample(observation):
                nonlocal agent
                actions, agent, _ = agent.sample_actions(observation)
                return np.asarray(jax.device_get(actions))
            def transition(robot, step, record):
                channel.send('transition', key(session, round_id, robot, step), {'version': version, 'transition': record})
                if dashboard is not None:
                    dashboard.step(robot, step + 1, record['is_hil'], trainable=not record.get('is_handoff', False))
            def end(robot, length, success):
                channel.send('episode_end', key(session, round_id, robot), {'version': version, 'length': length, 'success': bool(success)})
                if dashboard is not None:
                    dashboard.episode_done(robot, success)
            def wait_for_start(robot, stopped):
                return dashboard.wait_for_start(robot, stopped, reset=envs[robot].reset_only)
            collect_round(envs, sample, flags.replan_steps, flags.config_task.control_hz,
                          mirror_robot=mirror_robot, on_transition=transition, on_episode_end=end,
                          check_session=check_session, reset_done=True, canonical_frame=True, mark_handoff=True,
                          **({'wait_for_start': wait_for_start} if dashboard is not None else {}))
            if dashboard is not None:
                dashboard.set_phase('Round complete; waiting for learner / replay')
            channel.send('round_finished', key(session, round_id), {
                'version': version, 'inference_rng': np.asarray(jax.device_get(agent.rng)).tolist(),
            })
            channel.flush()
            round_id += 1
        if dashboard is not None:
            dashboard.set_phase('Training complete')
    except BaseException:
        if dashboard is not None:
            dashboard.set_phase('Stopped / error; see inference log')
            logging.exception('Inference stopped')
        # Interrupt reset RPC waits before publishing an abort (which itself may
        # be delayed by a failed transport). Already issued NUC motion cannot be undone.
        for env in envs:
            env.close()
        _abort(channel, session)
        raise
    finally:
        for reset in resets:
            reset.cancel()
        try:
            for env in envs:
                env.close()
        finally:
            try:
                reset_workers.shutdown(wait=True)
                channel.close()
            finally:
                if dashboard is not None:
                    dashboard.close()
