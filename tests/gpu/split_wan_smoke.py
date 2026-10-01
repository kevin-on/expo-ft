"""Cross-cluster integration with read-only recorded robot trajectories.

Prepare a separate fixture; run learner/inference on their own allocated hosts
with production transports. Two loopback mock clients exercise the real WS RPC
protocol. They replay recorded actions/observations, not physical dynamics.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def record_hash(records):
    import numpy as np
    digest = hashlib.sha256()
    for row in records:
        obs = row['observations']
        for key in ('exterior_image_1_left', 'exterior_image_2_left', 'wrist_image_left'):
            digest.update(np.ascontiguousarray(obs[key], dtype=np.uint8).tobytes())
        for value in (obs['cartesian_position'], obs['gripper_position'], row['actions'],
                      [row['rewards'], row['masks'], row['dones'], row.get('is_hil', False)]):
            digest.update(np.asarray(value, dtype='<f8').ravel().tobytes())
        digest.update(obs.get('prompt', '').encode())
    return digest.hexdigest()


def prepare(args):
    args.fixture.mkdir(parents=True, exist_ok=False)
    manifest = {'source': str(args.recordings), 'rounds': args.rounds, 'robots': [],
                'original_files_modified': False, 'frame': 'canonical robot0'}
    reference = None
    for robot in range(2):
        destination = args.fixture / f'robot-{robot}'
        destination.mkdir()
        episodes, records, sources = [], [], []
        for path in sorted((args.recordings / 'checkpoints' / f'robot-{robot}' / 'buffers').glob('*.pkl')):
            raw = path.read_bytes()
            value = pickle.loads(raw)
            if '-' in path.stem:
                start, end = map(int, path.stem.split('-'))
                if (value.get('format') != 'replay-batch-v1' or value.get('start_step') != start
                        or value.get('end_step') != end or len(value['transitions']) != end - start + 1):
                    raise ValueError(f'Invalid replay batch: {path}')
                rows = value['transitions']
            else:
                rows = [value]
            source = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'size': len(raw)}
            for row in rows:
                if reference is None:
                    reference = deepcopy(row)
                records.append(row)
                if not sources or sources[-1]['path'] != str(path):
                    sources.append(source)
                if bool(row['dones']):
                    success = bool(row.get('is_success', row['rewards'] > 0))
                    name = f'{len(episodes):03d}.pkl'
                    (destination / name).write_bytes(pickle.dumps(records, protocol=5))
                    episodes.append({'file': f'robot-{robot}/{name}', 'length': len(records), 'success': success,
                                     'records_sha256': record_hash(records), 'sources': sources})
                    records, sources = [], []
                    if len(episodes) == args.rounds:
                        break
            if len(episodes) == args.rounds:
                break
        if len(episodes) != args.rounds:
            raise ValueError(f'robot {robot} has only {len(episodes)} complete episodes')
        manifest['robots'].append(episodes)
    manifest['transitions'] = sum(e['length'] for rows in manifest['robots'] for e in rows)
    (args.fixture / 'reference.pkl').write_bytes(pickle.dumps(reference, protocol=5))
    (args.fixture / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'rounds': args.rounds, 'transitions': manifest['transitions'],
                      'episode_lengths': [[e['length'] for e in rows] for rows in manifest['robots']],
                      'successes': [sum(e['success'] for e in rows) for rows in manifest['robots']]}), flush=True)


def mock_client(args):
    import numpy as np
    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect
    from websockets.exceptions import ConnectionClosedOK
    from expo_ft.env.sft_eval import canonical_observation, physical_action
    manifest = json.loads((args.fixture / 'manifest.json').read_text())
    episodes = manifest['robots'][args.robot]
    round_id, index, rows, policy_commands = -1, 0, [], 0
    deadline = time.monotonic() + 1800
    while True:
        try:
            ws = connect(f'ws://127.0.0.1:{args.port + args.robot}', max_size=None, compression=None)
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.2)
    packer = msgpack_numpy.Packer()
    with ws:
        try:
            while True:
                request = msgpack_numpy.unpackb(ws.recv())
                operation = request['operation']
                if operation == 'create_env':
                    reply = {'env_id': f'mock-{args.robot}', 'task_description': 'pick up the cube'}
                elif operation in ('reset', 'reset_only'):
                    round_id += 1
                    index = 0
                    rows = pickle.loads((args.fixture / episodes[round_id]['file']).read_bytes())
                    reply = {'status': 'success'}
                    if operation == 'reset':
                        reply.update(observation=canonical_observation(rows[0]['observations'], args.robot == 1), done=False)
                elif operation == 'start_episode':
                    # Reset already selected this episode; read its first frame
                    # without advancing the episode or the recorded action cursor.
                    reply = {'status': 'success',
                             'observation': canonical_observation(rows[0]['observations'], args.robot == 1)}
                elif operation == 'step':
                    command = np.asarray(request['action'])
                    assert command.shape == (7,) and np.isfinite(command).all()
                    row = rows[index]
                    reply = {'action': physical_action(row['actions'], args.robot == 1),
                             'action_type': 'human' if row.get('is_hil', False) else 'policy'}
                    index += 1
                    policy_commands += 1
                elif operation == 'get_observation':
                    row = rows[index-1]
                    reply = {'observation': canonical_observation(rows[min(index, len(rows)-1)]['observations'], args.robot == 1),
                             'done': bool(row['dones']), 'success': episodes[round_id]['success'],
                             'reward': float(row['rewards']), 'mask': float(row['masks'])}
                else:
                    raise AssertionError(operation)
                ws.send(packer.pack(reply))
        except ConnectionClosedOK:
            pass
    assert round_id + 1 == args.rounds and index == len(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f'mock-{args.robot}-passed.json').write_text(json.dumps({
        'rounds': round_id+1, 'commands_received': policy_commands, 'recorded_actions_replayed': True}))


def parameter_hash(buffer):
    from expo_ft.distributed.policy import _read_manifest
    digest = hashlib.sha256()
    with buffer.view() as payload:
        for item in _read_manifest(payload)['arrays']:
            digest.update(json.dumps([item[k] for k in ('group', 'path', 'shape', 'dtype')]).encode())
            digest.update(payload[item['offset']:item['offset'] + item['nbytes']])
    return digest.hexdigest()


def tree_hash(tree):
    import jax
    import numpy as np
    digest = hashlib.sha256()
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        array = np.asarray(jax.device_get(value))
        digest.update(repr((path, array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def diagnostic_actions(agent, observation):
    """Observe candidate scores without changing sampling or selection."""
    import jax
    import numpy as np
    from expo_ft.agents.alg import expo_ft as algorithm
    original = algorithm.compute_q
    details = {}
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        details['candidate_actions'] = np.asarray(jax.device_get(args[3]))
        details['q_values'] = np.asarray(jax.device_get(result))
        return result
    algorithm.compute_q = capture
    try:
        actions, _, _ = agent.replace(rng=jax.random.PRNGKey(123)).sample_actions(deepcopy(observation))
        details['actions'] = np.asarray(jax.device_get(actions))
        details['selected_index'] = int(np.argmax(details['q_values']))
        return details
    finally:
        algorithm.compute_q = original


def node(args):
    """Supervise only this test's model, transport and optional mock processes."""
    args.output.mkdir(parents=True, exist_ok=True)
    children, logs = [], []
    def interrupted(*_):
        raise KeyboardInterrupt()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    def launch(name, command):
        suffix = '-rank-' + os.environ.get('EXPO_PROCESS_ID', '0') if args.node_role == 'learner' else ''
        log = (args.output / f'{name}{suffix}.log').open('w')
        logs.append(log)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        children.append(process)
        return process
    passed = False
    try:
        leader = args.node_role == 'inference' or int(os.environ.get('EXPO_PROCESS_ID', '0')) == 0
        transport = launch('transport', [sys.executable, '-u', '-m', 'expo_ft.distributed.transport',
                                         '--config', args.transport_config]) if leader else None
        base = [sys.executable, '-u', __file__, '--fixture', str(args.fixture), '--params', str(args.params),
                '--assets', str(args.assets), '--asset-id', args.asset_id, '--session', args.session,
                '--rounds', str(args.rounds), '--warmup-episodes', str(args.warmup_episodes),
                '--playback-hz', str(args.playback_hz), '--port', str(args.port), '--mailbox', args.mailbox]
        if args.performance:
            base.append('--performance')
        model = launch(args.node_role, base + ['--role', args.node_role, '--output', str(args.output / args.node_role)])
        workload = [model]
        if args.node_role == 'inference':
            for robot in range(2):
                workload.append(launch(f'mock-{robot}', base + ['--role', 'mock', '--robot', str(robot),
                                                               '--output', str(args.output)]))
        deadline = time.monotonic() + 5400
        while any(child.poll() is None for child in workload):
            if (transport is not None and transport.poll() is not None) or any(child.poll() not in (None, 0) for child in workload):
                raise RuntimeError('test child failed; inspect the role logs')
            if time.monotonic() >= deadline:
                raise TimeoutError('cross-cluster integration')
            time.sleep(1)
        passed = all(child.returncode == 0 for child in workload)
        if not passed:
            raise RuntimeError('test child failed')
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for log in logs:
            log.close()
        (args.output / ('node-result-' + os.environ.get('EXPO_PROCESS_ID', '0') + '.json')).write_text(
            json.dumps({'passed': passed, 'role': args.node_role}))


def model_role(args):
    import faulthandler
    faulthandler.register(signal.SIGUSR1, all_threads=False)
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    import jax
    import numpy as np
    import wandb
    import expo_ft.agents  # keep the established agents/data import order
    from configs.model.expo_ft_pi_config import get_config
    from configs.task.pick import get_config as get_task
    from expo_ft.distributed import runner
    from expo_ft.agents import initialize_checkpoint_dir
    from expo_ft.agents.alg.batch_utils import CRITIC_CAMERA_KEYS
    from expo_ft.agents.alg.expo_ft import load_agent, save_checkpoint, restore_checkpoint
    from expo_ft.agents.vla.pi05 import build_pi05
    from expo_ft.data.batch_processor import BatchProcessor
    from expo_ft.data.replay_buffer import create_replay_buffer, _critic_key_to_storage
    from openpi.training import sharding

    from expo_ft.distributed.learner_group import initialize_learner, LearnerGroup, local_value
    from expo_ft.data.distributed_sampler import DistributedReplaySampler
    if args.role == 'learner':
        initialize_learner(SimpleNamespace(split_role='learner', fsdp_devices=1, offline_ratio=0, num_robot=2))
    group = LearnerGroup()
    assert jax.default_backend() == 'gpu'
    assert jax.device_count() == (4 * group.size if args.role == 'learner' else 1)
    if group.size > 1 and not args.performance:
        raise ValueError('Multi-host WAN test uses --performance; action diagnostics require a local inference cache')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.fixture / 'manifest.json').read_text())
    reference = pickle.loads((args.fixture / 'reference.pkl').read_bytes())
    task, config = get_task(), get_config()
    task.control_hz = args.playback_hz  # recorded playback only; never real hardware
    from config_fixture import configure
    configure(config, args.output / ('config-fixture-'+str(jax.process_index())+'-'+args.role), args.params, args.assets, args.asset_id)
    flags = SimpleNamespace(config=config, config_task=task, seed=42, replan_steps=8, num_robot=2,
        split_session=args.session, split_mailbox=args.mailbox, split_timeout=1800, resume=False,
        split_warmup_episodes=args.warmup_episodes,
        client_host='127.0.0.1', client_port=args.port, output_dir=str(args.output), run_name='recorded-wan',
        max_steps=manifest['transitions'], batch_size=64, utd_ratio=20, num_updates=3, step_interval=50,
        checkpoint_model=True, checkpoint_buffer=True, checkpoint_interval=0)
    cumulative = np.cumsum([sum(robot[i]['length'] for robot in manifest['robots'])
                            for i in range(args.rounds)])
    update_rounds = [i + 1 >= args.warmup_episodes and steps >= flags.batch_size
                     for i, steps in enumerate(cumulative)]
    channel = None
    original_channel = runner._channel
    def capture_channel(flags):
        nonlocal channel
        channel = original_channel(flags)
        return channel
    runner._channel = capture_channel
    versions, parameter_hashes = [], []

    if args.role == 'inference':
        from expo_ft.utils import robot_round
        original_collect = robot_round.collect_round
        rollout_index = 0
        def measured_collect(*values, **kwargs):
            nonlocal rollout_index
            started = time.monotonic()
            result = original_collect(*values, **kwargs)
            with (args.output / 'rollout-timing.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(round=rollout_index, start=started,
                                             end=time.monotonic())) + '\n')
            rollout_index += 1
            return result
        robot_round.collect_round = measured_collect
        original = runner.import_policy
        def checked_import(agent, buffer, contract, version):
            if args.performance:
                # Initial and first updated policies get exact device-parameter
                # verification, outside the five subsequent measured cycles.
                verify = len(versions) < 2
                expected = channel.receive('validation-policy', str(version)) if verify else None
                if verify:
                    channel.release('validation-policy', str(version))
                    assert parameter_hash(buffer) == expected['parameter_sha256']
                agent = original(agent, buffer, contract, version)
                if verify:
                    with runner.export_policy(agent, contract, version) as installed:
                        assert parameter_hash(installed) == expected['parameter_sha256']
                    (args.output / f'installed-{version}.json').write_text(json.dumps({
                        'version': version, 'installed_parameters_exact': True,
                        'parameter_sha256': expected['parameter_sha256']}))
                versions.append(version)
                return agent
            validation_start = time.monotonic()
            expected = channel.receive('validation-policy', str(version))
            channel.release('validation-policy', str(version))
            digest = parameter_hash(buffer)
            assert digest == expected['parameter_sha256']
            before_rng = np.asarray(agent.rng).copy()
            install_started = time.monotonic()
            agent = original(agent, buffer, contract, version)
            install_seconds = time.monotonic() - install_started
            np.testing.assert_array_equal(agent.rng, before_rng)
            assert agent.critic is None and not jax.tree.leaves(agent.actor_train_state.opt_state)
            # Validate the actual device parameters after installation, not just
            # the received bytes. Cross-platform action arithmetic is reported
            # separately; it is not a substitute for this exact equality check.
            with runner.export_policy(agent, contract, version) as installed:
                assert parameter_hash(installed) == digest
            inputs = agent.actor.process_raw_inputs(deepcopy(reference['observations']), agent.action_dim, agent.resize_size)
            assert tree_hash(inputs) == expected['inputs_sha256']
            actual = diagnostic_actions(agent, reference['observations'])
            actions = actual['actions']
            repeated, _, _ = agent.replace(rng=jax.random.PRNGKey(123)).sample_actions(deepcopy(reference['observations']))
            np.testing.assert_array_equal(actions, np.asarray(jax.device_get(repeated)))
            assert np.isfinite(actions).all()
            cross_host_close = bool(np.allclose(actions, expected['actions'], rtol=1e-4, atol=1e-5))
            np.savez(args.output / f'actions-{version}.npz', learner=expected['actions'], inference=actions,
                learner_candidates=expected['candidate_actions'], inference_candidates=actual['candidate_actions'],
                learner_q=expected['q_values'], inference_q=actual['q_values'])
            versions.append(version)
            parameter_hashes.append(digest)
            (args.output / f'installed-{version}.json').write_text(json.dumps({
                'version': version, 'parameter_sha256': digest, 'installed_parameters_exact': True,
                'model_inputs_exact': True, 'same_host_repeat_exact': True,
                'cross_host_action_allclose': cross_host_close, 'production_install_seconds': install_seconds,
                'learner_selected_index': expected['selected_index'], 'inference_selected_index': actual['selected_index'],
                'max_candidate_difference': float(np.max(np.abs(actual['candidate_actions']-expected['candidate_actions']))),
                'max_q_difference': float(np.max(np.abs(actual['q_values']-expected['q_values']))),
                'learner_top_q_gap': float(np.diff(np.sort(expected['q_values'].ravel())[-2:])[0]),
                'inference_top_q_gap': float(np.diff(np.sort(actual['q_values'].ravel())[-2:])[0]),
                'max_action_difference': float(np.max(np.abs(actions-expected['actions']))),
                'install_with_validation_seconds': time.monotonic()-validation_start}))
            return agent
        runner.import_policy = checked_import
        runner.run_inference(flags)
        assert len(versions) == 1 + sum(update_rounds[:-1])
        assert len(set(parameter_hashes)) == len(parameter_hashes)
        (args.output / 'inference-passed.json').write_text(json.dumps({'versions': versions, 'parameter_hashes': parameter_hashes}))
        return

    mesh = sharding.make_mesh(1)  # four GPUs, data parallel, FSDP=1
    data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    actor, actor_state, target, kwargs, metadata = build_pi05(config, 42, mesh, data, replicated, False, task.language_instruction)
    from expo_ft.utils.model_config import make_record
    actor.checkpoint_record = make_record(config, task, 8, 2)
    buffer_args = dict(config=config, example_action=reference['actions'][None],
        capacity=manifest['transitions'] + 1024, task_description=task.language_instruction,
        replan_steps=8, delay=0, critic_camera_keys=CRITIC_CAMERA_KEYS)
    buffers = [create_replay_buffer(**buffer_args, seed=42+i+100003*jax.process_index()) for i in range(2)]
    offline = create_replay_buffer(**dict(buffer_args, capacity=1), seed=42)
    sampler = (DistributedReplaySampler(seed=42, global_batch_size=64,
                rank=jax.process_index(), world_size=group.size) if group.size > 1 else None)
    if sampler is not None:
        original_sample = sampler.sample
        def measured_sample(candidates, **options):
            result = original_sample(candidates, **options)
            counts = [int(c) if np.isscalar(c) else len(c) for c in candidates]
            with (args.output / f'sampling-rank-{jax.process_index()}.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(**options, counts=counts,
                    selected=None if result is None else result.tolist())) + '\n')
            return result
        sampler.sample = measured_sample
    processor = BatchProcessor(buffers[0], offline, data, 64 // group.size, 20, 0, config.actor_success_only, False,
                               replay_buffers=buffers, utd_axis=group.size > 1, distributed_sampler=sampler)
    original_next_batch = processor.next_batch
    def measured_next_batch(rng, **sample_options):
        started = time.monotonic()
        result = original_next_batch(rng, **sample_options)
        if not (args.output / f'batch-shapes-rank-{jax.process_index()}.json').exists():
            summary = jax.tree.map(lambda x: dict(shape=list(x.shape), dtype=str(x.dtype),
                                                  sharding=str(x.sharding)), result[:2])
            (args.output / f'batch-shapes-rank-{jax.process_index()}.json').write_text(json.dumps(summary))
        with (args.output / f'batch-timing-rank-{jax.process_index()}.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(seconds=time.monotonic()-started)) + '\n')
        return result
    processor.next_batch = measured_next_batch
    example = {_critic_key_to_storage(k): buffers[0].dataset_dict[_critic_key_to_storage(k)][:1] for k in CRITIC_CAMERA_KEYS}
    example.update(state=buffers[0].dataset_dict['state'][:1], actions=buffers[0].dataset_dict['actions'][:1])
    obs, state, action = buffers[0].convert_to_critic_format(example)
    actor.action_dim, actor.state_dim = action.squeeze().shape[-1], state.squeeze().shape[-1]
    kwargs.update(critic_camera_keys=CRITIC_CAMERA_KEYS, rollout_cache=False)
    agent = load_agent(seed=42, example_observation=obs.squeeze(), example_action=action.squeeze(), example_state=state.squeeze(),
        actor=actor, actor_train_state=actor_state, target_actor_params=target, agent_kwargs=kwargs, metadata=metadata,
        mesh=mesh, data_sharding=data, replicated_sharding=replicated, resume=False, replan_steps=8,
        default_prompt=task.language_instruction, edit_action_xyzg=task.edit_action_xyzg)
    manager, _ = initialize_checkpoint_dir(args.output / 'checkpoints', keep_period=None, overwrite=False, resume=False)
    wandb.init(mode='disabled')
    original_log = wandb.log
    def record_metrics(values, *args_, **kwargs_):
        with (args.output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': kwargs_.get('step'), **values},
                                    default=lambda value: np.asarray(value).tolist()) + '\n')
        if 'split/update_seconds' in values:
            with (args.output / 'update-timing.jsonl').open('a') as stream:
                stream.write(json.dumps({'step': kwargs_.get('step'), 'updates': values['updates'],
                    'seconds': values['split/update_seconds']})+'\n')
        return original_log(values, *args_, **kwargs_)
    wandb.log = record_metrics
    # Measure the actual durable writer separately from transport and replay work.
    from expo_ft.data import replay_buffer as replay_persistence
    original_save_batch = replay_persistence.save_replay_buffer_batch
    def timed_save_batch(path, records, *, start_step):
        started = time.monotonic()
        original_save_batch(path, records, start_step=start_step)
        with (args.output / 'replay-save-timing.jsonl').open('a') as stream:
            stream.write(json.dumps({'robot': Path(path).name, 'start_step': start_step,
                                     'records': len(records), 'seconds': time.monotonic()-started})+'\n')
    replay_persistence.save_replay_buffer_batch = timed_save_batch
    original_export, original_round = runner.export_policy, runner.receive_round
    round_number = 0
    def checked_round(*args_):
        nonlocal round_number
        episodes = original_round(*args_)
        for robot, (records, success) in enumerate(episodes):
            expected = manifest['robots'][robot][round_number]
            assert len(records) == expected['length'] and bool(success) == expected['success']
            if not args.performance:
                assert record_hash(records) == expected['records_sha256'], (robot, round_number)
        print('RECORDED_ROUND_VERIFIED', round_number, [len(e[0]) for e in episodes], flush=True)
        round_number += 1
        return episodes
    def checked_export(agent, contract, version):
        start = time.monotonic()
        buffer = original_export(agent, contract, version)
        export_seconds = time.monotonic() - start
        if args.performance:
            if len(versions) < 2:
                channel.send('validation-policy', str(version), {'parameter_sha256': parameter_hash(buffer)})
                channel.flush()
            versions.append(version)
            print('EXPORT_TIMING', json.dumps(dict(version=version, seconds=export_seconds)), flush=True)
            return buffer
        digest = parameter_hash(buffer)
        details = diagnostic_actions(agent.cache_infer_params(), reference['observations'])
        inputs = agent.actor.process_raw_inputs(deepcopy(reference['observations']), agent.action_dim, agent.resize_size)
        channel.send('validation-policy', str(version), {'parameter_sha256': digest,
            'inputs_sha256': tree_hash(inputs), **details})
        channel.flush()
        versions.append(version)
        parameter_hashes.append(digest)
        print('EXPORT_VALIDATED', version, buffer.size, time.monotonic()-start, flush=True)
        (args.output / f'export-{version}.json').write_text(json.dumps({
            'version': version, 'bytes': buffer.size, 'production_export_seconds': export_seconds,
            'export_with_validation_seconds': time.monotonic()-start, 'parameter_sha256': digest}))
        return buffer
    runner.receive_round, runner.export_policy = checked_round, checked_export
    agent = runner.run_learner(flags, agent, buffers, processor, manager, args.output / 'checkpoints', save_checkpoint,
                              0, False, replicated, 1)
    updates = sum(update_rounds) * flags.num_updates
    restored = restore_checkpoint(manager, agent)
    assert int(local_value(restored.actor_train_state.step)) == updates
    assert int(local_value(restored.critic.step)) == updates * flags.utd_ratio
    if sampler is not None:
        # Compare the next batch after an actual checkpoint restore, including
        # indexed n-step/image gathering, before any further gradient update.
        next_step = int(local_value(restored.actor_train_state.step))
        expected = processor.next_batch(jax.random.PRNGKey(42), update_step=next_step)
        resumed = BatchProcessor(buffers[0], offline, data, 64 // group.size, 20, 0,
            config.actor_success_only, False, replay_buffers=buffers, utd_axis=True,
            distributed_sampler=DistributedReplaySampler(seed=42, global_batch_size=64,
                rank=jax.process_index(), world_size=group.size))
        actual = resumed.next_batch(jax.random.PRNGKey(42), update_step=next_step)
        for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
            local_a = [s.data for s in a.addressable_shards] if isinstance(a, jax.Array) else [a]
            local_b = [s.data for s in b.addressable_shards] if isinstance(b, jax.Array) else [b]
            assert len(local_a) == len(local_b)
            for x, y in zip(local_a, local_b):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        group.barrier('sampling-logs-ready')
        logs = [[json.loads(line) for line in
                 (args.output / f'sampling-rank-{rank}.jsonl').read_text().splitlines()]
                for rank in range(group.size)]
        assert all(len(rows) == len(logs[0]) for rows in logs)
        for rows in zip(*logs):
            for row in rows[1:]:
                assert {k: v for k, v in row.items() if k != 'selected'} == {
                    k: v for k, v in rows[0].items() if k != 'selected'}
            if rows[0]['selected'] is None:
                assert all(row['selected'] is None for row in rows)
                continue
            plan = np.concatenate([np.asarray(row['selected']) for row in rows], axis=1)
            assert plan.shape == (rows[0]['num_batches'], 64, 2)
            # Repeated (robot, row) pairs are expected with replacement. Rank
            # slices still compose one global batch; exact seeded plans are
            # covered by cpu/test_distributed_sampler.py.
    round_number = group.call(lambda: round_number)
    assert round_number == args.rounds
    assert len(set(parameter_hashes)) == len(parameter_hashes)
    checkpoint_dir = args.output / 'checkpoints'
    replay_persistence.prepare_robot_replay_resume(checkpoint_dir, up_to_step=manifest['transitions'], num_robot=2)
    assert not list(checkpoint_dir.rglob('*.records'))
    for robot in range(2):
        files = replay_persistence._replay_files(checkpoint_dir / f'robot-{robot}/buffers')
        assert len(files) == args.rounds
        for (start, end, path), expected in zip(files, manifest['robots'][robot]):
            records = replay_persistence._load_replay_file(start, end, path)
            assert '-' in path.stem and len(records) == expected['length']
            assert record_hash(records) == expected['records_sha256']
            assert all(bool(row['is_success']) == expected['success'] for row in records)
    manager.close()
    wandb.finish()
    (args.output / f'learner-passed-rank-{jax.process_index()}.json').write_text(json.dumps({'rounds': round_number, 'updates': updates,
        'devices': jax.device_count(), 'batch_size': 64, 'utd_ratio': 20, 'critic_updates': int(local_value(restored.critic.step)), 'versions': versions,
        'transitions': manifest['transitions'], 'records_verified': True, 'checkpoint_restore': True,
        'batch_replay_files': args.rounds * 2, 'batch_replay_verified': True, 'duplicate_round_archives': False,
        'distributed_sampling_verified': sampler is not None,
        'restored_update_sampling_exact': sampler is not None}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['prepare', 'mock', 'learner', 'inference', 'node'], required=True)
    parser.add_argument('--node-role', choices=['learner', 'inference'])
    parser.add_argument('--transport-config')
    parser.add_argument('--recordings', type=Path)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--params', type=Path)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--asset-id', default='expo_ft/pick_cube_balance_0923_mixed_20_seed3')
    parser.add_argument('--mailbox')
    parser.add_argument('--session', default='wan-recorded-20260925')
    parser.add_argument('--rounds', type=int, default=12)
    parser.add_argument('--warmup-episodes', type=int, default=10)
    parser.add_argument('--performance', action='store_true', help='Check initial and first updated GPU parameters exactly; omit expensive action diagnostics in measured rounds')
    parser.add_argument('--playback-hz', type=int, default=1000,
                        help='Mock playback rate; use 10 for real-time collection cadence')
    parser.add_argument('--robot', type=int)
    parser.add_argument('--port', type=int, default=19400)
    args = parser.parse_args()
    if args.playback_hz <= 0:
        parser.error('--playback-hz must be positive')
    if args.rounds < args.warmup_episodes + 2:
        parser.error('Need warmup plus at least two rounds to update and infer with a replacement policy')
    {'prepare': prepare, 'mock': mock_client, 'node': node}.get(args.role, model_role)(args)
