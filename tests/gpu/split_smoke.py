"""Robot-free split integration on a GPU test allocation, with recorded RGB.

Launch via --role driver. Two model processes and two transport processes run on
the allocated node; the only environment is RecordedEnv below (no WS connection).
Checks warmup, two updates, snapshot replacement, seeded action parity and save.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def driver(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    processes, logs = [], []
    # Only Unix sockets/session markers live here; payloads stay in anonymous RAM.
    with tempfile.TemporaryDirectory(prefix='expo-split-') as local:
        local = Path(local)
        token = local / 'token'
        token.write_text(secrets.token_hex(32))
        token.chmod(0o600)
        ports = []
        for _ in range(2):
            with socket.socket() as s:
                s.bind(('127.0.0.1', 0))
                ports.append(s.getsockname()[1])
        def launch(name, command, env=None):
            log = (output / (name + '.log')).open('w')
            logs.append(log)
            p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            processes.append(p)
            return p
        try:
            for i in range(2):
                config = local / f'transport-{i}.json'
                config.write_text(json.dumps(dict(mailbox=str(local / str(i)), token_file=str(token),
                    listen=['127.0.0.1', ports[i]], peers=[['127.0.0.1', ports[1-i]]],
                    parallel_connections=32, record_connections=4, allow_plain_loopback=True)))
                launch(f'transport-{i}', [sys.executable, '-u', '-m', 'expo_ft.distributed.transport', '--config', str(config)])
            env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.65')
            base = [sys.executable, '-u', __file__, '--output', str(output), '--dataset', str(args.dataset), '--params', str(args.params)]
            learner = launch('learner', base + ['--role', 'learner', '--mailbox', str(local / '0')], env)
            deadline = time.monotonic() + 2700
            ready = output / 'learner-ready'
            while not ready.exists():
                if learner.poll() is not None:
                    raise RuntimeError('learner initialization failed; see learner.log')
                if time.monotonic() > deadline:
                    raise TimeoutError('learner initialization')
                time.sleep(1)
            inference = launch('inference', base + ['--role', 'inference', '--mailbox', str(local / '1')], env)
            while learner.poll() is None or inference.poll() is None:
                if any(p.poll() not in (None, 0) for p in processes):
                    raise RuntimeError('split process failed; inspect logs')
                if time.monotonic() > deadline:
                    raise TimeoutError('split integration')
                time.sleep(1)
            assert learner.returncode == inference.returncode == 0
            state = json.loads((output / 'checkpoints/split-384.json').read_text())
            assert state['episode_count'] == 24
            assert len(list((output / 'received-policies').glob('*.json'))) == 2
            assert (output / 'checkpoint-restored.json').exists()
            (output / 'PASSED.json').write_text(json.dumps({'episodes': 24, 'transitions': 384,
                'updates': 2, 'versions': [0, 352], 'seeded_action_parity': True, 'checkpoint_restore': True}))
            print('SPLIT INTEGRATION PASSED', flush=True)
        finally:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            for p in processes:
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            for log in logs:
                log.close()


def model_role(args):
    import logging
    logging.basicConfig(level=logging.INFO)
    import jax
    import numpy as np
    import wandb
    from configs.model.expo_ft_pi_config import get_config
    from configs.task.pick import get_config as get_task
    from expo_ft.distributed import runner
    from expo_ft.env.droid_utils import process_droid_dataset
    from openpi.shared import normalize

    assert jax.default_backend() == 'gpu'
    task, config = get_task(), get_config()
    task.control_hz = 100000  # no physical clock/hardware in this test
    dataset = process_droid_dataset(str(args.dataset), task, num_data=1)
    config.pi05_weight_loader_path = str(args.params)
    config.pi05_assets_dir, config.pi05_asset_id = str(args.output / 'assets'), 'split-fixture'
    config.N, config.n_edit_samples = 2, 2
    config.actor_success_only = False
    flags = SimpleNamespace(config=config, config_task=task, seed=42, replan_steps=8, num_robot=2,
        split_session='gpu-smoke', split_mailbox=args.mailbox, split_timeout=1200, resume=False,
        client_host='127.0.0.1', client_port=8100, output_dir=str(args.output), run_name='robot-free',
        max_steps=384, batch_size=2, utd_ratio=1, num_updates=1, step_interval=50,
        checkpoint_model=True, checkpoint_buffer=True, checkpoint_interval=384)
    reference = deepcopy(dataset[0]['observations'])

    if args.role == 'inference':
        class RecordedEnv:
            def __init__(self, **kwargs):
                self.robot = kwargs['port'] - flags.client_port
                self.step_number = 0
            def reset(self):
                self.step_number = 0
                return self.get_observation()
            def get_observation(self):
                from expo_ft.env.sft_eval import canonical_observation
                return canonical_observation(deepcopy(reference), self.robot == 1)
            def step(self, action):
                self.step_number += 1
                return np.clip(action, -1, 1), 'policy'
            def get_info_for_step(self):
                done = self.step_number == 16
                return done, done, float(done), float(not done)
            def close(self):
                pass
        original = runner.import_policy
        def checked_import(agent, path, contract, version):
            before_rng = np.asarray(agent.rng).copy()
            agent = original(agent, path, contract, version)
            assert agent.critic is None and agent.target_actor_params is None and agent.temp is None
            assert not jax.tree.leaves(agent.actor_train_state.opt_state)
            assert not jax.tree.leaves(agent.batch_encoder.opt_state)
            np.testing.assert_array_equal(agent.rng, before_rng)
            action, _, _ = agent.replace(rng=jax.random.PRNGKey(123)).sample_actions(deepcopy(reference))
            np.testing.assert_allclose(np.asarray(action), np.load(args.output / f'expected-{version}.npy'), rtol=1e-4, atol=1e-5)
            dest = args.output / 'received-policies'
            dest.mkdir(exist_ok=True)
            (dest / f'{version}.json').write_text(json.dumps({'parity': True, 'optimizer_arrays': 0}))
            return agent
        runner.import_policy = checked_import
        runner.run_inference(flags, env_factory=RecordedEnv)
        return

    from expo_ft.agents import initialize_checkpoint_dir
    from expo_ft.agents.alg.batch_utils import CRITIC_CAMERA_KEYS
    from expo_ft.agents.alg.expo_ft import load_agent, save_checkpoint, restore_checkpoint
    from expo_ft.agents.vla.pi05 import build_pi05
    from expo_ft.data.batch_processor import BatchProcessor
    from expo_ft.data.replay_buffer import create_replay_buffer, _critic_key_to_storage
    from openpi.training import sharding
    values = {'state': np.stack([np.r_[d['observations']['cartesian_position'], d['observations']['gripper_position']] for d in dataset]),
              'actions': np.stack([d['actions'] for d in dataset])}
    stats = {}
    for k, v in values.items():
        lo, hi = np.quantile(v, [.01, .99], axis=0)
        constant = hi - lo < 1e-6
        stats[k] = normalize.NormStats(mean=v.mean(0), std=np.maximum(v.std(0), 1e-6),
                                      q01=np.where(constant, lo-.5, lo), q99=np.where(constant, hi+.5, hi))
    normalize.save(args.output / 'assets' / 'split-fixture', stats)
    mesh = sharding.make_mesh(1)
    data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    actor, actor_state, target, kwargs, metadata = build_pi05(config, 42, mesh, data, replicated, False, task.language_instruction)
    rb = dict(config=config, example_action=dataset[0]['actions'][None], capacity=2048,
              task_description=task.language_instruction, replan_steps=8, delay=0, critic_camera_keys=CRITIC_CAMERA_KEYS)
    buffers = [create_replay_buffer(**rb, seed=42+i) for i in range(2)]
    offline = create_replay_buffer(**rb, seed=42)
    processor = BatchProcessor(buffers[0], offline, data, 2, 1, 0, False, False, dataset, replay_buffers=buffers)
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
    original = runner.export_policy
    def checked_export(agent, contract, version):
        start = time.monotonic()
        buffer = original(agent, contract, version)
        action, _, _ = agent.replace(rng=jax.random.PRNGKey(123)).cache_infer_params().sample_actions(deepcopy(reference))
        np.save(args.output / f'expected-{version}.npy', np.asarray(action))
        print('EXPORT', version, buffer.size, time.monotonic()-start, flush=True)
        (args.output / 'learner-ready').touch()
        return buffer
    runner.export_policy = checked_export
    agent = runner.run_learner(flags, agent, buffers, processor, manager, args.output / 'checkpoints',
                              save_checkpoint, 0, False, replicated, 1)
    restored = restore_checkpoint(manager, agent)
    assert int(restored.actor_train_state.step) == 2
    (args.output / 'checkpoint-restored.json').write_text(json.dumps({'actor_step': 2}))
    manager.close()
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['driver', 'learner', 'inference'], default='driver')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--params', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mailbox')
    args = parser.parse_args()
    driver(args) if args.role == 'driver' else model_role(args)
