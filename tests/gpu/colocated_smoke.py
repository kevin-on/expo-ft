"""Run the real colocated CLI/model/update/checkpoint against recorded episodes.

Only dataset loading and robot RPC endpoints are replaced. No robot, camera,
relay or network listener is opened. Use a fresh output directory on an allocated
GPU node; fixture is the read-only output of split_wan_smoke.py prepare.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import pickle
import runpy
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def verify_records(checkpoint_dir, selected, start_step=0):
    """Replay persistence converts scalar strings to 0-D NumPy arrays."""
    import numpy as np
    from split_wan_smoke import record_hash
    def normalized_hash(rows):
        return record_hash([dict(row, observations=dict(row['observations'],
            prompt=str(np.asarray(row['observations'].get('prompt', '')).item()))) for row in rows])
    step = start_step
    for round_id in range(len(selected[0])):
        for robot, episodes in enumerate(selected):
            rows = episodes[round_id]
            path = checkpoint_dir / f'robot-{robot}' / 'buffers' / f'{step+1:012d}-{step+len(rows):012d}.pkl'
            saved = pickle.loads(path.read_bytes())['transitions']
            assert normalized_hash(saved) == normalized_hash(rows), str(path)
            step += len(rows)
    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-robot', type=int, choices=(1, 2), required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--utd-ratio', type=int, default=1)
    parser.add_argument('--num-updates', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--expected-actor-step', type=int, help='Resume check when the prior process has no report')
    parser.add_argument('--rounds-only', action='store_true',
                        help='Skip snapshot parity and model checkpoint save; retain replay checks')
    args = parser.parse_args()

    import jax
    import numpy as np
    import wandb
    from expo_ft.distributed import local, policy, runner
    from expo_ft.env import droid_utils, env_client
    from expo_ft.env.sft_eval import canonical_observation, physical_action

    assert jax.process_count() == 1 and all(d.platform == 'gpu' for d in jax.devices())
    assert len(jax.devices()) == 2, jax.devices()
    started = time.monotonic()
    manifest = json.loads((args.fixture / 'manifest.json').read_text())
    episodes = [[pickle.loads((args.fixture / item['file']).read_bytes())
                 for item in robot] for robot in manifest['robots'][:args.num_robot]]
    checkpoint_dir = args.output / 'run' / 'checkpoints'
    if not args.resume and checkpoint_dir.exists():
        raise FileExistsError(f'Fresh validation requires an unused output: {args.output}')
    ledgers = list(checkpoint_dir.glob('split-*.json'))
    start_step = max(int(p.stem.split('-')[1]) for p in ledgers) if args.resume else 0
    previous = json.loads((checkpoint_dir / f'split-{start_step}.json').read_text()) if args.resume else None
    first_round = previous['episode_count'] // args.num_robot if previous else 0
    selected = [[robot[(first_round + i) % len(robot)] for i in range(args.rounds)] for robot in episodes]
    final_step = start_step + sum(len(rows) for robot in selected for rows in robot)
    demo = deepcopy(episodes[0][0])
    # Replay seeding requires per-transition success marks, as dataset conversion supplies.
    for row in demo:
        row['is_success'] = bool(demo[-1]['rewards'] > 0)
    if not demo[-1]['is_success']:
        raise ValueError('Select a fixture whose first robot0 episode succeeded')
    droid_utils.process_droid_dataset = lambda *a, **kw: deepcopy(demo)

    def forbidden(*args, **kwargs):
        raise AssertionError('Colocated runtime attempted network / second model / snapshot transport')
    runner._channel = runner.build_inference = forbidden
    runner.export_policy = runner.import_policy = forbidden
    env_client.EnvClientWrapper = forbidden
    log = []
    original_log = wandb.log
    def logged(data, **kwargs):
        log.append({k: np.asarray(v).tolist() for k, v in data.items()})
        return original_log(data, **kwargs)
    envs = []

    class Env:
        def __init__(self, *, port, **kwargs):
            self.robot = port - 8102
            self.round = -1
            self.commands = 0
            envs.append(self)
        def reset_only(self):
            self.round += 1
            self.index = 0
            self.rows = selected[self.robot][self.round]
        def start_episode(self):
            return self.get_observation()
        def get_observation(self):
            return canonical_observation(self.rows[min(self.index, len(self.rows)-1)]['observations'], self.robot == 1)
        def step(self, command):
            command = np.asarray(command)
            assert command.shape == (7,) and np.isfinite(command).all()
            row = self.rows[self.index]
            self.index += 1
            self.commands += 1
            return physical_action(row['actions'], self.robot == 1), 'human' if row.get('is_hil', False) else 'policy'
        def get_info_for_step(self):
            row = self.rows[self.index-1]
            return bool(row['dones']), bool(self.rows[-1]['rewards'] > 0), float(row['rewards']), float(row['masks'])
        def close(self):
            pass

    published, installed, parity, counts = [], [], [], []
    original_publish, original_install = local.LocalPolicyExchange.publish, local.LocalPolicyExchange.install
    def publish(self, agent, contract, version):
        view = local.inference_view(agent, jax.random.PRNGKey(91))
        leaves = 0
        for field in ('actor_train_state', 'batch_encoder', 'edit_actor', 'target_critic'):
            source, target = getattr(agent, field).params, getattr(view, field).params
            for x, y in zip(jax.tree.leaves(source), jax.tree.leaves(target)):
                if isinstance(x, jax.Array):
                    replica = next(s.data for s in x.addressable_shards if s.device == agent.actor.infer_device)
                    assert replica.unsafe_buffer_pointer() == y.unsafe_buffer_pointer()
                    leaves += 1
        counts.append(leaves)
        # Validation-only snapshot: seeded actions must match the split import path.
        # Production runner exports/imports are replaced with forbidden() above.
        if not args.rounds_only and len(parity) < 2:
            with policy.export_policy(agent, contract, version) as snapshot:
                imported = policy.import_policy(view, snapshot, contract, version)
            obs = selected[0][0][0]['observations']
            left, _, _ = view.sample_actions(obs)
            right, _, _ = imported.sample_actions(obs)
            np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=1e-5, atol=1e-6)
            parity.append(version)
            del imported, left, right
        published.append(version)
        print(f'LOCAL_POLICY_SHARED version={version} leaves={leaves} parity={parity}', flush=True)
        return original_publish(self, agent, contract, version)
    def install(self, previous_agent, contract, version, **kwargs):
        result, timing = original_install(self, previous_agent, contract, version, **kwargs)
        if previous_agent is not None:
            np.testing.assert_array_equal(result.rng, previous_agent.rng)
        installed.append(version)
        return result, timing
    local.LocalPolicyExchange.publish, local.LocalPolicyExchange.install = publish, install
    original_run = local.run_colocated
    final_agent = []
    initial_actor_step = []
    def run(*a, **kw):
        nonlocal original_log
        # wandb.init replaces wandb.log; intercept after the CLI initializes it.
        original_log = wandb.log
        wandb.log = logged
        initial_actor_step.append(int(np.asarray(a[1].actor_train_state.step)))
        if args.resume:
            report_path = args.output / 'passed.json'
            expected = json.loads(report_path.read_text())['actor_step'] if report_path.exists() else args.expected_actor_step
            assert expected is not None and initial_actor_step[0] == expected
            assert a[7] == start_step and a[8]
        result = original_run(*a, **kw, env_factory=Env)
        final_agent.append(result)
        return result
    local.run_colocated = run

    sys.argv = ['train_pi_robo.py', '--split_role=colocated', '--fsdp_devices=1',
                f'--num_robot={args.num_robot}', f'--initial_sft_checkpoint={args.checkpoint}',
                f'--output_dir={args.output}', '--run_name=run', '--dataset_path=/recorded-fixture',
                f'--batch_size={args.batch_size}', f'--utd_ratio={args.utd_ratio}',
                f'--num_updates={args.num_updates}', '--split_warmup_episodes=1',
                f'--max_steps={final_step}', '--checkpoint_buffer',
                '--checkpoint_interval=0', '--notqdm', '--config_task.control_hz=10',
                '--split_timeout=1800'] + (['--resume'] if args.resume else ['--overwrite'])
    if not args.rounds_only:
        sys.argv.append('--checkpoint_model')
    try:
        runpy.run_path(str(Path(__file__).resolve().parents[2] / 'train_pi_robo.py'), run_name='__main__')
    except SystemExit as error:
        if error.code not in (None, 0):
            raise
    assert len(final_agent) == 1 and published == installed
    assert len(published) >= (1 if args.resume else 2)
    assert all(env.round + 1 == args.rounds for env in envs) and len(envs) == args.num_robot
    if not args.rounds_only:
        ledger = json.loads((checkpoint_dir / f'split-{final_step}.json').read_text())
        assert ledger['episode_count'] == (first_round + args.rounds) * args.num_robot
    rounds = [x for x in log if 'round_steps' in x]
    assert len(rounds) == args.rounds
    cursor = start_step
    for metrics in rounds:
        cursor += metrics['round_steps']
        # Warmup=1 counts this completed round; the transition-count gate remains.
        assert metrics['updates'] == (args.num_updates if cursor >= args.batch_size else 0)
    assert int(np.asarray(final_agent[0].actor_train_state.step)) > initial_actor_step[0]
    assert verify_records(checkpoint_dir, selected, start_step) == final_step
    report = dict(num_robot=args.num_robot, devices=[str(d) for d in jax.devices()],
                  rounds=args.rounds, resumed=args.resume, start_step=start_step, final_step=final_step,
                  published=published, installed=installed, shared_parameter_leaves=counts,
                  action_parity_versions=parity, metrics=log,
                  initial_actor_step=initial_actor_step[0],
                  actor_step=int(np.asarray(final_agent[0].actor_train_state.step)),
                  seconds=time.monotonic()-started, replay_matches_recordings=True)
    report_path = args.output / ('resume-passed.json' if args.resume else 'passed.json')
    report_path.write_text(json.dumps(report, indent=2))
    print(f'COLOCATED_SMOKE_PASSED {report_path} elapsed={report["seconds"]:.2f}s', flush=True)


if __name__ == '__main__':
    main()
