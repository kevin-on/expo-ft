"""Full/compact SFT online-init parity and one update; allocated GPU, recorded data only.

Run export with JAX_PLATFORMS=cpu, then full/compact/restore in separate GPU
processes. Output should be node-local scratch (includes a full online checkpoint).
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['export', 'full', 'compact', 'restore'])
    p.add_argument('--sft', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--demo', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reports', type=Path, required=True)
    p.add_argument('--inference-only', action='store_true')
    p.add_argument('--action-atol', type=float, default=1e-5,
                   help='Cross-process action tolerance; establish any relaxation with a full-checkpoint repeat control')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()

    def report(event, **kw):
        row = dict(event=event, phase=args.phase, elapsed=time.monotonic()-began, **kw)
        print(json.dumps(row), flush=True)
        with (args.reports/(args.phase+'.jsonl')).open('a') as f:
            f.write(json.dumps(row)+'\n')

    from expo_ft.eval import checkpoint as ck
    packet = args.output/'sft/fixture/checkpoints/4999/trainable_weights.bin'
    if args.phase == 'export':
        with ck.export(args.sft, 'sft', checkpoint_path='sft/fixture/checkpoints/4999',
                       training_run_id='fixture') as payload:
            assert ck.save(payload, args.output) == packet
            report('EXPORT_OK', bytes=payload.size)
        return

    import jax
    import flax.serialization
    from configs.task.pick import get_config as get_task
    from configs.model.expo_ft_pi_config import get_config
    from expo_ft.env.droid_utils import process_droid_dataset
    from expo_ft.env.checkpoint_policy import build_agent, OnlinePolicy
    from openpi.training import sharding
    assert jax.default_backend() == 'gpu' and jax.device_count() == 1
    task = get_task()
    dataset = process_droid_dataset(args.demo, task, num_data=1)
    obs = {**dataset[0]['observations'], 'prompt': task.language_instruction}
    report('DATA_READY', transitions=len(dataset))

    if args.phase == 'restore':
        policy = OnlinePolicy(args.output/'online/1', task=task,
                              initial_sft_checkpoint=packet, initial_sft_base=args.base)
        assert int(policy.agent.actor_train_state.step) == 1
        actual = np.asarray(policy.sample_actions(obs)[0])
        expected = np.load(args.output/'compact-after.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=args.action_atol)
        report('COMPACT_INITIALIZED_ONLINE_RESUME_OK', max_abs=float(np.max(np.abs(actual-expected))), atol=args.action_atol)
        return

    config = get_config()
    config.initial_sft_checkpoint = str(args.sft if args.phase == 'full' else packet)
    if args.phase == 'compact':
        config.initial_sft_base = str(args.base)
    config.N = 2
    config.n_edit_samples = 2
    agent = build_agent(config, task, 8)
    report('AGENT_READY')
    hashes = {'actor': ck.fingerprint(agent.actor_train_state.params.to_pure_dict())}
    for group in ['batch_encoder', 'edit_actor', 'critic', 'target_critic']:
        hashes[group] = ck.fingerprint(flax.serialization.to_state_dict(getattr(agent, group).params))
    inputs = agent.actor.process_raw_inputs(dict(obs), agent.action_dim, agent.resize_size)
    hashes['inputs'] = ck.fingerprint({k: v for k, v in inputs.items() if v is not None})
    hashes['rng'] = ck.fingerprint({'rng': agent.rng})
    if args.phase == 'full':
        (args.output/'reference-hashes.json').write_text(json.dumps(hashes))
    else:
        assert hashes == json.loads((args.output/'reference-hashes.json').read_text())
        report('INITIAL_PARAMETERS_EXACTLY_EQUAL', hashes=hashes)
    actions, agent, _ = agent.cache_infer_params().sample_actions(obs)
    actions = np.asarray(actions)
    if args.phase == 'full':
        np.save(args.output/'reference-before.npy', actions)
    else:
        expected = np.load(args.output/'reference-before.npy')
        np.testing.assert_allclose(actions, expected, rtol=1e-4, atol=args.action_atol)
        report('ONLINE_INITIAL_INFERENCE_PARITY_OK', max_abs=float(np.max(np.abs(actions-expected))), atol=args.action_atol)
    if args.inference_only:
        report('INFERENCE_ONLY_DONE')
        return

    from expo_ft.data.replay_buffer import create_replay_buffer
    from expo_ft.data.batch_processor import BatchProcessor
    from expo_ft.agents.alg.batch_utils import CRITIC_CAMERA_KEYS
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    def buf():
        return create_replay_buffer(config, task.example_action, len(dataset)+16,
            task.language_instruction, 8, 42, critic_camera_keys=CRITIC_CAMERA_KEYS)
    processor = BatchProcessor(replay_buffer=buf(), offline_replay_buffer=buf(),
        data_sharding=data_sharding, batch_size=2, utd_ratio=1, offline_ratio=0,
        actor_success_only=config.actor_success_only, use_dagger_hil_sampling=False, dataset=dataset)
    batch, actor_batch, _ = processor.next_batch(jax.random.PRNGKey(2))
    report('UPDATE_BEGIN')
    agent, info = agent.update(agent, batch, 1, actor_batch)
    jax.block_until_ready((agent, info))
    assert int(agent.actor_train_state.step) == 1
    metrics = {k: np.asarray(v).tolist() for k, v in info.items()}
    assert all(np.isfinite(np.asarray(v)).all() for v in metrics.values())
    report('UPDATE_OK', metrics=metrics)
    if args.phase == 'compact':
        from expo_ft.agents import initialize_checkpoint_dir
        from expo_ft.agents.alg.expo_ft import save_checkpoint
        manager, _ = initialize_checkpoint_dir(args.output/'online', keep_period=None, overwrite=False, resume=False)
        save_checkpoint(manager, agent, 1)
        manager.wait_until_finished(); manager.close()
        report('ONLINE_CHECKPOINT_OK')
    agent = agent.cache_infer_params()
    after = np.asarray(agent.sample_actions(obs)[0])
    np.save(args.output/(args.phase+'-after.npy'), after)
    if args.phase == 'compact':
        expected = np.load(args.output/'full-after.npy')
        np.testing.assert_allclose(after, expected, rtol=1e-4, atol=args.action_atol)
        report('POST_UPDATE_PARITY_OK', max_abs=float(np.max(np.abs(after-expected))), atol=args.action_atol)
    report('DONE')


if __name__ == '__main__':
    main()
