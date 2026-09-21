"""Robot-free EXPO inference/update and independent-process checkpoint restore.

Uses recorded observations/actions, not a robot environment. Statistics computed
here are fixtures for this validation recording, not approved task statistics.
Run train and restore as separate processes to avoid two resident model copies.
"""
import argparse
import dataclasses
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--params', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--utd-ratio', type=int, default=1)
    parser.add_argument('--updates', type=int, default=3)
    parser.add_argument('--num-devices', type=int, default=1)
    parser.add_argument('--fsdp-devices', type=int, default=1)
    parser.add_argument('--phase', choices=['train', 'restore'], default='train')
    parser.add_argument('--skip-checkpoint', action='store_true')
    parser.add_argument('--profile-updates', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING)
    import jax
    import numpy as np
    import openpi.training.sharding as sharding
    from openpi.shared import normalize
    from configs.model.expo_ft_pi_config import get_config
    from configs.task.pick import get_config as get_task
    from expo_ft.agents import initialize_checkpoint_dir
    from expo_ft.agents.alg.batch_utils import CRITIC_CAMERA_KEYS
    from expo_ft.agents.alg.expo_ft import load_agent, save_checkpoint, restore_checkpoint
    from expo_ft.agents.vla.pi05 import build_pi05
    from expo_ft.data.batch_processor import BatchProcessor
    from expo_ft.data.replay_buffer import create_replay_buffer, _critic_key_to_storage
    from expo_ft.env.droid_utils import process_droid_dataset

    started = time.monotonic()
    event_file = args.output / (args.phase + '-events.jsonl')
    def report(event, **fields):
        row = dict(event=event, seconds=time.monotonic()-started, **fields)
        row['device_memory'] = [{k:int(v) for k,v in (d.memory_stats() or {}).items()
                                 if isinstance(v, (int,np.integer))} for d in jax.devices()]
        text = json.dumps(row)
        print('[EXPO-GPU] ' + text, flush=True)
        with event_file.open('a') as stream:
            stream.write(text+'\n')

    assert jax.default_backend() == 'gpu', jax.devices()
    assert args.num_devices > 0 and args.fsdp_devices > 0
    assert len(jax.devices()) == args.num_devices, jax.devices()
    assert args.num_devices % args.fsdp_devices == 0
    assert args.batch_size % args.num_devices == 0
    cache_root = args.output if args.profile_updates else args.output.parent
    jax.config.update('jax_compilation_cache_dir', str(cache_root/'jax-cache'))
    report('runtime', devices=[str(d) for d in jax.devices()], jax=jax.__version__,
           batch_size=args.batch_size, utd_ratio=args.utd_ratio, updates=args.updates,
           preallocate=os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),
           memory_fraction=os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'),
           fsdp_devices=args.fsdp_devices, phase=args.phase)
    task = get_task()
    dataset = process_droid_dataset(str(args.dataset), task, num_data=1)
    assert len(dataset) >= 16
    config = get_config()
    config.pi05_weight_loader_path = str(args.params.resolve())
    config.pi05_assets_dir = str((args.output/'assets').resolve())
    config.pi05_asset_id = 'validation-recording'
    state = np.stack([np.concatenate([np.asarray(d['observations']['cartesian_position']).reshape(-1),
                          np.asarray(d['observations']['gripper_position']).reshape(-1)]) for d in dataset])
    actions = np.stack([d['actions'] for d in dataset])
    stats = {}
    for key, values in [('state',state),('actions',actions)]:
        assert np.isfinite(values).all()
        lo, hi = np.quantile(values,[0.01,0.99],axis=0)
        # Degenerate dimensions need a nonzero range for finite quantile scaling.
        constant = hi-lo < 1e-6
        lo, hi = np.where(constant,lo-0.5,lo), np.where(constant,hi+0.5,hi)
        stats[key] = normalize.NormStats(mean=values.mean(axis=0),
            std=np.where(values.std(axis=0)<1e-6,1.0,values.std(axis=0)),q01=lo,q99=hi)
    if args.phase == 'train':
        normalize.save(args.output/'assets'/config.pi05_asset_id,stats)
    report('data', transitions=len(dataset), N=config.N, n_edit_samples=config.n_edit_samples,
           num_qs=config.num_qs, initial_weights='pi05_base + initialized LoRA',
           normalization='recording-derived validation fixture')
    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    report('model_init_start')
    actor, actor_state, target, kwargs, metadata = build_pi05(config,42,mesh,data_sharding,
        replicated,args.phase=='restore',task.language_instruction)
    report('model_init_done')
    rb_args=dict(config=config,example_action=dataset[0]['actions'][None],capacity=len(dataset)+16,
                 task_description=task.language_instruction,replan_steps=8,seed=42,delay=0,
                 critic_camera_keys=CRITIC_CAMERA_KEYS)
    buffer=create_replay_buffer(**rb_args)
    offline=create_replay_buffer(**rb_args)
    processor=BatchProcessor(replay_buffer=buffer,offline_replay_buffer=offline,
        data_sharding=data_sharding,batch_size=args.batch_size,utd_ratio=args.utd_ratio,
        offline_ratio=0,actor_success_only=config.actor_success_only,use_dagger_hil_sampling=False,
        dataset=dataset)
    example={_critic_key_to_storage(k):buffer.dataset_dict[_critic_key_to_storage(k)][0][None]
             for k in CRITIC_CAMERA_KEYS}
    example.update(state=buffer.dataset_dict['state'][0][None],actions=buffer.dataset_dict['actions'][0][None])
    obs,state,action=buffer.convert_to_critic_format(example)
    actor.action_dim=action.squeeze().shape[-1]
    actor.state_dim=state.squeeze().shape[-1]
    kwargs['critic_camera_keys']=CRITIC_CAMERA_KEYS
    agent=load_agent(seed=42,example_observation=obs.squeeze(),example_action=action.squeeze(),
        example_state=state.squeeze(),actor=actor,actor_train_state=actor_state,target_actor_params=target,
        agent_kwargs=kwargs,metadata=metadata,mesh=mesh,data_sharding=data_sharding,
        replicated_sharding=replicated,resume=args.phase=='restore',replan_steps=8,
        default_prompt=task.language_instruction,edit_action_xyzg=task.edit_action_xyzg)
    del actor_state,target,kwargs
    gc.collect()
    report('agent_ready')

    def check_parameter_sharding(a):
        leaves = [v for v in jax.tree.leaves(a.actor_train_state.params)
                  if isinstance(v, jax.Array)]
        partitioned = [v for v in leaves if not v.is_fully_replicated]
        if args.fsdp_devices > 1:
            assert partitioned, 'FSDP requested but actor parameters are all replicated'
        report('parameter_sharding', arrays=len(leaves), partitioned_arrays=len(partitioned),
               partitioned_global_bytes=sum(v.nbytes for v in partitioned),
               example_specs=[str(v.sharding) for v in partitioned[:2]])

    if args.phase == 'train':
        check_parameter_sharding(agent)

    def fingerprint(tree):
        digest=hashlib.sha256()
        for leaf in jax.tree.leaves(tree):
            value=np.asarray(jax.device_get(leaf))
            assert np.isfinite(value).all(), 'nonfinite parameter'
            digest.update(value.tobytes())
        return digest.hexdigest()
    def hashes(a):
        return dict(actor=fingerprint(a.actor_train_state.params.filter(a.actor.train_config.trainable_filter)),
                    critic=fingerprint(a.critic.params),edit_actor=fingerprint(a.edit_actor.params),
                    encoder=fingerprint(a.batch_encoder.params))
    def infer(a,label):
        began=time.monotonic()
        action,new_agent,_=a.sample_actions(dataset[0]['observations'])
        action=np.asarray(jax.device_get(action))
        assert action.shape==(16,7) and np.isfinite(action).all()
        report('inference',label=label,wall_seconds=time.monotonic()-began,shape=list(action.shape))
        return action,new_agent

    checkpoint_path=args.output/'checkpoint'
    if args.phase=='restore':
        manager,resuming=initialize_checkpoint_dir(checkpoint_path,keep_period=None,overwrite=False,resume=True)
        assert resuming
        agent=restore_checkpoint(manager,agent)
        agent=agent.cache_infer_params()
        check_parameter_sharding(agent)
        expected=json.loads((args.output/'expected.json').read_text())
        assert hashes(agent)==expected['hashes']
        assert int(agent.actor_train_state.step)==expected['actor_step']
        action,_=infer(agent,'restored')
        np.testing.assert_allclose(action,np.load(args.output/'expected-action.npy'),rtol=1e-4,atol=1e-5)
        manager.close()
        report('passed',checkpoint_restored=True,action_matches=True)
        return

    _,agent=infer(agent,'first_compile')
    for i in range(3):
        _,agent=infer(agent,f'warm_{i}')
    rng=jax.random.PRNGKey(2026)
    previous_trees = {}
    previous_leaves = {}
    if args.profile_updates:
        jax.config.update('jax_log_compiles', True)
        jax.config.update('jax_explain_cache_misses', True)
    for i in range(args.updates):
        report('batch_start',update=i+1)
        batch,actor_batch,rng=processor.next_batch(rng)
        jax.block_until_ready((batch,actor_batch))
        before=hashes(agent)
        report('update_start',update=i+1,critic_batch=list(batch['actions'].shape),
               actor_batch=list(actor_batch['actions'].shape))
        if args.profile_updates:
            clean = agent.replace(_infer_cache=None)
            fields = {f.name: getattr(clean, f.name) for f in dataclasses.fields(clean)
                      if f.name != '_infer_cache'}
            trees = {name: str(jax.tree.structure(value)) for name, value in fields.items()}
            flat, structure = jax.tree_util.tree_flatten_with_path((clean, batch, actor_batch))
            leaves = {jax.tree_util.keystr(path): dict(aval=str(jax.core.get_aval(value)),
                       sharding=str(getattr(value, 'sharding', None))) for path, value in flat}
            changed_trees = [k for k in trees if trees[k] != previous_trees.get(k)]
            changed_leaves = {k: dict(before=previous_leaves.get(k), after=v)
                              for k,v in leaves.items() if v != previous_leaves.get(k)}
            (args.output/f'jit-input-{i+1}.json').write_text(json.dumps(
                dict(trees=trees, leaves=leaves), indent=2))
            report('jit_signature', update=i+1, changed_tree_fields=changed_trees,
                   changed_leaf_count=len(changed_leaves),
                   changed_leaves=changed_leaves if i else {},
                   update_cache_size=type(agent)._update_jit._cache_size())
            previous_trees, previous_leaves = trees, leaves
            del clean, fields, flat, structure
        began=time.monotonic()
        agent,info=agent.update(agent,batch,args.utd_ratio,actor_batch)
        jax.block_until_ready((agent,info))
        wall=time.monotonic()-began
        if args.profile_updates:
            report('jit_call_done', update=i+1,
                   update_cache_size=type(agent)._update_jit._cache_size(), wall_seconds=wall)
        metrics={key:float(value) for key,value in jax.device_get(info).items()}
        assert all(np.isfinite(v) for v in metrics.values()),metrics
        after=hashes(agent)
        assert all(before[key]!=after[key] for key in before),'Expected trainable group did not change'
        assert int(agent.actor_train_state.step)==i+1
        report('update_passed',update=i+1,wall_seconds=wall,changed=list(after),metrics=metrics)
        del batch,actor_batch,info
        gc.collect()
    final_action,agent=infer(agent,'after_updates')
    if args.profile_updates:
        trainable = dict(
            actor=agent.actor_train_state.params.filter(agent.actor.train_config.trainable_filter),
            critic=agent.critic.params, edit_actor=agent.edit_actor.params,
            encoder=agent.batch_encoder.params, temperature=agent.temp.params)
        snapshot = {jax.tree_util.keystr(path): np.asarray(jax.device_get(value))
                    for path, value in jax.tree_util.tree_flatten_with_path(trainable)[0]}
        np.savez(args.output/'final-trainable.npz', **snapshot)
        np.save(args.output/'final-action.npy', final_action)
        del trainable, snapshot
    if not args.skip_checkpoint:
        manager,_=initialize_checkpoint_dir(checkpoint_path,keep_period=None,overwrite=False,resume=False)
        report('save_start')
        save_checkpoint(manager,agent,args.updates)
        manager.wait_until_finished()
        manager.close()
        (args.output/'expected.json').write_text(json.dumps(dict(hashes=hashes(agent),actor_step=int(agent.actor_train_state.step))))
        expected_action,_=infer(agent,'restore_reference')
        np.save(args.output/'expected-action.npy',expected_action)
        report('save_done')
    report('passed',updates=args.updates,checkpoint_saved=not args.skip_checkpoint)


if __name__=='__main__':
    main()
