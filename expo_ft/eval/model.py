"""Build each eval model afresh from CPU buffers. No checkpoint disk reads."""
import dataclasses
import json

import numpy as np

from .checkpoint import arrays, config_and_norm, split_actor, validate_base, put, leaves


def ram_config(config, stats):
    # Reuse the existing factory/transforms, overriding only its asset reader.
    class RAMData(type(config.data)):
        def _load_norm_stats(self, assets_dir, asset_id):
            return stats
    factory = RAMData(**{f.name: getattr(config.data, f.name) for f in dataclasses.fields(config.data)})
    return dataclasses.replace(config, data=factory)


def merge_actor(base, trained, config):
    frozen, _ = split_actor(base, config)
    import flax.nnx as nnx
    import jax
    model_shape = nnx.eval_shape(config.model.create, jax.random.key(0))
    expected = dict(leaves(nnx.state(model_shape).to_pure_dict()))
    merged = dict(leaves(frozen))
    update = dict(leaves(trained))
    if merged.keys() & update.keys() or merged.keys() | update.keys() != expected.keys():
        raise ValueError('Missing, overlapping or extra actor parameters')
    merged.update(update)
    result = {}
    for path, value in merged.items():
        if value.shape != expected[path].shape:
            raise ValueError(f'Actor shape mismatch: {path}')
        put(result, path, value)
    return result


def build(base, payload, task, seed=42, *, base_verified=False):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.policies.policy import Policy
    from openpi import transforms
    from expo_ft.env.checkpoint_policy import SFTPolicy, OnlinePolicy
    if not base_verified: validate_base(base, payload)
    with arrays(base) as (base_trees, _), arrays(payload) as (trees, meta):
        original, stats = config_and_norm(meta)
        cfg = ram_config(original, stats)
        if not getattr(cfg.data, 'use_cartesian_state', False) or cfg.data.output_action_dim != 7:
            raise ValueError('Expected Cartesian state / 7D actions')
        full = merge_actor(base_trees['base'], trees['actor'], cfg)
        if meta['kind'] == 'sft':
            if set(trees) != {'actor'}: raise ValueError('Unexpected SFT weight groups')
            # Match the existing SFT loader's BF16 inference casting.
            params = jax.tree.map(lambda a: jnp.asarray(a, dtype=jnp.bfloat16), full)
            model = cfg.model.load(params, remove_extra_params=False)
            data = cfg.data.create(cfg.assets_dirs, cfg.model)
            result = SFTPolicy.__new__(SFTPolicy)
            result.config = cfg
            result.policy = Policy(model, rng=jax.random.key(seed), transforms=[
                transforms.InjectDefaultPrompt(task.language_instruction), *data.data_transforms.inputs,
                transforms.Normalize(stats, use_quantiles=data.use_quantile_norm), *data.model_transforms.inputs],
                output_transforms=[*data.model_transforms.outputs,
                    transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm), *data.data_transforms.outputs],
                metadata=cfg.policy_metadata)
            jax.block_until_ready(params)
            return result
        from expo_ft.utils.model_config import check_task, defaults_signature, unpack
        record = json.loads(meta['files']['model_config/config.json'])
        if record['factory_defaults'] != defaults_signature(): raise ValueError('EXPO defaults changed')
        if record['sft']['resolved'] != json.loads(meta['files']['assets/config.json'])['resolved']:
            raise ValueError('SFT recipe mismatch')
        check_task(record, task, meta['replan_steps'])
        config = unpack(record['config'])
        from openpi.training import sharding, utils
        import optax
        from expo_ft.agents.vla.pi05 import Pi05Agent
        from expo_ft.agents.alg.expo_ft import EXPOLearner
        from expo_ft.data.replay_buffer import PiReplayBuffer, _critic_key_to_storage
        import flax.serialization
        if jax.device_count() != 1: raise ValueError('Expose exactly one eval GPU')
        mesh = sharding.make_mesh(1)
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        params = jax.tree.map(lambda a: jax.device_put(a, replicated), full)
        model = cfg.model.load(params, remove_extra_params=False)
        graph, state = nnx.split(model)
        train_state = utils.TrainState(step=0, params=state, model_def=graph,
            tx=optax.set_to_zero(), opt_state=(), ema_decay=None, ema_params=None)
        state_sharding = sharding.fsdp_sharding(jax.eval_shape(lambda: train_state), mesh, log=False)
        freeze = config.get('freeze_pi05_encoder', False)
        actor = Pi05Agent(train_config=cfg, mesh=mesh, train_state_sharding=state_sharding,
            data_sharding=data_sharding, replicated_sharding=replicated,
            default_prompt=task.language_instruction, freeze_pi05_encoder=freeze)
        cameras = tuple(record['task']['critic_camera_keys'])
        replay = PiReplayBuffer(example_action=task.example_action.squeeze(), capacity=1,
            pi_train_config=cfg, skip_norm_stats=False, resize_size=cfg.data.model_image_resize,
            task_description=task.language_instruction, replan_steps=meta['replan_steps'],
            discount=config['discount'], delay=0, critic_camera_keys=cameras)
        sample = { _critic_key_to_storage(k): np.zeros_like(replay.dataset_dict[_critic_key_to_storage(k)][:1]) for k in cameras }
        sample.update(state=np.zeros_like(replay.dataset_dict['state'][:1]), actions=np.zeros_like(replay.dataset_dict['actions'][:1]))
        obs, state_example, action = replay.convert_to_critic_format(sample)
        actor.action_dim, actor.state_dim = action.squeeze().shape[-1], state_example.squeeze().shape[-1]
        kwargs = {k:v for k,v in config.items() if k not in ('model_cls','initial_sft_checkpoint','initial_sft_base','freeze_pi05_encoder') and not k.startswith('pi05_')}
        kwargs.update(actor=actor, actor_train_state=train_state, target_actor_params=None,
            action_horizon=cfg.model.action_horizon, mesh=mesh, freeze_encoder=freeze,
            data_sharding=data_sharding, replicated_sharding=replicated, default_prompt=task.language_instruction,
            resize_size=cfg.data.model_image_resize, critic_camera_keys=cameras, inference_only=True,
            rollout_cache=False, replan_steps=meta['replan_steps'], edit_action_xyzg=task.edit_action_xyzg, resume=False)
        agent = EXPOLearner.create(seed, obs.squeeze(), action.squeeze(), state_example.squeeze(), **kwargs)
        if set(trees) != {'actor','encoder','edit','target_critic'}: raise ValueError('Invalid EXPO weight groups')
        def installed(template, group):
            expected = dict(leaves(flax.serialization.to_state_dict(template)))
            received = dict(leaves(trees[group]))
            if expected.keys() != received.keys() or any(expected[k].shape != received[k].shape or
                    np.dtype(expected[k].dtype) != np.dtype(received[k].dtype) for k in expected):
                raise ValueError(f'Invalid {group} parameters')
            owned = jax.tree.map(lambda a: jax.device_put(a, replicated, may_alias=False), trees[group])
            jax.block_until_ready(owned)
            return flax.serialization.from_state_dict(template, owned)
        agent = agent.replace(batch_encoder=agent.batch_encoder.replace(params=installed(agent.batch_encoder.params,'encoder')),
            edit_actor=agent.edit_actor.replace(params=installed(agent.edit_actor.params,'edit')),
            target_critic=agent.target_critic.replace(params=installed(agent.target_critic.params,'target_critic')))
        result = OnlinePolicy.__new__(OnlinePolicy)
        result.agent = agent.replace(rng=jax.device_put(jax.random.PRNGKey(seed),replicated))
        result.config, result.record, result.prompt = cfg, record, task.language_instruction
        jax.block_until_ready(params)
        return result
