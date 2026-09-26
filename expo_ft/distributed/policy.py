"""Inference snapshot format, independent of the network transport.

Only the actual rollout parameters are exported (never checkpoint EMA params).
Frozen actor leaves and normalization assets are fingerprinted once at startup.
"""
import dataclasses
import hashlib
import json
import zipfile

import flax
import jax
import numpy as np

from .buffer import Buffer


def _leaves(tree, path=()):
    if isinstance(tree, dict):
        for key in sorted(tree, key=str):
            yield from _leaves(tree[key], path + (key,))
    else:
        yield path, tree


def _set(tree, path, value):
    for key in path[:-1]:
        tree = tree[key]
    tree[path[-1]] = value


def parameter_trees(agent):
    return {
        'actor': agent.actor_train_state.params.filter(agent.actor.train_config.trainable_filter).to_pure_dict(),
        'encoder': flax.serialization.to_state_dict(agent.batch_encoder.params),
        'edit': flax.serialization.to_state_dict(agent.edit_actor.params),
        'target_critic': flax.serialization.to_state_dict(agent.target_critic.params),
    }


def identity(agent, task_contract):
    _, frozen = agent.actor_train_state.params.split(agent.actor.train_config.trainable_filter, ...)
    h = hashlib.sha256()
    for path, value in _leaves(frozen.to_pure_dict()):
        a = np.asarray(jax.device_get(value))
        h.update(json.dumps([path, a.shape, a.dtype.name]).encode())
        h.update(a.tobytes())
    stats = hashlib.sha256()
    for name, value in sorted((agent.actor.data_config.norm_stats or {}).items()):
        for field in dataclasses.fields(value):
            a = getattr(value, field.name)
            if a is not None:
                a = np.asarray(a)
                stats.update(json.dumps([name, field.name, a.shape, a.dtype.name]).encode())
                stats.update(a.tobytes())
    sampling = {k: getattr(agent, k) for k in ('N', 'n_edit_samples', 'edit_scale', 'edit_action_xyzg',
                'replan_steps', 'action_horizon', 'action_dim', 'state_dim', 'num_qs', 'num_min_qs',
                'resize_size', 'freeze_encoder', 'critic_camera_keys')}
    # JSON normalization permits tuple/list differences between peers.
    model = dataclasses.asdict(agent.actor.model_config)
    return json.loads(json.dumps({'schema': 1, 'frozen_actor': h.hexdigest(), 'norm_stats': stats.hexdigest(),
                                 'model': model, 'pi05_config': agent.actor.train_config.name,
                                 'use_quantile_norm': agent.actor.data_config.use_quantile_norm,
                                 'policy_metadata': agent.actor.train_config.policy_metadata,
                                 'sampling': sampling, 'task': task_contract}))


def export_policy(agent, contract, version):
    """Serialize directly into anonymous RAM; return a sealed shared buffer."""
    buffer = Buffer.create()
    arrays = []
    try:
        with buffer.open() as stream, zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for group, tree in parameter_trees(agent).items():
                for keys, value in _leaves(tree):
                    array = np.asarray(jax.device_get(value))
                    name = 'array-{}.npy'.format(len(arrays))
                    arrays.append({'group': group, 'path': keys, 'name': name, 'shape': array.shape, 'dtype': array.dtype.name})
                    with archive.open(name, 'w', force_zip64=True) as f:
                        np.lib.format.write_array(f, array, allow_pickle=False)
            archive.writestr('manifest.json', json.dumps({'identity': contract, 'version': version, 'arrays': arrays}))
        buffer.seal()
        return buffer
    except BaseException:
        buffer.close()
        raise


def import_policy(agent, buffer, contract, expected_version):
    """Build and validate an entire candidate before exposing any new parameters.

    This is called at a round barrier, with no inference still in flight. The
    receiver's RNG remains its own: snapshot install does not rewind sampling.
    """
    trees = parameter_trees(agent)
    expected = {(group, keys): value for group, tree in trees.items() for keys, value in _leaves(tree)}
    seen = set()
    with buffer.open() as stream, zipfile.ZipFile(stream) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        if manifest['identity'] != contract or manifest['version'] != expected_version:
            raise ValueError('policy version/base/assets/task identity mismatch')
        for item in manifest['arrays']:
            pair = (item['group'], tuple(item['path']))
            if pair in seen or pair not in expected:
                raise ValueError('unexpected or duplicate snapshot leaf')
            seen.add(pair)
            target = expected[pair]
            if list(target.shape) != item['shape'] or np.dtype(target.dtype).name != item['dtype']:
                raise ValueError('snapshot shape/dtype mismatch')
            with archive.open(item['name']) as f:
                value = np.lib.format.read_array(f, allow_pickle=False)
            # NumPy's .npy stores bfloat16 as a 2-byte void dtype; preserve it by metadata.
            value = value.view(np.dtype(item['dtype']))
            if value.shape != target.shape:
                raise ValueError('array shape mismatch')
            if not np.isfinite(value).all():
                raise ValueError('nonfinite policy parameter')
            _set(trees[pair[0]], pair[1], jax.device_put(value, agent.actor.infer_sharding))
    if seen != set(expected):
        raise ValueError('snapshot missing parameters')
    params = jax.tree_util.tree_map(lambda x: x, agent.actor_train_state.params)
    params.replace_by_pure_dict(trees['actor'])
    actor_state = dataclasses.replace(agent.actor_train_state, params=params, opt_state=(), ema_params=None)
    candidate = agent.replace(
        actor_train_state=actor_state,
        batch_encoder=agent.batch_encoder.replace(params=flax.serialization.from_state_dict(agent.batch_encoder.params, trees['encoder'])),
        edit_actor=agent.edit_actor.replace(params=flax.serialization.from_state_dict(agent.edit_actor.params, trees['edit'])),
        target_critic=agent.target_critic.replace(params=flax.serialization.from_state_dict(agent.target_critic.params, trees['target_critic'])),
        _infer_cache=None,
    )
    jax.block_until_ready((candidate.actor_train_state.params, candidate.batch_encoder.params,
                           candidate.edit_actor.params, candidate.target_critic.params))
    candidate = candidate.cache_infer_params()
    jax.block_until_ready(candidate._infer_cache)
    return candidate
