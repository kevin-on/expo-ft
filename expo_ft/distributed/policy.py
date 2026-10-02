"""Inference snapshot format, independent of the network transport.

Only the actual rollout parameters are exported (never checkpoint EMA params).
Frozen actor leaves and normalization assets are fingerprinted once at startup.
"""
import dataclasses
import hashlib
import json
import math
import struct

import flax
import jax
import numpy as np

from .buffer import Buffer
from .learner_group import local_value


# Versioned, uncompressed wire format: header, aligned C-order array bytes,
# then a JSON manifest. Checkpoint files are independent of this format.
_HEADER = struct.Struct('<8sQQ')
_MAGIC = b'EXPOARR1'
_ALIGNMENT = 64
_MAX_MANIFEST_BYTES = 16 * 1024**2


def _aligned(size):
    return (size + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


def _read_manifest(payload):
    """Validate the entire byte layout before creating views or copying to GPU."""
    if len(payload) < _ALIGNMENT:
        raise ValueError('truncated snapshot header')
    magic, offset, length = _HEADER.unpack_from(payload)
    if magic != _MAGIC:
        raise ValueError('unsupported snapshot format; update both peers')
    if (offset < _ALIGNMENT or length > _MAX_MANIFEST_BYTES
            or offset + length != len(payload)):
        raise ValueError('invalid snapshot manifest range')
    manifest = json.loads(bytes(payload[offset:]))
    if not isinstance(manifest, dict) or not isinstance(manifest.get('arrays'), list):
        raise ValueError('invalid snapshot manifest')
    cursor = _ALIGNMENT
    for item in manifest['arrays']:
        shape = item['shape']
        dtype = np.dtype(item['dtype'])
        if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                or dtype.hasobject or dtype.fields is not None or not dtype.isnative
                or dtype.itemsize == 0):
            raise ValueError('invalid snapshot array type')
        size = math.prod(shape) * dtype.itemsize
        if (type(item['offset']) is not int or type(item['nbytes']) is not int
                or item['offset'] != _aligned(cursor) or item['nbytes'] != size
                or item['offset'] + size > offset):
            raise ValueError('invalid snapshot array range')
        cursor = item['offset'] + size
    if cursor != offset:
        raise ValueError('unexpected bytes before snapshot manifest')
    return manifest


def _array_view(payload, item):
    # frombuffer retains an exported memoryview, so an accidentally live view
    # prevents mmap.close instead of becoming a dangling NumPy pointer.
    return np.frombuffer(payload, dtype=np.dtype(item['dtype']),
                         count=math.prod(item['shape']), offset=item['offset']).reshape(item['shape'])


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
        a = local_value(value)
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
                                 'sampling': sampling, 'task': task_contract,
                                 'online_config': {k: v for k, v in getattr(agent.actor, 'checkpoint_record', {}).get('config', {}).items()
                                                   if k not in ('initial_sft_checkpoint', 'initial_sft_base')}}))


def export_policy(agent, contract, version):
    """Serialize directly into anonymous RAM; return a sealed shared buffer."""
    buffer = Buffer.create()
    arrays = []
    try:
        with buffer.open() as stream:
            stream.write(bytes(_ALIGNMENT))
            for group, tree in parameter_trees(agent).items():
                for keys, value in _leaves(tree):
                    array = local_value(value)
                    if array.dtype.hasobject or array.dtype.fields is not None or not array.dtype.isnative:
                        raise ValueError('unsupported snapshot array dtype')
                    if not array.flags.c_contiguous:
                        array = np.ascontiguousarray(array)
                    offset = _aligned(stream.tell())
                    stream.write(bytes(offset - stream.tell()))
                    arrays.append({'group': group, 'path': keys, 'offset': offset, 'nbytes': array.nbytes,
                                   'shape': array.shape, 'dtype': array.dtype.name})
                    # The uint8 view also works for bfloat16, which does not
                    # implement Python's typed buffer protocol.
                    stream.write(memoryview(array.reshape(-1).view(np.uint8)))
            offset = stream.tell()
            manifest = json.dumps({'identity': contract, 'version': version, 'arrays': arrays}).encode()
            if len(manifest) > _MAX_MANIFEST_BYTES:
                raise ValueError('oversized snapshot manifest')
            stream.write(manifest)
            stream.seek(0)
            stream.write(_HEADER.pack(_MAGIC, offset, len(manifest)))
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
    sharding = agent.actor.infer_sharding
    devices = getattr(sharding, 'device_set', {sharding})
    # JAX's CPU backend may retain host memory even with may_alias=False.
    # GPU installs view the received RAM; CPU validation needs owned host memory.
    copy_host = any(device.platform == 'cpu' for device in devices)
    seen = set()
    with buffer.view() as payload:
        manifest = _read_manifest(payload)
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
        if seen != set(expected):
            raise ValueError('snapshot missing parameters')
        host_values, device_values = [], []
        value = None
        try:
            for item in manifest['arrays']:
                value = _array_view(payload, item)
                if copy_host:
                    value = value.copy()
                host_values.append(value)
                placed = jax.device_put(value, agent.actor.infer_sharding, may_alias=False)
                device_values.append(placed)
                _set(trees[item['group']], tuple(item['path']), placed)
        finally:
            # Drain earlier asynchronous copies even if a later transfer fails.
            # Keep host views alive until then, and release them before Buffer.close.
            try:
                jax.block_until_ready(device_values)
            finally:
                value = None
                host_values.clear()
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
    candidate = candidate.cache_infer_params()
    jax.block_until_ready(candidate._infer_cache)
    return candidate
