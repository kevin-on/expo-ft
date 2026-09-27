"""Snapshot layout, atomic install, and mapped-RAM lifetime without model weights."""
import dataclasses
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from expo_ft.distributed import policy
from expo_ft.distributed.buffer import Buffer


class Trainable(nnx.Param):
    pass


@dataclasses.dataclass(frozen=True)
class State:
    params: object
    opt_state: object = ()
    ema_params: object = None
    replace = dataclasses.replace


@dataclasses.dataclass(frozen=True)
class Agent:
    actor: object
    actor_train_state: State
    batch_encoder: State
    edit_actor: State
    target_critic: State
    rng: object
    _infer_cache: object = None
    replace = dataclasses.replace

    def cache_infer_params(self):
        return self.replace(_infer_cache=tuple(jax.tree.leaves(self.actor_train_state.params)))


def agent(offset=0):
    class Model(nnx.Module):
        def __init__(self):
            self.train = Trainable(jnp.arange(12, dtype=jnp.bfloat16).reshape(3, 4) + offset)
            self.frozen = nnx.Param(jnp.arange(7, dtype=jnp.float32))
    params = nnx.state(Model())
    return Agent(SimpleNamespace(train_config=SimpleNamespace(trainable_filter=Trainable),
                                 infer_sharding=jax.devices()[0]),
                 State(params, opt_state=(jnp.ones(3),), ema_params={'unused': jnp.zeros(1)}),
                 State({'scalar': jnp.array(offset, dtype=jnp.float32),
                        'half': jnp.arange(8, dtype=jnp.float16) + offset,
                        'empty': jnp.empty((0, 3), dtype=jnp.float32)}),
                 State({'kernel': jnp.arange(15, dtype=jnp.float32).reshape(3, 5) + offset}),
                 State({'count': jnp.array([1, 2], dtype=jnp.int32)}), jax.random.PRNGKey(73))


def contents(model):
    return {(group, path): np.asarray(v).copy() for group, tree in policy.parameter_trees(model).items()
            for path, v in policy._leaves(tree)}


def rewrite(buffer, mutate):
    with buffer.view() as view:
        _, offset, _ = policy._HEADER.unpack_from(view)
        manifest = policy._read_manifest(view)
        body = bytearray(view[:offset])
    mutate(manifest)
    metadata = json.dumps(manifest).encode()
    policy._HEADER.pack_into(body, 0, policy._MAGIC, offset, len(metadata))
    return Buffer.from_bytes(body + metadata)


class SnapshotTest(unittest.TestCase):
    def setUp(self):
        self.source, self.target = agent(2), agent(-7)
        self.contract = {'test': 'snapshot'}

    def test_exact_values_frozen_rng_and_buffer_lifetime(self):
        before = contents(self.target)
        with policy.export_policy(self.source, self.contract, 12) as buffer:
            with buffer.view() as payload:
                manifest = policy._read_manifest(payload)
                for item in manifest['arrays']:
                    array = policy._array_view(payload, item)
                    self.assertFalse(array.flags.owndata)
                    self.assertFalse(array.flags.writeable)
                    self.assertEqual(array.ctypes.data % 64, 0)
                    expected = contents(self.source)[item['group'], tuple(item['path'])]
                    self.assertEqual(array.dtype, expected.dtype)
                    self.assertEqual(array.tobytes(), expected.tobytes())
                del array
            result = policy.import_policy(self.target, buffer, self.contract, 12)
        # Read device values after the entire input RAM has been closed.
        for key, value in contents(result).items():
            np.testing.assert_array_equal(value, contents(self.source)[key])
            np.testing.assert_array_equal(contents(self.target)[key], before[key])
        np.testing.assert_array_equal(result.rng, self.target.rng)
        self.assertIs(result.actor_train_state.params['frozen'].value,
                      self.target.actor_train_state.params['frozen'].value)
        self.assertFalse(result.actor_train_state.opt_state)
        self.assertIsNone(result.actor_train_state.ema_params)

    def test_noncontiguous_export_keeps_shape_and_values(self):
        self.source = self.source.replace(edit_actor=State({'kernel': np.arange(15, dtype=np.float32).reshape(5, 3).T}))
        with policy.export_policy(self.source, self.contract, 0) as buffer:
            result = policy.import_policy(self.target, buffer, self.contract, 0)
        np.testing.assert_array_equal(result.edit_actor.params['kernel'], self.source.edit_actor.params['kernel'])

    def test_invalid_metadata_is_rejected_before_any_device_copy(self):
        mutations = [
            lambda m: m.update(identity={}),
            lambda m: m.update(version=99),
            lambda m: m['arrays'][0].update(dtype='float32'),
            lambda m: m['arrays'][0].update(shape=[-1]),
            lambda m: m['arrays'][0].update(shape=[True]),
            lambda m: m['arrays'][0].update(dtype='object'),
            lambda m: m['arrays'][0].update(offset=0),
            lambda m: m['arrays'][0].update(offset=65),
            lambda m: m['arrays'][0].update(nbytes=999999999),
            lambda m: m['arrays'][1].update(offset=m['arrays'][0]['offset']),
            lambda m: m['arrays'][0].update(group='unknown'),
            lambda m: m['arrays'][-1].update(group=m['arrays'][0]['group'], path=m['arrays'][0]['path']),
            lambda m: m['arrays'].pop(),
        ]
        with policy.export_policy(self.source, self.contract, 0) as original:
            for mutate in mutations:
                with self.subTest(mutate=mutate), rewrite(original, mutate) as corrupted:
                    with patch.object(policy.jax, 'device_put', side_effect=AssertionError('unexpected copy')):
                        with self.assertRaises((ValueError, TypeError)):
                            policy.import_policy(self.target, corrupted, self.contract, 0)

    def test_bad_header_and_truncation(self):
        with policy.export_policy(self.source, self.contract, 0) as original:
            with original.view() as payload:
                data = bytes(payload)
        for blob in (data[:20], b'old-zip!' + data[8:], data[:-1], data+b'x'):
            with Buffer.from_bytes(blob) as buffer:
                with self.assertRaises(ValueError):
                    policy.import_policy(self.target, buffer, self.contract, 0)


if __name__ == '__main__':
    unittest.main()
