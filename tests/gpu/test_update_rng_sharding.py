"""Run with four virtual CPU devices to reproduce the inference/update boundary."""
from dataclasses import dataclass, replace
import importlib.util
from pathlib import Path

import jax
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('sharding_utils',
    Path(__file__).resolve().parents[2] / 'expo_ft/agents/alg/sharding_utils.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@dataclass(frozen=True)
class Learner:
    rng: object
    params: object
    data_sharding: object

    def replace(self, **kwargs):
        return replace(self, **kwargs)


def test_rng_repair_preserves_partitioned_parameters():
    if jax.device_count() != 4:
        pytest.skip('requires four virtual CPU devices')
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ('fsdp',))
    partitioned = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('fsdp'))
    key = jax.device_put(jax.random.PRNGKey(42), jax.sharding.SingleDeviceSharding(jax.devices()[0]))
    params = jax.device_put(np.arange(16, dtype=np.float32), partitioned)
    learner = Learner(key, params, partitioned)
    operation = jax.jit(lambda rng, p: p + rng[0].astype(p.dtype))
    with pytest.raises(ValueError, match='incompatible devices'):
        operation(learner.rng, learner.params)
    repaired, alias = module.place_update_rng(learner, learner)
    assert repaired is alias
    assert repaired.params is params
    assert not repaired.params.is_fully_replicated
    assert repaired.rng.sharding.device_set == set(jax.devices())
    assert repaired.rng.is_fully_replicated
    np.testing.assert_array_equal(repaired.rng, key)
    np.testing.assert_array_equal(operation(repaired.rng, repaired.params), np.arange(16))
    other = learner.replace(rng=jax.device_put(jax.random.PRNGKey(7), jax.devices()[0]))
    left, right = module.place_update_rng(learner, other)
    assert left is not right
    assert right.params is other.params
    np.testing.assert_array_equal(right.rng, other.rng)
