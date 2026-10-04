"""Tiny CPU regressions for SFT eval precision; no large weights or hardware.

Run with JAX_PLATFORMS=cpu and the CPU test dependencies. OpenPI's factory,
config and policy wrapper are replaced at their import boundaries; the parent
SFT caller and compact reconstruction, serialization, and NNX state are real.
The OpenPI factory's actual checkpoint restore has its own OpenPI tests.
"""
from contextlib import contextmanager
import dataclasses
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from flax import nnx
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from expo_ft.env.checkpoint_policy import SFTPolicy
from expo_ft.eval import checkpoint, model as compact_model


class TinyProjection(nnx.Module):
    def __init__(self):
        self.kernel = nnx.Param(jnp.zeros((2, 2), dtype=jnp.float32))


class TinyBackbone(nnx.Module):
    def __init__(self):
        self.kernel = nnx.Param(jnp.zeros((2, 2), dtype=jnp.bfloat16))
        self.lora_a = nnx.Param(jnp.zeros((2, 2), dtype=jnp.float32))
        self.lora_b = nnx.Param(jnp.zeros((2, 2), dtype=jnp.float32))


class TinyActor(nnx.Module):
    def __init__(self):
        self.backbone = TinyBackbone()
        self.action_in_proj = TinyProjection()
        self.action_out_proj = TinyProjection()


@dataclasses.dataclass(frozen=True)
class TinyModelConfig:
    def create(self, rng):
        return TinyActor()

    def load(self, params, *, remove_extra_params=True):
        graph, state = nnx.split(self.create(jax.random.key(0)))
        state.replace_by_pure_dict(params)
        return nnx.merge(graph, state)


@dataclasses.dataclass
class TransformGroup:
    inputs: tuple = ()
    outputs: tuple = ()


@dataclasses.dataclass(frozen=True)
class TinyDataConfig:
    use_cartesian_state: bool = True
    output_action_dim: int = 7

    def create(self, assets_dirs, model):
        return SimpleNamespace(data_transforms=TransformGroup(), model_transforms=TransformGroup(),
                               use_quantile_norm=False)


@dataclasses.dataclass(frozen=True)
class TinyTrainConfig:
    model: TinyModelConfig = dataclasses.field(default_factory=TinyModelConfig)
    data: TinyDataConfig = dataclasses.field(default_factory=TinyDataConfig)
    assets_dirs: tuple = ()
    policy_metadata: dict = dataclasses.field(default_factory=dict)

    @property
    def freeze_filter(self):
        return lambda path, value: path == ('backbone', 'kernel')


class CapturedPolicy:
    def __init__(self, model, **kwargs):
        self.model = model


def parameters(policy):
    return nnx.state(policy.policy.model).to_pure_dict()


class SFTPrecisionTest(unittest.TestCase):
    def setUp(self):
        self.config = TinyTrainConfig()
        # Every trainable value loses information when rounded to BF16.
        values = np.array([[1.0001, -.33331], [.123456, 2.001]], dtype=np.float32)
        self.saved = {
            'backbone': {
                'kernel': np.array([[1., .5], [-.25, 2.]], dtype=ml_dtypes.bfloat16),
                'lora_a': values.copy(),
                'lora_b': values * np.float32(.731),
            },
            'action_in_proj': {'kernel': values * np.float32(1.17)},
            'action_out_proj': {'kernel': values * np.float32(.917)},
        }
        self.factory_calls = []

    @contextmanager
    def openpi_boundary(self):
        """Scope stand-ins so this test does not alter other tests' OpenPI modules."""
        names = ('openpi', 'openpi.training', 'openpi.training.checkpoint_config',
                 'openpi.policies', 'openpi.policies.policy', 'openpi.policies.policy_config',
                 'openpi.transforms')
        modules = {name: ModuleType(name) for name in names}
        for name, module in modules.items():
            module.__path__ = []
            if '.' in name:
                parent, child = name.rsplit('.', 1)
                setattr(modules[parent], child, module)
        modules['openpi.training.checkpoint_config'].load = lambda path: self.config

        def create_trained_policy(config, path, *, params_dtype=jnp.bfloat16, **kwargs):
            # Model the factory's explicit dtype contract. Its real restore is
            # separately covered in OpenPI; this catches a missing caller opt-out.
            self.factory_calls.append((params_dtype, kwargs))
            params = jax.tree.map(lambda value: jnp.asarray(value, dtype=params_dtype), self.saved)
            return CapturedPolicy(config.model.load(params))

        modules['openpi.policies.policy_config'].create_trained_policy = create_trained_policy
        modules['openpi.policies.policy'].Policy = CapturedPolicy
        transforms = modules['openpi.transforms']
        transforms.InjectDefaultPrompt = lambda prompt: ('prompt', prompt)
        transforms.Normalize = lambda stats, **kwargs: ('normalize', stats)
        transforms.Unnormalize = lambda stats, **kwargs: ('unnormalize', stats)
        with patch.dict(sys.modules, modules):
            yield

    def assert_training_precision(self, actual):
        expected = dict(checkpoint.leaves(self.saved))
        found = dict(checkpoint.leaves(actual))
        self.assertEqual(found.keys(), expected.keys())
        for path, value in found.items():
            with self.subTest(parameter='/'.join(path)):
                self.assertEqual(value.dtype, expected[path].dtype)
                np.testing.assert_array_equal(np.asarray(value), expected[path])
                if expected[path].dtype == np.float32:
                    rounded = expected[path].astype(ml_dtypes.bfloat16).astype(np.float32)
                    self.assertFalse(np.array_equal(np.asarray(value), rounded))

    def test_full_sft_caller_preserves_frozen_and_trainable_precision(self):
        with self.openpi_boundary():
            policy = SFTPolicy(Path('/unused/tiny-checkpoint'), seed=17, prompt='pick')
        self.assertEqual(len(self.factory_calls), 1)
        dtype, kwargs = self.factory_calls[0]
        self.assertIsNone(dtype)
        self.assertEqual(kwargs, {'seed': 17, 'default_prompt': 'pick'})
        self.assert_training_precision(parameters(policy))

    def test_compact_reconstruction_matches_full_after_payloads_close(self):
        # Frozen base starts as FP32, as a separate base checkpoint may do.
        # Trainable base values are deliberately wrong: only the payload wins.
        base = jax.tree.map(lambda value: np.full(value.shape, -9., dtype=np.float32), self.saved)
        base['backbone']['kernel'] = self.saved['backbone']['kernel'].astype(np.float32)
        _, trained = checkpoint.split_actor(self.saved, self.config)
        metadata = dict(kind='sft', training_run_id='tiny', checkpoint_path='sft/tiny/checkpoints/1',
                        hash_algorithm='xxh3_128', files={})
        with self.openpi_boundary(), patch.object(compact_model, 'config_and_norm',
                                                 return_value=(self.config, {})):
            full = SFTPolicy(Path('/unused/tiny-checkpoint'), seed=17, prompt='pick')
            with checkpoint.pack({'base': base}, {'kind': 'base'}) as base_payload:
                with checkpoint.pack({'actor': trained}, metadata) as trained_payload:
                    compact = compact_model.build(base_payload, trained_payload,
                        SimpleNamespace(language_instruction='pick'), seed=17, base_verified=True)

        # Access after both sealed RAM buffers close: model leaves must own their
        # storage and preserve FP32 LoRA/projections alongside the BF16 backbone.
        self.assert_training_precision(parameters(compact))
        full_params = dict(checkpoint.leaves(parameters(full)))
        for path, value in checkpoint.leaves(parameters(compact)):
            reference = full_params[path]
            self.assertEqual(value.dtype, reference.dtype)
            np.testing.assert_array_equal(np.asarray(value), np.asarray(reference))


if __name__ == '__main__':
    unittest.main()
