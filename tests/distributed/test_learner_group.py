"""Small host tests for the multi-node layout; no robots or model weights."""
import unittest
from unittest import mock

import jax
import numpy as np

from expo_ft.distributed.learner_group import LearnerGroup, local_value, replicate
from expo_ft.utils.augmentation import make_data_augmentation_fn


class LayoutTest(unittest.TestCase):
    def test_augmentation_preserves_per_image_rng(self):
        images = np.linspace(-1, 1, 6 * 16 * 16 * 3, dtype=np.float32).reshape(6, 16, 16, 3)
        flat = {'base_0_rgb': images, 'left_wrist_0_rgb': images[:, :, ::-1].copy()}
        blocked = jax.tree.map(lambda x: x.reshape((2, 3) + x.shape[1:]), flat)
        for full in (False, True):
            augment = make_data_augmentation_fn(full)
            expected = augment(jax.random.PRNGKey(5), flat)
            actual = augment(jax.random.PRNGKey(5), blocked)
            for camera in flat:
                np.testing.assert_allclose(np.asarray(actual[camera]).reshape(images.shape),
                                           expected[camera], atol=1e-6, rtol=1e-6)

    def test_single_host_call_preserves_exceptions_and_values(self):
        group = LearnerGroup()
        value = object()
        self.assertIs(group.call(lambda: value), value)
        with self.assertRaisesRegex(ValueError, 'example'):
            group.call(lambda: (_ for _ in ()).throw(ValueError('example')))

    def test_replicated_value(self):
        mesh = jax.sharding.Mesh(np.array(jax.devices()), ('batch',))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        np.testing.assert_array_equal(local_value(replicate(np.array([1, 2], np.uint32), sharding)), [1, 2])

    def test_multihost_cache_uses_explicit_shared_root(self):
        from expo_ft.utils.train_utils import set_compilation_cache_dir
        with mock.patch('jax.process_count', return_value=2), \
             mock.patch('jax.device_count', return_value=8), \
             mock.patch('jax.config.update') as update, \
             mock.patch.dict('os.environ', {'JAX_COMPILATION_CACHE_DIR': '/shared/expo-jax'}):
            self.assertEqual(set_compilation_cache_dir('sync'), '/shared/expo-jax/sync-n8')
            update.assert_called_once_with('jax_compilation_cache_dir', '/shared/expo-jax/sync-n8')
        with mock.patch('jax.process_count', return_value=2), \
             mock.patch.dict('os.environ', {'JAX_COMPILATION_CACHE_DIR': ''}):
            with self.assertRaisesRegex(ValueError, 'shared storage'):
                set_compilation_cache_dir('sync')
