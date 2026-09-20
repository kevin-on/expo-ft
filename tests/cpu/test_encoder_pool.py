"""Keep the TPU pooling workaround equivalent to the original encoder forward pass."""
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from expo_ft.networks.encoders import ResNetV2Encoder


@pytest.mark.parametrize("batched", [False, True])
def test_encoder_matches_original_same_pool_and_has_finite_gradients(monkeypatch, batched):
    shape = (1, 224, 224, 6) if batched else (224, 224, 6)
    images = jax.random.normal(jax.random.PRNGKey(1), shape)
    encoder = ResNetV2Encoder(stage_sizes=(1,), num_filters=4)
    variables = encoder.init(jax.random.PRNGKey(2), images)
    actual = encoder.apply(variables, images)
    grads = jax.grad(lambda p: jnp.mean(encoder.apply({"params": p}, images) ** 2))(variables["params"])
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grads))
    assert np.any(np.asarray(grads["Conv_0"]["kernel"]) != 0)

    original_pool = nn.max_pool
    def legacy_pool(x, window_shape, strides, padding):
        assert padding == "VALID"
        # A 224px input produces a 112px stem. Remove its explicit bottom/right
        # padding to compare with the previous SAME pooling on the same values.
        return original_pool(x[..., :-1, :-1, :], window_shape, strides, padding="SAME")
    monkeypatch.setattr(nn, "max_pool", legacy_pool)
    expected = encoder.apply(variables, images)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
