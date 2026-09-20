import jax
import numpy as np
import pytest

from conftest import transition
from expo_ft.data.batch_processor import BatchProcessor, sample_robot_buffers
from expo_ft.data.replay_buffer import save_replay_buffer_transition, restore_replay_buffer


def fill(buffer, robot, length=12, success=True):
    for index in range(length):
        buffer.insert(transition(robot, index, index == length - 1, success))


def test_chunks_and_nstep_do_not_cross_robots(make_buffer):
    buffers = [make_buffer(1), make_buffer(2)]
    # Interleaving insertion in wall-clock time must not interleave storage.
    # Include the following episode's rows: the existing sampler excludes the
    # final n_step rows until subsequent observations have been inserted.
    for index in range(14):
        for robot, buffer in enumerate(buffers):
            buffer.insert(transition(robot, index, index in (11, 13), True))
    batch = sample_robot_buffers(buffers, 256, np.random.default_rng(4))
    assert set(batch["state"][:, 0]) == {0, 1}
    np.testing.assert_array_equal(batch["state"][:, 0], batch["next_state"][:, 0])
    assert np.all(batch["actions"][:, :, 0] == batch["state"][:, 0, None])
    np.testing.assert_array_equal(batch["next_state"][:, 1], batch["state"][:, 1] + 2)
    # The n-step discounted terminal reward is still computed by PiReplayBuffer.
    terminal_window = batch["state"][:, 1] == 10
    assert terminal_window.any()
    np.testing.assert_allclose(batch["rewards"][terminal_window], 0.99)
    assert np.all(batch["masks"][terminal_window] == 0)
    for buffer in buffers:
        assert np.all(buffer.dataset_dict["actions"][11, :, 1] == 11)


@pytest.mark.parametrize("offline_ratio", [0, 0.5, 1])
def test_batch_size_offline_ratio_and_success_filter(make_buffer, offline_ratio):
    online = [make_buffer(1), make_buffer(2)]
    offline = make_buffer(3)
    fill(online[0], 0, success=False)
    fill(online[1], 1, success=True)
    fill(offline, 9, success=True)
    processor = BatchProcessor(online[0], offline, None, 16, 2, offline_ratio,
                               True, False, replay_buffers=online)
    batch, actor, _ = processor.next_batch(jax.random.PRNGKey(0))
    assert batch["state"].shape == (32, 2)
    assert actor["state"].shape == (16, 2)
    assert np.count_nonzero(np.asarray(batch["state"])[:, 0] == 9) == int(32 * offline_ratio)
    assert set(np.asarray(actor["state"])[:, 0]) <= {1, 9}


def test_empty_worker_and_offline_success_fallback(make_buffer):
    online = [make_buffer(1), make_buffer(2), make_buffer(3)]
    offline = make_buffer(4)
    fill(online[0], 0, success=False)
    fill(offline, 9, success=True)
    processor = BatchProcessor(online[0], offline, None, 8, 1, 0.5,
                               True, False, replay_buffers=online)
    _, actor, _ = processor.next_batch(jax.random.PRNGKey(0))
    assert np.all(np.asarray(actor["state"])[:, 0] == 9)
    assert sample_robot_buffers(online, 8, np.random.default_rng(), success_only=True) is None


def test_new_round_is_sampled_without_stale_prefetch(make_buffer):
    online = [make_buffer(1), make_buffer(2)]
    offline = make_buffer(3)
    fill(online[0], 0)
    processor = BatchProcessor(online[0], offline, None, 64, 1, 0, False, False, replay_buffers=online)
    first, _, rng = processor.next_batch(jax.random.PRNGKey(0))
    assert np.all(np.asarray(first["state"])[:, 0] == 0)
    fill(online[1], 1, length=100)
    second, _, _ = processor.next_batch(rng)
    assert np.count_nonzero(np.asarray(second["state"])[:, 0] == 1) > 32


def test_replay_save_restore_isolated_and_cutoff(tmp_path, make_buffer):
    for robot in range(2):
        for step in range(6):
            save_replay_buffer_transition(tmp_path / f"robot-{robot}",
                                          transition(robot, step, step in (3, 5), True), step=step + 1)
        buffer = make_buffer(robot)
        restore_replay_buffer(tmp_path / f"robot-{robot}", buffer, up_to_step=4)
        assert len(buffer) == 4
        assert np.all(buffer.dataset_dict["state"][:4, 0] == robot)
        assert np.all(buffer.dataset_dict["is_success"][:4])


@pytest.mark.parametrize("sample_kwargs", [{}, {"success_only": True}, {"hil_only": True}])
def test_sample_into_slice_matches_existing_sampler(make_buffer, sample_kwargs):
    buffer = make_buffer()
    for index in range(24):
        buffer.insert(transition(3, index, index in (11, 23), index < 12, hil=index % 3 == 0))
    buffer.rng = jax.random.PRNGKey(12)
    expected = buffer.sample_jax(320, **sample_kwargs)
    buffer.rng = jax.random.PRNGKey(12)
    storage = buffer.allocate_sample_batch(640)
    for value in storage.values():
        value[:] = 0
    actual = buffer.sample_jax(320, out={key: value[160:480] for key, value in storage.items()}, **sample_kwargs)
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], np.asarray(value), err_msg=key)
        assert not np.any(storage[key][:160])
        assert not np.any(storage[key][480:])


def test_full_batch_storage_reused_across_variable_robot_allocations(make_buffer):
    online = [make_buffer(1), make_buffer(2)]
    for index, buffer in enumerate(online):
        fill(buffer, index, length=100)
    processor = BatchProcessor(online[0], make_buffer(3), None, 64, 20, 0,
                               True, False, replay_buffers=online)
    rng = jax.random.PRNGKey(4)
    identities = None
    allocations = set()
    first = None
    for _ in range(30):
        batch, actor, rng = processor.next_batch(rng)
        assert batch["state"].shape == (1280, 2)
        assert actor["state"].shape == (64, 2)
        allocations.add(int(np.count_nonzero(np.asarray(batch["state"])[:, 0] == 0)))
        current = {(size, key): id(value) for size, values in processor._robot_batches.items()
                   for key, value in values.items()}
        if identities is None:
            identities = current
            first = batch
            first_copy = jax.tree.map(lambda value: np.asarray(value).copy(), first)
        assert current == identities
        assert set(processor._robot_batches) == {64, 1280}
        assert not any(buffer._stage for buffer in online)
        for size, values in processor._robot_batches.items():
            assert all(value.shape[0] == size for value in values.values())
    assert len(allocations) > 10
    for actual, expected in zip(jax.tree.leaves(first), jax.tree.leaves(first_copy)):
        np.testing.assert_array_equal(actual, expected)
