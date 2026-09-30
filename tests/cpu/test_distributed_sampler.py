import jax
import numpy as np
import pytest

from conftest import transition
from expo_ft.data.batch_processor import BatchProcessor
from expo_ft.data.distributed_sampler import DistributedReplaySampler
from expo_ft.data.replay_buffer import save_replay_buffer_batch, restore_replay_buffer


def sampler(rank=0, world_size=2):
    return DistributedReplaySampler(seed=42, global_batch_size=64, rank=rank, world_size=world_size)


@pytest.mark.parametrize('counts', [(20, 80), (20, 44), (0, 70), (100,)])
def test_global_minibatches_are_unique_and_split_exactly(counts):
    args = dict(update_step=17, num_batches=20)
    expected = sampler(world_size=1).sample(counts, **args)
    halves = [sampler(rank).sample(counts, **args) for rank in range(2)]
    actual = np.concatenate(halves, axis=1)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (20, 64, 2)
    for row in actual:
        assert len({tuple(pair) for pair in row}) == 64
    # There need not be 20*64 distinct candidates: each minibatch is independent.
    assert len({tuple(pair) for pair in actual.reshape(-1, 2)}) < 20 * 64


def test_small_and_empty_candidate_pools_are_handled_in_sampler():
    candidates = [0, np.array([3, 7, 20]), 5]
    actual = np.concatenate([sampler(rank).sample(candidates, update_step=1, num_batches=20)
                             for rank in range(2)], axis=1)
    allowed = {(1, 3), (1, 7), (1, 20)} | {(2, row) for row in range(5)}
    for batch in actual:
        assert set(map(tuple, batch)) <= allowed
        assert len(set(map(tuple, batch))) < 64
    assert sampler().sample([0, np.array([], dtype=int)], update_step=1, num_batches=20) is None


def test_sampling_is_stateless_and_streams_are_separate():
    current = sampler()
    args = dict(candidates=[100, 200], update_step=7, num_batches=20)
    expected = current.sample(**args)
    assert not np.array_equal(expected, current.sample(**dict(args, update_step=8)))
    assert not np.array_equal(expected, current.sample(**args, stream='actor'))
    current.sample([500, 600], update_step=900, num_batches=3)
    np.testing.assert_array_equal(current.sample(**args), expected)
    # Restoring only the update counter recreates the next plan; no RNG cursor.
    np.testing.assert_array_equal(sampler().sample(**args), expected)


def processor(buffers, offline, rank, world_size):
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('batch',))
    data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('batch'))
    return BatchProcessor(buffers[0], offline, data, 64 // world_size, 20, 0, True, False,
                          replay_buffers=buffers, utd_axis=True,
                          distributed_sampler=sampler(rank, world_size))


def populate(buffers):
    records = []
    for robot, buffer in enumerate(buffers):
        rows = [transition(robot, index, index in (59, 119), success=robot == 1)
                for index in range(120)]
        for row in rows:
            buffer.insert(row)
        records.append(rows)
    return records


def test_actual_batches_match_unsplit_reference_and_preserve_order(make_buffer):
    buffers = [make_buffer(1), make_buffer(2)]
    populate(buffers)
    procs = [processor(buffers, make_buffer(), rank, 2) for rank in range(2)]
    full = processor(buffers, make_buffer(), 0, 1)
    for step in (0, 7, 100):
        expected, actor_expected, _ = full.next_batch(jax.random.PRNGKey(9), update_step=step)
        parts = [p.next_batch(jax.random.PRNGKey(9), update_step=step) for p in procs]
        combined = jax.tree.map(lambda a, b: np.concatenate((a, b), axis=1), parts[0][0], parts[1][0])
        actor = jax.tree.map(lambda a, b: np.concatenate((a, b), axis=0), parts[0][1], parts[1][1])
        for actual, reference in zip(jax.tree.leaves((combined, actor)), jax.tree.leaves((expected, actor_expected))):
            np.testing.assert_array_equal(actual, reference)
        plan = sampler(world_size=1).sample([118, 118], update_step=step, num_batches=20)
        np.testing.assert_array_equal(combined['state'], plan)
        assert np.all(np.asarray(actor['state'])[:, 0] == 1)  # only successful robot
        assert np.all(np.asarray(actor['state'])[:, 1] < 118)  # final n-step starts excluded
    with pytest.raises(ValueError, match='update step'):
        procs[0].next_batch(jax.random.PRNGKey(9))


@pytest.mark.parametrize("num_robot", [1, 2])
def test_replay_restore_reproduces_sampling_and_data(tmp_path, make_buffer, num_robot):
    buffers = [make_buffer(i + 1) for i in range(num_robot)]
    records = populate(buffers)
    for robot, rows in enumerate(records):
        save_replay_buffer_batch(tmp_path / f'robot-{robot}', rows, start_step=1 + 120 * robot)
    before = processor(buffers, make_buffer(), 0, 2).next_batch(jax.random.PRNGKey(4), update_step=13)
    restored = [make_buffer(999 - i) for i in range(num_robot)]
    for robot, buffer in enumerate(restored):
        restore_replay_buffer(tmp_path / f'robot-{robot}', buffer, up_to_step=240)
        buffer.restore_success_marks()
    after = processor(restored, make_buffer(), 0, 2).next_batch(jax.random.PRNGKey(4), update_step=13)
    for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after)):
        np.testing.assert_array_equal(a, b)


def test_actor_with_no_successes_returns_none(make_buffer):
    buffers = [make_buffer(), make_buffer()]
    for i in range(80):
        buffers[0].insert(transition(0, i, i == 79, success=False))
    batch, actor, _ = processor(buffers, make_buffer(), 0, 2).next_batch(
        jax.random.PRNGKey(0), update_step=0)
    assert batch['state'].shape == (20, 32, 2)
    assert actor is None


def test_single_robot_distributed_batches(make_buffer, monkeypatch):
    from types import SimpleNamespace
    from expo_ft.distributed.learner_group import initialize_learner
    monkeypatch.setenv('EXPO_PROCESS_COUNT', '2')
    monkeypatch.setenv('EXPO_PROCESS_ID', '0')
    monkeypatch.setenv('EXPO_COORDINATOR', 'localhost:29451')
    calls = []
    monkeypatch.setattr(jax.distributed, 'initialize', lambda **kw: calls.append(kw))
    initialize_learner(SimpleNamespace(split_role='learner', fsdp_devices=1,
                                      offline_ratio=0, num_robot=1))
    assert calls[0]['num_processes'] == 2
    buffers = [make_buffer()]
    for i in range(120):
        buffers[0].insert(transition(0, i, i in (59, 119), success=True))
    full = processor(buffers, make_buffer(), 0, 1).next_batch(jax.random.PRNGKey(4), update_step=7)
    parts = [processor(buffers, make_buffer(), rank, 2).next_batch(
        jax.random.PRNGKey(4), update_step=7) for rank in range(2)]
    for item, axis in [(0, 1), (1, 0)]:
        combined = jax.tree.map(lambda a, b: np.concatenate((a, b), axis=axis),
                                parts[0][item], parts[1][item])
        for actual, expected in zip(jax.tree.leaves(combined), jax.tree.leaves(full[item])):
            np.testing.assert_array_equal(actual, expected)
