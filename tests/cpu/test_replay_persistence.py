"""Batch persistence, legacy compatibility and multi-robot checkpoint cursors."""
import importlib.util
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from conftest import transition
from expo_ft.data import replay_buffer as persistence


def save_round(root, start=1, lengths=(5, 7), *, legacy=False):
    for robot, length in enumerate(lengths):
        records = [transition(robot, i, i == length - 1, True, hil=i == 2)
                   for i in range(length)]
        directory = root / f"robot-{robot}"
        if legacy:
            for step, record in enumerate(records, start):
                persistence.save_replay_buffer_transition(directory, record, step=step)
        else:
            persistence.save_replay_buffer_batch(directory, records, start_step=start)
        start += length
    return start - 1


def restore(root, make_buffer, cutoff):
    buffers = [make_buffer(1), make_buffer(2)]
    for robot, buffer in enumerate(buffers):
        persistence.restore_replay_buffer(root / f"robot-{robot}", buffer,
                                          up_to_step=cutoff, skip_dummy_actions=False)
        buffer.restore_success_marks()
    return buffers


def equal_buffers(a, b):
    assert (len(a), a._insert_index, a._ep_step_counter, a._prev_is_hil) == (
        len(b), b._insert_index, b._ep_step_counter, b._prev_is_hil)
    for key in a.dataset_dict:
        np.testing.assert_array_equal(a.dataset_dict[key][:len(a)], b.dataset_dict[key][:len(b)], err_msg=key)
    a.rng = b.rng = jax.random.PRNGKey(10)
    if len(a) > 2:
        for mode in ({}, {"hil_only": True}, {"success_only": True}):
            for x, y in zip(jax.tree.leaves(a.sample_jax(8, **mode)),
                            jax.tree.leaves(b.sample_jax(8, **mode))):
                np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("cutoff", [0, 1, 5, 6, 12])
def test_batch_reader_equals_legacy_at_any_cutoff(tmp_path, make_buffer, cutoff):
    old, new = tmp_path / "old", tmp_path / "new"
    save_round(old, legacy=True)
    save_round(new)
    for a, b in zip(restore(old, make_buffer, cutoff), restore(new, make_buffer, cutoff)):
        equal_buffers(a, b)


def test_mixed_formats_and_repeated_resume(tmp_path, make_buffer):
    save_round(tmp_path, legacy=True)  # old checkpoint, 1..12
    save_round(tmp_path, 13)          # abandoned branch, 13..24
    abandoned = tmp_path / "abandoned-replay/first"
    persistence.prepare_robot_replay_resume(tmp_path, up_to_step=12, num_robot=2,
                                            abandoned_dir=abandoned)
    assert len(list(abandoned.glob("robot-*/*.pkl"))) == 2
    assert save_round(tmp_path, 13, lengths=(3, 4)) == 19
    persistence.prepare_robot_replay_resume(tmp_path, up_to_step=19, num_robot=2)
    assert [len(b) for b in restore(tmp_path, make_buffer, 19)] == [8, 11]
    persistence.prepare_robot_replay_resume(tmp_path, up_to_step=12, num_robot=2,
                                            abandoned_dir=tmp_path / "abandoned-replay/second")
    assert save_round(tmp_path, 13, lengths=(4, 6)) == 22
    persistence.prepare_robot_replay_resume(tmp_path, up_to_step=22, num_robot=2)
    assert [len(b) for b in restore(tmp_path, make_buffer, 22)] == [9, 13]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "overlap", "truncated",
                                     "count", "header", "version", "robot"])
def test_invalid_prefix_rejected_before_suffix_cleanup(tmp_path, failure):
    save_round(tmp_path)
    suffix = tmp_path / "robot-0/buffers/000000000013.pkl"
    suffix.write_bytes(b"not part of the checkpoint")
    path = next((tmp_path / "robot-0/buffers").glob("*-*.pkl"))
    if failure == "missing":
        path.unlink()
    elif failure == "duplicate":
        (tmp_path / "robot-1/buffers" / path.name).write_bytes(path.read_bytes())
    elif failure == "overlap":
        persistence.save_replay_buffer_transition(tmp_path / "robot-0", transition(0, 0), step=1)
    elif failure == "truncated":
        path.write_bytes(path.read_bytes()[:40])
    elif failure == "robot":
        (tmp_path / "robot-0").rename(tmp_path / "robot-2")
        suffix = tmp_path / "robot-2/buffers/000000000013.pkl"
    else:
        payload = pickle.loads(path.read_bytes())
        if failure == "count":
            payload["transitions"].pop()
        elif failure == "header":
            payload["start_step"] += 1
        else:
            payload["format"] = "unknown"
        path.write_bytes(pickle.dumps(payload))
    with pytest.raises((ValueError, EOFError, pickle.UnpicklingError)):
        persistence.prepare_robot_replay_resume(tmp_path, up_to_step=12, num_robot=2)
    assert suffix.exists()


def test_interior_robot_checkpoint_rejected_without_mutation(tmp_path):
    save_round(tmp_path)
    before = {str(p): p.read_bytes() for p in tmp_path.glob("robot-*/buffers/*.pkl")}
    with pytest.raises(ValueError, match="batch boundary"):
        persistence.prepare_robot_replay_resume(tmp_path, up_to_step=8, num_robot=2)
    assert {str(p): p.read_bytes() for p in tmp_path.glob("robot-*/buffers/*.pkl")} == before


def test_unpublished_tmp_and_newer_corrupt_suffix_are_not_restored(tmp_path, make_buffer):
    save_round(tmp_path)
    directory = tmp_path / "robot-0/buffers"
    (directory / "000000000001-000000000005.pkl.tmp").write_bytes(b"partial")
    suffix = directory / "000000000013-000000000020.pkl"
    suffix.write_bytes(b"partial newer round")
    persistence.prepare_robot_replay_resume(tmp_path, up_to_step=12, num_robot=2,
                                            abandoned_dir=tmp_path / "abandoned-replay/test")
    assert not suffix.exists()
    assert [len(b) for b in restore(tmp_path, make_buffer, 12)] == [5, 7]


@pytest.mark.parametrize("fail_at", ["file_fsync", "rename"])
def test_interrupted_write_keeps_old_published_batch(tmp_path, monkeypatch, make_buffer, fail_at):
    save_round(tmp_path)
    path = next((tmp_path / "robot-0/buffers").glob("*.pkl"))
    before = path.read_bytes()
    def fail(*args):
        raise OSError("injected save failure")
    monkeypatch.setattr(persistence.os, "fsync" if fail_at == "file_fsync" else "replace", fail)
    with pytest.raises(OSError, match="injected save failure"):
        persistence.save_replay_buffer_batch(tmp_path / "robot-0",
            [transition(9, i, i == 4, False) for i in range(5)], start_step=1)
    assert path.read_bytes() == before
    assert [len(b) for b in restore(tmp_path, make_buffer, 12)] == [5, 7]


def test_batch_fsyncs_before_and_after_rename(tmp_path, monkeypatch):
    directory = tmp_path / "robot-0"
    (directory / "buffers").mkdir(parents=True)
    events = []
    rename = persistence.os.replace
    monkeypatch.setattr(persistence.os, "fsync", lambda fd: events.append("fsync"))
    def replace(*args):
        events.append("rename")
        rename(*args)
    monkeypatch.setattr(persistence.os, "replace", replace)
    persistence.save_replay_buffer_batch(directory, [transition(0, 0, True, True)], start_step=1)
    assert events == ["fsync", "rename", "fsync"]


@pytest.mark.parametrize("skip_dummy", [False, True])
def test_max_transitions_and_dummy_filter_match_legacy(tmp_path, make_buffer, skip_dummy):
    records = [transition(0, i, i == 6, True, hil=i % 2 == 0) for i in range(7)]
    records[1]["actions"][:] = -1
    for step, record in enumerate(records, 1):
        persistence.save_replay_buffer_transition(tmp_path / "old", record, step=step)
    persistence.save_replay_buffer_batch(tmp_path / "new", records, start_step=1)
    buffers = [make_buffer(3), make_buffer(3)]
    for directory, buffer in zip(("old", "new"), buffers):
        persistence.restore_replay_buffer(tmp_path / directory, buffer, up_to_step=6,
                                          max_transitions=4, skip_dummy_actions=skip_dummy)
    assert all(len(b) == 4 for b in buffers)
    equal_buffers(*buffers)


def test_recorded_wan_fixture_accepts_legacy_and_batch_files(tmp_path):
    path = Path(__file__).resolve().parents[1] / "gpu/split_wan_smoke.py"
    spec = importlib.util.spec_from_file_location("wan_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    for layout in ("legacy", "batch"):
        step = 1
        for episode in range(2):
            for robot in range(2):
                records = [transition(robot, i, done=i == 2, success=True) for i in range(3)]
                for record in records:
                    record["observations"] = dict(cartesian_position=np.arange(6), gripper_position=1.,
                        **{key: np.full((2, 2, 3), 100 + episode, np.uint8) for key in
                           ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left")})
                directory = tmp_path / layout / "checkpoints" / f"robot-{robot}"
                if layout == "legacy":
                    for offset, record in enumerate(records):
                        persistence.save_replay_buffer_transition(directory, record, step=step + offset)
                else:
                    persistence.save_replay_buffer_batch(directory, records, start_step=step)
                step += len(records)
        fixture.prepare(SimpleNamespace(fixture=tmp_path / f"fixture-{layout}",
                                        recordings=tmp_path / layout, rounds=2))
    old, new = [json.loads((tmp_path / f"fixture-{layout}/manifest.json").read_text())
                for layout in ("legacy", "batch")]
    assert old["transitions"] == new["transitions"] == 12
    for old_episodes, new_episodes in zip(old["robots"], new["robots"]):
        for a, b in zip(old_episodes, new_episodes):
            assert (a["length"], a["success"], a["records_sha256"]) == (
                b["length"], b["success"], b["records_sha256"])
            assert len(a["sources"]) == 3 and len(b["sources"]) == 1
