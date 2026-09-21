"""CPU-only checks for TPU topology and reproducibility-control verdicts."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('profile_comparison', Path(__file__).with_name('compare_profile.py'))
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def make_run(path, *, loss=1.0, batch_size=4, cache_size=1):
    path.mkdir()
    rows = [dict(event='runtime', backend='tpu', devices=['TPU_0', 'TPU_1', 'TPU_2', 'TPU_3'],
                 jax='0.5.3', batch_size=batch_size, utd_ratio=1, updates=6, fsdp_devices=4)]
    for i in range(1, 7):
        rows += [dict(event='update_passed', update=i, wall_seconds=1.0, metrics={'loss': loss}),
                 dict(event='comparison_input', update=i, batch_hash=str(i), actor_batch_hash=str(i),
                      learner_rng=[0,i], parameter_hashes={'actor': 'same'}),
                 dict(event='jit_call_done', update=i, update_cache_size=cache_size, wall_seconds=1.0)]
    (path/'train-events.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    np.savez(path/'final-trainable.npz', **{"['actor']['weight']": np.array([1.0], dtype=np.float32)})
    np.save(path/'final-action.npy', np.ones((16,7), dtype=np.float32))
    return path


def test_four_device_runtime(tmp_path):
    result = comparison.compare(make_run(tmp_path/'baseline'), make_run(tmp_path/'fixed'))
    assert result['passed']
    assert (result['batch_size'], result['devices'], result['fsdp_devices']) == (4,4,4)


def test_runtime_mismatch_rejected(tmp_path):
    with pytest.raises(AssertionError, match='batch_size'):
        comparison.compare(make_run(tmp_path/'baseline'), make_run(tmp_path/'fixed', batch_size=8))


def test_recompile_rejected(tmp_path):
    result = comparison.compare(make_run(tmp_path/'baseline'), make_run(tmp_path/'fixed', cache_size=2))
    assert not result['passed']


def test_control_failure_fails_overall(tmp_path, monkeypatch, capsys):
    baseline = make_run(tmp_path/'baseline')
    fixed = make_run(tmp_path/'fixed')
    control = make_run(tmp_path/'control', loss=2.0)
    monkeypatch.setattr(sys, 'argv', ['compare', str(baseline), str(fixed), '--control', str(control)])
    with pytest.raises(SystemExit):
        comparison.main()
    result = json.loads(capsys.readouterr().out)
    assert not result['passed']
    assert not result['baseline_repeat']['passed']


def test_warm_mean_excludes_remaining_startup_recompile():
    updates = [{'wall_seconds': x} for x in (130., 131., 2., 2., 2., 2.)]
    cache = [{'update_cache_size': x} for x in (1, 2, 2, 2, 2, 2)]
    assert comparison.warm_mean(updates, cache) == 2.0
    assert comparison.warm_mean(updates[:2], cache[:2]) is None
