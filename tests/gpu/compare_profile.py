"""Compare identical-input accelerator runs and retain diagnostics on failure."""
import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np


def events(root, name):
    return [row for line in (root / 'train-events.jsonl').read_text().splitlines()
            if (row := json.loads(line))['event'] == name]


def compare_arrays(before, after, rtol, atol):
    assert before.shape == after.shape and before.dtype == after.dtype
    delta = after - before
    return dict(
        exact=bool(np.array_equal(before, after)),
        allclose=bool(np.allclose(after, before, rtol=rtol, atol=atol)),
        count=int(before.size),
        outside_tolerance=int(np.count_nonzero(np.abs(delta) > atol + rtol * np.abs(before))),
        max_abs=float(np.max(np.abs(delta))),
        delta_sq=float(np.sum(np.square(delta, dtype=np.float64))),
        reference_sq=float(np.sum(np.square(before, dtype=np.float64))))


def warm_mean(updates, cache_events):
    """Exclude startup calls through the last JIT executable-cache expansion."""
    assert len(updates) == len(cache_events) and updates
    sizes = [row['update_cache_size'] for row in cache_events]
    last_change = max(i for i in range(len(sizes)) if i == 0 or sizes[i] != sizes[i-1])
    warm = [row['wall_seconds'] for row in updates[last_change+1:]]
    return statistics.mean(warm) if warm else None


def compare(baseline, fixed, require_one_compile=True):
    runtime_before, runtime_after = [events(root, 'runtime')[0] for root in (baseline, fixed)]
    for key in ('backend', 'devices', 'jax', 'batch_size', 'utd_ratio', 'updates', 'fsdp_devices'):
        assert runtime_before.get(key) == runtime_after.get(key), f'Runtime mismatch: {key}'
    before, after = [events(root, 'update_passed') for root in (baseline, fixed)]
    assert len(before) == len(after) >= 6
    rtol, atol = 1e-4, 1e-5
    inputs_before, inputs_after = [events(root, 'comparison_input') for root in (baseline, fixed)]
    inputs_match = len(inputs_before) == len(inputs_after) == len(before)
    if inputs_match:
        inputs_match = all(
            all(b[key] == a[key] for key in ('update', 'batch_hash', 'actor_batch_hash', 'learner_rng'))
            for b, a in zip(inputs_before, inputs_after))
        inputs_match &= inputs_before[0]['parameter_hashes'] == inputs_after[0]['parameter_hashes']
    metric_differences = []
    max_metric_error = 0.0
    for b, a in zip(before, after):
        assert b['metrics'].keys() == a['metrics'].keys()
        for key, expected in b['metrics'].items():
            actual = a['metrics'][key]
            error = abs(actual - expected)
            max_metric_error = max(max_metric_error, error)
            if not np.isclose(actual, expected, rtol=rtol, atol=atol):
                metric_differences.append(dict(update=a['update'], metric=key,
                    baseline=expected, fixed=actual, abs_error=error,
                    relative_error=error / max(abs(expected), 1e-12)))
    groups = {}
    worst_leaves = []
    with np.load(baseline / 'final-trainable.npz') as b, np.load(fixed / 'final-trainable.npz') as a:
        assert a.files == b.files
        for key in a.files:
            stats = compare_arrays(b[key], a[key], rtol, atol)
            group = key.split(']')[0].strip("['")
            total = groups.setdefault(group, dict(exact=True, allclose=True, count=0,
                outside_tolerance=0, max_abs=0.0, delta_sq=0.0, reference_sq=0.0))
            for flag in ('exact', 'allclose'):
                total[flag] &= stats[flag]
            for number in ('count', 'outside_tolerance', 'delta_sq', 'reference_sq'):
                total[number] += stats[number]
            total['max_abs'] = max(total['max_abs'], stats['max_abs'])
            worst_leaves.append(dict(path=key, max_abs=stats['max_abs'],
                                     outside_tolerance=stats['outside_tolerance']))
    for total in groups.values():
        total['relative_l2'] = (total['delta_sq'] / max(total['reference_sq'], 1e-30)) ** 0.5
    action = compare_arrays(np.load(baseline / 'final-action.npy'),
                            np.load(fixed / 'final-action.npy'), rtol, atol)
    action['relative_l2'] = (action['delta_sq'] / max(action['reference_sq'], 1e-30)) ** 0.5
    base_cache, fixed_cache = [events(root, 'jit_call_done') for root in (baseline, fixed)]
    no_recompile = [r['update_cache_size'] for r in fixed_cache] == [1] * len(after)
    numerics_match = not metric_differences and all(g['allclose'] for g in groups.values()) and action['allclose']
    return dict(
        passed=bool(inputs_match and (no_recompile or not require_one_compile) and numerics_match),
        identical_inputs=bool(inputs_match), no_recompilation=bool(no_recompile),
        strict_numerical_match=bool(numerics_match),
        batch_size=runtime_before['batch_size'], utd_ratio=runtime_before['utd_ratio'],
        devices=len(runtime_before['devices']), fsdp_devices=runtime_before['fsdp_devices'],
        backend=runtime_before.get('backend', 'gpu'),
        baseline_seconds=[r['wall_seconds'] for r in before],
        fixed_seconds=[r['wall_seconds'] for r in after],
        baseline_cache_sizes=[r['update_cache_size'] for r in base_cache],
        fixed_cache_sizes=[r['update_cache_size'] for r in fixed_cache],
        baseline_warm_mean_seconds=warm_mean(before, base_cache),
        fixed_warm_mean_seconds=warm_mean(after, fixed_cache),
        max_absolute_metric_error=max_metric_error, metric_differences=metric_differences,
        parameter_groups=groups, action=action,
        worst_parameter_leaves=sorted(worst_leaves, key=lambda r:r['max_abs'], reverse=True)[:10],
        rtol=rtol, atol=atol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('fixed', type=Path)
    parser.add_argument('--control', type=Path)
    args = parser.parse_args()
    result = compare(args.baseline, args.fixed)
    if args.control is not None:
        result['baseline_repeat'] = compare(args.baseline, args.control, require_one_compile=False)
        result['passed'] = result['passed'] and result['baseline_repeat']['passed']
    print(json.dumps(result, indent=2), flush=True)
    if not result['passed']:
        sys.exit('Profile comparison failed; inspect the JSON diagnostics above.')


if __name__ == '__main__':
    main()
