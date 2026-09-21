"""Compare single-GPU update timings and numerical results from two runs."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np


def events(root, name):
    return [row for line in (root / 'train-events.jsonl').read_text().splitlines()
            if (row := json.loads(line))['event'] == name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('fixed', type=Path)
    args = parser.parse_args()
    before, after = [events(root, 'update_passed') for root in (args.baseline, args.fixed)]
    assert len(before) == len(after) >= 6
    rtol, atol = 1e-4, 1e-5
    max_metric_error = 0.0
    for b, a in zip(before, after):
        assert b['metrics'].keys() == a['metrics'].keys()
        for key in b['metrics']:
            np.testing.assert_allclose(a['metrics'][key], b['metrics'][key],
                                       rtol=rtol, atol=atol, err_msg=f"update {a['update']}: {key}")
            max_metric_error = max(max_metric_error, abs(a['metrics'][key] - b['metrics'][key]))
    max_param_error = 0.0
    exact_parameters = True
    parameter_count = 0
    with np.load(args.baseline / 'final-trainable.npz') as b, np.load(args.fixed / 'final-trainable.npz') as a:
        assert a.files == b.files
        for key in a.files:
            av, bv = a[key], b[key]
            np.testing.assert_allclose(av, bv, rtol=rtol, atol=atol, err_msg=key)
            exact_parameters &= np.array_equal(av, bv)
            max_param_error = max(max_param_error, float(np.max(np.abs(av - bv))))
            parameter_count += av.size
    np.testing.assert_allclose(np.load(args.fixed / 'final-action.npy'),
                               np.load(args.baseline / 'final-action.npy'), rtol=rtol, atol=atol)
    base_cache, fixed_cache = [events(root, 'jit_call_done') for root in (args.baseline, args.fixed)]
    assert [r['update_cache_size'] for r in fixed_cache] == [1] * len(after), fixed_cache
    assert len({r['update_cache_size'] for r in base_cache[3:]}) == 1, base_cache
    result = dict(
        passed=True, batch_size=2, utd_ratio=1, devices=1,
        baseline_seconds=[r['wall_seconds'] for r in before],
        fixed_seconds=[r['wall_seconds'] for r in after],
        baseline_cache_sizes=[r['update_cache_size'] for r in base_cache],
        fixed_cache_sizes=[r['update_cache_size'] for r in fixed_cache],
        baseline_warm_mean_seconds=statistics.mean(r['wall_seconds'] for r in before[3:]),
        fixed_warm_mean_seconds=statistics.mean(r['wall_seconds'] for r in after[1:]),
        compared_trainable_scalars=parameter_count,
        exact_final_parameters=bool(exact_parameters),
        max_absolute_metric_error=max_metric_error,
        max_absolute_parameter_error=max_param_error, rtol=rtol, atol=atol)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
