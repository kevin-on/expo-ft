"""Freeze the baseline and current update sources for the A40 comparison job."""
import argparse
from pathlib import Path
import shutil
import subprocess

BASE = 'f92c32a0d1eae3aca2282ff1787228c60b55a9be'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--base', default=BASE)
    parser.add_argument('--replicate-update-rng', action='store_true',
                        help='Apply the common multi-device RNG compatibility fix to baseline')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    baseline = {
        'baseline_expo_ft.py': 'expo_ft/agents/alg/expo_ft.py',
        'baseline_temperature.py': 'expo_ft/networks/temperature.py',
    }
    # Resolve everything before creating a snapshot; never overwrite one.
    commit = subprocess.check_output(['git', 'rev-parse', args.base], cwd=repo)
    originals = {name: subprocess.check_output(['git', 'show', f'{args.base}:{path}'], cwd=repo)
                 for name, path in baseline.items()}
    if args.replicate_update_rng:
        marker = b'    def update(self, agent, batch: DatasetDict, utd_ratio: int, actor_batch: DatasetDict = None):\n'
        repair = (b'        from expo_ft.agents.alg.sharding_utils import place_update_rng\n'
                  b'        self, agent = place_update_rng(self, agent)\n')
        source = originals['baseline_expo_ft.py']
        assert source.count(marker) == 1
        assert marker + repair in (repo / baseline['baseline_expo_ft.py']).read_bytes()
        originals['baseline_expo_ft.py'] = source.replace(marker, marker + repair)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'compatibility-patches.txt').write_text(
        'baseline and fixed: RNG-only placement on training mesh\n'
        if args.replicate_update_rng else 'baseline: unmodified\n')
    for name, contents in originals.items():
        (args.output / name).write_bytes(contents)
    for dest, source in {
        'fixed_expo_ft.py': 'expo_ft/agents/alg/expo_ft.py',
        'fixed_temperature.py': 'expo_ft/networks/temperature.py',
        'learner_smoke.py': 'tests/gpu/learner_smoke.py',
        'compare_profile.py': 'tests/gpu/compare_profile.py',
        'compare_lowering.py': 'tests/gpu/compare_lowering.py',
        'run_profile.sbatch': 'profiling/run_profile.sbatch',
    }.items():
        shutil.copyfile(repo / source, args.output / dest)
    (args.output / 'base-commit.txt').write_bytes(commit)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
