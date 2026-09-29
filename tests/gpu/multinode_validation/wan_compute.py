"""Recorded-data WAN test using the two currently authorized allocations."""
import argparse
import json
import os
from pathlib import Path

import compute as runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('stage', 'run'), required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--processes', type=int, default=2)
    parser.add_argument('--coordinator', default='gh042:29451')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--warmup-episodes', type=int, default=1)
    parser.add_argument('--asset-id', default='expo_ft/pick_cube_balance_0923_mixed_20_seed3')
    args = parser.parse_args()
    args.utd_axis = False
    arm = os.uname().machine == 'aarch64'
    role = 'learner' if arm else 'inference'
    if not arm:
        runtime.JOB = '17637291'
        runtime.STAGE = Path('/tmp/expo-gh200x8-iliad-' + runtime.JOB)
        runtime.IMAGE = Path('/iliad/u/kevinon/artifacts/expo-ft/access-runtime/expo-ft-sft-jax053-20260922.sif')
        runtime.CHECKPOINT = Path('/iliad/u/kevinon/artifacts/expo-ft/sft-eval/balance0923-seed3-5k-20260923/mixed-020/4999')
        runtime.TOKENIZER = Path('/iliad/u/kevinon/artifacts/expo-ft/openpi-cache/big_vision/paligemma_tokenizer.model')
        runtime.FIXTURE = Path('/iliad/u/kevinon/outputs/expo-ft/split-validation/20260927-xxh3-batch-e2e/fixture')
        args.processes = 1
    if os.environ['SLURM_JOB_ID'] != runtime.JOB:
        raise RuntimeError('Unexpected allocation; recheck the authorized job before running')
    if args.phase == 'stage':
        runtime.stage(fabric=arm)
        return
    expanded = runtime.STAGE / ('wan-fixture-' + args.attempt)
    if not expanded.exists():
        expanded.mkdir()
        for path in (runtime.STAGE / 'fixture').rglob('*'):
            target = expanded / path.relative_to(runtime.STAGE / 'fixture')
            if path.is_dir():
                target.mkdir(exist_ok=True)
            elif path.name != 'manifest.json':
                os.link(path, target)
        manifest = json.loads((runtime.STAGE / 'fixture/manifest.json').read_text())
        manifest['robots'] = [rows[:args.warmup_episodes] +
            [dict(rows[-1]) for _ in range(args.rounds - args.warmup_episodes)] for rows in manifest['robots']]
        assert all(len(rows) == args.rounds for rows in manifest['robots'])
        assert all(row['length'] == 80 for rows in manifest['robots'] for row in rows[args.warmup_episodes:])
        manifest['rounds'] = args.rounds
        manifest['transitions'] = sum(row['length'] for rows in manifest['robots'] for row in rows)
        (expanded / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    mailbox = runtime.STAGE / ('mailbox-' + args.attempt)
    mailbox.mkdir(exist_ok=False)
    link = runtime.ROOT / 'link'
    command = ['python', '-u', 'tests/gpu/split_wan_smoke.py', '--role', 'node', '--node-role', role,
        '--fixture', '/wan-fixture', '--output', '/output', '--params', '/checkpoint/params',
        '--assets', '/checkpoint/assets', '--asset-id', args.asset_id,
        '--mailbox', '/mailbox', '--transport-config', '/link/transport-' + role + '.json',
        '--rounds', str(args.rounds), '--warmup-episodes', str(args.warmup_episodes),
        '--playback-hz', '10', '--performance', '--session', 'gh200x8-' + args.attempt]
    runtime.execute(args, command=command, output_name='wan-' + args.attempt,
        extra_bindings=[str(link) + ':/link:ro', str(expanded) + ':/wan-fixture:ro', str(mailbox) + ':/mailbox'],
        fabric=arm)


if __name__ == '__main__':
    main()
