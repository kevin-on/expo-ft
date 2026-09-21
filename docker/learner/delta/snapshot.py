"""Snapshot tracked learner sources; keep OpenPI and libraries inside the SIF."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def snapshot(repo, destination):
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=repo).decode().split('\0')
    hashes = {}
    for name in paths:
        if not name:
            continue
        path = Path(name)
        selected = (name.startswith(('expo_ft/', 'configs/', 'tests/gpu/')) or len(path.parts) == 1)
        if not selected or path.suffix not in ('.py', '.json', '.toml', '.lock', '.yaml', '.yml'):
            continue
        if name.startswith('expo_ft/agents/vla/openpi/'):
            continue
        source = repo / path
        if source.is_symlink():
            raise ValueError(f'Symlink is not a reproducible source file: {name}')
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    if 'tests/gpu/learner_smoke.py' not in hashes:
        raise ValueError('Missing tracked learner smoke harness')
    return {
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=repo, text=True).strip(),
        'tracked_changes': subprocess.check_output(['git', 'status', '--short', '--untracked-files=no'], cwd=repo, text=True),
        'source_sha256': hashes,
        'scope': 'Tracked learner working-tree files; local edits are included, untracked files are excluded. OpenPI comes from SIF.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    args.manifest.write_text(json.dumps(snapshot(args.repo.resolve(), args.destination), indent=2) + '\n')
