"""Verify an old SFT CLI recipe against its original OpenPI checkout before upgrading.

Run with the EXPO Python environment. No devices are initialized; original model
weights are never written. Without --write, print the verified record to stdout.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--reference-openpi', required=True, type=Path)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    from openpi.training import checkpoint_config as recipe, config
    path = args.checkpoint/'assets/config.json'
    original = json.loads(path.read_text())
    if original.get('version') == 2:
        recipe.restore(original)
        print('Already verified version 2', file=sys.stderr)
        return
    if original.get('version') != 1 or not original.get('openpi_commit'):
        raise ValueError('Migration requires recorded CLI arguments and original OpenPI commit')
    reference = args.reference_openpi.resolve()
    revision = subprocess.check_output(['git', '-C', str(reference), 'rev-parse', 'HEAD'], text=True).strip()
    recorded = original['openpi_commit']
    if not revision.startswith(recorded) or len(recorded) < 7:
        raise ValueError('Reference checkout does not match the recorded OpenPI revision')
    if subprocess.check_output(['git', '-C', str(reference), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Reference OpenPI has modified tracked files')
    # The comparison encoder belongs to this implementation, but the parser and
    # preset being fingerprinted are imported exclusively from the original source.
    code = '''import importlib.util,json,sys
from openpi.training import config
spec=importlib.util.spec_from_file_location("recipe_encoder",sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
args=json.loads(sys.argv[2]);sys.argv=["config"]+args
print(json.dumps(module.describe(config.cli())))
'''
    env = dict(os.environ, PYTHONPATH=str(reference/'src')+os.pathsep+str(reference/'packages/openpi-client/src'), JAX_PLATFORMS='cpu')
    result = subprocess.check_output([sys.executable, '-c', code, recipe.__file__, json.dumps(original['config_args'])],
                                     env=env, cwd=reference, text=True)
    previous = json.loads(result)
    current = config.cli(original['config_args'])
    if previous != recipe.describe(current):
        raise ValueError('Current parser/preset does not reproduce original settings; use matching source')
    record = recipe.make_record(current, original['config_args'])
    record['openpi_commit'] = revision
    record['migration'] = {'original': original, 'verified_against': revision}
    if args.write:
        backup = path.with_name('config.v1.json')
        with backup.open('x') as f: f.write(path.read_text())
        temporary = path.with_name('config.json.tmp')
        temporary.write_text(json.dumps(record, indent=2)+'\n'); temporary.replace(path)
    else:
        print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
