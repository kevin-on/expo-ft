"""Verify the regional copy against the original GPU checkpoint manifest."""
import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    stage = Path(sys.argv[1])
    cache = stage / 'openpi-cache'
    entries = json.loads((cache / 'manifest.json').read_text())
    for item in entries:
        path = cache / item['bucket'] / item['name']
        assert path.stat().st_size == int(item['size']), path
        assert sha256(path) == item['sha256'], path
    demo = stage / 'demo/0/traj.hdf5'
    assert sha256(demo) == '5887eb33238b5b770fd05b8e71c3fc16be181c63dfbea8a7651ef0cbefe0952a'
    print(json.dumps(dict(checkpoint_files=len(entries), checkpoint_bytes=sum(int(i['size']) for i in entries),
                          demo_bytes=demo.stat().st_size, sha256='passed')))


if __name__ == '__main__':
    main()
