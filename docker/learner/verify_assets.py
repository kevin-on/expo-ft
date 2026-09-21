"""Validate a staged OpenPI cache against GCS CRC32C and the recorded SHA256."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import google_crc32c


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('cache',type=Path)
    args=parser.parse_args()
    manifest=json.loads((args.cache/'manifest.json').read_text())
    total=0
    for entry in manifest:
        path=args.cache/entry['bucket']/entry['name']
        crc=google_crc32c.Checksum()
        sha=hashlib.sha256()
        assert path.stat().st_size==int(entry['size']),path
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):
                crc.update(chunk)
                sha.update(chunk)
        assert base64.b64encode(crc.digest()).decode()==entry['crc32c'],f'GCS CRC32C mismatch: {path}'
        assert sha.hexdigest()==entry['sha256'],f'SHA256 mismatch: {path}'
        total+=int(entry['size'])
        print('Verified',entry['name'],flush=True)
    print(json.dumps({'files':len(manifest),'bytes':total,'crc32c':'passed','sha256':'passed'}))


if __name__=='__main__':
    main()
