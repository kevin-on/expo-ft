"""Download the public pi05 base params and tokenizer, checking size and available MD5, recording SHA256 and GCS CRC32C."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


def request_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    params = {'prefix': 'checkpoints/pi05_base/params/', 'maxResults': 1000,
              'fields': 'items(name,size,md5Hash,crc32c,generation),nextPageToken'}
    objects = []
    while True:
        page = request_json('https://storage.googleapis.com/storage/v1/b/openpi-assets/o?' + urllib.parse.urlencode(params))
        objects.extend(dict(item, bucket='openpi-assets') for item in page.get('items', []))
        if 'nextPageToken' not in page:
            break
        params['pageToken'] = page['nextPageToken']
    tokenizer = request_json('https://storage.googleapis.com/storage/v1/b/big_vision/o/paligemma_tokenizer.model')
    objects.append(dict(tokenizer, bucket='big_vision'))

    def fetch(item):
        target = args.destination / item['bucket'] / item['name']
        target.parent.mkdir(parents=True, exist_ok=True)
        def valid(path):
            if not path.exists() or path.stat().st_size != int(item['size']):
                return False
            digest = hashlib.md5()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            return 'md5Hash' not in item or base64.b64encode(digest.digest()).decode() == item['md5Hash']
        if not valid(target):
            temporary = target.with_name(target.name + '.partial')
            url = ('https://storage.googleapis.com/download/storage/v1/b/' + item['bucket'] + '/o/'
                   + urllib.parse.quote(item['name'], safe='') + '?alt=media&generation=' + item['generation'])
            if not valid(temporary):
                with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
                    for chunk in iter(lambda: response.read(8 * 1024 * 1024), b''):
                        output.write(chunk)
            if not valid(temporary):
                raise RuntimeError('Checksum failed: ' + str(temporary))
            temporary.replace(target)
        digest = hashlib.sha256()
        with target.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        print(json.dumps({'downloaded': str(target), 'bytes': int(item['size']),
                          'md5_verified': 'md5Hash' in item}), flush=True)
        return dict({k: item[k] for k in ('bucket', 'name', 'size', 'crc32c', 'generation')},
                    sha256=digest.hexdigest(), md5_verified='md5Hash' in item)

    with ThreadPoolExecutor(max_workers=3) as pool:
        manifest = list(pool.map(fetch, objects))
    (args.destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Verified bytes:', sum(int(item['size']) for item in manifest), flush=True)


if __name__ == '__main__':
    main()
