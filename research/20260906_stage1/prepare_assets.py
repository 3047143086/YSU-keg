"""Download public input assets into the isolated stage-one experiment directory."""

import concurrent.futures
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE = Path('/data/sunzixuan/rave_20260905_inputs/baby')
WEIGHT_SHA = '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
WEIGHT_URL = 'https://openaipublic.azureedge.net/clip/models/' + WEIGHT_SHA + '/ViT-B-16.pt'


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def download(url, path, limit, expected_hash=None):
    if path.exists():
        if expected_hash and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError('Existing asset has wrong checksum: ' + str(path))
        return
    partial = path.with_suffix(path.suffix + '.partial')
    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'multimodal-research/1.0'})
            checksum = hashlib.sha256()
            count = 0
            with urllib.request.urlopen(request, timeout=40) as response, partial.open('wb') as output:
                expected_size = int(response.headers.get('Content-Length', '0'))
                if expected_size > limit:
                    raise RuntimeError('Asset exceeds size limit')
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    checksum.update(chunk)
                    count += len(chunk)
                    if count > limit:
                        raise RuntimeError('Asset exceeds size limit')
                if expected_size and expected_size != count:
                    raise RuntimeError('Truncated asset')
            if expected_hash and checksum.hexdigest() != expected_hash:
                raise RuntimeError('Downloaded asset checksum mismatch')
            partial.replace(path)
            return
        except Exception as exc:
            last_error = exc
            print('download_retry', path.name, attempt + 1, str(exc), flush=True)
    raise RuntimeError('Asset download failed: ' + str(last_error))


def prepare_encoder():
    vendor = ROOT / 'vendor'
    vendor.mkdir(exist_ok=True)
    command = [sys.executable, '-m', 'pip', 'install', '--target', str(vendor), '--no-deps',
               'ftfy==6.1.3', 'regex==2023.12.25', 'tqdm==4.66.5', 'wcwidth==0.2.13']
    subprocess.run(command, check=True, timeout=600)
    request = urllib.request.Request('https://api.github.com/repos/openai/CLIP/commits/main',
                                     headers={'User-Agent': 'multimodal-research/1.0'})
    with urllib.request.urlopen(request, timeout=30) as response:
        revision = json.load(response)['sha']
    clip_dir = vendor / 'clip'
    clip_dir.mkdir(exist_ok=True)
    hashes = {}
    for name in ('__init__.py', 'clip.py', 'model.py', 'simple_tokenizer.py', 'bpe_simple_vocab_16e6.txt.gz'):
        path = clip_dir / name
        download('https://raw.githubusercontent.com/openai/CLIP/' + revision + '/clip/' + name,
                 path, 20 * 1024 * 1024)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    download('https://raw.githubusercontent.com/openai/CLIP/' + revision + '/LICENSE',
             vendor / 'CLIP_LICENSE', 1024 * 1024)
    weights = ROOT / 'weights'
    weights.mkdir(exist_ok=True)
    print('encoder_weights_downloading', flush=True)
    download(WEIGHT_URL, weights / 'ViT-B-16.pt', 500 * 1024 * 1024, WEIGHT_SHA)
    manifest = {'encoder': 'CLIP ViT-B/16', 'revision': revision, 'source_hashes': hashes,
                'weight_url': WEIGHT_URL, 'weight_sha256': WEIGHT_SHA, 'frozen': True}
    atomic_json(ROOT / 'encoder_manifest.json', manifest)
    print('encoder_ready', flush=True)
    return manifest


def fetch_image(row):
    item_id = row['item_id']
    destination = ROOT / 'images' / ('%05d.image' % item_id)
    result = {'item_id': item_id, 'asin': row['asin'], 'path': str(destination)}
    if not row.get('imUrl'):
        return dict(result, available=False, error='missing_url')
    url = urllib.parse.urlsplit(row['imUrl'])._replace(scheme='https').geturl()
    for attempt in range(2):
        try:
            if destination.exists():
                payload = destination.read_bytes()
            else:
                with urllib.request.urlopen(url, timeout=15) as response:
                    payload = response.read(8 * 1024 * 1024 + 1)
            if len(payload) > 8 * 1024 * 1024:
                raise RuntimeError('Image too large')
            image = Image.open(io.BytesIO(payload))
            size = list(image.size)
            image.verify()
            if not destination.exists():
                with destination.open('xb') as output:
                    output.write(payload)
            return dict(result, available=True, size=size,
                        sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
        except Exception as exc:
            result['error'] = str(exc)
    return dict(result, available=False)


def prepare_images():
    (ROOT / 'images').mkdir(exist_ok=True)
    rows = [json.loads(line) for line in (SOURCE / 'items.jsonl').read_text().splitlines()]
    if [row['item_id'] for row in rows] != list(range(7050)):
        raise RuntimeError('Recovered item mapping is not the audited full Baby catalog')
    shutil.copy2(SOURCE / 'items.jsonl', ROOT / 'items.jsonl')
    results = []
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_image, row) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            if len(results) % 250 == 0:
                progress = {'finished': len(results), 'total': len(rows),
                            'available': sum(row['available'] for row in results),
                            'elapsed_seconds': time.monotonic() - start}
                atomic_json(ROOT / 'download_progress.json', progress)
                print('image_progress', json.dumps(progress), flush=True)
    results.sort(key=lambda row: row['item_id'])
    atomic_json(ROOT / 'image_manifest.json', results)
    count = sum(row['available'] for row in results)
    if count / len(rows) < 0.95:
        raise RuntimeError('Less than 95% image coverage; inspect download failures before training')
    print('images_ready', count, len(rows), flush=True)
    return count


def main():
    started = time.monotonic()
    # Network preparation does not reserve a GPU or alter the base environment.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        images = executor.submit(prepare_images)
        encoder = executor.submit(prepare_encoder)
        image_count = images.result()
        encoder.result()
    atomic_json(ROOT / 'assets_ready.json', {'image_count': image_count,
                'elapsed_seconds': time.monotonic() - started})
    print('ASSETS_READY', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        atomic_json(ROOT / 'assets_failed.json', {'error': str(exc)})
        raise
