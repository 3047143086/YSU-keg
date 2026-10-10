"""Bounded, read-only probe of public Amazon data and exact feature identities."""

import hashlib
import json
import urllib.request

import numpy as np


base = 'https://snap.stanford.edu/data/amazon/productGraph/'
urls = {
    'metadata': base + 'categoryFiles/meta_Baby.json.gz',
    'visual': base + 'image_features/categoryFiles/image_features_Baby.b',
}
report = {'purpose': 'public_input_recoverability_probe', 'files_written': False}
for name, url in urls.items():
    try:
        request = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(request, timeout=15) as response:
            report[name] = {'url': url, 'status': response.status,
                            'bytes': response.headers.get('Content-Length'),
                            'accept_ranges': response.headers.get('Accept-Ranges')}
    except Exception as exc:
        report[name] = {'url': url, 'error': str(exc)}
print(json.dumps(report), flush=True)

try:
    dtype = np.dtype([('asin', 'S10'), ('features', '<f4', (4096,))])
    byte_limit = dtype.itemsize * 256
    request = urllib.request.Request(urls['visual'], headers={'Range': 'bytes=0-%d' % (byte_limit - 1)})
    with urllib.request.urlopen(request, timeout=20) as response:
        chunk = response.read(byte_limit)
    records = np.frombuffer(chunk[:len(chunk) // dtype.itemsize * dtype.itemsize], dtype=dtype)
    matrix = np.load('/data/sunzixuan/expirement/LOBSTER/data/baby/image_feat.npy', mmap_mode='r')
    lookup = {}
    for item_id, vector in enumerate(matrix):
        fingerprint = hashlib.sha256(np.asarray(vector, dtype='<f4').tobytes()).digest()
        lookup.setdefault(fingerprint, []).append(item_id)
    matches = []
    for record in records:
        fingerprint = hashlib.sha256(record['features'].tobytes()).digest()
        if fingerprint in lookup:
            matches.append({'asin': record['asin'].decode('ascii'), 'item_ids': lookup[fingerprint]})
    print(json.dumps({'record_bytes': dtype.itemsize, 'records_read': len(records),
                      'exact_float32_matches': matches,
                      'note': 'A match establishes vector identity, not unique ASIN identity; duplicates require checks.'}), flush=True)
except Exception as exc:
    print(json.dumps({'visual_probe_error': str(exc)}), flush=True)
