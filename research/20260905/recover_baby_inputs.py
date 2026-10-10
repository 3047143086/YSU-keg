"""Recover item identities from public records, never using reviews as features.

Mapping is verified by complete per-item timestamp/rating multisets. Ambiguous
identities are rejected, and no recommendation metric is computed here.
"""

import ast
import collections
import gzip
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path('/data/sunzixuan/rave_20260905_inputs/baby')
SOURCE = 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/'


def download(url, destination, limit=128 * 1024 * 1024):
    start = time.monotonic()
    checksum = hashlib.sha256()
    count = 0
    with urllib.request.urlopen(url, timeout=25) as response, destination.open('xb') as output:
        expected = int(response.headers.get('Content-Length', 0))
        if expected > limit:
            raise RuntimeError('Download exceeds size bound')
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            count += len(chunk)
            if count > limit or time.monotonic() - start > 180:
                raise RuntimeError('Download exceeded size or elapsed-time bound')
            output.write(chunk)
            checksum.update(chunk)
        if expected and expected != count:
            raise RuntimeError('Truncated download')
    return {'url': url, 'bytes': count, 'sha256': checksum.hexdigest()}


def records(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield ast.literal_eval(line)


def signature(entries):
    return tuple(sorted(entries))


def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    downloaded = {}
    reviews_file = ROOT / 'reviews_Baby_5.json.gz'
    downloaded['reviews_for_mapping_only'] = download(SOURCE + reviews_file.name, reviews_file)
    print('downloaded_review_records', downloaded['reviews_for_mapping_only']['bytes'], flush=True)
    raw_item_events = collections.defaultdict(list)
    raw_count = 0
    for row in records(reviews_file):
        raw_item_events[row['asin']].append((int(row['unixReviewTime']), float(row['overall'])))
        raw_count += 1
    signature_to_asins = collections.defaultdict(list)
    for asin, events in raw_item_events.items():
        signature_to_asins[signature(events)].append(asin)
    interactions = pd.read_csv('/data/sunzixuan/expirement/LOBSTER/data/baby/baby.inter', sep='\t')
    mapping = {}
    unresolved = []
    for item_id, group in interactions.groupby('itemID'):
        events = zip(group['timestamp'].astype(int), group['rating'].astype(float))
        matches = signature_to_asins.get(signature(events), [])
        if len(matches) == 1:
            mapping[int(item_id)] = matches[0]
        else:
            unresolved.append({'item_id': int(item_id), 'candidate_count': len(matches)})
    audit = {'purpose': 'identity_and_metadata_recovery_only',
             'test_metrics_computed': False, 'review_text_used_as_model_input': False,
             'identity_method': 'unique exact per-item multiset of (timestamp, rating), across full source records',
             'source_interactions': raw_count, 'local_interactions': len(interactions),
             'source_items': len(raw_item_events), 'local_items': int(interactions.itemID.nunique()),
             'mapped_items': len(mapping), 'unresolved_items': unresolved, 'downloads': downloaded}
    print(json.dumps(audit), flush=True)
    if unresolved or len(set(mapping.values())) != len(mapping):
        (ROOT / 'mapping_audit_incomplete.json').write_text(json.dumps(audit, indent=2))
        raise RuntimeError('Mapping is incomplete or not one-to-one; no training-ready data exported')
    if sorted(mapping) != list(range(len(mapping))):
        raise RuntimeError('Item IDs are not contiguous; inspect feature alignment before export')
    metadata_file = ROOT / 'meta_Baby.json.gz'
    downloaded['metadata'] = download(SOURCE + metadata_file.name, metadata_file)
    print('downloaded_metadata', downloaded['metadata']['bytes'], flush=True)
    selected_asins = set(mapping.values())
    metadata = {}
    for row in records(metadata_file):
        if row['asin'] in selected_asins:
            if row['asin'] in metadata:
                raise RuntimeError('Duplicate metadata ASIN')
            metadata[row['asin']] = {key: row.get(key) for key in
                                     ('title', 'description', 'brand', 'categories', 'imUrl')}
    exported = []
    for item_id, asin in sorted(mapping.items()):
        exported.append({'item_id': item_id, 'asin': asin, **metadata.get(asin, {})})
    with (ROOT / 'items.jsonl').open('x', encoding='utf-8') as output:
        for row in exported:
            output.write(json.dumps(row, ensure_ascii=True) + '\n')
    with (ROOT / 'item_id_to_asin.json').open('x') as output:
        json.dump(mapping, output, indent=2)
    audit['metadata_coverage'] = {
        'records': len(metadata), 'title': sum(bool(row.get('title')) for row in exported),
        'description': sum(bool(row.get('description')) for row in exported),
        'image_url': sum(bool(row.get('imUrl')) for row in exported),
    }
    # Sampling verifies access only; it does not establish full-catalog coverage.
    from PIL import Image
    samples = []
    for row in [entry for entry in exported if entry.get('imUrl')][:8]:
        url = urllib.parse.urlsplit(row['imUrl'])._replace(scheme='https').geturl()
        sample = {'item_id': row['item_id'], 'asin': row['asin'], 'url': url}
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                payload = response.read(8 * 1024 * 1024 + 1)
            if len(payload) > 8 * 1024 * 1024:
                raise RuntimeError('Image exceeds size bound')
            image = Image.open(io.BytesIO(payload))
            sample['size'] = list(image.size)
            sample['format'] = image.format
            image.verify()
            name = 'sample_%05d.%s' % (row['item_id'], sample['format'].lower())
            with (ROOT / name).open('xb') as output:
                output.write(payload)
            sample['bytes'] = len(payload)
        except Exception as exc:
            sample['error'] = str(exc)
        samples.append(sample)
    audit['image_access_sample'] = samples
    audit['sorted_asin_order_matches'] = [mapping[index] for index in range(len(mapping))] == sorted(mapping.values())
    (ROOT / 'mapping_audit.json').write_text(json.dumps(audit, indent=2))
    print('FINAL_AUDIT', json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
