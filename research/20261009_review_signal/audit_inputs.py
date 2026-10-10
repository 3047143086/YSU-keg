"""Audit TRAIN review alignment without printing review content or owner IDs."""
import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from mapping import (compute_holdout_pairs, file_sha256, iter_raw_reviews,
                     load_support_reviews, read_train_events)


INTER = Path('/data/sunzixuan/bridge_official/data/baby/baby.inter')
MAPPING = Path('/data/sunzixuan/rave_20260905_inputs/baby/item_id_to_asin.json')
REVIEWS = Path('/data/sunzixuan/rave_20260905_inputs/baby/reviews_Baby_5.json.gz')


def audit(inter_path=INTER, mapping_path=MAPPING, reviews_path=REVIEWS):
    started = time.monotonic()
    files = {name: {'path': str(path), 'bytes': Path(path).stat().st_size,
                    'sha256': file_sha256(path)}
             for name, path in (('interactions', inter_path), ('item_mapping', mapping_path),
                                 ('reviews', reviews_path))}
    with Path(inter_path).open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        inter_fields = reader.fieldnames
        split_counts = Counter(row['x_label'] for row in reader)
    item_mapping = json.loads(Path(mapping_path).read_text(encoding='utf8'))
    raw_first = next(iter_raw_reviews(reviews_path))
    raw_schema = {key: type(value).__name__ for key, value in raw_first.items()}
    del raw_first
    records, matching = load_support_reviews(inter_path, mapping_path, reviews_path,
                                             heldout_pairs=None, strict=False)
    train_events = read_train_events(inter_path, mapping_path)
    targets = sorted(compute_holdout_pairs(train_events))
    target_digest = hashlib.sha256(json.dumps(targets, separators=(',', ':')).encode('ascii')).hexdigest()
    by_item = Counter(record['itemID'] for record in records)
    donor_eligible = sum(by_item[record['itemID']] > 1 for record in records)
    matching['cross_user_same_item_donor_edges'] = donor_eligible
    matching['support_text_coverage'] = (matching['support_nonempty_review_text'] /
                                         max(matching['expected_support_edges'], 1))
    matching['strict_identity_passed'] = matching['matched_support_edges'] == matching['expected_support_edges']
    matching['same_item_donor_coverage'] = donor_eligible / max(len(records), 1)
    for info in files.values():
        if file_sha256(info['path']) != info['sha256']:
            raise RuntimeError('input changed during review audit')
    return {'status': 'completed', 'elapsed_seconds': time.monotonic() - started,
            'inputs': files, 'inter_fields': inter_fields, 'inter_rows': sum(split_counts.values()),
            'split_counts_for_inventory_only': dict(split_counts), 'raw_review_schema': raw_schema,
            'item_mapping_items': len(item_mapping), 'item_mapping_unique_asins': len(set(item_mapping.values())),
            'heldout_target_hash': target_digest, 'heldout_target_rule': 'dual-witness-target:v1:',
            'matching': matching, 'review_text_exported': False, 'reviewer_identity_exported': False,
            'validation_or_test_content_used': False, 'ratings_used_as_model_values': False,
            'training_or_encoding_started': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inter', type=Path, default=INTER)
    parser.add_argument('--mapping', type=Path, default=MAPPING)
    parser.add_argument('--reviews', type=Path, default=REVIEWS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.inter, args.mapping, args.reviews)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

