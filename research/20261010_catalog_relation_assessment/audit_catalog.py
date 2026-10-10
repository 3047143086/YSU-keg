"""Read-only CPU audit of explicit Baby/Sports catalog relations.

The remote payload emits aggregates and cryptographic file hashes only. It does
not print metadata values, titles, reviews, validation labels, or test labels.
"""
from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/20260905"))
from remote_session import Session


REMOTE = r'''
import hashlib, json, math, os, random, re, time, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path('/data/sunzixuan')
INPUT_ROOT = BASE / 'rave_20260905_inputs'
MSCA_ROOT = BASE / 'MSCA-main'
DATASETS = ('baby', 'sports')
PAIR_CAP = 20000
MIN_FIT = 1000
MIN_HOLDOUT = 200
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '4'

def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def norm(value):
    if not isinstance(value, str):
        return None
    value = unicodedata.normalize('NFKC', value)
    value = re.sub(r'\s+', ' ', value.strip()).casefold()
    return value or None

def categories(value):
    """Return explicit normalized root-to-leaf paths, without title parsing."""
    if not isinstance(value, list):
        return ()
    paths = []
    # Amazon metadata normally stores categories as list[list[str]]. A flat
    # list is one explicit path and is retained rather than guessed.
    if value and all(isinstance(x, str) for x in value):
        value = [value]
    for path in value:
        if not isinstance(path, (list, tuple)):
            continue
        normalized = tuple(norm(x) for x in path)
        if normalized and all(normalized):
            paths.append(normalized)
    return tuple(sorted(set(paths)))

def item_key(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if re.fullmatch(r'[0-9]+', value):
            return int(value)
    return None

def inventory_candidates(dataset):
    found = []
    for root in (INPUT_ROOT, BASE / 'bridge_official' / 'data' / dataset,
                 BASE / 'MSCA-main' / 'data' / dataset,
                 BASE / 'PRAVI_remote' / 'data' / dataset):
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if path.is_file() and (path.suffix.lower() in ('.jsonl', '.json', '.item', '.csv', '.tsv')
                                   or 'item' in path.name.lower() or 'catalog' in path.name.lower()):
                found.append(path)
    return list({str(path): path for path in found}.values())

def locate_metadata(dataset):
    expected = INPUT_ROOT / dataset / 'items.jsonl'
    if expected.is_file():
        return expected
    names = [path for path in inventory_candidates(dataset)
             if path.name.lower() in (dataset + '.jsonl', 'items.jsonl', dataset + '.item', 'items.item')]
    return names[0] if names else None

def locate_inter(dataset):
    for root in (MSCA_ROOT / 'data' / dataset, BASE / 'bridge_official' / 'data' / dataset):
        path = root / (dataset + '.inter')
        if path.is_file():
            return path
    return None

def stable_item_id(value):
    return item_key(value)

def read_metadata(path):
    rows = 0
    fields = Counter()
    id_types = Counter()
    ids = {}
    malformed = 0
    duplicate_ids = 0
    structure = Counter()
    path_lengths = Counter()
    text_nonempty = Counter()
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                malformed += 1
                continue
            rows += 1
            if not isinstance(row, dict):
                malformed += 1
                continue
            fields.update(row.keys())
            raw_categories = row.get('categories')
            shape = type(raw_categories).__name__
            if isinstance(raw_categories, list):
                shape += ':' + ('flat_strings' if all(isinstance(x, str) for x in raw_categories)
                                else 'nested_lists' if all(isinstance(x, list) for x in raw_categories)
                                else 'mixed')
            structure[shape] += 1
            for field in ('title', 'description'):
                value = row.get(field)
                text_nonempty[field] += bool(value.strip()) if isinstance(value, str) else bool(value)
            raw = row.get('item_id', row.get('itemID', row.get('iid')))
            id_types[type(raw).__name__] += 1
            item_id = stable_item_id(raw)
            if item_id is None:
                continue
            if item_id in ids:
                duplicate_ids += 1
                continue
            category = categories(row.get('categories'))
            path_lengths.update(len(path) for path in category)
            ids[item_id] = {'brand': norm(row.get('brand')), 'categories': category, 'asin': row.get('asin')}
    return {'rows': rows, 'malformed': malformed, 'duplicate_ids': duplicate_ids,
            'fields': sorted(fields), 'id_types': dict(id_types), 'ids': ids,
            'category_schema_counts': dict(structure), 'taxonomy_path_length_counts': dict(path_lengths),
            'nonempty_text_counts': dict(text_nonempty),
            'sha256': sha(path), 'bytes': path.stat().st_size}

def read_train(path):
    import csv
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        fields = reader.fieldnames or []
        if 'x_label' not in fields or 'itemID' not in fields:
            raise ValueError('Original MSCA interaction schema needs x_label and itemID')
        label = 'x_label'
        train = set()
        degree = Counter()
        rows = 0
        item_types = Counter()
        for row in reader:
            rows += 1
            if row.get(label) != '0':
                continue
            raw = row.get('itemID', row.get('item_id'))
            item_types[type(raw).__name__] += 1
            try:
                item_id = int(raw)
            except (TypeError, ValueError):
                continue
            train.add(item_id)
            degree[item_id] += 1
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size,
            'fields': fields, 'physical_rows_scanned_for_split0_only': rows,
            'train_events': sum(degree.values()), 'train_item_degrees': degree,
            'item_types': dict(item_types), 'train_items': train}

def partition(item_id, dataset):
    # SHA256 hash of the canonical dataset/item endpoint; exactly deterministic.
    return 'holdout' if int(hashlib.sha256((dataset + ':' + str(item_id)).encode()).hexdigest(), 16) % 5 == 0 else 'fit'

def relation(a, b):
    brand_a, cat_a = a['brand'], set(a['categories'])
    brand_b, cat_b = b['brand'], set(b['categories'])
    same_brand = brand_a == brand_b
    same_category = cat_a == cat_b
    if same_brand and same_category:
        return None  # excluded same-brand/same-category
    if same_brand and not same_category:
        return 'same-brand_diff-category'
    if not same_brand and same_category:
        return 'same-category_diff-brand'
    return 'neither'

def pair_summary(items, metadata, dataset):
    usable = sorted(item for item in items if item in metadata and metadata[item]['brand']
                    and metadata[item]['categories'])
    fit = [item for item in usable if partition(item, dataset) == 'fit']
    held = [item for item in usable if partition(item, dataset) == 'holdout']
    result = {'usable_items': len(usable), 'fit_items': len(fit), 'holdout_items': len(held),
              'fit_fraction': len(fit) / max(len(usable), 1),
              'holdout_fraction': len(held) / max(len(usable), 1), 'relations': {}, 'partition_distributions': {}}
    for label, endpoints in (('fitting', fit), ('heldout', held)):
        brands = Counter(metadata[item]['brand'] for item in endpoints)
        signatures = Counter(metadata[item]['categories'] for item in endpoints)
        result['partition_distributions'][label] = {'known_items': len(endpoints),
            'brand_count': len(brands), 'brand_frequency_histogram': dict(Counter(brands.values())),
            'category_signature_count': len(signatures),
            'category_frequency_histogram': dict(Counter(signatures.values()))}
        counts = Counter()
        samples = defaultdict(list)
        for kind in ('same-brand_diff-category', 'same-category_diff-brand', 'neither'):
            used = set()
            for pos, left in enumerate(endpoints):
                if left in used:
                    continue
                for right in endpoints[pos + 1:]:
                    if right in used or relation(metadata[left], metadata[right]) != kind:
                        continue
                    counts[kind] += 1
                    used.update((left, right))
                    # Store only deterministic endpoint digests, never raw IDs.
                    h = hashlib.sha256((dataset + ':' + str(left)).encode()).hexdigest()
                    k = hashlib.sha256((dataset + ':' + str(right)).encode()).hexdigest()
                    samples[kind].append([h, k])
                    break
                if counts[kind] >= PAIR_CAP:
                    break
        for kind in ('same-brand_diff-category', 'same-category_diff-brand', 'neither'):
            result['relations'].setdefault(kind, {})[label] = {
                'unique_matching_pair_count': int(counts[kind]),
                'available_count_semantics': 'deterministic matching under one-partner-per-item constraint; not total combinatorial pairs',
                'stored_count': len(samples[kind]),
                'meets_1000_fit': bool(label != 'fitting' or counts[kind] >= MIN_FIT),
                'meets_200_holdout': bool(label != 'heldout' or counts[kind] >= MIN_HOLDOUT),
                'endpoint_sha256_digest': hashlib.sha256(json.dumps(samples[kind], separators=(',', ':')).encode()).hexdigest(),
            }
    return result

def null_label_summary(items, metadata, dataset, degrees):
    result = {'seed': 2026, 'degree_strata': 'floor(log2(split0 item event degree))',
              'permutation_scope': 'within each item-hash partition and degree bin; joint brand/category labels',
              'partitions': {}}
    for name in ('fit', 'holdout'):
        endpoints = sorted(item for item in items if partition(item, dataset) == name
                           and item in metadata and metadata[item]['brand'] and metadata[item]['categories'])
        strata = defaultdict(list)
        for item in endpoints:
            strata[int(math.log2(degrees[item]))].append(item)
        rng = random.Random('catalog_null:2026:' + dataset + ':' + name)
        changed = brand_changed = category_changed = 0
        for rows in strata.values():
            donors = rows.copy()
            rng.shuffle(donors)
            for owner, donor in zip(rows, donors):
                a, b = metadata[owner], metadata[donor]
                brand_changed += a['brand'] != b['brand']
                category_changed += a['categories'] != b['categories']
                changed += (a['brand'], a['categories']) != (b['brand'], b['categories'])
        rate = changed / max(len(endpoints), 1)
        result['partitions'][name] = {'known_items': len(endpoints), 'joint_label_changed': changed,
            'joint_label_changed_rate': rate, 'brand_changed_rate': brand_changed / max(len(endpoints), 1),
            'category_changed_rate': category_changed / max(len(endpoints), 1),
            'degree_bin_counts': {str(key): len(value) for key, value in sorted(strata.items())},
            'passes_50pct_joint_changed': rate >= .50}
    result['passes_both_partitions'] = all(row['passes_50pct_joint_changed'] for row in result['partitions'].values())
    return result

def audit_dataset(dataset):
    metadata_path = locate_metadata(dataset)
    inter_path = locate_inter(dataset)
    row = {'dataset': dataset, 'metadata_candidates': [{'path': str(p), 'bytes': p.stat().st_size,
            'sha256': sha(p)} for p in inventory_candidates(dataset)],
           'metadata_path': str(metadata_path) if metadata_path else None,
           'inter_path': str(inter_path) if inter_path else None}
    if metadata_path is None or inter_path is None:
        row['status'] = 'blocked_missing_metadata_or_inter'
        return row
    metadata = read_metadata(metadata_path)
    inter = read_train(inter_path)
    mapping_path = metadata_path.parent / 'item_id_to_asin.json'
    mapping_audit_path = metadata_path.parent / 'mapping_audit.json'
    join = {'mapping_path': str(mapping_path), 'mapping_exists': mapping_path.is_file(),
            'mapping_audit_path': str(mapping_audit_path), 'mapping_audit_exists': mapping_audit_path.is_file(),
            'trusted': False}
    if mapping_path.is_file():
        mapping = json.loads(mapping_path.read_text())
        join['mapping_sha256'] = sha(mapping_path)
        join['mapping_rows'] = len(mapping)
        join['mapping_unique_asins'] = len(set(mapping.values()))
        join['metadata_asin_matches'] = sum(mapping.get(str(item)) == row['asin'] for item, row in metadata['ids'].items())
        join['metadata_asin_mismatches'] = len(metadata['ids']) - join['metadata_asin_matches']
    if mapping_audit_path.is_file():
        audit = json.loads(mapping_audit_path.read_text())
        join['mapping_audit_sha256'] = sha(mapping_audit_path)
        join['mapping_method'] = audit.get('identity_method')
        join['mapped_items'] = audit.get('mapped_items')
        join['unresolved_items_count'] = len(audit.get('unresolved_items', []))
    original_bridge_inter = BASE / 'bridge_official/data' / dataset / (dataset + '.inter')
    join['original_bridge_inter_path'] = str(original_bridge_inter)
    if original_bridge_inter.is_file():
        join['original_bridge_inter_sha256'] = sha(original_bridge_inter)
        join['original_bridge_same_bytes_as_msca'] = join['original_bridge_inter_sha256'] == inter['sha256']
    mapping_source_inter = BASE / 'expirement/LOBSTER/data' / dataset / (dataset + '.inter')
    join['mapping_source_inter_path'] = str(mapping_source_inter)
    if mapping_source_inter.is_file():
        join['mapping_source_inter_sha256'] = sha(mapping_source_inter)
        join['mapping_source_same_bytes_as_msca'] = join['mapping_source_inter_sha256'] == inter['sha256']
    join['trusted'] = bool(join.get('metadata_asin_mismatches') == 0 and join.get('unresolved_items_count') == 0
                           and join.get('mapping_method') == 'unique exact per-item multiset of (timestamp, rating), across full source records'
                           and join.get('original_bridge_same_bytes_as_msca') and join.get('mapping_source_same_bytes_as_msca')
                           and metadata['duplicate_ids'] == 0 and metadata['malformed'] == 0
                           and join.get('mapping_rows') == join.get('mapping_unique_asins')
                           and join.get('mapped_items') == join.get('mapping_rows'))
    train_items = inter['train_items']
    mapped = train_items & set(metadata['ids'])
    brand_known = sum(bool(metadata['ids'][item]['brand']) for item in mapped)
    category_known = sum(bool(metadata['ids'][item]['categories']) for item in mapped)
    both_known = sum(bool(metadata['ids'][item]['brand']) and bool(metadata['ids'][item]['categories']) for item in mapped)
    null_summary = null_label_summary(train_items, metadata['ids'], dataset, inter['train_item_degrees'])
    row.update({'status': 'audited', 'metadata': {key: value for key, value in metadata.items() if key != 'ids'},
        'inter': {key: value for key, value in inter.items() if key not in ('train_items', 'train_item_degrees')},
        'explicit_mapping_join': join,
        'train_unique_items': len(train_items), 'mapped_train_items': len(mapped),
        'mapping_coverage': len(mapped) / max(len(train_items), 1),
        'metadata_missing_train_items': len(train_items) - len(mapped),
        'brand_known': brand_known, 'brand_coverage': brand_known / max(len(train_items), 1),
        'category_known': category_known, 'category_coverage': category_known / max(len(train_items), 1),
        'both_known': both_known, 'both_coverage': both_known / max(len(train_items), 1),
        'metadata_null_brand_on_mapped': len(mapped) - brand_known,
        'metadata_null_category_on_mapped': len(mapped) - category_known,
        'relation_definition': {'normalization': 'NFKC, trim, whitespace collapse, casefold',
            'categories': 'explicit root-to-leaf sorted unique path sets; same-category iff exact signature equals; different iff unequal; partial overlap remains different',
            'brand': 'normalized explicit brand; unknown labels excluded',
            'excluded': 'same-brand and same-category pairs', 'pair_cap_per_class_partition': PAIR_CAP,
            'partner_limit': 'greedy ascending TRAIN itemID matching; each item at most one partner per relation class per partition; unordered deduplicated',
            'hash_partition': 'SHA256(dataset:item)%5==0 heldout; both endpoints heldout for diagnostic pairs'},
        'pairs': pair_summary(train_items, metadata['ids'], dataset),
        'pre_gate': bool(join['trusted'] and len(mapped) == len(train_items) and brand_known / max(len(train_items), 1) >= .70 and category_known / max(len(train_items), 1) >= .70 and both_known / max(len(train_items), 1) >= .70),
        'null_label_audit_completed': True, 'null_label_audit': null_summary,
        'overall_training_eligible': False})
    pair_gate = all(parts['fitting']['unique_matching_pair_count'] >= MIN_FIT
                    and parts['heldout']['unique_matching_pair_count'] >= MIN_HOLDOUT
                    for parts in row['pairs']['relations'].values())
    row['pair_gate'] = pair_gate
    row['overall_training_eligible'] = bool(row['pre_gate'] and pair_gate and null_summary['passes_both_partitions'])
    return row

record = {'checked_unix': time.time(), 'read_only': True, 'gpu_initialized': False,
          'reviews_read': False, 'validation_or_test_used': False, 'datasets': {}}
baby = audit_dataset('baby')
record['datasets']['baby'] = baby
if baby.get('pre_gate'):
    record['datasets']['sports'] = audit_dataset('sports')
else:
    record['datasets']['sports'] = {'status': 'not_audited_after_baby_pregate_failure',
        'limited_known_candidates': [str(p) for p in inventory_candidates('sports')], 'overall_training_eligible': False}
print(json.dumps(record, indent=2, allow_nan=False))
'''


def main():
    target = Path(__file__).resolve().parent
    target.mkdir(exist_ok=True)
    source_path = Path(__file__).resolve()
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    git = ['git', '-c', 'safe.directory=' + target.as_posix()]
    commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], cwd=str(target), text=True).strip()
    dirty = subprocess.check_output(git + ['status', '--porcelain', '--untracked-files=all'], cwd=str(target), text=True).strip()
    if dirty:
        raise RuntimeError('Commit diagnostic source/protocol before executing')
    compile(REMOTE, 'remote_catalog_audit', 'exec')
    session = Session("213")
    try:
        code, output = session.run_python(REMOTE, "/data/sunzixuan/miniconda3/envs/MMRec/bin/python", 900)
    finally:
        session.close()
    if code:
        raise RuntimeError(output[-4000:])
    record, _ = json.JSONDecoder().raw_decode(output.lstrip())
    post_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if post_sha != source_sha:
        raise RuntimeError('Diagnostic source changed while running')
    record['source_commit'] = commit
    record['source_sha256'] = source_sha
    record['post_source_sha256'] = post_sha
    (target / "catalog_relation_audit.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({'output': str(target / 'catalog_relation_audit.json'),
                      'datasets': {key: {'status': value.get('status'), 'pre_gate': value.get('pre_gate'),
                                         'mapping_coverage': value.get('mapping_coverage'),
                                         'brand_coverage': value.get('brand_coverage'),
                                         'category_coverage': value.get('category_coverage'),
                                         'pairs': value.get('pairs', {}).get('relations')}
                                    for key, value in record['datasets'].items()}}, indent=2))


if __name__ == '__main__':
    main()
