"""Recover TRAIN review identities without using held-out review content."""
import ast
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path


TARGET_PREFIX = 'dual-witness-target:v1:'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def iter_raw_reviews(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = ast.literal_eval(line)
            if not isinstance(row, dict):
                raise ValueError('raw review record is not a dictionary')
            yield row


def _number(value, integer=False):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError('invalid identity numeric field')
    if not number.is_finite():
        raise ValueError('nonfinite identity numeric field')
    if integer:
        if number != number.to_integral_value():
            raise ValueError('nonintegral review timestamp')
        return int(number)
    return number.normalize()


def fingerprint(asin, timestamp, rating):
    if not isinstance(asin, str) or not asin:
        raise ValueError('missing ASIN identity')
    return (asin, _number(timestamp, integer=True), _number(rating))


def read_train_events(inter_path, item_mapping_path):
    mapping = json.loads(Path(item_mapping_path).read_text(encoding='utf-8'))
    item_mapping = {int(key): value for key, value in mapping.items()}
    if len(set(item_mapping.values())) != len(item_mapping):
        raise ValueError('itemID-to-ASIN mapping must be one-to-one')
    events, seen = [], set()
    with Path(inter_path).open('r', encoding='utf-8-sig', newline='') as stream:
        rows = csv.DictReader(stream, delimiter='\t')
        required = {'userID', 'itemID', 'rating', 'timestamp', 'x_label'}
        if not required.issubset(set(rows.fieldnames or [])):
            raise ValueError('interaction file lacks required identity columns')
        for row in rows:
            if row['x_label'] != '0':
                continue
            user, item = int(row['userID']), int(row['itemID'])
            if user < 0 or item < 0 or item not in item_mapping:
                raise ValueError('invalid TRAIN user or mapped item ID')
            if (user, item) in seen:
                raise ValueError('duplicate TRAIN user-item edge')
            seen.add((user, item))
            identity = fingerprint(item_mapping[item], row['timestamp'], row['rating'])
            events.append({'userID': user, 'itemID': item, 'timestamp': identity[1],
                           'identity': identity})
    return events


def compute_holdout_pairs(train_events):
    histories = defaultdict(list)
    for event in train_events:
        histories[int(event['userID'])].append(int(event['itemID']))
    result = set()
    for user, items in histories.items():
        if len(items) >= 4:
            item = min(items, key=lambda value: hashlib.sha256(
                (TARGET_PREFIX + str(user) + ':' + str(value)).encode('ascii')).digest())
            result.add((user, item))
    return result


def match_support_identities(support_events, raw_reviews):
    """Return exact raw row alignment; text and ratings never enter features.

    Owner candidates are identified only by fingerprints in support events.
    Every support user must have one candidate owner, and every support event
    must correspond to one raw row for that owner.  Ambiguity is retained.
    """
    support_events = list(support_events)
    support_by_user = defaultdict(list)
    keys = set()
    for event in support_events:
        support_by_user[int(event['userID'])].append(event)
        keys.add(event['identity'])
    inverted = defaultdict(set)
    owner_rows = defaultdict(list)
    raw_count, owner_fingerprints, owner_items = 0, Counter(), Counter()
    for raw_index, row in enumerate(raw_reviews):
        raw_count += 1
        identity = fingerprint(row['asin'], row['unixReviewTime'], row['overall'])
        owner = row.get('reviewerID')
        if not isinstance(owner, str) or not owner:
            raise ValueError('missing reviewer identity')
        owner_fingerprints[(owner, identity)] += 1
        owner_items[(owner, identity[0])] += 1
        if identity in keys:
            inverted[identity].add(owner)
            owner_rows[(owner, identity)].append(raw_index)
    alignments, unresolved_users, ambiguous_users = [], 0, 0
    duplicate_event_users, mapped_owners = 0, set()
    resolved_user_owners = {}
    for user, events in sorted(support_by_user.items()):
        candidates = None
        for event in events:
            owners = inverted.get(event['identity'], set())
            candidates = set(owners) if candidates is None else candidates.intersection(owners)
        if not candidates:
            unresolved_users += 1
            continue
        if len(candidates) != 1:
            ambiguous_users += 1
            continue
        owner = next(iter(candidates))
        if any(len(owner_rows[(owner, event['identity'])]) != 1 for event in events):
            duplicate_event_users += 1
            continue
        resolved_user_owners[user] = owner
    # A unique fingerprint match is insufficient if two encoded users map to
    # one reviewer.  Preserve that identity collision instead of guessing.
    owner_users = defaultdict(list)
    for user, owner in resolved_user_owners.items():
        owner_users[owner].append(user)
    owner_collision_users = {user for users in owner_users.values() if len(users) > 1 for user in users}
    for user, owner in sorted(resolved_user_owners.items()):
        if user in owner_collision_users:
            continue
        mapped_owners.add(owner)
        for event in support_by_user[user]:
            alignments.append({'userID': user, 'itemID': int(event['itemID']),
                               'timestamp': int(event['timestamp']),
                               'raw_row_index': owner_rows[(owner, event['identity'])][0]})
    duplicate_records = sum(value - 1 for value in owner_fingerprints.values() if value > 1)
    audit = {
        'raw_review_records': raw_count,
        'raw_unique_owner_fingerprints': len(owner_fingerprints),
        'raw_duplicate_owner_fingerprint_records': duplicate_records,
        'raw_duplicate_owner_item_records': sum(value - 1 for value in owner_items.values() if value > 1),
        'support_edges': len(support_events),
        'support_users': len(support_by_user),
        'matched_support_edges': len(alignments),
        'matched_support_users': len(mapped_owners),
        'unresolved_support_users': unresolved_users,
        'ambiguous_support_users': ambiguous_users,
        'duplicate_raw_event_users': duplicate_event_users,
        'owner_collision_support_users': len(owner_collision_users),
        'ambiguous_support_fingerprints': sum(len(inverted[key]) > 1 for key in keys),
        'missing_support_fingerprints': sum(not inverted[key] for key in keys),
        'identity_uses_support_only': True,
        'ratings_used_only_for_identity': True,
        'reviewer_identity_exported': False,
    }
    return alignments, audit


def load_support_reviews(inter_path, item_mapping_path, reviews_path,
                         heldout_pairs=None, strict=True):
    """Return (records, audit) for globally held-out TRAIN support reviews.

    Records contain userID, itemID, timestamp, reviewText and summary only.
    Official validation/test rows are ignored before any identity is formed.
    """
    train = read_train_events(inter_path, item_mapping_path)
    frozen_targets = compute_holdout_pairs(train)
    targets = frozen_targets if heldout_pairs is None else set(
        (int(user), int(item)) for user, item in heldout_pairs)
    if targets != frozen_targets:
        raise ValueError('heldout pairs differ from the frozen TRAIN target rule')
    support = [event for event in train if (event['userID'], event['itemID']) not in targets]
    alignments, audit = match_support_identities(support, iter_raw_reviews(reviews_path))
    audit.update(original_train_edges=len(train), globally_removed_target_edges=len(targets),
                 expected_support_edges=len(train) - len(targets),
                 heldout_reviews_used=False, official_validation_reviews_used=False,
                 official_test_reviews_used=False)
    if strict and len(alignments) != len(support):
        raise ValueError('support review identities are not completely unique: ' + json.dumps(audit))
    by_raw_index = {event['raw_row_index']: event for event in alignments}
    if len(by_raw_index) != len(alignments):
        raise ValueError('a raw review was assigned to multiple TRAIN support edges')
    records = []
    for index, raw in enumerate(iter_raw_reviews(reviews_path)):
        if index not in by_raw_index:
            continue
        event = by_raw_index[index]
        text = raw.get('reviewText')
        summary = raw.get('summary')
        records.append({'userID': event['userID'], 'itemID': event['itemID'],
                        'timestamp': event['timestamp'],
                        'reviewText': text if isinstance(text, str) else '',
                        'summary': summary if isinstance(summary, str) else ''})
    if len(records) != len(alignments):
        raise RuntimeError('matched review alignment changed between passes')
    records.sort(key=lambda row: (row['userID'], row['itemID']))
    audit.update(support_nonempty_review_text=sum(bool(row['reviewText'].strip()) for row in records),
                 support_nonempty_summary=sum(bool(row['summary'].strip()) for row in records),
                 support_review_text_word_count=sum(len(row['reviewText'].split()) for row in records),
                 all_targets_absent_from_export=not any(
                     (row['userID'], row['itemID']) in targets for row in records))
    return records, audit
