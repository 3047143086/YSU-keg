"""Fixed same-item review controls without a dependency on the text encoder."""
import hashlib
import html
from collections import defaultdict
from html.parser import HTMLParser

import numpy as np


class _TextOnly(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, value):
        self.parts.append(value)


def review_text(record):
    parser = _TextOnly()
    parser.feed(html.unescape(str(record.get('summary', '')) + ' ' +
                            str(record.get('reviewText', ''))))
    return ' '.join(' '.join(parser.parts).split())


def same_item_donors(records):
    groups = defaultdict(list)
    for index, row in enumerate(records):
        text = review_text(row)
        if text:
            owner, item = int(row['userID']), int(row['itemID'])
            tie = hashlib.sha256(('review-owner-v1:{}:{}'.format(owner, item)).encode('ascii')).digest()
            groups[item].append((len(text.split()), tie, owner, index))
    for item in groups:
        groups[item].sort()
    donors = np.full(len(records), -1, dtype=np.int64)
    for index, row in enumerate(records):
        text = review_text(row)
        if not text:
            continue
        entries = groups[int(row['itemID'])]
        owner, size = int(row['userID']), len(text.split())
        candidates = [entry for entry in entries if entry[2] != owner]
        if candidates:
            donors[index] = min(candidates, key=lambda entry: (abs(entry[0] - size), entry[1]))[3]
    return donors


def normalize(rows):
    rows = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(rows).all():
        raise ValueError('nonfinite representation')
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-8)


def profile(vectors, user_ids, n_users):
    result = np.zeros((n_users, vectors.shape[1]), dtype=np.float32)
    np.add.at(result, user_ids, vectors)
    return normalize(result)


def score_rows(profiles, item_vectors, users, items):
    return np.einsum('ud,umd->um', profiles[users], item_vectors[items], optimize=True)
