"""Measure whether disjoint-history witnesses beat ordinary collaborative evidence."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from evidence import build_user_evidence, gather_scores, prune_topk


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def key(prefix, *values):
    return hashlib.sha256((prefix + ':'.join(map(str, values))).encode('ascii')).digest()


def internal_split(train):
    support = train.copy().tolil()
    users, targets = [], []
    for user in range(train.shape[0]):
        items = train.indices[train.indptr[user]:train.indptr[user + 1]]
        if len(items) >= 4:
            target = min(items, key=lambda item: key('dual-witness-target:v1:', user, int(item)))
            users.append(user)
            targets.append(int(target))
            support[user, target] = 0
    support = support.tocsr()
    support.eliminate_zeros()
    return support, np.asarray(users, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def comparisons(train, users, targets):
    rng = np.random.default_rng(2026)
    negatives = np.empty((len(users), 16), dtype=np.int64)
    for row, user in enumerate(users):
        observed = set(train.indices[train.indptr[user]:train.indptr[user + 1]].tolist())
        if len(observed) == train.shape[1]:
            raise ValueError('No unobserved comparisons for a diagnostic user')
        for index in range(16):
            item = int(rng.integers(train.shape[1]))
            while item in observed:
                item = int(rng.integers(train.shape[1]))
            negatives[row, index] = item
    return np.column_stack([targets, negatives])


def auc(scores):
    if scores.ndim != 2 or scores.shape[1] != 17 or not np.isfinite(scores).all():
        raise ValueError('Expected 1+16 finite comparison scores')
    difference = scores[:, :1] - scores[:, 1:]
    return ((difference > 0) + .5 * (difference == 0)).mean(axis=1)


def item_scores(item_weights, support, users, items, block_size=256):
    scores = np.zeros(items.shape, dtype=np.float64)
    for start in range(0, len(users), block_size):
        stop = min(start + block_size, len(users))
        raw = (support[users[start:stop]] @ item_weights.T).toarray()
        scores[start:stop] = raw[np.arange(stop - start)[:, None], items[start:stop]]
    return scores


def paired(delta):
    rng = np.random.default_rng(2026)
    draws = [float(delta[rng.integers(len(delta), size=len(delta))].mean()) for _ in range(2000)]
    return {'mean_delta': float(delta.mean()),
            'user_bootstrap_95ci': np.quantile(draws, [.025, .975]).tolist()}


def decide(per_user, coverage, users):
    controls = ('itemknn', 'userknn', 'min2')
    pairs = {name: paired(per_user['dual'] - per_user[name])
             for name in controls + ('shuffled',)}
    halves = np.asarray([key('dual-witness-half:v1:', int(user))[0] % 2 for user in users])
    half_deltas = {name: [float((per_user['dual'] - per_user[name])[halves == h].mean())
                          for h in (0, 1)] for name in controls}
    passed = (coverage >= .25
              and all(pairs[name]['mean_delta'] >= .005
                      and pairs[name]['user_bootstrap_95ci'][0] > 0 for name in controls)
              and pairs['shuffled']['mean_delta'] >= .005
              and all(value > 0 for values in half_deltas.values() for value in values))
    return {'status': 'keep' if passed else 'discard', 'passed': bool(passed),
            'positive_nonzero_coverage': coverage, 'paired_vs_controls': pairs,
            'user_hash_half_deltas': half_deltas, 'formal_training_authorized_by_this_screen': False,
            'novelty_established': False, 'publication_evidence': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('baby', 'sports'), required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    inter = args.data_root / args.dataset / (args.dataset + '.inter')
    shape_file = args.data_root / args.dataset / 'image_feat.npy'
    source_hashes = {str(path): sha(path) for path in (inter, shape_file)}
    frame = pd.read_csv(inter, sep='\t', usecols=['userID', 'itemID', 'x_label'])
    n_users = int(frame.userID.max()) + 1
    n_items = int(np.load(shape_file, mmap_mode='r').shape[0])
    frame = frame.loc[frame.x_label == 0, ['userID', 'itemID']].copy()
    if frame.duplicated(['userID', 'itemID']).any():
        raise ValueError('Binary TRAIN interactions must be unique')
    train = sparse.csr_matrix((np.ones(len(frame)), (frame.userID, frame.itemID)),
                              shape=(n_users, n_items))
    support, users, targets = internal_split(train)
    if not len(users):
        raise ValueError('No eligible internal diagnostic targets')
    if np.any(support[users, targets].A1 != 0):
        raise RuntimeError('Internal target survived in support')
    items = comparisons(train, users, targets)
    per_user, records, coverage = {}, {}, None
    for mode in ('itemknn', 'userknn', 'min2', 'dual', 'shuffled'):
        arm_start = time.monotonic()
        if mode == 'itemknn':
            pop = np.asarray(support.sum(axis=0)).reshape(-1)
            cooc = (support.T @ support).tocsr()
            cooc.setdiag(0)
            cooc.eliminate_zeros()
            inverse = 1 / np.sqrt(np.maximum(pop, 1e-6))
            sim = cooc.multiply(inverse[:, None]).multiply(inverse[None, :]).tocsr()
            values = item_scores(prune_topk(sim, 1000), support, users, items)
            del sim, cooc
            neighbor_nnz = None
        else:
            weights = build_user_evidence(support, mode=mode, topk=100, block_size=256)
            values = gather_scores(weights, support, users, items)
            neighbor_nnz = int(weights.nnz)
            del weights
        per_user[mode] = auc(values)
        arm_coverage = float(np.mean(values[:, 0] > 0))
        if mode == 'dual':
            coverage = arm_coverage
        records[mode] = {'mean_auc': float(per_user[mode].mean()),
                         'positive_nonzero_coverage': arm_coverage,
                         'neighbor_weights_nnz': neighbor_nnz,
                         'seconds': time.monotonic() - arm_start}
        print(json.dumps({'dataset': args.dataset, 'arm': mode, **records[mode]}), flush=True)
        del values
    decision = decide(per_user, coverage, users)
    np.savez_compressed(args.output / 'per_user.npz', user_ids=users, targets=targets,
                        comparisons=items, **per_user)
    if any(sha(path) != value for path, value in source_hashes.items()):
        raise RuntimeError('Original data changed during the diagnostic')
    record = {'dataset': args.dataset, 'status': 'completed', 'decision': decision,
              'arms': records, 'diagnostic_users': len(users), 'original_train_edges': train.nnz,
              'support_edges': support.nnz, 'source_sha256': source_hashes,
              'elapsed_seconds': time.monotonic() - started, 'validation_evaluated': False,
              'test_evaluated': False, 'training_internal_only': True,
              'all_internal_targets_absent': True, 'comparison_ties_count': .5,
              'source_revision': json.loads((Path(__file__).parent / 'source_revision.json').read_text())['commit']}
    with (args.output / 'diagnosis.json').open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps({'dataset': args.dataset, 'decision': decision}, indent=2), flush=True)


if __name__ == '__main__':
    main()
