"""One fixed leave-query-owner-out review-bank geometry and joint gate."""
import hashlib

import numpy as np
from scipy import sparse


CONTROLS = ("item_content", "old_target_geometry", "raw_review_bank", "raw_donor_bank",
            "residual_donor_bank", "popularity", "itemknn", "userknn")
ARMS = ("candidate",) + CONTROLS
EPS = 1e-12


def matrix(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("Expected finite rank-2 vectors")
    return value


def aligned_ids(users, items, n_users, n_items):
    users, items = np.asarray(users), np.asarray(items)
    if (users.ndim != 1 or items.shape != users.shape or not len(users)
            or not np.issubdtype(users.dtype, np.integer) or not np.issubdtype(items.dtype, np.integer)
            or np.any(users < 0) or np.any(users >= n_users)
            or np.any(items < 0) or np.any(items >= n_items)):
        raise ValueError("Invalid support owner/item IDs")
    users, items = users.astype(np.int64), items.astype(np.int64)
    if len(set(zip(users.tolist(), items.tolist()))) != len(users):
        raise ValueError("Support owner-item pairs must be unique")
    return users, items


def binary_support(users, items, n_users, n_items):
    users, items = aligned_ids(users, items, n_users, n_items)
    # F23 builds its initial support from default float64 ones; UserKNN's
    # frozen helper separately casts to float32 while ItemKNN uses this dtype.
    result = sparse.csr_matrix((np.ones(len(users), dtype=np.float64), (users, items)),
                              shape=(n_users, n_items), dtype=np.float64)
    result.sort_indices()
    return result


def validate_pairs(support, users, targets, comparisons):
    users, targets, choices = np.asarray(users), np.asarray(targets), np.asarray(comparisons)
    if (users.ndim != 1 or targets.shape != users.shape or choices.shape != (len(users), 17)
            or not all(np.issubdtype(v.dtype, np.integer) for v in (users, targets, choices))
            or len(np.unique(users)) != len(users) or np.any(users < 0)
            or np.any(users >= support.shape[0]) or np.any(choices < 0)
            or np.any(choices >= support.shape[1]) or not np.array_equal(choices[:, 0], targets)):
        raise ValueError("Invalid frozen 1+16 target/comparison rows")
    for user, row in zip(users, choices):
        if np.any(support.getrow(int(user))[:, row].toarray() != 0):
            raise ValueError("A globally held-out target or comparison survived in query support")
    pairs = sorted(zip(users.tolist(), targets.tolist()))
    import json
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode("ascii")).hexdigest()


def item_bank(vectors, owner_ids, item_ids, n_users, n_items):
    values = matrix(vectors)
    owners, items = aligned_ids(owner_ids, item_ids, n_users, n_items)
    if len(values) != len(owners):
        raise ValueError("Review rows must align with support edges")
    sums = np.zeros((n_items, values.shape[1]), dtype=np.float64)
    counts = np.zeros(n_items, dtype=np.int64)
    np.add.at(sums, items, values)
    np.add.at(counts, items, 1)
    return {"sums": sums, "counts": counts, "values": values,
            "lookup": {(int(u), int(i)): index for index, (u, i) in enumerate(zip(owners, items))}}


def bank_scores(profiles, bank, users, comparisons, block_size=128, checkpoint=None):
    profiles, choices = matrix(profiles), np.asarray(comparisons)
    users = np.asarray(users)
    if (users.ndim != 1 or choices.ndim != 2 or len(users) != len(choices)
            or not np.issubdtype(users.dtype, np.integer) or not np.issubdtype(choices.dtype, np.integer)
            or np.any(users < 0) or np.any(users >= len(profiles))
            or np.any(choices < 0) or np.any(choices >= len(bank["sums"]))
            or profiles.shape[1] != bank["sums"].shape[1]):
        raise ValueError("Invalid row-paired bank scores")
    if isinstance(block_size, bool) or int(block_size) != block_size or block_size < 1:
        raise ValueError("block_size must be positive")
    result = np.zeros(choices.shape, dtype=np.float64)
    availability = np.zeros(choices.shape, dtype=bool)
    removed = 0
    for start in range(0, len(users), int(block_size)):
        if checkpoint is not None: checkpoint()
        stop = min(start + int(block_size), len(users))
        vectors = bank["sums"][choices[start:stop]].copy()
        counts = bank["counts"][choices[start:stop]].copy()
        for offset, user in enumerate(users[start:stop]):
            for column, item in enumerate(choices[start + offset]):
                edge = bank["lookup"].get((int(user), int(item)))
                if edge is not None:
                    vectors[offset, column] -= bank["values"][edge]
                    counts[offset, column] -= 1
                    removed += 1
        norms = np.linalg.norm(vectors, axis=2)
        valid = (counts > 0) & (norms > EPS)
        vectors /= np.maximum(norms[:, :, None], EPS)
        vectors[~valid] = 0
        result[start:stop] = np.einsum("ud,umd->um", profiles[users[start:stop]], vectors, optimize=True)
        availability[start:stop] = valid
    if not np.isfinite(result).all(): raise ValueError("Nonfinite bank scores")
    return result, availability, removed


def auc(scores):
    scores = matrix(scores)
    if scores.shape[1] != 17: raise ValueError("Exactly 1+16 comparisons required")
    differences = scores[:, :1] - scores[:, 1:]
    return ((differences > 0) + .5 * (differences == 0)).mean(axis=1)


def statistics(per_user, coverage, users, bootstrap=2000, checkpoint=None):
    users = np.asarray(users)
    if (set(per_user) != set(ARMS) or users.ndim != 1 or not len(users)
            or not np.issubdtype(users.dtype, np.integer) or len(np.unique(users)) != len(users)
            or isinstance(coverage, bool) or not np.isfinite(coverage) or not 0 <= coverage <= 1):
        raise ValueError("Missing arms, invalid user population or coverage")
    if isinstance(bootstrap, bool) or int(bootstrap) != bootstrap or bootstrap < 1:
        raise ValueError("bootstrap must be positive integer")
    for value in per_user.values():
        if (np.asarray(value).shape != users.shape or not np.isfinite(value).all()
                or np.any(value < 0) or np.any(value > 1)):
            raise ValueError("Every user AUC must be a finite aligned probability")
    halves = np.asarray([hashlib.sha256(("dual-witness-half:v1:" + str(int(u))).encode("ascii")).digest()[0] % 2
                         for u in users])
    if not all(np.any(halves == half) for half in (0, 1)):
        raise ValueError("Both fixed hash halves required")
    deltas = np.column_stack([per_user["candidate"]-per_user[name] for name in CONTROLS])
    rng = np.random.default_rng(2026)
    draws = np.empty((int(bootstrap), len(CONTROLS)), dtype=np.float64)
    for index in range(int(bootstrap)):
        if checkpoint is not None and index % 100 == 0: checkpoint()
        draws[index] = deltas[rng.integers(len(users), size=len(users))].mean(axis=0)
    intervals = np.quantile(draws, [.025, .975], axis=0)
    pairs, failed = {}, []
    if coverage < .80: failed.append("all17_nonzero_coverage")
    if float(per_user["candidate"].mean()) < .505: failed.append("candidate_mean_above_chance")
    for column, name in enumerate(CONTROLS):
        delta = deltas[:, column]
        pair = {"delta_name": "candidate_minus_"+name, "mean_delta":float(delta.mean()),
                "user_bootstrap_95ci":intervals[:, column].tolist(),
                "user_hash_half_deltas":[float(delta[halves==half].mean()) for half in (0, 1)]}
        pairs[name] = pair
        if pair["mean_delta"] < .005: failed.append(name+":mean_gain")
        if pair["user_bootstrap_95ci"][0] <= 0: failed.append(name+":bootstrap_lower_bound")
        if any(value <= 0 for value in pair["user_hash_half_deltas"]): failed.append(name+":both_hash_halves")
    return {"status":"review_bank_alignment_supported_in_train_internal_diagnostic" if not failed
                    else "discard_this_fixed_bank_geometry", "passed":not failed,
            "candidate_mean_auc":float(per_user["candidate"].mean()),"coverage":float(coverage),
            "paired_vs_controls":pairs,"failed_conditions":failed,
            "formal_training_authorized":False,"preference_established":False,
            "novelty_established":False,"publication_evidence":False}, halves
