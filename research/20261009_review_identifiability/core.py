"""Fixed same-item, exact-degree author retrieval from TRAIN review vectors."""
import hashlib
from collections import defaultdict

import numpy as np


SPLIT_PREFIX = "review-identifiability-split:v1:"
HALF_PREFIX = "dual-witness-half:v1:"


def digest(prefix, *values):
    return hashlib.sha256((prefix + ":".join(map(str, values))).encode("ascii")).digest()


def validate_edges(user_ids, item_ids):
    users, items = np.asarray(user_ids), np.asarray(item_ids)
    if users.ndim != 1 or items.ndim != 1 or len(users) != len(items) or not len(users):
        raise ValueError("Expected nonempty aligned user/item vectors")
    if not np.issubdtype(users.dtype, np.integer) or not np.issubdtype(items.dtype, np.integer):
        raise ValueError("User and item IDs must be integers")
    if (users < 0).any() or (items < 0).any():
        raise ValueError("Negative edge ID")
    users, items = users.astype(np.int64), items.astype(np.int64)
    if len(set(zip(users.tolist(), items.tolist()))) != len(users):
        raise ValueError("Cached user-item edges must be unique")
    return users, items


def split_edges(user_ids, item_ids):
    users, items = validate_edges(user_ids, item_ids)
    return np.asarray([digest(SPLIT_PREFIX, int(u), int(i))[0] % 2 == 0
                       for u, i in zip(users, items)], dtype=bool)


def plan_groups(user_ids, item_ids, n_users, is_a=None):
    users, items = validate_edges(user_ids, item_ids)
    if is_a is None:
        is_a = split_edges(users, items)
    is_a = np.asarray(is_a)
    if is_a.dtype != np.bool_ or is_a.shape != users.shape:
        raise ValueError("A mask must be aligned boolean values")
    if isinstance(n_users, bool) or int(n_users) != n_users or n_users < 1 or (users >= n_users).any():
        raise ValueError("Cached owners must fit the frozen complete support population")
    population = np.arange(int(n_users), dtype=np.int64)
    positions = {int(user): row for row, user in enumerate(population)}
    a_counts = {int(user): 0 for user in population}
    b_counts = np.zeros(len(population), dtype=np.int64)
    for user, a in zip(users, is_a):
        if a:
            a_counts[int(user)] += 1
        else:
            b_counts[positions[int(user)]] += 1
    candidates = defaultdict(list)
    a_history = defaultdict(set)
    for index in np.flatnonzero(is_a):
        a_history[int(users[index])].add(int(items[index]))
    for index in np.flatnonzero(~is_a):
        owner = int(users[index])
        if int(items[index]) in a_history[owner]:
            raise ValueError("A profile contains its B-query item")
        degree = a_counts[owner]
        if degree >= 2:
            candidates[(int(items[index]), degree)].append(int(index))
    rng = np.random.RandomState(2026)
    groups = []
    scorable_queries = np.zeros(len(users), dtype=bool)
    fixed_points, assignments = 0, 0
    for key, indices in sorted(candidates.items()):
        if len(indices) < 2:
            continue
        indices = np.asarray(sorted(indices, key=lambda index: int(users[index])), dtype=np.int64)
        permutation = rng.permutation(len(indices))
        groups.append({"item": key[0], "a_degree": key[1], "indices": indices,
                       "owners": users[indices], "permutation": permutation})
        scorable_queries[indices] = True
        fixed_points += int((permutation == np.arange(len(indices))).sum())
        assignments += len(indices)
    scorable_users = np.zeros(len(population), dtype=bool)
    scorable_users[np.unique(users[scorable_queries])] = True
    return {"users": users, "items": items, "population": population,
            "population_positions": positions, "is_a": is_a, "a_counts": a_counts,
            "b_counts": b_counts, "groups": groups, "scorable_queries": scorable_queries,
            "scorable_users": scorable_users, "null_fixed_points": fixed_points,
            "null_assignments": assignments}


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite rank-2 vectors")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def profile_a(vectors, plan):
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(plan["users"]) or not np.isfinite(values).all():
        raise ValueError("Vectors must align with cached edges")
    result = np.zeros((len(plan["population"]), values.shape[1]), dtype=np.float64)
    positions = np.asarray([plan["population_positions"][int(user)] for user in plan["users"]])
    np.add.at(result, positions[plan["is_a"]], values[plan["is_a"]])
    return normalize_rows(result)


def _row_auc(scores, true_column):
    values = np.asarray(scores, dtype=np.float64)
    difference = values[true_column] - values
    credits = (difference > 0).astype(np.float64) + .5 * (difference == 0)
    return float((credits.sum() - .5) / (len(values) - 1))


def evaluate_view(vectors, plan, block_size=128, checkpoint=None):
    if isinstance(block_size, bool) or int(block_size) != block_size or block_size < 1:
        raise ValueError("block_size must be positive")
    values = np.asarray(vectors, dtype=np.float64)
    profiles = profile_a(values, plan)
    real = np.full(len(values), .5, dtype=np.float64)
    null = np.full(len(values), .5, dtype=np.float64)
    for group in plan["groups"]:
        if checkpoint is not None:
            checkpoint()
        indices, owners = group["indices"], group["owners"]
        profile_rows = profiles[[plan["population_positions"][int(user)] for user in owners]]
        review_rows = values[indices]
        permutation = group["permutation"]
        # Score true and permuted profiles together in bounded row blocks.
        for start in range(0, len(indices), int(block_size)):
            if checkpoint is not None:
                checkpoint()
            stop = min(start + int(block_size), len(indices))
            actual = profile_rows[start:stop] @ review_rows.T
            shuffled = profile_rows[permutation[start:stop]] @ review_rows.T
            for offset, column in enumerate(range(start, stop)):
                real[indices[column]] = _row_auc(actual[offset], column)
                null[indices[column]] = _row_auc(shuffled[offset], column)
    sums_real = np.zeros(len(plan["population"]), dtype=np.float64)
    sums_null = np.zeros_like(sums_real)
    for index in np.flatnonzero(~plan["is_a"]):
        position = plan["population_positions"][int(plan["users"][index])]
        sums_real[position] += real[index]
        sums_null[position] += null[index]
    divisor = np.maximum(plan["b_counts"], 1)
    user_real = sums_real / divisor
    user_null = sums_null / divisor
    user_real[plan["b_counts"] == 0] = .5
    user_null[plan["b_counts"] == 0] = .5
    return {"user_real": user_real, "user_null": user_null,
            "query_real": real, "query_null": null,
            "zero_a_profiles": int((np.linalg.norm(profiles, axis=1) == 0).sum()),
            "zero_b_vectors": int((np.linalg.norm(values[~plan["is_a"]], axis=1) == 0).sum())}


def summarize(values, plan, bootstrap=2000, checkpoint=None):
    real = np.asarray(values["user_real"], dtype=np.float64)
    null = np.asarray(values["user_null"], dtype=np.float64)
    if (real.shape != plan["population"].shape or null.shape != real.shape
            or plan["scorable_users"].shape != real.shape
            or not np.isfinite(real).all() or not np.isfinite(null).all()
            or np.any(real < 0) or np.any(real > 1) or np.any(null < 0) or np.any(null > 1)):
        raise ValueError("User AUCs must be finite, aligned probabilities over the complete population")
    if isinstance(bootstrap, bool) or int(bootstrap) != bootstrap or bootstrap < 1:
        raise ValueError("bootstrap must be a positive integer")
    delta = real - null
    if not len(delta):
        raise ValueError("Invalid user deltas")
    halves = np.asarray([digest(HALF_PREFIX, int(user))[0] % 2 for user in plan["population"]])
    if not all(np.any(halves == half) for half in (0, 1)):
        raise ValueError("Both frozen user hash halves must be nonempty")
    rng = np.random.RandomState(2026)
    draws = []
    for index in range(bootstrap):
        if checkpoint is not None and index % 100 == 0:
            checkpoint()
        draws.append(float(delta[rng.randint(len(delta), size=len(delta))].mean()))
    interval = np.quantile(draws, [.025, .975]).tolist()
    half_deltas = [float(delta[halves == half].mean()) for half in (0, 1)]
    coverage = float(plan["scorable_users"].mean())
    mean_delta = float(delta.mean())
    real_mean = float(values["user_real"].mean())
    passed = (coverage >= .25 and real_mean >= .505 and mean_delta >= .005 and interval[0] > 0
              and all(value > 0 for value in half_deltas))
    return {"real_mean_user_auc": real_mean,
            "null_mean_user_auc": float(values["user_null"].mean()),
            "mean_delta": mean_delta, "paired_user_bootstrap_95ci": interval,
            "user_hash_half_deltas": half_deltas, "scorable_user_coverage": coverage,
            "status": "expression_identifiable" if passed else "expression_not_established_by_this_audit",
            "passed": bool(passed), "preference_established": False,
            "recommendation_gain_established": False,
            "formal_training_authorized": False, "publication_evidence": False}
