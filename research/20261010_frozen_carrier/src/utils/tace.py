"""Temporal-affinity behavior evidence for checkpoint diagnostics."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def _prune_csr_topk(matrix, topk):
    """Keep each row's strongest entries without densifying the matrix."""
    matrix = matrix.tocsr()
    topk = int(topk)
    if topk <= 0:
        return matrix

    new_indptr = [0]
    new_indices = []
    new_data = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        row_indices = matrix.indices[start:end]
        row_data = matrix.data[start:end]
        if row_data.size > topk:
            keep = np.argpartition(row_data, -topk)[-topk:]
            keep = keep[np.argsort(row_indices[keep])]
            row_indices = row_indices[keep]
            row_data = row_data[keep]
        new_indices.extend(row_indices.tolist())
        new_data.extend(row_data.tolist())
        new_indptr.append(len(new_indices))

    output = sp.csr_matrix(
        (
            np.asarray(new_data, dtype=np.float32),
            np.asarray(new_indices, dtype=np.int32),
            np.asarray(new_indptr, dtype=np.int64),
        ),
        shape=matrix.shape,
        dtype=np.float32,
    )
    output.eliminate_zeros()
    return output


def build_temporal_affinity(
    user_ids,
    item_ids,
    timestamps,
    n_users,
    n_items,
    topk=1000,
):
    """Build cosine-normalized item affinity from chronological train episodes.

    Each user contributes ``1 / max(episode_rank_distance, 1)`` to an item
    pair. Equal timestamps share an episode rank. Repeated user-item events are
    collapsed to the earliest training timestamp before pair construction.
    """
    users = np.asarray(user_ids, dtype=np.int64).reshape(-1)
    items = np.asarray(item_ids, dtype=np.int64).reshape(-1)
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if not (users.size == items.size == times.size):
        raise ValueError("user_ids, item_ids, and timestamps must have equal length")
    if users.size:
        if users.min() < 0 or users.max() >= int(n_users):
            raise ValueError("user_ids contains an out-of-range identifier")
        if items.min() < 0 or items.max() >= int(n_items):
            raise ValueError("item_ids contains an out-of-range identifier")
        if not np.isfinite(times).all():
            raise ValueError("timestamps must be finite")

    history_users = []
    history_items = []
    affinity_rows = []
    affinity_cols = []
    affinity_values = []
    pair_count = 0

    order = np.lexsort((items, times, users))
    users = users[order]
    items = items[order]
    times = times[order]
    boundaries = (
        np.flatnonzero(np.r_[True, users[1:] != users[:-1], True])
        if users.size
        else np.asarray([0], dtype=np.int64)
    )

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        user = int(users[start])
        group_items = items[start:end]
        group_times = times[start:end]

        # Sorting puts the earliest occurrence first; retain that event per item.
        _, first = np.unique(group_items, return_index=True)
        first = np.sort(first)
        group_items = group_items[first]
        group_times = group_times[first]
        history_users.append(np.full(group_items.size, user, dtype=np.int64))
        history_items.append(group_items)

        if group_items.size < 2:
            continue
        _, episode_rank = np.unique(group_times, return_inverse=True)
        left, right = np.triu_indices(group_items.size, k=1)
        distance = np.abs(episode_rank[left] - episode_rank[right])
        weight = 1.0 / np.maximum(distance, 1).astype(np.float32)
        affinity_rows.extend((group_items[left], group_items[right]))
        affinity_cols.extend((group_items[right], group_items[left]))
        affinity_values.extend((weight, weight))
        pair_count += int(left.size)

    if history_users:
        history_users = np.concatenate(history_users)
        history_items = np.concatenate(history_items)
    else:
        history_users = np.empty(0, dtype=np.int64)
        history_items = np.empty(0, dtype=np.int64)
    train = sp.csr_matrix(
        (
            np.ones(history_users.size, dtype=np.float32),
            (history_users, history_items),
        ),
        shape=(int(n_users), int(n_items)),
        dtype=np.float32,
    )
    train.sum_duplicates()
    if train.nnz:
        train.data.fill(1.0)

    if affinity_values:
        rows = np.concatenate(affinity_rows)
        cols = np.concatenate(affinity_cols)
        values = np.concatenate(affinity_values).astype(np.float32, copy=False)
        affinity = sp.csr_matrix(
            (values, (rows, cols)),
            shape=(int(n_items), int(n_items)),
            dtype=np.float32,
        )
    else:
        affinity = sp.csr_matrix((int(n_items), int(n_items)), dtype=np.float32)
    affinity.sum_duplicates()
    affinity.setdiag(0.0)
    affinity.eliminate_zeros()

    item_popularity = np.asarray(train.sum(axis=0)).reshape(-1).astype(np.float32)
    denom = np.sqrt(np.maximum(item_popularity, 1e-6)).astype(np.float32)
    inv = np.divide(1.0, denom, out=np.zeros_like(denom), where=denom > 0)
    affinity = affinity.multiply(inv[:, None]).multiply(inv[None, :]).tocsr()
    affinity = _prune_csr_topk(affinity, topk)
    affinity.sort_indices()

    user_degree = np.diff(train.indptr)
    nonempty = user_degree[user_degree > 0]
    stats = {
        "train_interactions": int(train.nnz),
        "nonempty_users": int(nonempty.size),
        "mean_history_length": float(nonempty.mean()) if nonempty.size else 0.0,
        "raw_unordered_pairs": int(pair_count),
        "affinity_nnz": int(affinity.nnz),
        "kernel": "inverse_distinct_timestamp_rank",
    }
    return affinity, train, stats


def _gather_csr_rows(matrix, column_indices):
    matrix = matrix.tocsr()
    matrix.sort_indices()
    output = np.zeros(column_indices.shape, dtype=np.float32)
    for row in range(column_indices.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        if start == end:
            continue
        columns = matrix.indices[start:end]
        values = matrix.data[start:end]
        targets = column_indices[row]
        positions = np.searchsorted(columns, targets)
        clipped = positions.clip(max=columns.size - 1)
        valid = (positions < columns.size) & (columns[clipped] == targets)
        if np.any(valid):
            output[row, valid] = values[positions[valid]]
    return output


class TemporalAffinityScorer:
    """Compute candidate-only scores with BRIDGE-compatible normalization."""

    def __init__(self, train_csr, item_affinity, aggregation="sum_sqrt", norm="z"):
        self.train_csr = train_csr.tocsr()
        self.item_affinity = item_affinity.tocsr()
        self.aggregation = str(aggregation)
        self.norm = str(norm)

    def score(self, users, candidate_items):
        device = candidate_items.device
        users_np = users.detach().cpu().numpy().astype(np.int64, copy=False)
        candidates_np = (
            candidate_items.detach().cpu().numpy().astype(np.int64, copy=False)
        )
        history = self.train_csr[users_np]
        score_rows = history.dot(self.item_affinity.T).tocsr()
        if self.aggregation in ("mean", "sum_sqrt"):
            lengths = np.maximum(np.diff(history.indptr).astype(np.float32), 1.0)
            scale = 1.0 / lengths
            if self.aggregation == "sum_sqrt":
                scale = 1.0 / np.sqrt(lengths)
            score_rows = sp.diags(scale).dot(score_rows).tocsr()
        elif self.aggregation != "sum":
            raise ValueError("unknown aggregation: {}".format(self.aggregation))

        scores = torch.from_numpy(_gather_csr_rows(score_rows, candidates_np)).to(device)
        if self.norm == "z":
            scores = (scores - scores.mean(dim=1, keepdim=True)) / scores.std(
                dim=1, keepdim=True, unbiased=False
            ).clamp_min(1e-6)
        elif self.norm == "minmax":
            low = scores.min(dim=1, keepdim=True).values
            high = scores.max(dim=1, keepdim=True).values
            scores = (scores - low) / (high - low).clamp_min(1e-6)
        elif self.norm != "none":
            raise ValueError("unknown normalization: {}".format(self.norm))
        return scores
