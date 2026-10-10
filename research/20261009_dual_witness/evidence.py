"""Sparse, deterministic dual-witness co-user evidence.

The module is deliberately independent of Torch and the BRIDGE model.  It
turns a binary training interaction matrix into fixed user-neighbour evidence
using blocked sparse products.  No fallback is applied when a user has no
neighbour or no witness in either item view.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix


ITEM_HASH_PREFIX = "dual-witness:v1:"


def _as_binary_csr(matrix) -> csr_matrix:
    """Copy and canonicalize a binary user-item matrix."""
    result = sparse.csr_matrix(matrix, dtype=np.float32, copy=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    if result.nnz and not np.all(result.data == 1):
        raise ValueError("train matrix must be binary after duplicate aggregation")
    return result


def item_view_labels(n_items: int) -> np.ndarray:
    """Return the immutable A/B label for every item ID."""
    n_items = int(n_items)
    if n_items < 0:
        raise ValueError("n_items must be nonnegative")
    labels = np.empty(n_items, dtype=np.int8)
    for item_id in range(n_items):
        digest = hashlib.sha256((ITEM_HASH_PREFIX + str(item_id)).encode("ascii")).digest()
        labels[item_id] = int.from_bytes(digest[:8], "big") & 1
    return labels


def item_partition(n_items: int) -> np.ndarray:
    """Public boolean A/B partition (False=A, True=B)."""
    return item_view_labels(n_items).astype(bool, copy=False)


def split_item_views(train) -> Tuple[csr_matrix, csr_matrix]:
    """Split a binary matrix into disjoint A and B item views."""
    train = _as_binary_csr(train)
    labels = item_view_labels(train.shape[1])
    rows, cols = train.nonzero()
    data = np.ones(rows.size, dtype=np.float32)
    a = csr_matrix((data[labels[cols] == 0],
                    (rows[labels[cols] == 0], cols[labels[cols] == 0])),
                   shape=train.shape, dtype=np.float32)
    b = csr_matrix((data[labels[cols] == 1],
                    (rows[labels[cols] == 1], cols[labels[cols] == 1])),
                   shape=train.shape, dtype=np.float32)
    a.sort_indices()
    b.sort_indices()
    return a, b


def _l2_rows(matrix: csr_matrix) -> csr_matrix:
    squared = np.asarray(matrix.multiply(matrix).sum(axis=1)).reshape(-1)
    inv = np.zeros_like(squared, dtype=np.float32)
    nonzero = squared > 0
    inv[nonzero] = 1.0 / np.sqrt(squared[nonzero])
    return sparse.diags(inv, format="csr").dot(matrix).tocsr()


def _topk_rows(values: np.ndarray, start: int, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one dense block to CSR triplets with an explicit tie rule."""
    stop, n_users = values.shape
    rows = []
    cols = []
    data = []
    k = max(0, int(k))
    for offset in range(stop):
        row_id = start + offset
        row = values[offset]
        if 0 <= row_id < n_users:
            row[row_id] = 0.0
        ids = np.flatnonzero(np.isfinite(row) & (row > 0))
        if ids.size == 0 or k == 0:
            continue
        # np.lexsort uses the last key as primary: descending score, then ID.
        order = np.lexsort((ids, -row[ids]))
        chosen = ids[order[:k]]
        rows.extend([row_id] * chosen.size)
        cols.extend(chosen.tolist())
        data.extend(row[chosen].astype(np.float32, copy=False).tolist())
    return (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64),
            np.asarray(data, dtype=np.float32))


def _blocked_weights(left: csr_matrix, right: csr_matrix, k: int,
                     block_size: int, minimum_overlap: Optional[int] = None) -> csr_matrix:
    """Build top-k rows of left @ right.T without materializing user cooc."""
    if left.shape != right.shape:
        raise ValueError("left and right matrices must have equal shapes")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive")
    left_norm = _l2_rows(left)
    right_norm = _l2_rows(right)
    overlap_left = left if minimum_overlap is not None else None
    overlap_right = right if minimum_overlap is not None else None
    row_parts = []
    col_parts = []
    value_parts = []
    for start in range(0, left.shape[0], int(block_size)):
        stop = min(start + int(block_size), left.shape[0])
        values = left_norm[start:stop].dot(right_norm.T).toarray().astype(np.float32, copy=False)
        if minimum_overlap is not None:
            overlap = overlap_left[start:stop].dot(overlap_right.T).toarray()
            values[overlap < int(minimum_overlap)] = 0.0
        r, c, d = _topk_rows(values, start, int(k))
        row_parts.append(r)
        col_parts.append(c)
        value_parts.append(d)
    if not row_parts:
        return csr_matrix((left.shape[0], right.shape[0]), dtype=np.float32)
    rows = np.concatenate(row_parts) if any(x.size for x in row_parts) else np.empty(0, dtype=np.int64)
    cols = np.concatenate(col_parts) if any(x.size for x in col_parts) else np.empty(0, dtype=np.int64)
    data = np.concatenate(value_parts) if any(x.size for x in value_parts) else np.empty(0, dtype=np.float32)
    result = csr_matrix((data, (rows, cols)), shape=(left.shape[0], right.shape[0]), dtype=np.float32)
    result.sort_indices()
    return result


def build_user_knn(train, k: int = 100, block_size: int = 512) -> csr_matrix:
    """Standard cosine UserKNN evidence with deterministic top-k ties."""
    train = _as_binary_csr(train)
    return _blocked_weights(train, train, k, block_size)


def build_overlap_user_knn(train, k: int = 100, minimum_overlap: int = 2,
                           block_size: int = 512) -> csr_matrix:
    """Cosine UserKNN restricted to users sharing ``minimum_overlap`` items."""
    if int(minimum_overlap) < 1:
        raise ValueError("minimum_overlap must be positive")
    train = _as_binary_csr(train)
    return _blocked_weights(train, train, k, block_size, minimum_overlap=int(minimum_overlap))


def _shuffle_rows_within_degree(matrix: csr_matrix, seed: int) -> Tuple[csr_matrix, np.ndarray]:
    """Permute rows only within equal-degree strata; derangement is not forced."""
    matrix = _as_binary_csr(matrix)
    rng = np.random.RandomState(int(seed))
    degrees = np.diff(matrix.indptr)
    permutation = np.arange(matrix.shape[0], dtype=np.int64)
    for degree in np.unique(degrees):
        ids = np.flatnonzero(degrees == degree)
        permutation[ids] = ids[rng.permutation(ids.size)]
    return matrix[permutation].tocsr(), permutation


def build_dual_witness(train, k: int = 100, block_size: int = 512,
                       shuffled: bool = False, seed: int = 2026) -> csr_matrix:
    """Build sqrt(cos(A)*cos(B)) co-user witness evidence.

    For the shuffled control, B rows are permuted within equal B-degree
    strata before its pairwise cosine matrix is formed.  This preserves the
    marginal row-degree distribution and does not force a derangement.
    """
    train = _as_binary_csr(train)
    view_a, view_b = split_item_views(train)
    if shuffled:
        view_b, _ = _shuffle_rows_within_degree(view_b, seed)
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive")
    a_norm = _l2_rows(view_a)
    b_norm = _l2_rows(view_b)
    parts = []
    for start in range(0, train.shape[0], int(block_size)):
        stop = min(start + int(block_size), train.shape[0])
        cos_a = a_norm[start:stop].dot(a_norm.T).toarray().astype(np.float32, copy=False)
        cos_b = b_norm[start:stop].dot(b_norm.T).toarray().astype(np.float32, copy=False)
        values = np.sqrt(np.maximum(cos_a, 0.0) * np.maximum(cos_b, 0.0)).astype(np.float32, copy=False)
        parts.append(_topk_rows(values, start, int(k)))
    rows = np.concatenate([x[0] for x in parts]) if parts else np.empty(0, dtype=np.int64)
    cols = np.concatenate([x[1] for x in parts]) if parts else np.empty(0, dtype=np.int64)
    data = np.concatenate([x[2] for x in parts]) if parts else np.empty(0, dtype=np.float32)
    result = csr_matrix((data, (rows, cols)), shape=(train.shape[0], train.shape[0]), dtype=np.float32)
    result.sort_indices()
    return result


def build_all_evidence(train, k: int = 100, block_size: int = 512,
                       seed: int = 2026) -> Dict[str, csr_matrix]:
    """Build all preregistered evidence controls."""
    train = _as_binary_csr(train)
    return {
        "user_knn": build_user_knn(train, k=k, block_size=block_size),
        "overlap_user_knn": build_overlap_user_knn(train, k=k, minimum_overlap=2,
                                                    block_size=block_size),
        "dual_witness": build_dual_witness(train, k=k, block_size=block_size,
                                             shuffled=False, seed=seed),
        "shuffled_dual": build_dual_witness(train, k=k, block_size=block_size,
                                              shuffled=True, seed=seed),
    }


def build_user_evidence(train, mode: str = "dual", topk: int = 100,
                        block_size: int = 256, seed: int = 2026) -> csr_matrix:
    """Build one preregistered user evidence matrix.

    ``mode`` is one of ``dual``, ``userknn``, ``min2`` and ``shuffled``.
    The returned matrix contains raw cosine/product weights; callers choose
    any normalization used by the model after this fixed construction.
    """
    mode = str(mode).lower()
    if mode == "userknn":
        return build_user_knn(train, k=topk, block_size=block_size)
    if mode == "min2":
        return build_overlap_user_knn(train, k=topk, minimum_overlap=2,
                                      block_size=block_size)
    if mode == "dual":
        return build_dual_witness(train, k=topk, block_size=block_size,
                                  shuffled=False, seed=seed)
    if mode == "shuffled":
        return build_dual_witness(train, k=topk, block_size=block_size,
                                  shuffled=True, seed=seed)
    raise ValueError("unknown evidence mode: " + mode)


def prune_topk(matrix, topk: int) -> csr_matrix:
    """Deterministically retain the largest positive value per sparse row."""
    matrix = sparse.csr_matrix(matrix, dtype=np.float32, copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    rows, cols, data = [], [], []
    for row_id in range(matrix.shape[0]):
        start, stop = matrix.indptr[row_id], matrix.indptr[row_id + 1]
        ids = matrix.indices[start:stop]
        values = matrix.data[start:stop]
        keep = np.flatnonzero(np.isfinite(values) & (values > 0))
        if keep.size == 0 or int(topk) <= 0:
            continue
        order = np.lexsort((ids[keep], -values[keep]))[:int(topk)]
        chosen = keep[order]
        rows.extend([row_id] * chosen.size)
        cols.extend(ids[chosen].tolist())
        data.extend(values[chosen].tolist())
    return csr_matrix((np.asarray(data, dtype=np.float32),
                       (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
                      shape=matrix.shape, dtype=np.float32)


def evidence_scores(weights: csr_matrix, train, users: Sequence[int],
                    items: Sequence[int]) -> np.ndarray:
    """Cartesian gather for small inspection matrices only.

    The result has shape ``(len(users), len(items))`` and is NOT a row-pair
    gather.  Formal paired diagnostics must use :func:`gather_scores`.
    """
    weights = sparse.csr_matrix(weights)
    train = _as_binary_csr(train)
    users = np.asarray(users, dtype=np.int64).reshape(-1)
    items = np.asarray(items, dtype=np.int64)
    if items.ndim != 1:
        raise ValueError("Cartesian items must be one-dimensional; use gather_scores for row pairs")
    if np.any(users < 0) or np.any(users >= train.shape[0]):
        raise IndexError("user index out of range")
    if np.any(items < 0) or np.any(items >= train.shape[1]):
        raise IndexError("item index out of range")
    if weights.shape != (train.shape[0], train.shape[0]):
        raise ValueError("weights must be a square user matrix")
    return weights[users].dot(train[:, items]).toarray().astype(np.float32, copy=False)


def gather_scores(weights: csr_matrix, train, users: Sequence[int], items,
                  block_size: int = 256) -> np.ndarray:
    """Gather raw evidence for each user's own ``items[N, M]``.

    A block of sparse ``W[users] @ R`` is indexed directly.  This never
    materializes a dense user-by-user or user-by-catalog matrix.
    """
    weights = sparse.csr_matrix(weights)
    train = _as_binary_csr(train)
    users = np.asarray(users, dtype=np.int64)
    items = np.asarray(items, dtype=np.int64)
    if users.ndim != 1 or items.ndim != 2 or items.shape[0] != users.size:
        raise ValueError("expected one-dimensional users and items[N, M]")
    if weights.shape != (train.shape[0], train.shape[0]):
        raise ValueError("weights must be a square user matrix")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive")
    if np.any(users < 0) or np.any(users >= train.shape[0]):
        raise IndexError("user index out of range")
    if np.any(items < 0) or np.any(items >= train.shape[1]):
        raise IndexError("item index out of range")
    result = np.zeros(items.shape, dtype=np.float32)
    if items.shape[1] == 0:
        return result
    for start in range(0, users.size, int(block_size)):
        stop = min(start + int(block_size), users.size)
        raw = weights[users[start:stop]].dot(train).tocsr()
        for local, row_items in enumerate(items[start:stop]):
            result[start + local] = np.asarray(
                raw.getrow(local)[:, row_items].toarray(), dtype=np.float32
            ).reshape(-1)
    return result


def shuffle_rows_within_degree(matrix: csr_matrix, seed: int = 2026) -> Tuple[csr_matrix, np.ndarray]:
    """Public test/receipt helper for the B-view permutation."""
    return _shuffle_rows_within_degree(matrix, seed)
