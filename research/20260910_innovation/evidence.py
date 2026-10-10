"""Training-edge deletion for cosine ItemKNN evidence, with frozen KNN support.

This is an exact edge-deletion calculation CONDITIONAL ON the original KNN
support. It removes the queried training edge from history, cooccurrence, and
item popularity. It does not reselect neighbours after deleting that edge.
No validation/test interactions are required by this module.

NumPy supports tiny reference cases. SciPy CSR/CSC inputs provide the production
path without allocating an item-by-item matrix per training edge.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _sparse(matrix):
    return hasattr(matrix, "tocsr")


def _dense(matrix):
    return matrix.toarray() if _sparse(matrix) else np.asarray(matrix)


def _sum(matrix, axis):
    # Older SciPy can reduce float32 CSR values before applying sum(dtype=...).
    # Cast BEFORE reducing: tiny column-sum errors corrupt zero-row z-scores.
    return np.asarray(matrix.astype(np.float64, copy=False).sum(axis=axis)).ravel()


def _gather(matrix, rows, columns):
    shape = np.broadcast_shapes(np.shape(rows), np.shape(columns))
    rows = np.broadcast_to(rows, shape).ravel()
    columns = np.broadcast_to(columns, shape).ravel()
    return np.asarray(matrix[rows, columns]).ravel().reshape(shape)


def cosine_similarity(train, topk=1000):
    """Match BRIDGE's dense cosine construction and np.argpartition selection.

    One item-by-item temporary is used while constructing the graph, as in
    BRIDGE. The returned graph is sparse when the training matrix is sparse.
    """
    popularity = _sum(train, 0).astype(np.float32)
    cooccurrence = _dense(train.T @ train).astype(np.float32, copy=False)
    np.fill_diagonal(cooccurrence, 0.0)
    denominator = np.sqrt(np.maximum(popularity, 1e-6))
    similarity = cooccurrence / denominator[:, None] / denominator[None, :]
    np.fill_diagonal(similarity, 0.0)
    n_items = train.shape[1]
    if 0 < int(topk) < n_items:
        indices = np.argpartition(similarity, -int(topk), axis=1)[:, -int(topk):]
        values = np.take_along_axis(similarity, indices, axis=1)
        pruned = np.zeros_like(similarity)
        np.put_along_axis(pruned, indices, values, axis=1)
        similarity = pruned
    if _sparse(train):
        from scipy import sparse
        return sparse.csr_matrix(similarity)
    return similarity


def _validate_train(train):
    train = train.astype(np.float32, copy=True)
    if _sparse(train):
        train = train.tocsr()
        train.sum_duplicates()
        train.eliminate_zeros()
        train.sort_indices()
        values = train.data
    else:
        values = train.ravel()
    if not np.all(np.isfinite(values)) or np.any((values != 0) & (values != 1)):
        raise ValueError("Training interactions must be finite and binary; duplicate counts are invalid")
    if len(train.shape) != 2 or train.shape[1] < 2:
        raise ValueError("A user-item matrix with at least two items is required")
    return train


def _normalise(raw, mean, std, degree):
    scale = 1.0 / np.sqrt(np.maximum(degree, 1.0))
    return (raw - mean) * scale / np.maximum(std * scale, 1e-6)


@dataclass
class CounterfactualCache:
    train: object
    similarity: object
    raw_scores: np.ndarray
    item_popularity: np.ndarray
    user_degree: np.ndarray
    edge_keys: np.ndarray
    edge_users: np.ndarray
    edge_items: np.ndarray
    original_mean: np.ndarray
    original_std: np.ndarray
    history_mean: np.ndarray
    history_std: np.ndarray
    edge_mean: np.ndarray
    edge_std: np.ndarray
    edge_target_raw: np.ndarray
    self_vote_raw: np.ndarray

    def edge_indices(self, users, positive):
        users = np.asarray(users, dtype=np.int64)
        positive = np.asarray(positive, dtype=np.int64)
        if users.ndim != 1 or positive.shape != users.shape:
            raise ValueError("users and positive must be matching one-dimensional arrays")
        if np.any((users < 0) | (users >= self.train.shape[0])):
            raise ValueError("User index outside training matrix")
        if np.any((positive < 0) | (positive >= self.train.shape[1])):
            raise ValueError("Positive item outside training matrix")
        keys = users * self.train.shape[1] + positive
        indices = np.searchsorted(self.edge_keys, keys)
        if np.any(indices >= len(self.edge_keys)):
            raise ValueError("Every queried positive must be an observed training edge")
        if np.any(self.edge_keys[indices] != keys):
            raise ValueError("Every queried positive must be an observed training edge")
        return indices

    def get_pair(self, users, positive, negative, mode="edge"):
        """Return positive/negative z-scores for matching event queries.

        ``edge`` deletes the target edge from history AND cooccurrence;
        ``history`` deletes it only from the query history (ablation);
        ``original`` leaves evidence unchanged (matched baseline).

        Negatives may have shape [batch] or [batch, count]. They must exclude
        every known TRAINING positive, including the temporarily withheld target.
        No validation/test positives are used to filter training negatives.
        """
        users = np.asarray(users, dtype=np.int64)
        positive = np.asarray(positive, dtype=np.int64)
        negative = np.asarray(negative, dtype=np.int64)
        indices = self.edge_indices(users, positive)
        if negative.ndim not in (1, 2) or negative.shape[0] != len(users):
            raise ValueError("negative must have shape [batch] or [batch, count]")
        if np.any((negative < 0) | (negative >= self.train.shape[1])):
            raise ValueError("Negative item outside training matrix")
        query_users = users[:, None] if negative.ndim == 2 else users
        query_positive = positive[:, None] if negative.ndim == 2 else positive
        if np.any(_gather(self.train, query_users, negative) != 0):
            raise ValueError("Negative samples include an observed training positive")
        negative_raw = self.raw_scores[query_users, negative].astype(np.float64)
        positive_raw = self.raw_scores[users, positive].astype(np.float64)
        degree = self.user_degree[users]
        if mode == "original":
            mean, std = self.original_mean[users], self.original_std[users]
        elif mode in ("edge", "history"):
            negative_raw -= _gather(self.similarity, negative, query_positive)
            degree = degree - 1
            if mode == "edge":
                positive_raw = self.edge_target_raw[indices]
                mean, std = self.edge_mean[indices], self.edge_std[indices]
            else:
                mean, std = self.history_mean[indices], self.history_std[indices]
        else:
            raise ValueError("Unknown evidence mode: " + str(mode))
        positive_z = _normalise(positive_raw, mean, std, degree)
        if negative.ndim == 2:
            mean, std, degree = mean[:, None], std[:, None], degree[:, None]
        negative_z = _normalise(negative_raw, mean, std, degree)
        return positive_z.astype(np.float32), negative_z.astype(np.float32)

    def save(self, path):
        """Write a new cache artifact; refuse to overwrite an existing result."""
        payload = {name: value for name, value in vars(self).items()
                   if name not in ("train", "similarity")}
        for name in ("train", "similarity"):
            matrix = getattr(self, name)
            if _sparse(matrix):
                matrix = matrix.tocsr()
                payload[name + "_format"] = np.array("csr")
                for component in ("data", "indices", "indptr"):
                    payload[name + "_" + component] = getattr(matrix, component)
                payload[name + "_shape"] = np.asarray(matrix.shape)
            else:
                payload[name + "_format"] = np.array("dense")
                payload[name] = matrix
        with Path(path).open("xb") as stream:
            np.savez(stream, **payload)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            payload = {name: archive[name] for name in archive.files}
        for name in ("train", "similarity"):
            matrix_format = str(payload.pop(name + "_format"))
            if matrix_format == "csr":
                from scipy import sparse
                parts = [payload.pop(name + "_" + component)
                         for component in ("data", "indices", "indptr")]
                payload[name] = sparse.csr_matrix(tuple(parts), shape=tuple(payload.pop(name + "_shape")))
            elif matrix_format != "dense":
                raise ValueError("Unknown matrix cache format")
        return cls(**payload)


def build_counterfactual_cache(train, topk=1000, similarity=None,
                               raw_scores=None, progress=None):
    """Precompute per-edge moments with sparse column products, not full rows.

    ``similarity`` can be BRIDGE's actual float32 cosine graph. ``raw_scores``,
    when supplied, must be R.astype(float64) @ S.astype(float64).T: casting an
    already accumulated float32 row to float64 is insufficient. Double-precision
    accumulation and self-vote weights S_ij / integer_cooccurrence_ij prevent a
    spurious nonzero row when the held edge was the only evidence. This matters
    because the z-score epsilon floor otherwise amplifies float32 cancellation.
    The output costs one float64 user-item raw score matrix (about 1.10 GB Baby
    or 5.23 GB Sports), one sparse graph, and O(train edges) moment arrays. The
    original float32 model and GPU evidence matrix are unchanged.

    For target t and raw user score b, the counterfactual row is b - S[:, t]
    except at t. At t its value is sqrt(d_t/(d_t-1)) * (b_t - self_vote).
    Row moments follow by a column sum and a sparse dot product. Global z-score
    statistics are recomputed over the complete catalogue, including seen items,
    matching BRIDGE's original normalization convention.
    """
    train = _validate_train(train)
    n_users, n_items = train.shape
    if similarity is None:
        similarity = cosine_similarity(train, topk)
    else:
        similarity = similarity.astype(np.float32, copy=True)
    if similarity.shape != (n_items, n_items):
        raise ValueError("Similarity must have shape [n_items, n_items]")
    if _sparse(similarity):
        similarity = similarity.tocsr()
        similarity.eliminate_zeros()
        similarity.sort_indices()
        values = similarity.data
    else:
        values = similarity.ravel()
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Cosine interaction evidence must be finite and nonnegative")
    if np.any(np.asarray(similarity.diagonal()) != 0):
        raise ValueError("Similarity diagonal must be zero")
    if raw_scores is None:
        raw_scores = _dense(train.astype(np.float64) @ similarity.astype(np.float64).T)
    else:
        raw_scores = np.asarray(raw_scores)
        if raw_scores.dtype != np.float64:
            raise ValueError("raw_scores must be accumulated in float64, not rounded float32 scores")
    if raw_scores.shape != train.shape or not np.all(np.isfinite(raw_scores)):
        raise ValueError("raw_scores must be a finite matrix matching train")
    popularity, user_degree = _sum(train, 0), _sum(train, 1)
    original_mean = np.mean(raw_scores, axis=1, dtype=np.float64)
    original_std = np.std(raw_scores, axis=1, dtype=np.float64)
    column_sum = _sum(similarity, 0)
    if _sparse(similarity):
        columns = similarity.tocsc()
        column_square = _sum(similarity.astype(np.float64).power(2), 0)
        support = similarity.copy().astype(np.float64)
        row_indices = np.repeat(np.arange(n_items), np.diff(support.indptr))
        cooccurrence = (train.T @ train).tocsr()
        counts = np.asarray(cooccurrence[row_indices, support.indices]).ravel()
        if np.any(counts <= 0):
            raise ValueError("Similarity support contains an edge without training cooccurrence")
        support.data /= counts
        del cooccurrence, counts, row_indices
    else:
        columns = similarity
        column_square = np.sum(similarity.astype(np.float64) ** 2, axis=0)
        cooccurrence = train.T @ train
        if np.any((similarity != 0) & (cooccurrence <= 0)):
            raise ValueError("Similarity support contains an edge without training cooccurrence")
        support = np.divide(similarity.astype(np.float64), cooccurrence,
                            out=np.zeros_like(similarity, dtype=np.float64), where=cooccurrence > 0)
        del cooccurrence
    edge_users, edge_items = train.nonzero()
    order = np.lexsort((edge_items, edge_users))
    edge_users = np.asarray(edge_users, dtype=np.int64)[order]
    edge_items = np.asarray(edge_items, dtype=np.int64)[order]
    edge_keys = edge_users * n_items + edge_items
    fields = {name: np.zeros(len(edge_keys), dtype=np.float64) for name in
              ("history_mean", "history_std", "edge_mean", "edge_std",
               "edge_target_raw", "self_vote_raw")}
    offsets = np.searchsorted(edge_users, np.arange(n_users + 1))
    for user in range(n_users):
        begin, end = offsets[user], offsets[user + 1]
        if begin == end:
            continue
        history = edge_items[begin:end]
        raw = raw_scores[user].astype(np.float64)
        selected_columns = columns[:, history]
        cross = np.asarray(selected_columns.T @ raw).ravel()
        target_original = raw[history]
        votes = _sum(support[history][:, history], 1)
        fields["self_vote_raw"][begin:end] = votes
        target_deleted = np.zeros(len(history), dtype=np.float64)
        has_other_users = popularity[history] > 1
        factor = np.sqrt(popularity[history][has_other_users] /
                         (popularity[history][has_other_users] - 1))
        target_deleted[has_other_users] = factor * (
            target_original[has_other_users] - votes[has_other_users])
        target_deleted = np.maximum(target_deleted, 0.0)
        history_sum = raw.sum() - column_sum[history]
        history_square = raw @ raw + column_square[history] - 2 * cross
        delta = target_deleted - target_original
        edge_sum = history_sum + delta
        edge_square = history_square + 2 * delta * target_original + delta ** 2
        fields["history_mean"][begin:end] = history_sum / n_items
        fields["history_std"][begin:end] = np.sqrt(np.maximum(
            history_square / n_items - (history_sum / n_items) ** 2, 0.0))
        fields["edge_mean"][begin:end] = edge_sum / n_items
        fields["edge_std"][begin:end] = np.sqrt(np.maximum(
            edge_square / n_items - (edge_sum / n_items) ** 2, 0.0))
        fields["edge_target_raw"][begin:end] = target_deleted
        if len(history) == 1:
            # An empty history produces an exactly zero row, not roundoff noise.
            for name in ("history_mean", "history_std", "edge_mean", "edge_std", "edge_target_raw"):
                fields[name][begin:end] = 0.0
        if progress is not None and (user + 1) % 1000 == 0:
            progress(user + 1, n_users)
    return CounterfactualCache(train, similarity, raw_scores, popularity,
                               user_degree, edge_keys, edge_users, edge_items,
                               original_mean, original_std, **fields)
