"""Cross-fitted collaborative evidence calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_1d(values, dtype):
    return np.asarray(values, dtype=dtype).reshape(-1)


def weighted_isotonic(values, weights):
    """Return the weighted non-decreasing PAVA fit of a one-dimensional curve."""
    y = _as_1d(values, np.float64)
    w = _as_1d(weights, np.float64)
    if y.size != w.size:
        raise ValueError("values and weights must have equal size")
    if y.size == 0:
        return y.astype(np.float32)
    if np.any(w <= 0):
        raise ValueError("isotonic weights must be positive")

    block_values = []
    block_weights = []
    block_starts = []
    block_stops = []
    for index, (value, weight) in enumerate(zip(y, w)):
        block_values.append(float(value))
        block_weights.append(float(weight))
        block_starts.append(index)
        block_stops.append(index + 1)
        while (
            len(block_values) >= 2
            and block_values[-2] > block_values[-1]
        ):
            merged_weight = block_weights[-2] + block_weights[-1]
            merged_value = (
                block_values[-2] * block_weights[-2]
                + block_values[-1] * block_weights[-1]
            ) / merged_weight
            block_values[-2:] = [merged_value]
            block_weights[-2:] = [merged_weight]
            block_stops[-2:] = [block_stops[-1]]
            block_starts.pop()

    fitted = np.empty_like(y)
    for value, start, stop in zip(
        block_values, block_starts, block_stops
    ):
        fitted[start:stop] = value
    return fitted.astype(np.float32)


def quantile_edges(values, n_bins):
    """Build unique finite interior quantile edges."""
    array = _as_1d(values, np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0 or int(n_bins) <= 1:
        return np.empty(0, dtype=np.float32)
    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1]
    return np.unique(np.quantile(array, quantiles)).astype(np.float32)


def make_degree_edges(item_degrees, n_bins):
    degrees = _as_1d(item_degrees, np.float64)
    observed = degrees[degrees > 0]
    return quantile_edges(np.log1p(observed), n_bins)


def assign_degree_bins(item_degrees, degree_edges):
    degrees = _as_1d(item_degrees, np.float64)
    edges = _as_1d(degree_edges, np.float64)
    return np.searchsorted(edges, np.log1p(degrees), side="right").astype(
        np.int16
    )


def _llr_curve(positive_counts, negative_counts, alpha):
    positive = _as_1d(positive_counts, np.float64) + float(alpha)
    negative = _as_1d(negative_counts, np.float64) + float(alpha)
    positive /= positive.sum()
    negative /= negative.sum()
    raw = np.log(positive) - np.log(negative)
    return weighted_isotonic(raw, positive_counts + negative_counts + 2 * alpha)


@dataclass(frozen=True)
class EvidenceCalibrator:
    """Degree-conditioned monotone semantic-to-collaborative LLR map."""

    similarity_edges: np.ndarray
    degree_edges: np.ndarray
    global_curve: np.ndarray
    cell_curves: np.ndarray
    cell_support: np.ndarray

    @classmethod
    def fit(
        cls,
        positive_similarity,
        negative_similarity,
        positive_left,
        positive_right,
        negative_left,
        negative_right,
        item_degrees,
        degree_edges,
        n_similarity_bins=16,
        alpha=1.0,
        shrinkage=200.0,
        clip=4.0,
    ):
        pos_sim = _as_1d(positive_similarity, np.float32)
        neg_sim = _as_1d(negative_similarity, np.float32)
        pos_left = _as_1d(positive_left, np.int64)
        pos_right = _as_1d(positive_right, np.int64)
        neg_left = _as_1d(negative_left, np.int64)
        neg_right = _as_1d(negative_right, np.int64)
        if pos_sim.size == 0 or neg_sim.size == 0:
            raise ValueError("both positive and negative calibration pairs are required")
        if not (pos_sim.size == pos_left.size == pos_right.size):
            raise ValueError("positive pair arrays must have equal size")
        if not (neg_sim.size == neg_left.size == neg_right.size):
            raise ValueError("negative pair arrays must have equal size")

        similarity_edges = quantile_edges(
            np.concatenate((pos_sim, neg_sim)), n_similarity_bins
        )
        n_similarity_cells = similarity_edges.size + 1
        degrees = _as_1d(item_degrees, np.float64)
        degree_edges = _as_1d(degree_edges, np.float32)
        degree_bin = assign_degree_bins(degrees, degree_edges)
        n_degree_cells = degree_edges.size + 1

        def cell_ids(left, right):
            left_bin = degree_bin[left]
            right_bin = degree_bin[right]
            low = np.minimum(left_bin, right_bin).astype(np.int64)
            high = np.maximum(left_bin, right_bin).astype(np.int64)
            return low * n_degree_cells + high

        pos_similarity_bin = np.searchsorted(
            similarity_edges, pos_sim, side="right"
        )
        neg_similarity_bin = np.searchsorted(
            similarity_edges, neg_sim, side="right"
        )
        pos_cell = cell_ids(pos_left, pos_right)
        neg_cell = cell_ids(neg_left, neg_right)
        count_shape = (n_degree_cells * n_degree_cells, n_similarity_cells)
        positive_counts = np.zeros(count_shape, dtype=np.float64)
        negative_counts = np.zeros(count_shape, dtype=np.float64)
        np.add.at(positive_counts, (pos_cell, pos_similarity_bin), 1.0)
        np.add.at(negative_counts, (neg_cell, neg_similarity_bin), 1.0)

        global_positive = positive_counts.sum(axis=0)
        global_negative = negative_counts.sum(axis=0)
        global_curve = _llr_curve(global_positive, global_negative, alpha)
        global_curve = np.clip(global_curve, -float(clip), float(clip))

        curves = np.broadcast_to(
            global_curve, (count_shape[0], n_similarity_cells)
        ).copy()
        support = positive_counts.sum(axis=1) + negative_counts.sum(axis=1)
        for cell in np.flatnonzero(support):
            local_curve = _llr_curve(
                positive_counts[cell], negative_counts[cell], alpha
            )
            reliability = support[cell] / (support[cell] + float(shrinkage))
            blended = reliability * local_curve + (1.0 - reliability) * global_curve
            curves[cell] = weighted_isotonic(
                blended,
                positive_counts[cell] + negative_counts[cell] + 2 * alpha,
            )
        curves = np.clip(curves, -float(clip), float(clip)).astype(np.float32)
        return cls(
            similarity_edges=similarity_edges,
            degree_edges=degree_edges,
            global_curve=global_curve.astype(np.float32),
            cell_curves=curves.reshape(
                n_degree_cells, n_degree_cells, n_similarity_cells
            ),
            cell_support=support.reshape(n_degree_cells, n_degree_cells).astype(
                np.int64
            ),
        )

    def transform(
        self,
        similarity,
        left_items,
        right_items,
        item_degrees,
        degree_conditioned=True,
    ):
        similarity = np.asarray(similarity, dtype=np.float32)
        shape = similarity.shape
        flat_similarity = similarity.reshape(-1)
        similarity_bin = np.searchsorted(
            self.similarity_edges, flat_similarity, side="right"
        )
        if not degree_conditioned:
            return self.global_curve[similarity_bin].reshape(shape)

        left = _as_1d(left_items, np.int64)
        right = _as_1d(right_items, np.int64)
        if left.size != flat_similarity.size or right.size != flat_similarity.size:
            raise ValueError("one item pair is required per similarity value")
        degree_bin = assign_degree_bins(item_degrees, self.degree_edges)
        left_bin = degree_bin[left]
        right_bin = degree_bin[right]
        low = np.minimum(left_bin, right_bin)
        high = np.maximum(left_bin, right_bin)
        return self.cell_curves[low, high, similarity_bin].reshape(shape)

    def summary(self):
        return {
            "similarity_edges": self.similarity_edges.tolist(),
            "degree_edges_log1p": self.degree_edges.tolist(),
            "global_curve": self.global_curve.tolist(),
            "cell_support": self.cell_support.tolist(),
        }


def topq_mean(values, q):
    """Average the largest q values in each candidate-by-history row."""
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("values must be a non-empty two-dimensional matrix")
    count = min(int(q), matrix.shape[1])
    if count <= 0:
        raise ValueError("q must be positive")
    if count == matrix.shape[1]:
        return matrix.mean(axis=1)
    return np.partition(matrix, matrix.shape[1] - count, axis=1)[:, -count:].mean(
        axis=1
    )


def deterministic_user_folds(user_ids, n_folds, seed):
    """Assign users to stable folds without depending on Python hash state."""
    users = _as_1d(user_ids, np.uint64)
    folds = int(n_folds)
    if folds < 2:
        raise ValueError("cross-fitting requires at least two folds")
    with np.errstate(over="ignore"):
        mixed = users + np.uint64(int(seed)) + np.uint64(0x9E3779B97F4A7C15)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
        mixed ^= mixed >> np.uint64(31)
    return np.asarray(mixed % np.uint64(folds), dtype=np.int16)


def diagnostic_gate(
    c3e_auc,
    exposure_delta,
    exposure_delta_ci_low,
    raw_topq_delta,
    raw_topq_delta_ci_low,
    fold_deltas,
    min_auc=0.6192,
    min_exposure_delta=0.02,
    min_raw_topq_delta=0.005,
):
    """Apply the preregistered signal-screen thresholds."""
    fold_deltas = _as_1d(fold_deltas, np.float64)
    checks = {
        "absolute_auc": float(c3e_auc) >= float(min_auc),
        "exposure_delta": float(exposure_delta) >= float(min_exposure_delta),
        "exposure_delta_ci": float(exposure_delta_ci_low) > 0.0,
        "raw_topq_delta": float(raw_topq_delta) >= float(min_raw_topq_delta),
        "raw_topq_delta_ci": float(raw_topq_delta_ci_low) > 0.0,
        "fold_consistency": bool(
            fold_deltas.size > 0 and np.all(fold_deltas > 0.0)
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_auc": float(min_auc),
            "min_exposure_delta": float(min_exposure_delta),
            "min_raw_topq_delta": float(min_raw_topq_delta),
            "paired_ci_low_must_exceed": 0.0,
            "every_fold_delta_must_exceed": 0.0,
        },
    }
