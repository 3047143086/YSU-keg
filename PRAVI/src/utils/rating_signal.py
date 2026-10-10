"""Utilities for diagnosing explicit-rating signal in implicit MMR data."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _as_1d(values, dtype):
    return np.asarray(values, dtype=dtype).reshape(-1)


def build_profile_operators(
    user_ids,
    item_ids,
    ratings,
    n_users,
    n_items,
):
    """Build inverse-popularity exposure and user-centered rating operators.

    The exposure operator follows SG-URInit's inverse-item-degree aggregation.
    The rating operator uses the same weights but removes each user's weighted
    mean rating. Its rows therefore encode within-user polarity rather than a
    user's general tendency to assign high or low scores.
    """
    users = _as_1d(user_ids, np.int64)
    items = _as_1d(item_ids, np.int64)
    values = _as_1d(ratings, np.float32)
    if not (users.size == items.size == values.size):
        raise ValueError("user_ids, item_ids, and ratings must have equal size")
    if users.size == 0:
        shape = (int(n_users), int(n_items))
        empty = sp.csr_matrix(shape, dtype=np.float32)
        return empty, empty.copy(), np.zeros(n_users, dtype=np.float32)
    if users.min() < 0 or users.max() >= n_users:
        raise ValueError("user id outside declared shape")
    if items.min() < 0 or items.max() >= n_items:
        raise ValueError("item id outside declared shape")

    item_degree = np.bincount(items, minlength=n_items).astype(np.float32)
    inverse_degree = np.zeros_like(item_degree)
    observed_items = item_degree > 0
    inverse_degree[observed_items] = 1.0 / item_degree[observed_items]
    exposure_weight = inverse_degree[items]

    user_mass = np.bincount(
        users, weights=exposure_weight, minlength=n_users
    ).astype(np.float32)
    user_rating_sum = np.bincount(
        users, weights=exposure_weight * values, minlength=n_users
    ).astype(np.float32)
    weighted_mean = np.divide(
        user_rating_sum,
        user_mass,
        out=np.zeros_like(user_rating_sum),
        where=user_mass > 0,
    )
    rating_weight = exposure_weight * (values - weighted_mean[users])

    shape = (int(n_users), int(n_items))
    exposure = sp.coo_matrix(
        (exposure_weight, (users, items)), shape=shape, dtype=np.float32
    ).tocsr()
    rating = sp.coo_matrix(
        (rating_weight, (users, items)), shape=shape, dtype=np.float32
    ).tocsr()
    exposure.sum_duplicates()
    rating.sum_duplicates()
    return exposure, rating, weighted_mean


def shuffle_values_within_users(user_ids, values, seed):
    """Shuffle values independently within users while preserving multisets."""
    users = _as_1d(user_ids, np.int64)
    source = np.asarray(values).reshape(-1)
    if users.size != source.size:
        raise ValueError("user_ids and values must have equal size")
    shuffled = source.copy()
    order = np.argsort(users, kind="stable")
    ordered_users = users[order]
    boundaries = np.flatnonzero(np.diff(ordered_users)) + 1
    groups = np.split(order, boundaries)
    rng = np.random.default_rng(int(seed))
    for indices in groups:
        if indices.size > 1:
            shuffled[indices] = source[indices][rng.permutation(indices.size)]
    return shuffled


def l2_normalize_rows(values, eps=1e-12):
    """Return row-normalized float32 values and a nonzero-row mask."""
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1)
    valid = norms > float(eps)
    output = np.zeros_like(array, dtype=np.float32)
    output[valid] = array[valid] / norms[valid, None]
    return output, valid


def event_pairwise_auc(positive_scores, negative_scores, mask=None):
    """Return event-level pairwise AUC values, including half credit for ties."""
    positive = _as_1d(positive_scores, np.float64)
    negative = np.asarray(negative_scores, dtype=np.float64)
    if negative.ndim != 2 or negative.shape[0] != positive.size:
        raise ValueError("negative_scores must have one row per positive score")
    event_auc = (
        (positive[:, None] > negative).mean(axis=1)
        + 0.5 * (positive[:, None] == negative).mean(axis=1)
    )
    if mask is not None:
        event_auc = event_auc[np.asarray(mask, dtype=bool)]
    return event_auc


def mean_and_ci(values):
    """Return a normal-approximation mean and paired 95% confidence interval."""
    array = _as_1d(values, np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    mean = float(array.mean())
    if array.size == 1:
        half_width = 0.0
    else:
        half_width = 1.96 * float(array.std(ddof=1)) / np.sqrt(array.size)
    return {
        "count": int(array.size),
        "mean": mean,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def diagnostic_gate(
    coverage,
    high_rating_auc,
    shuffled_high_rating_auc,
    overall_delta,
    overall_delta_ci_low,
    min_coverage=0.70,
    min_high_auc=0.53,
    min_shuffle_delta=0.02,
    min_overall_delta=0.005,
):
    """Apply the preregistered rating-signal go/no-go thresholds."""
    checks = {
        "coverage": float(coverage) >= float(min_coverage),
        "high_rating_auc": float(high_rating_auc) >= float(min_high_auc),
        "shuffle_delta": (
            float(high_rating_auc) - float(shuffled_high_rating_auc)
            >= float(min_shuffle_delta)
        ),
        "overall_delta": float(overall_delta) >= float(min_overall_delta),
        "overall_delta_ci": float(overall_delta_ci_low) > 0.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_coverage": float(min_coverage),
            "min_high_auc": float(min_high_auc),
            "min_shuffle_delta": float(min_shuffle_delta),
            "min_overall_delta": float(min_overall_delta),
            "overall_delta_ci_low_must_exceed": 0.0,
        },
    }
