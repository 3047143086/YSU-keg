"""Utilities for candidate-wise multimodal late interaction."""

from __future__ import annotations

import numpy as np
import torch


def build_history_table(user_ids, item_ids, n_users):
    """Pack each user's unique training items into a padded table."""
    users = np.asarray(user_ids, dtype=np.int64).reshape(-1)
    items = np.asarray(item_ids, dtype=np.int64).reshape(-1)
    if users.size != items.size:
        raise ValueError("user_ids and item_ids must have equal length")
    if users.size and (users.min() < 0 or users.max() >= int(n_users)):
        raise ValueError("user_ids contains an out-of-range identifier")

    histories = [[] for _ in range(int(n_users))]
    for user, item in zip(users, items):
        histories[int(user)].append(int(item))
    histories = [np.unique(values).astype(np.int64) for values in histories]
    max_length = max((len(values) for values in histories), default=0)
    if max_length == 0:
        return (
            torch.zeros((int(n_users), 1), dtype=torch.long),
            torch.zeros((int(n_users), 1), dtype=torch.bool),
        )

    table = np.zeros((int(n_users), max_length), dtype=np.int64)
    mask = np.zeros((int(n_users), max_length), dtype=np.bool_)
    for user, values in enumerate(histories):
        if values.size:
            table[user, : values.size] = values
            mask[user, : values.size] = True
    return torch.from_numpy(table), torch.from_numpy(mask)


def topq_history_similarity(
    candidate_items,
    history_items,
    history_mask,
    normalized_item_features,
    top_q=3,
):
    """Average each candidate's strongest valid history matches."""
    if candidate_items.ndim != 2 or history_items.ndim != 2:
        raise ValueError("candidate_items and history_items must be matrices")
    if history_items.shape != history_mask.shape:
        raise ValueError("history_items and history_mask must have equal shape")
    if candidate_items.size(0) != history_items.size(0):
        raise ValueError("candidate and history batch sizes must match")
    if normalized_item_features.ndim != 2:
        raise ValueError("normalized_item_features must be a matrix")

    count = min(int(top_q), int(history_items.size(1)))
    if count <= 0:
        raise ValueError("top_q and history width must be positive")

    candidate_features = normalized_item_features[candidate_items]
    history_features = normalized_item_features[history_items]
    similarities = torch.bmm(
        candidate_features,
        history_features.transpose(1, 2),
    )
    similarities = similarities.masked_fill(
        ~history_mask.unsqueeze(1), float("-inf")
    )
    strongest = torch.topk(similarities, k=count, dim=-1).values

    available = history_mask.sum(dim=1).clamp(max=count)
    valid_rank = torch.arange(count, device=history_mask.device).view(1, 1, -1)
    valid_rank = valid_rank < available.view(-1, 1, 1)
    strongest = torch.where(valid_rank, strongest, torch.zeros_like(strongest))
    return strongest.sum(dim=-1) / available.clamp_min(1).view(-1, 1)


def standardize_candidate_evidence(evidence, eps=1e-6):
    """Remove user-specific evidence location and scale."""
    centered = evidence - evidence.mean(dim=1, keepdim=True)
    scale = evidence.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return centered / scale


def local_gap_scale(candidate_scores, anchor_k=50, eps=1e-8):
    """Estimate one local rank step from sorted candidate scores."""
    width = min(max(int(anchor_k), 2), int(candidate_scores.size(1)))
    head = candidate_scores[:, :width]
    gaps = (head[:, :-1] - head[:, 1:]).clamp_min(0.0)
    median_gap = gaps.median(dim=1).values
    fallback = head.std(dim=1, unbiased=False) / float(max(width - 1, 1))
    return torch.where(median_gap > eps, median_gap, fallback.clamp_min(eps))


def apply_local_evidence(candidate_scores, evidence, anchor_k=50, clip_z=2.0):
    """Apply a bounded, parameter-free residual measured in local rank gaps."""
    if candidate_scores.shape != evidence.shape:
        raise ValueError("candidate_scores and evidence must have equal shape")
    standardized = standardize_candidate_evidence(evidence)
    scale = local_gap_scale(candidate_scores, anchor_k=anchor_k)
    residual = standardized.clamp(-float(clip_z), float(clip_z))
    return candidate_scores + scale.unsqueeze(1) * residual


def per_user_ranking_metrics(hit_matrix, positive_lengths, cutoff):
    """Return unrounded per-user Recall and NDCG at one cutoff."""
    hits = np.asarray(hit_matrix, dtype=np.bool_)
    lengths = np.asarray(positive_lengths, dtype=np.int64).reshape(-1)
    cutoff = min(int(cutoff), hits.shape[1])
    if hits.ndim != 2 or hits.shape[0] != lengths.size or cutoff <= 0:
        raise ValueError("invalid hit matrix, positive lengths, or cutoff")

    selected = hits[:, :cutoff]
    recall = selected.sum(axis=1) / np.maximum(lengths, 1)
    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2, dtype=np.float64))
    dcg = (selected * discounts.reshape(1, -1)).sum(axis=1)
    ideal_count = np.minimum(lengths, cutoff)
    cumulative_discount = np.cumsum(discounts)
    idcg = cumulative_discount[np.maximum(ideal_count, 1) - 1]
    ndcg = dcg / idcg
    return recall.astype(np.float64), ndcg.astype(np.float64)


def mean_ci95(values):
    """Normal-approximation confidence interval for paired user deltas."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("at least one value is required")
    mean = float(array.mean())
    if array.size == 1:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    half_width = 1.96 * float(array.std(ddof=1)) / np.sqrt(array.size)
    return {
        "mean": mean,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def validation_gate(baseline, treatment, tolerance=0.0002):
    """Apply the preregistered ECLI validation-only screening rule."""
    deltas = {
        key: float(treatment[key]) - float(baseline[key])
        for key in ("recall@10", "recall@20", "ndcg@10", "ndcg@20")
    }
    checks = {
        "recall20_positive": deltas["recall@20"] > 0.0,
        "ndcg20_positive": deltas["ndcg@20"] > 0.0,
        "recall10_guard": deltas["recall@10"] >= -float(tolerance),
        "ndcg10_guard": deltas["ndcg@10"] >= -float(tolerance),
    }
    return {"passed": all(checks.values()), "checks": checks, "deltas": deltas}
