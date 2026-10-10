"""Degree-preserving virtual transport on bipartite interactions."""

from __future__ import annotations

from numbers import Integral
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


def _binary_csr(matrix: sp.spmatrix, name: str) -> sp.csr_matrix:
    matrix = matrix.tocsr().astype(np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if matrix.data.size and not np.all(matrix.data == 1.0):
        raise ValueError(f"{name} must be binary")
    return matrix


def _positive_csr(matrix: sp.spmatrix, name: str) -> sp.csr_matrix:
    matrix = matrix.tocsr().astype(np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if matrix.data.size and (
        not np.all(np.isfinite(matrix.data)) or not np.all(matrix.data > 0.0)
    ):
        raise ValueError(f"{name} must contain finite positive values")
    return matrix


def _align_observed_costs(
    observed: sp.csr_matrix, costs: sp.spmatrix
) -> np.ndarray:
    """Align a sparse cost matrix to observed CSR edges and reject leakage."""
    bounds = costs.tocsr().astype(np.float64, copy=True)
    bounds.sum_duplicates()
    bounds.sort_indices()
    if bounds.shape != observed.shape:
        raise ValueError("donor_upper_bounds must match observed shape")
    if bounds.data.size and (
        not np.all(np.isfinite(bounds.data))
        or np.any(bounds.data < 0.0)
        or np.any(bounds.data > 1.0)
    ):
        raise ValueError("donor_upper_bounds must lie in [0, 1]")

    n_items = observed.shape[1]
    observed_rows = np.repeat(
        np.arange(observed.shape[0], dtype=np.int64), np.diff(observed.indptr)
    )
    observed_keys = observed_rows * n_items + observed.indices.astype(
        np.int64, copy=False
    )
    bounds_coo = bounds.tocoo()
    bound_keys = bounds_coo.row.astype(np.int64) * n_items + bounds_coo.col
    positions = np.searchsorted(observed_keys, bound_keys)
    valid = positions < observed_keys.size
    valid[valid] &= observed_keys[positions[valid]] == bound_keys[valid]
    if not np.all(valid):
        raise ValueError("donor_upper_bounds may only score observed edges")
    aligned = np.zeros(observed.nnz, dtype=np.float64)
    aligned[positions] = bounds_coo.data
    return aligned


def build_degree_preserving_transport(
    observed: sp.spmatrix,
    candidates: sp.spmatrix,
    max_swaps_per_user: int = 1,
    weight_mode: str = "unit",
) -> Tuple[sp.csr_matrix, Dict[str, object]]:
    """Greedily match reciprocal alternating 2-switches.

    A matched switch replaces observed edges ``(u, i), (v, j)`` with
    candidate edges ``(u, j), (v, i)``. Each input edge is used at most once.
    In confidence-weighted mode, the common switch weight is the lower of its
    two candidate confidences so the signed row and column mass still cancels.
    """
    if (
        isinstance(max_swaps_per_user, (bool, np.bool_))
        or not isinstance(max_swaps_per_user, Integral)
        or max_swaps_per_user < 1
    ):
        raise ValueError("max_swaps_per_user must be a positive integer")
    if weight_mode not in {"unit", "confidence"}:
        raise ValueError("weight_mode must be 'unit' or 'confidence'")

    train = _binary_csr(observed, "observed")
    proposal = _positive_csr(candidates, "candidates")
    if train.shape != proposal.shape:
        raise ValueError("observed and candidates must have the same shape")
    if train.multiply(proposal).nnz:
        raise ValueError("candidate edges must be unobserved")

    n_users, _ = train.shape
    observed_rows = np.repeat(
        np.arange(n_users, dtype=np.int64), np.diff(train.indptr)
    )
    observed_cols = train.indices.astype(np.int64, copy=False)
    candidate_rows = np.repeat(
        np.arange(n_users, dtype=np.int64), np.diff(proposal.indptr)
    )
    candidate_cols = proposal.indices.astype(np.int64, copy=False)
    confidence = proposal.data

    candidate_lookup = {
        (int(user), int(item)): edge
        for edge, (user, item) in enumerate(zip(candidate_rows, candidate_cols))
    }
    item_observations = [[] for _ in range(train.shape[1])]
    for edge, (user, item) in enumerate(zip(observed_rows, observed_cols)):
        item_observations[int(item)].append((int(user), edge))

    used_observed = np.zeros(train.nnz, dtype=bool)
    used_candidates = np.zeros(proposal.nnz, dtype=bool)
    order = np.lexsort((candidate_cols, candidate_rows, -confidence))
    delta_rows = []
    delta_cols = []
    delta_values = []
    selected_confidences = []
    switch_weights = []
    switched_users = set()
    user_swap_count = np.zeros(n_users, dtype=np.int64)
    swaps = 0

    for first_candidate in order:
        first_candidate = int(first_candidate)
        if used_candidates[first_candidate]:
            continue
        user = int(candidate_rows[first_candidate])
        if user_swap_count[user] >= max_swaps_per_user:
            continue
        new_item = int(candidate_cols[first_candidate])
        best = None

        for first_observed in range(train.indptr[user], train.indptr[user + 1]):
            if used_observed[first_observed]:
                continue
            old_item = int(train.indices[first_observed])
            for other_user, second_observed in item_observations[new_item]:
                if (
                    other_user == user
                    or user_swap_count[other_user] >= max_swaps_per_user
                    or used_observed[second_observed]
                ):
                    continue
                second_candidate = candidate_lookup.get((other_user, old_item))
                if second_candidate is None or used_candidates[second_candidate]:
                    continue
                key = (
                    -float(confidence[second_candidate]),
                    old_item,
                    other_user,
                    int(second_candidate),
                )
                if best is None or key < best[0]:
                    best = (
                        key,
                        first_observed,
                        second_observed,
                        int(second_candidate),
                        other_user,
                        old_item,
                    )

        if best is None:
            continue
        (
            _,
            first_observed,
            second_observed,
            second_candidate,
            other_user,
            old_item,
        ) = best
        weight = (
            1.0
            if weight_mode == "unit"
            else min(
                float(confidence[first_candidate]),
                float(confidence[second_candidate]),
            )
        )

        used_observed[[first_observed, second_observed]] = True
        used_candidates[[first_candidate, second_candidate]] = True
        user_swap_count[[user, other_user]] += 1
        delta_rows.extend([user, other_user, user, other_user])
        delta_cols.extend([new_item, old_item, old_item, new_item])
        delta_values.extend([weight, weight, -weight, -weight])
        selected_confidences.extend(
            [float(confidence[first_candidate]), float(confidence[second_candidate])]
        )
        switch_weights.append(weight)
        switched_users.update((user, other_user))
        swaps += 1

    delta = sp.coo_matrix(
        (delta_values, (delta_rows, delta_cols)),
        shape=train.shape,
        dtype=np.float64,
    ).tocsr()
    delta.sum_duplicates()
    delta.eliminate_zeros()
    delta.sort_indices()

    row_residual = np.asarray(delta.sum(axis=1)).ravel()
    column_residual = np.asarray(delta.sum(axis=0)).ravel()
    row_residual_max = (
        float(np.max(np.abs(row_residual))) if row_residual.size else 0.0
    )
    column_residual_max = (
        float(np.max(np.abs(column_residual))) if column_residual.size else 0.0
    )
    if weight_mode == "unit" and (
        row_residual_max != 0.0 or column_residual_max != 0.0
    ):
        raise RuntimeError("unit-weight DVT failed to preserve degrees exactly")

    selected = np.asarray(selected_confidences, dtype=np.float64)
    weights = np.asarray(switch_weights, dtype=np.float64)
    confidence_stats = {
        "mean": float(selected.mean()) if selected.size else 0.0,
        "min": float(selected.min()) if selected.size else 0.0,
        "max": float(selected.max()) if selected.size else 0.0,
        "std": float(selected.std()) if selected.size else 0.0,
    }
    used_candidate_count = int(used_candidates.sum())
    stats: Dict[str, object] = {
        "method": "Degree-preserving Virtual Transport",
        "swaps": int(swaps),
        "users": int(len(switched_users)),
        "user_coverage": float(len(switched_users) / max(n_users, 1)),
        "positive_nnz": int(np.count_nonzero(delta.data > 0.0)),
        "negative_nnz": int(np.count_nonzero(delta.data < 0.0)),
        "positive_mass": float(delta.data[delta.data > 0.0].sum()),
        "negative_mass_abs": float(-delta.data[delta.data < 0.0].sum()),
        "candidate_edges": int(proposal.nnz),
        "used_candidate_edges": used_candidate_count,
        "used_observed_edges": int(used_observed.sum()),
        "candidate_coverage": float(used_candidate_count / max(proposal.nnz, 1)),
        "row_residual_max": row_residual_max,
        "column_residual_max": column_residual_max,
        "confidence": confidence_stats,
        "confidence_mean": confidence_stats["mean"],
        "confidence_min": confidence_stats["min"],
        "confidence_max": confidence_stats["max"],
        "confidence_std": confidence_stats["std"],
        "switch_weight_mean": float(weights.mean()) if weights.size else 0.0,
        "switch_weight_min": float(weights.min()) if weights.size else 0.0,
        "switch_weight_max": float(weights.max()) if weights.size else 0.0,
        "max_swaps_per_user": int(max_swaps_per_user),
        "weight_mode": weight_mode,
    }
    return delta, stats


def build_evidence_cycle_transport(
    observed: sp.spmatrix,
    candidates: sp.spmatrix,
    weight_mode: str = "unit",
    min_arc_confidence: float = 0.0,
    donor_upper_bounds: Optional[sp.spmatrix] = None,
    min_net_margin: float = 0.0,
) -> Tuple[sp.csr_matrix, Dict[str, object]]:
    """Find a maximum-confidence partial cycle cover over candidate transfers.

    A directed arc ``u -> v`` means that user ``u`` receives a candidate item
    currently observed for user ``v``. If donor upper bounds are supplied, an
    arc is eligible only when its receiver LCB exceeds the donor UCB by more
    than ``min_net_margin``. A sparse assignment with identity dummy arcs then
    selects a partial permutation of users. Its non-identity arcs decompose into
    disjoint cycles, so every selected user receives and relinquishes one edge
    while every selected item loses and gains one edge.
    """
    if weight_mode not in {"unit", "confidence"}:
        raise ValueError("weight_mode must be 'unit' or 'confidence'")
    if not np.isfinite(min_arc_confidence) or min_arc_confidence < 0.0:
        raise ValueError("min_arc_confidence must be finite and non-negative")
    if not np.isfinite(min_net_margin) or min_net_margin < 0.0:
        raise ValueError("min_net_margin must be finite and non-negative")

    train = _binary_csr(observed, "observed")
    proposal = _positive_csr(candidates, "candidates")
    if train.shape != proposal.shape:
        raise ValueError("observed and candidates must have the same shape")
    if train.multiply(proposal).nnz:
        raise ValueError("candidate edges must be unobserved")

    n_users, n_items = train.shape
    candidate_rows = np.repeat(
        np.arange(n_users, dtype=np.int64), np.diff(proposal.indptr)
    )
    candidate_cols = proposal.indices.astype(np.int64, copy=False)
    item_observations = [[] for _ in range(n_items)]
    observed_rows = np.repeat(
        np.arange(n_users, dtype=np.int64), np.diff(train.indptr)
    )
    for edge, (user, item) in enumerate(zip(observed_rows, train.indices)):
        item_observations[int(item)].append((int(user), int(edge)))
    donor_aware = donor_upper_bounds is not None
    donor_costs = (
        _align_observed_costs(train, donor_upper_bounds)
        if donor_aware
        else np.zeros(train.nnz, dtype=np.float64)
    )

    # Keep the strongest item-level explanation for each directed user pair.
    best_arcs = {}
    considered_owner_arcs = 0
    donor_rejected_owner_arcs = 0
    for candidate_edge, (source, item, confidence) in enumerate(
        zip(candidate_rows, candidate_cols, proposal.data)
    ):
        source = int(source)
        item = int(item)
        confidence = float(confidence)
        if confidence < min_arc_confidence:
            continue
        for target, observed_edge in item_observations[item]:
            if target == source:
                continue
            considered_owner_arcs += 1
            donor_cost = float(donor_costs[observed_edge])
            net_margin = confidence - donor_cost
            if donor_aware and net_margin <= float(min_net_margin):
                donor_rejected_owner_arcs += 1
                continue
            arc_score = net_margin if donor_aware else confidence
            key = source * n_users + target
            value = (
                arc_score,
                confidence,
                -donor_cost,
                -item,
                -candidate_edge,
                -observed_edge,
            )
            current = best_arcs.get(key)
            if current is None or value > current:
                best_arcs[key] = value

    diagonal = np.arange(n_users, dtype=np.int64)
    if best_arcs:
        keys = np.fromiter(best_arcs.keys(), dtype=np.int64)
        arc_rows = keys // n_users
        arc_cols = keys % n_users
        arc_score = np.fromiter(
            (value[0] for value in best_arcs.values()), dtype=np.float64
        )
        # All costs stay strictly positive because sparse matching drops zeros.
        arc_cost = 1.0 - arc_score / (1.0 + arc_score)
        rows = np.concatenate([diagonal, arc_rows])
        cols = np.concatenate([diagonal, arc_cols])
        costs = np.concatenate([np.ones(n_users), arc_cost])
    else:
        rows = diagonal
        cols = diagonal
        costs = np.ones(n_users, dtype=np.float64)

    assignment_graph = sp.coo_matrix(
        (costs, (rows, cols)), shape=(n_users, n_users)
    ).tocsr()
    matched_rows, matched_cols = min_weight_full_bipartite_matching(
        assignment_graph
    )
    successor = np.full(n_users, -1, dtype=np.int64)
    successor[matched_rows] = matched_cols
    if np.any(successor < 0):
        raise RuntimeError("cycle transport did not produce a full assignment")

    visited = np.zeros(n_users, dtype=bool)
    cycles = []
    for start in range(n_users):
        if visited[start] or successor[start] == start:
            visited[start] = True
            continue
        cycle = []
        user = start
        while not visited[user]:
            visited[user] = True
            cycle.append(int(user))
            user = int(successor[user])
        if user != start or len(cycle) < 2:
            raise RuntimeError("non-identity assignment did not form a cycle")
        cycles.append(cycle)

    delta_rows = []
    delta_cols = []
    delta_values = []
    selected_confidences = []
    selected_donor_costs = []
    selected_net_margins = []
    selected_candidate_edges = []
    selected_observed_edges = []
    cycle_weights = []
    for cycle in cycles:
        arc_values = [
            best_arcs[user * n_users + int(successor[user])] for user in cycle
        ]
        weight = (
            1.0
            if weight_mode == "unit"
            else min(value[1] for value in arc_values)
        )
        cycle_weights.append(weight)
        for source, target, value in zip(
            cycle, (int(successor[user]) for user in cycle), arc_values
        ):
            (
                _,
                confidence,
                negative_donor_cost,
                negative_item,
                negative_candidate,
                negative_observed_edge,
            ) = value
            item = -int(negative_item)
            candidate_edge = -int(negative_candidate)
            observed_edge = -int(negative_observed_edge)
            donor_cost = -float(negative_donor_cost)
            delta_rows.extend([source, target])
            delta_cols.extend([item, item])
            delta_values.extend([weight, -weight])
            selected_confidences.append(float(confidence))
            selected_donor_costs.append(donor_cost)
            selected_net_margins.append(float(confidence) - donor_cost)
            selected_candidate_edges.append(candidate_edge)
            selected_observed_edges.append(int(observed_edge))

    delta = sp.coo_matrix(
        (delta_values, (delta_rows, delta_cols)),
        shape=train.shape,
        dtype=np.float64,
    ).tocsr()
    delta.sum_duplicates()
    delta.eliminate_zeros()
    delta.sort_indices()

    row_residual = np.asarray(delta.sum(axis=1)).ravel()
    column_residual = np.asarray(delta.sum(axis=0)).ravel()
    row_residual_max = (
        float(np.max(np.abs(row_residual))) if row_residual.size else 0.0
    )
    column_residual_max = (
        float(np.max(np.abs(column_residual))) if column_residual.size else 0.0
    )
    tolerance = 0.0 if weight_mode == "unit" else 1e-12
    if row_residual_max > tolerance or column_residual_max > tolerance:
        raise RuntimeError("cycle transport failed to preserve degrees")

    selected = np.asarray(selected_confidences, dtype=np.float64)
    selected_donors = np.asarray(selected_donor_costs, dtype=np.float64)
    selected_margins = np.asarray(selected_net_margins, dtype=np.float64)
    lengths = np.asarray([len(cycle) for cycle in cycles], dtype=np.int64)
    weights = np.asarray(cycle_weights, dtype=np.float64)
    used_candidate_count = int(len(set(selected_candidate_edges)))
    used_observed_count = int(len(set(selected_observed_edges)))
    length_histogram = {
        str(int(length)): int(np.count_nonzero(lengths == length))
        for length in np.unique(lengths)
    }
    confidence_stats = {
        "mean": float(selected.mean()) if selected.size else 0.0,
        "min": float(selected.min()) if selected.size else 0.0,
        "max": float(selected.max()) if selected.size else 0.0,
        "std": float(selected.std()) if selected.size else 0.0,
    }
    donor_stats = {
        "mean": float(selected_donors.mean()) if selected_donors.size else 0.0,
        "min": float(selected_donors.min()) if selected_donors.size else 0.0,
        "max": float(selected_donors.max()) if selected_donors.size else 0.0,
        "std": float(selected_donors.std()) if selected_donors.size else 0.0,
    }
    margin_stats = {
        "sum": float(selected_margins.sum()) if selected_margins.size else 0.0,
        "mean": float(selected_margins.mean()) if selected_margins.size else 0.0,
        "min": float(selected_margins.min()) if selected_margins.size else 0.0,
        "max": float(selected_margins.max()) if selected_margins.size else 0.0,
        "std": float(selected_margins.std()) if selected_margins.size else 0.0,
    }
    transported_users = int(lengths.sum()) if lengths.size else 0
    stats: Dict[str, object] = {
        "method": (
            "Donor-Aware Net-Utility Cycle Transport"
            if donor_aware
            else "Evidence-Constrained Cycle Transport"
        ),
        "cycles": int(len(cycles)),
        "transport_arcs": transported_users,
        "users": transported_users,
        "user_coverage": float(transported_users / max(n_users, 1)),
        "cycle_length_mean": float(lengths.mean()) if lengths.size else 0.0,
        "cycle_length_min": int(lengths.min()) if lengths.size else 0,
        "cycle_length_max": int(lengths.max()) if lengths.size else 0,
        "cycle_length_histogram": length_histogram,
        "candidate_graph_arcs": int(len(best_arcs)),
        "considered_owner_arcs": int(considered_owner_arcs),
        "donor_rejected_owner_arcs": int(donor_rejected_owner_arcs),
        "candidate_edges": int(proposal.nnz),
        "used_candidate_edges": used_candidate_count,
        "used_observed_edges": used_observed_count,
        "candidate_coverage": float(
            used_candidate_count / max(proposal.nnz, 1)
        ),
        "positive_nnz": int(np.count_nonzero(delta.data > 0.0)),
        "negative_nnz": int(np.count_nonzero(delta.data < 0.0)),
        "positive_mass": float(delta.data[delta.data > 0.0].sum()),
        "negative_mass_abs": float(-delta.data[delta.data < 0.0].sum()),
        "row_residual_max": row_residual_max,
        "column_residual_max": column_residual_max,
        "confidence": confidence_stats,
        "donor_upper_bound": donor_stats,
        "net_margin": margin_stats,
        "cycle_weight_mean": float(weights.mean()) if weights.size else 0.0,
        "cycle_weight_min": float(weights.min()) if weights.size else 0.0,
        "cycle_weight_max": float(weights.max()) if weights.size else 0.0,
        "weight_mode": weight_mode,
        "min_arc_confidence": float(min_arc_confidence),
        "donor_aware": bool(donor_aware),
        "min_net_margin": float(min_net_margin),
    }
    return delta, stats


def build_signed_transport_operator(
    observed: sp.spmatrix,
    delta: sp.spmatrix,
    device: object = "cpu",
) -> torch.Tensor:
    """Normalize signed delta by observed degrees and extend it symmetrically."""
    train = _binary_csr(observed, "observed")
    signed = delta.tocoo().astype(np.float64)
    signed.sum_duplicates()
    signed.eliminate_zeros()
    if train.shape != signed.shape:
        raise ValueError("observed and delta must have the same shape")
    if signed.data.size and not np.all(np.isfinite(signed.data)):
        raise ValueError("delta must contain finite values")

    user_degree = np.asarray(train.sum(axis=1)).ravel()
    item_degree = np.asarray(train.sum(axis=0)).ravel()
    denominator = np.sqrt(user_degree[signed.row] * item_degree[signed.col])
    if denominator.size and np.any(denominator == 0.0):
        raise ValueError("delta touches a zero-degree observed node")
    values = signed.data / denominator
    rows = np.concatenate([signed.row, train.shape[0] + signed.col])
    cols = np.concatenate([train.shape[0] + signed.col, signed.row])
    symmetric_values = np.concatenate([values, values])
    indices = torch.as_tensor(np.stack([rows, cols]), dtype=torch.long)
    weights = torch.as_tensor(symmetric_values, dtype=torch.float32)
    size = train.shape[0] + train.shape[1]
    return torch.sparse_coo_tensor(
        indices, weights, (size, size), device=device
    ).coalesce()
