"""Cross-fitted, uncertainty-aware virtual interaction construction."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp


EPS = 1e-12


def torch_graph_to_scipy(graph) -> sp.csr_matrix:
    """Convert a dense or sparse torch item graph to SciPy CSR."""
    if graph.is_sparse:
        graph = graph.coalesce().detach().cpu()
        indices = graph.indices().numpy()
        values = graph.values().float().numpy()
        matrix = sp.coo_matrix(
            (values, (indices[0], indices[1])), shape=tuple(graph.shape)
        )
    else:
        matrix = sp.csr_matrix(graph.detach().cpu().float().numpy())
    matrix = matrix.tocsr().astype(np.float32)
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    return matrix


def sparse_fingerprint(matrix: sp.spmatrix) -> str:
    """Return a stable fingerprint for cache invalidation."""
    matrix = matrix.tocsr().astype(np.float32)
    digest = hashlib.sha1()
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    digest.update(matrix.indptr.astype(np.int64, copy=False).tobytes())
    digest.update(matrix.indices.astype(np.int64, copy=False).tobytes())
    digest.update(matrix.data.tobytes())
    return digest.hexdigest()


def _row_normalize_positive(graph: sp.spmatrix) -> sp.csr_matrix:
    graph = graph.tocsr().astype(np.float32).copy()
    graph.setdiag(0.0)
    graph.data = np.maximum(graph.data, 0.0)
    graph.eliminate_zeros()
    row_sum = np.asarray(graph.sum(axis=1)).ravel()
    inv = np.divide(
        1.0,
        row_sum,
        out=np.zeros_like(row_sum, dtype=np.float32),
        where=row_sum > EPS,
    )
    graph = (sp.diags(inv) @ graph).tocsr()
    graph.sort_indices()
    return graph


def _accumulate_scores(graph: sp.csr_matrix, items: np.ndarray) -> Dict[int, float]:
    """Accumulate transition mass from a set of context items."""
    scores: Dict[int, float] = {}
    for item in items:
        start, end = graph.indptr[item], graph.indptr[item + 1]
        for candidate, value in zip(graph.indices[start:end], graph.data[start:end]):
            scores[int(candidate)] = scores.get(int(candidate), 0.0) + float(value)
    return scores


def _noisy_or_scores(graph: sp.csr_matrix, items: np.ndarray) -> Dict[int, float]:
    """Combine absolute transition evidence without row-wise max rescaling."""
    miss_probability: Dict[int, float] = {}
    for item in items:
        start, end = graph.indptr[item], graph.indptr[item + 1]
        for candidate, value in zip(graph.indices[start:end], graph.data[start:end]):
            value = float(np.clip(value, 0.0, 1.0 - 1e-7))
            previous = miss_probability.get(int(candidate), 1.0)
            miss_probability[int(candidate)] = previous * (1.0 - value)
    return {candidate: 1.0 - miss for candidate, miss in miss_probability.items()}


def _target_noisy_or_score(
    graph: sp.csr_matrix, items: np.ndarray, target: int
) -> float:
    """Return leave-one-out evidence for one target without materializing a map."""
    miss_probability = 1.0
    for item in items:
        if int(item) == target:
            continue
        start, end = graph.indptr[item], graph.indptr[item + 1]
        row_indices = graph.indices[start:end]
        position = int(np.searchsorted(row_indices, target))
        if position >= row_indices.size or int(row_indices[position]) != target:
            continue
        value = float(np.clip(graph.data[start + position], 0.0, 1.0 - 1e-7))
        miss_probability *= 1.0 - value
    return 1.0 - miss_probability


def _profile_drift_scores(
    graph: sp.csr_matrix, items: np.ndarray
) -> np.ndarray:
    """Measure each edge's leave-one-out change to the mean graph profile."""
    history_size = int(items.size)
    if history_size < 2:
        return np.ones(history_size, dtype=np.float32)
    profile_sum = _accumulate_scores(graph, items)
    full_profile = {
        item: value / float(history_size) for item, value in profile_sum.items()
    }
    scale = 0.5 / float(history_size - 1)
    drift = np.empty(history_size, dtype=np.float32)
    for position, item in enumerate(items):
        start, end = graph.indptr[item], graph.indptr[item + 1]
        row = {
            int(candidate): float(value)
            for candidate, value in zip(
                graph.indices[start:end], graph.data[start:end]
            )
        }
        l1_distance = sum(
            abs(row.get(candidate, 0.0) - mean_value)
            for candidate, mean_value in full_profile.items()
        )
        drift[position] = float(np.clip(scale * l1_distance, 0.0, 1.0))
    return drift


def _confidence_bounds(
    evidence: np.ndarray,
    reliability: np.ndarray,
    reliability_variance: np.ndarray,
    uncertainty_kappa: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute delta-method lower and upper bounds for multimodal evidence."""
    terms = 1.0 - reliability[:, None] * evidence
    consensus = 1.0 - np.prod(terms, axis=0)
    variance = np.zeros_like(consensus)
    for modality in range(evidence.shape[0]):
        if evidence.shape[0] > 1:
            other_product = np.prod(np.delete(terms, modality, axis=0), axis=0)
        else:
            other_product = np.ones_like(consensus)
        derivative = evidence[modality] * other_product
        variance += derivative * derivative * reliability_variance[modality]
    uncertainty = np.sqrt(np.maximum(variance, 0.0))
    radius = float(uncertainty_kappa) * uncertainty
    lower = np.maximum(consensus - radius, 0.0)
    upper = np.minimum(consensus + radius, 1.0)
    return consensus, uncertainty, lower, upper


def _crossfit_reliability(
    train: sp.csr_matrix,
    graphs: Sequence[sp.csr_matrix],
    probe_topl: int,
    shrinkage: float,
    reliability_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate independent user-modality retrieval rates from training folds.

    Each modality receives its own empirical-Bayes Beta posterior.  Failures
    are retained explicitly; this avoids forcing weak modalities to share one
    unit of probability mass as a Dirichlet model would.
    """
    n_modalities = len(graphs)
    n_users = train.shape[0]
    successes = np.zeros((n_modalities, n_users), dtype=np.float32)
    trials = np.zeros(n_users, dtype=np.float32)

    for user in range(n_users):
        start, end = train.indptr[user], train.indptr[user + 1]
        history = train.indices[start:end]
        if history.size < 2:
            continue

        folds = (np.arange(history.size) + user) % 2
        for probe_fold in (0, 1):
            probe = history[folds == probe_fold]
            context = history[folds != probe_fold]
            if probe.size == 0 or context.size == 0:
                continue
            trials[user] += float(probe.size)
            context_set = set(context.tolist())
            probe_set = set(probe.tolist())
            for modality, graph in enumerate(graphs):
                scores = _accumulate_scores(graph, context)
                for item in context_set:
                    scores.pop(item, None)
                if not scores:
                    continue
                ranked = sorted(scores, key=scores.get, reverse=True)[:probe_topl]
                successes[modality, user] += len(probe_set.intersection(ranked))

    total_trials = float(trials.sum())
    modality_successes = successes.sum(axis=1)
    global_mean = (modality_successes + 0.5) / (total_trials + 1.0)
    alpha = successes + float(shrinkage) * global_mean[:, None]
    beta = (
        trials[None, :]
        - successes
        + float(shrinkage) * (1.0 - global_mean[:, None])
    )
    total = np.maximum(alpha + beta, EPS)
    posterior_mean = alpha / total
    posterior_variance = alpha * beta / (total * total * (total + 1.0))

    if reliability_mode == "global":
        global_alpha = modality_successes + 0.5
        global_beta = total_trials - modality_successes + 0.5
        global_total = global_alpha + global_beta
        global_posterior = global_alpha / np.maximum(global_total, EPS)
        global_variance = (
            global_alpha
            * global_beta
            / np.maximum(global_total * global_total * (global_total + 1.0), EPS)
        )
        posterior_mean[:] = global_posterior[:, None]
        posterior_variance[:] = global_variance[:, None]
    elif reliability_mode == "uniform":
        # Common absolute retrieval rate, not a forced simplex allocation.
        posterior_mean[:] = float(np.mean(global_mean))
        posterior_variance[:] = 0.0

    return posterior_mean, posterior_variance, global_mean, successes, trials


def estimate_pravi_reliability(
    interactions: sp.spmatrix,
    modality_graphs: Sequence[sp.spmatrix],
    probe_topl: int = 10,
    shrinkage: float = 10.0,
    reliability_mode: str = "personalized",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Estimate train-only user/modality reliability for adaptive alignment.

    The returned posterior arrays are independent of model parameters and use
    only the observed training graph.  Keeping this estimator separate from
    candidate generation lets a model condition its loss without adding any
    virtual interactions or leaking validation/test edges.
    """
    if reliability_mode not in {"personalized", "global", "uniform"}:
        raise ValueError(f"Unknown reliability mode: {reliability_mode}")
    if probe_topl < 1 or shrinkage <= 0:
        raise ValueError("probe_topl must be positive and shrinkage must be > 0")
    if not modality_graphs:
        raise ValueError("At least one modality graph is required")

    train = interactions.tocsr().astype(np.float32).copy()
    train.data[:] = 1.0
    train.sort_indices()
    graphs = [_row_normalize_positive(graph) for graph in modality_graphs]
    for graph in graphs:
        if graph.shape != (train.shape[1], train.shape[1]):
            raise ValueError("Modality graph shape does not match the item count")

    reliability, variance, global_mean, successes, trials = _crossfit_reliability(
        train,
        graphs,
        probe_topl=int(probe_topl),
        shrinkage=float(shrinkage),
        reliability_mode=reliability_mode,
    )
    stats: Dict[str, object] = {
        "method": "train_crossfit_beta_reliability",
        "n_users": int(train.shape[0]),
        "n_items": int(train.shape[1]),
        "n_train_edges": int(train.nnz),
        "probe_topl": int(probe_topl),
        "shrinkage": float(shrinkage),
        "reliability_mode": reliability_mode,
        "global_reliabilities": [float(value) for value in global_mean],
        "global_reliability_sum": float(global_mean.sum()),
        "crossfit_successes": [float(value) for value in successes.sum(axis=1)],
        "crossfit_trials": float(trials.sum()),
        "user_reliability_mean": [
            float(value) for value in np.mean(reliability, axis=1)
        ],
        "user_reliability_std": [
            float(value) for value in np.std(reliability, axis=1)
        ],
        "user_reliability_variance_mean": [
            float(value) for value in np.mean(variance, axis=1)
        ],
    }
    return reliability.astype(np.float32), variance.astype(np.float32), stats


def _build_pravi_evidence(
    interactions: sp.spmatrix,
    modality_graphs: Sequence[sp.spmatrix],
    topk: int = 5,
    shrinkage: float = 10.0,
    reliability_mode: str = "personalized",
    probe_topl: int = 10,
    uncertainty_kappa: float = 1.0,
    mass_budget_ratio: float = 0.1,
    min_confidence: float = 0.0,
    min_modal_support: int = 1,
    binary_weights: bool = False,
    raw_candidate_weights: bool = False,
    include_donor_bounds: bool = False,
    min_donor_history: int = 2,
    donor_cost_mode: str = "loo_retrieval",
) -> Union[
    Tuple[sp.csr_matrix, Dict[str, object]],
    Tuple[sp.csr_matrix, sp.csr_matrix, Dict[str, object]],
]:
    """Build CF-LCB confidence-weighted virtual user-item interactions.

    Reliability is cross-fitted using only training edges. Candidate weights are
    lower confidence bounds whose total mass per user is capped relative to the
    number of observed training interactions.
    """
    if reliability_mode not in {"personalized", "global", "uniform"}:
        raise ValueError(f"Unknown reliability mode: {reliability_mode}")
    if topk < 1 or probe_topl < 1:
        raise ValueError("topk and probe_topl must be positive")
    if shrinkage <= 0 or uncertainty_kappa < 0 or mass_budget_ratio <= 0:
        raise ValueError("invalid CF-LCB hyperparameter")
    if not modality_graphs:
        raise ValueError("At least one modality graph is required")
    if min_donor_history < 2:
        raise ValueError("min_donor_history must be at least 2")
    if donor_cost_mode not in {"loo_retrieval", "profile_drift"}:
        raise ValueError(f"Unknown donor cost mode: {donor_cost_mode}")

    train = interactions.tocsr().astype(np.float32).copy()
    train.data[:] = 1.0
    train.sort_indices()
    graphs = [_row_normalize_positive(graph) for graph in modality_graphs]
    for graph in graphs:
        if graph.shape != (train.shape[1], train.shape[1]):
            raise ValueError("Modality graph shape does not match the item count")

    reliability, reliability_variance, global_mean, successes, trials = (
        _crossfit_reliability(
            train,
            graphs,
            probe_topl=probe_topl,
            shrinkage=shrinkage,
            reliability_mode=reliability_mode,
        )
    )

    virtual_users: List[int] = []
    virtual_items: List[int] = []
    virtual_weights: List[float] = []
    accepted_uncertainties: List[float] = []
    rejected_by_lcb = 0
    users_with_edges = 0
    donor_users: List[int] = []
    donor_items: List[int] = []
    donor_weights: List[float] = []
    donor_signals: List[List[float]] = [list() for _ in graphs]
    protected_donor_edges = 0

    for user in range(train.shape[0]):
        start, end = train.indptr[user], train.indptr[user + 1]
        history = train.indices[start:end]
        if history.size == 0:
            continue

        if include_donor_bounds:
            if history.size < int(min_donor_history):
                upper_bounds = np.ones(history.size, dtype=np.float32)
                protected_donor_edges += int(history.size)
            else:
                if donor_cost_mode == "profile_drift":
                    donor_evidence = np.stack(
                        [_profile_drift_scores(graph, history) for graph in graphs]
                    )
                else:
                    donor_evidence = np.empty(
                        (len(graphs), history.size), dtype=np.float32
                    )
                    for position, donor_item in enumerate(history):
                        donor_evidence[:, position] = np.fromiter(
                            (
                                _target_noisy_or_score(
                                    graph, history, int(donor_item)
                                )
                                for graph in graphs
                            ),
                            dtype=np.float32,
                            count=len(graphs),
                        )
                _, _, _, upper_bounds = _confidence_bounds(
                    donor_evidence,
                    reliability[:, user],
                    reliability_variance[:, user],
                    uncertainty_kappa,
                )
                for modality in range(len(graphs)):
                    donor_signals[modality].extend(
                        donor_evidence[modality].astype(float).tolist()
                    )
            donor_users.extend([user] * int(history.size))
            donor_items.extend(history.astype(int).tolist())
            donor_weights.extend(upper_bounds.astype(float).tolist())

        history_set = set(history.tolist())
        evidence_maps = [_noisy_or_scores(graph, history) for graph in graphs]
        candidate_ids = sorted(
            set().union(*(scores.keys() for scores in evidence_maps)) - history_set
        )
        if not candidate_ids:
            continue

        candidate_ids_array = np.asarray(candidate_ids, dtype=np.int64)
        evidence = np.zeros((len(graphs), len(candidate_ids)), dtype=np.float32)
        for modality, scores in enumerate(evidence_maps):
            evidence[modality] = np.fromiter(
                (scores.get(item, 0.0) for item in candidate_ids),
                dtype=np.float32,
                count=len(candidate_ids),
            )

        support = np.count_nonzero(evidence > 0.0, axis=0)
        _, uncertainty, confidence, _ = _confidence_bounds(
            evidence,
            reliability[:, user],
            reliability_variance[:, user],
            uncertainty_kappa,
        )

        valid = support >= int(min_modal_support)
        threshold_mask = confidence > float(min_confidence)
        rejected_by_lcb += int(np.count_nonzero(valid & ~threshold_mask))
        valid &= threshold_mask
        positions = np.flatnonzero(valid)
        if positions.size == 0:
            continue
        keep_count = min(int(topk), positions.size)
        if keep_count < positions.size:
            local = np.argpartition(confidence[positions], -keep_count)[-keep_count:]
            positions = positions[local]
        positions = positions[np.argsort(confidence[positions])[::-1]]

        selected = confidence[positions]
        mass_cap = float(mass_budget_ratio) * float(history.size)
        raw_mass = float(selected.sum())
        if raw_candidate_weights:
            selected = selected.copy()
        else:
            scale = min(1.0, mass_cap / max(raw_mass, EPS))
            selected = selected * scale
            if binary_weights:
                selected[:] = min(raw_mass, mass_cap) / max(len(selected), 1)

        virtual_users.extend([user] * len(positions))
        virtual_items.extend(candidate_ids_array[positions].tolist())
        virtual_weights.extend(selected.astype(float).tolist())
        accepted_uncertainties.extend(uncertainty[positions].astype(float).tolist())
        users_with_edges += 1

    virtual = sp.coo_matrix(
        (virtual_weights, (virtual_users, virtual_items)),
        shape=train.shape,
        dtype=np.float32,
    ).tocsr()
    virtual.sum_duplicates()
    if virtual.multiply(train).nnz:
        raise RuntimeError("PRAVI generated an observed training interaction")

    train_mass = float(train.sum())
    stats: Dict[str, object] = {
        "method": "CF-Beta-LCB-PRAVI",
        "n_users": int(train.shape[0]),
        "n_items": int(train.shape[1]),
        "n_train_edges": int(train.nnz),
        "n_virtual_edges": int(virtual.nnz),
        "users_with_virtual_edges": int(users_with_edges),
        "virtual_edges_per_user": float(virtual.nnz / max(train.shape[0], 1)),
        "virtual_mass": float(virtual.sum()),
        "virtual_to_train_mass": float(virtual.sum() / max(train_mass, EPS)),
        "mean_confidence": float(np.mean(virtual.data)) if virtual.nnz else 0.0,
        "mean_posterior_uncertainty": (
            float(np.mean(accepted_uncertainties)) if accepted_uncertainties else 0.0
        ),
        "lcb_rejections": int(rejected_by_lcb),
        "global_reliabilities": [float(value) for value in global_mean],
        "global_reliability_sum": float(global_mean.sum()),
        "crossfit_successes": [float(value) for value in successes.sum(axis=1)],
        "crossfit_trials": float(trials.sum()),
        "crossfit_recall": [
            float(value / max(float(trials.sum()), 1.0))
            for value in successes.sum(axis=1)
        ],
        "user_reliability_mean": [
            float(value) for value in np.mean(reliability, axis=1)
        ],
        "user_reliability_std": [
            float(value) for value in np.std(reliability, axis=1)
        ],
        "reliability_mode": reliability_mode,
        "uncertainty_kappa": float(uncertainty_kappa),
        "mass_budget_ratio": float(mass_budget_ratio),
        "binary_weights": bool(binary_weights),
        "candidate_weight_semantics": (
            "raw_lcb" if raw_candidate_weights else "budgeted_lcb"
        ),
    }
    if not include_donor_bounds:
        return virtual, stats

    donor_upper_bounds = sp.coo_matrix(
        (donor_weights, (donor_users, donor_items)),
        shape=train.shape,
        dtype=np.float32,
    ).tocsr()
    donor_upper_bounds.sum_duplicates()
    if donor_upper_bounds.nnz != train.nnz:
        raise RuntimeError("donor bounds must cover every observed training edge")
    donor_values = donor_upper_bounds.data
    stats["donor_upper_bound"] = {
        "mean": float(donor_values.mean()) if donor_values.size else 0.0,
        "min": float(donor_values.min()) if donor_values.size else 0.0,
        "max": float(donor_values.max()) if donor_values.size else 0.0,
        "std": float(donor_values.std()) if donor_values.size else 0.0,
        "protected_edges": int(protected_donor_edges),
        "min_history": int(min_donor_history),
        "semantics": f"training_leave_one_out_{donor_cost_mode}_ucb",
        "cost_mode": donor_cost_mode,
        "raw_signal_mean": [
            float(np.mean(values)) if values else 0.0
            for values in donor_signals
        ],
        "raw_signal_std": [
            float(np.std(values)) if values else 0.0
            for values in donor_signals
        ],
    }
    return virtual, donor_upper_bounds, stats


def build_pravi_interactions(
    interactions: sp.spmatrix,
    modality_graphs: Sequence[sp.spmatrix],
    topk: int = 5,
    shrinkage: float = 10.0,
    reliability_mode: str = "personalized",
    probe_topl: int = 10,
    uncertainty_kappa: float = 1.0,
    mass_budget_ratio: float = 0.1,
    min_confidence: float = 0.0,
    min_modal_support: int = 1,
    binary_weights: bool = False,
) -> Tuple[sp.csr_matrix, Dict[str, object]]:
    """Build budgeted CF-LCB virtual interactions for additive PRAVI variants."""
    result = _build_pravi_evidence(
        interactions,
        modality_graphs,
        topk=topk,
        shrinkage=shrinkage,
        reliability_mode=reliability_mode,
        probe_topl=probe_topl,
        uncertainty_kappa=uncertainty_kappa,
        mass_budget_ratio=mass_budget_ratio,
        min_confidence=min_confidence,
        min_modal_support=min_modal_support,
        binary_weights=binary_weights,
    )
    return result


def build_pravi_transport_evidence(
    interactions: sp.spmatrix,
    modality_graphs: Sequence[sp.spmatrix],
    topk: int = 5,
    shrinkage: float = 10.0,
    reliability_mode: str = "personalized",
    probe_topl: int = 10,
    uncertainty_kappa: float = 1.0,
    min_confidence: float = 0.0,
    min_modal_support: int = 1,
    min_donor_history: int = 2,
    donor_cost_mode: str = "loo_retrieval",
) -> Tuple[sp.csr_matrix, sp.csr_matrix, Dict[str, object]]:
    """Build receiver LCBs and leave-one-out donor UCBs from training only."""
    result = _build_pravi_evidence(
        interactions,
        modality_graphs,
        topk=topk,
        shrinkage=shrinkage,
        reliability_mode=reliability_mode,
        probe_topl=probe_topl,
        uncertainty_kappa=uncertainty_kappa,
        mass_budget_ratio=1.0,
        min_confidence=min_confidence,
        min_modal_support=min_modal_support,
        binary_weights=False,
        raw_candidate_weights=True,
        include_donor_bounds=True,
        min_donor_history=min_donor_history,
        donor_cost_mode=donor_cost_mode,
    )
    return result
