"""Leak-free reconstruction of the VI-MMRec virtual interaction formulas."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


EPS = 1e-12


@torch.no_grad()
def build_feature_knn(
    features: Sequence[torch.Tensor], topk: int, chunk_size: int = 512
) -> sp.csr_matrix:
    """Build top-k neighbors from the sum of modality cosine similarities."""
    if not features:
        raise ValueError("At least one feature tensor is required")
    n_items = features[0].shape[0]
    if any(feature.shape[0] != n_items for feature in features):
        raise ValueError("Feature tensors have inconsistent item counts")
    if not 0 < topk < n_items:
        raise ValueError("topk must be between 1 and n_items - 1")

    normalized = [F.normalize(feature.detach(), p=2, dim=1) for feature in features]
    rows = []
    cols = []
    for start in range(0, n_items, chunk_size):
        end = min(start + chunk_size, n_items)
        similarities = sum(
            feature[start:end] @ feature.transpose(0, 1) for feature in normalized
        )
        local_rows = torch.arange(end - start, device=similarities.device)
        global_rows = torch.arange(start, end, device=similarities.device)
        similarities[local_rows, global_rows] = -float("inf")
        neighbors = torch.topk(similarities, topk, dim=1).indices
        rows.append(global_rows[:, None].expand(-1, topk).reshape(-1).cpu().numpy())
        cols.append(neighbors.reshape(-1).cpu().numpy())

    row = np.concatenate(rows)
    col = np.concatenate(cols)
    data = np.ones(row.size, dtype=np.float32)
    return sp.coo_matrix((data, (row, col)), shape=(n_items, n_items)).tocsr()


def _binary(matrix: sp.spmatrix) -> sp.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32).copy()
    matrix.data[:] = 1.0
    matrix.eliminate_zeros()
    return matrix


def build_vi_interactions(
    interactions: sp.spmatrix,
    item_graphs: Sequence[sp.spmatrix],
    strategy: str = "overlay",
    lambda_weight: float = 0.005,
    topk: int = 10,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, Dict[str, object]]:
    """Construct VI-MMRec Eq. 10-13 using training interactions only."""
    if strategy not in {"overlay", "synergistic"}:
        raise ValueError(f"Unknown VI-MMRec strategy: {strategy}")
    expected_graphs = 1 if strategy == "synergistic" else None
    if not item_graphs or (expected_graphs and len(item_graphs) != expected_graphs):
        raise ValueError("Unexpected item graphs for VI-MMRec strategy")
    if lambda_weight <= 0 or topk <= 0:
        raise ValueError("lambda_weight and topk must be positive")

    train = _binary(interactions)
    modality_virtual = []
    overlap_rates = []
    for graph in item_graphs:
        graph = _binary(graph)
        candidates = _binary(train @ graph)
        modality_virtual.append(candidates)
        overlap_rates.append(float(candidates.multiply(train).nnz / max(train.nnz, 1)))

    random_overlap = min(
        1.0,
        float(topk * train.nnz / max(train.shape[0] * train.shape[1], 1)),
    )
    weights = [rate / max(random_overlap, EPS) for rate in overlap_rates]

    combined = sp.csr_matrix(train.shape, dtype=np.float32)
    for weight, candidates in zip(weights, modality_virtual):
        combined = combined + float(weight) * candidates
    augmented = (train + float(lambda_weight) * combined).tocsr()
    augmented.data = np.minimum(augmented.data, 1.0)
    augmented.eliminate_zeros()

    virtual = (augmented - train).tocsr()
    virtual.data = np.maximum(virtual.data, 0.0)
    virtual.eliminate_zeros()
    if virtual.multiply(train).nnz:
        raise RuntimeError("VI-MMRec virtual matrix overlaps observed training edges")

    stats: Dict[str, object] = {
        "method": f"VI-MMRec-{strategy[0].upper()}",
        "strategy": strategy,
        "lambda": float(lambda_weight),
        "topk": int(topk),
        "overlap_rates": overlap_rates,
        "random_overlap": float(random_overlap),
        "modality_weights": weights,
        "n_train_edges": int(train.nnz),
        "n_virtual_edges": int(virtual.nnz),
        "virtual_mass": float(virtual.sum()),
        "virtual_to_train_mass": float(virtual.sum() / max(float(train.sum()), EPS)),
    }
    return augmented, virtual, stats
