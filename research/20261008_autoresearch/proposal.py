"""Change BRIDGE training evidence while inheriting its prediction path intact.

The edge arm deletes the supervised interaction from the evidence calculation,
conditional on the cached original KNN support. The history arm removes only
the history entry. This class does not alter adjacency, model parameters,
candidate masks, objectives, or prediction-time behavior evidence.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from evidence import CounterfactualCache
from models.bridge import BRIDGE


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_csr(matrix):
    matrix = sparse.csr_matrix(matrix, dtype=np.float32, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def _verify_training_matrix(cache, dataset, n_users, n_items):
    current = _canonical_csr(dataset.inter_matrix(form="csr"))
    stored = _canonical_csr(cache.train)
    if current.shape != (n_users, n_items) or stored.shape != current.shape:
        raise ValueError("Evidence cache training matrix shape does not match the model")
    for field in ("indptr", "indices", "data"):
        if not np.array_equal(getattr(current, field), getattr(stored, field)):
            raise ValueError("Evidence cache training matrix differs in " + field)
    if not np.all(current.data == 1):
        raise ValueError("Evidence requires binary training interactions")
    users = np.repeat(np.arange(n_users, dtype=np.int64), np.diff(current.indptr))
    items = current.indices.astype(np.int64, copy=False)
    expected = {"edge_users": users, "edge_items": items,
                "edge_keys": users * n_items + items}
    for field, values in expected.items():
        if not np.array_equal(getattr(cache, field), values):
            raise ValueError("Evidence cache training edge IDs differ in " + field)


class TargetIsolatedBRIDGE(BRIDGE):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.evidence_mode = str(self._cfg(config, "evidence_mode", "original"))
        if self.evidence_mode not in ("original", "history", "edge", "soft_edge"):
            raise ValueError("Unknown evidence_mode: " + self.evidence_mode)
        self.isolation_strength = float(self._cfg(config, "isolation_strength", 1.0))
        if not np.isfinite(self.isolation_strength) or not 0.0 <= self.isolation_strength <= 1.0:
            raise ValueError("isolation_strength must be finite and in [0, 1]")
        if self.evidence_mode == "soft_edge" and self.isolation_strength in (0.0, 1.0):
            # Endpoints are useful as exact regression controls, but named arms
            # should make the interpolation explicit in their configuration.
            pass
        self._evidence_context = None
        self._evidence_cache = None
        self._evidence_statistics = {}
        self._assert_baseline_configuration()
        if self.evidence_mode == "original":
            # No cache construction, file access, or random draws in this arm.
            return
        if self.behavior_score_matrix is None:
            raise ValueError("Target-isolated evidence currently requires BRIDGE's dense backend")
        path = Path(str(self._cfg(config, "evidence_cache", "")))
        expected_hash = str(self._cfg(config, "evidence_cache_sha256", "")).lower()
        if not path.is_absolute() or not path.is_file():
            raise ValueError("evidence_cache must identify an existing absolute file path")
        if len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash):
            raise ValueError("An explicit evidence_cache_sha256 is required")
        if _sha256(path) != expected_hash:
            raise ValueError("Evidence cache SHA256 does not match the frozen artifact")
        cache = CounterfactualCache.load(path)
        _verify_training_matrix(cache, dataset, self.n_users, self.n_items)
        self._evidence_cache = cache

    def _assert_baseline_configuration(self):
        required = {"behavior_sim_method": "cosine", "behavior_aggregation": "sum_sqrt",
                    "behavior_score_norm": "z"}
        for name, expected in required.items():
            if getattr(self, name) != expected:
                raise ValueError("Target-isolated evidence requires " + name + "=" + expected)
        for name in ("use_behavior_proto", "use_private_residual", "use_residual_gate",
                     "use_activity_simplex", "use_user_graph"):
            if getattr(self, name):
                raise ValueError("Experimental add-on must be disabled: " + name)
        if self.msr_weight != 0 or self.private_residual_weight != 0:
            raise ValueError("MSR and private residual weights must be zero")
        if not self.use_behavior_correction:
            raise ValueError("The controlled carrier must retain behavior correction")

    def _pair_behavior_scores(self, users, items):
        context = self._evidence_context
        if context is None:
            return super()._pair_behavior_scores(users, items)
        index = context["index"]
        if index >= 2:
            raise RuntimeError("BRIDGE requested more than two training evidence pairs")
        if not torch.equal(users, context["users"]) or not torch.equal(items, context["items"][index]):
            raise RuntimeError("BRIDGE training evidence dispatch does not match the supervised pair")
        context["index"] += 1
        return context["values"][index]

    def calculate_loss(self, interaction):
        if self.evidence_mode == "original":
            return super().calculate_loss(interaction)
        if self._evidence_context is not None:
            raise RuntimeError("Nested target-isolated loss calls are unsupported")
        users, positive, negative = interaction[:3]
        if any(value.ndim != 1 for value in (users, positive, negative)):
            raise ValueError("BRIDGE expects one-dimensional user, positive, and negative tensors")
        query = [value.detach().cpu().numpy() for value in (users, positive, negative)]
        cache_mode = "edge" if self.evidence_mode == "soft_edge" else self.evidence_mode
        adjusted = self._evidence_cache.get_pair(*query, mode=cache_mode)
        original = [BRIDGE._pair_behavior_scores(self, users, items).detach()
                    for items in (positive, negative)]
        if self.evidence_mode == "soft_edge":
            strength = self.isolation_strength
            adjusted = [orig.detach().cpu().numpy() + strength * (value - orig.detach().cpu().numpy())
                        for value, orig in zip(adjusted, original)]
        values = [torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
                  for value, reference in zip(adjusted, original)]
        context = {"users": users.detach().clone(),
                   "items": [positive.detach().clone(), negative.detach().clone()],
                   "values": values, "index": 0}
        self._evidence_context = context
        try:
            result = super().calculate_loss(interaction)
            if context["index"] != 2:
                raise RuntimeError("BRIDGE did not consume both target-isolated evidence pairs")
        finally:
            self._evidence_context = None
        self._record_diagnostics(original, values)
        return result

    def _record_diagnostics(self, original, adjusted):
        values = {"original_positive_z": original[0], "original_negative_z": original[1],
                  "adjusted_positive_z": adjusted[0], "adjusted_negative_z": adjusted[1]}
        statistics = self._evidence_statistics
        statistics["count"] = statistics.get("count", 0) + original[0].numel()
        statistics["batches"] = statistics.get("batches", 0) + 1
        for name, value in values.items():
            statistics[name] = statistics.get(name, 0.0) + value.detach().double().sum().item()

    def train_diagnostics(self, reset=True):
        """Summaries describe processed training examples, never alter learning."""
        statistics = self._evidence_statistics
        count = statistics.get("count", 0)
        result = {"mode": self.evidence_mode, "count": count,
                  "batches": statistics.get("batches", 0)}
        if count:
            for name in ("original_positive_z", "original_negative_z",
                         "adjusted_positive_z", "adjusted_negative_z"):
                result["mean_" + name] = statistics[name] / count
        if reset:
            self._evidence_statistics = {}
        return result
