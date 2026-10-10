"""Small, framework-independent helpers for the fixed feature precheck.

The remote runner imports the same formulas, but this module intentionally has
no import-time dependency on torch so that the CPU fixtures can run in the
workspace image.  The runner is supplied separately as ``remote_precheck.py``.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

EPSILON_REL = 0.01
NUM_BATCHES = 3
GATE_MIN_COUNT = 2


def _is_torch(value: Any) -> bool:
    return hasattr(value, "index_copy") and hasattr(value, "detach")


def _finite(value: Any) -> bool:
    if _is_torch(value):
        import torch
        return bool(torch.isfinite(value).all().item())
    import numpy as np
    return bool(np.isfinite(value).all())


def _row_norm(value: Any):
    if _is_torch(value):
        return value.norm(dim=1, keepdim=True)
    import numpy as np
    return np.linalg.norm(value, axis=1, keepdims=True)


def replace_rows(base: Any, row_ids: Any, deltas: Any):
    """Return a functional row replacement and leave ``base`` untouched."""
    if _is_torch(base):
        rows = base.index_select(0, row_ids)
        return base.index_copy(0, row_ids, rows + deltas)
    import numpy as np
    result = np.array(base, copy=True)
    result[np.asarray(row_ids)] = result[np.asarray(row_ids)] + deltas
    return result


def targeted_deltas(feature_rows: Any, gradient_rows: Any, epsilon: float = EPSILON_REL):
    """Make the image-BPR ascent step with each row's relative norm."""
    feature_norm = _row_norm(feature_rows)
    gradient_norm = _row_norm(gradient_rows)
    if not (_finite(feature_norm) and _finite(gradient_norm)):
        raise FloatingPointError("nonfinite feature or gradient norm")
    if _is_torch(feature_rows):
        import torch
        if bool((feature_norm <= 0).any().item()) or bool((gradient_norm <= 0).any().item()):
            raise ValueError("zero feature or gradient norm")
    else:
        if bool((feature_norm <= 0).any()) or bool((gradient_norm <= 0).any()):
            raise ValueError("zero feature or gradient norm")
    return epsilon * feature_norm * gradient_rows / gradient_norm


def random_deltas(feature_rows: Any, rng: Any, epsilon: float = EPSILON_REL):
    """Make a same-row-norm Gaussian null using a private RNG."""
    if _is_torch(feature_rows):
        import torch
        noise = torch.randn(feature_rows.shape, dtype=feature_rows.dtype,
                            device="cpu", generator=rng).to(feature_rows.device)
        noise_norm = noise.norm(dim=1, keepdim=True)
        feature_norm = feature_rows.norm(dim=1, keepdim=True)
        if bool((noise_norm <= 0).any().item()):
            raise ValueError("zero random direction norm")
        return epsilon * feature_norm * noise / noise_norm
    import numpy as np
    noise = rng.normal(size=np.asarray(feature_rows).shape).astype(np.asarray(feature_rows).dtype)
    noise_norm = np.linalg.norm(noise, axis=1, keepdims=True)
    feature_norm = np.linalg.norm(feature_rows, axis=1, keepdims=True)
    if bool((noise_norm <= 0).any()):
        raise ValueError("zero random direction norm")
    return epsilon * feature_norm * noise / noise_norm


def gate_pass(targeted_greater: Any, minimum: int = GATE_MIN_COUNT) -> bool:
    """Pre-registered two-of-three strict targeted-vs-random gate."""
    values = [bool(value) for value in targeted_greater]
    if len(values) != NUM_BATCHES:
        raise ValueError("expected exactly three fixed batches")
    return sum(values) >= minimum


def bpr_gap(clean: float, perturbed: float) -> float:
    if not all(math.isfinite(float(value)) for value in (clean, perturbed)):
        raise FloatingPointError("nonfinite BPR value")
    return float(perturbed) - float(clean)


def tensor_sha(value: Any) -> str:
    """Stable digest used for public state/feature receipts."""
    if _is_torch(value):
        value = value.detach().cpu().contiguous().numpy()
    payload = value.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "EPSILON_REL", "NUM_BATCHES", "GATE_MIN_COUNT", "replace_rows",
    "targeted_deltas", "random_deltas", "gate_pass", "bpr_gap", "tensor_sha",
]
