"""Fixed item-direction residual and profile operations for the diagnostic.

The projection is evaluated in float64.  The only zeroing rule is a machine
round-off guard for an exactly parallel vector; it is not a tunable model
parameter and is fixed before any metric is computed.
"""
import numpy as np


_EPS = np.finfo(np.float64).eps
_NORMALIZE_EPS = 1e-12


def _matrix(value, name):
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError("{} must be a rank-2 array".format(name))
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("{} must be numeric".format(name))
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("{} contains nonfinite values".format(name))
    return array


def normalize_rows(rows):
    """Unit-normalize rows, retaining an explicit zero for zero rows."""
    array = _matrix(rows, "rows")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    result = np.zeros_like(array)
    nonzero = norms[:, 0] > _NORMALIZE_EPS
    result[nonzero] = array[nonzero] / norms[nonzero]
    return result.astype(np.float32)


def residualize(review_vectors, item_text_vectors):
    """Apply r=normalize(z - t dot(z,t)) row by row.

    ``t`` is normalized independently.  A zero item vector has no direction,
    so its residual is the normalized review vector.  A zero review remains
    zero.  The round-off guard removes only residual norms within a fixed
    double-precision error bound for collinear inputs.
    """
    z = _matrix(review_vectors, "review_vectors")
    t_raw = _matrix(item_text_vectors, "item_text_vectors")
    if z.shape != t_raw.shape:
        raise ValueError("review and item arrays must have equal shape")
    z_norm = np.linalg.norm(z, axis=1)
    t_norm = np.linalg.norm(t_raw, axis=1)
    t = np.zeros_like(t_raw)
    has_item = t_norm > _NORMALIZE_EPS
    t[has_item] = t_raw[has_item] / t_norm[has_item, None]
    dots = np.einsum("ij,ij->i", z, t)
    residual = z - t * dots[:, None]
    residual_norm = np.linalg.norm(residual, axis=1)
    # The bound follows the accumulated dot-product error in one row.  It is
    # solely a numerical exactness guard and is fixed independently of data.
    bound = 8.0 * _EPS * np.maximum(1.0, np.sqrt(float(z.shape[1]))) * np.maximum(z_norm, 1.0)
    residual[residual_norm <= bound] = 0.0
    result = normalize_rows(residual)
    if not np.isfinite(result).all():
        raise ValueError("residualization produced nonfinite values")
    return result


def profile(vectors, owner_ids, n_users):
    """Sum unit review vectors per user and unit-normalize the result."""
    values = _matrix(vectors, "vectors")
    owners = np.asarray(owner_ids)
    if owners.ndim != 1 or len(owners) != len(values):
        raise ValueError("owner_ids must align with vectors")
    if not np.issubdtype(owners.dtype, np.integer):
        raise ValueError("owner_ids must be integers")
    if isinstance(n_users, bool) or int(n_users) != n_users or n_users < 1:
        raise ValueError("n_users must be positive")
    owners = owners.astype(np.int64, copy=False)
    if np.any(owners < 0) or np.any(owners >= int(n_users)):
        raise ValueError("owner id outside user range")
    result = np.zeros((int(n_users), values.shape[1]), dtype=np.float64)
    np.add.at(result, owners, values)
    return normalize_rows(result)


def score_rows(profiles, target_vectors, users, comparisons):
    """Score each user's profile against its fixed target/comparison rows."""
    user_profiles = _matrix(profiles, "profiles")
    targets = np.asarray(target_vectors)
    if targets.ndim != 2 or targets.shape[1] != user_profiles.shape[1]:
        raise ValueError("target_vectors has incompatible shape")
    targets = targets.astype(np.float64, copy=False)
    if not np.isfinite(targets).all():
        raise ValueError("target_vectors contains nonfinite values")
    user_ids = np.asarray(users)
    choices = np.asarray(comparisons)
    if user_ids.ndim != 1 or choices.ndim != 2 or len(user_ids) != len(choices):
        raise ValueError("users and comparisons must align")
    if not np.issubdtype(user_ids.dtype, np.integer) or np.any(user_ids < 0):
        raise ValueError("invalid user ids")
    if np.any(choices < 0) or np.any(choices >= targets.shape[0]):
        raise ValueError("comparison item outside target bank")
    return np.einsum("ud,umd->um", user_profiles[user_ids], targets[choices], optimize=True)
