"""Official-formula SG-URInit preprocessing for model adapters."""

from __future__ import annotations

import hashlib

import numpy as np
import scipy.sparse as sp


def inverse_item_degree_operator(interactions):
    """Return SG-URInit's sum of neighbors weighted by inverse item degree."""
    observed = interactions.tocsr().astype(np.float32)
    observed.sum_duplicates()
    observed.data.fill(1.0)
    item_degree = np.asarray(observed.sum(axis=0)).ravel().astype(np.float32)
    inverse_degree = np.zeros_like(item_degree)
    nonzero = item_degree > 0
    inverse_degree[nonzero] = 1.0 / item_degree[nonzero]

    coo = observed.tocoo()
    values = inverse_degree[coo.col]
    return sp.coo_matrix(
        (values, (coo.row, coo.col)),
        shape=observed.shape,
        dtype=np.float32,
    ).tocsr()


def cluster_aligned_features(features, num_clusters, random_state=0):
    """Replace each item feature with its nearest K-Means centroid."""
    from sklearn.cluster import KMeans

    array = np.asarray(features, dtype=np.float32)
    num_clusters = int(num_clusters)
    if array.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if num_clusters <= 0 or num_clusters > array.shape[0]:
        raise ValueError("num_clusters must be between one and item count")
    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=int(random_state),
        n_init=10,
    )
    assignments = kmeans.fit_predict(array)
    return np.asarray(kmeans.cluster_centers_[assignments], dtype=np.float32)


def build_sg_user_profile(
    interactions,
    features,
    num_clusters=4,
    cluster_weight=0.1,
    random_state=0,
):
    """Construct SG-URInit local/global user representations (Eq. 1--5)."""
    cluster_weight = float(cluster_weight)
    if not 0.0 <= cluster_weight <= 1.0:
        raise ValueError("cluster_weight must lie in [0, 1]")
    array = np.asarray(features, dtype=np.float32)
    if interactions.shape[1] != array.shape[0]:
        raise ValueError("interaction item count and feature rows must match")

    aggregator = inverse_item_degree_operator(interactions)
    local_profile = np.asarray(aggregator @ array, dtype=np.float32)
    if cluster_weight == 0.0:
        return local_profile
    aligned = cluster_aligned_features(array, num_clusters, random_state)
    global_profile = np.asarray(aggregator @ aligned, dtype=np.float32)
    return np.asarray(
        (1.0 - cluster_weight) * local_profile
        + cluster_weight * global_profile,
        dtype=np.float32,
    )


def profile_cache_signature(interactions, features):
    """Return a compact signature guarding profile-cache reuse."""
    observed = interactions.tocsr()
    array = np.asarray(features)
    digest = hashlib.sha256()
    digest.update(np.asarray(observed.shape, dtype=np.int64).tobytes())
    digest.update(observed.indptr.astype(np.int64, copy=False).tobytes())
    digest.update(observed.indices.astype(np.int64, copy=False).tobytes())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(str(array.dtype).encode("ascii"))
    sample_size = min(32, array.shape[0])
    if sample_size:
        digest.update(np.ascontiguousarray(array[:sample_size]).tobytes())
        digest.update(np.ascontiguousarray(array[-sample_size:]).tobytes())
    return digest.hexdigest()[:16]
