"""TRAIN-only interest groups and private, rejection-free BPR negative sampling.

This is a PerPEFT-inspired sampling adaptation, not its official implementation.
The shuffled arm reassigns users to the SAME fixed semantic item pools.  It never
rebuilds pools from shuffled labels.  No Torch, evaluator or split loader is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


N_GROUPS = 8
GROUP_PROBABILITY = 0.5
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 300
KMEANS_TOL = 1e-4


class NoEligibleNegativeError(ValueError):
    """The user has interacted with the complete TRAIN item universe."""


def _integer_vector(values, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return values.astype(np.int64, copy=False)


def _readonly(values) -> np.ndarray:
    values = np.asarray(values).copy()
    values.flags.writeable = False
    return values


def _array_sha(values) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(str(values.shape).encode("ascii"))
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def _histories(train) -> tuple[tuple[np.ndarray, ...], int]:
    """Accept canonical CSR protocol or a small binary dense matrix.

    Duplicate CSR columns are rejected instead of silently changing the observed
    positive-event count.  Explicit stored zeros are omitted, positive values
    must be one.  The input matrix is never mutated.
    """
    if all(hasattr(train, name) for name in ("indptr", "indices", "data", "shape")):
        if len(train.shape) != 2:
            raise ValueError("TRAIN must have two dimensions")
        n_users, n_items = map(int, train.shape)
        indptr = _integer_vector(train.indptr, "CSR indptr")
        indices = _integer_vector(train.indices, "CSR indices")
        data = np.asarray(train.data)
        if (indptr.size != n_users + 1 or indptr[0] != 0
                or indptr[-1] != indices.size or data.shape != indices.shape
                or np.any(np.diff(indptr) < 0)):
            raise ValueError("invalid CSR layout")
        if indices.size and (np.any(indices < 0) or np.any(indices >= n_items)):
            raise ValueError("CSR item ID out of range")
        if not np.all(np.isfinite(data)) or np.any((data != 0) & (data != 1)):
            raise ValueError("TRAIN must be binary")
        rows = []
        for user in range(n_users):
            left, right = indptr[user:user + 2]
            row = np.sort(indices[left:right][data[left:right] != 0])
            if row.size > 1 and np.any(np.diff(row) == 0):
                raise ValueError("TRAIN has duplicate user/item positives")
            rows.append(_readonly(row))
    else:
        train = np.asarray(train)
        if train.ndim != 2 or not np.all(np.isfinite(train)):
            raise ValueError("TRAIN must be a finite binary matrix")
        if np.any((train != 0) & (train != 1)):
            raise ValueError("TRAIN must be binary")
        n_users, n_items = train.shape
        rows = [_readonly(np.flatnonzero(row).astype(np.int64)) for row in train]
    if n_users <= 0 or n_items <= 0:
        raise ValueError("TRAIN population and catalog must be nonempty")
    return tuple(rows), int(n_items)


def _normalize_rows(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("feature arrays must be finite matrices")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def train_profiles(histories, image_features, text_features, n_items) -> np.ndarray:
    """Each item view L2 -> equal TRAIN mean -> concat -> joint L2.

    Missing/zero feature rows remain zero.  They never cause event or user drops.
    A cold TRAIN user has an all-zero grouping profile.
    """
    image = _normalize_rows(image_features)
    text = _normalize_rows(text_features)
    if image.shape[0] != n_items or text.shape[0] != n_items:
        raise ValueError("feature row count must equal TRAIN catalog size")
    if image.shape[1] == 0 or text.shape[1] == 0:
        raise ValueError("both feature views must have at least one column")
    profiles = np.zeros((len(histories), image.shape[1] + text.shape[1]),
                        dtype=np.float64)
    for user, seen in enumerate(histories):
        if seen.size:
            profiles[user] = np.concatenate((image[seen].mean(axis=0),
                                            text[seen].mean(axis=0)))
    return _normalize_rows(profiles)


def _distances(points, centers):
    result = (np.einsum("ij,ij->i", points, points)[:, None]
              + np.einsum("ij,ij->i", centers, centers)[None, :]
              - 2.0 * points.dot(centers.T))
    return np.maximum(result, 0.0)


def _numpy_kmeans(points, n_groups, seed):
    """Explicit deterministic kmeans++/Lloyd fallback, not sklearn equivalence.

    Ten sequential starts share a private RandomState.  Squared Euclidean
    distance ties choose lowest center ID.  Empty centers are re-seeded from
    largest-distance rows with user-ID ties.  Best-start inertia ties choose the
    first start.  Relative convergence uses mean feature variance times 1e-4.
    """
    rng = np.random.RandomState(seed)
    best = None
    tolerance = float(np.var(points, axis=0).mean()) * KMEANS_TOL
    for start in range(KMEANS_N_INIT):
        chosen = [int(rng.randint(points.shape[0]))]
        centers = [points[chosen[0]].copy()]
        closest = _distances(points, np.asarray(centers)).reshape(-1)
        for _ in range(1, n_groups):
            total = float(closest.sum())
            if total > 0:
                selected = int(np.searchsorted(np.cumsum(closest),
                                                rng.random_sample() * total,
                                                side="right"))
                selected = min(selected, points.shape[0] - 1)
            else:
                selected = next(row for row in range(points.shape[0])
                                if row not in chosen)
            chosen.append(selected)
            centers.append(points[selected].copy())
            closest = np.minimum(closest,
                                 _distances(points, points[selected:selected + 1]).reshape(-1))
        centers = np.asarray(centers)
        previous = None
        for iteration in range(KMEANS_MAX_ITER):
            distances = _distances(points, centers)
            labels = distances.argmin(axis=1).astype(np.int64)
            new = centers.copy()
            occupied_rows = set()
            for group in range(n_groups):
                selected = labels == group
                if selected.any():
                    new[group] = points[selected].mean(axis=0)
                else:
                    remaining = distances[np.arange(points.shape[0]), labels]
                    order = np.lexsort((np.arange(points.shape[0]), -remaining))
                    row = next(int(row) for row in order if int(row) not in occupied_rows)
                    occupied_rows.add(row)
                    new[group] = points[row]
            shift = float(np.square(new - centers).sum())
            centers = new
            if (previous is not None and np.array_equal(labels, previous)) or shift <= tolerance:
                break
            previous = labels.copy()
        distances = _distances(points, centers)
        labels = distances.argmin(axis=1).astype(np.int64)
        inertia = float(distances[np.arange(points.shape[0]), labels].sum())
        if best is None or inertia < best[0]:
            best = inertia, labels.copy(), centers.copy(), iteration + 1, start
    return best[1], {"backend": "numpy_explicit_kmeanspp_lloyd_v1",
                     "inertia": best[0], "iterations": best[3],
                     "best_start": best[4], "n_init": KMEANS_N_INIT,
                     "max_iter": KMEANS_MAX_ITER, "tol": KMEANS_TOL,
                     "sklearn_equivalent_claim": False,
                     "centers_sha256": _array_sha(best[2])}


def _fit_groups(profiles, n_groups, seed, backend):
    if backend not in ("auto", "sklearn", "numpy"):
        raise ValueError("backend must be auto, sklearn or numpy")
    if profiles.shape[0] < n_groups:
        raise ValueError("at least n_groups TRAIN users are required")
    import_error = None
    if backend != "numpy":
        try:
            from sklearn.cluster import KMeans
            import sklearn
        except ImportError as exc:
            if backend == "sklearn":
                raise
            import_error = type(exc).__name__
        else:
            model = KMeans(n_clusters=n_groups, random_state=seed,
                           n_init=KMEANS_N_INIT, max_iter=KMEANS_MAX_ITER,
                           tol=KMEANS_TOL, algorithm="lloyd")
            labels = model.fit_predict(profiles).astype(np.int64)
            return labels, {"backend": "sklearn_kmeans_lloyd",
                            "sklearn_version": sklearn.__version__,
                            "inertia": float(model.inertia_),
                            "iterations": int(model.n_iter_),
                            "n_init": KMEANS_N_INIT, "max_iter": KMEANS_MAX_ITER,
                            "tol": KMEANS_TOL,
                            "centers_sha256": _array_sha(model.cluster_centers_)}
    labels, details = _numpy_kmeans(profiles, n_groups, seed)
    if import_error is not None:
        details["auto_import_failure"] = import_error
    return labels, details


def _distribution(values) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.int64)
    degree, counts = np.unique(values, return_counts=True)
    return {"count": int(values.size),
            "mean": float(values.mean()) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "min": int(values.min()) if values.size else None,
            "max": int(values.max()) if values.size else None,
            "train_degree_histogram": {str(int(k)): int(v)
                                       for k, v in zip(degree, counts)}}


@dataclass(frozen=True)
class GroupAssets:
    histories: tuple[np.ndarray, ...]
    n_items: int
    n_groups: int
    labels: np.ndarray
    shuffled_labels: np.ndarray
    degrees: np.ndarray
    item_degrees: np.ndarray
    universe: np.ndarray
    control_user_permutation: np.ndarray
    pools: tuple[np.ndarray, ...]
    diagnostics: dict[str, Any]

    @property
    def n_users(self):
        return len(self.histories)


def assemble_group_assets(train, labels, *, n_groups=N_GROUPS, seed=2026,
                          grouping_diagnostics=None) -> GroupAssets:
    """Build fixed TRAIN pools and exact-degree assignment permutation.

    Exposed for cache restoration and literal unit controls.  Formal semantic
    labels are constructed by build_group_assets, not supplied from targets.
    """
    histories, n_items = _histories(train)
    labels = _integer_vector(labels, "group labels")
    if n_groups <= 0 or labels.size != len(histories):
        raise ValueError("group label population mismatch")
    if np.any(labels < 0) or np.any(labels >= n_groups):
        raise ValueError("group label out of range")
    degrees = np.asarray([row.size for row in histories], dtype=np.int64)
    item_degrees = np.zeros(n_items, dtype=np.int64)
    pooled = [[] for _ in range(n_groups)]
    for user, seen in enumerate(histories):
        item_degrees[seen] += 1
        pooled[int(labels[user])].extend(seen.tolist())
    pools = tuple(_readonly(np.unique(row).astype(np.int64)) for row in pooled)
    universe = _readonly(np.flatnonzero(item_degrees > 0).astype(np.int64))
    rng = np.random.RandomState(seed)
    shuffled = labels.copy()
    source_users = np.arange(len(histories), dtype=np.int64)
    strata_counts = {}
    for degree in np.unique(degrees):
        owners = np.flatnonzero(degrees == degree)
        permutation = rng.permutation(owners.size)
        source_users[owners] = owners[permutation]
        shuffled[owners] = labels[source_users[owners]]
        before = np.bincount(labels[owners], minlength=n_groups)
        after = np.bincount(shuffled[owners], minlength=n_groups)
        if not np.array_equal(before, after):
            raise AssertionError("exact-degree label-count conservation failed")
        strata_counts[str(int(degree))] = before.tolist()
    group_counts = np.bincount(labels, minlength=n_groups)
    shuffled_counts = np.bincount(shuffled, minlength=n_groups)
    pair_masses = np.bincount(labels, weights=degrees, minlength=n_groups).astype(np.int64)
    shuffled_masses = np.bincount(shuffled, weights=degrees,
                                 minlength=n_groups).astype(np.int64)
    if (not np.array_equal(group_counts, shuffled_counts)
            or not np.array_equal(pair_masses, shuffled_masses)):
        raise AssertionError("group size/positive-pair mass conservation failed")
    def own_filter_rates(assignments):
        rates = []
        for group in range(n_groups):
            owners = np.flatnonzero(assignments == group)
            pool = pools[group]
            if owners.size == 0:
                rates.append({"users": 0, "mean_eligible_fraction": None,
                              "users_with_nonempty_eligible": 0,
                              "own_positive_filter_rate": None})
                continue
            eligible = []
            for user in owners:
                blocked = _blocked_pool_positions(pool, histories[int(user)])
                eligible.append(pool.size - blocked.size)
            eligible = np.asarray(eligible, dtype=np.int64)
            rates.append({"users": int(owners.size),
                          "mean_eligible_fraction": float(np.mean(eligible / max(pool.size, 1))),
                          "users_with_nonempty_eligible": int(np.sum(eligible > 0)),
                          "own_positive_filter_rate": float(np.sum(eligible > 0) / owners.size)})
        return rates
    indptr = np.concatenate(([0], np.cumsum(degrees)))
    flat = np.concatenate(histories) if degrees.sum() else np.empty(0, dtype=np.int64)
    details = {"n_users": len(histories), "n_items": n_items,
               "n_train_positives": int(degrees.sum()), "n_groups": n_groups,
               "uniform_universe": "unique_original_train_item_union",
               "uniform_universe_size": int(universe.size),
               "uniform_universe_sha256": _array_sha(universe),
               "seed": int(seed), "group_probability": GROUP_PROBABILITY,
               "group_user_counts": group_counts.tolist(),
               "shuffled_group_user_counts": shuffled_counts.tolist(),
               "group_positive_pair_masses": pair_masses.tolist(),
               "shuffled_group_positive_pair_masses": shuffled_masses.tolist(),
               "exact_degree_group_counts": strata_counts,
               "exact_degree_group_counts_preserved": True,
               "shuffled_uses_original_semantic_pools": True,
               "assignment_fixed_points": int(np.sum(shuffled == labels)),
               "permutation_source_user_fixed_points": int(np.sum(source_users == np.arange(len(histories)))),
               "permutation_source_user_fixed_point_fraction": float(np.mean(source_users == np.arange(len(histories)))),
               "assignment_mismatches": int(np.sum(shuffled != labels)),
               "assignment_mismatch_fraction": float(np.mean(shuffled != labels)),
               "pool_sizes": [int(row.size) for row in pools],
               "pool_item_popularity": [_distribution(item_degrees[row]) for row in pools],
               "pool_own_filter_true_assignment": own_filter_rates(labels),
               "pool_own_filter_shuffled_assignment": own_filter_rates(shuffled),
               "train_item_popularity": _distribution(item_degrees),
               "train_indptr_sha256": _array_sha(indptr),
               "train_indices_sha256": _array_sha(flat),
               "semantic_labels_sha256": _array_sha(labels),
               "shuffled_labels_sha256": _array_sha(shuffled),
               "control_user_permutation_sha256": _array_sha(source_users),
               "pool_sha256": [_array_sha(row) for row in pools],
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if grouping_diagnostics is not None:
        details["grouping"] = dict(grouping_diagnostics)
    return GroupAssets(histories, n_items, int(n_groups), _readonly(labels),
                       _readonly(shuffled), _readonly(degrees),
                       _readonly(item_degrees), universe, _readonly(source_users), pools, details)


def build_group_assets(train, image_features, text_features, *, seed=2026,
                       n_groups=N_GROUPS, kmeans_backend="auto") -> GroupAssets:
    histories, n_items = _histories(train)
    profiles = train_profiles(histories, image_features, text_features, n_items)
    labels, details = _fit_groups(profiles, n_groups, seed, kmeans_backend)
    details.update({"profile_formula": "item_l2_each_view_train_mean_concat_joint_l2",
                    "profiles_sha256": _array_sha(profiles),
                    "image_features_sha256": _array_sha(image_features),
                    "text_features_sha256": _array_sha(text_features),
                    "zero_profile_users": int(np.sum(np.linalg.norm(profiles, axis=1) == 0))})
    return assemble_group_assets(train, labels, n_groups=n_groups, seed=seed,
                                 grouping_diagnostics=details)


def _blocked_pool_positions(pool, seen):
    positions = np.searchsorted(pool, seen)
    in_range = positions < pool.size
    positions = positions[in_range]
    seen = seen[in_range]
    return positions[pool[positions] == seen].astype(np.int64)


def _eligible_rank(rank: int, forbidden_shifted: np.ndarray) -> int:
    return int(rank + np.searchsorted(forbidden_shifted, rank, side="right"))


class InterestNegativeSampler:
    """sample(user_ids) -> int64 negatives, with no global RNG mutation.

    All modes consume exactly three private variates per example in order
    (mixture coin, uniform rank, group rank), even when a branch is unused.
    Hence batching a stream differently preserves it.  Each draw is uniform
    over the eligible set via rank mapping, not bounded rejection or clipping.
    """

    def __init__(self, assets: GroupAssets, *, mode="semantic", seed=2026):
        if mode not in ("uniform", "semantic", "shuffled"):
            raise ValueError("mode must be uniform, semantic or shuffled")
        self.assets = assets
        self.mode = mode
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.assignments = assets.shuffled_labels if mode == "shuffled" else assets.labels
        self._global_shifted = []
        self._group_shifted = []
        self._group_eligible = np.zeros(assets.n_users, dtype=np.int64)
        for user, seen in enumerate(assets.histories):
            blocked_global = _blocked_pool_positions(assets.universe, seen)
            if blocked_global.size != seen.size:
                raise AssertionError("own TRAIN positives absent from TRAIN item universe")
            self._global_shifted.append(blocked_global - np.arange(blocked_global.size))
            pool = assets.pools[int(self.assignments[user])]
            blocked = _blocked_pool_positions(pool, seen)
            self._group_shifted.append(blocked - np.arange(blocked.size))
            self._group_eligible[user] = pool.size - blocked.size
        self._draws = self._requested_group = self._actual_group = self._fallback = 0
        self._calls = 0
        self._drawn_item_counts = np.zeros(assets.n_items, dtype=np.int64)

    def sample(self, user_ids) -> np.ndarray:
        users = _integer_vector(user_ids, "user IDs")
        if users.size and (np.any(users < 0) or np.any(users >= self.assets.n_users)):
            raise ValueError("user ID out of range")
        if np.any(self.assets.degrees[users] >= self.assets.universe.size):
            raise NoEligibleNegativeError("a requested TRAIN user has no eligible TRAIN-universe item")
        draws = self.rng.random_sample((users.size, 3))
        negatives = np.empty(users.size, dtype=np.int64)
        requested = actual = fallback = 0
        for index, user in enumerate(users):
            user = int(user)
            choose_group = self.mode != "uniform" and draws[index, 0] < GROUP_PROBABILITY
            if choose_group:
                requested += 1
            if choose_group and self._group_eligible[user] > 0:
                rank = int(draws[index, 2] * self._group_eligible[user])
                position = _eligible_rank(rank, self._group_shifted[user])
                negatives[index] = self.assets.pools[int(self.assignments[user])][position]
                actual += 1
            else:
                eligible = self.assets.universe.size - self.assets.degrees[user]
                rank = int(draws[index, 1] * eligible)
                position = _eligible_rank(rank, self._global_shifted[user])
                negatives[index] = self.assets.universe[position]
                fallback += int(choose_group)
        self._draws += users.size
        self._requested_group += requested
        self._actual_group += actual
        self._fallback += fallback
        self._calls += 1
        self._drawn_item_counts += np.bincount(negatives, minlength=self.assets.n_items)
        return negatives

    def stats(self) -> dict[str, Any]:
        available = self._group_eligible > 0
        positive_mass = int(self.assets.degrees.sum())
        positive_available = int(self.assets.degrees[available].sum())
        degree_sum = int(self._drawn_item_counts.dot(self.assets.item_degrees))
        by_degree = {}
        for degree in np.unique(self.assets.item_degrees):
            by_degree[str(int(degree))] = int(self._drawn_item_counts[
                self.assets.item_degrees == degree].sum())
        return {"mode": self.mode, "seed": self.seed, "sample_calls": self._calls,
                "draws": int(self._draws), "group_requested_draws": self._requested_group,
                "group_actual_draws": self._actual_group,
                "empty_group_uniform_fallback_draws": self._fallback,
                "group_requested_rate": self._requested_group / max(self._draws, 1),
                "group_actual_rate": self._actual_group / max(self._draws, 1),
                "private_variates_per_example": 3,
                "rejection_free_uniform_eligible": True,
                "own_train_positives_excluded": True,
                "uniform_universe": "unique_original_train_item_union",
                "uniform_universe_size": int(self.assets.universe.size),
                "uniform_universe_sha256": self.assets.diagnostics["uniform_universe_sha256"],
                "eligible_group_users": int(available.sum()),
                "eligible_group_user_fraction": float(available.mean()),
                "eligible_group_positive_pair_mass": positive_available,
                "eligible_group_positive_pair_mass_fraction": positive_available / max(positive_mass, 1),
                "group_eligible_counts": _distribution(self._group_eligible),
                "drawn_distinct_items": int(np.sum(self._drawn_item_counts > 0)),
                "drawn_catalog_coverage": float(np.mean(self._drawn_item_counts > 0)),
                "drawn_train_universe_coverage": int(np.sum(self._drawn_item_counts > 0)) / max(self.assets.universe.size, 1),
                "drawn_negative_mean_train_degree": degree_sum / max(self._draws, 1),
                "drawn_negative_train_degree_histogram": by_degree,
                "drawn_item_counts_sha256": _array_sha(self._drawn_item_counts)}
