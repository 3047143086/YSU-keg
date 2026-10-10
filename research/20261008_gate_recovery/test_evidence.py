"""Check the analytical estimator against literal held-edge graph rebuilds."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from evidence import CounterfactualCache, _sum, build_counterfactual_cache, cosine_similarity


def literal_scores(train, user, target, support, mode):
    history = train[user].copy()
    if mode != "original":
        history[target] = 0
    graph_train = train.copy()
    if mode == "edge":
        graph_train[user, target] = 0
    popularity = graph_train.sum(0)
    cooccurrence = graph_train.T @ graph_train
    np.fill_diagonal(cooccurrence, 0)
    denominator = np.sqrt(np.maximum(popularity, 1e-6))
    similarity = cooccurrence / denominator[:, None] / denominator[None, :]
    similarity *= support
    scores = history @ similarity.T / np.sqrt(max(history.sum(), 1))
    return (scores - scores.mean()) / max(scores.std(), 1e-6)


class EdgeDeletionTests(unittest.TestCase):
    def check_rebuild(self, train, topk):
        cache = build_counterfactual_cache(train, topk=topk)
        support = cache.similarity != 0
        for user, target in zip(cache.edge_users, cache.edge_items):
            negatives = np.flatnonzero(train[user] == 0)
            if not len(negatives):
                continue
            for mode in ("original", "history", "edge"):
                positive, negative = cache.get_pair([user], [target], negatives[None, :], mode)
                reference = literal_scores(train, user, target, support, mode)
                np.testing.assert_allclose(positive, reference[[target]], atol=4e-6, rtol=4e-6)
                np.testing.assert_allclose(negative[0], reference[negatives], atol=4e-6, rtol=4e-6)

    def test_exact_rebuild_with_asymmetric_pruned_graph(self):
        train = np.array([[1, 1, 1, 0, 0, 0], [1, 0, 1, 1, 0, 0],
                          [0, 1, 1, 0, 1, 0], [1, 0, 0, 1, 1, 0],
                          [0, 0, 0, 0, 0, 1]], dtype=np.float64)
        self.check_rebuild(train, topk=2)

    def test_random_graphs_against_literal_rebuild(self):
        generator = np.random.default_rng(17)
        for _ in range(6):
            train = (generator.random((12, 17)) < 0.23).astype(np.float64)
            self.check_rebuild(train, topk=4)

    def test_zero_other_user_support_and_empty_user(self):
        train = np.array([[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=float)
        self.check_rebuild(train, topk=3)
        cache = build_counterfactual_cache(train, topk=3)
        positive, negative = cache.get_pair([0], [0], [3], mode="edge")
        np.testing.assert_array_equal(positive, [0])
        np.testing.assert_array_equal(negative, [0])

    def test_nonempty_counterfactual_history_with_zero_evidence(self):
        # After holding out (1,2), history still contains item0 but has no
        # remaining candidate evidence under the frozen top-2 support. Float32
        # cancellation previously produced .014901 after z-score normalization.
        train = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1]], dtype=float)
        cache = build_counterfactual_cache(train, topk=2)
        positive, negative = cache.get_pair([1], [2], [3], mode="edge")
        np.testing.assert_allclose(positive, [0], atol=1e-7, rtol=0)
        np.testing.assert_allclose(negative, [0], atol=1e-7, rtol=0)
        isolated = np.array([[1, 1, 0], [1, 0, 1]], dtype=float)
        cache = build_counterfactual_cache(isolated, topk=2)
        positive, negative = cache.get_pair([0], [0], [2], mode="edge")
        np.testing.assert_allclose(positive, [0], atol=1e-7, rtol=0)
        np.testing.assert_allclose(negative, [0], atol=1e-7, rtol=0)
        self.assertEqual(cache.raw_scores.dtype, np.float64)

    def test_reject_preaccumulated_float32_rows(self):
        train = np.array([[1, 1, 0], [1, 0, 1]], dtype=float)
        with self.assertRaisesRegex(ValueError, "accumulated in float64"):
            build_counterfactual_cache(train, raw_scores=np.zeros_like(train, dtype=np.float32))

    def test_full_training_history_negative_mask(self):
        cache = build_counterfactual_cache(np.array([[1, 1, 0], [0, 1, 1]], dtype=float))
        for forbidden in (0, 1):
            with self.assertRaisesRegex(ValueError, "observed training positive"):
                cache.get_pair([0], [0], [forbidden])
        with self.assertRaisesRegex(ValueError, "observed training edge"):
            cache.get_pair([0], [2], [2])

    def test_invalid_duplicate_or_weighted_interactions(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            build_counterfactual_cache(np.array([[2, 0, 0]], dtype=float))

    def test_roundtrip_and_no_overwrite(self):
        cache = build_counterfactual_cache(np.array([[1, 1, 0], [0, 1, 1]], dtype=float))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.npz"
            cache.save(path)
            restored = CounterfactualCache.load(path)
            for mode in ("edge", "history", "original"):
                expected = cache.get_pair([0, 1], [1, 2], [2, 0], mode)
                actual = restored.get_pair([0, 1], [1, 2], [2, 0], mode)
                for left, right in zip(expected, actual):
                    np.testing.assert_array_equal(left, right)
            with self.assertRaises(FileExistsError):
                cache.save(path)

    def test_sparse_matches_dense_when_scipy_available(self):
        try:
            from scipy import sparse
        except ImportError:
            self.skipTest("SciPy is installed on the experiment server, not this local environment")
        train = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1]], dtype=float)
        dense = build_counterfactual_cache(train, topk=2)
        csr = build_counterfactual_cache(sparse.csr_matrix(train), topk=2)
        for mode in ("original", "history", "edge"):
            actual = csr.get_pair([0, 1, 2], [0, 2, 3], [2, 3, 0], mode)
            expected = dense.get_pair([0, 1, 2], [0, 2, 3], [2, 3, 0], mode)
            for left, right in zip(expected, actual):
                np.testing.assert_allclose(left, right, atol=4e-6)

    def test_sparse_column_sums_accumulate_before_rounding(self):
        try:
            from scipy import sparse
        except ImportError:
            self.skipTest("SciPy is installed on the experiment server, not this local environment")
        # Older SciPy versions can reduce float32 CSR values before applying a
        # requested float64 sum dtype. Casting must precede accumulation.
        values = np.array([[0, 0.70710677, 0.5], [0.5, 0.5, 0.70710677],
                           [0.70710677, 0, 0.5]], dtype=np.float32)
        for axis in (0, 1):
            expected = values.astype(np.float64).sum(axis=axis)
            for matrix in (sparse.csr_matrix(values), sparse.csc_matrix(values)):
                np.testing.assert_array_equal(_sum(matrix, axis), expected)

    def test_sparse_batched_zero_row_with_identical_frozen_support(self):
        try:
            from scipy import sparse
        except ImportError:
            self.skipTest("SciPy is installed on the experiment server, not this local environment")
        train = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1]], dtype=np.float32)
        # Freeze the specific tied top-2 support that exposed the remote bug;
        # np.argpartition tie choices differ across NumPy versions.
        similarity = cosine_similarity(train, topk=0)
        similarity[1, 0] = 0
        similarity[2, 1] = 0
        dense = build_counterfactual_cache(train, similarity=similarity)
        csr = build_counterfactual_cache(sparse.csr_matrix(train),
                                         similarity=sparse.csr_matrix(similarity))
        for field in ("raw_scores", "original_mean", "history_mean", "edge_mean",
                      "edge_target_raw", "self_vote_raw"):
            np.testing.assert_allclose(getattr(csr, field), getattr(dense, field),
                                       atol=1e-15, rtol=0, err_msg=field)
        for mode in ("original", "history", "edge"):
            expected = dense.get_pair([0, 1, 2], [0, 2, 3], [2, 3, 0], mode)
            actual = csr.get_pair([0, 1, 2], [0, 2, 3], [2, 3, 0], mode)
            for left, right in zip(expected, actual):
                np.testing.assert_allclose(left, right, atol=4e-6)
            single = csr.get_pair([1], [2], [3], mode)
            for batch, one in zip(actual, single):
                np.testing.assert_array_equal(batch[1:2], one)
            if mode == "edge":
                for values in actual:
                    np.testing.assert_allclose(values[1], 0, atol=1e-7, rtol=0)


if __name__ == "__main__":
    unittest.main()
