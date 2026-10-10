import unittest

import numpy as np
from scipy import sparse

from evidence import (
    build_dual_witness,
    build_overlap_user_knn,
    build_user_evidence,
    build_user_knn,
    evidence_scores,
    gather_scores,
    item_partition,
    prune_topk,
    shuffle_rows_within_degree,
    split_item_views,
)


def dense_cos(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norm, 1e-12)) @ (matrix / np.maximum(norm, 1e-12)).T


class DualWitnessTests(unittest.TestCase):
    def setUp(self):
        self.x = sparse.csr_matrix(np.array([
            [1, 1, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 1],
        ], dtype=np.float32))

    def test_partition_is_stable_and_boolean(self):
        a = item_partition(80)
        b = item_partition(80)
        self.assertEqual(a.dtype, np.bool_)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.any(a))
        self.assertTrue(np.any(~a))

    def test_views_are_disjoint_and_reconstruct_input(self):
        a, b = split_item_views(self.x)
        self.assertEqual((a.multiply(b)).nnz, 0)
        np.testing.assert_array_equal((a + b).toarray(), self.x.toarray())

    def test_user_knn_matches_dense_formula_and_block_size(self):
        expected = dense_cos(self.x.toarray())
        np.fill_diagonal(expected, 0.0)
        got1 = build_user_knn(self.x, k=20, block_size=2).toarray()
        got2 = build_user_knn(self.x, k=20, block_size=7).toarray()
        np.testing.assert_allclose(got1, expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(got1, got2)
        self.assertTrue(np.all(np.diag(got1) == 0))

    def test_dual_matches_literal_formula_and_block_size(self):
        a, b = split_item_views(self.x)
        expected = np.sqrt(np.maximum(dense_cos(a.toarray()), 0) *
                           np.maximum(dense_cos(b.toarray()), 0))
        np.fill_diagonal(expected, 0.0)
        got1 = build_dual_witness(self.x, k=20, block_size=2).toarray()
        got2 = build_dual_witness(self.x, k=20, block_size=7).toarray()
        np.testing.assert_allclose(got1, expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(got1, got2)
        self.assertTrue(np.all(np.diag(got1) == 0))

    def test_topk_tie_is_id_deterministic(self):
        raw = sparse.csr_matrix(np.array([[0, .5, .5, .2], [0, 0, 0, 0]], dtype=np.float32))
        got = prune_topk(raw, 1).toarray()
        self.assertEqual(got[0, 1], .5)
        self.assertEqual(got[0, 2], 0)

    def test_empty_view_has_zero_witness_without_fallback(self):
        x = sparse.csr_matrix(np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32))
        got = build_dual_witness(x, k=10, block_size=1).toarray()
        self.assertEqual(np.count_nonzero(got), 0)

    def test_shuffling_preserves_degree_and_is_seeded(self):
        shuffled1, permutation1 = shuffle_rows_within_degree(self.x, 2026)
        shuffled2, permutation2 = shuffle_rows_within_degree(self.x, 2026)
        np.testing.assert_array_equal(permutation1, permutation2)
        np.testing.assert_array_equal(np.diff(shuffled1.indptr), np.diff(self.x.indptr))
        np.testing.assert_array_equal(shuffled1.toarray(), shuffled2.toarray())
        np.testing.assert_array_equal(np.asarray(shuffled1.sum(axis=0)),
                                      np.asarray(self.x.sum(axis=0)))

    def test_shuffled_dual_matches_dense_and_excludes_self(self):
        a, b = split_item_views(self.x)
        b, _ = shuffle_rows_within_degree(b, 2026)
        expected = np.sqrt(np.maximum(dense_cos(a.toarray()), 0) *
                           np.maximum(dense_cos(b.toarray()), 0))
        np.fill_diagonal(expected, 0.0)
        got = build_dual_witness(self.x, k=20, block_size=2, shuffled=True, seed=2026)
        np.testing.assert_allclose(got.toarray(), expected, rtol=2e-6, atol=2e-6)
        self.assertEqual(np.count_nonzero(got.diagonal()), 0)

    def test_empty_and_cold_users_keep_square_zero_weights(self):
        for x in (sparse.csr_matrix((0, 8), dtype=np.float32),
                  sparse.csr_matrix((4, 8), dtype=np.float32)):
            for mode in ("dual", "userknn", "min2", "shuffled"):
                got = build_user_evidence(x, mode=mode, block_size=2)
                self.assertEqual(got.shape, (x.shape[0], x.shape[0]))
                self.assertEqual(got.nnz, 0)

    def test_construction_does_not_change_support_or_add_targets(self):
        x = self.x.copy()
        before = x.toarray().copy()
        heldout_items = [2, 3]
        for mode in ("dual", "userknn", "min2", "shuffled"):
            weights = build_user_evidence(x, mode=mode, topk=3, block_size=2)
            evidence_scores(weights, x, [0, 1], heldout_items)
            np.testing.assert_array_equal(x.toarray(), before)
            self.assertEqual(x[0, 2], 0)
            self.assertEqual(x[1, 3], 0)

    def test_overlap_control_removes_single_overlap(self):
        got = build_overlap_user_knn(self.x, k=20, minimum_overlap=2, block_size=3).toarray()
        full = dense_cos(self.x.toarray())
        overlap = self.x.dot(self.x.T).toarray()
        for u, v in zip(*np.nonzero(got)):
            self.assertGreaterEqual(overlap[u, v], 2)
            self.assertAlmostEqual(got[u, v], full[u, v], places=5)

    def test_public_modes_and_gather_scores(self):
        for mode in ("dual", "userknn", "min2", "shuffled"):
            weights = build_user_evidence(self.x, mode=mode, topk=3, block_size=2)
            self.assertEqual(weights.shape, (self.x.shape[0], self.x.shape[0]))
            scores = evidence_scores(weights, self.x, [0, 1], [2, 3, 4])
            np.testing.assert_array_equal(scores, weights[[0, 1]].dot(self.x[:, [2, 3, 4]]).toarray())

    def test_row_gather_matches_dense_for_different_items_and_repeated_users(self):
        users = np.array([0, 1, 0, 5])
        items = np.array([[2, 3, 4], [5, 1, 7], [0, 1, 2], [2, 4, 6]])
        for mode in ("dual", "userknn", "min2", "shuffled"):
            weights = build_user_evidence(self.x, mode=mode, topk=3, block_size=2)
            raw = weights.dot(self.x).toarray()
            expected = raw[users[:, None], items]
            got1 = gather_scores(weights, self.x, users, items, block_size=1)
            got2 = gather_scores(weights, self.x, users, items, block_size=4)
            np.testing.assert_array_equal(got1, expected)
            np.testing.assert_array_equal(got1, got2)

    def test_row_gather_rejects_cartesian_misuse_and_handles_empty(self):
        weights = build_user_knn(self.x, block_size=2)
        with self.assertRaises(ValueError):
            evidence_scores(weights, self.x, [0, 1], [[2], [3]])
        with self.assertRaises(ValueError):
            gather_scores(weights, self.x, [0, 1], [2, 3])
        with self.assertRaises(ValueError):
            gather_scores(weights, self.x, [0], [[2], [3]])
        empty = gather_scores(weights, self.x, [], np.empty((0, 17), dtype=np.int64))
        self.assertEqual(empty.shape, (0, 17))

    def test_invalid_mode_and_block_are_rejected(self):
        with self.assertRaises(ValueError):
            build_user_evidence(self.x, mode="bad")
        with self.assertRaises(ValueError):
            build_dual_witness(self.x, block_size=0)


if __name__ == "__main__":
    unittest.main()
