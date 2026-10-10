from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.dvt import (
    build_degree_preserving_transport,
    build_evidence_cycle_transport,
    build_signed_transport_operator,
)


class DVTTest(unittest.TestCase):
    def test_exact_reciprocal_switch(self):
        observed = sp.csr_matrix(
            ([1, 1], ([0, 1], [0, 1])), shape=(2, 2)
        )
        candidates = sp.csr_matrix(
            ([0.8, 0.6], ([0, 1], [1, 0])), shape=(2, 2)
        )

        delta, stats = build_degree_preserving_transport(observed, candidates)

        np.testing.assert_array_equal(
            delta.toarray(), np.asarray([[-1.0, 1.0], [1.0, -1.0]])
        )
        self.assertEqual(stats["swaps"], 1)
        self.assertEqual(stats["users"], 2)
        self.assertEqual(stats["positive_nnz"], 2)
        self.assertEqual(stats["negative_nnz"], 2)
        self.assertEqual(stats["candidate_coverage"], 1.0)
        self.assertAlmostEqual(stats["confidence"]["mean"], 0.7)

    def test_degrees_are_preserved_exactly(self):
        observed = sp.eye(4, format="csr")
        candidates = sp.csr_matrix(
            (
                [0.9, 0.8, 0.7, 0.6],
                ([0, 1, 2, 3], [1, 0, 3, 2]),
            ),
            shape=(4, 4),
        )

        delta, stats = build_degree_preserving_transport(observed, candidates)

        np.testing.assert_array_equal(np.asarray(delta.sum(axis=1)).ravel(), 0.0)
        np.testing.assert_array_equal(np.asarray(delta.sum(axis=0)).ravel(), 0.0)
        self.assertEqual(stats["swaps"], 2)
        self.assertEqual(stats["row_residual_max"], 0.0)
        self.assertEqual(stats["column_residual_max"], 0.0)

    def test_input_edges_are_not_reused(self):
        observed = sp.eye(3, format="csr")
        candidates = sp.csr_matrix(
            (
                [0.9, 0.8, 0.7, 0.6],
                ([0, 1, 0, 2], [1, 0, 2, 0]),
            ),
            shape=(3, 3),
        )

        delta, stats = build_degree_preserving_transport(observed, candidates)

        self.assertEqual(stats["swaps"], 1)
        self.assertEqual(stats["used_observed_edges"], 2)
        self.assertEqual(stats["used_candidate_edges"], 2)
        self.assertEqual(delta.nnz, 4)

    def test_default_limits_each_user_to_one_swap(self):
        observed = sp.csr_matrix(
            (
                [1, 1, 1, 1],
                ([0, 0, 1, 2], [0, 3, 1, 2]),
            ),
            shape=(3, 4),
        )
        candidates = sp.csr_matrix(
            (
                [0.9, 0.8, 0.7, 0.6],
                ([0, 1, 0, 2], [1, 0, 2, 3]),
            ),
            shape=(3, 4),
        )

        _, limited = build_degree_preserving_transport(observed, candidates)
        _, unrestricted = build_degree_preserving_transport(
            observed, candidates, max_swaps_per_user=2
        )

        self.assertEqual(limited["swaps"], 1)
        self.assertEqual(unrestricted["swaps"], 2)

    def test_signed_operator_preserves_sign_and_normalization(self):
        observed = sp.csr_matrix(
            ([1, 1, 1], ([0, 0, 1], [0, 1, 2])), shape=(2, 3)
        )
        delta = sp.csr_matrix(
            ([-1, 1, 1, -1], ([0, 0, 1, 1], [0, 2, 0, 2])),
            shape=(2, 3),
        )

        operator = build_signed_transport_operator(
            observed, delta, "cpu"
        ).to_dense().numpy()

        expected = np.zeros((5, 5), dtype=np.float32)
        expected[0, 2] = expected[2, 0] = -1.0 / np.sqrt(2.0)
        expected[0, 4] = expected[4, 0] = 1.0 / np.sqrt(2.0)
        expected[1, 2] = expected[2, 1] = 1.0
        expected[1, 4] = expected[4, 1] = -1.0
        np.testing.assert_allclose(operator, expected, rtol=1e-6, atol=0.0)
        self.assertEqual(np.count_nonzero(operator > 0), 4)
        self.assertEqual(np.count_nonzero(operator < 0), 4)

    def test_cycle_transport_finds_non_reciprocal_three_cycle(self):
        observed = sp.eye(3, format="csr")
        candidates = sp.csr_matrix(
            ([0.9, 0.8, 0.7], ([0, 1, 2], [1, 2, 0])), shape=(3, 3)
        )

        delta, stats = build_evidence_cycle_transport(observed, candidates)

        np.testing.assert_array_equal(
            delta.toarray(),
            np.asarray([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [1.0, 0.0, -1.0]]),
        )
        self.assertEqual(stats["cycles"], 1)
        self.assertEqual(stats["transport_arcs"], 3)
        self.assertEqual(stats["cycle_length_histogram"], {"3": 1})
        self.assertEqual(stats["candidate_coverage"], 1.0)
        self.assertEqual(stats["row_residual_max"], 0.0)
        self.assertEqual(stats["column_residual_max"], 0.0)

    def test_cycle_transport_leaves_unclosed_arc_unused(self):
        observed = sp.eye(2, format="csr")
        candidates = sp.csr_matrix(([0.9], ([0], [1])), shape=(2, 2))

        delta, stats = build_evidence_cycle_transport(observed, candidates)

        self.assertEqual(delta.nnz, 0)
        self.assertEqual(stats["cycles"], 0)
        self.assertEqual(stats["users"], 0)

    def test_confidence_cycle_uses_common_bottleneck_weight(self):
        observed = sp.eye(3, format="csr")
        candidates = sp.csr_matrix(
            ([0.9, 0.8, 0.7], ([0, 1, 2], [1, 2, 0])), shape=(3, 3)
        )

        delta, stats = build_evidence_cycle_transport(
            observed, candidates, weight_mode="confidence"
        )

        np.testing.assert_allclose(
            delta.toarray(),
            0.7
            * np.asarray(
                [[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [1.0, 0.0, -1.0]]
            ),
        )
        self.assertAlmostEqual(stats["cycle_weight_mean"], 0.7)
        self.assertLessEqual(stats["row_residual_max"], 1e-12)
        self.assertLessEqual(stats["column_residual_max"], 1e-12)

    def test_cycle_confidence_floor_rejects_weak_cycle(self):
        observed = sp.eye(3, format="csr")
        candidates = sp.csr_matrix(
            ([0.9, 0.8, 0.7], ([0, 1, 2], [1, 2, 0])), shape=(3, 3)
        )

        delta, stats = build_evidence_cycle_transport(
            observed, candidates, min_arc_confidence=0.75
        )

        self.assertEqual(delta.nnz, 0)
        self.assertEqual(stats["cycles"], 0)
        self.assertEqual(stats["min_arc_confidence"], 0.75)

    def test_net_utility_cycle_chooses_the_safer_donor(self):
        observed = sp.csr_matrix(
            ([1, 1, 1], ([0, 1, 2], [0, 1, 1])), shape=(3, 2)
        )
        candidates = sp.csr_matrix(
            ([0.9, 0.7], ([0, 2], [1, 0])), shape=(3, 2)
        )
        donor_upper_bounds = sp.csr_matrix(
            ([0.1, 0.8, 0.2], ([0, 1, 2], [0, 1, 1])), shape=(3, 2)
        )

        delta, stats = build_evidence_cycle_transport(
            observed,
            candidates,
            donor_upper_bounds=donor_upper_bounds,
        )

        np.testing.assert_array_equal(
            delta.toarray(),
            np.asarray([[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0]]),
        )
        self.assertTrue(stats["donor_aware"])
        self.assertEqual(stats["cycles"], 1)
        self.assertEqual(stats["transport_arcs"], 2)
        self.assertAlmostEqual(stats["net_margin"]["min"], 0.6)
        self.assertAlmostEqual(stats["donor_upper_bound"]["max"], 0.2)
        self.assertEqual(stats["row_residual_max"], 0.0)
        self.assertEqual(stats["column_residual_max"], 0.0)

    def test_net_utility_cycle_rejects_nonpositive_margins(self):
        observed = sp.eye(2, format="csr")
        candidates = sp.csr_matrix(
            ([0.8, 0.8], ([0, 1], [1, 0])), shape=(2, 2)
        )
        donor_upper_bounds = sp.csr_matrix(
            ([0.9, 0.9], ([0, 1], [0, 1])), shape=(2, 2)
        )

        delta, stats = build_evidence_cycle_transport(
            observed,
            candidates,
            donor_upper_bounds=donor_upper_bounds,
        )

        self.assertEqual(delta.nnz, 0)
        self.assertEqual(stats["candidate_graph_arcs"], 0)
        self.assertEqual(stats["donor_rejected_owner_arcs"], 2)


if __name__ == "__main__":
    unittest.main()
