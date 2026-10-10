from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.pravi import (
    build_pravi_interactions,
    build_pravi_transport_evidence,
    estimate_pravi_reliability,
    sparse_fingerprint,
)
from models.msca_pravi import build_anchored_virtual_operator


class PraviGraphTest(unittest.TestCase):
    def setUp(self):
        # User 0 consumed items 0/1; user 1 consumed items 2/3.
        self.interactions = sp.csr_matrix(
            ([1, 1, 1, 1], ([0, 0, 1, 1], [0, 1, 2, 3])), shape=(2, 5)
        )
        visual = np.zeros((5, 5), dtype=np.float32)
        textual = np.zeros((5, 5), dtype=np.float32)
        visual[0, 1] = visual[1, 0] = 1.0
        textual[0, 1] = textual[1, 0] = 0.8
        visual[0, 4] = visual[1, 4] = 0.9
        textual[0, 4] = textual[1, 4] = 0.85
        visual[2, 3] = visual[3, 2] = 0.7
        textual[2, 3] = textual[3, 2] = 0.7
        visual[2, 4] = visual[3, 4] = 0.9
        self.graphs = [sp.csr_matrix(visual), sp.csr_matrix(textual)]

    def test_virtual_edges_are_unobserved_and_bounded(self):
        virtual, stats = build_pravi_interactions(
            self.interactions,
            self.graphs,
            topk=1,
            shrinkage=1.0,
            uncertainty_kappa=0.0,
            mass_budget_ratio=0.1,
        )
        self.assertEqual(virtual.multiply(self.interactions).nnz, 0)
        self.assertLessEqual(virtual.nnz, self.interactions.shape[0])
        self.assertTrue(np.all(virtual.data > 0))
        self.assertTrue(np.all(virtual.data <= 1))
        row_mass = np.asarray(virtual.sum(axis=1)).ravel()
        degree = np.asarray(self.interactions.sum(axis=1)).ravel()
        self.assertTrue(np.all(row_mass <= 0.1 * degree + 1e-7))
        self.assertEqual(stats["n_virtual_edges"], virtual.nnz)
        self.assertEqual(stats["method"], "CF-Beta-LCB-PRAVI")
        # Independent Beta posteriors are absolute retrieval rates and are not
        # forced to sum to one across modalities.
        self.assertNotAlmostEqual(stats["global_reliability_sum"], 1.0)

    def test_missing_modality_support_is_not_negative_evidence(self):
        virtual, _ = build_pravi_interactions(
            self.interactions,
            self.graphs,
            topk=1,
            shrinkage=1.0,
            uncertainty_kappa=0.0,
            mass_budget_ratio=1.0,
        )
        # User 1's candidate 4 has visual support and no textual support.
        self.assertGreater(virtual[1, 4], 0.0)

    def test_lcb_is_pessimistic_and_fingerprint_tracks_data(self):
        mean_only, _ = build_pravi_interactions(
            self.interactions,
            self.graphs,
            topk=1,
            shrinkage=1.0,
            uncertainty_kappa=0.0,
            mass_budget_ratio=1.0,
        )
        lower_bound, _ = build_pravi_interactions(
            self.interactions,
            self.graphs,
            topk=1,
            shrinkage=1.0,
            uncertainty_kappa=1.0,
            mass_budget_ratio=1.0,
        )
        self.assertLessEqual(float(lower_bound.sum()), float(mean_only.sum()))

        changed = self.interactions.copy()
        changed[0, 2] = 1
        self.assertNotEqual(
            sparse_fingerprint(self.interactions), sparse_fingerprint(changed)
        )

    def test_virtual_operator_uses_only_observed_degrees(self):
        virtual = sp.csr_matrix(([0.5], ([0], [2])), shape=(2, 5))
        operator = build_anchored_virtual_operator(
            self.interactions, virtual, "cpu"
        ).coalesce()
        self.assertEqual(operator._nnz(), 2)
        expected_sum = 2.0 * 0.5 / np.sqrt(2.5 * 1.5)
        self.assertAlmostEqual(float(operator.values().sum()), expected_sum)

        unsupported = sp.csr_matrix(([0.5], ([0], [4])), shape=(2, 5))
        unsupported_operator = build_anchored_virtual_operator(
            self.interactions, unsupported, "cpu"
        ).coalesce()
        self.assertEqual(unsupported_operator._nnz(), 2)
        cold_sum = 2.0 * 0.5 / np.sqrt(2.5 * 0.5)
        self.assertAlmostEqual(float(unsupported_operator.values().sum()), cold_sum)

    def test_transport_evidence_protects_short_history_donors(self):
        candidates, donor_bounds, stats = build_pravi_transport_evidence(
            self.interactions,
            self.graphs,
            topk=1,
            shrinkage=1.0,
            uncertainty_kappa=1.0,
            min_donor_history=3,
        )

        self.assertEqual(donor_bounds.nnz, self.interactions.nnz)
        np.testing.assert_array_equal(donor_bounds.data, 1.0)
        self.assertEqual(
            stats["donor_upper_bound"]["protected_edges"],
            self.interactions.nnz,
        )
        self.assertEqual(stats["candidate_weight_semantics"], "raw_lcb")
        self.assertEqual(candidates.multiply(self.interactions).nnz, 0)

    def test_profile_drift_cost_is_nonzero_without_exact_retrieval(self):
        _, donor_bounds, stats = build_pravi_transport_evidence(
            self.interactions,
            self.graphs,
            topk=1,
            shrinkage=1.0,
            uncertainty_kappa=0.0,
            min_donor_history=2,
            donor_cost_mode="profile_drift",
        )

        self.assertTrue(np.all(donor_bounds.data > 0.0))
        self.assertEqual(
            stats["donor_upper_bound"]["cost_mode"], "profile_drift"
        )
        self.assertTrue(
            all(
                value > 0.0
                for value in stats["donor_upper_bound"]["raw_signal_mean"]
            )
        )

    def test_reliability_estimator_is_train_only_and_finite(self):
        reliability, variance, stats = estimate_pravi_reliability(
            self.interactions,
            self.graphs,
            probe_topl=2,
            shrinkage=1.0,
            reliability_mode="personalized",
        )
        self.assertEqual(reliability.shape, (2, 2))
        self.assertEqual(variance.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(reliability)))
        self.assertTrue(np.all(np.isfinite(variance)))
        self.assertTrue(np.all(reliability > 0.0))
        self.assertTrue(np.all(variance >= 0.0))
        self.assertGreater(stats["crossfit_trials"], 0.0)
        self.assertEqual(stats["method"], "train_crossfit_beta_reliability")

    def test_uniform_reliability_is_a_noop_signal(self):
        reliability, variance, _ = estimate_pravi_reliability(
            self.interactions,
            self.graphs,
            probe_topl=2,
            shrinkage=1.0,
            reliability_mode="uniform",
        )
        np.testing.assert_allclose(
            reliability, np.repeat(reliability[:, :1], reliability.shape[1], axis=1)
        )
        np.testing.assert_allclose(variance, 0.0)


if __name__ == "__main__":
    unittest.main()
