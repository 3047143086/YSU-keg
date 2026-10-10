import importlib.util
import unittest
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location("alignment_core", Path(__file__).with_name("core.py"))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


class BankGeometryTests(unittest.TestCase):
    def test_query_owner_review_is_subtracted_before_normalization(self):
        bank = core.item_bank(np.eye(2), np.array([0, 1]), np.array([3, 3]), 3, 5)
        scores, available, removed = core.bank_scores(np.eye(2), bank, np.array([0]), np.array([[3]]))
        self.assertAlmostEqual(scores[0, 0], 0.)
        self.assertTrue(available[0, 0])
        self.assertEqual(removed, 1)
        np.testing.assert_array_equal(bank["sums"][3], [1., 1.])

    def test_repeated_candidates_do_not_subtract_owner_twice_from_same_bank(self):
        bank = core.item_bank(np.eye(2), np.array([0, 1]), np.array([3, 3]), 3, 5)
        scores, available, removed = core.bank_scores(np.eye(2), bank, np.array([0]), np.array([[3, 3]]))
        np.testing.assert_array_equal(scores, [[0., 0.]])
        self.assertTrue(available.all())
        self.assertEqual(removed, 2)
        np.testing.assert_array_equal(bank["counts"], [0, 0, 0, 2, 0])

    def test_only_owner_review_leaves_empty_bank_and_neutral_auc(self):
        bank = core.item_bank(np.array([[1., 0.]]), np.array([0]), np.array([3]), 2, 5)
        scores, available, removed = core.bank_scores(np.array([[1., 0.]]), bank,
                                                     np.array([0]), np.full((1, 17), 3))
        np.testing.assert_array_equal(scores, np.zeros((1, 17)))
        self.assertFalse(available.any())
        self.assertEqual(removed, 17)
        self.assertEqual(core.auc(scores)[0], .5)

    def test_zero_vectors_are_retained_without_content_fallback(self):
        bank = core.item_bank(np.zeros((2, 2)), np.array([0, 1]), np.array([3, 3]), 3, 5)
        scores, available, removed = core.bank_scores(np.array([[1., 0.], [0., 1.], [0., 0.]]),
                                                     bank, np.array([2]), np.full((1, 17), 3))
        np.testing.assert_array_equal(scores, np.zeros((1, 17)))
        self.assertFalse(available.any())
        self.assertEqual(removed, 0)

    def test_blocked_scores_equal_unblocked_and_no_cartesian_expansion(self):
        users, items = np.array([0, 1, 2]), np.array([0, 0, 1])
        bank = core.item_bank(np.eye(3), users, items, 3, 3)
        queries, choices = np.array([0, 1, 2]), np.array([[0, 1], [1, 0], [0, 1]])
        first = core.bank_scores(np.eye(3), bank, queries, choices, block_size=1)
        second = core.bank_scores(np.eye(3), bank, queries, choices, block_size=128)
        for index in (0, 1): np.testing.assert_array_equal(first[index], second[index])
        self.assertEqual(first[0].shape, choices.shape)

    def test_duplicate_edges_and_nonfinite_reviews_are_rejected(self):
        with self.assertRaises(ValueError):
            core.item_bank(np.eye(2), np.array([0, 0]), np.array([3, 3]), 3, 5)
        with self.assertRaises(ValueError):
            core.item_bank(np.array([[np.nan, 0]]), np.array([0]), np.array([3]), 3, 5)

    def test_target_or_negative_support_leakage_is_rejected(self):
        support = core.binary_support(np.array([0, 1]), np.array([1, 2]), 3, 20)
        choices = np.tile(np.arange(3, 20), (2, 1))
        users, targets = np.array([0, 1]), choices[:, 0]
        first_hash = core.validate_pairs(support, users, targets, choices)
        self.assertEqual(first_hash, core.validate_pairs(support, users, targets, choices))
        choices[0, 3] = 1
        with self.assertRaisesRegex(ValueError, "survived"):
            core.validate_pairs(support, users, targets, choices)

    def test_true_target_must_be_column_zero_and_users_unique(self):
        support = core.binary_support(np.array([0]), np.array([1]), 3, 20)
        choices = np.tile(np.arange(3, 20), (2, 1))
        with self.assertRaises(ValueError):
            core.validate_pairs(support, np.array([0, 0]), choices[:, 0], choices)
        with self.assertRaises(ValueError):
            core.validate_pairs(support, np.array([0, 1]), np.array([4, 4]), choices)

    def test_auc_retains_all_16_comparisons_and_half_credit_ties(self):
        values = np.zeros((1, 17))
        values[0, 9:] = -1
        self.assertEqual(core.auc(values)[0], .75)
        with self.assertRaises(ValueError): core.auc(np.zeros((2, 16)))


class JointGateTests(unittest.TestCase):
    def setUp(self):
        self.users = np.arange(32)
        self.values = {name:np.full(32, .6) for name in core.ARMS}
        self.values["candidate"][:] = .7

    def test_all_eight_controls_must_pass_without_best_arm_selection(self):
        decision, halves = core.statistics(self.values, .8, self.users, bootstrap=40)
        self.assertTrue(decision["passed"])
        self.assertEqual(set(decision["paired_vs_controls"]), set(core.CONTROLS))
        self.values["itemknn"][:] = .696
        decision, _ = core.statistics(self.values, .8, self.users, bootstrap=40)
        self.assertFalse(decision["passed"])
        self.assertIn("itemknn:mean_gain", decision["failed_conditions"])
        self.assertFalse(decision["formal_training_authorized"])

    def test_fixed_coverage_and_above_chance_gates(self):
        self.assertFalse(core.statistics(self.values, .799999, self.users, bootstrap=40)[0]["passed"])
        values = {name:np.full(32, .49) for name in core.ARMS}
        values["candidate"][:] = .504
        self.assertFalse(core.statistics(values, 1., self.users, bootstrap=40)[0]["passed"])

    def test_negative_hash_half_prevents_passing_positive_overall_delta(self):
        _, halves = core.statistics(self.values, 1., self.users, bootstrap=40)
        self.values["raw_review_bank"][halves == 0] = .71
        decision, _ = core.statistics(self.values, 1., self.users, bootstrap=40)
        self.assertFalse(decision["passed"])
        self.assertIn("raw_review_bank:both_hash_halves", decision["failed_conditions"])

    def test_missing_control_and_nonfinite_probability_are_technical_errors(self):
        del self.values["itemknn"]
        with self.assertRaises(ValueError): core.statistics(self.values, 1., self.users, bootstrap=40)
        self.values["itemknn"] = np.full(32, np.nan)
        with self.assertRaises(ValueError): core.statistics(self.values, 1., self.users, bootstrap=40)
        self.values["itemknn"] = np.full(32, 1.1)
        with self.assertRaises(ValueError): core.statistics(self.values, 1., self.users, bootstrap=40)


if __name__ == "__main__":
    unittest.main()
