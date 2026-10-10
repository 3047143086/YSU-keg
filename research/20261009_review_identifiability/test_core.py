import importlib.util
import unittest
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location("ident_core", Path(__file__).with_name("core.py"))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


class IdentifiabilityTests(unittest.TestCase):
    def test_split_is_deterministic_and_edges_must_be_unique(self):
        users = np.array([0, 0, 1, 1], dtype=np.int64)
        items = np.array([0, 1, 0, 1], dtype=np.int64)
        first = core.split_edges(users, items)
        np.testing.assert_array_equal(first, core.split_edges(users, items))
        with self.assertRaises(ValueError):
            core.plan_groups(np.array([0, 0]), np.array([1, 1]), 2)

    def test_population_keeps_users_without_cached_edges(self):
        users = np.array([0, 0, 1, 1], dtype=np.int64)
        items = np.array([0, 1, 0, 1], dtype=np.int64)
        plan = core.plan_groups(users, items, 4, np.array([True, False, True, False]))
        np.testing.assert_array_equal(plan["population"], np.arange(4))
        self.assertEqual(len(plan["b_counts"]), 4)

    def test_query_item_cannot_appear_in_own_a_history(self):
        users = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        items = np.array([0, 1, 9, 2, 3, 9], dtype=np.int64)
        is_a = np.array([True, True, False, True, True, False])
        plan = core.plan_groups(users, items, 2, is_a)
        for group in plan["groups"]:
            for owner in group["owners"]:
                self.assertNotIn(group["item"], items[is_a & (users == owner)])

    def test_exact_degree_groups_exclude_mismatched_authors(self):
        users = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
        items = np.array([0, 1, 9, 0, 1, 9, 0, 1, 2, 9], dtype=np.int64)
        is_a = np.array([True, True, False, True, True, False, True, True, True, False])
        plan = core.plan_groups(users, items, 3, is_a)
        self.assertEqual([(g["item"], g["a_degree"], len(g["indices"]))
                          for g in plan["groups"]], [(9, 2, 2)])
        self.assertEqual(int(plan["scorable_queries"].sum()), 2)

    def test_null_is_group_local_and_fixed_points_are_recorded(self):
        users = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        items = np.array([0, 1, 0, 2, 0, 3], dtype=np.int64)
        is_a = np.array([True, True, True, True, True, True])
        # Add one B event per author for item 4 and exact degree two.
        users = np.r_[users, [0, 1, 2]]
        items = np.r_[items, [4, 4, 4]]
        is_a = np.r_[is_a, [False, False, False]]
        plan = core.plan_groups(users, items, 3, is_a)
        self.assertEqual(len(plan["groups"]), 1)
        self.assertEqual(plan["null_assignments"], 3)
        self.assertGreaterEqual(plan["null_fixed_points"], 0)
        np.testing.assert_array_equal(np.sort(plan["groups"][0]["permutation"]), np.arange(3))

    def test_auc_excludes_true_column_and_ties_are_half(self):
        self.assertAlmostEqual(core._row_auc(np.array([1., 1., 0.]), 0), .75)
        self.assertAlmostEqual(core._row_auc(np.array([0., 0., 0.]), 2), .5)

    def test_real_and_null_retrieval_are_scored_without_full_matrix(self):
        users = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        items = np.array([0, 1, 0, 2, 0, 3], dtype=np.int64)
        is_a = np.array([True, True, True, True, True, True])
        users = np.r_[users, [0, 1, 2]]
        items = np.r_[items, [4, 4, 4]]
        is_a = np.r_[is_a, [False, False, False]]
        plan = core.plan_groups(users, items, 3, is_a)
        vectors = np.eye(3, dtype=np.float64)[users % 3]
        result = core.evaluate_view(vectors, plan, block_size=1)
        self.assertEqual(result["query_real"].shape, (len(users),))
        self.assertEqual(result["query_null"].shape, (len(users),))
        np.testing.assert_array_equal(result["user_real"], np.ones(3))
        permutation = plan["groups"][0]["permutation"]
        np.testing.assert_allclose(result["user_null"], np.where(permutation == np.arange(3), 1., .25))
        blocked = core.evaluate_view(vectors, plan, block_size=100)
        np.testing.assert_array_equal(blocked["user_null"], result["user_null"])

    def test_summary_requires_both_hash_halves_and_real_chance_gate(self):
        plan = {"population": np.arange(16), "scorable_users": np.ones(16, dtype=bool)}
        values = {"user_real": np.full(16, .51), "user_null": np.full(16, .50)}
        summary = core.summarize(values, plan, bootstrap=20)
        self.assertTrue(summary["passed"])
        self.assertFalse(summary["preference_established"])
        values = {"user_real": np.full(16, .504), "user_null": np.full(16, .49)}
        self.assertFalse(core.summarize(values, plan, bootstrap=20)["passed"])
        with self.assertRaises(ValueError):
            core.summarize({"user_real": np.array([.6]), "user_null": np.array([.5])},
                           {"population": np.array([0]), "scorable_users": np.array([True])}, bootstrap=20)

    def test_ineligible_and_missing_users_are_neutral(self):
        users = np.array([0, 0, 1, 1], dtype=np.int64)
        items = np.array([0, 1, 0, 1], dtype=np.int64)
        plan = core.plan_groups(users, items, 4, np.array([True, False, True, False]))
        result = core.evaluate_view(np.ones((4, 2)), plan)
        np.testing.assert_array_equal(result["user_real"], np.full(4, .5))
        np.testing.assert_array_equal(result["user_null"], np.full(4, .5))

    def test_nonfinite_vectors_and_bad_block_size_rejected(self):
        with self.assertRaises(ValueError):
            core.normalize_rows(np.array([[np.nan, 0.]]))
        users = np.array([0, 0], dtype=np.int64)
        items = np.array([0, 1], dtype=np.int64)
        plan = core.plan_groups(users, items, 1, np.array([True, False]))
        with self.assertRaises(ValueError):
            core.evaluate_view(np.ones((2, 2)), plan, block_size=0)

    def test_gate_requires_coverage_and_strict_positive_lower_bound(self):
        plan = {"population": np.arange(16), "scorable_users": np.r_[np.ones(3, dtype=bool),
                                                                    np.zeros(13, dtype=bool)]}
        values = {"user_real": np.full(16, .6), "user_null": np.full(16, .5)}
        self.assertFalse(core.summarize(values, plan, bootstrap=20)["passed"])
        plan["scorable_users"][:] = True
        tied = {"user_real": np.full(16, .6), "user_null": np.full(16, .6)}
        summary = core.summarize(tied, plan, bootstrap=20)
        self.assertEqual(summary["paired_user_bootstrap_95ci"], [0., 0.])
        self.assertFalse(summary["passed"])

    def test_summary_rejects_missing_nonfinite_or_out_of_range_auc(self):
        plan = {"population": np.arange(16), "scorable_users": np.ones(16, dtype=bool)}
        for values in ({"user_real": np.full(15, .6), "user_null": np.full(15, .5)},
                       {"user_real": np.full(16, np.nan), "user_null": np.full(16, .5)},
                       {"user_real": np.full(16, 1.1), "user_null": np.full(16, .5)}):
            with self.assertRaises(ValueError):
                core.summarize(values, plan, bootstrap=20)

    def test_each_users_unscorable_b_queries_are_retained_at_chance(self):
        users = np.array([0, 0, 1, 1, 0, 1, 0])
        items = np.array([0, 1, 0, 1, 2, 2, 3])
        mask = np.array([True, True, True, True, False, False, False])
        plan = core.plan_groups(users, items, 3, mask)
        vectors = np.eye(2)[users % 2]
        result = core.evaluate_view(vectors, plan)
        self.assertAlmostEqual(result["user_real"][0], .75)
        self.assertAlmostEqual(result["user_real"][1], 1.)
        self.assertAlmostEqual(result["user_real"][2], .5)


if __name__ == "__main__":
    unittest.main()
