"""Literal mechanism tests; no recommendation metrics or GPU calls."""

import types
import unittest

import numpy as np

from sampling import (InterestNegativeSampler, NoEligibleNegativeError,
                      _eligible_rank, assemble_group_assets, build_group_assets,
                      train_profiles)


def toy_assets():
    train = np.zeros((8, 6), dtype=np.int64)
    for user, seen in enumerate(([0, 2], [1, 3], [0, 1], [2, 3],
                                 [4], [5], [4], [])):
        train[user, seen] = 1
    return assemble_group_assets(train, np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                                 n_groups=2, seed=2026)


class SamplingTests(unittest.TestCase):
    def test_literal_profile_formula(self):
        image = np.array([[3., 0.], [0., 4.], [0., 0.]])
        text = np.array([[0., 7., 0.], [0., 0., 8.], [0., 0., 0.]])
        histories = (np.array([0, 1]), np.array([2]), np.array([], dtype=np.int64))
        actual = train_profiles(histories, image, text, 3)
        expected = np.array([[.5, .5, 0., .5, .5], [0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0.]])
        expected[0] /= np.linalg.norm(expected[0])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_exact_degree_counts_and_pair_masses_conserved(self):
        assets = toy_assets()
        for degree in np.unique(assets.degrees):
            rows = assets.degrees == degree
            np.testing.assert_array_equal(np.bincount(assets.labels[rows], minlength=2),
                                          np.bincount(assets.shuffled_labels[rows], minlength=2))
        self.assertEqual(assets.diagnostics["group_positive_pair_masses"],
                         assets.diagnostics["shuffled_group_positive_pair_masses"])
        self.assertGreater(assets.diagnostics["assignment_fixed_points"], 0)
        self.assertEqual(assets.diagnostics["assignment_fixed_points"]
                         + assets.diagnostics["assignment_mismatches"], assets.n_users)

    def test_shuffle_reuses_fixed_pools_not_shuffled_rebuild(self):
        train = np.eye(8, dtype=np.int64)
        assets = assemble_group_assets(train, np.arange(8, dtype=np.int64) % 2,
                                      n_groups=2, seed=2026)
        self.assertGreater(assets.diagnostics["assignment_mismatches"], 0)
        for group in range(2):
            np.testing.assert_array_equal(assets.pools[group],
                                          np.flatnonzero(np.arange(8) % 2 == group))
        sampler = InterestNegativeSampler(assets, mode="shuffled")
        self.assertIs(sampler.assets.pools, assets.pools)
        self.assertTrue(assets.diagnostics["shuffled_uses_original_semantic_pools"])

    def test_private_rng_does_not_advance_global_numpy(self):
        np.random.seed(17)
        before = np.random.get_state()
        assets = toy_assets()
        sampler = InterestNegativeSampler(assets, seed=2026)
        sampler.sample(np.arange(8, dtype=np.int64))
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_identical_seed_and_batch_partition_stream(self):
        assets = toy_assets()
        users = np.tile(np.arange(8, dtype=np.int64), 100)
        for mode in ("uniform", "semantic", "shuffled"):
            one = InterestNegativeSampler(assets, mode=mode, seed=2026)
            split = InterestNegativeSampler(assets, mode=mode, seed=2026)
            expected = one.sample(users)
            actual = np.concatenate((split.sample(users[:17]), split.sample(users[17:201]),
                                     split.sample(users[201:])))
            np.testing.assert_array_equal(expected, actual)

    def test_all_modes_exclude_every_own_train_positive(self):
        assets = toy_assets()
        users = np.tile(np.arange(8, dtype=np.int64), 1000)
        for mode in ("uniform", "semantic", "shuffled"):
            negatives = InterestNegativeSampler(assets, mode=mode).sample(users)
            for user in range(assets.n_users):
                self.assertFalse(np.isin(negatives[users == user], assets.histories[user]).any())
            self.assertTrue(np.all((negatives >= 0) & (negatives < assets.n_items)))

    def test_unseen_items_eligible_without_validation_test_inputs(self):
        assets = toy_assets()
        sampler = InterestNegativeSampler(assets, mode="uniform")
        negatives = sampler.sample(np.zeros(10000, dtype=np.int64))
        self.assertEqual(set(negatives.tolist()), {1, 3, 4, 5})
        # These may coincide with unknown validation/test labels.  The sampler
        # never loads them and must not silently filter them using held labels.
        self.assertNotIn("validation", sampler.__dict__)
        self.assertNotIn("test", sampler.__dict__)

    def test_eligible_rank_matches_literal_catalog_complement(self):
        for forbidden in ([], [0], [5], [0, 1], [1, 3], [0, 2, 4, 5]):
            forbidden = np.asarray(forbidden, dtype=np.int64)
            shifted = forbidden - np.arange(forbidden.size)
            eligible = np.setdiff1d(np.arange(6), forbidden)
            actual = [_eligible_rank(rank, shifted) for rank in range(eligible.size)]
            np.testing.assert_array_equal(actual, eligible)

    def test_empty_group_fallback_uses_identical_uniform_stream(self):
        train = np.eye(3, dtype=np.int64)
        assets = assemble_group_assets(train, np.array([0, 1, 1]), n_groups=2)
        users = np.zeros(10000, dtype=np.int64)
        candidate = InterestNegativeSampler(assets, mode="semantic", seed=2026)
        uniform = InterestNegativeSampler(assets, mode="uniform", seed=2026)
        np.testing.assert_array_equal(candidate.sample(users), uniform.sample(users))
        stats = candidate.stats()
        self.assertGreater(stats["empty_group_uniform_fallback_draws"], 4000)
        self.assertEqual(stats["group_actual_draws"], 0)

    def test_no_catalog_eligible_is_explicit_error_without_rng_advance(self):
        train = np.array([[1, 1], [0, 0]], dtype=np.int64)
        assets = assemble_group_assets(train, np.array([0, 1]), n_groups=2)
        sampler = InterestNegativeSampler(assets)
        before = sampler.rng.get_state()
        with self.assertRaises(NoEligibleNegativeError):
            sampler.sample(np.array([0]))
        after = sampler.rng.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(sampler.stats()["draws"], 0)

    def test_cold_user_and_empty_batch_preserved(self):
        assets = toy_assets()
        sampler = InterestNegativeSampler(assets)
        self.assertEqual(sampler.sample(np.empty(0, dtype=np.int64)).shape, (0,))
        negatives = sampler.sample(np.full(10000, 7, dtype=np.int64))
        self.assertTrue(np.all((negatives >= 0) & (negatives < 6)))
        self.assertEqual(sampler.stats()["draws"], 10000)

    def test_deterministic_statistical_mixture(self):
        assets = toy_assets()
        sampler = InterestNegativeSampler(assets, mode="semantic", seed=2026)
        negatives = sampler.sample(np.zeros(100000, dtype=np.int64))
        frequencies = np.bincount(negatives, minlength=6) / negatives.size
        np.testing.assert_allclose(frequencies, [0., .375, 0., .375, .125, .125],
                                   rtol=0, atol=.006)
        stats = sampler.stats()
        self.assertAlmostEqual(stats["group_actual_rate"], .5, delta=.006)
        expected_degree = float(assets.item_degrees[negatives].mean())
        self.assertEqual(stats["drawn_negative_mean_train_degree"], expected_degree)
        self.assertEqual(sum(stats["drawn_negative_train_degree_histogram"].values()), 100000)

    def test_csr_protocol_canonicalization_and_duplicate_rejection(self):
        csr = types.SimpleNamespace(shape=(2, 4), indptr=np.array([0, 3, 4]),
                                    indices=np.array([3, 1, 2, 0]),
                                    data=np.array([1., 1., 0., 1.]))
        assets = assemble_group_assets(csr, np.array([0, 1]), n_groups=2)
        np.testing.assert_array_equal(assets.histories[0], [1, 3])
        np.testing.assert_array_equal(assets.histories[1], [0])
        csr.indices[0] = 1
        with self.assertRaisesRegex(ValueError, "duplicate"):
            assemble_group_assets(csr, np.array([0, 1]), n_groups=2)

    def test_inputs_arrays_immutable_and_sha_repeatable(self):
        first, second = toy_assets(), toy_assets()
        self.assertEqual(first.diagnostics, second.diagnostics)
        for values in (first.labels, first.shuffled_labels, first.degrees,
                       first.item_degrees, first.universe, first.control_user_permutation,
                       *first.histories, *first.pools):
            self.assertFalse(values.flags.writeable)

    def test_noncontiguous_train_universe_excludes_catalog_only_items(self):
        train = np.zeros((4, 9), dtype=np.int64)
        train[0, [1, 5]] = 1
        train[1, [3, 8]] = 1
        train[2, [1, 3]] = 1
        assets = assemble_group_assets(train, np.array([0, 0, 1, 1]), n_groups=2)
        np.testing.assert_array_equal(assets.universe, [1, 3, 5, 8])
        for mode in ("uniform", "semantic", "shuffled"):
            sampler = InterestNegativeSampler(assets, mode=mode)
            users = np.tile(np.arange(4, dtype=np.int64), 1000)
            negatives = sampler.sample(users)
            self.assertTrue(np.isin(negatives, [1, 3, 5, 8]).all())
            for user in range(4):
                self.assertFalse(np.isin(negatives[users == user], assets.histories[user]).any())

    def test_source_permutation_fixed_points_distinct_from_unchanged_labels(self):
        assets = toy_assets()
        np.testing.assert_array_equal(assets.degrees[assets.control_user_permutation], assets.degrees)
        np.testing.assert_array_equal(assets.shuffled_labels,
                                      assets.labels[assets.control_user_permutation])
        exact_fixed = int(np.sum(assets.control_user_permutation == np.arange(assets.n_users)))
        self.assertEqual(exact_fixed, assets.diagnostics["permutation_source_user_fixed_points"])
        self.assertLessEqual(exact_fixed, assets.diagnostics["assignment_fixed_points"])

    def test_numpy_kmeans_repeatable_and_global_rng_unchanged(self):
        train = np.eye(8, dtype=np.int64)
        image = np.array([[1, 0]] * 4 + [[0, 1]] * 4, dtype=np.float64)
        text = image.copy()
        np.random.seed(23)
        before = np.random.get_state()
        first = build_group_assets(train, image, text, n_groups=2, kmeans_backend="numpy")
        second = build_group_assets(train, image, text, n_groups=2, kmeans_backend="numpy")
        after = np.random.get_state()
        np.testing.assert_array_equal(first.labels, second.labels)
        self.assertEqual(first.diagnostics, second.diagnostics)
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertNotEqual(first.labels[0], first.labels[7])
        self.assertEqual(first.diagnostics["grouping"]["n_init"], 10)

    def test_auto_backend_used_in_actual_environment_is_repeatable(self):
        train = np.eye(8, dtype=np.int64)
        image = np.array([[1, 0]] * 4 + [[0, 1]] * 4, dtype=np.float64)
        np.random.seed(29)
        before = np.random.get_state()
        first = build_group_assets(train, image, image, n_groups=2)
        second = build_group_assets(train, image, image, n_groups=2)
        after = np.random.get_state()
        self.assertEqual(first.diagnostics, second.diagnostics)
        np.testing.assert_array_equal(first.labels, second.labels)
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertIn(first.diagnostics["grouping"]["backend"],
                      ("numpy_explicit_kmeanspp_lloyd_v1", "sklearn_kmeans_lloyd"))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            assemble_group_assets(np.array([[0, 2]]), np.array([0]), n_groups=2)
        with self.assertRaises(ValueError):
            build_group_assets(np.eye(8), np.eye(7), np.eye(8))
        with self.assertRaises(ValueError):
            build_group_assets(np.eye(8), np.eye(8) * np.nan, np.eye(8))
        sampler = InterestNegativeSampler(toy_assets())
        for users in (np.array([-1]), np.array([8]), np.array([1.5]), np.array([[1]])):
            with self.assertRaises(ValueError):
                sampler.sample(users)


if __name__ == "__main__":
    unittest.main()
