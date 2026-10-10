from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.rating_signal import (
    build_profile_operators,
    diagnostic_gate,
    event_pairwise_auc,
    shuffle_values_within_users,
)


class RatingSignalTest(unittest.TestCase):
    def test_exposure_uses_inverse_item_degree(self):
        exposure, _, _ = build_profile_operators(
            [0, 0, 1], [0, 1, 0], [5, 1, 4], 2, 2
        )
        np.testing.assert_allclose(
            exposure.toarray(), [[0.5, 1.0], [0.5, 0.0]]
        )

    def test_rating_residual_is_user_centered_and_shift_invariant(self):
        _, rating_a, _ = build_profile_operators(
            [0, 0, 1, 1], [0, 1, 0, 2], [1, 5, 2, 4], 2, 3
        )
        _, rating_b, _ = build_profile_operators(
            [0, 0, 1, 1], [0, 1, 0, 2], [2, 6, 5, 7], 2, 3
        )
        np.testing.assert_allclose(
            rating_a.toarray(), rating_b.toarray(), atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(rating_a.sum(axis=1)).ravel(), 0.0, atol=1e-6
        )

    def test_constant_user_ratings_have_no_polarity(self):
        _, rating, _ = build_profile_operators(
            [0, 0, 1], [0, 1, 2], [5, 5, 3], 2, 3
        )
        np.testing.assert_allclose(rating.toarray(), 0.0)

    def test_shuffle_is_deterministic_and_preserves_user_values(self):
        users = np.array([1, 0, 1, 0, 1])
        values = np.array([2, 4, 3, 5, 1])
        first = shuffle_values_within_users(users, values, 2026)
        second = shuffle_values_within_users(users, values, 2026)
        np.testing.assert_array_equal(first, second)
        for user in np.unique(users):
            np.testing.assert_array_equal(
                np.sort(first[users == user]), np.sort(values[users == user])
            )

    def test_pairwise_auc_and_gate(self):
        auc = event_pairwise_auc([2.0, 0.0], [[1.0, 2.0], [1.0, -1.0]])
        np.testing.assert_allclose(auc, [0.75, 0.5])
        gate = diagnostic_gate(0.8, 0.56, 0.52, 0.01, 0.003)
        self.assertTrue(gate["passed"])
        self.assertFalse(
            diagnostic_gate(0.8, 0.56, 0.55, 0.01, 0.003)["passed"]
        )


if __name__ == "__main__":
    unittest.main()
