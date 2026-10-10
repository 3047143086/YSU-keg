"""CPU-only fixtures for the frozen feature perturbation contract."""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from precheck import (EPSILON_REL, bpr_gap, gate_pass, random_deltas,
                      replace_rows, targeted_deltas)


class PrecheckFixtures(unittest.TestCase):
    def test_functional_replace_and_exact_restore(self):
        base = np.arange(20, dtype=np.float64).reshape(5, 4)
        before = base.copy()
        rows = np.array([1, 3], dtype=np.int64)
        delta = np.full((2, 4), 0.25)
        changed = replace_rows(base, rows, delta)
        np.testing.assert_array_equal(base, before)
        np.testing.assert_allclose(changed[rows], before[rows] + delta)
        restored = replace_rows(changed, rows, -delta)
        np.testing.assert_array_equal(restored, before)

    def test_target_and_random_have_identical_row_norms(self):
        features = np.array([[3., 4., 0.], [0., 5., 12.]], dtype=np.float64)
        gradients = np.array([[1., 2., 2.], [4., 0., 3.]], dtype=np.float64)
        target = targeted_deltas(features, gradients)
        random = random_deltas(features, np.random.RandomState(20261030))
        expected = EPSILON_REL * np.linalg.norm(features, axis=1)
        np.testing.assert_allclose(np.linalg.norm(target, axis=1), expected)
        np.testing.assert_allclose(np.linalg.norm(random, axis=1), expected)

    def test_two_of_three_gate_and_gap(self):
        self.assertTrue(gate_pass([True, False, True]))
        self.assertFalse(gate_pass([True, False, False]))
        self.assertGreater(bpr_gap(1.0, 1.2), 0)
        self.assertLess(bpr_gap(1.0, 0.8), 0)

    def test_zero_rows_are_rejected(self):
        with self.assertRaises(ValueError):
            targeted_deltas(np.zeros((1, 2)), np.ones((1, 2)))
        with self.assertRaises(ValueError):
            targeted_deltas(np.ones((1, 2)), np.zeros((1, 2)))


if __name__ == "__main__":
    unittest.main()
