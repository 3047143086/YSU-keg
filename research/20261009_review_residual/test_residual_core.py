import importlib.util
import unittest
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location("residual_core", Path(__file__).with_name("residual_core.py"))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


class ResidualCoreTests(unittest.TestCase):
    def test_parallel_rows_are_zero_after_roundoff_guard(self):
        z = np.array([[1.0, 0.0, 0.0], [0.0, -3.0, 0.0]], dtype=np.float32)
        t = z.copy()
        result = core.residualize(z, t)
        np.testing.assert_array_equal(result, np.zeros_like(result))

    def test_zero_item_keeps_normalized_review(self):
        z = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        result = core.residualize(z, np.zeros_like(z))
        np.testing.assert_allclose(result[0], [0.6, 0.8], atol=1e-6)
        np.testing.assert_array_equal(result[1], [0.0, 0.0])

    def test_orthogonal_and_antiparallel(self):
        z = np.array([[0.0, 2.0], [0.0, -2.0]], dtype=np.float64)
        t = np.array([[3.0, 0.0], [3.0, 0.0]], dtype=np.float64)
        result = core.residualize(z, t)
        np.testing.assert_allclose(result, [[0.0, 1.0], [0.0, -1.0]], atol=1e-6)

    def test_scale_of_item_direction_does_not_change_result(self):
        z = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        t = np.array([[4.0, 0.0, 0.0]], dtype=np.float64)
        np.testing.assert_allclose(core.residualize(z, t), core.residualize(z, 17 * t), atol=1e-6)

    def test_profile_and_scores(self):
        vectors = np.array([[1., 0.], [0., 1.], [1., 0.]])
        profiles = core.profile(vectors, np.array([0, 1, 0]), 2)
        np.testing.assert_allclose(profiles[0], [1., 0.], atol=1e-6)
        targets = np.array([[1., 0.], [0., 1.]])
        scores = core.score_rows(profiles, targets, np.array([0, 1]), np.array([[0, 1], [1, 0]]))
        np.testing.assert_allclose(scores, [[1., 0.], [1., 0.]], atol=1e-6)

    def test_shape_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            core.residualize([[1., 2.]], [[1., 2., 3.]])
        with self.assertRaises(ValueError):
            core.normalize_rows([[float("nan")]])
        with self.assertRaises(ValueError):
            core.score_rows(np.zeros((1, 2)), np.zeros((1, 2)), [0], [[1]])


if __name__ == "__main__":
    unittest.main()
