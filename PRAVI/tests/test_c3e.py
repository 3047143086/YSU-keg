from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.c3e import (
    EvidenceCalibrator,
    deterministic_user_folds,
    diagnostic_gate,
    topq_mean,
    weighted_isotonic,
)


class C3ETest(unittest.TestCase):
    def test_weighted_isotonic_is_monotone(self):
        fitted = weighted_isotonic([0.0, 2.0, 1.0, 3.0], [1, 1, 1, 1])
        np.testing.assert_allclose(fitted, [0.0, 1.5, 1.5, 3.0])
        self.assertTrue(np.all(np.diff(fitted) >= 0.0))

    def test_calibrator_is_monotone_and_finite(self):
        degrees = np.asarray([1, 2, 3, 4, 8, 16], dtype=np.float32)
        positive_left = np.asarray([0, 1, 2, 3, 4, 5])
        positive_right = np.asarray([1, 2, 3, 4, 5, 0])
        negative_left = np.asarray([0, 1, 2, 3, 4, 5])
        negative_right = np.asarray([3, 4, 5, 0, 1, 2])
        calibrator = EvidenceCalibrator.fit(
            [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2],
            positive_left,
            positive_right,
            negative_left,
            negative_right,
            degrees,
            degree_edges=np.asarray([1.2, 2.0]),
            n_similarity_bins=4,
            shrinkage=2.0,
        )
        self.assertTrue(np.all(np.isfinite(calibrator.cell_curves)))
        self.assertTrue(np.all(np.diff(calibrator.global_curve) >= -1e-7))
        self.assertTrue(
            np.all(np.diff(calibrator.cell_curves, axis=-1) >= -1e-7)
        )

    def test_topq_mean_uses_multiple_interests(self):
        values = np.asarray([[0.9, 0.8, -0.5], [0.4, 0.3, 0.2]])
        np.testing.assert_allclose(topq_mean(values, 2), [0.85, 0.35])

    def test_folds_are_deterministic_and_gate_is_strict(self):
        first = deterministic_user_folds(np.arange(100), 2, 2026)
        second = deterministic_user_folds(np.arange(100), 2, 2026)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(first.tolist()), {0, 1})
        self.assertTrue(
            diagnostic_gate(0.63, 0.025, 0.01, 0.006, 0.001, [0.02, 0.03])[
                "passed"
            ]
        )
        self.assertFalse(
            diagnostic_gate(0.63, 0.025, 0.01, 0.004, 0.001, [0.02, 0.03])[
                "passed"
            ]
        )


if __name__ == "__main__":
    unittest.main()
