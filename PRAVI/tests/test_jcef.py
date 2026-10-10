from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.jcef import (
    RidgeLogisticCalibrator,
    diagnostic_gate,
    evidence_features,
)


class JCEFTest(unittest.TestCase):
    def test_evidence_features_encode_agreement_and_conflict(self):
        features = evidence_features([2.0, 2.0], [2.0, -2.0], True)
        np.testing.assert_allclose(features[:, 2], [4.0, -4.0])
        np.testing.assert_allclose(features[:, 3], [0.0, 4.0])

    def test_ridge_logistic_separates_synthetic_pairs(self):
        positive = np.asarray([[1.2, 1.0], [1.0, 1.1], [0.9, 1.3]])
        negative = np.asarray([[-1.0, -0.8], [-0.9, -1.2], [-1.1, -1.0]])
        x = np.vstack((positive, negative))
        y = np.asarray([1, 1, 1, 0, 0, 0])
        first = RidgeLogisticCalibrator.fit(x, y, l2=0.01)
        second = RidgeLogisticCalibrator.fit(x, y, l2=0.01)
        scores = first.decision_function(x)
        self.assertGreater(scores[:3].min(), scores[3:].max())
        np.testing.assert_allclose(first.coefficients, second.coefficients)

    def test_gate_requires_synergy_not_only_reweighting(self):
        self.assertTrue(
            diagnostic_gate(0.64, 0.008, 0.003, 0.003, 0.001, [0.006, 0.010])[
                "passed"
            ]
        )
        self.assertFalse(
            diagnostic_gate(0.64, 0.008, 0.003, 0.001, 0.0005, [0.006, 0.010])[
                "passed"
            ]
        )


if __name__ == "__main__":
    unittest.main()
