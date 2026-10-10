from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.vi_mmrec import build_vi_interactions


class ViMMRecGraphTest(unittest.TestCase):
    def test_overlay_is_bounded_and_training_only(self):
        interactions = sp.csr_matrix(
            ([1, 1, 1], ([0, 0, 1], [0, 1, 2])), shape=(2, 4)
        )
        visual = sp.csr_matrix(
            ([1, 1, 1, 1], ([0, 1, 1, 2], [1, 0, 3, 3])), shape=(4, 4)
        )
        textual = sp.csr_matrix(
            ([1, 1, 1], ([0, 1, 2], [1, 3, 3])), shape=(4, 4)
        )
        augmented, virtual, stats = build_vi_interactions(
            interactions,
            [visual, textual],
            strategy="overlay",
            lambda_weight=0.01,
            topk=1,
        )
        self.assertEqual(virtual.multiply(interactions).nnz, 0)
        self.assertTrue(np.all(augmented.data <= 1.0))
        self.assertTrue(np.all(virtual.data > 0.0))
        self.assertEqual(stats["method"], "VI-MMRec-O")
        self.assertEqual(len(stats["modality_weights"]), 2)


if __name__ == "__main__":
    unittest.main()
