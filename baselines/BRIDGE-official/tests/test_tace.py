import sys
import unittest
from pathlib import Path

import numpy as np
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from utils.tace import TemporalAffinityScorer, build_temporal_affinity  # noqa: E402


class TACETest(unittest.TestCase):
    def test_temporal_distance_downweights_far_pairs(self):
        affinity, train, stats = build_temporal_affinity(
            [0, 0, 0], [0, 1, 2], [10, 20, 30], n_users=1, n_items=3
        )
        self.assertAlmostEqual(float(affinity[0, 1]), 1.0)
        self.assertAlmostEqual(float(affinity[1, 2]), 1.0)
        self.assertAlmostEqual(float(affinity[0, 2]), 0.5)
        self.assertEqual(train.nnz, 3)
        self.assertEqual(stats["raw_unordered_pairs"], 3)

    def test_equal_timestamps_share_an_episode(self):
        affinity, _, _ = build_temporal_affinity(
            [0, 0, 0], [0, 1, 2], [10, 10, 30], n_users=1, n_items=3
        )
        self.assertAlmostEqual(float(affinity[0, 1]), 1.0)
        self.assertAlmostEqual(float(affinity[0, 2]), 1.0)
        self.assertAlmostEqual(float(affinity[1, 2]), 1.0)

    def test_repeated_user_item_keeps_earliest_event(self):
        affinity, train, _ = build_temporal_affinity(
            [0, 0, 0, 0], [0, 0, 1, 2], [10, 40, 20, 30], 1, 3
        )
        self.assertEqual(train.nnz, 3)
        self.assertAlmostEqual(float(affinity[0, 2]), 0.5)

    def test_candidate_scores_match_bridge_normalization(self):
        affinity, train, _ = build_temporal_affinity(
            [0, 0, 1, 1, 2, 2],
            [0, 1, 0, 2, 1, 3],
            [10, 20, 10, 20, 10, 30],
            n_users=3,
            n_items=4,
        )
        scorer = TemporalAffinityScorer(train, affinity, norm="z")
        scores = scorer.score(torch.tensor([0]), torch.tensor([[1, 2, 3]]))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertAlmostEqual(float(scores.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(scores.std(unbiased=False)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
