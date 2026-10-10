import sys
import unittest
from pathlib import Path

import numpy as np
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from utils.ecli import (  # noqa: E402
    apply_local_evidence,
    build_history_table,
    mean_ci95,
    per_user_ranking_metrics,
    topq_history_similarity,
    validation_gate,
)


class ECLITest(unittest.TestCase):
    def test_history_table_is_unique_and_padded(self):
        table, mask = build_history_table(
            np.array([0, 0, 0, 1]),
            np.array([2, 2, 3, 4]),
            n_users=3,
        )
        self.assertEqual(table.shape, (3, 2))
        self.assertEqual(table[0].tolist(), [2, 3])
        self.assertEqual(mask.sum(dim=1).tolist(), [2, 1, 0])

    def test_topq_ignores_padding_and_uses_multiple_interests(self):
        features = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.6, 0.8]]
        )
        features = torch.nn.functional.normalize(features, dim=1)
        candidates = torch.tensor([[2, 3]])
        history = torch.tensor([[0, 1, 0]])
        mask = torch.tensor([[True, True, False]])
        score = topq_history_similarity(
            candidates, history, mask, features, top_q=2
        )
        self.assertTrue(torch.allclose(score, torch.tensor([[0.5, 0.7]])))

    def test_constant_evidence_is_identity(self):
        scores = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        evidence = torch.ones_like(scores)
        refined = apply_local_evidence(scores, evidence, anchor_k=4)
        self.assertTrue(torch.allclose(refined, scores))

    def test_unrounded_per_user_metrics(self):
        hits = np.array([[True, False, True], [False, True, False]])
        recall, ndcg = per_user_ranking_metrics(hits, np.array([2, 1]), 2)
        self.assertTrue(np.allclose(recall, [0.5, 1.0]))
        expected_first = 1.0 / (1.0 + 1.0 / np.log2(3.0))
        self.assertTrue(np.allclose(ndcg, [expected_first, 1.0 / np.log2(3.0)]))

    def test_paired_ci_contains_constant_delta(self):
        summary = mean_ci95(np.array([0.25, 0.25, 0.25]))
        self.assertEqual(summary["mean"], 0.25)
        self.assertEqual(summary["ci95_low"], 0.25)
        self.assertEqual(summary["ci95_high"], 0.25)

    def test_gate_requires_both_primary_metrics(self):
        baseline = {
            "recall@10": 0.07,
            "recall@20": 0.10,
            "ndcg@10": 0.04,
            "ndcg@20": 0.05,
        }
        treatment = dict(baseline, **{"recall@20": 0.101, "ndcg@20": 0.0505})
        self.assertTrue(validation_gate(baseline, treatment)["passed"])
        treatment["ndcg@20"] = 0.0499
        self.assertFalse(validation_gate(baseline, treatment)["passed"])


if __name__ == "__main__":
    unittest.main()
