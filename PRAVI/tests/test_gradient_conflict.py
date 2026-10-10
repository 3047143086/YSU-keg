from __future__ import annotations

import sys
import unittest
from collections import OrderedDict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.gradient_conflict import (
    ALIGNMENT_COMPONENTS,
    gradient_statistics,
    loss_gradients,
    sum_loss_components,
    summarize_probe_records,
    weighted_msca_loss_components,
)


class _Weights:
    cl_weight = 0.25
    reg_weight = 0.10


def _observation(cosine, norm_ratio, dot):
    return {
        "cosine": cosine,
        "dot": dot,
        "reference_norm": 1.0,
        "auxiliary_norm": norm_ratio,
        "norm_ratio": norm_ratio,
        "used_parameters": 2,
    }


class GradientConflictTest(unittest.TestCase):
    def test_gradient_statistics_distinguishes_alignment_and_conflict(self):
        parameter = torch.tensor([1.0, 2.0], requires_grad=True)
        rank = parameter[0] + 2.0 * parameter[1]
        aligned = 2.0 * parameter[0] + 4.0 * parameter[1]
        opposed = -parameter[0] - 2.0 * parameter[1]

        rank_gradient = loss_gradients(rank, [parameter], retain_graph=True)
        aligned_gradient = loss_gradients(aligned, [parameter], retain_graph=True)
        opposed_gradient = loss_gradients(opposed, [parameter], retain_graph=True)

        aligned_stats = gradient_statistics(rank_gradient, aligned_gradient)
        opposed_stats = gradient_statistics(rank_gradient, opposed_gradient)
        self.assertAlmostEqual(aligned_stats["cosine"], 1.0, places=6)
        self.assertAlmostEqual(aligned_stats["norm_ratio"], 2.0, places=6)
        self.assertAlmostEqual(opposed_stats["cosine"], -1.0, places=6)

    def test_weighted_components_reconstruct_expected_total(self):
        raw = OrderedDict((name, torch.tensor(1.0)) for name in ("rank",))
        for name in ALIGNMENT_COMPONENTS:
            raw[name] = torch.tensor(2.0)
        raw["regularization"] = torch.tensor(3.0)
        raw["feature_learning"] = torch.tensor(4.0)

        weighted = weighted_msca_loss_components(_Weights(), raw)
        total = sum_loss_components(weighted)
        self.assertAlmostEqual(float(total), 8.3, places=6)

    def test_summary_gate_requires_material_repeated_conflict(self):
        records = []
        for probe_index in range(4):
            terms = OrderedDict()
            for term_index, name in enumerate(ALIGNMENT_COMPONENTS):
                if term_index < 2:
                    value = _observation(-0.4, 0.08, -0.04)
                else:
                    value = _observation(0.2, 0.08, 0.02)
                terms[name] = {"id_backbone": value}
            terms["alignment_total"] = {
                "id_backbone": _observation(-0.2, 0.10, -0.02)
            }
            records.append(
                {"epoch": 1, "batch": probe_index, "terms": terms}
            )

        summary = summarize_probe_records(records)
        self.assertTrue(summary["gate"]["passed"])
        self.assertEqual(
            summary["gate"]["material_terms"],
            ["align_user_image", "align_user_text"],
        )

    def test_summary_gate_rejects_small_or_rare_conflicts(self):
        records = []
        for probe_index in range(4):
            terms = OrderedDict(
                (
                    name,
                    {"id_backbone": _observation(0.1, 0.005, 0.01)},
                )
                for name in ALIGNMENT_COMPONENTS
            )
            terms["alignment_total"] = {
                "id_backbone": _observation(0.1, 0.01, 0.01)
            }
            records.append(
                {"epoch": 1, "batch": probe_index, "terms": terms}
            )

        summary = summarize_probe_records(records)
        self.assertFalse(summary["gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
