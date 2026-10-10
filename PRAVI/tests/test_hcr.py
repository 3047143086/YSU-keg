from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.msca_hcr import (
    MSCA_HCR,
    build_history_mean_operator,
    history_coefficients,
    leave_one_out_prototype,
)


class HCRTest(unittest.TestCase):
    def test_history_operator_is_a_row_mean(self):
        interactions = sp.csr_matrix(
            ([1, 1, 1], ([0, 0, 1], [0, 1, 2])), shape=(2, 3)
        )
        operator, degree = build_history_mean_operator(interactions, "cpu")

        np.testing.assert_allclose(
            operator.to_dense().numpy(),
            np.asarray([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(degree.numpy(), [2.0, 1.0])

    def test_leave_one_out_removes_the_training_target(self):
        history_mean = torch.tensor([[0.5, 0.5]])
        positive_content = torch.tensor([[0.0, 1.0]])
        direction, coherence, remaining = leave_one_out_prototype(
            history_mean,
            torch.tensor([2.0]),
            positive_content,
            min_context=1,
        )

        torch.testing.assert_close(direction, torch.tensor([[1.0, 0.0]]))
        torch.testing.assert_close(coherence, torch.tensor([1.0]))
        torch.testing.assert_close(remaining, torch.tensor([1.0]))

    def test_too_short_leave_one_out_context_is_disabled(self):
        direction, coherence, _ = leave_one_out_prototype(
            torch.tensor([[0.5, 0.5]]),
            torch.tensor([2.0]),
            torch.tensor([[0.0, 1.0]]),
            min_context=2,
        )

        torch.testing.assert_close(direction, torch.zeros_like(direction))
        torch.testing.assert_close(coherence, torch.zeros_like(coherence))

    def test_history_coefficients_are_bounded_and_sparse_aware(self):
        coefficients = history_coefficients(
            torch.tensor([[0.8, 0.2], [0.8, 0.2]]),
            torch.tensor([2.0, 20.0]),
            activity_prior=10.0,
            temperature=0.25,
            min_context=2,
        )

        self.assertTrue(torch.all(coefficients >= 0.0))
        self.assertTrue(torch.all(coefficients.sum(dim=1) <= 1.0))
        self.assertGreater(
            float(coefficients[0].sum()), float(coefficients[1].sum())
        )

    def test_zero_initial_scale_is_exact_but_trainable(self):
        model = object.__new__(MSCA_HCR)
        nn.Module.__init__(model)
        model.hcr_max_scale = 0.1
        model.hcr_raw_scale = nn.Parameter(torch.zeros(2))

        scales = model._bounded_scales()
        torch.testing.assert_close(scales, torch.zeros(2))
        scales.sum().backward()
        self.assertTrue(torch.all(model.hcr_raw_scale.grad != 0.0))


if __name__ == "__main__":
    unittest.main()
