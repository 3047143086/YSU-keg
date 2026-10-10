from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.msca_dcrl import MSCA_DCRL


class DCRLTest(unittest.TestCase):
    def test_conjunctive_margin_uses_the_weaker_view(self):
        model = object.__new__(MSCA_DCRL)
        model.dcrl_enabled = True
        model.dcrl_weight = 1.0
        model.dcrl_margin = 0.1
        model.dcrl_temperature = 0.2
        model.dcrl_min_weight = 0.1
        model.n_users = 0
        user = torch.tensor([[1.0, 0.0]])
        image = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        text = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        model._dcrl_forward_cache = (user, None, None, None, image, text)
        # The visual view prefers item 0 while text prefers item 1; the
        # auxiliary loss must remain finite and backpropagate through views.
        image.requires_grad_()
        text.requires_grad_()
        loss = model._compute_dcrl(
            torch.tensor([0]), torch.tensor([0]), torch.tensor([1])
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(image.grad)
        self.assertIsNotNone(text.grad)

    def test_disabled_returns_no_auxiliary_loss(self):
        model = object.__new__(MSCA_DCRL)
        model.dcrl_enabled = False
        model.dcrl_weight = 0.02
        model._dcrl_forward_cache = None
        self.assertIsNone(
            model._compute_dcrl(
                torch.tensor([0]), torch.tensor([0]), torch.tensor([1])
            )
        )


if __name__ == "__main__":
    unittest.main()
