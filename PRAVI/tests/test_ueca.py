from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.msca import MSCA
from models.msca_ueca import MSCA_UECA, inverse_inclusion_weights


class UECATest(unittest.TestCase):
    def test_inverse_inclusion_weights_are_exact(self):
        weights = inverse_inclusion_weights(
            torch.tensor([0.0, 1.0, 2.0, 4.0]),
            population_size=4,
            batch_size=2,
        )
        torch.testing.assert_close(
            weights,
            torch.tensor([0.0, 2.0, 1.2, 1.0]),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_duplicate_entities_create_one_contrastive_row(self):
        model = object.__new__(MSCA_UECA)
        nn.Module.__init__(model)
        model.register_buffer(
            "ueca_user_inverse_inclusion",
            torch.tensor([4.0, 3.0, 2.0, 1.0]),
            persistent=False,
        )

        ids, weights = model._unique_cl_anchors(
            torch.tensor([3, 1, 3, 2, 1]), is_user=True
        )
        torch.testing.assert_close(ids, torch.tensor([1, 2, 3]))
        torch.testing.assert_close(weights, torch.tensor([3.0, 2.0, 1.0]))
        self.assertEqual(ids.numel(), torch.unique(ids).numel())

    def test_duplicate_copies_do_not_change_user_alignment(self):
        model = object.__new__(MSCA_UECA)
        nn.Module.__init__(model)
        model.register_buffer(
            "ueca_user_inverse_inclusion",
            torch.tensor([2.0, 1.25, 1.0]),
            persistent=False,
        )
        first = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
        )
        second = torch.tensor(
            [[0.8, 0.2], [0.1, 0.9], [0.7, 0.7]], dtype=torch.float32
        )

        ids_a, weights_a = model._unique_cl_anchors(
            torch.tensor([0, 1, 2]), is_user=True
        )
        ids_b, weights_b = model._unique_cl_anchors(
            torch.tensor([0, 0, 1, 2, 2]), is_user=True
        )
        loss_a = model.cal_cl_loss(
            first[ids_a], second[ids_a], tau=0.2, sample_weight=weights_a
        )
        loss_b = model.cal_cl_loss(
            first[ids_b], second[ids_b], tau=0.2, sample_weight=weights_b
        )
        torch.testing.assert_close(loss_a, loss_b)

    def test_disabled_ueca_delegates_exactly_to_msca(self):
        model = object.__new__(MSCA_UECA)
        nn.Module.__init__(model)
        model.ueca_enabled = False
        expected = torch.tensor(3.25)
        interaction = torch.zeros((3, 2), dtype=torch.long)

        with patch.object(MSCA, "calculate_loss", return_value=expected) as base:
            actual = model.calculate_loss(interaction)

        self.assertIs(actual, expected)
        base.assert_called_once_with(interaction)

    def test_invalid_sampling_population_is_rejected(self):
        with self.assertRaises(ValueError):
            inverse_inclusion_weights(torch.tensor([1.0]), 0, 1)
        with self.assertRaises(ValueError):
            inverse_inclusion_weights(torch.tensor([5.0]), 4, 2)


if __name__ == "__main__":
    unittest.main()

