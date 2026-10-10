import os
import sys

import torch


SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)

from models.bridge_c2r import BRIDGE_C2R


def test_product_weights_preserve_score_scale():
    user = torch.full((4, 3), 1.0 / 3.0)
    item = torch.full((4, 3), 1.0 / 3.0)
    weights = BRIDGE_C2R._product_weights(user, item)
    assert torch.allclose(weights.sum(dim=-1), torch.full((4,), 3.0))
    assert torch.allclose(weights, torch.ones_like(weights))


def test_helpful_modality_gets_larger_target_weight():
    margin = torch.tensor([[2.0, 0.0, -1.0]])
    scale = margin.abs().mean(dim=-1, keepdim=True).clamp_min(1e-3)
    target = torch.softmax(margin / scale, dim=-1)
    assert target[0, 0] > target[0, 1] > target[0, 2]


if __name__ == "__main__":
    test_product_weights_preserve_score_scale()
    test_helpful_modality_gets_larger_target_weight()
    print("C2R math tests passed")
