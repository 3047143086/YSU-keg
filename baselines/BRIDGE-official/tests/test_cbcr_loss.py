import os
import sys

import torch


SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)

from models.bridge_cbcr import BRIDGE_CBCR


def test_inactive_pairs_contribute_zero():
    base = torch.tensor([-1.0, 1.0])
    correction = torch.tensor([0.5, -0.5], requires_grad=True)
    mask = torch.zeros(2)
    loss, repair, guard = BRIDGE_CBCR._cbcr_loss(base, correction, mask)
    assert loss.item() == 0.0
    assert repair.item() == 0.0
    assert guard.item() == 0.0


def test_helpful_repair_reduces_boundary_loss():
    base = torch.tensor([-1.0, -0.5])
    mask = torch.ones(2)
    zero = torch.zeros(2)
    helpful = torch.tensor([1.5, 1.0])
    zero_loss = BRIDGE_CBCR._cbcr_loss(base, zero, mask, guard_weight=0.0)[0]
    repaired_loss = BRIDGE_CBCR._cbcr_loss(
        base, helpful, mask, guard_weight=0.0
    )[0]
    assert repaired_loss < zero_loss


def test_confident_pairs_enforce_correction_budget():
    base = torch.tensor([2.0, 3.0])
    mask = torch.ones(2)
    small = torch.tensor([0.01, -0.01])
    large = torch.tensor([2.0, -2.0])
    small_guard = BRIDGE_CBCR._cbcr_loss(base, small, mask)[2]
    large_guard = BRIDGE_CBCR._cbcr_loss(base, large, mask)[2]
    assert large_guard > small_guard


if __name__ == "__main__":
    test_inactive_pairs_contribute_zero()
    test_helpful_repair_reduces_boundary_loss()
    test_confident_pairs_enforce_correction_budget()
    print("CBCR loss tests passed")
