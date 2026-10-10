"""Check that observation preserves finite updates and blocks poisoned ones."""
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from numerics import FiniteTrainingGuard, rng_state


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1., 2.]))

    def calculate_loss(self, interaction):
        return (self.weight * interaction).square().sum()


class NumericTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def guard(self, model, optimizer):
        guard = FiniteTrainingGuard(model, optimizer, Path(self.directory.name))
        self.addCleanup(guard.close)
        return guard

    def test_finite_adam_update_and_rng_unchanged(self):
        reference, observed = Tiny(), Tiny()
        reference_optimizer = torch.optim.Adam(reference.parameters(), lr=.0001)
        observed_optimizer = torch.optim.Adam(observed.parameters(), lr=.0001)
        guard = self.guard(observed, observed_optimizer)
        before = rng_state()
        interaction = torch.tensor([3., 4.])
        reference.calculate_loss(interaction).backward()
        reference_optimizer.step()
        guard.loss(interaction).backward()
        observed_optimizer.step()
        torch.testing.assert_close(observed.weight, reference.weight, rtol=0, atol=0)
        for key in ('exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(observed_optimizer.state[observed.weight][key],
                                       reference_optimizer.state[reference.weight][key], rtol=0, atol=0)
        after = rng_state()
        self.assertEqual(before['python'], after['python'])
        np.testing.assert_array_equal(before['numpy'][1], after['numpy'][1])
        self.assertEqual(before['numpy'][2:], after['numpy'][2:])
        self.assertTrue(torch.equal(before['torch'], after['torch']))
        self.assertEqual(guard.steps, 1)

    def test_invalid_gradient_is_saved_and_never_applied(self):
        model = Tiny()
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
        guard = self.guard(model, optimizer)
        guard.enter_epoch(39)
        interaction = torch.tensor([3., 4.])
        guard.loss(interaction).backward()
        model.weight.grad[0] = float('nan')
        before = model.weight.detach().clone()
        with self.assertRaisesRegex(FloatingPointError, 'before optimizer.step'):
            optimizer.step()
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
        self.assertEqual(len(optimizer.state), 0)
        record = json.loads((Path(self.directory.name) / 'failure.json').read_text())
        self.assertTrue(record['parameters_finite'])
        self.assertTrue(record['optimizer_update_blocked'])
        packet = torch.load(Path(self.directory.name) / 'failure_replay.pt', map_location='cpu')
        torch.testing.assert_close(packet['interaction'], interaction, rtol=0, atol=0)
        self.assertEqual(packet['rng']['python'], random.getstate())

    def test_forward_failure_is_captured(self):
        model = Tiny()
        guard = self.guard(model, torch.optim.Adam(model.parameters()))
        with self.assertRaisesRegex(FloatingPointError, 'Nonfinite loss'):
            guard.loss(torch.tensor([float('inf'), 1.]))
        record = json.loads((Path(self.directory.name) / 'failure.json').read_text())
        self.assertEqual(record['stage'], 'forward')
        self.assertTrue(record['parameters_finite'])

    def test_svd_probe_preserves_decomposition_gradient(self):
        model = Tiny()
        guard = self.guard(model, torch.optim.Adam(model.parameters()))
        original = guard.original_svd
        rep = torch.tensor([[3., .2], [.1, 1.], [.4, .3]], requires_grad=True)
        reference = rep.detach().clone().requires_grad_()
        u, s, vh = original(reference, full_matrices=False)
        expected = ((u[:, :1] * s[:1]) @ vh[:1]).square().sum()
        expected.backward()
        u, s, vh = torch.linalg.svd(rep, full_matrices=False)
        actual = ((u[:, :1] * s[:1]) @ vh[:1]).square().sum()
        actual.backward()
        torch.testing.assert_close(rep.grad, reference.grad, rtol=0, atol=0)
        self.assertEqual(guard.bad_backward, [])


if __name__ == '__main__':
    unittest.main()
