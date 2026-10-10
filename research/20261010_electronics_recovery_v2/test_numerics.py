"""Actual CPU arithmetic checks that no invalid optimizer step can slip through."""
import json
import tempfile
import unittest
from pathlib import Path
try:
    import torch
except ModuleNotFoundError:
    torch = None
if torch is not None:
    from numerics import FiniteTrainingGuard
    from projector import projector_bands


@unittest.skipUnless(torch is not None, 'Actual MMRec CPU tests require PyTorch')
class NumericalGuardTests(unittest.TestCase):
    def test_nonfinite_parameter_gradient_blocks_optimizer_and_preserves_parameters(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(2.0))
            def calculate_loss(self, interaction):
                return self.weight.square()
        model = Model()
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
        before = model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            guard = FiniteTrainingGuard(model, optimizer, directory)
            try:
                guard.enter_epoch(0)
                model.calculate_loss(torch.tensor([0])).backward()
                model.weight.grad.fill_(float('nan'))
                with self.assertRaises(FloatingPointError):
                    optimizer.step()
                self.assertTrue(torch.equal(before, model.weight))
                self.assertEqual(guard.steps, 0)
                row = json.loads((Path(directory) / 'failure.json').read_text())
                self.assertTrue(row['optimizer_update_blocked'])
                self.assertEqual(row['stage'], 'backward')
            finally:
                guard.close()

    def test_repeated_internal_singular_values_take_a_finite_guarded_adam_step(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                value = torch.zeros((8, 6), dtype=torch.float32)
                value[:6] = torch.diag(torch.tensor([7., 7., 4., 4., 1., 1.]))
                self.weight = torch.nn.Parameter(value)
            def calculate_loss(self, interaction):
                outputs = projector_bands(self.weight, 3)
                return sum((index + 1) * output.square().sum() for index, output in enumerate(outputs))
        model = Model()
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
        with tempfile.TemporaryDirectory() as directory:
            guard = FiniteTrainingGuard(model, optimizer, directory)
            try:
                model.calculate_loss(torch.tensor([0])).backward()
                optimizer.step()
                self.assertTrue(torch.isfinite(model.weight).all())
                self.assertEqual(guard.steps, 1)
                self.assertFalse((Path(directory) / 'failure.json').exists())
            finally:
                guard.close()


if __name__ == '__main__':
    unittest.main()
