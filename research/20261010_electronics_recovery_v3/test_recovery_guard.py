"""Actual CPU checks of no-grad custom-forward capture and unmodified Adam."""
import json
import tempfile
import unittest
import warnings
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None
if torch is not None:
    from projector import projector_bands
    from recovery_guard import RecoveryTrainingGuard
    from numerics import rng_state


def make_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            value = torch.zeros((8, 6), dtype=torch.float32)
            value[:6] = torch.diag(torch.tensor([7., 7., 4., 4., 1., 1.]))
            self.weight = torch.nn.Parameter(value)

        def calculate_loss(self, interaction):
            guard = getattr(self, '_spectral_recovery_guard', None)
            outputs = (projector_bands(self.weight, 3) if guard is None else
                       guard.projector_call(self.weight, 3, projector_bands))
            return sum((index + 1) * output.square().sum() for index, output in enumerate(outputs))
    return Model()


@unittest.skipUnless(torch is not None, 'Actual MMRec CPU tests require PyTorch')
class RecoveryGuardTests(unittest.TestCase):
    def test_custom_forward_captures_actual_input_without_generic_output_gradients(self):
        model = make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
        with tempfile.TemporaryDirectory() as directory:
            guard = RecoveryTrainingGuard(model, optimizer, directory)
            try:
                loss = model.calculate_loss(torch.tensor([0]))
                self.assertEqual(len(guard.captures), 1)
                capture = guard.captures[0]
                self.assertFalse(capture['svd_grad_mode_enabled'])
                self.assertFalse(capture['generic_output_gradients_available'])
                self.assertEqual(capture['gradients'], {})
                self.assertTrue(torch.equal(capture['input'], model.weight.detach()))
                self.assertEqual(capture['singular'].shape, (6,))
                loss.backward()
                self.assertIsNotNone(capture['input_gradient'])
                self.assertTrue(torch.isfinite(capture['input_gradient']).all())
            finally:
                guard.close()

    def test_two_guarded_steps_equal_ordinary_adam_and_rng_without_capture_accumulation(self):
        reference, observed = make_model(), make_model()
        left = torch.optim.Adam(reference.parameters(), lr=.0001)
        right = torch.optim.Adam(observed.parameters(), lr=.0001)
        rng_before = rng_state()
        original_svd = torch.linalg.svd
        with tempfile.TemporaryDirectory() as directory:
            guard = RecoveryTrainingGuard(observed, right, directory)
            try:
                for step in range(2):
                    left.zero_grad(set_to_none=True)
                    # Temporarily bypass only the observer while computing the control.
                    torch.linalg.svd = original_svd
                    reference.calculate_loss(torch.tensor([0])).backward()
                    left.step()
                    torch.linalg.svd = guard.svd
                    right.zero_grad(set_to_none=True)
                    observed.calculate_loss(torch.tensor([0])).backward()
                    self.assertEqual(len(guard.captures), 1)
                    self.assertEqual(len(guard.handles), 1)
                    right.step()
                    self.assertEqual(guard.steps, step + 1)
                    self.assertEqual(guard.captures, [])
                    self.assertEqual(guard.handles, [])
                    self.assertTrue(torch.equal(reference.weight, observed.weight))
                    for a, b in zip(left.state.values(), right.state.values()):
                        self.assertEqual(set(a), set(b))
                        for key in a:
                            if torch.is_tensor(a[key]):
                                self.assertTrue(torch.equal(a[key], b[key]))
                            else:
                                self.assertEqual(a[key], b[key])
                rng_after = rng_state()
                self.assertEqual(rng_before['python'], rng_after['python'])
                self.assertTrue(torch.equal(rng_before['torch'], rng_after['torch']))
                self.assertEqual(rng_before['cuda'], rng_after['cuda'])
                self.assertEqual(rng_before['numpy'][0], rng_after['numpy'][0])
                self.assertTrue((rng_before['numpy'][1] == rng_after['numpy'][1]).all())
                self.assertEqual(rng_before['numpy'][2:], rng_after['numpy'][2:])
            finally:
                guard.close()
        self.assertIs(torch.linalg.svd, original_svd)

    def test_failure_packet_retains_actual_inputs_and_honest_gradient_kind(self):
        model = make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
        before = model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            guard = RecoveryTrainingGuard(model, optimizer, directory)
            try:
                model.calculate_loss(torch.tensor([0])).backward()
                model.weight.grad.fill_(float('nan'))
                with self.assertRaises(FloatingPointError):
                    optimizer.step()
                packet = torch.load(Path(directory) / 'failure_replay.pt', map_location='cpu')
                self.assertEqual(len(packet['svd_captures']), 1)
                capture = packet['svd_captures'][0]
                self.assertTrue(torch.equal(capture['input'], before))
                self.assertEqual(capture['output_gradients'], {})
                self.assertEqual(capture['gradient_kind'], 'actual_projector_input_only')
                self.assertTrue(capture['actual_input_gradient_observed'])
                self.assertTrue(torch.isfinite(capture['actual_input_gradient']).all())
                self.assertFalse(capture['generic_output_gradients_available'])
                self.assertTrue(torch.equal(model.weight, before))
                self.assertEqual(guard.steps, 0)
                self.assertEqual(packet['record']['stage'], 'backward')
            finally:
                guard.close()

    def test_boundary_failure_retains_capture_with_unobserved_gradient(self):
        model = make_model()
        with torch.no_grad():
            model.weight[:6] = torch.diag(torch.tensor([7., 4., 4., 2., 1., 1.]))
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
        with tempfile.TemporaryDirectory() as directory:
            guard = RecoveryTrainingGuard(model, optimizer, directory)
            try:
                with self.assertRaisesRegex(FloatingPointError, 'Unresolved spectral band boundary'):
                    model.calculate_loss(torch.tensor([0]))
                packet = torch.load(Path(directory) / 'failure_replay.pt', map_location='cpu')
                capture = packet['svd_captures'][0]
                self.assertEqual(packet['record']['stage'], 'forward')
                self.assertIsNone(capture['actual_input_gradient'])
                self.assertFalse(capture['actual_input_gradient_observed'])
                self.assertEqual(capture['output_gradients'], {})
                self.assertEqual(guard.steps, 0)
            finally:
                guard.close()

    def test_eval_does_not_capture_projector_input(self):
        model = make_model().eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
        with tempfile.TemporaryDirectory() as directory:
            guard = RecoveryTrainingGuard(model, optimizer, directory)
            try:
                with torch.no_grad():
                    model.calculate_loss(torch.tensor([0]))
                self.assertEqual(guard.captures, [])
                self.assertEqual(guard.handles, [])
            finally:
                guard.close()

    def test_nonfinite_first_input_saves_actual_input_without_singular_values(self):
        model = make_model()
        with torch.no_grad():
            model.weight[0, 0] = float('nan')
        optimizer = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            guard = RecoveryTrainingGuard(model, optimizer, directory)
            try:
                with self.assertRaisesRegex(FloatingPointError, 'projector input/domain validation'):
                    model.calculate_loss(torch.tensor([0]))
                packet = torch.load(Path(directory) / 'failure_replay.pt', map_location='cpu')
                self.assertTrue(torch.isnan(packet['svd_captures'][0]['input'][0, 0]))
                self.assertIsNone(packet['svd_captures'][0]['singular'])
                self.assertEqual(guard.steps, 0)
            finally:
                guard.close()

    def test_original_solver_runtime_error_saves_pending_input_without_retry(self):
        model = make_model()
        optimizer = torch.optim.Adam(model.parameters())
        calls = []
        def broken(*args, **kwargs):
            calls.append(1)
            raise RuntimeError('synthetic solver failure')
        with tempfile.TemporaryDirectory() as directory:
            guard = RecoveryTrainingGuard(model, optimizer, directory)
            original = guard.original_svd
            guard.original_svd = broken
            try:
                with self.assertRaisesRegex(FloatingPointError, 'Original SVD failed without retry'):
                    model.calculate_loss(torch.tensor([0]))
                packet = torch.load(Path(directory) / 'failure_replay.pt', map_location='cpu')
                capture = packet['svd_captures'][0]
                self.assertTrue(torch.equal(capture['input'], model.weight.detach()))
                self.assertIsNone(capture['singular'])
                self.assertIn('synthetic solver failure', capture['solver_error'])
                self.assertEqual(calls, [1])
            finally:
                guard.original_svd = original
                guard.close()

    def test_reported_svd_fallback_blocks_and_restores_warning_observer(self):
        model = make_model()
        optimizer = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as directory, warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            guard = RecoveryTrainingGuard(model, optimizer, directory)
            original, warning_method = guard.original_svd, warnings.showwarning
            filters = list(warnings.filters)
            def warned(*args, **kwargs):
                result = original(*args, **kwargs)
                warnings.warn('SVD failed to converge; fallback recomputation', UserWarning)
                return result
            guard.original_svd = warned
            try:
                with self.assertRaisesRegex(FloatingPointError, 'Reported SVD'):
                    model.calculate_loss(torch.tensor([0]))
                self.assertIs(warnings.showwarning, warning_method)
                self.assertEqual(warnings.filters, filters)
                packet = torch.load(Path(directory) / 'failure_replay.pt', map_location='cpu')
                self.assertIsNotNone(packet['svd_captures'][0]['singular'])
                self.assertEqual(len(packet['svd_captures'][0]['reported_warnings']), 1)
                self.assertEqual(guard.steps, 0)
            finally:
                guard.original_svd = original
                guard.close()

    def test_nonfinite_actual_input_blocks_even_when_later_hook_restores_parameter_gradient(self):
        model = make_model()
        optimizer = torch.optim.Adam(model.parameters())
        before = model.weight.detach().clone()
        first = model.weight.register_hook(lambda gradient: torch.full_like(gradient, float('nan')))
        with tempfile.TemporaryDirectory() as directory:
            guard = RecoveryTrainingGuard(model, optimizer, directory)
            last = None
            try:
                loss = model.calculate_loss(torch.tensor([0]))
                last = model.weight.register_hook(lambda gradient: torch.zeros_like(gradient))
                loss.backward()
                self.assertTrue(torch.isfinite(model.weight.grad).all())
                self.assertEqual(len(guard.bad_backward), 1)
                with self.assertRaisesRegex(FloatingPointError, 'intermediate/parameter'):
                    optimizer.step()
                self.assertTrue(torch.equal(model.weight, before))
                self.assertEqual(guard.steps, 0)
            finally:
                first.remove()
                if last is not None:
                    last.remove()
                guard.close()

    def test_optimizer_nonfinite_update_is_not_counted_as_finite_step(self):
        model = make_model()
        optimizer = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            guard = RecoveryTrainingGuard(model, optimizer, directory)
            original = guard.original_step
            def corrupt(*args, **kwargs):
                result = original(*args, **kwargs)
                with torch.no_grad():
                    model.weight[0, 0] = float('nan')
                return result
            guard.original_step = corrupt
            try:
                model.calculate_loss(torch.tensor([0])).backward()
                with self.assertRaisesRegex(FloatingPointError, 'Optimizer created'):
                    optimizer.step()
                self.assertEqual(guard.steps, 0)
                record = json.loads((Path(directory) / 'failure.json').read_text())
                self.assertFalse(record['optimizer_update_blocked'])
                self.assertEqual(record['stage'], 'optimizer')
            finally:
                guard.close()


if __name__ == '__main__':
    unittest.main()
