"""Capture the actual projector SVD inside custom forward without changing it."""
import json
import time
import warnings

import torch

from numerics import FiniteTrainingGuard, all_finite


class RecoveryTrainingGuard(FiniteTrainingGuard):
    """Retain original finite-step protection and honest failure-packet gradients."""
    def __init__(self, model, optimizer, output):
        if hasattr(model, '_spectral_recovery_guard'):
            raise RuntimeError('Recovery input observer may only be installed once')
        self.pending_input_capture = None
        super().__init__(model, optimizer, output)
        model._spectral_recovery_guard = self

    def new_capture(self, rep, generic):
        capture = {'input': rep.detach().clone(), 'singular': None, 'gradients': {},
                   'input_gradient': None, 'svd_grad_mode_enabled': generic,
                   'gradient_kind': ('generic_svd_outputs_and_actual_input' if generic
                                     else 'actual_projector_input_only'),
                   'generic_output_gradients_available': generic, 'reported_warnings': [],
                   'solver_error': None}
        self.captures.append(capture)
        return capture

    def projector_call(self, rep, bands, function):
        # Frozen input validation precedes SVD, so observe at the instance entry
        # as well. Successful calls reuse this packet in svd, never duplicate it.
        if self.pending_input_capture is not None:
            raise RuntimeError('Unexpected nested projector input observation')
        if self.model.training and rep.requires_grad:
            self.pending_input_capture = (rep, self.new_capture(rep, False))
        try:
            return function(rep, bands)
        except ValueError as error:
            raise FloatingPointError('Frozen projector input/domain validation failed: ' + str(error)) from error
        finally:
            self.pending_input_capture = None

    def svd(self, rep, *args, **kwargs):
        recording = self.model.training and rep.requires_grad
        generic = torch.is_grad_enabled()
        index = len(self.captures)
        capture = None
        if recording:
            if self.pending_input_capture is not None:
                actual_input, capture = self.pending_input_capture
                if actual_input is not rep:
                    raise FloatingPointError('Projector/SVD observed input identity changed')
                index = len(self.captures) - 1
                capture['svd_grad_mode_enabled'] = generic
                capture['generic_output_gradients_available'] = generic
                capture['gradient_kind'] = ('generic_svd_outputs_and_actual_input' if generic
                                            else 'actual_projector_input_only')
            else:
                capture = self.new_capture(rep, generic)
        if not all_finite([rep]):
            raise FloatingPointError('Nonfinite SVD input')
        reported = []
        original_warning = warnings.showwarning
        def observe_warning(message, category, filename, lineno, file=None, line=None):
            reported.append(str(message))
            return original_warning(message, category, filename, lineno, file=file, line=line)
        warnings.showwarning = observe_warning
        try:
            result = self.original_svd(rep, *args, **kwargs)
        except RuntimeError as error:
            if capture is not None:
                capture['solver_error'] = str(error)
            raise FloatingPointError('Original SVD failed without retry: ' + str(error)) from error
        finally:
            warnings.showwarning = original_warning
            if capture is not None:
                capture['reported_warnings'] = reported
        if capture is not None:
            capture['singular'] = result[1].detach().clone()
        for message in reported:
            message = message.lower()
            if (any(word in message for word in ('svd', 'gesvd', 'cusolver')) and
                    any(word in message for word in ('converg', 'fallback', 'fall back', 'recomput', 'failed'))):
                raise FloatingPointError('Reported SVD nonconvergence/fallback; no update permitted')
        if not all_finite(result):
            raise FloatingPointError('Nonfinite SVD output from finite input')
        # Autograd.Function.forward has grad mode disabled even though its actual
        # input participates in the upstream graph. Evaluation remains excluded.
        if not recording:
            return result
        if generic:
            for name, value in zip(('U', 'S', 'Vh'), result):
                if value.requires_grad:
                    def observe_output(gradient, name=name):
                        capture['gradients'][name] = gradient.detach().clone()
                        return None
                    self.handles.append(value.register_hook(observe_output))

        def observe_input(gradient):
            capture['input_gradient'] = gradient.detach().clone()
            if not all_finite([gradient]):
                self.bad_backward.append({'call': index, 'shape': list(rep.shape),
                    'gradient_kind': capture['gradient_kind'], 'input_gradient_finite': False,
                    'nonfinite_elements': int((~torch.isfinite(gradient)).sum().item()),
                    'generic_output_gradients_available': generic})
            return None
        self.handles.append(rep.register_hook(observe_input))
        return result

    def step(self, *args, **kwargs):
        if self.bad_backward or not all_finite(value.grad for value in self.model.parameters()):
            self.fail('backward', 'Nonfinite actual intermediate/parameter gradients blocked before optimizer.step')
        result = self.original_step(*args, **kwargs)
        values = list(self.model.parameters())
        values += [value for state in self.optimizer.state.values() for value in state.values()
                   if torch.is_tensor(value)]
        if not all_finite(values):
            self.fail('optimizer', 'Optimizer created nonfinite state from finite gradients')
        self.steps += 1
        self.clear_capture()
        return result

    def close(self):
        super().close()
        if getattr(self.model, '_spectral_recovery_guard', None) is not self:
            raise RuntimeError('Recovery input observer identity changed')
        del self.model._spectral_recovery_guard
        self.pending_input_capture = None

    def fail(self, stage, message):
        invalid = [name for name, value in self.model.named_parameters()
                   if value.grad is not None and not all_finite([value.grad])]
        record = {'stage': stage, 'error': message, 'epoch': self.epoch, 'batch': self.batch,
                  'finite_steps_completed': self.steps, 'unix': time.time(),
                  'parameters_finite': all_finite(self.model.parameters()),
                  'optimizer_update_blocked': stage in ('forward', 'backward'),
                  'nonfinite_parameter_gradients': invalid, 'bad_svd_backward': self.bad_backward,
                  'test_evaluated': False, 'replay_packet': str(self.output / 'failure_replay.pt'),
                  'backward_implementation_changed': True,
                  'capture_semantics': 'actual SVD input/singular values; actual dL/dX when reached; no fabricated dL/dU,S,Vh'}
        captures = []
        for capture in self.captures:
            singular = capture['singular']
            gaps = None if singular is None else singular[:-1] - singular[1:]
            gradient = capture['input_gradient']
            captures.append({'input': capture['input'].cpu(), 'singular': None if singular is None else singular.cpu(),
                'output_gradients': {key: value.cpu() for key, value in capture['gradients'].items()},
                'shape': list(capture['input'].shape),
                'equal_adjacent_singular_values': None if gaps is None else int((gaps == 0).sum().item()),
                'min_adjacent_gap': gaps.min().item() if gaps is not None and len(gaps) else None,
                'reported_warnings': capture['reported_warnings'], 'solver_error': capture['solver_error'],
                'gradient_kind': capture['gradient_kind'],
                'generic_output_gradients_available': capture['generic_output_gradients_available'],
                'svd_grad_mode_enabled': capture['svd_grad_mode_enabled'],
                'actual_input_gradient_observed': gradient is not None,
                'actual_input_gradient': None if gradient is None else gradient.cpu()})
        torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                    'interaction': self.interaction, 'rng': self.rng, 'record': record,
                    'svd_captures': captures}, self.output / 'failure_replay.pt')
        with (self.output / 'failure.json').open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
        raise FloatingPointError(message)
