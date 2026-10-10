"""Capture invalid training arithmetic before it reaches an optimizer update."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def all_finite(values):
    checks = {}
    for value in values:
        if value is not None:
            checks.setdefault(value.device, []).append(torch.isfinite(value).all())
    return all(bool(torch.stack(group).all().item()) for group in checks.values())


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


class FiniteTrainingGuard:
    """Observe arithmetic and save a replay packet; never repair or skip a step."""
    def __init__(self, model, optimizer, output):
        self.model, self.optimizer, self.output = model, optimizer, Path(output)
        self.epoch, self.batch, self.steps = -1, -1, 0
        self.interaction, self.rng, self.captures, self.handles = None, None, [], []
        self.bad_backward = []
        self.original_loss, self.original_step = model.calculate_loss, optimizer.step
        self.original_svd = torch.linalg.svd
        model.calculate_loss, optimizer.step, torch.linalg.svd = self.loss, self.step, self.svd

    def clear_capture(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.captures.clear()

    def enter_epoch(self, epoch):
        self.epoch, self.batch = epoch, -1

    def fail(self, stage, message):
        invalid_gradients = [name for name, value in self.model.named_parameters()
                             if value.grad is not None and not all_finite([value.grad])]
        record = {'stage': stage, 'error': message, 'epoch': self.epoch, 'batch': self.batch,
                  'finite_steps_completed': self.steps, 'unix': time.time(),
                  'parameters_finite': all_finite(self.model.parameters()),
                  'optimizer_update_blocked': stage in ('forward', 'backward'),
                  'nonfinite_parameter_gradients': invalid_gradients,
                  'bad_svd_backward': self.bad_backward, 'test_evaluated': False,
                  'replay_packet': str(self.output / 'failure_replay.pt')}
        captures = []
        for capture in self.captures:
            singular = capture['singular']
            gaps = singular[:-1] - singular[1:]
            captures.append({'input': capture['input'].cpu(), 'singular': singular.cpu(),
                             'output_gradients': {key: value.cpu() for key, value in capture['gradients'].items()},
                             'shape': list(capture['input'].shape),
                             'equal_adjacent_singular_values': int((gaps == 0).sum().item()),
                             'min_adjacent_gap': gaps.min().item() if len(gaps) else None})
        torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                    'interaction': self.interaction, 'rng': self.rng, 'record': record,
                    'svd_captures': captures}, self.output / 'failure_replay.pt')
        with (self.output / 'failure.json').open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
        raise FloatingPointError(message)

    def loss(self, interaction):
        self.clear_capture()
        self.bad_backward.clear()
        self.batch += 1
        self.interaction = (interaction.detach().cpu().clone() if torch.is_tensor(interaction)
                            else [value.detach().cpu().clone() for value in interaction])
        self.rng = rng_state()
        try:
            losses = self.original_loss(interaction)
        except FloatingPointError as error:
            self.fail('forward', str(error))
        if not all_finite(losses if isinstance(losses, (tuple, list)) else [losses]):
            self.fail('forward', 'Nonfinite loss before backward')
        return losses

    def svd(self, rep, *args, **kwargs):
        if not all_finite([rep]):
            raise FloatingPointError('Nonfinite SVD input')
        result = self.original_svd(rep, *args, **kwargs)
        if not all_finite(result):
            raise FloatingPointError('Nonfinite SVD output from finite input')
        if not torch.is_grad_enabled() or not rep.requires_grad:
            return result
        index = len(self.captures)
        capture = {'input': rep.detach(), 'singular': result[1].detach(), 'gradients': {}}
        self.captures.append(capture)
        for name, value in zip(('U', 'S', 'Vh'), result):
            if value.requires_grad:
                def observe_output(gradient, name=name):
                    capture['gradients'][name] = gradient.detach()
                    return gradient
                self.handles.append(value.register_hook(observe_output))

        def observe_input(gradient):
            if not all_finite([gradient]):
                self.bad_backward.append({'call': index, 'shape': list(rep.shape),
                    'output_gradients_finite': all_finite(capture['gradients'].values()),
                    'input_gradient_finite': False,
                    'nonfinite_elements': int((~torch.isfinite(gradient)).sum().item())})
            return gradient
        self.handles.append(rep.register_hook(observe_input))
        return result

    def step(self, *args, **kwargs):
        if not all_finite(value.grad for value in self.model.parameters()):
            self.fail('backward', 'Nonfinite gradients blocked before optimizer.step')
        result = self.original_step(*args, **kwargs)
        self.steps += 1
        values = list(self.model.parameters())
        values += [value for state in self.optimizer.state.values() for value in state.values()
                   if torch.is_tensor(value)]
        if not all_finite(values):
            self.fail('optimizer', 'Optimizer created nonfinite state from finite gradients')
        self.clear_capture()
        return result

    def close(self):
        self.clear_capture()
        self.model.calculate_loss, self.optimizer.step = self.original_loss, self.original_step
        torch.linalg.svd = self.original_svd
