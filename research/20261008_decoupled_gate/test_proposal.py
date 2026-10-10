"""Test gradient ownership with real BRIDGE loss/scoring and a tiny encoder."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

local_carrier = Path(__file__).resolve().parents[2] / 'baselines/BRIDGE-official/src'
if local_carrier.is_dir():
    sys.path.insert(0, str(local_carrier))

import numpy as np
import torch
from scipy import sparse

from evidence import build_counterfactual_cache
from models.bridge import BRIDGE
from models.dual_frequency_encoder import DualFrequencyEncoder
from proposal import DecoupledGateBRIDGE


TRAIN = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0]], dtype=np.float32)
GATE_NAMES = ('behavior_user_bias.weight', 'behavior_item_bias.weight',
              'gate_base_scale', 'gate_behavior_scale', 'gate_residual_scale')


class Dataset:
    def inter_matrix(self, form='csr'):
        return sparse.csr_matrix(TRAIN).asformat(form)


def tiny_init(model, config, dataset):
    torch.nn.Module.__init__(model)
    model.n_users, model.n_items = TRAIN.shape
    model.behavior_sim_method = 'cosine'
    model.behavior_aggregation = 'sum_sqrt'
    model.behavior_score_norm = 'z'
    for name in ('use_behavior_proto', 'use_private_residual', 'use_residual_gate',
                 'use_activity_simplex', 'use_user_graph'):
        setattr(model, name, False)
    model.use_behavior_correction = model.use_calibration_gate = model.use_behavior_gate = True
    model.private_residual_weight = model.msr_weight = model.behavior_proto_loss_weight = 0.0
    model.behavior_proto_gate_reg_weight = 0.0
    model.low_band_count = 1
    model.private_residual_transform = 'raw'
    model.behavior_weight, model.gate_floor, model.gate_target = 0.4, 0.0, 0.5
    model.ib_weight, model.reg_weight = 0.1, 0.02
    model.aux_base_weight, model.aux_low_weight, model.aux_high_weight = 0.2, 0.11, 0.07
    model.gate_reg_weight = 0.001
    model.behavior_user_bias = torch.nn.Embedding(3, 1)
    model.behavior_item_bias = torch.nn.Embedding(4, 1)
    torch.nn.init.zeros_(model.behavior_user_bias.weight)
    torch.nn.init.zeros_(model.behavior_item_bias.weight)
    model.gate_base_scale = torch.nn.Parameter(torch.tensor(0.15))
    model.gate_behavior_scale = torch.nn.Parameter(torch.tensor(0.8))
    model.gate_residual_scale = torch.nn.Parameter(torch.tensor(0.0))
    model.encoder_users = torch.nn.Parameter(torch.tensor([[.2, .3, -.1], [.5, -.2, .1], [.3, .1, .2]]))
    model.encoder_items = torch.nn.Parameter(torch.tensor([[.1, .2, .4], [.2, -.1, .3], [.5, .1, -.2], [.1, .3, .2]]))
    model.behavior_score_matrix = torch.tensor([[1., 2., .5, -.3], [2., .1, 1., -.2], [.2, 1., 2., -.4]])
    model.frequency_contrastive_loss = lambda users, items: torch.stack(
        [value.square().mean() for value in users + items]).mean()


def tiny_encoder(model):
    users, items = model.encoder_users, model.encoder_items
    ib = users.square().mean() + items.square().mean()
    return users, items, ib, [users * f for f in (.5, .3, .2)], [items * f for f in (.5, .3, .2)]


class GradientTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = build_counterfactual_cache(sparse.csr_matrix(TRAIN), topk=2)
        self.path = Path(self.directory.name) / 'cache.npz'
        self.cache.save(self.path)
        self.hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.interaction = (torch.tensor([0, 1]), torch.tensor([1, 0]), torch.tensor([2, 3]))
        self.masks = (torch.tensor([1., 1.]), torch.tensor([1., 1.]))

    def make(self, mode='edge', strength=0.5):
        config = {'evidence_mode': 'edge', 'gate_evidence_mode': mode,
                  'isolation_strength': strength, 'evidence_cache': str(self.path),
                  'evidence_cache_sha256': self.hash}
        with mock.patch.object(BRIDGE, '__init__', tiny_init):
            model = DecoupledGateBRIDGE(config, Dataset())
            reference = BRIDGE(config, Dataset())
        reference.load_state_dict(model.state_dict(), strict=True)
        return model, reference

    def loss(self, model):
        with mock.patch.object(DualFrequencyEncoder, 'forward', tiny_encoder), mock.patch.object(
                BRIDGE, '_sampled_candidate_masks', return_value=self.masks):
            return model.calculate_loss(self.interaction)

    def gate_objective(self, model, mode, strength):
        users, pos, neg = self.interaction
        base = [(model.encoder_users[users] * model.encoder_items[items]).sum(dim=1)
                for items in (pos, neg)]
        original = [model.behavior_score_matrix[users, items] for items in (pos, neg)]
        if mode == 'original':
            adjusted = original
        else:
            isolated = self.cache.get_pair(*[v.numpy() for v in self.interaction], mode=mode)
            adjusted = [value + strength * (torch.tensor(cf) - value)
                        for value, cf in zip(original, isolated)]
        final, original_gates = [], []
        for index, items in enumerate((pos, neg)):
            gate = BRIDGE._pair_gate(model, users, items, base[index].detach(), adjusted[index],
                                    torch.zeros_like(base[index]))
            final.append(base[index].detach() + .4 * self.masks[index] * gate * adjusted[index])
            original_gates.append(BRIDGE._pair_gate(model, users, items, base[index], original[index],
                                                   torch.zeros_like(base[index])))
        active = torch.cat(self.masks) > 0
        gates = torch.cat(original_gates)
        reg = (gates[active].mean() - .5).square() if active.any() else base[0].new_tensor(0.)
        return torch.nn.functional.softplus(-(final[0] - final[1])).mean() + .001 * reg

    def assert_grad(self, actual, expected):
        if actual is None or expected is None:
            self.assertIs(actual, expected)
        else:
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)

    def test_encoder_gradient_equals_original_with_auxiliary_losses_intact(self):
        for mode in ('edge', 'history'):
            model, reference = self.make(mode)
            actual, original = self.loss(model), self.loss(reference)
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
            actual.backward()
            original.backward()
            for name in ('encoder_users', 'encoder_items'):
                self.assert_grad(dict(model.named_parameters())[name].grad,
                                 dict(reference.named_parameters())[name].grad)

    def test_gate_gradients_equal_independent_counterfactual_objective_only(self):
        for mode in ('edge', 'history'):
            model, reference = self.make(mode)
            self.loss(model).backward()
            self.gate_objective(reference, mode, .5).backward()
            for name in GATE_NAMES:
                self.assert_grad(dict(model.named_parameters())[name].grad,
                                 dict(reference.named_parameters())[name].grad)
            self.assertIsNone(reference.encoder_users.grad)
            self.assertIsNone(reference.encoder_items.grad)

    def test_original_evidence_recovers_all_original_gradients_and_rng(self):
        model, reference = self.make('original')
        state = torch.get_rng_state().clone()
        numpy_state = np.random.get_state()
        self.loss(model).backward()
        self.loss(reference).backward()
        for name, parameter in model.named_parameters():
            self.assert_grad(parameter.grad, dict(reference.named_parameters())[name].grad)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
        self.assertEqual(numpy_state[2:], np.random.get_state()[2:])

    def test_counterfactual_changes_gate_gradients_without_changing_schema(self):
        model, reference = self.make()
        self.assertEqual(list(model.state_dict()), list(reference.state_dict()))
        self.loss(model).backward()
        self.loss(reference).backward()
        delta = (model.behavior_user_bias.weight.grad - reference.behavior_user_bias.weight.grad).abs().sum()
        self.assertGreater(delta.item(), 1e-6)
        self.assertIs(DecoupledGateBRIDGE.full_sort_predict, BRIDGE.full_sort_predict)
        self.assertIs(DecoupledGateBRIDGE._pair_behavior_scores,
                      DecoupledGateBRIDGE.__mro__[1]._pair_behavior_scores)
        self.assertIsNone(model._evidence_context)
        self.assertIsNone(model._gate_context)

    def test_masked_pairs_have_no_counterfactual_gate_ranking_gradients(self):
        self.masks = (torch.zeros(2), torch.zeros(2))
        model, reference = self.make()
        self.loss(model).backward()
        self.loss(reference).backward()
        for name, parameter in model.named_parameters():
            self.assert_grad(parameter.grad, dict(reference.named_parameters())[name].grad)
        diagnostics = model.train_diagnostics()
        self.assertEqual(diagnostics['active_positive'], 0)
        self.assertEqual(diagnostics['active_negative'], 0)

    def test_strength_zero_recovers_original_and_one_matches_full_gate_isolation(self):
        for strength in (0., 1.):
            model, reference = self.make(strength=strength)
            self.loss(model).backward()
            if strength == 0:
                self.loss(reference).backward()
                for name, parameter in model.named_parameters():
                    self.assert_grad(parameter.grad, dict(reference.named_parameters())[name].grad)
            else:
                self.gate_objective(reference, 'edge', 1.).backward()
                for name in GATE_NAMES:
                    self.assert_grad(dict(model.named_parameters())[name].grad,
                                     dict(reference.named_parameters())[name].grad)

    def test_original_scoring_is_restored_after_loss_exception(self):
        model, _ = self.make()
        with mock.patch.object(BRIDGE, 'calculate_loss', side_effect=ArithmeticError('forced')):
            with self.assertRaises(ArithmeticError):
                model.calculate_loss(self.interaction)
        self.assertIsNone(model._gate_context)
        users, positive, _ = self.interaction
        torch.testing.assert_close(model._pair_behavior_scores(users, positive),
                                   model.behavior_score_matrix[users, positive])
        self.assertEqual(model.train_diagnostics()['count'], 0)

    def test_dispatch_or_missing_main_objective_fails_and_restores_context(self):
        model, _ = self.make()

        def wrong_dispatch(self, interaction):
            return self._combine_score_components(interaction[0], interaction[2],
                                                  torch.ones(2), torch.ones(2), torch.zeros(2))

        for function in (wrong_dispatch, lambda self, interaction: torch.tensor(0.)):
            with mock.patch.object(BRIDGE, 'calculate_loss', function):
                with self.assertRaises(RuntimeError):
                    model.calculate_loss(self.interaction)
            self.assertIsNone(model._gate_context)

    def test_diagnostics_report_separate_objectives_and_reset(self):
        model, _ = self.make()
        self.loss(model)
        report = model.train_diagnostics()
        self.assertEqual(report['count'], 2)
        self.assertEqual(report['active_positive'], 2)
        self.assertGreater(report['mean_positive_credit_abs_difference'], 0)
        self.assertNotEqual(report['mean_original_main'], report['mean_gate_main'])
        self.assertTrue(report['surrogate_forward_value_is_original_BPR'])
        self.assertEqual(model.train_diagnostics()['count'], 0)


if __name__ == '__main__':
    unittest.main()
