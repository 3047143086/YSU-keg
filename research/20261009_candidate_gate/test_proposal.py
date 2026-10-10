"""Verify candidate gradient ownership using the actual BRIDGE loss hooks."""
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

local_carrier = Path(__file__).resolve().parents[2] / 'baselines/BRIDGE-official/src'
if local_carrier.is_dir():
    sys.path.insert(0, str(local_carrier))

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse

from models.bridge import BRIDGE
from models.dual_frequency_encoder import DualFrequencyEncoder
from proposal import CandidateGateBRIDGE


TRAIN = np.zeros((3, 60), dtype=np.float32)
TRAIN[0, [0, 1]] = 1
TRAIN[1, [58, 59]] = 1
TRAIN[2, [2, 3]] = 1
GATE_NAMES = ('behavior_user_bias.weight', 'behavior_item_bias.weight',
              'gate_base_scale', 'gate_behavior_scale', 'gate_residual_scale')


class Dataset:
    def __init__(self, train=None):
        self.train = TRAIN.copy() if train is None else np.asarray(train).copy()

    def inter_matrix(self, form='csr'):
        return sparse.csr_matrix(self.train).asformat(form)


def tiny_init(model, config, dataset):
    torch.nn.Module.__init__(model)
    model.n_users, model.n_items = dataset.train.shape
    model.device = torch.device('cpu')
    model.behavior_sim_method = 'cosine'
    model.behavior_aggregation = 'sum_sqrt'
    model.behavior_score_norm = 'z'
    for name in ('use_behavior_proto', 'use_private_residual', 'use_residual_gate',
                 'use_activity_simplex', 'use_user_graph'):
        setattr(model, name, False)
    model.use_behavior_correction = model.use_calibration_gate = model.use_behavior_gate = True
    model.use_topk_correction = model.train_candidate_aware = True
    model.correction_scope = 'topk'
    model.behavior_eval_topk = model.train_candidate_topk = 50
    model.private_residual_weight = model.msr_weight = model.behavior_proto_loss_weight = 0.0
    model.behavior_proto_gate_reg_weight = 0.0
    model.low_band_count = 1
    model.private_residual_transform = 'raw'
    model.behavior_weight, model.gate_floor, model.gate_target = 0.4, 0.0, 0.5
    model.ib_weight, model.reg_weight = 0.1, 0.02
    model.aux_base_weight, model.aux_low_weight, model.aux_high_weight = 0.2, 0.11, 0.07
    model.gate_reg_weight = 0.001
    model.behavior_user_bias = torch.nn.Embedding(model.n_users, 1)
    model.behavior_item_bias = torch.nn.Embedding(model.n_items, 1)
    torch.nn.init.zeros_(model.behavior_user_bias.weight)
    torch.nn.init.zeros_(model.behavior_item_bias.weight)
    model.gate_base_scale = torch.nn.Parameter(torch.tensor(0.15))
    model.gate_behavior_scale = torch.nn.Parameter(torch.tensor(0.8))
    model.gate_residual_scale = torch.nn.Parameter(torch.tensor(0.0))
    model.encoder_users = torch.nn.Parameter(torch.eye(3))
    ascending = torch.arange(1, model.n_items + 1, dtype=torch.float32) / 20
    model.encoder_items = torch.nn.Parameter(torch.stack(
        [ascending.flip(0), ascending, torch.roll(ascending, 20)], dim=1))
    model.behavior_score_matrix = torch.stack([
        torch.linspace(2., -1., model.n_items), torch.linspace(-.7, 1.8, model.n_items),
        torch.linspace(.2, 2.2, model.n_items)])
    model.frequency_contrastive_loss = lambda users, items: torch.stack(
        [value.square().mean() for value in users + items]).mean()


def tiny_encoder(model):
    users, items = model.encoder_users, model.encoder_items
    ib = users.square().mean() + items.square().mean()
    return users, items, ib, [users * f for f in (.5, .3, .2)], [items * f for f in (.5, .3, .2)]


def independent_gate(model, users, items, base, evidence):
    user_bias = model.behavior_user_bias.weight[users, 0]
    if items.ndim == 2:
        user_bias = user_bias[:, None]
    logits = (user_bias + model.behavior_item_bias.weight[items, 0]
              + model.gate_base_scale * torch.tanh(base.detach())
              + model.gate_behavior_scale * torch.tanh(evidence))
    return torch.sigmoid(logits)


def independent_objectives(model, interaction, train, detach_base=True):
    users, pos, neg = interaction
    matrix = model.encoder_users[users] @ model.encoder_items.T
    idx = matrix.detach().topk(50, dim=1).indices
    masks = [(idx == items[:, None]).any(dim=1).to(matrix.dtype) for items in (pos, neg)]
    sampled, gates = [], []
    for items, mask in zip((pos, neg), masks):
        base = (model.encoder_users[users] * model.encoder_items[items]).sum(dim=1)
        base = base.detach() if detach_base else base
        evidence = model.behavior_score_matrix[users, items]
        gate = independent_gate(model, users, items, base, evidence)
        sampled.append(base + .4 * gate * evidence * mask)
        gates.append(gate)
    fallback_rows = F.softplus(-(sampled[0] - sampled[1]))
    rows, counts, eligible = [], [], []
    for row, user in enumerate(users.tolist()):
        unseen = [item for item in idx[row].tolist() if train[user, item] == 0]
        counts.append(len(unseen))
        eligible.append(bool(masks[0][row].item()) and bool(unseen))
        if eligible[-1]:
            items = torch.tensor(unseen)
            base = matrix[row, items].detach() if detach_base else matrix[row, items]
            evidence = model.behavior_score_matrix[user, items]
            gate = independent_gate(model, users[row].expand_as(items), items, base, evidence)
            scores = base + .4 * gate * evidence
            rows.append(F.softplus(-(sampled[0][row] - scores)).mean())
        else:
            rows.append(fallback_rows[row])
    gate_values = torch.cat(gates)
    active = torch.cat(masks) > 0
    reg = (gate_values[active].mean() - .5).square() if active.any() else matrix.new_tensor(0.)
    return {'selected': torch.stack(rows).mean(), 'original': fallback_rows.mean(),
            'reg': .001 * reg, 'idx': idx, 'masks': masks,
            'counts': counts, 'eligible': eligible}


class GradientTests(unittest.TestCase):
    def setUp(self):
        self.interaction = (torch.tensor([0, 1]), torch.tensor([0, 59]), torch.tensor([59, 0]))

    def make(self, mode='candidate', train=None):
        dataset = Dataset(train)
        config = {'gate_supervision': mode}
        with mock.patch.object(BRIDGE, '__init__', tiny_init):
            model = CandidateGateBRIDGE(config, dataset)
            reference = BRIDGE(config, dataset)
        reference.load_state_dict(model.state_dict(), strict=True)
        return model, reference, dataset.train

    def loss(self, model, interaction=None):
        with mock.patch.object(DualFrequencyEncoder, 'forward', tiny_encoder):
            return model.calculate_loss(self.interaction if interaction is None else interaction)

    def assert_grad(self, actual, expected):
        if actual is None or expected is None:
            self.assertIs(actual, expected)
        else:
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)

    def compare_gradients(self, actual, expected, names=None):
        expected_parameters = dict(expected.named_parameters())
        for name, parameter in actual.named_parameters():
            if names is None or name in names:
                self.assert_grad(parameter.grad, expected_parameters[name].grad)

    def test_candidate_encoder_gradient_and_forward_equal_original_with_all_auxiliary_terms(self):
        model, reference, _ = self.make()
        actual, original = self.loss(model), self.loss(reference)
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
        actual.backward()
        original.backward()
        self.compare_gradients(model, reference, ('encoder_users', 'encoder_items'))

    def test_gate_gradient_matches_independent_candidate_formula_and_original_regularizer(self):
        model, reference, train = self.make()
        self.loss(model).backward()
        expected = independent_objectives(reference, self.interaction, train)
        (expected['selected'] + expected['reg']).backward()
        self.compare_gradients(model, reference, GATE_NAMES)
        self.assertIsNone(reference.encoder_users.grad)
        self.assertIsNone(reference.encoder_items.grad)

    def test_joint_encoder_gradient_matches_independent_candidate_formula_with_original_auxiliaries(self):
        model, reference, train = self.make('joint')
        actual, original = self.loss(model), self.loss(reference)
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
        expected = independent_objectives(reference, self.interaction, train, detach_base=False)
        total = original - expected['original'] + expected['selected']
        actual.backward()
        total.backward()
        self.compare_gradients(model, reference)

    def test_joint_and_candidate_gate_gradients_agree_at_fixed_parameters(self):
        candidate, _, _ = self.make('candidate')
        joint, _, _ = self.make('joint')
        joint.load_state_dict(candidate.state_dict(), strict=True)
        self.loss(candidate).backward()
        self.loss(joint).backward()
        self.compare_gradients(candidate, joint, GATE_NAMES)
        self.assertGreater((candidate.encoder_items.grad - joint.encoder_items.grad).abs().sum().item(), 1e-4)

    def test_original_mode_is_strict_gradient_parity_and_does_not_build_candidates(self):
        model, reference, _ = self.make('original')
        with mock.patch.object(model, '_candidate_objective', side_effect=AssertionError('unexpected')):
            actual, original = self.loss(model), self.loss(reference)
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
        actual.backward()
        original.backward()
        self.compare_gradients(model, reference)
        self.assertIsNone(model._gate_context)
        self.assertEqual(model.train_diagnostics()['count'], 0)

    def test_candidate_filters_every_training_positive_without_refilling_raw_topk(self):
        model, _, train = self.make()
        captured = {}
        original_route = model._route_pair_scores

        def inspect_route(*args, **kwargs):
            context = model._gate_context
            captured['idx'] = context['candidate_idx'].clone()
            captured['valid'] = context['valid_negative'].clone()
            return original_route(*args, **kwargs)

        with mock.patch.object(model, '_route_pair_scores', side_effect=inspect_route):
            self.loss(model)
        expected = independent_objectives(model, self.interaction, train)
        self.assertTrue(torch.equal(captured['idx'], expected['idx']))
        self.assertEqual(captured['idx'].shape, (2, 50))
        self.assertEqual(captured['valid'].sum(dim=1).tolist(), [48, 48])
        for row, user in enumerate(self.interaction[0].tolist()):
            selected = captured['idx'][row][captured['valid'][row]].tolist()
            self.assertTrue(all(train[user, item] == 0 for item in selected))
        self.assertNotIn(59, captured['idx'][0].tolist())
        self.assertNotIn(0, captured['idx'][1].tolist())

    def test_positive_outside_candidates_falls_back_to_original_sampled_gradient(self):
        train = TRAIN.copy()
        train[0, 59] = 1
        interaction = (torch.tensor([0]), torch.tensor([59]), torch.tensor([2]))
        for mode in ('candidate', 'joint'):
            model, reference, _ = self.make(mode, train)
            self.loss(model, interaction).backward()
            self.loss(reference, interaction).backward()
            self.compare_gradients(model, reference)
            report = model.train_diagnostics()
            self.assertEqual(report['eligible'], 0)
            self.assertEqual(report['fallback'], 1)
            self.assertEqual(report['active_positive'], 0)

    def test_no_valid_candidate_negative_falls_back_without_zero_division(self):
        train = TRAIN.copy()
        train[0, :50] = 1
        interaction = (torch.tensor([0]), torch.tensor([0]), torch.tensor([59]))
        for mode in ('candidate', 'joint'):
            model, reference, _ = self.make(mode, train)
            actual, original = self.loss(model, interaction), self.loss(reference, interaction)
            self.assertTrue(torch.isfinite(actual))
            actual.backward()
            original.backward()
            self.compare_gradients(model, reference)
            report = model.train_diagnostics()
            self.assertEqual(report['eligible'], 0)
            self.assertEqual(report['candidate_negative_pairs'], 0)
            self.assertEqual(report['fallback'], 1)

    def test_mixed_rows_use_mean_per_candidate_set_then_mean_over_entire_batch(self):
        train = TRAIN.copy()
        train[0, :25] = 1
        train[1, 10:60] = 1
        model, reference, train = self.make('candidate', train)
        self.loss(model).backward()
        expected = independent_objectives(reference, self.interaction, train)
        self.assertEqual(expected['counts'], [25, 0])
        self.assertEqual(expected['eligible'], [True, False])
        (expected['selected'] + expected['reg']).backward()
        self.compare_gradients(model, reference, GATE_NAMES)
        report = model.train_diagnostics()
        self.assertEqual(report['eligible_pair_rate'], .5)
        self.assertEqual(report['mean_valid_candidate_negatives_eligible'], 25.)

    def test_rng_and_parameter_checkpoint_schema_are_unchanged(self):
        for mode in ('candidate', 'joint', 'original'):
            model, reference, _ = self.make(mode)
            self.assertEqual(list(model.state_dict()), list(reference.state_dict()))
            self.assertEqual(list(dict(model.named_parameters())), list(dict(reference.named_parameters())))
            torch_state = torch.get_rng_state().clone()
            numpy_state = np.random.get_state()
            python_state = random.getstate()
            self.loss(model).backward()
            self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
            np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
            self.assertEqual(numpy_state[2:], np.random.get_state()[2:])
            self.assertEqual(python_state, random.getstate())

    def test_constructor_consumes_exactly_the_original_rng_draws(self):
        torch_state = torch.get_rng_state().clone()
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        with mock.patch.object(BRIDGE, '__init__', tiny_init):
            model = CandidateGateBRIDGE({}, Dataset())
            candidate_torch_state = torch.get_rng_state().clone()
            candidate_numpy_state = np.random.get_state()
            candidate_python_state = random.getstate()
            torch.set_rng_state(torch_state)
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            reference = BRIDGE({}, Dataset())
        self.assertTrue(torch.equal(candidate_torch_state, torch.get_rng_state()))
        np.testing.assert_array_equal(candidate_numpy_state[1], np.random.get_state()[1])
        self.assertEqual(candidate_numpy_state[2:], np.random.get_state()[2:])
        self.assertEqual(candidate_python_state, random.getstate())
        self.assertEqual(list(model.state_dict()), list(reference.state_dict()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)

    def test_prediction_path_is_inherited_and_identical_before_and_after_training_loss(self):
        model, reference, _ = self.make()
        self.assertIs(CandidateGateBRIDGE.full_sort_predict, BRIDGE.full_sort_predict)
        self.assertIs(CandidateGateBRIDGE._pair_behavior_scores, BRIDGE._pair_behavior_scores)
        self.assertIs(CandidateGateBRIDGE._candidate_behavior_scores, BRIDGE._candidate_behavior_scores)
        interaction = [torch.tensor([0, 1])]
        with mock.patch.object(DualFrequencyEncoder, 'forward', tiny_encoder):
            expected = reference.full_sort_predict(interaction)
            actual = model.full_sort_predict(interaction)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            model.calculate_loss(self.interaction)
            torch.testing.assert_close(model.full_sort_predict(interaction), expected, rtol=0, atol=0)

    def test_exception_restores_context_and_does_not_record_partial_diagnostics(self):
        model, reference, _ = self.make()
        with mock.patch.object(BRIDGE, 'calculate_loss', side_effect=ArithmeticError('forced')):
            with self.assertRaises(ArithmeticError):
                self.loss(model)
        self.assertIsNone(model._gate_context)
        self.assertEqual(model.train_diagnostics()['count'], 0)
        self.loss(model).backward()
        self.loss(reference).backward()
        self.compare_gradients(model, reference, ('encoder_users', 'encoder_items'))

    def test_wrong_dispatch_and_missing_main_objective_fail_and_restore_context(self):
        model, _, _ = self.make()

        def wrong_dispatch(instance, interaction):
            return instance._combine_score_components(interaction[0], interaction[2],
                                                       torch.ones(2), torch.ones(2), torch.zeros(2))

        for function in (wrong_dispatch, lambda instance, interaction: torch.tensor(0.)):
            with mock.patch.object(BRIDGE, 'calculate_loss', function):
                with self.assertRaises(RuntimeError):
                    self.loss(model)
            self.assertIsNone(model._gate_context)

    def test_diagnostics_report_separate_objectives_coverage_and_reset(self):
        model, _, _ = self.make()
        self.loss(model)
        report = model.train_diagnostics()
        self.assertEqual(report['count'], 2)
        self.assertEqual(report['batches'], 1)
        self.assertEqual(report['random_negative_in_scope_rate'], 0.)
        self.assertEqual(report['eligible_pair_rate'], 1.)
        self.assertEqual(report['mean_valid_candidate_negatives_eligible'], 48.)
        self.assertEqual(report['candidate_negative_pairs'], 96)
        self.assertEqual(report['fallback'], 0)
        self.assertNotEqual(report['mean_original_main'], report['mean_gate_main'])
        self.assertTrue(report['surrogate_forward_value_is_original_BPR'])
        self.assertEqual(model.train_diagnostics()['count'], 0)

    def test_constructor_rejects_unsupported_mode_sparse_backend_and_invalid_training_matrix(self):
        with mock.patch.object(BRIDGE, '__init__', tiny_init):
            with self.assertRaises(ValueError):
                CandidateGateBRIDGE({'gate_supervision': 'bad'}, Dataset())
            train = TRAIN.copy()
            train[0, 0] = 2
            with self.assertRaises(ValueError):
                CandidateGateBRIDGE({}, Dataset(train))

        def sparse_init(instance, config, dataset):
            tiny_init(instance, config, dataset)
            instance.behavior_score_matrix = None

        with mock.patch.object(BRIDGE, '__init__', sparse_init):
            with self.assertRaises(ValueError):
                CandidateGateBRIDGE({}, Dataset())


if __name__ == '__main__':
    unittest.main()
