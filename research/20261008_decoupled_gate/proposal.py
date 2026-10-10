"""Assign counterfactual ranking supervision to the existing gate only."""
import numpy as np
import torch

from isolation import TargetIsolatedBRIDGE
from models.bridge import BRIDGE


class DecoupledGateBRIDGE(TargetIsolatedBRIDGE):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.gate_evidence_mode = str(self._cfg(config, 'gate_evidence_mode', 'edge'))
        if self.gate_evidence_mode not in ('original', 'history', 'edge'):
            raise ValueError('Unknown gate_evidence_mode: ' + self.gate_evidence_mode)
        if self._evidence_cache is None or self.evidence_mode != 'edge':
            raise ValueError('Decoupled supervision requires the verified edge cache')
        if not self.use_calibration_gate or not self.use_behavior_gate:
            raise ValueError('Decoupled supervision requires the original behavior gate')
        if not 0 <= self.gate_floor < 1:
            raise ValueError('The gate floor must be in [0, 1)')
        self._gate_context = None
        self._gate_statistics = {}

    def calculate_loss(self, interaction):
        if self._gate_context is not None:
            raise RuntimeError('Nested decoupled-gate loss calls are unsupported')
        users, positive, negative = interaction[:3]
        if any(value.ndim != 1 for value in (users, positive, negative)):
            raise ValueError('Expected one-dimensional training pair tensors')
        original = [BRIDGE._pair_behavior_scores(self, users, items).detach()
                    for items in (positive, negative)]
        if self.gate_evidence_mode == 'original':
            adjusted = original
        else:
            query = [value.detach().cpu().numpy() for value in (users, positive, negative)]
            isolated = self._evidence_cache.get_pair(*query, mode=self.gate_evidence_mode)
            adjusted = []
            for reference, value in zip(original, isolated):
                base = reference.cpu().numpy()
                blended = base + self.isolation_strength * (value - base)
                if not np.all(np.isfinite(blended)):
                    raise FloatingPointError('Non-finite counterfactual gate evidence')
                adjusted.append(torch.as_tensor(blended, device=reference.device,
                                                dtype=reference.dtype))
        context = {'users': users.detach().clone(),
                   'items': [positive.detach().clone(), negative.detach().clone()],
                   'adjusted': adjusted, 'original': original,
                   'pairs': [], 'bpr_calls': 0, 'main_consumed': False}
        self._gate_context = context
        try:
            # Bypass the parent's uniform evidence replacement. The encoder
            # and its auxiliary objectives see the original BRIDGE evidence.
            result = BRIDGE.calculate_loss(self, interaction)
            if len(context['pairs']) != 2 or not context['main_consumed']:
                raise RuntimeError('BRIDGE did not consume the expected ranking pair')
            self._record_gate_diagnostics(context)
            return result
        finally:
            self._gate_context = None

    def _combine_score_components(self, users, items, base_scores, behavior_scores,
                                  residual_scores, correction_mask=None):
        context = self._gate_context
        if context is None:
            return BRIDGE._combine_score_components(
                self, users, items, base_scores, behavior_scores, residual_scores, correction_mask)
        index = len(context['pairs'])
        if index >= 2 or not torch.equal(users, context['users']) or not torch.equal(
                items, context['items'][index]):
            raise RuntimeError('Decoupled-gate score dispatch differs from the training pair')
        original = BRIDGE._combine_score_components(
            self, users, items, base_scores, behavior_scores, residual_scores, correction_mask)
        counterfactual = BRIDGE._combine_score_components(
            self, users, items, base_scores.detach(), context['adjusted'][index],
            residual_scores.detach(), correction_mask)
        context['pairs'].append({'original_final': original[0],
                                 'counterfactual_final': counterfactual[0],
                                 'original_gate': original[1],
                                 'counterfactual_gate': counterfactual[1],
                                 'mask': (torch.ones_like(base_scores) if correction_mask is None
                                          else correction_mask.detach().to(base_scores.dtype))})
        encoder_final = base_scores + original[2].detach() + original[3].detach()
        # Original gates retain their original regularizer gradients. Ranking
        # gradients to gates come from the counterfactual main objective below.
        return encoder_final, original[1], original[2].detach(), original[3].detach()

    def _bpr_from_scores(self, positive, negative):
        original = BRIDGE._bpr_from_scores(positive, negative)
        context = self._gate_context
        if context is None:
            return original
        index = context['bpr_calls']
        context['bpr_calls'] += 1
        if index:
            return original
        if len(context['pairs']) != 2:
            raise RuntimeError('Main BPR called before the expected gate pair')
        counterfactual = BRIDGE._bpr_from_scores(
            context['pairs'][0]['counterfactual_final'],
            context['pairs'][1]['counterfactual_final'])
        context['original_main'] = original.detach()
        context['gate_main'] = counterfactual.detach()
        context['main_consumed'] = True
        return original + (counterfactual - counterfactual.detach())

    def _record_gate_diagnostics(self, context):
        statistics = self._gate_statistics
        count = context['users'].numel()
        statistics['count'] = statistics.get('count', 0) + count
        statistics['batches'] = statistics.get('batches', 0) + 1
        for key in ('original_main', 'gate_main'):
            statistics[key] = statistics.get(key, 0.0) + context[key].double().item() * count
        original_margin = (context['pairs'][0]['original_final']
                           - context['pairs'][1]['original_final']).detach()
        counterfactual_margin = (context['pairs'][0]['counterfactual_final']
                                - context['pairs'][1]['counterfactual_final']).detach()
        for index, label in enumerate(('positive', 'negative')):
            pair = context['pairs'][index]
            gate, adjusted_gate = pair['original_gate'].detach(), pair['counterfactual_gate'].detach()
            active = pair['mask'] > 0
            active_count = int(active.sum().item())
            key = 'active_' + label
            statistics[key] = statistics.get(key, 0) + active_count
            for name, value in (('original_' + label + '_z', context['original'][index]),
                                ('adjusted_' + label + '_z', context['adjusted'][index]),
                                ('original_' + label + '_gate', gate),
                                ('adjusted_' + label + '_gate', adjusted_gate)):
                statistics[name] = statistics.get(name, 0.0) + value.double().sum().item()
            # Log derivatives of the main mean BPR w.r.t. individual gate
            # logits, up to the shared 1/batch factor. No gradients are changed.
            direction = -1.0 if index == 0 else 1.0
            floor_scale = 1.0 - self.gate_floor
            sigma = (gate - self.gate_floor) / floor_scale
            sigma_cf = (adjusted_gate - self.gate_floor) / floor_scale
            credit = (direction * torch.sigmoid(-original_margin) * self.behavior_weight
                      * context['original'][index] * floor_scale * sigma * (1 - sigma) * pair['mask'])
            credit_cf = (direction * torch.sigmoid(-counterfactual_margin) * self.behavior_weight
                         * context['adjusted'][index] * floor_scale * sigma_cf * (1 - sigma_cf) * pair['mask'])
            for name, value in (('original_' + label + '_credit_abs', credit.abs()),
                                ('adjusted_' + label + '_credit_abs', credit_cf.abs()),
                                (label + '_credit_abs_difference', (credit_cf - credit).abs())):
                statistics[name] = statistics.get(name, 0.0) + value[active].double().sum().item()

    def train_diagnostics(self, reset=True):
        statistics = self._gate_statistics
        count = statistics.get('count', 0)
        result = {'mode': self.gate_evidence_mode, 'isolation_strength': self.isolation_strength,
                  'count': count, 'batches': statistics.get('batches', 0),
                  'surrogate_forward_value_is_original_BPR': True}
        if count:
            for key, value in statistics.items():
                if key not in ('count', 'batches') and not key.startswith('active_'):
                    denominator = count
                    for label in ('positive', 'negative'):
                        if label + '_credit' in key:
                            denominator = max(statistics.get('active_' + label, 0), 1)
                    result['mean_' + key] = value / denominator
            for label in ('positive', 'negative'):
                result['active_' + label] = statistics.get('active_' + label, 0)
        if reset:
            self._gate_statistics = {}
        return result
