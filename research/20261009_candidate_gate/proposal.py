"""Train the existing BRIDGE gate against its actual candidate negatives."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse

from models.bridge import BRIDGE


class CandidateGateBRIDGE(BRIDGE):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.gate_supervision = str(self._cfg(config, 'gate_supervision', 'candidate'))
        if self.gate_supervision not in ('candidate', 'joint', 'original'):
            raise ValueError('Unknown gate_supervision: ' + self.gate_supervision)
        if self.behavior_score_matrix is None:
            raise ValueError('Candidate supervision requires the dense behavior backend')
        for name in ('use_behavior_proto', 'use_private_residual', 'use_residual_gate',
                     'use_activity_simplex', 'use_user_graph'):
            if getattr(self, name):
                raise ValueError('Candidate supervision requires disabled ' + name)
        if not all((self.use_behavior_correction, self.use_calibration_gate, self.use_behavior_gate,
                    self.use_topk_correction, self.train_candidate_aware)):
            raise ValueError('Candidate supervision requires the original candidate gate')
        if self.correction_scope != 'topk' or self.behavior_eval_topk <= 0:
            raise ValueError('Candidate supervision requires top-K correction')
        if self.train_candidate_topk != self.behavior_eval_topk:
            raise ValueError('Training and prediction candidate budgets must match')
        if self.private_residual_weight != 0.0 or self.msr_weight != 0.0:
            raise ValueError('Private residual correction and MSR must be disabled')
        for name, expected in (('behavior_sim_method', 'cosine'),
                               ('behavior_aggregation', 'sum_sqrt'), ('behavior_score_norm', 'z')):
            if getattr(self, name) != expected:
                raise ValueError('Candidate supervision requires ' + name + '=' + expected)
        train = sparse.csr_matrix(dataset.inter_matrix(form='csr'), copy=True)
        train.sum_duplicates()
        train.eliminate_zeros()
        train.sort_indices()
        if train.shape != (self.n_users, self.n_items) or not np.all(train.data == 1):
            raise ValueError('Expected the binary training-only interaction matrix')
        self._candidate_train = train
        self._gate_context = None
        self._gate_statistics = {}

    def calculate_loss(self, interaction):
        if self.gate_supervision == 'original':
            return BRIDGE.calculate_loss(self, interaction)
        if self._gate_context is not None:
            raise RuntimeError('Nested candidate-gate loss calls are unsupported')
        users, positive, negative = interaction[:3]
        if any(value.ndim != 1 for value in (users, positive, negative)):
            raise ValueError('Expected one-dimensional training pair tensors')
        if not (users.numel() == positive.numel() == negative.numel()) or not users.numel():
            raise ValueError('Training pair tensors must have matching nonempty lengths')
        context = {'users': users.detach().clone(),
                   'items': [positive.detach().clone(), negative.detach().clone()],
                   'pairs': [], 'bpr_calls': 0, 'main_consumed': False}
        self._gate_context = context
        try:
            result = BRIDGE.calculate_loss(self, interaction)
            if len(context['pairs']) != 2 or not context['main_consumed']:
                raise RuntimeError('BRIDGE did not consume the expected training objective')
            self._record_gate_diagnostics(context)
            return result
        finally:
            self._gate_context = None

    def _sampled_candidate_masks(self, user_emb_batch, item_emb_all, pos_items, neg_items):
        context = self._gate_context
        if context is None:
            return BRIDGE._sampled_candidate_masks(
                self, user_emb_batch, item_emb_all, pos_items, neg_items)
        if 'candidate_idx' in context or not all(torch.equal(actual, expected) for actual, expected
                in zip((pos_items, neg_items), context['items'])):
            raise RuntimeError('Candidate-mask dispatch differs from the training pair')
        candidate_k = min(max(self.train_candidate_topk, 50), self.n_items)
        with torch.no_grad():
            base = torch.matmul(user_emb_batch.detach(), item_emb_all.detach().transpose(0, 1))
            candidate_base, candidate_idx = torch.topk(base, k=candidate_k, dim=1)
            pos_mask = (candidate_idx == pos_items.view(-1, 1)).any(dim=1)
            neg_mask = (candidate_idx == neg_items.view(-1, 1)).any(dim=1)
            users_np = context['users'].cpu().numpy()
            candidates_np = candidate_idx.cpu().numpy()
            valid_negative_np = np.ones(candidates_np.shape, dtype=np.bool_)
            # Filter the raw retrieval set; replacing removed items would
            # change the correction scope inherited from BRIDGE inference.
            for row, user in enumerate(users_np):
                start = self._candidate_train.indptr[int(user)]
                end = self._candidate_train.indptr[int(user) + 1]
                if start != end:
                    valid_negative_np[row] = ~np.isin(
                        candidates_np[row], self._candidate_train.indices[start:end], assume_unique=True)
            valid_negative = torch.as_tensor(valid_negative_np, device=candidate_idx.device)
        context.update(candidate_idx=candidate_idx, candidate_base=candidate_base,
                       valid_negative=valid_negative,
                       eligible=pos_mask & valid_negative.any(dim=1),
                       masks=[pos_mask.to(user_emb_batch.dtype), neg_mask.to(user_emb_batch.dtype)])
        return tuple(context['masks'])

    def _route_pair_scores(self, users, pos_items, neg_items, pos_base, neg_base,
                           user_emb, item_emb_all, pos_candidate_mask=None, neg_candidate_mask=None):
        context = self._gate_context
        if context is None:
            return BRIDGE._route_pair_scores(
                self, users, pos_items, neg_items, pos_base, neg_base, user_emb,
                item_emb_all, pos_candidate_mask, neg_candidate_mask)
        if 'candidate_idx' not in context or 'user_emb' in context:
            raise RuntimeError('Candidate retrieval must occur once before routing')
        for actual, expected in zip((users, pos_items, neg_items),
                                    [context['users']] + context['items']):
            if not torch.equal(actual, expected):
                raise RuntimeError('Candidate routing differs from the training pair')
        for actual, expected in zip((pos_candidate_mask, neg_candidate_mask), context['masks']):
            if actual is None or not torch.equal(actual, expected):
                raise RuntimeError('Candidate routing received different correction masks')
        context.update(user_emb=user_emb, item_emb_all=item_emb_all)
        return pos_base, neg_base, pos_base.new_tensor(0.0)

    def _combine_score_components(self, users, items, base_scores, behavior_scores,
                                  residual_scores, correction_mask=None):
        context = self._gate_context
        if context is None:
            return BRIDGE._combine_score_components(
                self, users, items, base_scores, behavior_scores, residual_scores, correction_mask)
        index = len(context['pairs'])
        if index >= 2 or not torch.equal(users, context['users']) or not torch.equal(
                items, context['items'][index]):
            raise RuntimeError('Candidate-gate scoring differs from the training pair')
        if correction_mask is None or not torch.equal(correction_mask, context['masks'][index]):
            raise RuntimeError('Candidate-gate scoring received different correction masks')
        original = BRIDGE._combine_score_components(
            self, users, items, base_scores, behavior_scores, residual_scores, correction_mask)
        context['pairs'].append({'base': base_scores, 'behavior': behavior_scores,
                                 'residual': residual_scores, 'original_final': original[0]})
        if self.gate_supervision == 'candidate':
            encoder_final = base_scores + original[2].detach() + original[3].detach()
            return encoder_final, original[1], original[2].detach(), original[3].detach()
        return original

    def _candidate_objective(self, context):
        detach_base = self.gate_supervision != 'joint'
        sampled = []
        for index, pair in enumerate(context['pairs']):
            base = pair['base'].detach() if detach_base else pair['base']
            sampled.append(BRIDGE._combine_score_components(
                self, context['users'], context['items'][index], base, pair['behavior'],
                pair['residual'].detach(), context['masks'][index])[0])
        fallback_rows = F.softplus(-(sampled[0] - sampled[1]))
        if detach_base:
            candidate_base = context['candidate_base']
        else:
            candidate_base = torch.matmul(
                context['user_emb'], context['item_emb_all'].transpose(0, 1)).gather(
                    1, context['candidate_idx'])
        candidate_behavior = self.behavior_score_matrix[
            context['users'].view(-1, 1), context['candidate_idx']]
        candidate_scores = BRIDGE._combine_score_components(
            self, context['users'], context['candidate_idx'], candidate_base, candidate_behavior,
            torch.zeros_like(candidate_base), torch.ones_like(candidate_base))[0]
        weights = context['valid_negative'].to(candidate_base.dtype)
        candidate_rows = (F.softplus(-(sampled[0].view(-1, 1) - candidate_scores))
                          * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        selected_rows = torch.where(context['eligible'], candidate_rows, fallback_rows)
        context['candidate_rows'] = candidate_rows.detach()
        context['selected_main'] = selected_rows.mean().detach()
        context['fallback_main'] = fallback_rows.mean().detach()
        return selected_rows.mean()

    def _bpr_from_scores(self, positive, negative):
        original = BRIDGE._bpr_from_scores(positive, negative)
        context = self._gate_context
        if context is None:
            return original
        index = context['bpr_calls']
        context['bpr_calls'] += 1
        if index:
            return original
        if len(context['pairs']) != 2 or 'user_emb' not in context:
            raise RuntimeError('Main BPR called before candidate pair scoring')
        objective = self._candidate_objective(context)
        context['original_main'] = original.detach()
        context['main_consumed'] = True
        if self.gate_supervision == 'joint':
            return original.detach() + (objective - objective.detach())
        return original + (objective - objective.detach())

    def _record_gate_diagnostics(self, context):
        statistics = self._gate_statistics
        count = context['users'].numel()
        statistics['count'] = statistics.get('count', 0) + count
        statistics['batches'] = statistics.get('batches', 0) + 1
        for index, label in enumerate(('positive', 'negative')):
            key = 'active_' + label
            statistics[key] = statistics.get(key, 0) + int(context['masks'][index].sum().item())
        eligible = context['eligible']
        valid_counts = context['valid_negative'].sum(dim=1)
        for key, value in (('eligible', int(eligible.sum().item())),
                           ('candidate_negative_pairs', int(valid_counts.sum().item())),
                           ('eligible_negative_pairs', int(valid_counts[eligible].sum().item()))):
            statistics[key] = statistics.get(key, 0) + value
        for key in ('original_main', 'selected_main', 'fallback_main'):
            statistics[key] = statistics.get(key, 0.0) + context[key].double().item() * count
        statistics['candidate_main_sum'] = statistics.get('candidate_main_sum', 0.0) + (
            context['candidate_rows'][eligible].double().sum().item())

    def train_diagnostics(self, reset=True):
        statistics = self._gate_statistics
        count = statistics.get('count', 0)
        result = {'mode': self.gate_supervision, 'count': count,
                  'batches': statistics.get('batches', 0),
                  'surrogate_forward_value_is_original_BPR': True}
        for key in ('active_positive', 'active_negative', 'eligible',
                    'candidate_negative_pairs', 'eligible_negative_pairs'):
            result[key] = statistics.get(key, 0)
        result['fallback'] = count - result['eligible']
        if count:
            result['positive_scope_rate'] = result['active_positive'] / count
            result['random_negative_scope_rate'] = result['active_negative'] / count
            result['eligible_rate'] = result['eligible'] / count
            result['mean_candidate_negative_count'] = result['candidate_negative_pairs'] / count
            result['mean_eligible_candidate_negative_count'] = (
                result['eligible_negative_pairs'] / max(result['eligible'], 1))
            for key in ('original_main', 'selected_main', 'fallback_main'):
                result['mean_' + key] = statistics[key] / count
            result['mean_candidate_main'] = statistics.get('candidate_main_sum', 0.0) / max(
                result['eligible'], 1)
            result['mean_gate_main'] = result['mean_selected_main']
            result['random_negative_in_scope_rate'] = result['random_negative_scope_rate']
            result['eligible_pair_rate'] = result['eligible_rate']
            result['mean_valid_candidate_negatives_eligible'] = (
                result['mean_eligible_candidate_negative_count'])
        if reset:
            self._gate_statistics = {}
        return result
