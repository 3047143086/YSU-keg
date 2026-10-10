# coding: utf-8

"""BRIDGE with confidence-bounded candidate repair (CBCR)."""

import torch
import torch.nn.functional as F

from models.bridge import BRIDGE
from models.dual_frequency_encoder import DualFrequencyEncoder


class BRIDGE_CBCR(BRIDGE):
    """Train BRIDGE's candidate correction with a base-aware repair objective.

    CBCR treats the base ranking as the retrieval anchor. It emphasizes sampled
    candidate pairs whose detached base margin is wrong or uncertain, while a
    confidence-weighted budget prevents the correction from perturbing pairs
    that the base model already ranks confidently.
    """

    def __init__(self, config, dataset):
        super(BRIDGE_CBCR, self).__init__(config, dataset)
        self.cbcr_weight = float(self._cfg(config, "cbcr_weight", 0.2))
        self.cbcr_temperature = float(self._cfg(config, "cbcr_temperature", 1.0))
        self.cbcr_guard_weight = float(self._cfg(config, "cbcr_guard_weight", 0.05))
        print(
            "CBCR loss: "
            f"weight={self.cbcr_weight}, temperature={self.cbcr_temperature}, "
            f"guard_weight={self.cbcr_guard_weight}"
        )

    @staticmethod
    def _cbcr_loss(
        base_margin,
        correction_delta,
        active_mask,
        temperature=1.0,
        guard_weight=0.05,
    ):
        """Return CBCR, boundary-repair, and confidence-budget losses.

        The base margin and its batch scale are detached, so this auxiliary
        objective optimizes only the candidate correction. The ordinary BRIDGE
        BPR terms continue to train the retrieval backbone.
        """
        mask = active_mask.to(dtype=base_margin.dtype)
        count = mask.sum().clamp_min(1.0)
        detached_margin = base_margin.detach()
        margin_scale = torch.sqrt(
            (mask * detached_margin.square()).sum() / count
        ).clamp_min(1e-6)
        scaled_margin = detached_margin / margin_scale
        temperature = max(float(temperature), 1e-6)

        repair_weight = torch.sigmoid(
            (-scaled_margin / temperature).clamp(min=-12.0, max=12.0)
        ) * mask
        repaired_margin = scaled_margin + correction_delta / margin_scale
        repair_loss = (
            repair_weight * F.softplus(-repaired_margin)
        ).sum() / repair_weight.sum().clamp_min(1.0)

        confidence_weight = (mask - repair_weight).clamp_min(0.0)
        normalized_correction = correction_delta / margin_scale
        guard_loss = (
            confidence_weight * normalized_correction.square()
        ).sum() / confidence_weight.sum().clamp_min(1.0)

        return repair_loss + float(guard_weight) * guard_loss, repair_loss, guard_loss

    def calculate_loss(self, interaction):
        user_indices = interaction[0]
        pos_item_indices = interaction[1]
        neg_item_indices = interaction[2]

        raw_user_emb_all, item_emb_all, ib_loss, user_frequencies, item_frequencies = (
            DualFrequencyEncoder.forward(self)
        )
        user_emb_all, proto_user_emb_all, proto_gate = self._apply_behavior_proto(
            raw_user_emb_all, item_emb_all
        )
        low_user, low_item, high_user, high_item = self._frequency_views(
            raw_user_emb_all, item_emb_all, user_frequencies, item_frequencies
        )

        user_emb_batch = user_emb_all[user_indices]
        pos_item_emb_batch = item_emb_all[pos_item_indices]
        neg_item_emb_batch = item_emb_all[neg_item_indices]

        pos_base = torch.sum(user_emb_batch * pos_item_emb_batch, dim=1)
        neg_base = torch.sum(user_emb_batch * neg_item_emb_batch, dim=1)

        low_user_batch = low_user[user_indices]
        high_user_batch = high_user[user_indices]
        pos_low = torch.sum(low_user_batch * low_item[pos_item_indices], dim=1)
        neg_low = torch.sum(low_user_batch * low_item[neg_item_indices], dim=1)
        pos_high = torch.sum(high_user_batch * high_item[pos_item_indices], dim=1)
        neg_high = torch.sum(high_user_batch * high_item[neg_item_indices], dim=1)

        pos_residual = self._transform_residual(pos_high - pos_low)
        neg_residual = self._transform_residual(neg_high - neg_low)
        pos_behavior = self._pair_behavior_scores(user_indices, pos_item_indices)
        neg_behavior = self._pair_behavior_scores(user_indices, neg_item_indices)
        if self.behavior_score_matrix is None and self.behavior_score_norm == "z":
            behavior_pair = torch.cat([pos_behavior, neg_behavior], dim=0)
            behavior_pair = (behavior_pair - behavior_pair.mean()) / behavior_pair.std(
                unbiased=False
            ).clamp_min(1e-6)
            pos_behavior, neg_behavior = torch.chunk(behavior_pair, 2, dim=0)
        elif self.behavior_score_matrix is None and self.behavior_score_norm == "minmax":
            behavior_pair = torch.cat([pos_behavior, neg_behavior], dim=0)
            lo = behavior_pair.min()
            hi = behavior_pair.max()
            behavior_pair = (behavior_pair - lo) / (hi - lo).clamp_min(1e-6)
            pos_behavior, neg_behavior = torch.chunk(behavior_pair, 2, dim=0)

        pos_candidate_mask, neg_candidate_mask = self._sampled_candidate_masks(
            user_emb_batch, item_emb_all, pos_item_indices, neg_item_indices
        )
        (
            pos_final,
            pos_gate,
            pos_behavior_correction,
            pos_residual_correction,
        ) = self._combine_score_components(
            user_indices,
            pos_item_indices,
            pos_base,
            pos_behavior,
            pos_residual,
            pos_candidate_mask,
        )
        (
            neg_final,
            neg_gate,
            neg_behavior_correction,
            neg_residual_correction,
        ) = self._combine_score_components(
            user_indices,
            neg_item_indices,
            neg_base,
            neg_behavior,
            neg_residual,
            neg_candidate_mask,
        )

        main_loss = self._bpr_from_scores(pos_final, neg_final)
        base_loss = self._bpr_from_scores(pos_base, neg_base)
        low_loss = self._bpr_from_scores(pos_low, neg_low)
        high_loss = self._bpr_from_scores(pos_high, neg_high)
        zero_loss = pos_base.new_tensor(0.0)

        base_margin = pos_base - neg_base
        correction_delta = (
            pos_behavior_correction
            + pos_residual_correction
            - neg_behavior_correction
            - neg_residual_correction
        )
        active_pair_mask = torch.maximum(pos_candidate_mask, neg_candidate_mask)
        cbcr_loss, _, _ = self._cbcr_loss(
            base_margin,
            correction_delta,
            active_pair_mask,
            temperature=self.cbcr_temperature,
            guard_weight=self.cbcr_guard_weight,
        )

        if self.use_behavior_proto and self.behavior_proto_loss_weight != 0.0:
            proto_user_batch = proto_user_emb_all[user_indices]
            pos_proto = torch.sum(proto_user_batch * pos_item_emb_batch, dim=1)
            neg_proto = torch.sum(proto_user_batch * neg_item_emb_batch, dim=1)
            proto_loss = self._bpr_from_scores(pos_proto, neg_proto)
        else:
            proto_loss = zero_loss

        user_band_emb_batch = [band[user_indices] for band in user_frequencies]
        item_band_emb_batch = [band[pos_item_indices] for band in item_frequencies]
        cl_loss = self.frequency_contrastive_loss(
            user_band_emb_batch, item_band_emb_batch
        )

        active_gate_mask = torch.cat(
            [pos_candidate_mask, neg_candidate_mask], dim=0
        ) > 0
        gate_values = torch.cat([pos_gate, neg_gate], dim=0)
        if active_gate_mask.any():
            gate_mean = gate_values[active_gate_mask].mean()
            gate_reg = (gate_mean - self.gate_target) ** 2
        else:
            gate_reg = zero_loss
        if self.use_behavior_proto and self.behavior_proto_gate_reg_weight != 0.0:
            proto_gate_reg = (
                proto_gate.mean() - self.behavior_proto_gate_target
            ) ** 2
        else:
            proto_gate_reg = zero_loss

        return (
            main_loss
            + ib_loss * self.ib_weight
            + cl_loss * self.reg_weight
            + self.aux_base_weight * base_loss
            + self.aux_low_weight * low_loss
            + self.aux_high_weight * high_loss
            + self.gate_reg_weight * gate_reg
            + self.behavior_proto_loss_weight * proto_loss
            + self.behavior_proto_gate_reg_weight * proto_gate_reg
            + self.cbcr_weight * cbcr_loss
        )

    def diagnostic_info(self):
        return (
            super(BRIDGE_CBCR, self).diagnostic_info()
            + f", CBCR(weight={self.cbcr_weight:.3g}, "
            + f"temperature={self.cbcr_temperature:.3g}, "
            + f"guard={self.cbcr_guard_weight:.3g})"
        )
