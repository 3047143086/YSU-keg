# coding: utf-8

"""Candidate-conditional reliability routing for BRIDGE."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bridge import BRIDGE


class BRIDGE_C2R(BRIDGE):
    """BRIDGE with Candidate-Conditional Reliability Routing (C2R).

    The official BRIDGE encoder remains the retriever. C2R only reweights the
    three final score blocks (ID, visual, text) inside the detached candidate
    set. A pairwise reliability target is obtained from the three modality
    margins, and its confidence is reduced when the modalities disagree weakly.
    """

    def __init__(self, config, dataset):
        super(BRIDGE_C2R, self).__init__(config, dataset)
        self.c2r_loss_weight = float(self._cfg(config, "c2r_loss_weight", 0.05))
        self.c2r_balance_weight = float(self._cfg(config, "c2r_balance_weight", 0.005))
        self.c2r_temperature = float(self._cfg(config, "c2r_temperature", 1.0))
        self.c2r_confidence_temperature = float(
            self._cfg(config, "c2r_confidence_temperature", 1.0)
        )
        self.c2r_route_mix = float(self._cfg(config, "c2r_route_mix", 0.25))
        self.c2r_hidden_dim = int(self._cfg(config, "c2r_hidden_dim", 64))

        self.c2r_user_gate = self._make_gate()
        self.c2r_item_gate = self._make_gate()
        print(
            "C2R routing: "
            f"loss_weight={self.c2r_loss_weight}, "
            f"balance_weight={self.c2r_balance_weight}, "
            f"temperature={self.c2r_temperature}, "
            f"confidence_temperature={self.c2r_confidence_temperature}, "
            f"route_mix={self.c2r_route_mix}"
        )

    def _make_gate(self):
        gate = nn.Sequential(
            nn.LayerNorm(self.final_embed_dim),
            nn.Linear(self.final_embed_dim, self.c2r_hidden_dim),
            nn.GELU(),
            nn.Linear(self.c2r_hidden_dim, 3),
        )
        # Start as the official BRIDGE: all three score blocks have weight one.
        nn.init.zeros_(gate[-1].weight)
        nn.init.zeros_(gate[-1].bias)
        return gate

    def _gate_weights(self, embeddings, item_side=False):
        gate = self.c2r_item_gate if item_side else self.c2r_user_gate
        temperature = max(self.c2r_temperature, 1e-6)
        return F.softmax(gate(embeddings) / temperature, dim=-1)

    @staticmethod
    def _product_weights(user_weights, item_weights):
        weights = user_weights * item_weights
        return 3.0 * weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _weighted_block_score(user_blocks, item_blocks, weights):
        component_scores = (user_blocks * item_blocks).sum(dim=-1)
        return (component_scores * weights).sum(dim=-1)

    @staticmethod
    def _routing_confidence(weights):
        """Use normalized gate entropy as a conservative residual strength."""
        probs = (weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)).clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1)
        return (1.0 - entropy / torch.log(probs.new_tensor(3.0))).clamp(0.0, 1.0)

    def _route_pair_scores(
        self,
        users,
        pos_items,
        neg_items,
        pos_base,
        neg_base,
        user_emb,
        item_emb_all,
        pos_candidate_mask=None,
        neg_candidate_mask=None,
    ):
        del users
        dim = self.dim_latent
        user_blocks = user_emb.reshape(-1, 3, dim)
        pos_blocks = item_emb_all[pos_items].reshape(-1, 3, dim)
        neg_blocks = item_emb_all[neg_items].reshape(-1, 3, dim)

        user_weights = self._gate_weights(user_emb)
        pos_item_weights = self._gate_weights(
            pos_blocks.reshape(-1, 3 * dim), item_side=True
        )
        neg_item_weights = self._gate_weights(
            neg_blocks.reshape(-1, 3 * dim), item_side=True
        )
        pos_weights = self._product_weights(user_weights, pos_item_weights)
        neg_weights = self._product_weights(user_weights, neg_item_weights)

        pos_routed = self._weighted_block_score(user_blocks, pos_blocks, pos_weights)
        neg_routed = self._weighted_block_score(user_blocks, neg_blocks, neg_weights)

        # Apply routing as an entropy-controlled residual. Uniform gates recover
        # the official BRIDGE exactly; confident gates can refine the candidate.
        pos_strength = self.c2r_route_mix * self._routing_confidence(pos_weights)
        neg_strength = self.c2r_route_mix * self._routing_confidence(neg_weights)

        if pos_candidate_mask is None:
            pos_candidate_mask = torch.ones_like(pos_base)
        if neg_candidate_mask is None:
            neg_candidate_mask = torch.ones_like(neg_base)
        pos_routed = pos_base + pos_strength * (pos_routed - pos_base) * pos_candidate_mask
        neg_routed = neg_base + neg_strength * (neg_routed - neg_base) * neg_candidate_mask

        # Detached modality margins define the reliability target. The gate is
        # trained by this target, while the routed BPR remains fully trainable.
        modality_margin = (
            user_blocks * (pos_blocks - neg_blocks)
        ).sum(dim=-1)
        target_scale = modality_margin.detach().abs().mean(dim=-1, keepdim=True).clamp_min(1e-3)
        target_logits = modality_margin.detach() / target_scale
        target = F.softmax(target_logits / max(self.c2r_temperature, 1e-6), dim=-1)

        pair_weights = self._product_weights(
            user_weights, 0.5 * (pos_item_weights + neg_item_weights)
        )
        spread = modality_margin.detach().max(dim=-1).values - modality_margin.detach().min(dim=-1).values
        confidence_logits = (
            spread / (max(self.c2r_confidence_temperature, 1e-6) * target_scale.squeeze(-1))
        ) - 1.0
        confidence = torch.sigmoid(confidence_logits)
        active = torch.maximum(pos_candidate_mask, neg_candidate_mask)
        weighted_confidence = confidence * active

        cross_entropy = -(target * pair_weights.clamp_min(1e-8).log()).sum(dim=-1)
        rel_loss = weighted_confidence.mul(cross_entropy).sum() / weighted_confidence.sum().clamp_min(1.0)

        active_rows = active > 0
        if active_rows.any():
            mean_gate = pair_weights[active_rows].mean(dim=0)
            balance_loss = (mean_gate - (1.0 / 3.0)).square().mean()
        else:
            balance_loss = pos_base.new_tensor(0.0)
        routing_loss = self.c2r_loss_weight * rel_loss + self.c2r_balance_weight * balance_loss
        return pos_routed, neg_routed, routing_loss

    def _route_candidate_scores(self, users, candidate_idx, candidate_base, user_emb, item_emb_all):
        del users
        batch_size, candidate_k = candidate_idx.shape
        dim = self.dim_latent
        user_blocks = user_emb.reshape(batch_size, 3, dim)
        candidate_blocks = item_emb_all[candidate_idx].reshape(batch_size, candidate_k, 3, dim)

        user_weights = self._gate_weights(user_emb).unsqueeze(1)
        item_weights = self._gate_weights(
            candidate_blocks.reshape(-1, 3 * dim), item_side=True
        ).reshape(batch_size, candidate_k, 3)
        weights = self._product_weights(user_weights, item_weights)
        expanded_users = user_blocks.unsqueeze(1).expand(-1, candidate_k, -1, -1)
        routed = self._weighted_block_score(expanded_users, candidate_blocks, weights)
        strength = self.c2r_route_mix * self._routing_confidence(weights)
        return candidate_base + strength * (routed - candidate_base)

    def diagnostic_info(self):
        return (
            super(BRIDGE_C2R, self).diagnostic_info()
            + f", C2R(loss={self.c2r_loss_weight:.3g}, "
            + f"balance={self.c2r_balance_weight:.3g}, mix={self.c2r_route_mix:.3g})"
        )
