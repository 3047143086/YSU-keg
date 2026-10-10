"""MSCA with disagreement-conditioned conjunctive ranking loss.

DCRL adds a score-level constraint that is independent of graph construction:
the positive item must beat the sampled negative in both modality views.  The
extra constraint is weighted by the detached disagreement between the two
modality margins, so already-consistent pairs receive little additional force.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from models.msca import MSCA


def _scalar(value, default):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


class MSCA_DCRL(MSCA):
    """Official MSCA plus a bounded, disagreement-focused ranking objective."""

    def __init__(self, config, dataset):
        super(MSCA_DCRL, self).__init__(config, dataset)
        self.dcrl_enabled = bool(_scalar(config.get("dcrl_enabled", True), True))
        self.dcrl_weight = max(
            0.0, float(_scalar(config.get("dcrl_weight", 0.02), 0.02))
        )
        self.dcrl_margin = max(
            0.0, float(_scalar(config.get("dcrl_margin", 0.10), 0.10))
        )
        self.dcrl_temperature = max(
            1e-4,
            float(_scalar(config.get("dcrl_temperature", 0.20), 0.20)),
        )
        self.dcrl_min_weight = max(
            0.0,
            float(_scalar(config.get("dcrl_min_weight", 0.10), 0.10)),
        )
        self._dcrl_forward_cache = None
        self._dcrl_last_loss = None
        have_both_modalities = (
            self.visual_feat is not None and self.textual_feat is not None
        )
        if not have_both_modalities:
            self.dcrl_enabled = False
        logging.getLogger().info(
            "DCRL %s: weight=%.6f margin=%.4f temperature=%.4f min_weight=%.4f",
            "enabled" if self.dcrl_enabled else "disabled",
            self.dcrl_weight,
            self.dcrl_margin,
            self.dcrl_temperature,
            self.dcrl_min_weight,
        )

    def forward(self, test=True):
        result = super(MSCA_DCRL, self).forward(test=test)
        if not test and self.dcrl_enabled:
            # Keep graph-connected tensors for the auxiliary loss.  The cache
            # is overwritten on every training batch and is never used at eval.
            self._dcrl_forward_cache = result
        return result

    def _compute_dcrl(self, users, positive_items, negative_items):
        if (
            not self.dcrl_enabled
            or self.dcrl_weight <= 0.0
            or self._dcrl_forward_cache is None
        ):
            return None

        final_user, _, _, _, image_view, text_view = self._dcrl_forward_cache
        image_items = image_view[self.n_users :]
        text_items = text_view[self.n_users :]
        user_vectors = F.normalize(final_user[users.long()], p=2, dim=1, eps=1e-6)
        image_items = F.normalize(image_items, p=2, dim=1, eps=1e-6)
        text_items = F.normalize(text_items, p=2, dim=1, eps=1e-6)

        positive_items = positive_items.long()
        negative_items = negative_items.long()
        visual_margin = (
            torch.sum(user_vectors * image_items[positive_items], dim=1)
            - torch.sum(user_vectors * image_items[negative_items], dim=1)
        )
        text_margin = (
            torch.sum(user_vectors * text_items[positive_items], dim=1)
            - torch.sum(user_vectors * text_items[negative_items], dim=1)
        )

        # The minimum margin is a conjunctive criterion: both views must rank
        # the positive above the negative.  Detaching the disagreement keeps
        # the weighting from collapsing by manipulating its own gate.
        conjunctive_margin = torch.minimum(visual_margin, text_margin)
        disagreement = (visual_margin - text_margin).abs().detach()
        scale = disagreement / disagreement.mean().clamp_min(1e-6)
        scale = scale.clamp(0.0, 2.0)
        weights = self.dcrl_min_weight + (1.0 - self.dcrl_min_weight) * scale
        weights = weights.clamp_min(0.0)
        loss = (
            weights
            * F.softplus((self.dcrl_margin - conjunctive_margin) / self.dcrl_temperature)
        ).mean()
        return loss

    def _refine_pairwise_scores(
        self, pos_scores, neg_scores, users, pos_items, neg_items
    ):
        if self.dcrl_enabled and self.dcrl_weight > 0.0:
            self._dcrl_last_loss = self._compute_dcrl(
                users, pos_items, neg_items
            )
        return pos_scores, neg_scores

    def calculate_loss(self, interaction):
        self._dcrl_last_loss = None
        base_loss = super(MSCA_DCRL, self).calculate_loss(interaction)
        if self._dcrl_last_loss is None:
            return base_loss
        return base_loss + self.dcrl_weight * self._dcrl_last_loss

    def post_epoch_processing(self):
        if self._dcrl_last_loss is None:
            return None
        return "DCRL auxiliary loss (last batch): {:.6f}".format(
            float(self._dcrl_last_loss.detach())
        )

