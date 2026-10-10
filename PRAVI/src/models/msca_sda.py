"""Structure-aware, modality-disentangled adaptation for MSCA.

This module is deliberately self-contained and uses the item content already
provided by the benchmark.  It does not require downloading a foundation model
or using validation/test interactions.  The fixed content views provide a
soft structural teacher, while the two low-rank paths are learned separately
for vision and text.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.msca import MSCA


def _scalar(value, default):
    """Read a scalar from either a runtime value or a one-element YAML list."""
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


class MSCA_SDA(MSCA):
    """MSCA with structural cross-modal adaptation and private low-rank paths.

    The name follows the recent structural/disentangled adaptation direction,
    but this implementation is adapted to the lightweight MSCA setting: the
    existing image/text features act as frozen teachers and only small
    recommendation-side adapters are trained.
    """

    def __init__(self, config, dataset):
        super(MSCA_SDA, self).__init__(config, dataset)

        self.sda_enabled = bool(_scalar(config.get("sda_enabled", True), True))
        self.sda_adapter_rank = int(
            _scalar(config.get("sda_adapter_rank", 8), 8)
        )
        self.sda_adapter_rank = max(1, min(self.sda_adapter_rank, self.embedding_dim))
        self.sda_bridge_mix = float(_scalar(config.get("sda_bridge_mix", 0.05), 0.05))
        self.sda_bridge_mix = max(0.0, min(1.0, self.sda_bridge_mix))
        self.sda_adapter_scale = float(
            _scalar(config.get("sda_adapter_scale", 0.10), 0.10)
        )
        self.sda_adapter_scale = max(0.0, min(2.0, self.sda_adapter_scale))
        self.sda_gate_temperature = float(
            _scalar(config.get("sda_gate_temperature", 4.0), 4.0)
        )
        self.sda_gate_center = float(
            _scalar(config.get("sda_gate_center", 0.0), 0.0)
        )
        self.sda_cmsa_weight = float(
            _scalar(config.get("sda_cmsa_weight", 0.01), 0.01)
        )
        self.sda_cmsa_weight = max(0.0, self.sda_cmsa_weight)
        self.sda_cmsa_batch = int(_scalar(config.get("sda_cmsa_batch", 256), 256))
        self.sda_cmsa_batch = max(2, self.sda_cmsa_batch)
        self.sda_tau = float(_scalar(config.get("sda_tau", 0.20), 0.20))
        self.sda_tau = max(self.sda_tau, 1e-3)
        self._sda_last_image_content = None
        self._sda_last_text_content = None

        have_both_modalities = (
            self.visual_feat is not None and self.textual_feat is not None
        )
        if not self.sda_enabled or not have_both_modalities:
            self.sda_enabled = False
            return

        # Separate low-rank paths prevent one modality's gradients from
        # overwriting the other modality's content adapter.
        rank = self.sda_adapter_rank
        self.sda_image_down = nn.Linear(self.embedding_dim, rank, bias=False)
        self.sda_image_up = nn.Linear(rank, self.embedding_dim, bias=False)
        self.sda_text_down = nn.Linear(self.embedding_dim, rank, bias=False)
        self.sda_text_up = nn.Linear(rank, self.embedding_dim, bias=False)
        for layer in (
            self.sda_image_down,
            self.sda_text_down,
        ):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        # A zero residual gives an exact MSCA initialization and lets the
        # structural objective grow the adapters only when useful.
        nn.init.zeros_(self.sda_image_up.weight)
        nn.init.zeros_(self.sda_text_up.weight)

        # Normalize fixed features once.  This removes the image/text scale
        # mismatch while retaining their intra-modal neighborhood geometry.
        image_teacher = self._teacher_normalize(self.visual_feat.detach())
        text_teacher = self._teacher_normalize(self.textual_feat.detach())
        self.register_buffer(
            "sda_image_teacher", image_teacher, persistent=False
        )
        self.register_buffer("sda_text_teacher", text_teacher, persistent=False)

        logging.getLogger().info(
            "SDA enabled: rank=%d bridge_mix=%.4f adapter_scale=%.4f "
            "cmsa_weight=%.4f cmsa_batch=%d",
            rank,
            self.sda_bridge_mix,
            self.sda_adapter_scale,
            self.sda_cmsa_weight,
            self.sda_cmsa_batch,
        )

    @staticmethod
    def _teacher_normalize(features):
        features = features.float()
        features = features - features.mean(dim=0, keepdim=True)
        return F.normalize(features, p=2, dim=1, eps=1e-6)

    def _refine_modality_content(self, image_content, text_content):
        if (
            not self.sda_enabled
            or image_content is None
            or text_content is None
        ):
            return image_content, text_content

        image_unit = F.normalize(image_content, p=2, dim=1, eps=1e-6)
        text_unit = F.normalize(text_content, p=2, dim=1, eps=1e-6)
        agreement = (image_unit * text_unit).sum(dim=1)
        # The gate is detached from the routing decision.  This keeps the
        # confidence mechanism from learning a degenerate all-open/all-closed
        # solution while the adapters still receive normal gradients.
        common_gate = torch.sigmoid(
            (agreement - self.sda_gate_center) * self.sda_gate_temperature
        ).detach().unsqueeze(1)

        image_delta = self.sda_image_up(
            torch.tanh(self.sda_image_down(image_content))
        )
        text_delta = self.sda_text_up(
            torch.tanh(self.sda_text_down(text_content))
        )
        shared = 0.5 * (image_content + text_content)
        image_content = (
            image_content
            + self.sda_bridge_mix * common_gate * (shared - image_content)
            + self.sda_adapter_scale * image_delta
        )
        text_content = (
            text_content
            + self.sda_bridge_mix * common_gate * (shared - text_content)
            + self.sda_adapter_scale * text_delta
        )
        self._sda_last_image_content = image_content
        self._sda_last_text_content = text_content
        return image_content, text_content

    def _get_feature_learning_loss(self, pos_items):
        """Match cross-modal neighbourhood distributions to a soft teacher."""
        if (
            not self.sda_enabled
            or self.sda_cmsa_weight <= 0.0
            or self._sda_last_image_content is None
            or self._sda_last_text_content is None
        ):
            return self.item_id_embedding.weight.new_zeros(())

        item_ids = torch.unique(pos_items.detach().long())
        if item_ids.numel() > self.sda_cmsa_batch:
            item_ids = item_ids[: self.sda_cmsa_batch]
        if item_ids.numel() < 2:
            return self.item_id_embedding.weight.new_zeros(())

        image_teacher = self.sda_image_teacher[item_ids]
        text_teacher = self.sda_text_teacher[item_ids]
        teacher_logits = 0.5 * (
            torch.matmul(image_teacher, image_teacher.transpose(0, 1))
            + torch.matmul(text_teacher, text_teacher.transpose(0, 1))
        ) / self.sda_tau
        teacher = torch.softmax(teacher_logits.detach(), dim=1)

        image = F.normalize(
            self._sda_last_image_content[item_ids], p=2, dim=1, eps=1e-6
        )
        text = F.normalize(
            self._sda_last_text_content[item_ids], p=2, dim=1, eps=1e-6
        )
        cross_logits = torch.matmul(image, text.transpose(0, 1)) / self.sda_tau
        row_loss = -(teacher * F.log_softmax(cross_logits, dim=1)).sum(dim=1).mean()
        col_loss = -(
            teacher.transpose(0, 1)
            * F.log_softmax(cross_logits.transpose(0, 1), dim=1)
        ).sum(dim=1).mean()
        return self.sda_cmsa_weight * 0.5 * (row_loss + col_loss)

