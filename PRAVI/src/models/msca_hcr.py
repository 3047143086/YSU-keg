"""Leave-one-out calibrated history residual for MSCA.

LCHR keeps every MSCA graph and representation objective unchanged.  It adds a
small score-space correction derived only from each user's observed training
history.  During BPR training, the current positive item is removed from its
history prototype so that the correction cannot win by copying the target.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.msca import MSCA


def _scalar(value, default):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


def build_history_mean_operator(interactions, device):
    """Return row-mean aggregation and observed train-history lengths."""
    observed = interactions.tocsr().astype(np.float32)
    observed.sum_duplicates()
    observed.data.fill(1.0)
    degree = np.asarray(observed.sum(axis=1)).ravel().astype(np.float32)
    coo = observed.tocoo()
    values = np.divide(
        coo.data,
        degree[coo.row],
        out=np.zeros_like(coo.data, dtype=np.float32),
        where=degree[coo.row] > 0,
    )
    indices = torch.as_tensor(
        np.stack([coo.row, coo.col]), dtype=torch.long, device=device
    )
    operator = torch.sparse_coo_tensor(
        indices,
        torch.as_tensor(values, dtype=torch.float32, device=device),
        observed.shape,
        device=device,
    ).coalesce()
    return operator, torch.as_tensor(degree, dtype=torch.float32, device=device)


def leave_one_out_prototype(
    history_mean, degree, positive_content, min_context=2
):
    """Remove a positive item before constructing its training prototype."""
    degree = degree.to(dtype=history_mean.dtype)
    remaining = (degree - 1.0).clamp_min(0.0)
    history_sum = history_mean * degree.unsqueeze(1)
    prototype = (history_sum - positive_content) / remaining.clamp_min(1.0).unsqueeze(1)
    coherence = torch.linalg.vector_norm(prototype, dim=1).clamp(0.0, 1.0)
    direction = F.normalize(prototype, p=2, dim=1, eps=1e-6)
    valid = remaining >= float(min_context)
    direction = torch.where(valid.unsqueeze(1), direction, torch.zeros_like(direction))
    coherence = torch.where(valid, coherence, torch.zeros_like(coherence))
    return direction, coherence, remaining


def history_coefficients(
    coherence,
    context_size,
    activity_prior=10.0,
    temperature=0.25,
    min_context=2,
):
    """Build bounded modality coefficients from detached history evidence."""
    evidence = coherence.detach().clamp(0.0, 1.0)
    temperature = max(float(temperature), 1e-4)
    modality_weight = torch.softmax(evidence / temperature, dim=1)

    count = context_size.to(dtype=evidence.dtype).clamp_min(0.0)
    prior = max(float(activity_prior), 1e-4)
    # The first term discounts tiny contexts; the second reserves more residual
    # capacity for sparse users whose collaborative estimate is less certain.
    support = torch.sqrt(count / (count + 1.0))
    sparse_need = torch.sqrt(prior / (count + prior))
    valid = (count >= float(min_context)).to(evidence.dtype)
    envelope = support * sparse_need * valid
    return modality_weight * evidence * envelope.unsqueeze(1)


class MSCA_HCR(MSCA):
    """MSCA with a target-safe, confidence-bounded history score residual."""

    def __init__(self, config, dataset):
        super(MSCA_HCR, self).__init__(config, dataset)
        self.hcr_enabled = bool(_scalar(config.get("hcr_enabled", True), True))
        self.hcr_max_scale = max(
            0.0, float(_scalar(config.get("hcr_max_scale", 0.10), 0.10))
        )
        self.hcr_activity_prior = max(
            1e-4,
            float(_scalar(config.get("hcr_activity_prior", 10.0), 10.0)),
        )
        self.hcr_coherence_temperature = max(
            1e-4,
            float(
                _scalar(config.get("hcr_coherence_temperature", 0.25), 0.25)
            ),
        )
        self.hcr_min_context = max(
            1, int(_scalar(config.get("hcr_min_context", 2), 2))
        )
        self._hcr_item_content = None

        have_both_modalities = (
            self.visual_feat is not None and self.textual_feat is not None
        )
        if not self.hcr_enabled or not have_both_modalities:
            self.hcr_enabled = False
            return

        history_operator, user_degree = build_history_mean_operator(
            self.interaction_matrix, self.device
        )
        self.register_buffer(
            "hcr_history_operator", history_operator, persistent=False
        )
        self.register_buffer("hcr_user_degree", user_degree, persistent=False)

        # tanh(0)=0 makes the initial scorer exactly equal to official MSCA.
        # Separate bounded scales let ranking supervision accept, reject, or
        # reverse each modality without perturbing any graph topology.
        self.hcr_raw_scale = nn.Parameter(torch.zeros(2, device=self.device))
        logging.getLogger().info(
            "LCHR enabled: max_scale=%.4f activity_prior=%.2f "
            "coherence_temperature=%.3f min_context=%d",
            self.hcr_max_scale,
            self.hcr_activity_prior,
            self.hcr_coherence_temperature,
            self.hcr_min_context,
        )

    def _refine_modality_content(self, image_content, text_content):
        if (
            not self.hcr_enabled
            or image_content is None
            or text_content is None
        ):
            return image_content, text_content

        # Catalog centering removes the global content direction before cosine
        # matching.  MSCA still receives its original, uncentered streams.
        image_centered = image_content - image_content.mean(dim=0, keepdim=True)
        text_centered = text_content - text_content.mean(dim=0, keepdim=True)
        self._hcr_item_content = (
            F.normalize(image_centered, p=2, dim=1, eps=1e-6),
            F.normalize(text_centered, p=2, dim=1, eps=1e-6),
        )
        return image_content, text_content

    def _bounded_scales(self):
        return self.hcr_max_scale * torch.tanh(self.hcr_raw_scale)

    def post_epoch_processing(self):
        scales = self._bounded_scales().detach().cpu().tolist()
        return "LCHR bounded scales: image={:.6f} text={:.6f}".format(*scales)

    def _history_means(self):
        if self._hcr_item_content is None:
            return None
        return tuple(
            torch.sparse.mm(self.hcr_history_operator, content)
            for content in self._hcr_item_content
        )

    def _coefficients(self, coherence, context_size):
        return history_coefficients(
            coherence,
            context_size,
            activity_prior=self.hcr_activity_prior,
            temperature=self.hcr_coherence_temperature,
            min_context=self.hcr_min_context,
        )

    def _refine_pairwise_scores(
        self, pos_scores, neg_scores, users, pos_items, neg_items
    ):
        if not self.hcr_enabled or self._hcr_item_content is None:
            return pos_scores, neg_scores

        user_ids = users.long().view(-1)
        pos_ids = pos_items.long().view(-1)
        neg_ids = neg_items.long().view(-1)
        degree = self.hcr_user_degree[user_ids]
        history_means = self._history_means()
        scales = self._bounded_scales()

        pos_directions = []
        pos_coherences = []
        neg_directions = []
        neg_coherences = []
        for history_mean, item_content in zip(
            history_means, self._hcr_item_content
        ):
            batch_mean = history_mean[user_ids]
            pos_direction, pos_coherence, remaining = leave_one_out_prototype(
                batch_mean,
                degree,
                item_content[pos_ids],
                min_context=self.hcr_min_context,
            )
            neg_direction = F.normalize(batch_mean, p=2, dim=1, eps=1e-6)
            neg_coherence = torch.linalg.vector_norm(
                batch_mean, dim=1
            ).clamp(0.0, 1.0)
            pos_directions.append(pos_direction)
            pos_coherences.append(pos_coherence)
            neg_directions.append(neg_direction)
            neg_coherences.append(neg_coherence)

        pos_coeff = self._coefficients(
            torch.stack(pos_coherences, dim=1), remaining
        )
        neg_coeff = self._coefficients(
            torch.stack(neg_coherences, dim=1), degree
        )
        for modality_index, item_content in enumerate(self._hcr_item_content):
            pos_similarity = torch.sum(
                pos_directions[modality_index] * item_content[pos_ids], dim=1
            )
            neg_similarity = torch.sum(
                neg_directions[modality_index] * item_content[neg_ids], dim=1
            )
            pos_scores = pos_scores + (
                scales[modality_index]
                * pos_coeff[:, modality_index]
                * pos_similarity
            )
            neg_scores = neg_scores + (
                scales[modality_index]
                * neg_coeff[:, modality_index]
                * neg_similarity
            )
        return pos_scores, neg_scores

    def _refine_full_sort_scores(
        self, scores, users, user_embeds, item_embeds
    ):
        if not self.hcr_enabled or self._hcr_item_content is None:
            return scores

        user_ids = users.long().view(-1)
        degree = self.hcr_user_degree[user_ids]
        history_means = self._history_means()
        directions = [
            F.normalize(history_mean[user_ids], p=2, dim=1, eps=1e-6)
            for history_mean in history_means
        ]
        coherence = torch.stack(
            [
                torch.linalg.vector_norm(history_mean[user_ids], dim=1).clamp(
                    0.0, 1.0
                )
                for history_mean in history_means
            ],
            dim=1,
        )
        coefficients = self._coefficients(coherence, degree)
        scales = self._bounded_scales()
        for modality_index, item_content in enumerate(self._hcr_item_content):
            weighted_history = directions[modality_index] * (
                scales[modality_index]
                * coefficients[:, modality_index]
            ).unsqueeze(1)
            scores = torch.addmm(
                scores, weighted_history, item_content.transpose(0, 1)
            )
        return scores
