"""User-equalized, collision-free contrastive alignment for MSCA.

MSCA builds its auxiliary InfoNCE batches from interaction records. Repeated
users and items therefore occupy several rows of the same contrastive matrix,
and high-degree users contribute to more minibatches even though full-ranking
metrics average users uniformly. UECA changes only these auxiliary alignment
terms; the scorer, BPR loss, graphs, features, and regularization stay intact.
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


def inverse_inclusion_weights(degrees, population_size, batch_size):
    """Return inverse probabilities of appearing in a shuffled minibatch.

    A minibatch is a simple random sample without replacement from the training
    interactions. For an entity with ``degree`` records, its inclusion
    probability is ``1 - C(N-degree, B) / C(N, B)``. Inverse-probability anchor
    weighting makes the expected user-side objective uniform over users.
    """
    population_size = int(population_size)
    batch_size = min(int(batch_size), population_size)
    if population_size <= 0 or batch_size <= 0:
        raise ValueError("population_size and batch_size must be positive")

    source = torch.as_tensor(degrees)
    work = source.to(dtype=torch.float64)
    if torch.any(work < 0) or torch.any(work > population_size):
        raise ValueError("degrees must be between zero and population_size")

    probability = torch.zeros_like(work)
    observed = work > 0
    certain = observed & (work > population_size - batch_size)
    probability[certain] = 1.0

    avoidable = observed & ~certain
    if torch.any(avoidable):
        degree = work[avoidable]
        n = work.new_tensor(float(population_size))
        b = work.new_tensor(float(batch_size))
        log_miss = (
            torch.lgamma(n - degree + 1.0)
            + torch.lgamma(n - b + 1.0)
            - torch.lgamma(n - degree - b + 1.0)
            - torch.lgamma(n + 1.0)
        )
        # expm1 remains accurate when a rare entity has a small probability.
        probability[avoidable] = -torch.expm1(log_miss.clamp(max=0.0))

    weights = torch.zeros_like(work)
    weights[observed] = probability[observed].clamp_min(1e-12).reciprocal()
    output_dtype = source.dtype if source.is_floating_point() else torch.float32
    return weights.to(dtype=output_dtype)


class MSCA_UECA(MSCA):
    """MSCA with user-equalized, entity-collision-free auxiliary InfoNCE."""

    def __init__(self, config, dataset):
        super(MSCA_UECA, self).__init__(config, dataset)
        self.ueca_enabled = bool(_scalar(config.get("ueca_enabled", True), True))

        interaction_coo = self.interaction_matrix.tocoo(copy=False)
        population_size = int(interaction_coo.nnz)
        user_degree = np.bincount(
            interaction_coo.row, minlength=self.n_users
        ).astype(np.float64)
        full_batch_size = min(int(self.batch_size), population_size)
        user_weight = inverse_inclusion_weights(
            torch.from_numpy(user_degree), population_size, full_batch_size
        ).float()
        self.register_buffer(
            "ueca_user_inverse_inclusion", user_weight, persistent=False
        )
        self.ueca_population_size = population_size
        self.ueca_inclusion_batch_size = full_batch_size
        self._reset_ueca_stats()

        if self.ueca_enabled:
            observed = user_weight > 0
            logging.getLogger().info(
                "UECA enabled: collision-free user/item CL; "
                "user inverse-inclusion range=[%.4f, %.4f], N=%d, B=%d",
                float(user_weight[observed].min()) if torch.any(observed) else 0.0,
                float(user_weight[observed].max()) if torch.any(observed) else 0.0,
                population_size,
                full_batch_size,
            )

    def _reset_ueca_stats(self):
        self._ueca_batches = 0
        self._ueca_user_rows = 0
        self._ueca_unique_users = 0
        self._ueca_item_rows = 0
        self._ueca_unique_items = 0
        self._ueca_weight_min = float("inf")
        self._ueca_weight_max = 0.0

    def pre_epoch_processing(self):
        self._reset_ueca_stats()

    def post_epoch_processing(self):
        if not self.ueca_enabled or self._ueca_batches == 0:
            return None
        user_duplicate_rate = 1.0 - (
            self._ueca_unique_users / max(self._ueca_user_rows, 1)
        )
        item_duplicate_rate = 1.0 - (
            self._ueca_unique_items / max(self._ueca_item_rows, 1)
        )
        return (
            "UECA diagnostics: user_duplicate_rate={:.4f} "
            "item_duplicate_rate={:.4f} user_weight_range=[{:.4f}, {:.4f}]"
        ).format(
            user_duplicate_rate,
            item_duplicate_rate,
            self._ueca_weight_min,
            self._ueca_weight_max,
        )

    def _unique_cl_anchors(self, ids, is_user):
        unique_ids = torch.unique(ids.detach().long(), sorted=True)
        weights = None
        if is_user:
            weights = self.ueca_user_inverse_inclusion[unique_ids]
        return unique_ids, weights

    def _record_batch_stats(
        self, users, unique_users, items, unique_items, user_weights
    ):
        self._ueca_batches += 1
        self._ueca_user_rows += int(users.numel())
        self._ueca_unique_users += int(unique_users.numel())
        self._ueca_item_rows += int(items.numel())
        self._ueca_unique_items += int(unique_items.numel())
        if user_weights.numel() > 0:
            self._ueca_weight_min = min(
                self._ueca_weight_min, float(user_weights.min().detach())
            )
            self._ueca_weight_max = max(
                self._ueca_weight_max, float(user_weights.max().detach())
            )

    def calculate_loss(self, interaction):
        if not self.ueca_enabled:
            return super(MSCA_UECA, self).calculate_loss(interaction)

        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        (
            final_user_embeds,
            final_item_embeds,
            collab_embeds,
            struct_embeds,
            image_embeds,
            text_embeds,
        ) = self.forward(test=False)

        batch_user_embeds = final_user_embeds[users]
        pos_scores = torch.sum(
            batch_user_embeds * final_item_embeds[pos_items], dim=1
        )
        neg_scores = torch.sum(
            batch_user_embeds * final_item_embeds[neg_items], dim=1
        )
        pos_scores, neg_scores = self._refine_pairwise_scores(
            pos_scores, neg_scores, users, pos_items, neg_items
        )
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        reg_loss = self.cal_reg_loss()

        collab_user, collab_item = torch.split(
            collab_embeds, [self.n_users, self.n_items], dim=0
        )
        struct_user, struct_item = torch.split(
            struct_embeds, [self.n_users, self.n_items], dim=0
        )
        image_user, image_item = torch.split(
            image_embeds, [self.n_users, self.n_items], dim=0
        )
        text_user, text_item = torch.split(
            text_embeds, [self.n_users, self.n_items], dim=0
        )

        unique_users, user_weights = self._unique_cl_anchors(users, is_user=True)
        unique_items, _ = self._unique_cl_anchors(pos_items, is_user=False)
        self._record_batch_stats(
            users,
            unique_users,
            pos_items,
            unique_items,
            user_weights,
        )

        # Only the auxiliary alignment sample is changed. All six terms retain
        # the official MSCA views, temperature, and global cl_weight.
        modality_user_loss = self.cal_cl_loss(
            collab_user[unique_users],
            image_user[unique_users],
            self.tau,
            sample_weight=user_weights,
        ) + self.cal_cl_loss(
            collab_user[unique_users],
            text_user[unique_users],
            self.tau,
            sample_weight=user_weights,
        )
        modality_item_loss = self.cal_cl_loss(
            collab_item[unique_items], image_item[unique_items], self.tau
        ) + self.cal_cl_loss(
            collab_item[unique_items], text_item[unique_items], self.tau
        )
        structural_user_loss = self.cal_cl_loss(
            collab_user[unique_users],
            struct_user[unique_users],
            self.tau,
            sample_weight=user_weights,
        )
        structural_item_loss = self.cal_cl_loss(
            collab_item[unique_items], struct_item[unique_items], self.tau
        )

        feature_learning_loss = self._get_feature_learning_loss(pos_items)
        alignment_loss = (
            modality_user_loss
            + modality_item_loss
            + structural_user_loss
            + structural_item_loss
        )
        return (
            bpr_loss
            + self.cl_weight * alignment_loss
            + self.reg_weight * reg_loss
            + feature_learning_loss
        )
