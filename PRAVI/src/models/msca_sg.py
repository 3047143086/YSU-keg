"""MSCA adapter for the official SG-URInit user semantic profiles."""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

from models.msca import MSCA
from utils.sg_urinit import build_sg_user_profile, profile_cache_signature


def _scalar(value, default):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


class MSCA_SG(MSCA):
    """Official MSCA plus an SG-URInit-style training-only semantic BPR."""

    def __init__(self, config, dataset):
        super(MSCA_SG, self).__init__(config, dataset)
        self.sg_num_clusters = int(_scalar(config.get("sg_num_clusters", 4), 4))
        self.sg_cluster_weight = float(
            _scalar(config.get("sg_cluster_weight", 0.1), 0.1)
        )
        self.sg_loss_weight = float(
            _scalar(config.get("sg_loss_weight", 0.001), 0.001)
        )
        self.sg_random_state = int(
            _scalar(config.get("sg_random_state", 0), 0)
        )
        if self.sg_loss_weight < 0.0:
            raise ValueError("sg_loss_weight must be nonnegative")

        dataset_path = os.path.abspath(
            os.path.join(config["data_path"], config["dataset"])
        )
        cache_dir = os.path.join(dataset_path, "sg_urinit_cache")
        os.makedirs(cache_dir, exist_ok=True)

        if self.visual_feat is not None:
            image_profile = self._load_or_build_profile(
                "image", self.visual_feat, cache_dir
            )
            self.register_buffer(
                "sg_image_user_profile", image_profile, persistent=False
            )
        else:
            self.sg_image_user_profile = None
        if self.textual_feat is not None:
            text_profile = self._load_or_build_profile(
                "text", self.textual_feat, cache_dir
            )
            self.register_buffer(
                "sg_text_user_profile", text_profile, persistent=False
            )
        else:
            self.sg_text_user_profile = None

        self._last_sg_loss = None
        logging.getLogger().info(
            "MSCA + SG-URInit adapter: K=%d cluster_weight=%.4f "
            "loss_weight=%.6f random_state=%d",
            self.sg_num_clusters,
            self.sg_cluster_weight,
            self.sg_loss_weight,
            self.sg_random_state,
        )

    def _load_or_build_profile(self, modality, feature_tensor, cache_dir):
        features = feature_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        signature = profile_cache_signature(self.interaction_matrix, features)
        weight_token = ("{:.6g}".format(self.sg_cluster_weight)).replace(".", "p")
        cache_name = (
            "{}_k{}_w{}_r{}_{}.pt".format(
                modality,
                self.sg_num_clusters,
                weight_token,
                self.sg_random_state,
                signature,
            )
        )
        cache_path = os.path.join(cache_dir, cache_name)
        if os.path.exists(cache_path):
            profile = torch.load(cache_path, map_location="cpu")
            if tuple(profile.shape) == (self.n_users, features.shape[1]):
                logging.getLogger().info(
                    "Loaded SG-URInit %s profile from %s", modality, cache_path
                )
                return profile.float()

        profile = build_sg_user_profile(
            self.interaction_matrix,
            features,
            num_clusters=self.sg_num_clusters,
            cluster_weight=self.sg_cluster_weight,
            random_state=self.sg_random_state,
        )
        profile_tensor = torch.from_numpy(profile)
        temporary_path = cache_path + ".tmp"
        torch.save(profile_tensor, temporary_path)
        os.replace(temporary_path, cache_path)
        logging.getLogger().info(
            "Built SG-URInit %s profile and cached it at %s",
            modality,
            cache_path,
        )
        return profile_tensor

    @staticmethod
    def semantic_bpr_loss(user_features, positive_features, negative_features):
        positive_scores = torch.sum(user_features * positive_features, dim=1)
        negative_scores = torch.sum(user_features * negative_features, dim=1)
        return -torch.mean(torch.nn.functional.logsigmoid(
            positive_scores - negative_scores
        ))

    def _modality_semantic_loss(
        self, profile, embedding, transform, users, positive_items, negative_items
    ):
        user_features = transform(profile[users])
        positive_features = transform(embedding(positive_items))
        negative_features = transform(embedding(negative_items))
        return self.semantic_bpr_loss(
            user_features, positive_features, negative_features
        )

    def calculate_loss(self, interaction):
        base_loss = super(MSCA_SG, self).calculate_loss(interaction)
        if self.sg_loss_weight == 0.0:
            return base_loss

        users, positive_items, negative_items = interaction[:3]
        semantic_losses = []
        if self.sg_image_user_profile is not None:
            semantic_losses.append(
                self._modality_semantic_loss(
                    self.sg_image_user_profile,
                    self.image_embedding,
                    self.image_trs,
                    users,
                    positive_items,
                    negative_items,
                )
            )
        if self.sg_text_user_profile is not None:
            semantic_losses.append(
                self._modality_semantic_loss(
                    self.sg_text_user_profile,
                    self.text_embedding,
                    self.text_trs,
                    users,
                    positive_items,
                    negative_items,
                )
            )
        if not semantic_losses:
            return base_loss
        semantic_loss = torch.stack(semantic_losses).sum()
        self._last_sg_loss = float(semantic_loss.detach())
        return base_loss + self.sg_loss_weight * semantic_loss

    def post_epoch_processing(self):
        if self._last_sg_loss is None:
            return None
        return "SG-URInit semantic BPR (last batch): {:.6f}".format(
            self._last_sg_loss
        )
