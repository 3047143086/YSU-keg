"""MSCA with a training-only reconstruction of VI-MMRec."""

from __future__ import annotations

import hashlib
import json
import logging
import os

import scipy.sparse as sp
import torch

from models.msca import MSCA
from utils.graph_cache import GraphCacheManager
from utils.pravi import sparse_fingerprint, torch_graph_to_scipy
from utils.vi_mmrec import build_feature_knn, build_vi_interactions


class MSCA_VI(MSCA):
    def __init__(self, config, dataset):
        super(MSCA_VI, self).__init__(config, dataset)
        observed = self.interaction_matrix.tocsr().astype("float32")
        source_graphs = []
        raw_features = []
        if hasattr(self, "image_original_adj"):
            source_graphs.append(torch_graph_to_scipy(self.image_original_adj))
            raw_features.append(self.image_embedding.weight.detach())
        if hasattr(self, "text_original_adj"):
            source_graphs.append(torch_graph_to_scipy(self.text_original_adj))
            raw_features.append(self.text_embedding.weight.detach())

        parameters = {
            "version": 1,
            "strategy": config["vi_strategy"],
            "lambda": float(config["vi_lambda"]),
            "topk": int(config["vi_topk"]),
            "train_fingerprint": sparse_fingerprint(observed),
            "source_graph_fingerprints": [
                sparse_fingerprint(graph) for graph in source_graphs
            ],
        }
        cache_key = hashlib.sha1(
            json.dumps(parameters, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        cache_manager = GraphCacheManager(config["data_path"], config["dataset"])
        cache_dir = cache_manager.get_model_cache_dir("MSCA_VI")
        augmented_path = os.path.join(cache_dir, f"augmented_{cache_key}.npz")
        virtual_path = os.path.join(cache_dir, f"virtual_{cache_key}.npz")
        stats_path = os.path.join(cache_dir, f"stats_{cache_key}.json")

        if all(os.path.exists(path) for path in (augmented_path, virtual_path, stats_path)):
            augmented = sp.load_npz(augmented_path).tocsr()
            virtual = sp.load_npz(virtual_path).tocsr()
            with open(stats_path, encoding="utf-8") as handle:
                stats = json.load(handle)
            stats["cache_hit"] = True
        else:
            if parameters["strategy"] == "overlay":
                item_graphs = [
                    build_feature_knn([feature], parameters["topk"])
                    for feature in raw_features
                ]
            else:
                item_graphs = [build_feature_knn(raw_features, parameters["topk"])]
            augmented, virtual, stats = build_vi_interactions(
                observed,
                item_graphs,
                strategy=parameters["strategy"],
                lambda_weight=parameters["lambda"],
                topk=parameters["topk"],
            )
            stats["cache_hit"] = False
            stats["parameters"] = parameters
            sp.save_npz(augmented_path, augmented)
            sp.save_npz(virtual_path, virtual)
            with open(stats_path, "w", encoding="utf-8") as handle:
                json.dump(stats, handle, ensure_ascii=False, indent=2)

        # VI-MMRec replaces R-bar throughout the host model.
        self.interaction_matrix = augmented.tocoo()
        self.norm_R, self.norm_adj = self.get_norm_adj_mat()
        self.collab_norm_adj = self.norm_adj
        self.struct_original_adj = self.get_struct_adj_mat()
        self.virtual_interaction_matrix = virtual
        virtual_coo = virtual.tocoo()
        keys = virtual_coo.row.astype("int64") * self.n_items + virtual_coo.col
        self.virtual_keys = torch.as_tensor(
            sorted(keys.tolist()), dtype=torch.long, device=self.device
        )
        logging.getLogger().info("VI-MMRec graph: %s", json.dumps(stats, sort_keys=True))

    def calculate_loss(self, interaction):
        if self.virtual_keys.numel() == 0:
            return super(MSCA_VI, self).calculate_loss(interaction)
        users, neg_items = interaction[0], interaction[2]
        keys = users.long() * self.n_items + neg_items.long()
        positions = torch.searchsorted(self.virtual_keys, keys)
        bounded = positions < self.virtual_keys.numel()
        matches = torch.zeros_like(bounded)
        matches[bounded] = self.virtual_keys[positions[bounded]] == keys[bounded]
        if not torch.any(matches):
            return super(MSCA_VI, self).calculate_loss(interaction)
        keep = ~matches
        if not torch.any(keep):
            return self.user_embedding.weight.sum() * 0.0
        return super(MSCA_VI, self).calculate_loss([field[keep] for field in interaction])
