"""MSCA enhanced with personalized reliability-aware virtual interactions."""

from __future__ import annotations

import hashlib
import json
import logging
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from models.msca import MSCA
from utils.graph_cache import GraphCacheManager
from utils.dvt import (
    build_degree_preserving_transport,
    build_evidence_cycle_transport,
    build_signed_transport_operator,
)
from utils.pravi import (
    build_pravi_interactions,
    build_pravi_transport_evidence,
    estimate_pravi_reliability,
    sparse_fingerprint,
    torch_graph_to_scipy,
)


def build_anchored_virtual_operator(observed, virtual, device):
    """Build a bounded virtual-only operator under an observed+virtual envelope."""
    observed = observed.tocsr().astype(np.float32)
    virtual = virtual.tocoo().astype(np.float32)
    virtual_csr = virtual.tocsr()
    user_degree = np.asarray(observed.sum(axis=1)).ravel() + np.asarray(
        virtual_csr.sum(axis=1)
    ).ravel()
    item_degree = np.asarray(observed.sum(axis=0)).ravel() + np.asarray(
        virtual_csr.sum(axis=0)
    ).ravel()
    if virtual.nnz:
        denominator = np.sqrt(
            user_degree[virtual.row] * item_degree[virtual.col]
        )
        values = np.divide(
            virtual.data,
            denominator,
            out=np.zeros_like(virtual.data, dtype=np.float32),
            where=denominator > 0,
        )
        keep = values > 0
        rows = np.concatenate(
            [virtual.row[keep], observed.shape[0] + virtual.col[keep]]
        )
        cols = np.concatenate(
            [observed.shape[0] + virtual.col[keep], virtual.row[keep]]
        )
        values = np.concatenate([values[keep], values[keep]])
    else:
        rows = np.empty(0, dtype=np.int64)
        cols = np.empty(0, dtype=np.int64)
        values = np.empty(0, dtype=np.float32)
    indices = torch.as_tensor(np.stack([rows, cols]), dtype=torch.long)
    weights = torch.as_tensor(values, dtype=torch.float32)
    size = observed.shape[0] + observed.shape[1]
    return torch.sparse_coo_tensor(
        indices, weights, (size, size), device=device
    ).coalesce()


class MSCA_PRAVI(MSCA):
    def __init__(self, config, dataset):
        # Build the official MSCA views first. Its structural view remains based
        # only on observed training interactions.
        super(MSCA_PRAVI, self).__init__(config, dataset)
        observed = self.interaction_matrix.tocsr().astype("float32")

        modality_graphs = []
        if hasattr(self, "image_original_adj"):
            modality_graphs.append(torch_graph_to_scipy(self.image_original_adj))
        if hasattr(self, "text_original_adj"):
            modality_graphs.append(torch_graph_to_scipy(self.text_original_adj))

        propagation_scope = config["pravi_propagation_scope"]
        transport_method = config["pravi_dvt_transport"]
        donor_aware_transport = (
            propagation_scope == "dvt"
            and transport_method in {"net_utility_cycle", "profile_drift_cycle"}
        )
        donor_cost_mode = (
            "profile_drift"
            if transport_method == "profile_drift_cycle"
            else "loo_retrieval"
        )
        parameters = {
            "version": 5,
            "topk": int(config["pravi_topk"]),
            "shrinkage": float(config["pravi_shrinkage"]),
            "reliability_mode": config["pravi_reliability_mode"],
            "probe_topl": int(config["pravi_probe_topl"]),
            "uncertainty_kappa": float(config["pravi_uncertainty_kappa"]),
            "mass_budget_ratio": float(config["pravi_mass_budget_ratio"]),
            "min_confidence": float(config["pravi_min_confidence"]),
            "min_modal_support": int(config["pravi_min_modal_support"]),
            "binary_weights": bool(config["pravi_binary_weights"]),
            "knn_k": int(config["knn_k"]),
            "is_sparse": bool(config["is_sparse"]),
            "evidence_mode": (
                f"receiver_lcb_donor_{donor_cost_mode}_ucb"
                if donor_aware_transport
                else "budgeted_receiver_lcb"
            ),
            "donor_cost_mode": donor_cost_mode,
            "min_donor_history": int(config["pravi_dvt_min_donor_history"]),
            "train_fingerprint": sparse_fingerprint(observed),
            "graph_fingerprints": [
                sparse_fingerprint(graph) for graph in modality_graphs
            ],
        }
        cache_key = hashlib.sha1(
            json.dumps(parameters, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        cache_manager = GraphCacheManager(config["data_path"], config["dataset"])
        cache_dir = cache_manager.get_model_cache_dir("MSCA_PRAVI")

        # Optional adaptive alignment uses only train-graph evidence.  It is
        # deliberately cached separately from virtual interactions so that the
        # identity path and all previous PRAVI transport experiments remain
        # byte-for-byte compatible.
        conditioning = str(config.get("pravi_reliability_conditioning", "none"))
        conditioning_strength = float(
            config.get("pravi_reliability_strength", 0.0)
        )
        valid_conditioning = {
            "none",
            "cl_weight",
            "fusion_route",
            "attention_prior",
            "both",
            "consensus_residual",
        }
        if conditioning not in valid_conditioning:
            raise ValueError(f"Unknown reliability conditioning: {conditioning}")
        if conditioning_strength < 0.0:
            raise ValueError("pravi_reliability_strength must be non-negative")

        reliability_log = {
            "enabled": False,
            "conditioning": conditioning,
            "strength": conditioning_strength,
        }
        if conditioning != "none" and conditioning_strength > 0.0:
            reliability_parameters = {
                "version": 1,
                "probe_topl": int(config["pravi_probe_topl"]),
                "shrinkage": float(config["pravi_shrinkage"]),
                "reliability_mode": config["pravi_reliability_mode"],
                "uncertainty_kappa": float(config["pravi_uncertainty_kappa"]),
                "train_fingerprint": sparse_fingerprint(observed),
                "graph_fingerprints": [
                    sparse_fingerprint(graph) for graph in modality_graphs
                ],
            }
            reliability_key = hashlib.sha1(
                json.dumps(reliability_parameters, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            reliability_path = os.path.join(
                cache_dir, f"reliability_{reliability_key}.npz"
            )
            reliability_stats_path = os.path.join(
                cache_dir, f"reliability_{reliability_key}.json"
            )
            reliability_cache_hit = os.path.exists(reliability_path) and os.path.exists(
                reliability_stats_path
            )
            if reliability_cache_hit:
                with np.load(reliability_path) as arrays:
                    reliability = arrays["mean"].astype(np.float32)
                    reliability_variance = arrays["variance"].astype(np.float32)
                with open(reliability_stats_path, encoding="utf-8") as handle:
                    reliability_stats = json.load(handle)
            else:
                reliability, reliability_variance, reliability_stats = (
                    estimate_pravi_reliability(
                        observed,
                        modality_graphs,
                        probe_topl=int(config["pravi_probe_topl"]),
                        shrinkage=float(config["pravi_shrinkage"]),
                        reliability_mode=config["pravi_reliability_mode"],
                    )
                )
                np.savez_compressed(
                    reliability_path,
                    mean=reliability,
                    variance=reliability_variance,
                )
                with open(reliability_stats_path, "w", encoding="utf-8") as handle:
                    json.dump(reliability_stats, handle, ensure_ascii=False, indent=2)
            expected_shape = (len(modality_graphs), self.n_users)
            if reliability.shape != expected_shape or reliability_variance.shape != expected_shape:
                raise RuntimeError(
                    "cached reliability shape does not match the current training graph"
                )
            self.register_buffer(
                "pravi_user_reliability",
                torch.as_tensor(reliability, dtype=torch.float32, device=self.device),
                persistent=False,
            )
            self.register_buffer(
                "pravi_user_reliability_variance",
                torch.as_tensor(
                    reliability_variance, dtype=torch.float32, device=self.device
                ),
                persistent=False,
            )
            # Use a robust posterior-median reference.  A mean-minus-standard-
            # deviation LCB collapses almost every sparse-user posterior to
            # zero on Baby; the median keeps the train-only signal while
            # precision-shrinking uncertain users toward a typical user.
            uncertainty = torch.sqrt(
                self.pravi_user_reliability_variance.clamp_min(0.0)
            )
            posterior = self.pravi_user_reliability.clamp_min(1e-6)
            reference = posterior.median(dim=1, keepdim=True).values.clamp_min(1e-6)
            precision = posterior / (posterior + uncertainty + 1e-6)
            stabilized = reference + precision * (posterior - reference)
            reliability_ratio = (stabilized / reference).clamp(0.5, 1.5)
            self.register_buffer(
                "pravi_user_reliability_ratio",
                reliability_ratio,
                persistent=False,
            )
            reliability_log = {
                **reliability_stats,
                "enabled": True,
                "cache_hit": reliability_cache_hit,
                "conditioning": conditioning,
                "strength": conditioning_strength,
                "effective_ratio_mean": [
                    float(value) for value in reliability_ratio.mean(dim=1).detach().cpu()
                ],
                "effective_ratio_std": [
                    float(value) for value in reliability_ratio.std(dim=1).detach().cpu()
                ],
            }
        self.pravi_reliability_conditioning = conditioning
        self.pravi_reliability_strength = conditioning_strength
        self.pravi_consensus_private_strength = float(
            config.get("pravi_consensus_private_strength", 0.25)
        )
        if self.pravi_consensus_private_strength < 0.0:
            raise ValueError("pravi_consensus_private_strength must be non-negative")
        self.pravi_reliability_log = reliability_log
        virtual_path = os.path.join(cache_dir, f"virtual_{cache_key}.npz")
        donor_path = os.path.join(cache_dir, f"donor_{cache_key}.npz")
        stats_path = os.path.join(cache_dir, f"virtual_{cache_key}.json")

        cache_complete = os.path.exists(virtual_path) and os.path.exists(stats_path)
        if donor_aware_transport:
            cache_complete = cache_complete and os.path.exists(donor_path)
        donor_upper_bounds = None
        if cache_complete:
            virtual = sp.load_npz(virtual_path).tocsr()
            if donor_aware_transport:
                donor_upper_bounds = sp.load_npz(donor_path).tocsr()
            with open(stats_path, encoding="utf-8") as handle:
                stats = json.load(handle)
            stats["cache_hit"] = True
        else:
            if donor_aware_transport:
                virtual, donor_upper_bounds, stats = (
                    build_pravi_transport_evidence(
                        observed,
                        modality_graphs,
                        topk=parameters["topk"],
                        shrinkage=parameters["shrinkage"],
                        reliability_mode=parameters["reliability_mode"],
                        probe_topl=parameters["probe_topl"],
                        uncertainty_kappa=parameters["uncertainty_kappa"],
                        min_confidence=parameters["min_confidence"],
                        min_modal_support=parameters["min_modal_support"],
                        min_donor_history=parameters["min_donor_history"],
                        donor_cost_mode=parameters["donor_cost_mode"],
                    )
                )
            else:
                virtual, stats = build_pravi_interactions(
                    observed,
                    modality_graphs,
                    topk=parameters["topk"],
                    shrinkage=parameters["shrinkage"],
                    reliability_mode=parameters["reliability_mode"],
                    probe_topl=parameters["probe_topl"],
                    uncertainty_kappa=parameters["uncertainty_kappa"],
                    mass_budget_ratio=parameters["mass_budget_ratio"],
                    min_confidence=parameters["min_confidence"],
                    min_modal_support=parameters["min_modal_support"],
                    binary_weights=parameters["binary_weights"],
                )
            stats["cache_hit"] = False
            stats["parameters"] = parameters
            sp.save_npz(virtual_path, virtual)
            if donor_aware_transport:
                sp.save_npz(donor_path, donor_upper_bounds)
            with open(stats_path, "w", encoding="utf-8") as handle:
                json.dump(stats, handle, ensure_ascii=False, indent=2)

        self.pravi_stats = {**stats, "adaptive_alignment": reliability_log}
        self.virtual_interaction_matrix = virtual
        self.transport_matrix = None
        self.virtual_response_adj = None
        self.pravi_response_scale = float(config["pravi_response_scale"])
        # DVT is an auxiliary signed response, not a source of BPR positives.
        self.pravi_filter_virtual_negatives = bool(
            config["pravi_filter_virtual_negatives"]
        ) and propagation_scope != "dvt"

        if propagation_scope in {"collab", "collab_semantic"}:
            self.interaction_matrix = (observed + virtual).tocoo()
            augmented_norm_R, self.collab_norm_adj = self.get_norm_adj_mat()
            if propagation_scope == "collab_semantic":
                self.norm_R = augmented_norm_R
        elif propagation_scope == "first_order":
            self.collab_norm_adj = self.norm_adj
            self.virtual_response_adj = build_anchored_virtual_operator(
                observed, virtual, self.device
            )
        elif propagation_scope == "dvt":
            if transport_method in {
                "cycle_cover",
                "net_utility_cycle",
                "profile_drift_cycle",
            }:
                transport, transport_stats = build_evidence_cycle_transport(
                    observed,
                    virtual,
                    weight_mode=config["pravi_dvt_weight_mode"],
                    min_arc_confidence=float(
                        config["pravi_dvt_min_arc_confidence"]
                    ),
                    donor_upper_bounds=(
                        donor_upper_bounds
                        if donor_aware_transport
                        else None
                    ),
                    min_net_margin=float(config["pravi_dvt_min_net_margin"]),
                )
            elif transport_method == "two_switch":
                transport, transport_stats = build_degree_preserving_transport(
                    observed,
                    virtual,
                    max_swaps_per_user=int(
                        config["pravi_dvt_max_swaps_per_user"]
                    ),
                    weight_mode=config["pravi_dvt_weight_mode"],
                )
            else:
                raise ValueError(f"unknown DVT transport: {transport_method}")
            self.transport_matrix = transport
            self.pravi_stats = {
                **stats,
                "adaptive_alignment": reliability_log,
                "dvt": transport_stats,
            }
            self.collab_norm_adj = self.norm_adj
            self.virtual_response_adj = build_signed_transport_operator(
                observed, transport, self.device
            )
        else:
            raise ValueError(f"Unknown PRAVI propagation scope: {propagation_scope}")
        self.interaction_matrix = observed.tocoo()

        virtual_coo = virtual.tocoo()
        virtual_keys = virtual_coo.row.astype("int64") * self.n_items + virtual_coo.col
        self.virtual_keys = torch.as_tensor(
            sorted(virtual_keys.tolist()), dtype=torch.long, device=self.device
        )

        logging.getLogger().info(
            "PRAVI graph: %s",
            json.dumps(
                {
                    **self.pravi_stats,
                    "propagation_scope": propagation_scope,
                    "response_scale": self.pravi_response_scale,
                    "filter_virtual_negatives": self.pravi_filter_virtual_negatives,
                    "response_operator_nnz": (
                        int(self.virtual_response_adj._nnz())
                        if self.virtual_response_adj is not None
                        else 0
                    ),
                },
                sort_keys=True,
            ),
        )

    def _adaptive_ratio(self, modality_index):
        """Return clipped train-only trust ratios for one modality."""
        if not hasattr(self, "pravi_user_reliability_ratio"):
            return None
        if modality_index >= self.pravi_user_reliability_ratio.shape[0]:
            return None
        return self.pravi_user_reliability_ratio[modality_index]

    def _adaptive_scale(self, modality_index):
        ratio = self._adaptive_ratio(modality_index)
        if ratio is None:
            return None
        # Keep the intervention bounded; the strength is selected before
        # training and is never fitted against validation/test metrics.
        return (
            1.0 + self.pravi_reliability_strength * (ratio - 1.0)
        ).clamp(0.25, 1.75)

    def _route_modality_view(self, embeds, modality_index):
        if self.pravi_reliability_conditioning not in {"fusion_route", "both"}:
            return embeds
        scale = self._adaptive_scale(modality_index)
        if scale is None:
            return embeds
        user_embeds = embeds[: self.n_users] * scale.unsqueeze(1)
        return torch.cat([user_embeds, embeds[self.n_users :]], dim=0)

    def _get_cl_sample_weight(self, ids, modality_index, is_user):
        if not is_user or self.pravi_reliability_conditioning not in {
            "cl_weight",
            "both",
        }:
            return None
        scale = self._adaptive_scale(modality_index)
        if scale is None:
            return None
        return scale[ids.long()]

    def _get_attention_bias(self, multi_embed_list):
        if self.pravi_reliability_conditioning != "attention_prior":
            return None
        if not hasattr(self, "pravi_user_reliability_ratio"):
            return None
        bias = torch.zeros(
            (multi_embed_list[0].shape[0], len(multi_embed_list)),
            dtype=multi_embed_list[0].dtype,
            device=multi_embed_list[0].device,
        )
        for modality_index in (0, 1):
            ratio = self._adaptive_ratio(modality_index)
            if ratio is not None:
                # A log prior is additive in attention logits and leaves the
                # semantic embedding geometry unchanged.
                bias[: self.n_users, modality_index] = (
                    self.pravi_reliability_strength * torch.log(ratio)
                )
        return bias

    def _get_user_modality_cl_loss(
        self,
        collab_user_embeds,
        image_user_embeds,
        text_user_embeds,
        users,
    ):
        """Align collaborative users to a reliable shared/private decomposition.

        MSCA aligns the same collaborative vector independently to each
        modality.  When the views disagree, those objectives can provide
        opposing gradients.  This branch first forms a train-only reliability
        weighted consensus (the shared component), then optionally aligns the
        collaborative residual to each modality residual.  The consensus basis
        used for residual projection is detached so the private term cannot
        rotate the shared teacher.
        """
        if self.pravi_reliability_conditioning != "consensus_residual":
            return super(MSCA_PRAVI, self)._get_user_modality_cl_loss(
                collab_user_embeds,
                image_user_embeds,
                text_user_embeds,
                users,
            )
        if not hasattr(self, "pravi_user_reliability_ratio"):
            return super(MSCA_PRAVI, self)._get_user_modality_cl_loss(
                collab_user_embeds,
                image_user_embeds,
                text_user_embeds,
                users,
            )

        user_ids = users.long()
        collab = collab_user_embeds[user_ids]
        image = F.normalize(image_user_embeds[user_ids], p=2, dim=1)
        text = F.normalize(text_user_embeds[user_ids], p=2, dim=1)

        # Ratios are posterior-median referenced and precision-shrunk in the
        # constructor.  Normalizing them per user gives a convex shared
        # semantic teacher without forcing the two modalities to sum to one
        # in the underlying reliability estimator.
        ratio = torch.stack(
            [
                self.pravi_user_reliability_ratio[0, user_ids],
                self.pravi_user_reliability_ratio[1, user_ids],
            ],
            dim=1,
        ).clamp_min(1e-6)
        modality_weights = ratio / ratio.sum(dim=1, keepdim=True).clamp_min(1e-6)
        consensus = F.normalize(
            modality_weights[:, 0:1] * image
            + modality_weights[:, 1:2] * text,
            p=2,
            dim=1,
        )

        agreement = ((image * text).sum(dim=1) + 1.0).mul(0.5).clamp(0.0, 1.0)
        precision = torch.ones_like(agreement)
        if hasattr(self, "pravi_user_reliability") and hasattr(
            self, "pravi_user_reliability_variance"
        ):
            posterior = self.pravi_user_reliability[:, user_ids].clamp_min(1e-6)
            uncertainty = torch.sqrt(
                self.pravi_user_reliability_variance[:, user_ids].clamp_min(0.0)
            )
            precision = (posterior / (posterior + uncertainty + 1e-6)).mean(dim=0)
        trust = agreement * (0.5 + 0.5 * precision)
        trust = trust / trust.mean().clamp_min(1e-6)
        trust = trust.clamp(0.5, 1.5)

        # Two consensus anchors preserve the scale of MSCA's two modality
        # terms, while making the shared direction explicit.
        shared_loss = 2.0 * self.cal_cl_loss(
            collab,
            consensus,
            self.tau,
            sample_weight=trust,
        )

        private_loss = collab.new_zeros(())
        if self.pravi_consensus_private_strength > 0.0:
            basis = consensus.detach()
            collab_unit = F.normalize(collab, p=2, dim=1)
            collab_residual = collab_unit - (collab_unit * basis).sum(
                dim=1, keepdim=True
            ) * basis
            image_residual = image - (image * basis).sum(dim=1, keepdim=True) * basis
            text_residual = text - (text * basis).sum(dim=1, keepdim=True) * basis
            disagreement = (1.0 - agreement).clamp(0.25, 1.0)
            disagreement = disagreement / disagreement.mean().clamp_min(1e-6)
            disagreement = disagreement.clamp(0.5, 1.5)
            private_loss = self.cal_cl_loss(
                collab_residual,
                image_residual,
                self.tau,
                sample_weight=disagreement,
            ) + self.cal_cl_loss(
                collab_residual,
                text_residual,
                self.tau,
                sample_weight=disagreement,
            )

        new_loss = shared_loss + self.pravi_consensus_private_strength * private_loss
        mix = float(self.pravi_reliability_strength)
        mix = max(0.0, min(1.0, mix))
        if mix >= 1.0:
            return new_loss
        baseline_loss = super(MSCA_PRAVI, self)._get_user_modality_cl_loss(
            collab_user_embeds,
            image_user_embeds,
            text_user_embeds,
            users,
        )
        return (1.0 - mix) * baseline_loss + mix * new_loss

    def collaborative_views(self, initial_embeds):
        if self.virtual_response_adj is None:
            return super(MSCA_PRAVI, self).collaborative_views(initial_embeds)

        clean = initial_embeds
        response = torch.zeros_like(initial_embeds)
        clean_layers = []
        response_layers = []
        for _ in range(self.n_layers):
            # This recurrence is the first-order response of graph propagation:
            # every retained path contains at most one virtual edge operator.
            response = torch.sparse.mm(self.collab_norm_adj, response) + torch.sparse.mm(
                self.virtual_response_adj, clean
            )
            clean = torch.sparse.mm(self.collab_norm_adj, clean)
            clean_layers.append(clean)
            response_layers.append(response)
        clean_collab = torch.mean(torch.stack(clean_layers, dim=1), dim=1)
        virtual_response = torch.mean(torch.stack(response_layers, dim=1), dim=1)
        scoring_collab = clean_collab + self.pravi_response_scale * virtual_response
        # Contrastive alignment remains exactly on the official clean view.
        return scoring_collab, clean_collab

    def calculate_loss(self, interaction):
        """Drop sampled negatives that are virtual positive evidence."""
        if (
            not self.pravi_filter_virtual_negatives
            or self.virtual_keys.numel() == 0
        ):
            return super(MSCA_PRAVI, self).calculate_loss(interaction)
        users, neg_items = interaction[0], interaction[2]
        keys = users.long() * self.n_items + neg_items.long()
        positions = torch.searchsorted(self.virtual_keys, keys)
        bounded = positions < self.virtual_keys.numel()
        matches = torch.zeros_like(bounded)
        matches[bounded] = self.virtual_keys[positions[bounded]] == keys[bounded]
        if not torch.any(matches):
            return super(MSCA_PRAVI, self).calculate_loss(interaction)
        keep = ~matches
        if not torch.any(keep):
            return self.user_embedding.weight.sum() * 0.0
        filtered = [field[keep] for field in interaction]
        return super(MSCA_PRAVI, self).calculate_loss(filtered)
