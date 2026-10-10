"""Validation-only checkpoint diagnostic for ECLI candidate refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.bridge import BRIDGE
from models.dual_frequency_encoder import DualFrequencyEncoder
from utils.configurator import Config
from utils.dataloader import EvalDataLoader, TrainDataLoader
from utils.dataset import RecDataset
from utils.ecli import (
    apply_local_evidence,
    build_history_table,
    mean_ci95,
    per_user_ranking_metrics,
    topq_history_similarity,
    validation_gate,
)
from utils.topk_evaluator import TopKEvaluator
from utils.utils import init_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="baby")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--top-q", type=int, default=3)
    parser.add_argument("--anchor-k", type=int, default=50)
    parser.add_argument("--clip-z", type=float, default=2.0)
    return parser.parse_args()


def scalarize_hyperparameters(config):
    for name in config["hyper_parameters"]:
        value = config[name]
        if isinstance(value, (list, tuple)):
            config[name] = value[0] if value else None


def cache_bridge_embeddings(model):
    raw_user, item, _, user_frequencies, item_frequencies = (
        DualFrequencyEncoder.forward(model)
    )
    user, _, _ = model._apply_behavior_proto(raw_user, item)
    low_user, low_item, high_user, high_item = model._frequency_views(
        raw_user, item, user_frequencies, item_frequencies
    )
    return user, item, low_user, low_item, high_user, high_item


def bridge_scores(model, users, cached):
    user_all, item_all, low_user, low_item, high_user, high_item = cached
    user = user_all[users]
    base_scores = torch.matmul(user, item_all.transpose(0, 1))
    if not model.use_topk_correction or model.behavior_eval_topk <= 0:
        return base_scores
    if (
        (not model.use_behavior_correction or model.behavior_weight == 0.0)
        and (not model.use_private_residual or model.private_residual_weight == 0.0)
    ):
        return base_scores
    if model.correction_scope != "topk":
        raise ValueError("ECLI diagnostic requires BRIDGE top-k correction")

    candidate_k = min(max(model.behavior_eval_topk, 50), model.n_items)
    _, candidate_idx = torch.topk(base_scores, k=candidate_k, dim=1)
    candidate_base = base_scores.gather(1, candidate_idx)
    candidate_behavior = model._candidate_behavior_scores(users, candidate_idx)
    high_scores = torch.matmul(
        high_user[users], high_item.transpose(0, 1)
    ).gather(1, candidate_idx)
    low_scores = torch.matmul(
        low_user[users], low_item.transpose(0, 1)
    ).gather(1, candidate_idx)
    candidate_residual = model._transform_residual(high_scores - low_scores)
    _, _, behavior_correction, residual_correction = model._combine_score_components(
        users,
        candidate_idx,
        candidate_base,
        candidate_behavior,
        candidate_residual,
    )
    correction = behavior_correction + residual_correction
    output = base_scores.clone()
    output.scatter_add_(1, candidate_idx, correction)
    return output


def projected_modality_features(model):
    features = []
    if model.v_feat is not None:
        visual = model.v_gcn.MLP_1(F.leaky_relu(model.v_gcn.MLP(model.v_feat)))
        features.append(F.normalize(visual, dim=1, eps=1e-6))
    if model.t_feat is not None:
        textual = model.t_gcn.MLP_1(F.leaky_relu(model.t_gcn.MLP(model.t_feat)))
        features.append(F.normalize(textual, dim=1, eps=1e-6))
    if not features:
        raise ValueError("ECLI requires at least one modality feature matrix")
    return features


def exact_paired_metrics(baseline_topk, treatment_topk, valid_data):
    positives = valid_data.get_eval_items()
    positive_lengths = valid_data.get_eval_len_list()

    def hits(recommendations):
        rows = []
        for positive, recommended in zip(positives, recommendations):
            rows.append(np.isin(recommended, positive, assume_unique=False))
        return np.asarray(rows, dtype=np.bool_)

    baseline_hits = hits(baseline_topk)
    treatment_hits = hits(treatment_topk)
    baseline = {}
    treatment = {}
    paired_delta = {}
    for cutoff in (10, 20):
        base_recall, base_ndcg = per_user_ranking_metrics(
            baseline_hits, positive_lengths, cutoff
        )
        treat_recall, treat_ndcg = per_user_ranking_metrics(
            treatment_hits, positive_lengths, cutoff
        )
        for name, base_rows, treat_rows in (
            ("recall", base_recall, treat_recall),
            ("ndcg", base_ndcg, treat_ndcg),
        ):
            key = "{}@{}".format(name, cutoff)
            baseline[key] = float(base_rows.mean())
            treatment[key] = float(treat_rows.mean())
            paired_delta[key] = mean_ci95(treat_rows - base_rows)
    return baseline, treatment, paired_delta


@torch.no_grad()
def evaluate(model, valid_data, config, history_items, history_mask, args):
    model.eval()
    cached = cache_bridge_embeddings(model)
    modality_features = projected_modality_features(model)
    evaluator = TopKEvaluator(config)
    baseline_batches = []
    treatment_batches = []
    max_k = max(config["topk"])
    candidate_k = min(max(int(args.candidate_k), max_k), model.n_items)

    for batched_data in valid_data:
        users = batched_data[0]
        scores = bridge_scores(model, users, cached)
        masked_items = batched_data[1]
        scores[masked_items[0], masked_items[1]] = -1e10
        baseline_batches.append(torch.topk(scores, max_k, dim=1).indices.cpu())

        candidate_scores, candidate_idx = torch.topk(
            scores, candidate_k, dim=1
        )
        batch_history = history_items[users]
        batch_mask = history_mask[users]
        evidence = torch.zeros_like(candidate_scores)
        for features in modality_features:
            evidence.add_(
                topq_history_similarity(
                    candidate_idx,
                    batch_history,
                    batch_mask,
                    features,
                    top_q=args.top_q,
                )
            )
        evidence.div_(float(len(modality_features)))
        refined = apply_local_evidence(
            candidate_scores,
            evidence,
            anchor_k=args.anchor_k,
            clip_z=args.clip_z,
        )
        treatment_scores = scores.clone()
        treatment_scores.scatter_(1, candidate_idx, refined)
        treatment_batches.append(
            torch.topk(treatment_scores, max_k, dim=1).indices.cpu()
        )

    baseline = evaluator.evaluate(baseline_batches, valid_data)
    treatment = evaluator.evaluate(treatment_batches, valid_data)
    baseline_topk = torch.cat(baseline_batches, dim=0).numpy()
    treatment_topk = torch.cat(treatment_batches, dim=0).numpy()
    exact = exact_paired_metrics(baseline_topk, treatment_topk, valid_data)
    return baseline, treatment, exact


def main():
    args = parse_args()
    config = Config(
        "BRIDGE",
        args.dataset,
        {
            "gpu_id": args.gpu_id,
            "seed": [args.seed],
            "eval_batch_size": args.batch_size,
            "save_recommended_topk": False,
        },
    )
    scalarize_hyperparameters(config)
    config["seed"] = args.seed
    init_seed(args.seed)

    dataset = RecDataset(config)
    train_dataset, valid_dataset, _ = dataset.split()
    # The upstream loader initializes inter_num in RecDataset.__str__.
    str(train_dataset)
    str(valid_dataset)
    train_data = TrainDataLoader(
        config,
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
    )
    valid_data = EvalDataLoader(
        config,
        valid_dataset,
        additional_dataset=train_dataset,
        batch_size=args.batch_size,
    )
    model = BRIDGE(config, train_data).to(config["device"])
    state = torch.load(args.checkpoint, map_location=config["device"])
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = {"user_activity_z"}
    allowed_unexpected = {"id_preference", "v_preference", "t_preference"}
    unknown_missing = set(incompatible.missing_keys) - allowed_missing
    unknown_unexpected = set(incompatible.unexpected_keys) - allowed_unexpected
    if unknown_missing or unknown_unexpected:
        raise RuntimeError(
            "Unsupported checkpoint mismatch: missing={} unexpected={}".format(
                sorted(unknown_missing), sorted(unknown_unexpected)
            )
        )
    print(
        "CHECKPOINT_LOADED missing={} unexpected={}".format(
            incompatible.missing_keys, incompatible.unexpected_keys
        ),
        flush=True,
    )

    history_items, history_mask = build_history_table(
        train_dataset.df[train_dataset.uid_field].to_numpy(),
        train_dataset.df[train_dataset.iid_field].to_numpy(),
        dataset.user_num,
    )
    history_items = history_items.to(config["device"])
    history_mask = history_mask.to(config["device"])
    baseline, treatment, exact = evaluate(
        model, valid_data, config, history_items, history_mask, args
    )
    exact_baseline, exact_treatment, paired_delta = exact
    gate = validation_gate(exact_baseline, exact_treatment)
    output = {
        "method": "ECLI",
        "split": "validation",
        "dataset": args.dataset,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "settings": {
            "candidate_k": args.candidate_k,
            "top_q": args.top_q,
            "anchor_k": args.anchor_k,
            "clip_z": args.clip_z,
            "fusion": "equal_projected_image_text",
            "residual_scale": "per_user_median_top_anchor_score_gap",
        },
        "baseline": baseline,
        "treatment": treatment,
        "exact": {
            "baseline": exact_baseline,
            "treatment": exact_treatment,
            "paired_delta_ci95": paired_delta,
        },
        "gate": gate,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
