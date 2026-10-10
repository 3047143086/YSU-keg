"""Validation-only TACE diagnostic on a frozen BRIDGE checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diagnose_ecli import (
    cache_bridge_embeddings,
    exact_paired_metrics,
    scalarize_hyperparameters,
)
from models.bridge import BRIDGE
from utils.configurator import Config
from utils.dataloader import EvalDataLoader, TrainDataLoader
from utils.dataset import RecDataset
from utils.ecli import validation_gate
from utils.tace import TemporalAffinityScorer, build_temporal_affinity
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
    return parser.parse_args()


def paired_bridge_scores(model, users, cached, temporal_scorer):
    """Return official and time-aware scores with an identical candidate scope."""
    user_all, item_all, low_user, low_item, high_user, high_item = cached
    base_scores = torch.matmul(user_all[users], item_all.transpose(0, 1))
    if not model.use_topk_correction or model.behavior_eval_topk <= 0:
        return base_scores, base_scores
    if model.correction_scope != "topk":
        raise ValueError("TACE diagnostic requires BRIDGE top-k correction")

    candidate_k = min(max(model.behavior_eval_topk, 50), model.n_items)
    _, candidate_idx = torch.topk(base_scores, k=candidate_k, dim=1)
    candidate_base = base_scores.gather(1, candidate_idx)
    original_behavior = model._candidate_behavior_scores(users, candidate_idx)
    temporal_behavior = temporal_scorer.score(users, candidate_idx)
    high_scores = torch.matmul(
        high_user[users], high_item.transpose(0, 1)
    ).gather(1, candidate_idx)
    low_scores = torch.matmul(
        low_user[users], low_item.transpose(0, 1)
    ).gather(1, candidate_idx)
    residual = model._transform_residual(high_scores - low_scores)

    _, _, original_correction, original_residual = model._combine_score_components(
        users, candidate_idx, candidate_base, original_behavior, residual
    )
    _, _, temporal_correction, temporal_residual = model._combine_score_components(
        users, candidate_idx, candidate_base, temporal_behavior, residual
    )
    official = base_scores.clone()
    official.scatter_add_(
        1, candidate_idx, original_correction + original_residual
    )
    treatment = base_scores.clone()
    treatment.scatter_add_(
        1, candidate_idx, temporal_correction + temporal_residual
    )
    return official, treatment


@torch.no_grad()
def evaluate(model, valid_data, config, temporal_scorer):
    model.eval()
    cached = cache_bridge_embeddings(model)
    evaluator = TopKEvaluator(config)
    baseline_batches = []
    treatment_batches = []
    max_k = max(config["topk"])

    for batched_data in valid_data:
        users = batched_data[0]
        baseline, treatment = paired_bridge_scores(
            model, users, cached, temporal_scorer
        )
        masked_items = batched_data[1]
        baseline[masked_items[0], masked_items[1]] = -1e10
        treatment[masked_items[0], masked_items[1]] = -1e10
        baseline_batches.append(torch.topk(baseline, max_k, dim=1).indices.cpu())
        treatment_batches.append(torch.topk(treatment, max_k, dim=1).indices.cpu())

    rounded_baseline = evaluator.evaluate(baseline_batches, valid_data)
    rounded_treatment = evaluator.evaluate(treatment_batches, valid_data)
    exact = exact_paired_metrics(
        torch.cat(baseline_batches, dim=0).numpy(),
        torch.cat(treatment_batches, dim=0).numpy(),
        valid_data,
    )
    return rounded_baseline, rounded_treatment, exact


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

    interaction_path = Path(dataset.dataset_path) / config["inter_file_name"]
    columns = [
        dataset.uid_field,
        dataset.iid_field,
        config["TIME_FIELD"],
        dataset.splitting_label,
    ]
    interactions = pd.read_csv(
        interaction_path,
        usecols=columns,
        sep=config["field_separator"],
    )
    train_interactions = interactions[
        interactions[dataset.splitting_label] == 0
    ]
    affinity, temporal_train, affinity_stats = build_temporal_affinity(
        train_interactions[dataset.uid_field].to_numpy(),
        train_interactions[dataset.iid_field].to_numpy(),
        train_interactions[config["TIME_FIELD"]].to_numpy(),
        dataset.user_num,
        dataset.item_num,
        topk=model.behavior_topk,
    )
    temporal_scorer = TemporalAffinityScorer(
        temporal_train,
        affinity,
        aggregation=model.behavior_aggregation,
        norm=model.behavior_score_norm,
    )
    rounded_baseline, rounded_treatment, exact = evaluate(
        model, valid_data, config, temporal_scorer
    )
    exact_baseline, exact_treatment, paired_delta = exact
    directional_gate = validation_gate(exact_baseline, exact_treatment)
    material_checks = {
        "recall20_delta_at_least_0.001": (
            exact_treatment["recall@20"] - exact_baseline["recall@20"] >= 0.001
        ),
        "ndcg20_delta_at_least_0.0004": (
            exact_treatment["ndcg@20"] - exact_baseline["ndcg@20"] >= 0.0004
        ),
    }
    gate = {
        "passed": directional_gate["passed"] and all(material_checks.values()),
        "directional": directional_gate,
        "material_checks": material_checks,
    }
    output = {
        "method": "TACE",
        "split": "validation",
        "dataset": args.dataset,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "settings": {
            "behavior_topk": model.behavior_topk,
            "behavior_aggregation": model.behavior_aggregation,
            "behavior_score_norm": model.behavior_score_norm,
            "kernel": "1/max(distinct_timestamp_rank_distance,1)",
            "selection": "frozen_no_coefficient_search",
        },
        "affinity_stats": affinity_stats,
        "baseline": rounded_baseline,
        "treatment": rounded_treatment,
        "exact": {
            "baseline": exact_baseline,
            "treatment": exact_treatment,
            "paired_delta_ci95": paired_delta,
        },
        "gate": gate,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
