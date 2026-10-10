"""Diagnose joint cross-fitted image/text collaborative evidence fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from diagnose_c3e import (
    FeatureScorer,
    build_histories,
    sample_co_consumption_pairs,
    sample_degree_matched_pairs,
    sample_unobserved_negatives,
)
from utils.c3e import (
    EvidenceCalibrator,
    deterministic_user_folds,
    make_degree_edges,
    topq_mean,
)
from utils.jcef import (
    RidgeLogisticCalibrator,
    diagnostic_gate,
    evidence_features,
)
from utils.rating_signal import event_pairwise_auc, mean_and_ci


ADDITIVE_NAMES = ("image_evidence", "text_evidence")
JOINT_NAMES = (
    "image_evidence",
    "text_evidence",
    "agreement_product",
    "evidence_disagreement",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="baby")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--negatives", type=int, default=16)
    parser.add_argument("--top-q", type=int, default=3)
    parser.add_argument("--max-calibration-pairs", type=int, default=100000)
    parser.add_argument("--max-pairs-per-user", type=int, default=8)
    parser.add_argument("--similarity-bins", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--pair-batch-size", type=int, default=8192)
    parser.add_argument("--event-chunk-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def fit_global_curve(
    similarity_positive,
    similarity_negative,
    positive_left,
    positive_right,
    negative_left,
    negative_right,
    degrees,
    bins,
):
    return EvidenceCalibrator.fit(
        similarity_positive,
        similarity_negative,
        positive_left,
        positive_right,
        negative_left,
        negative_right,
        degrees,
        degree_edges=np.empty(0, dtype=np.float32),
        n_similarity_bins=bins,
        shrinkage=0.0,
    )


def global_evidence(calibrator, similarity, left, right, degrees):
    return calibrator.transform(
        similarity,
        left,
        right,
        degrees,
        degree_conditioned=False,
    )


def score_target_fold(
    scorers,
    curves,
    additive_model,
    joint_model,
    event_indices,
    users,
    positives,
    negatives,
    histories,
    item_degrees,
    top_q,
    event_chunk_size,
):
    score_names = ("raw_topq", "fixed_global", "learned_additive", "jcef")
    output = {
        name: {
            "positive": np.zeros(len(event_indices), dtype=np.float32),
            "negative": np.zeros((len(event_indices), negatives.shape[1]), dtype=np.float32),
        }
        for name in score_names
    }
    candidate_count = negatives.shape[1] + 1
    for chunk_start in range(0, len(event_indices), int(event_chunk_size)):
        chunk_stop = min(chunk_start + int(event_chunk_size), len(event_indices))
        chunk_indices = event_indices[chunk_start:chunk_stop]
        left_parts = []
        right_parts = []
        history_sizes = []
        for event_index in chunk_indices:
            history = histories[int(users[event_index])]
            candidates = np.concatenate(
                ([positives[event_index]], negatives[event_index])
            )
            left_parts.append(np.repeat(candidates, history.size))
            right_parts.append(np.tile(history, candidate_count))
            history_sizes.append(int(history.size))
        left = np.concatenate(left_parts)
        right = np.concatenate(right_parts)

        raw = {}
        evidence = {}
        for modality in ("image", "text"):
            raw[modality] = scorers[modality].pair_cosine(left, right)
            evidence[modality] = global_evidence(
                curves[modality], raw[modality], left, right, item_degrees
            )
        additive_values = additive_model.decision_function(
            evidence_features(evidence["image"], evidence["text"], False)
        )
        joint_values = joint_model.decision_function(
            evidence_features(evidence["image"], evidence["text"], True)
        )

        cursor = 0
        for local_index, history_size in enumerate(history_sizes):
            value_count = candidate_count * history_size
            shape = (candidate_count, history_size)
            image_raw = topq_mean(
                raw["image"][cursor : cursor + value_count].reshape(shape), top_q
            )
            text_raw = topq_mean(
                raw["text"][cursor : cursor + value_count].reshape(shape), top_q
            )
            image_evidence = topq_mean(
                evidence["image"][cursor : cursor + value_count].reshape(shape),
                top_q,
            )
            text_evidence = topq_mean(
                evidence["text"][cursor : cursor + value_count].reshape(shape),
                top_q,
            )
            candidate_scores = {
                "raw_topq": 0.5 * (image_raw + text_raw),
                "fixed_global": 0.5 * (image_evidence + text_evidence),
                "learned_additive": topq_mean(
                    additive_values[cursor : cursor + value_count].reshape(shape),
                    top_q,
                ),
                "jcef": topq_mean(
                    joint_values[cursor : cursor + value_count].reshape(shape),
                    top_q,
                ),
            }
            for name, values in candidate_scores.items():
                output[name]["positive"][chunk_start + local_index] = values[0]
                output[name]["negative"][chunk_start + local_index] = values[1:]
            cursor += value_count
    return output


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_root = args.data_root or repo_root / "data"
    dataset_dir = data_root / args.dataset
    frame = pd.read_csv(
        dataset_dir / f"{args.dataset}.inter",
        sep="\t",
        usecols=["userID", "itemID", "x_label"],
    )
    train = frame[frame["x_label"] == 0]
    valid = frame[frame["x_label"] == 1]
    feature_paths = {
        name: dataset_dir / filename
        for name, filename in (
            ("image", "image_feat.npy"),
            ("text", "text_feat.npy"),
        )
    }
    missing = [name for name, path in feature_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"JCEF requires image and text features; missing {missing}")

    n_users = int(frame["userID"].max()) + 1
    n_items = int(np.load(feature_paths["image"], mmap_mode="r").shape[0])
    train_users = train["userID"].to_numpy(np.int64)
    train_items = train["itemID"].to_numpy(np.int64)
    valid_users = valid["userID"].to_numpy(np.int64)
    valid_items = valid["itemID"].to_numpy(np.int64)
    histories = build_histories(train_users, train_items, n_users)
    negatives = sample_unobserved_negatives(
        histories, valid_items, valid_users, n_items, args.negatives, args.seed
    )
    full_item_degrees = np.bincount(train_items, minlength=n_items).astype(np.float32)
    inverse_degree = np.divide(
        1.0,
        full_item_degrees,
        out=np.zeros_like(full_item_degrees),
        where=full_item_degrees > 0,
    )
    exposure_operator = sp.coo_matrix(
        (inverse_degree[train_items], (train_users, train_items)),
        shape=(n_users, n_items),
        dtype=np.float32,
    ).tocsr()
    scorers = {
        modality: FeatureScorer(
            np.asarray(np.load(path), dtype=np.float32),
            args.device,
            args.pair_batch_size,
        )
        for modality, path in feature_paths.items()
    }

    all_users = np.arange(n_users, dtype=np.int64)
    user_fold = deterministic_user_folds(all_users, args.folds, args.seed)
    score_names = (
        "exposure",
        "raw_topq",
        "fixed_global",
        "learned_additive",
        "jcef",
    )
    scores = {
        name: {
            "positive": np.zeros(len(valid), dtype=np.float32),
            "negative": np.zeros(negatives.shape, dtype=np.float32),
        }
        for name in score_names
    }
    for scorer in scorers.values():
        positive, negative = scorer.exposure_scores(
            exposure_operator, valid_users, valid_items, negatives
        )
        scores["exposure"]["positive"] += 0.5 * positive
        scores["exposure"]["negative"] += 0.5 * negative

    fold_models = []
    for fold in range(args.folds):
        calibration_users = all_users[user_fold != fold]
        calibration_mask = user_fold[train_users] != fold
        item_degrees = np.bincount(
            train_items[calibration_mask], minlength=n_items
        ).astype(np.float32)
        degree_edges = make_degree_edges(item_degrees, 4)
        positive_left, positive_right = sample_co_consumption_pairs(
            histories,
            calibration_users,
            args.max_pairs_per_user,
            args.max_calibration_pairs,
            args.seed + 101 * fold,
        )
        negative_left, negative_right = sample_degree_matched_pairs(
            item_degrees,
            degree_edges,
            positive_left,
            positive_right,
            args.seed + 1009 + 101 * fold,
        )
        positive_evidence = {}
        negative_evidence = {}
        curves = {}
        for modality in ("image", "text"):
            positive_similarity = scorers[modality].pair_cosine(
                positive_left, positive_right
            )
            negative_similarity = scorers[modality].pair_cosine(
                negative_left, negative_right
            )
            curves[modality] = fit_global_curve(
                positive_similarity,
                negative_similarity,
                positive_left,
                positive_right,
                negative_left,
                negative_right,
                item_degrees,
                args.similarity_bins,
            )
            positive_evidence[modality] = global_evidence(
                curves[modality],
                positive_similarity,
                positive_left,
                positive_right,
                item_degrees,
            )
            negative_evidence[modality] = global_evidence(
                curves[modality],
                negative_similarity,
                negative_left,
                negative_right,
                item_degrees,
            )
        labels = np.concatenate(
            (
                np.ones(positive_left.size, dtype=np.float64),
                np.zeros(negative_left.size, dtype=np.float64),
            )
        )
        image_evidence = np.concatenate(
            (positive_evidence["image"], negative_evidence["image"])
        )
        text_evidence = np.concatenate(
            (positive_evidence["text"], negative_evidence["text"])
        )
        additive_model = RidgeLogisticCalibrator.fit(
            evidence_features(image_evidence, text_evidence, False),
            labels,
            l2=args.ridge,
        )
        joint_model = RidgeLogisticCalibrator.fit(
            evidence_features(image_evidence, text_evidence, True),
            labels,
            l2=args.ridge,
        )
        event_indices = np.flatnonzero(user_fold[valid_users] == fold)
        fold_scores = score_target_fold(
            scorers,
            curves,
            additive_model,
            joint_model,
            event_indices,
            valid_users,
            valid_items,
            negatives,
            histories,
            item_degrees,
            args.top_q,
            args.event_chunk_size,
        )
        for name, values in fold_scores.items():
            scores[name]["positive"][event_indices] = values["positive"]
            scores[name]["negative"][event_indices] = values["negative"]
        fold_models.append(
            {
                "fold": fold,
                "events": int(event_indices.size),
                "calibration_pairs_per_class": int(positive_left.size),
                "additive": additive_model.summary(ADDITIVE_NAMES),
                "joint": joint_model.summary(JOINT_NAMES),
                "global_curves": {
                    modality: curves[modality].summary()
                    for modality in ("image", "text")
                },
            }
        )

    event_auc = {
        name: event_pairwise_auc(values["positive"], values["negative"])
        for name, values in scores.items()
    }
    metrics = {name: mean_and_ci(values) for name, values in event_auc.items()}
    delta_fixed = mean_and_ci(event_auc["jcef"] - event_auc["fixed_global"])
    delta_additive = mean_and_ci(
        event_auc["jcef"] - event_auc["learned_additive"]
    )
    delta_raw = mean_and_ci(event_auc["jcef"] - event_auc["raw_topq"])
    fold_metrics = []
    fold_deltas = []
    for fold in range(args.folds):
        mask = user_fold[valid_users] == fold
        fold_delta = mean_and_ci(
            event_auc["jcef"][mask] - event_auc["fixed_global"][mask]
        )
        fold_deltas.append(fold_delta["mean"])
        fold_metrics.append(
            {
                "fold": fold,
                "events": int(mask.sum()),
                "jcef_auc": mean_and_ci(event_auc["jcef"][mask]),
                "fixed_global_auc": mean_and_ci(event_auc["fixed_global"][mask]),
                "jcef_delta_vs_fixed_global": fold_delta,
            }
        )
    gate = diagnostic_gate(
        joint_auc=metrics["jcef"]["mean"],
        delta_fixed=delta_fixed["mean"],
        delta_fixed_ci_low=delta_fixed["ci95_low"],
        delta_additive=delta_additive["mean"],
        delta_additive_ci_low=delta_additive["ci95_low"],
        fold_deltas=fold_deltas,
    )
    output = {
        "method": "JCEF",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "device": next(iter(scorers.values())).device,
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "negative_samples_per_event": int(args.negatives),
        "folds": int(args.folds),
        "top_q": int(args.top_q),
        "metrics": metrics,
        "paired_delta_vs_fixed_global": delta_fixed,
        "paired_delta_vs_learned_additive": delta_additive,
        "paired_delta_vs_raw_topq": delta_raw,
        "fold_metrics": fold_metrics,
        "models": fold_models,
        "gate": gate,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("JCEF_SUMMARY " + json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
