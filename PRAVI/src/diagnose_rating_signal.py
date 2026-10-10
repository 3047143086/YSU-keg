"""Diagnose whether explicit ratings add target-safe multimodal preference signal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from utils.rating_signal import (
    build_profile_operators,
    diagnostic_gate,
    event_pairwise_auc,
    l2_normalize_rows,
    mean_and_ci,
    shuffle_values_within_users,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="baby")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--negatives", type=int, default=16)
    parser.add_argument("--correction-scale", type=float, default=0.25)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def sample_unobserved_negatives(
    train_operator, users, positives, n_items, count, seed
):
    rng = np.random.default_rng(int(seed))
    negatives = np.empty((len(users), int(count)), dtype=np.int64)
    indptr = train_operator.indptr
    indices = train_operator.indices
    for row, (user, positive) in enumerate(zip(users, positives)):
        forbidden = set(indices[indptr[user] : indptr[user + 1]].tolist())
        forbidden.add(int(positive))
        chosen = []
        while len(chosen) < count:
            candidates = rng.integers(
                0, n_items, size=max(8, 2 * (count - len(chosen)))
            )
            chosen.extend(int(x) for x in candidates if int(x) not in forbidden)
        negatives[row] = chosen[:count]
    return negatives


def score_profile(operator, features, users, positives, negatives, chunk=128):
    profile, valid_users = l2_normalize_rows(operator @ features)
    positive_scores = np.empty(len(users), dtype=np.float32)
    negative_scores = np.empty(negatives.shape, dtype=np.float32)
    for start in range(0, len(users), chunk):
        stop = min(start + chunk, len(users))
        batch_profile = profile[users[start:stop]]
        positive_scores[start:stop] = np.einsum(
            "bd,bd->b", batch_profile, features[positives[start:stop]]
        )
        negative_scores[start:stop] = np.einsum(
            "bd,bnd->bn",
            batch_profile,
            features[negatives[start:stop]],
            optimize=True,
        )
    return positive_scores, negative_scores, valid_users[users]


def evaluate_scores(positive, negative, masks):
    result = {}
    for name, mask in masks.items():
        result[name] = mean_and_ci(event_pairwise_auc(positive, negative, mask))
    return result


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_root = args.data_root or repo_root / "data"
    dataset_dir = data_root / args.dataset
    interaction_path = dataset_dir / f"{args.dataset}.inter"

    frame = pd.read_csv(
        interaction_path,
        sep="\t",
        usecols=["userID", "itemID", "rating", "x_label"],
    )
    train = frame[frame["x_label"] == 0].copy()
    valid = frame[frame["x_label"] == 1].copy()
    feature_paths = [
        ("image", dataset_dir / "image_feat.npy"),
        ("text", dataset_dir / "text_feat.npy"),
    ]
    feature_paths = [(name, path) for name, path in feature_paths if path.exists()]
    if not feature_paths:
        raise FileNotFoundError(f"No image/text features found under {dataset_dir}")

    n_users = int(frame["userID"].max()) + 1
    first_features = np.load(feature_paths[0][1], mmap_mode="r")
    n_items = int(first_features.shape[0])
    del first_features

    train_users = train["userID"].to_numpy(np.int64)
    train_items = train["itemID"].to_numpy(np.int64)
    train_ratings = train["rating"].to_numpy(np.float32)
    exposure_op, rating_op, _ = build_profile_operators(
        train_users, train_items, train_ratings, n_users, n_items
    )
    shuffled_ratings = shuffle_values_within_users(
        train_users, train_ratings, args.seed
    )
    _, shuffled_op, _ = build_profile_operators(
        train_users, train_items, shuffled_ratings, n_users, n_items
    )

    valid_users = valid["userID"].to_numpy(np.int64)
    valid_items = valid["itemID"].to_numpy(np.int64)
    valid_ratings = valid["rating"].to_numpy(np.float32)
    negatives = sample_unobserved_negatives(
        exposure_op,
        valid_users,
        valid_items,
        n_items,
        args.negatives,
        args.seed,
    )

    fused = {
        name: {
            "positive": np.zeros(len(valid), dtype=np.float32),
            "negative": np.zeros(negatives.shape, dtype=np.float32),
            "valid": np.ones(len(valid), dtype=bool),
        }
        for name in ("exposure", "rating", "rating_shuffled")
    }
    per_modality = {}
    operators = {
        "exposure": exposure_op,
        "rating": rating_op,
        "rating_shuffled": shuffled_op,
    }
    for modality, path in feature_paths:
        features = np.asarray(np.load(path), dtype=np.float32)
        features, _ = l2_normalize_rows(features)
        per_modality[modality] = {}
        for name, operator in operators.items():
            positive, negative, valid_profile = score_profile(
                operator,
                features,
                valid_users,
                valid_items,
                negatives,
            )
            fused[name]["positive"] += positive
            fused[name]["negative"] += negative
            fused[name]["valid"] &= valid_profile
            per_modality[modality][name] = {
                "coverage": float(valid_profile.mean()),
                "all_auc": mean_and_ci(
                    event_pairwise_auc(positive, negative, valid_profile)
                ),
            }
        del features

    modality_count = float(len(feature_paths))
    for values in fused.values():
        values["positive"] /= modality_count
        values["negative"] /= modality_count

    signed_valid = fused["rating"]["valid"]
    shuffled_valid = fused["rating_shuffled"]["valid"]
    masks = {
        "all": np.ones(len(valid), dtype=bool),
        "high_rating": valid_ratings >= 4.0,
        "low_rating": valid_ratings <= 2.0,
    }
    metrics = {}
    for name, values in fused.items():
        profile_mask = values["valid"]
        metrics[name] = {
            "coverage": float(profile_mask.mean()),
            "auc": evaluate_scores(
                values["positive"],
                values["negative"],
                {key: mask & profile_mask for key, mask in masks.items()},
            ),
        }

    scale = float(args.correction_scale)
    combined_positive = (
        fused["exposure"]["positive"] + scale * fused["rating"]["positive"]
    )
    combined_negative = (
        fused["exposure"]["negative"] + scale * fused["rating"]["negative"]
    )
    shuffled_combined_positive = (
        fused["exposure"]["positive"]
        + scale * fused["rating_shuffled"]["positive"]
    )
    shuffled_combined_negative = (
        fused["exposure"]["negative"]
        + scale * fused["rating_shuffled"]["negative"]
    )
    exposure_event_auc = event_pairwise_auc(
        fused["exposure"]["positive"], fused["exposure"]["negative"]
    )
    combined_event_auc = event_pairwise_auc(
        combined_positive, combined_negative
    )
    overall_delta = mean_and_ci(combined_event_auc - exposure_event_auc)
    combined = {
        "scale": scale,
        "auc": evaluate_scores(combined_positive, combined_negative, masks),
        "shuffled_auc": evaluate_scores(
            shuffled_combined_positive,
            shuffled_combined_negative,
            masks,
        ),
        "paired_delta_vs_exposure": overall_delta,
    }

    high_mask = masks["high_rating"] & signed_valid & shuffled_valid
    rating_high_auc = event_pairwise_auc(
        fused["rating"]["positive"],
        fused["rating"]["negative"],
        high_mask,
    ).mean()
    shuffled_high_auc = event_pairwise_auc(
        fused["rating_shuffled"]["positive"],
        fused["rating_shuffled"]["negative"],
        high_mask,
    ).mean()
    gate = diagnostic_gate(
        coverage=float(signed_valid.mean()),
        high_rating_auc=float(rating_high_auc),
        shuffled_high_rating_auc=float(shuffled_high_auc),
        overall_delta=float(overall_delta["mean"]),
        overall_delta_ci_low=float(overall_delta["ci95_low"]),
    )

    output = {
        "dataset": args.dataset,
        "seed": int(args.seed),
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "negative_samples_per_event": int(args.negatives),
        "modalities": [name for name, _ in feature_paths],
        "per_modality": per_modality,
        "fused": metrics,
        "combined": combined,
        "gate": gate,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("RATING_SIGNAL_SUMMARY " + json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
