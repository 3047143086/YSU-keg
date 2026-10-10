"""Run the preregistered C3E multimodal evidence diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from utils.c3e import (
    EvidenceCalibrator,
    deterministic_user_folds,
    diagnostic_gate,
    make_degree_edges,
    topq_mean,
)
from utils.rating_signal import event_pairwise_auc, l2_normalize_rows, mean_and_ci


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
    parser.add_argument("--degree-bins", type=int, default=4)
    parser.add_argument("--shrinkage", type=float, default=200.0)
    parser.add_argument("--pair-batch-size", type=int, default=8192)
    parser.add_argument("--event-chunk-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def build_histories(users, items, n_users):
    histories = [[] for _ in range(int(n_users))]
    for user, item in zip(users, items):
        histories[int(user)].append(int(item))
    return [np.unique(values).astype(np.int64) for values in histories]


def sample_unobserved_negatives(histories, positives, users, n_items, count, seed):
    rng = np.random.default_rng(int(seed))
    output = np.empty((len(users), int(count)), dtype=np.int64)
    for row, (user, positive) in enumerate(zip(users, positives)):
        forbidden = set(histories[int(user)].tolist())
        forbidden.add(int(positive))
        chosen = []
        needed = int(count)
        while len(chosen) < needed:
            candidates = rng.integers(0, n_items, size=max(16, 3 * needed))
            chosen.extend(int(x) for x in candidates if int(x) not in forbidden)
        output[row] = chosen[:needed]
    return output


def sample_co_consumption_pairs(
    histories,
    calibration_users,
    max_pairs_per_user,
    max_pairs,
    seed,
):
    rng = np.random.default_rng(int(seed))
    left = []
    right = []
    per_user = int(max_pairs_per_user)
    for user in calibration_users:
        items = histories[int(user)]
        size = int(items.size)
        if size < 2:
            continue
        total = size * (size - 1) // 2
        if total <= per_user:
            first, second = np.triu_indices(size, k=1)
            selected = np.arange(total)
        else:
            pairs = set()
            for _ in range(per_user * 4):
                first_index, second_index = rng.integers(0, size, size=2)
                if first_index == second_index:
                    continue
                pair = tuple(sorted((int(first_index), int(second_index))))
                pairs.add(pair)
                if len(pairs) == per_user:
                    break
            if len(pairs) < per_user:
                for pair in zip(*np.triu_indices(size, k=1)):
                    pairs.add((int(pair[0]), int(pair[1])))
                    if len(pairs) == per_user:
                        break
            ordered = sorted(pairs)
            first = np.asarray([pair[0] for pair in ordered], dtype=np.int64)
            second = np.asarray([pair[1] for pair in ordered], dtype=np.int64)
            selected = np.arange(len(ordered))
        left.extend(items[first[selected]].tolist())
        right.extend(items[second[selected]].tolist())

    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    if left.size > int(max_pairs):
        selected = rng.choice(left.size, size=int(max_pairs), replace=False)
        left = left[selected]
        right = right[selected]
    return left, right


def sample_degree_matched_pairs(item_degrees, degree_edges, left, right, seed):
    rng = np.random.default_rng(int(seed))
    degree_bins = np.searchsorted(
        degree_edges, np.log1p(np.asarray(item_degrees)), side="right"
    )
    observed = np.flatnonzero(np.asarray(item_degrees) > 0)
    pools = [
        np.flatnonzero(degree_bins == degree_bin)
        for degree_bin in range(len(degree_edges) + 1)
    ]
    pools = [pool if pool.size else observed for pool in pools]

    def draw(reference):
        reference_bins = degree_bins[reference]
        sampled = np.empty(reference.size, dtype=np.int64)
        for degree_bin, pool in enumerate(pools):
            mask = np.flatnonzero(reference_bins == degree_bin)
            if mask.size:
                sampled[mask] = pool[rng.integers(0, pool.size, size=mask.size)]
        return sampled

    negative_left = draw(left)
    negative_right = draw(right)
    collision = negative_left == negative_right
    for _ in range(4):
        if not collision.any():
            break
        negative_right[collision] = draw(right[collision])
        collision = negative_left == negative_right
    if collision.any():
        for index in np.flatnonzero(collision):
            pool = pools[degree_bins[right[index]]]
            alternatives = pool[pool != negative_left[index]]
            if alternatives.size == 0:
                alternatives = observed[observed != negative_left[index]]
            negative_right[index] = alternatives[0]
    return negative_left, negative_right


class FeatureScorer:
    def __init__(self, features, device, pair_batch_size):
        normalized, _ = l2_normalize_rows(features)
        self.features = normalized
        self.batch_size = int(pair_batch_size)
        self.device = "cpu"
        self.tensor = None
        if device != "cpu":
            try:
                import torch

                if torch.cuda.is_available():
                    self.device = "cuda"
                    self.tensor = torch.from_numpy(normalized).cuda()
                elif device == "cuda":
                    raise RuntimeError("CUDA was requested but is unavailable")
            except ImportError:
                if device == "cuda":
                    raise

    def pair_cosine(self, left, right):
        left = np.asarray(left, dtype=np.int64).reshape(-1)
        right = np.asarray(right, dtype=np.int64).reshape(-1)
        if left.size != right.size:
            raise ValueError("left and right pair arrays must have equal size")
        output = np.empty(left.size, dtype=np.float32)
        if self.device == "cuda":
            import torch

            with torch.no_grad():
                for start in range(0, left.size, self.batch_size):
                    stop = min(start + self.batch_size, left.size)
                    left_index = torch.as_tensor(left[start:stop], device="cuda")
                    right_index = torch.as_tensor(right[start:stop], device="cuda")
                    values = (
                        self.tensor[left_index] * self.tensor[right_index]
                    ).sum(dim=1)
                    output[start:stop] = values.cpu().numpy()
        else:
            for start in range(0, left.size, self.batch_size):
                stop = min(start + self.batch_size, left.size)
                output[start:stop] = np.einsum(
                    "ij,ij->i",
                    self.features[left[start:stop]],
                    self.features[right[start:stop]],
                    optimize=True,
                )
        return output

    def exposure_scores(self, operator, users, positives, negatives):
        profiles, _ = l2_normalize_rows(operator @ self.features)
        positive = np.empty(len(users), dtype=np.float32)
        negative = np.empty(negatives.shape, dtype=np.float32)
        event_batch_size = max(1, self.batch_size // negatives.shape[1])
        for start in range(0, len(users), event_batch_size):
            stop = min(start + event_batch_size, len(users))
            batch_profiles = profiles[users[start:stop]]
            positive[start:stop] = np.einsum(
                "ij,ij->i",
                batch_profiles,
                self.features[positives[start:stop]],
                optimize=True,
            )
            negative[start:stop] = np.einsum(
                "ij,ikj->ik",
                batch_profiles,
                self.features[negatives[start:stop]],
                optimize=True,
            )
        return positive.astype(np.float32), negative.astype(np.float32)


def candidate_history_scores(
    scorer,
    calibrator,
    event_indices,
    users,
    positives,
    negatives,
    histories,
    item_degrees,
    top_q,
    event_chunk_size,
):
    positive_raw = np.zeros(len(event_indices), dtype=np.float32)
    negative_raw = np.zeros((len(event_indices), negatives.shape[1]), dtype=np.float32)
    positive_global = np.zeros(len(event_indices), dtype=np.float32)
    negative_global = np.zeros_like(negative_raw)
    positive_c3e = np.zeros(len(event_indices), dtype=np.float32)
    negative_c3e = np.zeros_like(negative_raw)
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
        similarity = scorer.pair_cosine(left, right)
        global_evidence = calibrator.transform(
            similarity, left, right, item_degrees, degree_conditioned=False
        )
        c3e_evidence = calibrator.transform(
            similarity, left, right, item_degrees, degree_conditioned=True
        )

        cursor = 0
        for local_index, history_size in enumerate(history_sizes):
            value_count = candidate_count * history_size
            shape = (candidate_count, history_size)
            raw_score = topq_mean(similarity[cursor : cursor + value_count].reshape(shape), top_q)
            global_score = topq_mean(
                global_evidence[cursor : cursor + value_count].reshape(shape), top_q
            )
            c3e_score = topq_mean(
                c3e_evidence[cursor : cursor + value_count].reshape(shape), top_q
            )
            positive_raw[chunk_start + local_index] = raw_score[0]
            negative_raw[chunk_start + local_index] = raw_score[1:]
            positive_global[chunk_start + local_index] = global_score[0]
            negative_global[chunk_start + local_index] = global_score[1:]
            positive_c3e[chunk_start + local_index] = c3e_score[0]
            negative_c3e[chunk_start + local_index] = c3e_score[1:]
            cursor += value_count
    return {
        "raw_topq": (positive_raw, negative_raw),
        "global_calibrated": (positive_global, negative_global),
        "c3e": (positive_c3e, negative_c3e),
    }


def score_summary(positive, negative):
    return mean_and_ci(event_pairwise_auc(positive, negative))


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
    feature_paths = [
        (name, dataset_dir / filename)
        for name, filename in (
            ("image", "image_feat.npy"),
            ("text", "text_feat.npy"),
        )
        if (dataset_dir / filename).exists()
    ]
    if not feature_paths:
        raise FileNotFoundError(f"No image/text features found under {dataset_dir}")

    n_users = int(frame["userID"].max()) + 1
    first_features = np.load(feature_paths[0][1], mmap_mode="r")
    n_items = int(first_features.shape[0])
    del first_features
    train_users = train["userID"].to_numpy(np.int64)
    train_items = train["itemID"].to_numpy(np.int64)
    valid_users = valid["userID"].to_numpy(np.int64)
    valid_items = valid["itemID"].to_numpy(np.int64)
    histories = build_histories(train_users, train_items, n_users)
    negatives = sample_unobserved_negatives(
        histories,
        valid_items,
        valid_users,
        n_items,
        args.negatives,
        args.seed,
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

    all_users = np.arange(n_users, dtype=np.int64)
    user_fold = deterministic_user_folds(all_users, args.folds, args.seed)
    calibration_data = []
    for fold in range(args.folds):
        calibration_users = all_users[user_fold != fold]
        calibration_mask = user_fold[train_users] != fold
        item_degrees = np.bincount(
            train_items[calibration_mask], minlength=n_items
        ).astype(np.float32)
        degree_edges = make_degree_edges(item_degrees, args.degree_bins)
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
        calibration_data.append(
            {
                "fold": fold,
                "event_indices": np.flatnonzero(user_fold[valid_users] == fold),
                "item_degrees": item_degrees,
                "degree_edges": degree_edges,
                "positive_left": positive_left,
                "positive_right": positive_right,
                "negative_left": negative_left,
                "negative_right": negative_right,
            }
        )

    score_names = ("exposure", "raw_topq", "global_calibrated", "c3e")
    fused = {
        name: {
            "positive": np.zeros(len(valid), dtype=np.float32),
            "negative": np.zeros(negatives.shape, dtype=np.float32),
        }
        for name in score_names
    }
    per_modality = {}
    calibrator_summaries = {}
    selected_device = None
    for modality, path in feature_paths:
        features = np.asarray(np.load(path), dtype=np.float32)
        scorer = FeatureScorer(features, args.device, args.pair_batch_size)
        selected_device = scorer.device
        exposure_positive, exposure_negative = scorer.exposure_scores(
            exposure_operator, valid_users, valid_items, negatives
        )
        modality_scores = {
            name: {
                "positive": np.zeros(len(valid), dtype=np.float32),
                "negative": np.zeros(negatives.shape, dtype=np.float32),
            }
            for name in score_names
        }
        modality_scores["exposure"]["positive"] = exposure_positive
        modality_scores["exposure"]["negative"] = exposure_negative
        calibrator_summaries[modality] = []

        for fold_data in calibration_data:
            positive_similarity = scorer.pair_cosine(
                fold_data["positive_left"], fold_data["positive_right"]
            )
            negative_similarity = scorer.pair_cosine(
                fold_data["negative_left"], fold_data["negative_right"]
            )
            calibrator = EvidenceCalibrator.fit(
                positive_similarity,
                negative_similarity,
                fold_data["positive_left"],
                fold_data["positive_right"],
                fold_data["negative_left"],
                fold_data["negative_right"],
                fold_data["item_degrees"],
                fold_data["degree_edges"],
                n_similarity_bins=args.similarity_bins,
                shrinkage=args.shrinkage,
            )
            event_indices = fold_data["event_indices"]
            fold_scores = candidate_history_scores(
                scorer,
                calibrator,
                event_indices,
                valid_users,
                valid_items,
                negatives,
                histories,
                fold_data["item_degrees"],
                args.top_q,
                args.event_chunk_size,
            )
            for name, (positive, negative) in fold_scores.items():
                modality_scores[name]["positive"][event_indices] = positive
                modality_scores[name]["negative"][event_indices] = negative
            calibrator_summaries[modality].append(
                {
                    "fold": int(fold_data["fold"]),
                    "calibration_pairs_per_class": int(positive_similarity.size),
                    **calibrator.summary(),
                }
            )

        per_modality[modality] = {}
        for name, scores in modality_scores.items():
            per_modality[modality][name] = score_summary(
                scores["positive"], scores["negative"]
            )
            fused[name]["positive"] += scores["positive"]
            fused[name]["negative"] += scores["negative"]
        del scorer, features

    modality_count = float(len(feature_paths))
    event_auc = {}
    metrics = {}
    for name, scores in fused.items():
        scores["positive"] /= modality_count
        scores["negative"] /= modality_count
        event_auc[name] = event_pairwise_auc(
            scores["positive"], scores["negative"]
        )
        metrics[name] = mean_and_ci(event_auc[name])

    delta_exposure = mean_and_ci(event_auc["c3e"] - event_auc["exposure"])
    delta_raw = mean_and_ci(event_auc["c3e"] - event_auc["raw_topq"])
    fold_metrics = []
    fold_deltas = []
    for fold in range(args.folds):
        mask = user_fold[valid_users] == fold
        c3e_fold = event_auc["c3e"][mask]
        exposure_fold = event_auc["exposure"][mask]
        delta = mean_and_ci(c3e_fold - exposure_fold)
        fold_deltas.append(delta["mean"])
        fold_metrics.append(
            {
                "fold": fold,
                "events": int(mask.sum()),
                "c3e_auc": mean_and_ci(c3e_fold),
                "exposure_auc": mean_and_ci(exposure_fold),
                "c3e_delta_vs_exposure": delta,
            }
        )
    gate = diagnostic_gate(
        c3e_auc=metrics["c3e"]["mean"],
        exposure_delta=delta_exposure["mean"],
        exposure_delta_ci_low=delta_exposure["ci95_low"],
        raw_topq_delta=delta_raw["mean"],
        raw_topq_delta_ci_low=delta_raw["ci95_low"],
        fold_deltas=fold_deltas,
    )
    output = {
        "method": "C3E",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "device": selected_device,
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "negative_samples_per_event": int(args.negatives),
        "folds": int(args.folds),
        "top_q": int(args.top_q),
        "modalities": [name for name, _ in feature_paths],
        "metrics": metrics,
        "paired_delta_vs_exposure": delta_exposure,
        "paired_delta_vs_raw_topq": delta_raw,
        "fold_metrics": fold_metrics,
        "per_modality": per_modality,
        "calibrators": calibrator_summaries,
        "gate": gate,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("C3E_SUMMARY " + json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
