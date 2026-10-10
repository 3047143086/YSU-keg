"""Build a fixed-support edge-deletion cache and inspect training evidence only."""

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Non-finite diagnostic values")
    return {"count": int(values.size), "mean": float(values.mean()),
            "std": float(values.std()), "min": float(values.min()),
            "median": float(np.median(values)), "max": float(values.max()),
            "p10": float(np.quantile(values, 0.1)),
            "p90": float(np.quantile(values, 0.9))}


def sample_negatives(users, edge_keys, user_degree, n_items, seed=2026):
    """Filter full observed TRAINING histories; never inspect held-out labels."""
    if np.any(user_degree[users] >= n_items):
        raise ValueError("A training user has no unobserved negative item")
    generator = np.random.default_rng(seed)
    negatives = generator.integers(0, n_items, size=len(users), dtype=np.int64)
    pending = np.arange(len(users))
    while pending.size:
        query_keys = users[pending] * n_items + negatives[pending]
        positions = np.searchsorted(edge_keys, query_keys)
        inside = positions < len(edge_keys)
        collisions = np.zeros(len(pending), dtype=bool)
        collisions[inside] = edge_keys[positions[inside]] == query_keys[inside]
        pending = pending[collisions]
        if pending.size:
            negatives[pending] = generator.integers(0, n_items, size=len(pending))
    return negatives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("baby", "sports"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.bridge_root.resolve()
    output = args.output_dir.resolve()
    bundle = Path(__file__).resolve().parent
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.chdir(root / "src")
    sys.path[:0] = [str(bundle), str(root / "src")]
    import torch
    from scipy import sparse
    from evidence import build_counterfactual_cache
    from models.bridge import BRIDGE
    from utils.configurator import Config
    from utils.dataset import RecDataset

    torch.set_num_threads(4)
    config = Config("BRIDGE", args.dataset,
                    {"use_gpu": False, "gpu_id": "", "seed": [2026],
                     "save_recommended_topk": False})
    for key in config["hyper_parameters"]:
        value = config[key]
        if isinstance(value, (tuple, list)):
            if len(value) != 1:
                raise ValueError("A single frozen value is required for " + key)
            config[key] = value[0]
    settings = SimpleNamespace(
        behavior_topk=int(BRIDGE._cfg(config, "behavior_topk", 1000)),
        behavior_sim_method=str(BRIDGE._cfg(config, "behavior_sim_method", "cosine")),
        behavior_aggregation=str(BRIDGE._cfg(config, "behavior_aggregation", "sum_sqrt")),
        behavior_score_norm=str(BRIDGE._cfg(config, "behavior_score_norm", "z")))
    if (settings.behavior_topk != 1000 or settings.behavior_sim_method != "cosine"
            or settings.behavior_aggregation != "sum_sqrt" or settings.behavior_score_norm != "z"):
        raise ValueError("The frozen experiment requires cosine/topk1000/sum_sqrt/global-z evidence")

    # RecDataset reads the supplied split file to preserve the published global
    # ID space. Only rows labelled training enter any graph, score, or diagnostic.
    dataset = RecDataset(config)
    train_df = dataset.df.loc[dataset.df[config["inter_splitting_label"]] == 0]
    users = train_df[dataset.uid_field].to_numpy(dtype=np.int64)
    items = train_df[dataset.iid_field].to_numpy(dtype=np.int64)
    train = sparse.csr_matrix((np.ones(len(users), dtype=np.float32), (users, items)),
                              shape=(dataset.user_num, dataset.item_num))
    train.sum_duplicates()
    train.sort_indices()
    if np.any(train.data != 1):
        raise ValueError("Duplicate training interactions violate the binary evidence contract")
    edge_users, edge_items = train.nonzero()
    order = np.lexsort((edge_items, edge_users))
    edge_users = np.asarray(edge_users, dtype=np.int64)[order]
    edge_items = np.asarray(edge_items, dtype=np.int64)[order]
    edge_keys = edge_users * train.shape[1] + edge_items
    user_degree = np.asarray(train.sum(axis=1)).ravel()
    negative = sample_negatives(edge_users, edge_keys, user_degree, train.shape[1])
    if not len(edge_keys):
        raise ValueError("No training interactions")
    inter_path = Path(dataset.dataset_path) / config["inter_file_name"]
    del train_df, dataset, users, items
    gc.collect()

    def emit(message):
        print(json.dumps({"stage": message, "elapsed_seconds": time.monotonic() - started}), flush=True)

    # Invoke the real baseline method without initializing its encoder, GPU, or
    # graph-cache writers. Keep only queried values before building our cache.
    emit("checking_original_BRIDGE_behavior_method")
    adapter = SimpleNamespace(inter_matrix=lambda form="csr": train)
    reference, _, _ = BRIDGE._build_behavior_score_matrix(settings, adapter)
    reference_positive = reference[edge_users, edge_items].copy()
    reference_negative = reference[edge_users, negative].copy()
    del reference
    gc.collect()
    emit("building_fixed_support_edge_deletion_cache")
    cache = build_counterfactual_cache(
        train, topk=settings.behavior_topk,
        progress=lambda completed, total: emit("cache_users_%d_of_%d" % (completed, total)))
    if not np.array_equal(cache.edge_keys, edge_keys):
        raise RuntimeError("Counterfactual edge ordering changed")
    pairs = {mode: cache.get_pair(edge_users, edge_items, negative, mode=mode)
             for mode in ("original", "history", "edge")}
    positive_error = np.abs(pairs["original"][0] - reference_positive)
    negative_error = np.abs(pairs["original"][1] - reference_negative)
    maximum_error = float(max(positive_error.max(), negative_error.max()))
    baseline_parity = {"method": "BRIDGE._build_behavior_score_matrix",
                       "sample": "all training positives and one fixed unobserved negative per edge",
                       "positive_max_abs_difference": float(positive_error.max()),
                       "negative_max_abs_difference": float(negative_error.max()),
                       "tolerance": 0.0002, "passed": maximum_error <= 0.0002}

    raw_original = cache.raw_scores[edge_users, edge_items].astype(np.float64)
    raw_deleted = cache.edge_target_raw
    raw_decrease = raw_original - raw_deleted
    original_edge_z_decrease = pairs["original"][0] - pairs["edge"][0]
    history_edge_z_decrease = pairs["history"][0] - pairs["edge"][0]
    mean_raw_relative_decrease = float(raw_decrease.mean() / max(raw_original.mean(), 1e-12))
    fraction_z_decrease = float(np.mean(history_edge_z_decrease >= 0.1))
    gate = {"predeclared": True,
            "raw_mean_relative_decrease_definition": "(mean(original positive raw) - mean(edge-deleted positive raw)) / mean(original positive raw)",
            "raw_mean_relative_decrease_minimum": 0.10,
            "raw_mean_relative_decrease": mean_raw_relative_decrease,
            "z_decrease_definition": "history-only positive z minus edge-deleted positive z",
            "z_decrease_per_edge_minimum": 0.10,
            "fraction_edges_with_z_decrease_minimum": 0.25,
            "fraction_edges_with_z_decrease": fraction_z_decrease,
            "mechanism_gate_passed": mean_raw_relative_decrease >= 0.10 and fraction_z_decrease >= 0.25}
    formal_allowed = bool(gate["mechanism_gate_passed"] and baseline_parity["passed"])
    arms = {mode: {"positive_z": describe(positive), "negative_z": describe(negative_z),
                   "positive_minus_negative_z": describe(positive - negative_z)}
            for mode, (positive, negative_z) in pairs.items()}
    observed_user_counts = np.bincount(edge_users, minlength=train.shape[0])
    mean_user_z = np.bincount(edge_users, weights=history_edge_z_decrease,
                             minlength=train.shape[0]) / np.maximum(observed_user_counts, 1)
    strata = []
    for name, lower, upper in (("degree_1_to_5", 1, 5), ("degree_6_to_10", 6, 10),
                               ("degree_11_to_20", 11, 20), ("degree_over_20", 21, None)):
        user_mask = user_degree >= lower
        if upper is not None:
            user_mask &= user_degree <= upper
        edge_mask = user_mask[edge_users]
        strata.append({"group": name, "users": int(user_mask.sum()),
                       "training_edges": int(edge_mask.sum()),
                       "edge_weighted_history_minus_edge_positive_z": describe(history_edge_z_decrease[edge_mask]),
                       "user_weighted_mean_history_minus_edge_positive_z": describe(mean_user_z[user_mask]),
                       "raw_positive_decrease": describe(raw_decrease[edge_mask])})

    emit("writing_training_cache")
    cache_path = output / "cache.npz"
    cache.save(cache_path)
    files = [Path(__file__).resolve(), bundle / "evidence.py", inter_path,
             root / "src/models/bridge.py", root / "src/utils/dataset.py",
             root / "src/utils/configurator.py", root / "src/configs/overall.yaml",
             root / "src/configs/model/BRIDGE.yaml",
             root / "src/configs/dataset" / (args.dataset + ".yaml")]
    report = {"dataset": args.dataset, "seed": 2026,
              "shape": list(train.shape), "training_edges": int(train.nnz),
              "negative_sampling": "one uniform item outside the complete training history per training edge",
              "official_validation_evaluated": False, "official_test_evaluated": False,
              "held_out_labels_used_for_diagnostic_or_negative_filtering": False,
              "legacy_reader_loads_full_split_file_for_global_id_space": True,
              "encoder_instantiated": False, "encoder_graph_modified": False,
              "fixed_knn_support": True, "knn_support_reselected_after_edge_deletion": False,
              "inference_modified": False,
              "interpretation": "Training evidence mismatch diagnostic; not a performance or generalization result. Edge deletion is exact conditional on the original KNN support, with floating-point tolerance; encoder message passing is untouched.",
              "baseline_parity": baseline_parity, "mechanism_gate": gate,
              "formal_training_allowed": formal_allowed,
              "arms": arms, "strata": strata,
              "raw_positive_original": describe(raw_original),
              "raw_positive_edge_deleted": describe(raw_deleted),
              "raw_positive_self_vote": describe(cache.self_vote_raw),
              "raw_positive_decrease": describe(raw_decrease),
              "original_minus_edge_positive_z": describe(original_edge_z_decrease),
              "history_minus_edge_positive_z": describe(history_edge_z_decrease),
              "user_weighted_history_minus_edge_positive_z": describe(mean_user_z[observed_user_counts > 0]),
              "cache": {"path": str(cache_path), "sha256": digest(cache_path),
                        "size_bytes": cache_path.stat().st_size},
              "input_sha256": {str(path): digest(path) for path in files},
              "versions": {"numpy": np.__version__, "torch": torch.__version__},
              "elapsed_seconds": time.monotonic() - started}
    with (output / "prepare.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    emit("completed_formal_training_allowed_" + str(formal_allowed))
    if not baseline_parity["passed"]:
        raise RuntimeError("Baseline evidence equivalence failed; inspect prepare.json before any training")


if __name__ == "__main__":
    main()
