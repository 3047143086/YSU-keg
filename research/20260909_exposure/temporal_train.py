"""Lean chronological Baby feasibility screen using a frozen global feature bank."""

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

VARIANTS = ("long_mean", "recent3", "learned_two_scale")
METRICS = ("recall@10", "ndcg@10", "recall@20", "ndcg@20", "recall@200")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_interactions(path):
    """Read only the columns needed from a tab-separated interaction file."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"userID", "itemID", "timestamp"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("baby.inter must contain userID, itemID, timestamp")
        rows = [(int(row["userID"]), int(row["itemID"]), int(float(row["timestamp"])))
                for row in reader]
    if not rows:
        raise ValueError("empty interaction file")
    return rows


def chronological_split(rows):
    """Return train histories, calibration targets, and test targets per user."""
    grouped = {}
    for user, item, timestamp in rows:
        grouped.setdefault(user, []).append((timestamp, item))
    train, calibration, test = {}, {}, {}
    for user, events in grouped.items():
        events = sorted(events, key=lambda value: (value[0], value[1]))
        if len(events) < 3:
            raise ValueError("every user needs at least three events")
        train[user] = tuple(item for _, item in events[:-2])
        calibration[user] = events[-2][1]
        test[user] = events[-1][1]
        if calibration[user] in train[user] or test[user] in train[user] or test[user] == calibration[user]:
            raise ValueError("duplicate item makes chronological target impossible to evaluate")
    return train, calibration, test


def context_for(user, train, calibration, phase):
    if phase == "train":
        return tuple(train[user])
    if phase == "calibration":
        return tuple(train[user])
    if phase == "test":
        return tuple(train[user]) + (calibration[user],)
    raise ValueError("unknown phase: " + phase)


def summary_vector(history, item_features, variant):
    if not history:
        return np.zeros(item_features.shape[1], dtype=np.float32)
    selected = history if variant == "long_mean" else history[-3:]
    return item_features[np.asarray(selected, dtype=np.int64)].mean(axis=0).astype(np.float32)


def build_training_examples(train, item_features, variant):
    examples = []
    for user in sorted(train):
        history = []
        for item in train[user]:
            if history:
                examples.append((user, summary_vector(tuple(history), item_features, variant), item,
                                 frozenset(history)))
            history.append(item)
    if not examples:
        raise ValueError("no nonempty chronological training contexts")
    return examples


def metric_rows(scores, targets, excluded):
    rows = []
    items = np.arange(scores.shape[1])
    for row, target, forbidden in zip(scores, targets, excluded):
        values = np.asarray(row, dtype=np.float64).copy()
        values[list(forbidden)] = -np.inf
        if not np.isfinite(values[target]):
            raise ValueError("target is excluded or nonfinite")
        order = np.lexsort((items, -values))
        rank = int(np.flatnonzero(order == target)[0])
        rows.append([float(rank < 10), float(1.0 / np.log2(rank + 2) if rank < 10 else 0.0),
                     float(rank < 20), float(1.0 / np.log2(rank + 2) if rank < 20 else 0.0),
                     float(rank < 200)])
    return np.asarray(rows, dtype=np.float64)


def metric_means(values):
    return {name: float(value) for name, value in zip(METRICS, np.asarray(values).mean(axis=0))}


def fused_bank(path):
    bank = np.load(path, mmap_mode="r")
    if bank.ndim != 3 or bank.shape[1] != 2 or not np.isfinite(bank).all():
        raise ValueError("global_bank must be finite [items, 2, dimensions]")
    fused = np.asarray(bank, dtype=np.float32).mean(axis=1)
    fused /= np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-8)
    return fused.astype(np.float32)


def evaluate(model, vectors, target_map, train, calibration, phase, device):
    import torch
    users = sorted(target_map)
    contexts = [context_for(u, train, calibration, phase) for u in users]
    queries = np.stack([summary_vector(c, vectors, model.variant if model.variant != "learned_two_scale" else "long_mean")
                        for c in contexts]).astype(np.float32)
    recent = np.stack([summary_vector(c, vectors, "recent3") for c in contexts]).astype(np.float32)
    with torch.no_grad():
        scores = model(torch.from_numpy(queries).to(device), torch.from_numpy(recent).to(device)).cpu().numpy()
    excluded = [set(c) for c in contexts]
    return users, metric_rows(scores, [target_map[u] for u in users], excluded), scores


def train(args):
    import torch
    from torch import nn
    from torch.nn import functional as F

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise RuntimeError("refusing to overwrite existing result")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.manual_seed_all(args.seed)

    rows = read_interactions(args.interactions)
    train_histories, calibration_targets, test_targets = chronological_split(rows)
    if args.smoke:
        selected = sorted(train_histories)[:128]
        train_histories = {u: train_histories[u] for u in selected}
        calibration_targets = {u: calibration_targets[u] for u in selected}
        test_targets = {u: test_targets[u] for u in selected}
    vectors = fused_bank(args.feature_bank)
    max_item = max(max(items) for items in train_histories.values())
    max_item = max(max_item, max(calibration_targets.values()), max(test_targets.values()))
    if max_item >= len(vectors) or min(min(items) for items in train_histories.values()) < 0:
        raise ValueError("interaction item IDs do not fit global_bank")
    examples = build_training_examples(train_histories, vectors, args.variant)

    class Ranker(nn.Module):
        def __init__(self):
            super().__init__()
            width = vectors.shape[1]
            self.variant = args.variant
            self.query_long = nn.Linear(width, args.hidden, bias=False)
            self.query_recent = nn.Linear(width, args.hidden, bias=False)
            self.item = nn.Linear(width, args.hidden, bias=False)
            self.mix = nn.Parameter(torch.zeros(2))
            self.register_buffer("items", torch.from_numpy(vectors), persistent=False)

        def forward(self, long_query, recent_query, item_ids=None):
            item = self.items if item_ids is None else self.items[item_ids]
            item = F.normalize(self.item(item), dim=-1)
            long_query = F.normalize(self.query_long(long_query), dim=-1)
            if item_ids is not None:
                long_score = (long_query * item).sum(dim=-1)
            else:
                long_score = long_query @ item.T
            if self.variant == "long_mean":
                return long_score * 10.0
            recent_query = F.normalize(self.query_recent(recent_query), dim=-1)
            if item_ids is not None:
                recent_score = (recent_query * item).sum(dim=-1)
            else:
                recent_score = recent_query @ item.T
            if self.variant == "recent3":
                return recent_score * 10.0
            weights = F.softmax(self.mix, dim=0)
            return (weights[0] * long_score + weights[1] * recent_score) * 10.0

    model = Ranker().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    all_items = np.arange(len(vectors), dtype=np.int64)
    best_state, best_key, best_epoch = None, None, 0
    started = time.monotonic()
    atomic_json(output / "progress.json", {"status": "training", "seed": args.seed, "variant": args.variant})
    epochs = 2 if args.smoke else args.epochs
    for epoch in range(1, epochs + 1):
        rng = np.random.default_rng(args.seed + epoch)
        order = rng.permutation(len(examples))
        losses = []
        model.train()
        for start in range(0, len(order), args.batch_size):
            batch = [examples[i] for i in order[start:start + args.batch_size]]
            long_q = torch.from_numpy(np.stack([x[1] for x in batch])).to(device)
            recent_q = torch.from_numpy(np.stack([summary_vector(tuple(x[3]), vectors, "recent3") for x in batch])).to(device)
            positives = np.asarray([x[2] for x in batch], dtype=np.int64)
            negatives = []
            for example in batch:
                forbidden = set(example[3]) | {example[2]}
                choices = np.setdiff1d(all_items, np.fromiter(forbidden, dtype=np.int64), assume_unique=False)
                negatives.append(rng.choice(choices, size=args.negatives, replace=len(choices) < args.negatives))
            negative_ids = torch.from_numpy(np.asarray(negatives, dtype=np.int64)).to(device)
            positive_ids = torch.from_numpy(positives).to(device)
            positive_score = model(long_q, recent_q, positive_ids)
            negative_score = model(long_q[:, None, :].expand(-1, args.negatives, -1).reshape(-1, vectors.shape[1]),
                                   recent_q[:, None, :].expand(-1, args.negatives, -1).reshape(-1, vectors.shape[1]),
                                   negative_ids.reshape(-1)).reshape(len(batch), args.negatives)
            loss = F.softplus(-(positive_score[:, None] - negative_score)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        _, calibration_values, _ = evaluate(model, vectors, calibration_targets, train_histories,
                                            calibration_targets, "calibration", device)
        calibration_metrics = metric_means(calibration_values)
        key = (calibration_metrics["ndcg@20"], calibration_metrics["recall@20"],
               calibration_metrics["ndcg@10"], -epoch)
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        atomic_json(output / "progress.json", {"status": "training", "epoch": epoch,
                    "epochs": epochs, "loss": float(np.mean(losses)),
                    "calibration_metrics": calibration_metrics})
    model.load_state_dict(best_state)
    model.eval()
    _, calibration_values, _ = evaluate(model, vectors, calibration_targets, train_histories,
                                        calibration_targets, "calibration", device)
    if args.smoke:
        result = {"status": "smoke_passed", "seed": args.seed, "variant": args.variant,
                  "best_epoch": best_epoch, "parameters": sum(p.numel() for p in model.parameters()),
                  "calibration_metrics": metric_means(calibration_values),
                  "test_evaluated": False, "users": len(train_histories),
                  "training_examples": len(examples), "negative_samples": args.negatives,
                  "elapsed_seconds": time.monotonic() - started,
                  "provenance": {"interactions": str(args.interactions), "feature_bank": str(args.feature_bank),
                                 "interactions_sha256": sha256(args.interactions),
                                 "feature_bank_sha256": sha256(args.feature_bank)},
                  "split": "all but last two train; penultimate calibration; last test",
                  "scope": "smoke only; no test evaluation; feasibility screen"}
        atomic_json(output / "result.json", result)
        return result
    test_users, test_values, scores = evaluate(model, vectors, test_targets, train_histories,
                                                calibration_targets, "test", device)
    test_metrics = metric_means(test_values)
    users = sorted(test_targets)
    test_contexts = [context_for(u, train_histories, calibration_targets, "test") for u in users]
    top20 = np.stack([np.lexsort((np.arange(scores.shape[1]), -np.where(np.isin(np.arange(scores.shape[1]), list(c)), -np.inf, scores[i])))[:20]
                      for i, c in enumerate(test_contexts)])
    np.savez_compressed(output / "per_user.npz", users=np.asarray(test_users),
                        targets=np.asarray([test_targets[u] for u in test_users]), metrics=test_values,
                        top20=top20, metric_names=np.asarray(METRICS))
    result = {"status": "smoke_passed" if args.smoke else "completed", "seed": args.seed,
              "variant": args.variant, "best_epoch": best_epoch, "parameters": sum(p.numel() for p in model.parameters()),
              "calibration_metrics": metric_means(calibration_values), "test_metrics": test_metrics,
              "test_evaluated": not args.smoke, "dev_metrics": test_metrics,
              "users": len(train_histories), "training_examples": len(examples),
              "negative_samples": args.negatives, "elapsed_seconds": time.monotonic() - started,
              "provenance": {"interactions": str(args.interactions), "feature_bank": str(args.feature_bank),
                             "interactions_sha256": sha256(args.interactions), "feature_bank_sha256": sha256(args.feature_bank)},
              "split": "all but last two train; penultimate calibration; last test",
              "scope": "feasibility screen; internal result, not a novelty or publication claim"}
    atomic_json(output / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--feature-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--negatives", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.negatives, args.hidden, args.threads) < 1:
        raise ValueError("training budgets must be positive")
    result = train(args)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
