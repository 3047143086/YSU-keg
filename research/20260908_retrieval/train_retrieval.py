"""Bounded, train-internal candidate-complementarity screening.

Independent must finish first for each seed. It publishes a frozen shared
ranker under root/shared/<seed>/; every other variant reads that same ranker.
No official validation/test files are read. Results are internal screening.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np


VARIANTS = ("independent", "hard", "hard_positive", "clear", "conditional", "shuffled",
            "base_user", "feedback")
METRIC_NAMES = ("recall@10", "ndcg@10", "recall@20", "ndcg@20", "recall@200")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_rows(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def matched_context_permutation(degrees, candidate_popularity, seed=20260908):
    degree_bins = np.searchsorted(np.quantile(degrees, np.arange(0.1, 1.0, 0.1)), degrees)
    popularity_bins = np.searchsorted(np.quantile(candidate_popularity, np.arange(0.1, 1.0, 0.1)),
                                     candidate_popularity)
    labels = degree_bins * 10 + popularity_bins
    groups, singletons = [], []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if len(members) < 2:
            singletons.extend(members.tolist())
        else:
            groups.append(members)
    if len(singletons) == 1:
        if not groups:
            raise ValueError("At least two users required for mismatched-context control")
        groups[0] = np.concatenate((groups[0], np.asarray(singletons)))
    elif singletons:
        groups.append(np.asarray(singletons))
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(degrees))
    for members in groups:
        members = rng.permutation(members)
        permutation[members] = np.roll(members, 1)
    if len(np.unique(permutation)) != len(permutation) or np.any(permutation == np.arange(len(permutation))):
        raise AssertionError("Mismatched context must be a fixed derangement")
    return permutation


def standardize_scores(scores):
    finite = np.isfinite(scores)
    counts = finite.sum(axis=1, keepdims=True)
    if np.any(counts < 200):
        raise ValueError("Fewer than 200 finite eligible catalog items")
    safe = np.where(finite, scores, 0.0)
    means = safe.sum(axis=1, keepdims=True) / counts
    centered = np.where(finite, safe - means, 0.0)
    stds = np.sqrt((centered * centered).sum(axis=1, keepdims=True) / counts)
    result = centered / np.maximum(stds, 1e-6)
    result[~finite] = -np.inf
    return result.astype(np.float32)


def topk_sorted(scores, k):
    if not 0 < k <= scores.shape[1]:
        raise ValueError("Invalid top-k")
    # Stable sorting makes ties deterministic across all candidate controls.
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def fixed_budget_candidates(base_top, retrieval, support_mask, add_budget):
    if base_top.shape[1] != 200 or add_budget not in (20, 50):
        raise ValueError("This screen requires K=200 and add budget 20 or 50")
    kept = base_top[:, :200 - add_budget]
    masked = np.array(retrieval, copy=True)
    masked[support_mask] = -np.inf
    # Exclude the complete original top-200, not just the retained prefix.
    # Thus recovered and displaced hits have an unambiguous interpretation.
    np.put_along_axis(masked, base_top, -np.inf, axis=1)
    if np.any(np.isfinite(masked).sum(axis=1) < add_budget):
        raise ValueError("Insufficient eligible items outside base top-200")
    additions = topk_sorted(masked, add_budget)
    candidates = np.concatenate((kept, additions), axis=1)
    if np.any(np.diff(np.sort(candidates, axis=1), axis=1) == 0):
        raise AssertionError("Duplicate candidate")
    if np.take_along_axis(support_mask, candidates, axis=1).any():
        raise AssertionError("History item entered candidate pool")
    return candidates


def metric_arrays(candidates, scores, targets):
    selected = np.take_along_axis(scores, candidates, axis=1)
    ranks = np.argsort(-selected, axis=1, kind="stable")
    ranked = np.take_along_axis(candidates, ranks, axis=1)
    hits = ranked == np.asarray(targets)[:, None]
    columns = []
    for k in (10, 20):
        columns.append(hits[:, :k].any(axis=1).astype(np.float64))
        columns.append((hits[:, :k] / np.log2(np.arange(2, k + 2))).sum(axis=1))
    columns.append(hits.any(axis=1).astype(np.float64))
    return np.stack(columns, axis=1)


def metric_means(per_user):
    if not len(per_user):
        raise ValueError("Cannot evaluate an empty user role")
    return {name: float(value) for name, value in zip(METRIC_NAMES, per_user.mean(axis=0))}


def paired_bootstrap(delta, seed, draws=1000):
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(draws):
        means.append(delta[rng.integers(0, len(delta), len(delta))].mean(axis=0))
    lower, upper = np.quantile(means, (0.025, 0.975), axis=0)
    return {name: {"delta": float(value), "ci95": [float(lo), float(hi)]}
            for name, value, lo, hi in zip(METRIC_NAMES, delta.mean(axis=0), lower, upper)}


class Prepared:
    def __init__(self, root, seed):
        directory = root / "prepared" / str(seed)
        self.base = np.load(directory / "base_scores.npy", mmap_mode="r")
        self.top = np.load(directory / "base_top200.npy", mmap_mode="r")
        history_path = directory / "history.npy"
        self.profiles = np.load(history_path if history_path.exists() else directory / "profiles.npy").astype(np.float32)
        base_user_path = directory / "base_user.npy"
        self.user_base = np.load(base_user_path if base_user_path.exists() else directory / "user_base.npy").astype(np.float32)
        if (directory / "targets.npy").exists():
            targets = np.load(directory / "targets.npy").astype(np.int64)
            all_roles = np.load(directory / "roles.npy").astype(np.int64)
            self.users = np.flatnonzero(targets >= 0)
            self.targets = targets[self.users]
            self.roles = all_roles[self.users]
        elif (directory / "targets.npz").exists():
            with np.load(directory / "targets.npz") as labels:
                self.users = labels["users"].astype(np.int64)
                self.targets = labels["items"].astype(np.int64)
                self.roles = labels["roles"].astype(np.int64)
        else:
            self.users = np.load(directory / "held_users.npy").astype(np.int64)
            self.targets = np.load(directory / "held_items.npy").astype(np.int64)
            self.roles = np.load(directory / "roles.npy").astype(np.int64)
        self.features = np.load(root / "features" / "global_bank.npy").astype(np.float32)
        self.n_users, self.n_items = self.base.shape
        self.support_mask = None
        if (directory / "support_mask.npy").exists():
            self.support_mask = np.load(directory / "support_mask.npy", mmap_mode="r")
            support_users, self.indices = np.nonzero(self.support_mask)
            counts = np.bincount(support_users, minlength=self.n_users)
            self.indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        elif (directory / "support.npz").exists():
            with np.load(directory / "support.npz") as support:
                support_users = support["users"].astype(np.int64)
                support_items = support["items"].astype(np.int64)
            order = np.lexsort((support_items, support_users))
            self.indices = support_items[order]
            counts = np.bincount(support_users, minlength=self.n_users)
            self.indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        else:
            self.indptr = np.load(directory / "support_indptr.npy").astype(np.int64)
            self.indices = np.load(directory / "support_indices.npy").astype(np.int64)
        if (directory / "candidate_context.npy").exists():
            self.context = np.load(directory / "candidate_context.npy").astype(np.float32)
        else:
            weights = 1.0 / np.log2(np.arange(2, 202, dtype=np.float32))
            weights /= weights.sum()
            self.context = np.empty((self.n_users, 1024), dtype=np.float32)
            for offset in range(0, self.n_users, 64):
                end = min(offset + 64, self.n_users)
                pooled = np.einsum("bkmd,k->bmd", self.features[self.top[offset:end]], weights)
                pooled /= np.maximum(np.linalg.norm(pooled, axis=2, keepdims=True), 1e-8)
                self.context[offset:end] = pooled.reshape(end - offset, 1024)
        self.context = normalized_rows(self.context)
        if (directory / "shuffled_context.npy").exists():
            self.shuffled = np.load(directory / "shuffled_context.npy").astype(np.float32)
        else:
            degrees = np.diff(self.indptr)
            popularity = np.bincount(self.indices, minlength=self.n_items)
            means = np.log1p(popularity[self.top]).mean(axis=1)
            permutation = matched_context_permutation(degrees, means)
            self.shuffled = self.context[permutation]
        self.shuffled = normalized_rows(self.shuffled)
        if self.features.shape != (self.n_items, 2, 512):
            raise ValueError("Expected two frozen 512-dimensional modality features")
        if self.top.shape != (self.n_users, 200):
            raise ValueError("Base top-200 shape mismatch")
        if self.profiles.shape != (self.n_users, 1024):
            raise ValueError("Profile shape mismatch")
        if self.context.shape != self.profiles.shape or self.shuffled.shape != self.profiles.shape:
            raise ValueError("Context shape mismatch")
        if len(set(self.users.tolist())) != len(self.users):
            raise ValueError("Expected one held-out positive per user")
        if not (len(self.users) == len(self.targets) == len(self.roles)):
            raise ValueError("Held labels and roles disagree")
        if set(np.unique(self.roles).tolist()) != {0, 1, 2}:
            raise ValueError("Train, calibration, and dev roles are required")
        if self.indptr.shape != (self.n_users + 1,) or self.indptr[-1] != len(self.indices):
            raise ValueError("Malformed support CSR")
        for user, target in zip(self.users, self.targets):
            if target in self.indices[self.indptr[user]:self.indptr[user + 1]]:
                raise ValueError("Held target leaked into support history")
            if not np.isfinite(self.base[user, target]):
                raise ValueError("Held target was incorrectly excluded by base")
        for array in (self.profiles, self.context, self.shuffled, self.user_base, self.features):
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite feature or context")

    def mask(self, users):
        if self.support_mask is not None:
            return np.asarray(self.support_mask[users], dtype=bool)
        result = np.zeros((len(users), self.n_items), dtype=bool)
        for row, user in enumerate(users):
            result[row, self.indices[self.indptr[user]:self.indptr[user + 1]]] = True
        return result

    def base_rows(self, users):
        scores = np.array(self.base[users], dtype=np.float32, copy=True)
        scores[self.mask(users)] = -np.inf
        return scores

    def variant_context(self, variant):
        if variant == "conditional":
            return self.context
        if variant == "shuffled":
            return self.shuffled
        if variant == "base_user":
            source = normalized_rows(self.user_base)
            # Fixed isometric padding preserves all user dimensions when d<=1024.
            if source.shape[1] > 1024:
                raise ValueError("Base user embedding exceeds matched context width")
            return np.pad(source, ((0, 0), (0, 1024 - source.shape[1])))
        # Duplicate history keeps every query network trainable and capacity-matched.
        return self.profiles


def make_model(torch, feature_array, hidden=128, dimensions=64):
    nn = torch.nn
    functional = torch.nn.functional

    class Retriever(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("features", torch.from_numpy(feature_array), persistent=False)
            self.queries = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, hidden), nn.GELU(),
                              nn.Linear(hidden, dimensions)) for _ in range(2)])
            self.items = nn.ModuleList([nn.Linear(512, dimensions, bias=False) for _ in range(2)])

        def encode_items(self):
            return [functional.normalize(self.items[m](self.features[:, m]), dim=-1)
                    for m in range(2)]

        def forward(self, profiles, contexts, item_vectors=None):
            joined = torch.cat((profiles, contexts), dim=-1)
            vectors = self.encode_items() if item_vectors is None else item_vectors
            scores = [functional.normalize(self.queries[m](joined), dim=-1) @ vectors[m].T
                      for m in range(2)]
            return (scores[0] + scores[1]) * 5.0

    return Retriever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise RuntimeError("This run already has a result")
    spec = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    settings = {"epochs": 100, "patience_evaluations": 10, "eval_every": 2,
                "train_batch_size": 256, "eval_batch_size": 256,
                "learning_rate": 0.001, "weight_decay": 0.0001,
                "hard_positive_weight": 3.0, "hidden": 128, "dimensions": 64}
    settings.update({key: spec[key] for key in settings if key in spec})
    if "batch_size" in spec:
        settings["train_batch_size"] = spec["batch_size"]
    settings.update(spec.get("retrieval", {}))
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(spec.get("gpu_memory_fraction", 0.25)))
    data = Prepared(root, args.seed)
    training = np.flatnonzero(data.roles == 0)
    calibration = np.flatnonzero(data.roles == 1)
    development = np.flatnonzero(data.roles == 2)
    if args.smoke:
        training = training[:512]
        calibration = calibration[:64]
    shared = root / "shared" / str(args.seed)
    shared_meta = None
    shared_scores = None
    if args.variant != "independent" and not args.smoke:
        shared_meta = json.loads((shared / "ready.json").read_text(encoding="utf-8"))
        if shared_meta["seed"] != args.seed:
            raise RuntimeError("Shared ranker seed mismatch")
        shared_scores = np.load(shared / "relevance_scores.npy", mmap_mode="r")
        if shared_scores.shape != data.base.shape:
            raise RuntimeError("Shared ranking score dimensions mismatch")
    profiles = torch.from_numpy(data.profiles).to(device)
    contexts = torch.from_numpy(data.variant_context(args.variant)).to(device)
    model = make_model(torch, data.features, settings["hidden"], settings["dimensions"]).to(device)
    parameter_count = (0 if args.variant == "feedback" else
                       sum(parameter.numel() for parameter in model.parameters()))
    started = time.monotonic()
    atomic_json(output / "config.json", {
        "variant": args.variant, "seed": args.seed, "settings": settings,
        "experiment": spec, "parameters": parameter_count, "device": str(device),
        "role_counts": {str(role): int((data.roles == role).sum()) for role in (0, 1, 2)},
        "official_validation_evaluated": False, "official_test_evaluated": False,
        "training_labels": "held positives from role 0 only, absent from all support histories",
        "checkpoint_selection": "calibration candidate recall@200, then ranked ndcg@20",
        "shared_ranker": str(shared), "code_sha256": sha256(Path(__file__)),
        "scope": "Baby train-internal screen, not full-data baseline confirmation"})

    @torch.no_grad()
    def model_scores(users):
        model.eval()
        vectors = model.encode_items()
        all_scores = []
        for offset in range(0, len(users), settings["eval_batch_size"]):
            batch = users[offset:offset + settings["eval_batch_size"]]
            ids = torch.as_tensor(batch, device=device)
            scores = model(profiles[ids], contexts[ids], vectors).cpu().numpy()
            scores[data.mask(batch)] = -np.inf
            all_scores.append(scores)
        return np.concatenate(all_scores)

    def combined_scores(users, own_scores=None):
        base = standardize_scores(data.base_rows(users))
        if shared_scores is None:
            # Independent's checkpoint tie-break uses alpha=1 before publication.
            relevance = standardize_scores(own_scores)
            alpha = 1.0
        else:
            relevance = np.asarray(shared_scores[users])
            alpha = shared_meta["alpha"]
        if alpha == 0:
            return base
        return base + alpha * relevance

    def evaluate(positions, scores, alpha_override=None):
        users = data.users[positions]
        targets = data.targets[positions]
        base_top = np.asarray(data.top[users])
        support_mask = data.mask(users)
        ranking_scores = combined_scores(users, scores)
        if alpha_override is not None:
            base_z = standardize_scores(data.base_rows(users))
            ranking_scores = base_z if alpha_override == 0 else base_z + alpha_override * standardize_scores(scores)
        records = {}
        for budget in (20, 50):
            candidates = fixed_budget_candidates(base_top, scores, support_mask, budget)
            arrays = metric_arrays(candidates, ranking_scores, targets)
            records[budget] = {"metrics": metric_means(arrays), "arrays": arrays,
                               "candidates": candidates}
        return records

    def selection_key(record):
        metrics = record["metrics"]
        return (metrics["recall@200"], metrics["ndcg@20"])

    if args.variant == "feedback":
        # Label-free Rocchio-like positive/negative rank-mean feedback, calibrated only.
        raw_items = data.features.reshape(data.n_items, 1024)

        def feedback_scores(users, coefficient):
            query = normalized_rows(data.profiles[users] + coefficient * data.context[users])
            scores = query @ normalized_rows(raw_items).T
            scores[data.mask(users)] = -np.inf
            return scores

        feedback_trials = []
        for coefficient in (-1.0, -0.5, 0.0, 0.5, 1.0):
            scores = feedback_scores(data.users[calibration], coefficient)
            records = evaluate(calibration, scores)
            for budget, record in records.items():
                feedback_trials.append((selection_key(record), coefficient, budget, record["metrics"]))
        _, coefficient, selected_budget, calibration_metrics = max(feedback_trials, key=lambda x: x[0])
        best_record = {"epoch": 0, "feedback_coefficient": coefficient,
                       "budget": selected_budget, "metrics": calibration_metrics}
        atomic_json(output / "feedback_calibration.json", [
            {"coefficient": item[1], "budget": item[2], "metrics": item[3]} for item in feedback_trials])
        if args.smoke:
            atomic_json(output / "result.json", {"status": "smoke_passed",
                        "not_a_performance_result": True, "variant": args.variant})
            return
        final_scores = feedback_scores(data.users[development], coefficient)
        epoch = 0
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                      weight_decay=settings["weight_decay"])
        generator = np.random.default_rng(args.seed)
        best_key = (-1.0, -1.0)
        best_record = None
        stale = 0
        epochs = 1 if args.smoke else settings["epochs"]
        before = next(model.parameters()).detach().clone()
        for epoch in range(1, epochs + 1):
            model.train()
            order = generator.permutation(training)
            losses = []
            for offset in range(0, len(order), settings["train_batch_size"]):
                batch = order[offset:offset + settings["train_batch_size"]]
                users = data.users[batch]
                targets = data.targets[batch]
                ids = torch.as_tensor(users, device=device)
                labels = torch.as_tensor(targets, device=device)
                scores = model(profiles[ids], contexts[ids])
                support = torch.from_numpy(data.mask(users)).to(device)
                scores = scores.masked_fill(support, -torch.inf)
                per_example = F.cross_entropy(scores, labels, reduction="none")
                if args.variant in ("hard", "hard_positive", "conditional", "shuffled", "base_user"):
                    missed = ~(np.asarray(data.top[users]) == targets[:, None]).any(axis=1)
                    weights = torch.as_tensor(np.where(missed, settings["hard_positive_weight"], 1.0),
                                              dtype=torch.float32, device=device)
                    loss = (per_example * weights).sum() / weights.sum()
                else:
                    loss = per_example.mean()
                if args.variant == "clear":
                    # CLEAR-style adaptation, not an official reproduction: a hard
                    # candidate negative and base-score-dependent residual margin.
                    pool = np.array(data.top[users], copy=True)
                    columns = generator.integers(0, 200, len(users))
                    negatives = pool[np.arange(len(users)), columns]
                    collisions = negatives == targets
                    columns[collisions] = (columns[collisions] + 1) % 200
                    negatives = pool[np.arange(len(users)), columns]
                    if np.any(negatives == targets):
                        raise AssertionError("Positive used as CLEAR negative")
                    base_z = standardize_scores(data.base_rows(users))
                    margins = 1.0 - 0.5 * (base_z[np.arange(len(users)), targets]
                                           - base_z[np.arange(len(users)), negatives])
                    margins = torch.as_tensor(np.clip(margins, 0.0, 5.0), device=device)
                    rows = torch.arange(len(users), device=device)
                    negative_ids = torch.as_tensor(negatives, device=device)
                    residual = F.relu(margins - scores[rows, labels] + scores[rows, negative_ids]).mean()
                    loss = loss + residual
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite retrieval training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("Nonfinite retrieval gradient")
                optimizer.step()
                losses.append(float(loss.detach()))
            progress = {"status": "training", "variant": args.variant, "seed": args.seed,
                        "epoch": epoch, "train_loss": float(np.mean(losses)),
                        "elapsed_seconds": time.monotonic() - started}
            if epoch == 1 or epoch % settings["eval_every"] == 0 or epoch == epochs:
                scores = model_scores(data.users[calibration])
                records = evaluate(calibration, scores)
                budget = max(records, key=lambda candidate: selection_key(records[candidate]))
                record = records[budget]
                key = selection_key(record)
                progress["calibration"] = {str(k): v["metrics"] for k, v in records.items()}
                if key > best_key:
                    best_key, stale = key, 0
                    best_record = {"epoch": epoch, "budget": budget, "metrics": record["metrics"]}
                    temporary = output / "best.pth.tmp"
                    torch.save(model.state_dict(), temporary)
                    temporary.replace(output / "best.pth")
                    atomic_json(output / "best.json", best_record)
                else:
                    stale += 1
            progress["best"] = best_record
            progress["stale_evaluations"] = stale
            atomic_json(output / "progress.json", progress)
            with (output / "epochs.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(progress, allow_nan=False) + "\n")
            print(json.dumps(progress, allow_nan=False), flush=True)
            if stale >= settings["patience_evaluations"]:
                break
        if args.smoke:
            if torch.equal(before, next(model.parameters()).detach()):
                raise AssertionError("Retrieval model did not update")
            atomic_json(output / "result.json", {"status": "smoke_passed",
                        "not_a_performance_result": True, "variant": args.variant,
                        "parameters": parameter_count, "candidate_budget_verified": 200,
                        "training_examples": len(training), "calibration_examples": len(calibration),
                        "dev_evaluated": False, "elapsed_seconds": time.monotonic() - started})
            print("SMOKE_PASSED", flush=True)
            return
        model.load_state_dict(torch.load(output / "best.pth", map_location=device))
        if args.variant == "independent":
            shared.mkdir(parents=True, exist_ok=True)
            if (shared / "ready.json").exists():
                raise RuntimeError("Refusing to replace an existing shared ranker")
            calibration_scores = model_scores(data.users[calibration])
            base_z = standardize_scores(data.base_rows(data.users[calibration]))
            relevance_z = standardize_scores(calibration_scores)
            pools = [np.asarray(data.top[data.users[calibration]])]
            pools.extend(record["candidates"] for record in evaluate(calibration, calibration_scores).values())
            alpha_trials = []
            for alpha in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
                scores = base_z if alpha == 0 else base_z + alpha * relevance_z
                values = [metric_means(metric_arrays(pool, scores, data.targets[calibration])) for pool in pools]
                key = (float(np.mean([v["ndcg@20"] for v in values])),
                       float(np.mean([v["recall@20"] for v in values])))
                alpha_trials.append({"alpha": alpha, "key": list(key), "pool_metrics": values})
            chosen = max(alpha_trials, key=lambda trial: tuple(trial["key"]))
            destination = shared / "relevance_scores.npy"
            mapped = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32,
                                              shape=data.base.shape)
            for offset in range(0, data.n_users, settings["eval_batch_size"]):
                users = np.arange(offset, min(offset + settings["eval_batch_size"], data.n_users))
                mapped[users] = standardize_scores(model_scores(users))
            mapped.flush()
            del mapped
            torch.save(model.state_dict(), shared / "model.pth")
            shared_meta = {"seed": args.seed, "alpha": chosen["alpha"],
                           "model_sha256": sha256(shared / "model.pth"),
                           "source_run": str(output), "selection": "calibration mean N20 over base and independent pools",
                           "alpha_trials": alpha_trials, "train_role": 0, "calibration_role": 1}
            atomic_json(shared / "ready.json", shared_meta)
            shared_scores = np.load(destination, mmap_mode="r")
        # Re-select only the quota on calibration using the now-frozen common ranker.
        calibration_scores = model_scores(data.users[calibration])
        records = evaluate(calibration, calibration_scores)
        selected_budget = max(records, key=lambda budget: selection_key(records[budget]))
        best_record["final_budget"] = selected_budget
        best_record["final_calibration"] = {str(k): v["metrics"] for k, v in records.items()}
        final_scores = model_scores(data.users[development])

    # Dev labels are read for reporting only after checkpoint, alpha, and quota lock.
    records = evaluate(development, final_scores)
    users = data.users[development]
    targets = data.targets[development]
    base_top = np.asarray(data.top[users])
    ranking_scores = combined_scores(users, final_scores)
    pipeline_arrays = metric_arrays(base_top, ranking_scores, targets)
    fusion_top = topk_sorted(ranking_scores, 200)
    fusion_arrays = metric_arrays(fusion_top, ranking_scores, targets)
    raw_arrays = metric_arrays(base_top, np.asarray(data.base[users]), targets)
    selected = records[selected_budget]
    baseline_hits = (base_top == targets[:, None]).any(axis=1)
    new_hits = (selected["candidates"] == targets[:, None]).any(axis=1)
    recovered = new_hits & ~baseline_hits
    displaced = baseline_hits & ~new_hits
    if not np.allclose((recovered.astype(float) - displaced.astype(float)).mean(),
                       selected["metrics"]["recall@200"] - pipeline_arrays[:, -1].mean()):
        raise AssertionError("Net candidate-hit decomposition failed")
    np.savez_compressed(output / "dev_per_user.npz", users=users, targets=targets,
                        metrics=selected["arrays"], pipeline_baseline=pipeline_arrays,
                        full_catalog_fusion_baseline=fusion_arrays,
                        raw_baseline=raw_arrays, candidates=selected["candidates"],
                        recovered=recovered, displaced=displaced,
                        metric_names=np.asarray(METRIC_NAMES))
    result = {"status": "completed", "variant": args.variant, "seed": args.seed,
              "scope": "train-internal candidate retrieval screen", "best": best_record,
              "epochs": epoch, "selected_add_budget": selected_budget,
              "candidate_budget": 200, "common_ranker_alpha": shared_meta["alpha"],
              "common_ranker_sha256": shared_meta["model_sha256"],
              "dev_metrics": selected["metrics"],
              "dev_budgets": {str(k): v["metrics"] for k, v in records.items()},
              "pipeline_baseline": metric_means(pipeline_arrays),
              "full_catalog_fusion_baseline": metric_means(fusion_arrays),
              "raw_base_baseline": metric_means(raw_arrays),
              "candidate_recovered_rate": float(recovered.mean()),
              "candidate_displaced_rate": float(displaced.mean()),
              "candidate_net_delta": float(recovered.mean() - displaced.mean()),
              "paired_vs_pipeline": paired_bootstrap(selected["arrays"] - pipeline_arrays, args.seed),
              "elapsed_seconds": time.monotonic() - started,
              "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
              "official_validation_evaluated": False, "official_test_evaluated": False,
              "dev_used_for_training_or_selection": False, "performance_gain_not_guaranteed": True}
    atomic_json(output / "result.json", result)
    atomic_json(output / "progress.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output" in sys.argv and sys.argv.index("--output") + 1 < len(sys.argv):
            failure_output = Path(sys.argv[sys.argv.index("--output") + 1])
            failure_output.mkdir(parents=True, exist_ok=True)
            atomic_json(failure_output / "failure.json", {
                "status": "failed", "error": str(error), "traceback": traceback.format_exc(),
                "official_validation_evaluated": False, "official_test_evaluated": False})
        raise
