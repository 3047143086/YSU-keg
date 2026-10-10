"""Exploratory support-history pairing screen; never reads official val/test.

Existing frozen scores are shared by every arm. Only role 0 trains the new
score branch; role 1 chooses checkpoint and scale; role 2 is evaluated once.
Centered moments are observable statistics, not exact PID/ANOVA components.
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


VARIANTS = ("additive", "marginal_tensor", "paired_moment", "paired_cov", "shuffled_cov")
METRIC_NAMES = ("recall@10", "ndcg@10", "recall@20", "ndcg@20", "recall@200")
SCALES = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)


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


def normalized_rows(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def standardize_scores(scores):
    finite = np.isfinite(scores)
    counts = finite.sum(axis=1, keepdims=True)
    if np.any(counts < 200):
        raise ValueError("At least 200 eligible items are required")
    safe = np.where(finite, scores, 0.0)
    means = safe.sum(axis=1, keepdims=True) / counts
    centered = np.where(finite, safe - means, 0.0)
    stds = np.sqrt((centered * centered).sum(axis=1, keepdims=True) / counts)
    result = centered / np.maximum(stds, 1e-6)
    result[~finite] = -np.inf
    return result.astype(np.float32)


def combine_scores(baseline, residual, scale):
    if baseline.shape != residual.shape or not np.isfinite(residual).all():
        raise ValueError("Residual score dimensions or values are invalid")
    if scale == 0:
        return np.array(baseline, copy=True)
    return baseline + np.float32(scale) * residual


def metric_arrays(scores, targets):
    """Exact full-catalog target ranks with ascending item-ID tie breaking."""
    targets = np.asarray(targets, dtype=np.int64)
    if scores.ndim != 2 or targets.shape != (len(scores),):
        raise ValueError("Bad metric inputs")
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("Scores contain NaN or positive infinity")
    target_scores = scores[np.arange(len(scores)), targets]
    if not np.isfinite(target_scores).all():
        raise ValueError("Evaluation target is excluded or nonfinite")
    item_ids = np.arange(scores.shape[1])[None, :]
    ranks = (scores > target_scores[:, None]).sum(axis=1)
    ranks += ((scores == target_scores[:, None]) & (item_ids < targets[:, None])).sum(axis=1)
    columns = []
    for cutoff in (10, 20):
        hit = (ranks < cutoff).astype(np.float64)
        columns.extend((hit, hit / np.log2(ranks + 2)))
    columns.append((ranks < 200).astype(np.float64))
    return np.stack(columns, axis=1)


def metric_means(values):
    if not len(values):
        raise ValueError("Empty evaluation role")
    return {name: float(value) for name, value in zip(METRIC_NAMES, values.mean(axis=0))}


def paired_bootstrap(delta, seed, draws=1000):
    rng = np.random.default_rng(seed)
    means = np.empty((draws, delta.shape[1]), dtype=np.float64)
    for draw in range(draws):
        means[draw] = delta[rng.integers(0, len(delta), len(delta))].mean(axis=0)
    lower, upper = np.quantile(means, (0.025, 0.975), axis=0)
    return {name: {"delta": float(value), "ci95": [float(lo), float(hi)]}
            for name, value, lo, hi in zip(METRIC_NAMES, delta.mean(axis=0), lower, upper)}


def split_roles(targets, roles):
    targets, roles = np.asarray(targets), np.asarray(roles)
    if targets.ndim != 1 or roles.shape != targets.shape:
        raise ValueError("Roles and targets must be aligned")
    valid = targets >= 0
    if set(np.unique(roles[valid]).tolist()) != {0, 1, 2}:
        raise ValueError("All three roles are required")
    groups = tuple(np.flatnonzero(valid & (roles == role)) for role in (0, 1, 2))
    if len(np.unique(np.concatenate(groups))) != sum(map(len, groups)):
        raise AssertionError("Roles overlap")
    return groups


def validate_support(support, targets):
    if support.ndim != 2 or targets.shape != (support.shape[0],):
        raise ValueError("Support and targets disagree")
    users = np.flatnonzero(targets >= 0)
    if np.any(targets[users] >= support.shape[1]):
        raise ValueError("Target item outside catalog")
    if support[users, targets[users]].any():
        raise ValueError("Held target leaked into support history")
    if np.any(np.asarray(support.sum(axis=1))[users] == 0):
        raise ValueError("This screen requires nonempty support histories")


def orthogonal_projection(width, dimensions, seed):
    if dimensions > width or dimensions < 1:
        raise ValueError("Invalid projection dimension")
    rng = np.random.default_rng(seed)
    projected, triangular = np.linalg.qr(rng.standard_normal((width, dimensions)))
    signs = np.where(np.diag(triangular) < 0, -1.0, 1.0)
    return (projected * signs).astype(np.float32)


def history_statistics(visual, textual, permutation=None):
    visual = np.asarray(visual, dtype=np.float32)
    textual = np.asarray(textual, dtype=np.float32)
    if visual.ndim != 2 or visual.shape != textual.shape or not len(visual):
        raise ValueError("Paired nonempty history matrices are required")
    if permutation is not None:
        permutation = np.asarray(permutation, dtype=np.int64)
        if not np.array_equal(np.sort(permutation), np.arange(len(visual))):
            raise ValueError("Text permutation must preserve the whole history multiset")
        textual = textual[permutation]
    visual_mean = visual.mean(axis=0)
    textual_mean = textual.mean(axis=0)
    marginal = np.outer(visual_mean, textual_mean)
    moment = visual.T @ textual / len(visual)
    covariance = moment - marginal
    if len(visual) == 1:
        covariance.fill(0.0)
    return {"visual_mean": visual_mean, "textual_mean": textual_mean,
            "marginal_tensor": marginal.reshape(-1),
            "paired_moment": moment.reshape(-1), "paired_cov": covariance.reshape(-1)}


def rms_transform(values, fitting_users=None):
    fitting = values if fitting_users is None else values[fitting_users]
    rms = np.sqrt(np.mean(np.square(fitting, dtype=np.float64), axis=0))
    # A floor tied to nonzero scales avoids amplifying numerical zero covariance.
    floor = max(float(np.median(rms)) * 1e-3, 1e-8)
    return (values / np.maximum(rms, floor)).astype(np.float32), rms.astype(np.float32)


class Prepared:
    def __init__(self, source_root, seed, variant, dimensions=32):
        self.source = Path(source_root).resolve()
        directory = self.source / "prepared" / str(seed)
        self.base = np.load(directory / "base_scores.npy", mmap_mode="r")
        self.support = np.load(directory / "support_mask.npy", mmap_mode="r")
        self.targets = np.load(directory / "targets.npy").astype(np.int64)
        self.roles = np.load(directory / "roles.npy").astype(np.int64)
        self.training, self.calibration, self.development = split_roles(self.targets, self.roles)
        validate_support(self.support, self.targets)
        self.n_users, self.n_items = self.base.shape
        if self.support.shape != self.base.shape:
            raise ValueError("Support and score matrix dimensions disagree")
        features = np.load(self.source / "features" / "global_bank.npy").astype(np.float32)
        if features.shape != (self.n_items, 2, 512) or not np.isfinite(features).all():
            raise ValueError("Expected finite frozen feature bank [items, 2, 512]")
        features = normalized_rows(features)
        self.shared_path = self.source / "shared" / str(seed)
        self.shared_meta = json.loads((self.shared_path / "ready.json").read_text(encoding="utf-8"))
        if self.shared_meta.get("seed") != seed:
            raise ValueError("Shared ranker seed differs")
        self.alpha = float(self.shared_meta["alpha"])
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("Invalid frozen fusion alpha")
        self.relevance = np.load(self.shared_path / "relevance_scores.npy", mmap_mode="r")
        if self.relevance.shape != self.base.shape:
            raise ValueError("Shared ranker dimensions differ")
        projection_visual = orthogonal_projection(512, dimensions, seed + 17001)
        projection_text = orthogonal_projection(512, dimensions, seed + 17002)
        visual = (features[:, 0] - features[:, 0].mean(axis=0)) @ projection_visual
        textual = (features[:, 1] - features[:, 1].mean(axis=0)) @ projection_text
        self.projections = {"visual": projection_visual, "textual": projection_text}
        degree = self.support.sum(axis=1).astype(np.int64)
        if variant == "additive":
            user_values = np.zeros((self.n_users, 2, 512), dtype=np.float32)
            item_values = features
        else:
            user_values = np.zeros((self.n_users, dimensions * dimensions), dtype=np.float32)
            item_values = np.einsum("bi,bj->bij", visual, textual).reshape(self.n_items, -1)
        rng = np.random.default_rng(seed + 27001)
        raw_norms = np.zeros(self.n_users, dtype=np.float32)
        unchanged = 0
        shuffled_users = 0
        for user in range(self.n_users):
            history = np.flatnonzero(self.support[user])
            if not len(history):
                continue
            if variant == "additive":
                user_values[user] = features[history].mean(axis=0)
            else:
                permutation = None
                if variant == "shuffled_cov":
                    # Uniform permutations include fixed points. Derangements bias
                    # the mean shuffled covariance for short user histories.
                    permutation = rng.permutation(len(history))
                    unchanged += int(np.array_equal(permutation, np.arange(len(history))))
                    shuffled_users += 1
                stats = history_statistics(visual[history], textual[history], permutation)
                key = "paired_cov" if variant == "shuffled_cov" else variant
                user_values[user] = stats[key]
            raw_norms[user] = np.linalg.norm(user_values[user])
        if variant == "additive":
            self.user_values = normalized_rows(user_values)
            self.item_values = normalized_rows(item_values)
            user_rms = np.ones((2, 512), dtype=np.float32)
            item_rms = np.ones((2, 512), dtype=np.float32)
        else:
            self.user_values, user_rms = rms_transform(user_values, self.training)
            self.item_values, item_rms = rms_transform(item_values)
        if not np.isfinite(self.user_values).all() or not np.isfinite(self.item_values).all():
            raise ValueError("Nonfinite branch representation")
        self.rms = {"user": user_rms, "item": item_rms}
        self.diagnostics = {
            "history_min": int(degree.min()), "history_median": float(np.median(degree)),
            "history_max": int(degree.max()), "single_history_users": int((degree == 1).sum()),
            "raw_user_norm_mean": float(raw_norms.mean()),
            "raw_user_norm_median": float(np.median(raw_norms)),
            "raw_zero_user_fraction": float((raw_norms == 0).mean()),
            "unchanged_uniform_permutation_users": unchanged,
            "uniform_permutation_users": shuffled_users,
            "projection_dimensions_per_modality": dimensions,
            "catalog_centering": "label-free mean of frozen features; transductive catalog",
            "user_rms_fitting_role": 0,
            "short_history_rule": "n=1 covariance exactly zero; no derangement or artificial covariance"}
        self.provenance = {
            "source_root": str(self.source), "prepared_seed": seed,
            "targets_sha256": sha256(directory / "targets.npy"),
            "roles_sha256": sha256(directory / "roles.npy"),
            "shared_ready_sha256": sha256(self.shared_path / "ready.json"),
            "shared_ranker_sha256": self.shared_meta.get("model_sha256"),
            "feature_path": str(self.source / "features" / "global_bank.npy"),
            "base_scores_path": str(directory / "base_scores.npy"),
            "support_mask_path": str(directory / "support_mask.npy"),
            "feature_sha256": sha256(self.source / "features" / "global_bank.npy"),
            "support_mask_sha256": sha256(directory / "support_mask.npy")}

    def baseline_rows(self, users):
        mask = np.asarray(self.support[users], dtype=bool)
        base = np.array(self.base[users], dtype=np.float32, copy=True)
        base[mask] = -np.inf
        baseline = standardize_scores(base)
        if self.alpha:
            relevance = np.array(self.relevance[users], dtype=np.float32, copy=True)
            if not np.isfinite(relevance[~mask]).all():
                raise ValueError("Frozen relevance has nonfinite eligible scores")
            baseline += self.alpha * relevance
        baseline[mask] = -np.inf
        return baseline


def make_model(torch, user_values, item_values, variant):
    nn = torch.nn
    functional = torch.nn.functional

    def head(width, hidden, dimensions):
        return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(),
                             nn.Linear(hidden, dimensions))

    class ScoreBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("user_values", torch.from_numpy(user_values), persistent=False)
            self.register_buffer("item_values", torch.from_numpy(item_values), persistent=False)
            if variant == "additive":
                self.user_heads = nn.ModuleList([head(512, 128, 32) for _ in range(2)])
                self.item_heads = nn.ModuleList([head(512, 128, 32) for _ in range(2)])
            else:
                self.user_heads = nn.ModuleList([head(user_values.shape[1], 128, 64)])
                self.item_heads = nn.ModuleList([head(item_values.shape[1], 128, 64)])

        def encode_items(self):
            if variant == "additive":
                return [functional.normalize(module(self.item_values[:, index]), dim=-1)
                        for index, module in enumerate(self.item_heads)]
            return [functional.normalize(self.item_heads[0](self.item_values), dim=-1)]

        def forward(self, users, items=None):
            items = self.encode_items() if items is None else items
            if variant == "additive":
                queries = [functional.normalize(module(self.user_values[users, index]), dim=-1)
                           for index, module in enumerate(self.user_heads)]
            else:
                queries = [functional.normalize(self.user_heads[0](self.user_values[users]), dim=-1)]
            values = [query @ item.T for query, item in zip(queries, items)]
            # Equal score range for additive and tensor arms, not exact capacity equality.
            return sum(values) * (5.0 / len(values))

    return ScoreBranch()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", "--source", dest="source_root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs, args.patience, args.batch_size, args.eval_every, args.threads) < 1:
        raise ValueError("Training budgets must be positive")
    if not 0 < args.memory_fraction <= 1:
        raise ValueError("Invalid memory fraction")
    root = args.root.resolve()
    output = root / "results" / str(args.seed) / args.variant
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists() or (output / "best.pt").exists():
        raise RuntimeError("Refusing to overwrite an existing experiment")
    import torch
    from torch.nn import functional as F

    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    atomic_json(output / "progress.json", {"status": "preparing", "seed": args.seed,
                "variant": args.variant, "elapsed_seconds": time.monotonic() - started})
    data = Prepared(args.source_root, args.seed, args.variant)
    training = data.training[:512] if args.smoke else data.training
    calibration = data.calibration[:64] if args.smoke else data.calibration
    model = make_model(torch, data.user_values, data.item_values, args.variant).to(device)
    params = sum(parameter.numel() for parameter in model.parameters())
    np.savez_compressed(output / "representations.npz", **data.projections,
                        user_rms=data.rms["user"], item_rms=data.rms["item"])
    settings = {key: value for key, value in vars(args).items() if not isinstance(value, Path)}
    config = {"settings": settings, "parameters": params, "variant": args.variant,
              "seed": args.seed, "baseline_alpha": data.alpha,
              "role_counts": {str(role): int((data.roles == role).sum()) for role in (0, 1, 2)},
              "provenance": data.provenance, "representation_diagnostics": data.diagnostics,
              "selection": "calibration role 1 ndcg@20, then recall@20; lower scale on exact ties",
              "scale_grid": list(SCALES), "train_residual_scale": 1.0,
              "scope": "reused internal-development exploratory screen, not independent confirmation",
              "official_validation_evaluated": False, "official_test_evaluated": False,
              "training_labels": "role 0 held positives only, absent from support of every arm",
              "parameter_matching": "four tensor arms exact; additive comparable, not exact",
              "code_sha256": sha256(Path(__file__))}
    atomic_json(output / "config.json", config)
    # Keep frozen scores on host memory; only current batches occupy device memory.
    baseline = np.empty(data.base.shape, dtype=np.float32)
    for offset in range(0, data.n_users, args.batch_size):
        end = min(offset + args.batch_size, data.n_users)
        baseline[offset:end] = data.baseline_rows(np.arange(offset, end))
    calibration_base = baseline[calibration]
    calibration_baseline_arrays = metric_arrays(calibration_base, data.targets[calibration])
    baseline_calibration = metric_means(calibration_baseline_arrays)

    def selection_key(metrics):
        return metrics["ndcg@20"], metrics["recall@20"]

    @torch.no_grad()
    def residual_scores(users):
        model.eval()
        vectors = model.encode_items()
        values = np.empty((len(users), data.n_items), dtype=np.float32)
        for offset in range(0, len(users), args.batch_size):
            batch = users[offset:offset + args.batch_size]
            values[offset:offset + len(batch)] = model(torch.as_tensor(batch, device=device), vectors).cpu().numpy()
        if not np.isfinite(values).all():
            raise RuntimeError("Nonfinite residual predictions")
        return values

    def save_checkpoint():
        temporary = output / "best.pt.tmp"
        torch.save(model.state_dict(), temporary)
        temporary.replace(output / "best.pt")

    best = {"epoch": 0, "scale": 0.0, "metrics": baseline_calibration}
    best_key = selection_key(best["metrics"])
    save_checkpoint()
    trials = [{"epoch": 0, "scale": 0.0, "metrics": baseline_calibration}]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = np.random.default_rng(args.seed + 37001)
    first_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    stale = 0
    epochs = 1 if args.smoke else args.epochs
    for epoch in range(1, epochs + 1):
        model.train()
        order = generator.permutation(training)
        losses = []
        for offset in range(0, len(order), args.batch_size):
            users = order[offset:offset + args.batch_size]
            user_ids = torch.as_tensor(users, device=device)
            labels = torch.as_tensor(data.targets[users], device=device)
            frozen = torch.from_numpy(baseline[users]).to(device)
            scores = frozen + model(user_ids)
            loss = F.cross_entropy(scores, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Nonfinite training gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        progress = {"status": "training", "seed": args.seed, "variant": args.variant,
                    "epoch": epoch, "train_loss": float(np.mean(losses)),
                    "elapsed_seconds": time.monotonic() - started}
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == epochs:
            residual = residual_scores(calibration)
            epoch_trials = []
            for scale in SCALES:
                metrics = metric_means(metric_arrays(combine_scores(calibration_base, residual, scale),
                                                     data.targets[calibration]))
                record = {"epoch": epoch, "scale": scale, "metrics": metrics}
                epoch_trials.append(record)
                trials.append(record)
            chosen = max(epoch_trials, key=lambda record: selection_key(record["metrics"]))
            chosen_key = selection_key(chosen["metrics"])
            if chosen_key > best_key:
                best, best_key, stale = chosen, chosen_key, 0
                save_checkpoint()
            else:
                stale += 1
            progress["calibration"] = epoch_trials
            atomic_json(output / "calibration.json", {"best": best, "trials": trials})
        progress["best"] = best
        progress["stale_evaluations"] = stale
        atomic_json(output / "progress.json", progress)
        with (output / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(progress, allow_nan=False) + "\n")
        print(json.dumps(progress, allow_nan=False), flush=True)
        if stale >= args.patience:
            break
    if all(torch.equal(before, after.detach()) for before, after in zip(first_parameters, model.parameters())):
        raise AssertionError("No branch parameter changed")
    if args.smoke:
        result = {"status": "smoke_passed", "variant": args.variant, "seed": args.seed,
                  "parameters": params, "dev_evaluated": False, "training_examples": len(training),
                  "calibration_examples": len(calibration), "not_a_performance_result": True,
                  "elapsed_seconds": time.monotonic() - started}
        atomic_json(output / "result.json", result)
        atomic_json(output / "progress.json", result)
        print("SMOKE_PASSED", flush=True)
        return result
    model.load_state_dict(torch.load(output / "best.pt", map_location=device))
    atomic_json(output / "selection_locked.json", {"best": best, "development_evaluations_so_far": 0})
    development = data.development
    targets = data.targets[development]
    final_scores = combine_scores(baseline[development], residual_scores(development), best["scale"])
    baseline_scores = baseline[development]
    metrics = metric_arrays(final_scores, targets)
    baseline_arrays = metric_arrays(baseline_scores, targets)
    top20 = np.argsort(-final_scores, axis=1, kind="stable")[:, :20]
    baseline_top20 = np.argsort(-baseline_scores, axis=1, kind="stable")[:, :20]
    if np.take_along_axis(np.asarray(data.support[development]), top20, axis=1).any():
        raise AssertionError("Support item reached final recommendation list")
    np.savez_compressed(output / "per_user.npz", users=development, targets=targets,
                        metrics=metrics, baseline=baseline_arrays, top20=top20,
                        baseline_top20=baseline_top20, metric_names=np.asarray(METRIC_NAMES))
    changed = (top20 != baseline_top20).any(axis=1)
    set_changed = (np.sort(top20, axis=1) != np.sort(baseline_top20, axis=1)).any(axis=1)
    result = {"status": "completed", "variant": args.variant, "seed": args.seed,
              "dev_metrics": metric_means(metrics), "baseline_metrics": metric_means(baseline_arrays),
              "selected_scale": best["scale"], "best_epoch": best["epoch"], "epochs": epoch,
              "calibration_metrics": best["metrics"], "baseline_calibration_metrics": baseline_calibration,
              "parameters": params, "baseline_alpha": data.alpha,
              "zero_fallback_selected": best["scale"] == 0,
              "paired_vs_baseline": paired_bootstrap(metrics - baseline_arrays, args.seed),
              "top20_order_changed_fraction": float(changed.mean()),
              "top20_set_changed_fraction": float(set_changed.mean()),
              "elapsed_seconds": time.monotonic() - started,
              "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
              "provenance": data.provenance, "representation_diagnostics": data.diagnostics,
              "official_validation_evaluated": False, "official_test_evaluated": False,
              "dev_used_for_training_or_selection": False, "development_evaluations": 1,
              "scope": "reused internal-development exploratory screen, not independent confirmation",
              "performance_gain_not_guaranteed": True}
    atomic_json(output / "result.json", result)
    atomic_json(output / "progress.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        required = ("--root", "--seed", "--variant")
        if all(key in sys.argv and sys.argv.index(key) + 1 < len(sys.argv) for key in required):
            values = {key: sys.argv[sys.argv.index(key) + 1] for key in required}
            destination = Path(values["--root"]) / "results" / values["--seed"] / values["--variant"]
            destination.mkdir(parents=True, exist_ok=True)
            atomic_json(destination / "failure.json", {"status": "failed", "error": str(error),
                        "traceback": traceback.format_exc(), "official_validation_evaluated": False,
                        "official_test_evaluated": False})
        raise
