"""One bounded CPU-only author-expression audit of frozen TRAIN embeddings."""
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAX_SECONDS = 3600
N_USERS = 19445
N_EDGES = 105032
_DEADLINE = None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".pending." + str(os.getpid()))
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def checkpoint():
    if time.monotonic() >= _DEADLINE:
        raise TimeoutError("One-hour CPU audit budget exceeded")


def memory_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def verify_inputs(manifest):
    actual = {}
    for name, row in manifest["files"].items():
        digest = sha256(row["path"])
        if digest != row["sha256"]:
            raise RuntimeError("Frozen input changed: " + name)
        actual[name] = digest
    return actual


def verify_bundle():
    path = ROOT / "deployment.json"
    manifest = json.loads(path.read_text())
    actual = {name: sha256(ROOT / name) for name in manifest["bundle_sha256"]}
    if actual != manifest["bundle_sha256"]:
        raise RuntimeError("Immutable audit bundle changed")
    return {"manifest_sha256": sha256(path), "bundle_sha256": actual}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    global _DEADLINE
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "4"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = time.monotonic()
    _DEADLINE = started + MAX_SECONDS
    signal.signal(signal.SIGALRM, lambda signum, frame: (_ for _ in ()).throw(
        TimeoutError("One-hour CPU audit alarm exceeded")))
    signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
    record = {"status": "running", "pid": os.getpid(), "gpu_used": False, "cpu_threads": 4,
              "max_seconds": MAX_SECONDS, "new_encoding_performed": False,
              "raw_reviews_read": False, "interactions_read": False,
              "validation_evaluated": False, "test_evaluated": False,
              "formal_training_authorized": False, "publication_evidence": False}
    try:
        before_bundle = verify_bundle()
        input_path = ROOT / "input_audit.json"
        input_hash = sha256(input_path)
        manifest = json.loads(input_path.read_text())
        before_inputs = verify_inputs(manifest)
        f25 = json.loads(Path(manifest["files"]["predecessor_diagnosis"]["path"]).read_text())
        if (f25.get("status") != "completed" or f25.get("source_revision") != manifest["predecessor_commit"]
                or f25["matching_audit"]["support_users"] != N_USERS
                or not f25["matching_audit"]["all_targets_absent_from_export"]
                or not f25["masking_settings"]["all_internal_targets_absent_from_support_and_records"]):
            raise RuntimeError("F25 support-only cache provenance is invalid")
        while memory_available() < 12 * 1024 ** 3:
            checkpoint()
            atomic_json(ROOT / "progress.json", dict(record, stage="waiting_for_RAM", elapsed_seconds=time.monotonic()-started))
            time.sleep(min(30, max(0, _DEADLINE-time.monotonic())))
        record["available_ram_bytes_at_start"] = memory_available()
        import numpy as np
        core = load_module("fixed_identifiability_core", ROOT / "core.py")
        residual = load_module("frozen_review_residual_core", manifest["files"]["residual_core"]["path"])
        embeddings = np.load(manifest["files"]["review_embeddings"]["path"], mmap_mode="r", allow_pickle=False)
        with np.load(manifest["files"]["review_index"]["path"], allow_pickle=False) as index:
            users, items = index["user_ids"].copy(), index["item_ids"].copy()
            positions = index["record_indices"].copy()
            own_indices, donor_indices = index["own_record_indices"].copy(), index["donor_record_indices"].copy()
        if (embeddings.shape != (N_EDGES, 512) or len(users) != N_EDGES or len(items) != N_EDGES
                or positions.shape != (N_EDGES,) or len(np.unique(positions)) != N_EDGES
                or not np.array_equal(positions, np.sort(positions))
                or not np.isfinite(embeddings).all()):
            raise RuntimeError("F25 cache shapes, indices or vector values changed")
        if (own_indices.shape != (N_EDGES,) or donor_indices.shape != (N_EDGES,)
                or not set(own_indices.tolist()).issubset(set(positions.tolist()))
                or not set(donor_indices.tolist()).issubset(set(positions.tolist()))):
            raise RuntimeError("F25 encoded record mapping is incomplete")
        plan = core.plan_groups(users, items, N_USERS)
        b_indices = np.flatnonzero(~plan["is_a"])
        insufficient_a = sum(plan["a_counts"][int(users[index])] < 2 for index in b_indices)
        counts = {"support_users_denominator": N_USERS, "cached_users": len(np.unique(users)),
                  "cached_edges": N_EDGES, "a_edges": int(plan["is_a"].sum()),
                  "b_edges": int((~plan["is_a"]).sum()), "groups": len(plan["groups"]),
                  "scorable_queries": int(plan["scorable_queries"].sum()),
                  "scorable_users": int(plan["scorable_users"].sum()),
                  "users_without_cached_b_reviews": int((plan["b_counts"] == 0).sum()),
                  "b_queries_excluded_a_count_under_two": insufficient_a,
                  "b_queries_excluded_no_exact_degree_competitor": int(len(b_indices)
                      - insufficient_a - plan["scorable_queries"].sum()),
                  "null_fixed_points": plan["null_fixed_points"],
                  "null_assignments": plan["null_assignments"],
                  "null_fixed_point_rate": plan["null_fixed_points"] / max(1, plan["null_assignments"])}
        atomic_json(ROOT / "progress.json", dict(record, stage="fixed_split_complete", counts=counts))
        cache = ROOT / "cache"
        cache.mkdir(exist_ok=False)
        np.savez_compressed(cache / "fixed_split.npz", user_ids=users, item_ids=items,
                            population=plan["population"], a_mask=plan["is_a"],
                            b_counts=plan["b_counts"], scorable_queries=plan["scorable_queries"],
                            scorable_users=plan["scorable_users"])
        bank = np.load(manifest["files"]["global_bank"]["path"], mmap_mode="r", allow_pickle=False)
        if bank.shape != (7050, 2, 512) or not np.isfinite(bank).all() or (items >= len(bank)).any():
            raise RuntimeError("Frozen global item bank changed")
        views = {"raw": embeddings, "residual": residual.residualize(embeddings, bank[items, 1])}
        summaries = {}
        for name in ("raw", "residual"):
            checkpoint()
            atomic_json(ROOT / "progress.json", dict(record, stage="evaluating_"+name, counts=counts))
            values = core.evaluate_view(views[name], plan, block_size=128, checkpoint=checkpoint)
            summaries[name] = core.summarize(values, plan, bootstrap=2000, checkpoint=checkpoint)
            summaries[name].update(zero_a_profiles=values["zero_a_profiles"], zero_b_vectors=values["zero_b_vectors"])
            np.savez_compressed(cache / (name + "_per_user.npz"), user_ids=plan["population"],
                                real=values["user_real"], null=values["user_null"],
                                delta=values["user_real"]-values["user_null"],
                                scorable_users=plan["scorable_users"])
        checkpoint()
        after_inputs, after_bundle = verify_inputs(manifest), verify_bundle()
        if before_inputs != after_inputs or before_bundle != after_bundle or sha256(input_path) != input_hash:
            raise RuntimeError("Protected inputs or bundle changed during audit")
        if "torch" in sys.modules or "clip" in sys.modules:
            raise RuntimeError("CPU cache audit unexpectedly imported a GPU encoder")
        record.update(status="completed", stage="done", counts=counts, views=summaries,
                      predecessor_commit=manifest["predecessor_commit"],
                      population_scope="all 19445 support users, including neutral unscorable/no-B users",
                      preference_established=False, recommendation_gain_established=False,
                      inputs_sha256_before=before_inputs, inputs_sha256_after=after_inputs,
                      deployment_before=before_bundle, deployment_after=after_bundle,
                      elapsed_seconds=time.monotonic()-started,
                      peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      source_revision=json.loads((ROOT/"source_revision.json").read_text())["commit"],
                      outcome_scope="both views reported independently; no overall best-view success")
        lines = ["view\treal_user_auc\tnull_user_auc\tdelta\tcoverage\tstatus"]
        for name in ("raw", "residual"):
            row = summaries[name]
            lines.append("{}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{}".format(
                name, row["real_mean_user_auc"], row["null_mean_user_auc"], row["mean_delta"],
                row["scorable_user_coverage"], row["status"]))
        with (ROOT/"results.tsv").open("x", encoding="utf-8") as stream:
            stream.write("\n".join(lines)+"\n")
        atomic_json(ROOT/"diagnosis.json", record)
        atomic_json(ROOT/"progress.json", record)
        print(json.dumps(record, allow_nan=False), flush=True)
    except BaseException as error:
        record.update(status="technical_failure", error=repr(error), traceback=traceback.format_exc(),
                      elapsed_seconds=time.monotonic()-started)
        atomic_json(ROOT/"diagnosis.json", record)
        atomic_json(ROOT/"progress.json", record)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
