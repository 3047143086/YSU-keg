"""Frozen review-direction residual diagnostic; no model training or test labels."""
import hashlib
import importlib.util
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "input_audit.json"
CACHE = ROOT / "cache"
PROGRESS = ROOT / "progress.json"
DIAGNOSIS = ROOT / "diagnosis.json"
RESULTS = ROOT / "results.tsv"
MAX_SECONDS = 6 * 3600
EXPECTED_USERS = 13447
EXPECTED_ITEMS = 7050
TARGET_SHA256 = "1d019d38c5274cb02b5dfbd434f6e6d2ce5e904fca379cc009ba857e35bdfb1a"
PREDECESSOR_COMMIT = "9c35573aa6e132940439d8853e440083daf23c73"
PREDECESSOR_EXECUTOR_SHA256 = "589c313e1b2c25f0ff6aa5ee63126186890f4bc2e788af0a74b1ed8a04251574"
CONTROLS = ("raw_own", "same_item_donor", "item_content", "residual_donor")
ARMS = ("residual",) + CONTROLS
PROTECTED_BUNDLE_FILES = ("experiment.py", "residual_core.py", "PROTOCOL.md", "program.md",
                          "source_revision.json", "input_audit.json")
_STATE = {"status": "starting", "child_pid": os.getpid(), "counts": {}}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending." + str(os.getpid()))
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def progress(stage, **fields):
    _STATE["stage"] = stage
    if "counts" in fields:
        fields["counts"] = dict(_STATE["counts"], **fields["counts"])
    _STATE.update(fields)
    _STATE["updated_unix"] = time.time()
    atomic_json(PROGRESS, _STATE)
    print(json.dumps(_STATE, sort_keys=True, allow_nan=False), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle():
    """Check immutable deployment hashes before and after the diagnostic."""
    manifest_path = ROOT / "deployment.json"
    if not manifest_path.is_file():
        raise RuntimeError("Immutable deployment manifest is missing")
    deployment = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = deployment.get("bundle_sha256")
    if not isinstance(recorded, dict) or not all(name in recorded for name in PROTECTED_BUNDLE_FILES):
        raise RuntimeError("Deployment manifest lacks protected bundle hashes")
    actual = {}
    for name in PROTECTED_BUNDLE_FILES:
        path = ROOT / name
        if not path.is_file():
            raise RuntimeError("Protected bundle file is missing: " + name)
        actual[name] = sha256(path)
    expected = {name: recorded[name] for name in PROTECTED_BUNDLE_FILES}
    if actual != expected:
        raise RuntimeError("Protected runtime bundle changed")
    return expected


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(Path(path).resolve()))
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load protected module: " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_helpers(manifest, deadline):
    entry = manifest["dependencies"]["review_signal_experiment"]
    digest = sha256(entry["path"])
    if digest != entry["sha256"] or digest != PREDECESSOR_EXECUTOR_SHA256:
        raise RuntimeError("Frozen predecessor executor hash mismatch")
    helper = load_module("frozen_review_signal_executor", entry["path"])
    for name in ("ROOT", "AUDIT", "CACHE", "PROGRESS", "DIAGNOSIS", "RESULTS"):
        setattr(helper, name, globals()[name])
    helper._DEADLINE = deadline
    helper._STATE = _STATE
    helper.progress = progress

    def load_dependency(name):
        if name not in ("mapping", "screen_core"):
            raise ValueError("Unsupported predecessor helper: " + name)
        row = manifest["dependencies"]["review_signal_" + name]
        if sha256(row["path"]) != row["sha256"]:
            raise RuntimeError("Frozen predecessor helper hash mismatch: " + name)
        return load_module("frozen_review_signal_" + name, row["path"])

    helper.load_local = load_dependency
    return helper


def _finite(value, lower=-1, upper=1):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite numeric metric")
    value = float(value)
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError("Metric is nonfinite or outside its domain")
    return value


def decide(coverage, pairs, half_deltas):
    """Apply all four preregistered comparisons without selecting a control."""
    coverage = _finite(coverage, 0, 1)
    if set(pairs) != set(CONTROLS) or set(half_deltas) != set(CONTROLS):
        raise ValueError("All four frozen controls are required")
    clean_pairs, clean_halves = {}, {}
    for name in CONTROLS:
        row = pairs[name]
        if not isinstance(row, dict) or set(row) != {"mean_delta", "user_bootstrap_95ci"}:
            raise ValueError("Invalid paired statistic")
        mean = _finite(row["mean_delta"])
        interval = row["user_bootstrap_95ci"]
        halves = half_deltas[name]
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("Expected two confidence bounds")
        if not isinstance(halves, (list, tuple)) or len(halves) != 2:
            raise ValueError("Both frozen hash halves are required")
        interval = [_finite(value) for value in interval]
        if interval[0] > interval[1]:
            raise ValueError("Reversed confidence interval")
        clean_pairs[name] = {"mean_delta": mean, "user_bootstrap_95ci": interval}
        clean_halves[name] = [_finite(value) for value in halves]
    passed = (coverage >= .80
              and all(row["mean_delta"] >= .005 and row["user_bootstrap_95ci"][0] > 0
                      for row in clean_pairs.values())
              and all(value > 0 for halves in clean_halves.values() for value in halves))
    return {"status": "keep" if passed else "discard", "passed": bool(passed),
            "coverage": coverage, "paired_vs_controls": clean_pairs,
            "user_hash_half_deltas": clean_halves,
            "formal_training_authorized_by_this_screen": False,
            "novelty_established": False, "publication_evidence": False}


def comparison_statistics(diagnose, per_user, users):
    import numpy as np
    if set(per_user) != set(ARMS):
        raise ValueError("All five profile arms are required")
    if not all(values.shape == (len(users),) and np.isfinite(values).all()
               and ((values >= 0) & (values <= 1)).all() for values in per_user.values()):
        raise ValueError("Invalid or incomplete AUC arrays")
    halves = np.asarray([diagnose.key("dual-witness-half:v1:", int(user))[0] % 2
                         for user in users])
    if not all(np.any(halves == half) for half in (0, 1)):
        raise ValueError("Both fixed hash halves must be nonempty")
    pairs, half_deltas = {}, {}
    for name in CONTROLS:
        delta = per_user["residual"] - per_user[name]
        pairs[name] = diagnose.paired(delta)
        half_deltas[name] = [float(delta[halves == half].mean()) for half in (0, 1)]
    return pairs, half_deltas, halves


def write_results(helper, decision, metrics):
    lines = ["arm\tmean_auc\tnonzero_profile_coverage\tstatus"]
    for name in ARMS:
        if decision["status"] == "crash" or name not in metrics:
            lines.append(name + "\tn/a\tn/a\tcrash")
        else:
            row = metrics[name]
            lines.append("{}\t{:.10f}\t{:.10f}\t{}".format(
                name, row["mean_auc"], row["nonzero_profile_coverage"], decision["status"]))
    text = "\n".join(lines) + "\n"
    if helper is not None:
        helper.atomic_text(RESULTS, text)
    else:
        temporary = RESULTS.with_name(RESULTS.name + ".pending." + str(os.getpid()))
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(RESULTS)


def main():
    started = time.time()
    deadline = started + MAX_SECONDS
    helper = None
    entries, pre_hashes, post_hashes = [], {}, {}
    audit_hash, source_revision, resources = None, None, None
    bundle_hashes_before, bundle_hashes_after = None, None
    deployment_manifest_sha256 = None
    timer_installed = False
    stage_seconds, cache_hashes = {}, {}
    record = {"predecessor_source_revision": PREDECESSOR_COMMIT,
              "validation_evaluated": False, "test_evaluated": False,
              "training_started": False, "raw_review_text_exported": False,
              "encoding_settings": {"encoder": "CLIP ViT-B/16", "frozen": True, "batch_size": 128,
                  "content_bpe_tokens": 75, "context_tokens": 77, "unit_normalized": True,
                  "summary_and_review_text_html_whitespace_cleaned": True, "no_grad": True,
                  "reused_previous_encoded_bank": False, "new_encoding_performed": False,
                  "item_bank_max_chunks": 8},
              "masking_settings": {"keep_rule": "donors >= 0", "same_mask_all_five_arms": True,
                  "all_diagnostic_users_retained": True, "zero_profiles_retained": True,
                  "negative_comparisons": 16, "ties_credit": .5,
                  "donor_reuse_allowed": True, "degree_preserving_permutation": False},
              "residual_formula": "normalize(z - normalize(item_text) * dot(z,normalize(item_text)))",
              "projection_trainable": False, "projection_alpha": 1.0,
              "coverage_definition": "nonzero candidate residual profile over all 13447 diagnostic users",
              "time_budget_seconds": MAX_SECONDS}

    def timeout_handler(signum, frame):
        raise TimeoutError("Six-hour residual diagnostic deadline exceeded")

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "4"
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
            timer_installed = True
        bundle_hashes_before = verify_bundle()
        deployment_manifest_sha256 = sha256(ROOT / "deployment.json")
        progress("auditing_inputs", status="running", started_unix=started, deadline_unix=deadline)
        audit_hash = sha256(AUDIT)
        manifest = json.loads(AUDIT.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("heldout_target_hash") != TARGET_SHA256:
            raise RuntimeError("A completed audit with the frozen target digest is required")
        helper = load_helpers(manifest, deadline)
        entries = helper.manifest_entries(manifest)
        pre_hashes = helper.verify_entries(entries)
        source_revision = json.loads((ROOT / "source_revision.json").read_text())["commit"]
        previous = json.loads(Path(manifest["dependencies"]["review_signal_input_audit"]["path"]).read_text())
        for name, old in previous["files"].items():
            new = manifest["files"][name]
            if old["path"] != new["path"] or old["sha256"] != new["sha256"]:
                raise RuntimeError("Residual inputs differ from the frozen predecessor: " + name)
        if previous.get("heldout_target_hash") != TARGET_SHA256:
            raise RuntimeError("Predecessor target digest mismatch")
        CACHE.mkdir(exist_ok=False)
        import numpy as np
        core = load_module("review_residual_core", ROOT / "residual_core.py")
        inputs = helper.choose_inputs(manifest)
        global_bank = np.load(inputs["global_bank"], mmap_mode="r", allow_pickle=False)
        if global_bank.shape != (EXPECTED_ITEMS, 2, 512) or not np.isfinite(global_bank).all():
            raise RuntimeError("Frozen global bank shape or values changed")
        target_vectors = core.normalize_rows(global_bank.sum(axis=1))
        item_text = np.asarray(global_bank[:, 1], dtype=np.float32)
        train = helper.read_train_csr(inputs["interactions"], EXPECTED_ITEMS)
        diagnose = helper.load_diagnose(inputs["diagnose"], inputs["evidence"])
        support, users, targets = diagnose.internal_split(train)
        comparisons = diagnose.comparisons(train, users, targets)
        if len(users) != EXPECTED_USERS or comparisons.shape != (EXPECTED_USERS, 17):
            raise RuntimeError("Frozen diagnostic cardinality changed")
        if np.any(np.asarray(support[users, targets]).reshape(-1) != 0):
            raise RuntimeError("Internal targets survived support")
        target_pairs = sorted(zip(users.tolist(), targets.tolist()))
        target_hash = hashlib.sha256(json.dumps(target_pairs, separators=(",", ":")).encode("ascii")).hexdigest()
        if target_hash != TARGET_SHA256:
            raise RuntimeError("Internal target digest changed")
        records, matching_audit = helper.load_records(inputs)
        target_set = set(target_pairs)
        if any((int(row["userID"]), int(row["itemID"])) in target_set for row in records):
            raise RuntimeError("Internal target review entered support records")
        if matching_audit["matched_support_edges"] != matching_audit["expected_support_edges"]:
            raise RuntimeError("Support identity coverage changed")
        record["masking_settings"]["all_internal_targets_absent_from_support_and_records"] = True
        records, cleaned_texts, donors, keep = helper.clean_records_and_donors(records)
        needed = np.flatnonzero(keep)
        needed_indices = np.unique(np.concatenate([needed, donors[needed]]))
        if not len(needed):
            raise RuntimeError("No matched review edges")
        progress("matching_review_controls", counts={"diagnostic_users": len(users),
                 "support_records": len(records), "kept_records": int(keep.sum()),
                 "unique_reviews_to_encode": len(needed_indices)})
        stage_seconds["audit_mapping_matching"] = time.time() - started
        texts = [cleaned_texts[int(index)] for index in needed_indices]
        resources = helper.wait_resources()
        encode_started = time.time()
        record["encoding_settings"]["new_encoding_performed"] = True
        encoded = helper.encode_texts(texts, inputs["clip_weights"], inputs["vendor_root"], deadline)
        stage_seconds["encoding"] = time.time() - encode_started
        np.save(CACHE / "review_embeddings.npy", encoded, allow_pickle=False)
        np.savez_compressed(CACHE / "review_index.npz", record_indices=needed_indices,
                            user_ids=np.asarray([records[int(index)]["userID"] for index in needed_indices]),
                            item_ids=np.asarray([records[int(index)]["itemID"] for index in needed_indices]),
                            own_record_indices=needed, donor_record_indices=donors[needed])
        cache_hashes = {name: sha256(CACHE / name) for name in ("review_embeddings.npy", "review_index.npz")}
        build_started = time.time()
        index_to_encoded = np.full(len(records), -1, dtype=np.int64)
        index_to_encoded[needed_indices] = np.arange(len(needed_indices))
        owner_ids = np.asarray([records[int(index)]["userID"] for index in needed], dtype=np.int64)
        item_ids = np.asarray([records[int(index)]["itemID"] for index in needed], dtype=np.int64)
        own_rows = encoded[index_to_encoded[needed]]
        donor_rows = encoded[index_to_encoded[donors[needed]]]
        item_rows = item_text[item_ids]
        progress("projecting_review_directions", counts={"residual_rows": len(needed)})
        vectors = {"raw_own": own_rows, "same_item_donor": donor_rows, "item_content": item_rows,
                   "residual": core.residualize(own_rows, item_rows),
                   "residual_donor": core.residualize(donor_rows, item_rows)}
        profiles = {name: core.profile(values, owner_ids, train.shape[0]) for name, values in vectors.items()}
        nonzero = {name: np.linalg.norm(values[users], axis=1) > 0 for name, values in profiles.items()}
        coverage = float(nonzero["residual"].mean())
        per_user = {name: diagnose.auc(core.score_rows(profiles[name], target_vectors, users, comparisons))
                    for name in ARMS}
        stage_seconds["projection_profiles_scoring"] = time.time() - build_started
        progress("computing_fixed_statistics", counts={"diagnostic_users": len(users)})
        stats_started = time.time()
        pairs, half_deltas, halves = comparison_statistics(diagnose, per_user, users)
        decision = decide(coverage, pairs, half_deltas)
        stage_seconds["statistics"] = time.time() - stats_started
        metrics = {name: {"mean_auc": float(per_user[name].mean()),
                         "nonzero_profile_coverage": float(nonzero[name].mean())} for name in ARMS}
        np.savez_compressed(CACHE / "per_user.npz", user_ids=users, targets=targets,
                            comparisons=comparisons, user_hash_half=halves,
                            candidate_nonzero_profile=nonzero["residual"], **per_user)
        atomic_json(CACHE / "matching_audit.json", matching_audit)
        post_hashes = helper.verify_entries(entries)
        if pre_hashes != post_hashes or sha256(AUDIT) != audit_hash:
            raise RuntimeError("Protected sources or audit changed during diagnostic")
        bundle_hashes_after = verify_bundle()
        if (bundle_hashes_after != bundle_hashes_before
                or sha256(ROOT / "deployment.json") != deployment_manifest_sha256):
            raise RuntimeError("Protected runtime bundle changed during diagnostic")
        record.update(status="completed", decision=decision, metrics=metrics, coverage=coverage,
                      diagnostic_users=len(users), support_records=len(records), kept_records=int(keep.sum()),
                      unique_encoded_reviews=len(needed_indices), heldout_target_hash=target_hash,
                      matching_audit=matching_audit, source_revision=source_revision,
                      input_audit_sha256=audit_hash, source_hashes_before=pre_hashes,
                      source_hashes_after=post_hashes, encoding_cache_sha256=cache_hashes,
                      deployment_bundle_sha256=bundle_hashes_before,
                      deployment_manifest_sha256=deployment_manifest_sha256,
                      runtime_bundle_unchanged=True,
                      stage_seconds=stage_seconds, elapsed_seconds=time.time() - started,
                      resources=resources, deadline_exceeded=False,
                      peak_gpu_allocated_bytes=helper._PEAK_ALLOCATED,
                      peak_gpu_reserved_bytes=helper._PEAK_RESERVED)
        atomic_json(DIAGNOSIS, record)
        write_results(helper, decision, metrics)
        progress("completed", status="completed", decision=decision,
                 counts={"encoded_texts": len(texts), "diagnostic_users": len(users)},
                 peak_gpu_allocated_bytes=helper._PEAK_ALLOCATED,
                 peak_gpu_reserved_bytes=helper._PEAK_RESERVED)
    except BaseException as error:
        if timer_installed:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if helper is not None and "torch" in sys.modules:
            try:
                helper.update_memory_peaks(sys.modules["torch"])
            except Exception:
                pass
        integrity_error = None
        try:
            bundle_hashes_after = verify_bundle()
            if bundle_hashes_before is not None and bundle_hashes_after != bundle_hashes_before:
                raise RuntimeError("Protected runtime bundle changed")
            if (deployment_manifest_sha256 is not None
                    and sha256(ROOT / "deployment.json") != deployment_manifest_sha256):
                raise RuntimeError("Deployment manifest changed")
            if helper is not None and entries:
                post_hashes = helper.verify_entries(entries)
            if audit_hash is not None and sha256(AUDIT) != audit_hash:
                raise RuntimeError("Input audit changed")
        except Exception as check_error:
            integrity_error = repr(check_error)
        decision = {"status": "crash", "passed": False, "reason": "technical_failure",
                    "formal_training_authorized_by_this_screen": False,
                    "novelty_established": False, "publication_evidence": False}
        record.update(status="technical_failure", decision=decision, error=repr(error),
                      traceback=traceback.format_exc(), integrity_error=integrity_error,
                      source_revision=source_revision, input_audit_sha256=audit_hash,
                      source_hashes_before=pre_hashes, source_hashes_after=post_hashes,
                      deployment_bundle_sha256=bundle_hashes_before,
                      deployment_manifest_sha256=deployment_manifest_sha256,
                      encoding_cache_sha256=cache_hashes, stage_seconds=stage_seconds,
                      elapsed_seconds=time.time() - started, resources=resources,
                      deadline_exceeded=isinstance(error, TimeoutError),
                      peak_gpu_allocated_bytes=helper._PEAK_ALLOCATED if helper else 0,
                      peak_gpu_reserved_bytes=helper._PEAK_RESERVED if helper else 0)
        atomic_json(DIAGNOSIS, record)
        write_results(helper, decision, {})
        progress("failed", status="technical_failure", error=repr(error))
        raise
    finally:
        if timer_installed:
            signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
