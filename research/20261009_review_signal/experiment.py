"""Run the frozen TRAIN review-expression information diagnostic.

This process never trains BRIDGE or evaluates official validation/test labels.
All source paths and hashes are supplied by the immutable input_audit.json.
"""
import csv
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
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
GPU_FREE_REQUIRED = 14 * 1024 ** 3
RAM_REQUIRED = 12 * 1024 ** 3
CPU_THREADS = 4
GPU_FRACTION = 0.55
MAX_SECONDS = 6 * 3600
EXPECTED_USERS = 13447
EXPECTED_ITEMS = 7050
TARGET_SHA256 = "1d019d38c5274cb02b5dfbd434f6e6d2ce5e904fca379cc009ba857e35bdfb1a"
_STATE = {"status": "starting", "child_pid": os.getpid(), "counts": {}}
_PEAK_ALLOCATED = 0
_PEAK_RESERVED = 0
_DEADLINE = None


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending." + str(os.getpid()))
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def atomic_text(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".pending." + str(os.getpid()))
    with pending.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def progress(stage=None, **fields):
    if stage is not None:
        _STATE["stage"] = stage
    if "counts" in fields:
        fields["counts"] = dict(_STATE.get("counts", {}), **fields["counts"])
    _STATE.update(fields)
    _STATE["updated_unix"] = time.time()
    atomic_json(PROGRESS, _STATE)
    print(json.dumps({key: _STATE[key] for key in fields}, sort_keys=True), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry_path(row):
    if not isinstance(row, dict) or not row.get("path") or not row.get("sha256"):
        raise ValueError("Audit entries require path and sha256")
    path = Path(row["path"])
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def manifest_entries(manifest):
    files = manifest.get("files")
    dependencies = manifest.get("dependencies", {})
    if not isinstance(files, dict) or not files:
        raise ValueError("input_audit.json must contain nonempty files")
    entries = {}
    for group_name, group in (("files", files), ("dependencies", dependencies)):
        if not isinstance(group, dict):
            raise ValueError(group_name + " must be a mapping")
        for name, row in group.items():
            if not isinstance(row, dict):
                raise ValueError("Invalid audit row: " + str(name))
            path = _entry_path(row)
            key = str(path.resolve())
            if key in entries and entries[key]["sha256"] != row["sha256"]:
                raise ValueError("Conflicting hashes for " + key)
            entries[key] = {"name": str(name), "path": path,
                            "sha256": str(row["sha256"]), "group": group_name}
    return list(entries.values())


def verify_entries(entries):
    actual = {}
    for entry in entries:
        digest = sha256(entry["path"])
        if digest != entry["sha256"]:
            raise RuntimeError("Input hash mismatch: " + str(entry["path"]))
        actual[str(entry["path"])] = digest
    return actual


def choose_inputs(manifest):
    files, dependencies = manifest["files"], manifest["dependencies"]
    inputs = {name: Path(files[name]["path"]) for name in
              ("interactions", "item_mapping", "reviews", "global_bank", "clip_weights")}
    inputs["diagnose"] = Path(dependencies["dual_witness_diagnose"]["path"])
    inputs["evidence"] = Path(dependencies["dual_witness_evidence"]["path"])
    inputs["vendor_root"] = Path(files["clip_source_clip.py"]["path"]).parent.parent
    return inputs


def mem_available():
    try:
        values = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.split()[0]) * 1024
        return values.get("MemAvailable", 0)
    except FileNotFoundError:
        return 0


def gpu_free_bytes():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20, check=False)
    if result.returncode:
        return None
    values = {}
    for line in result.stdout.splitlines():
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) == 2:
            try:
                values[int(pieces[0])] = int(pieces[1]) * 1024 ** 2
            except ValueError:
                continue
    return values.get(1)


def wait_resources():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    started = time.time()
    while time.time() < _DEADLINE:
        free = gpu_free_bytes()
        ram = mem_available()
        if free is not None and free >= GPU_FREE_REQUIRED and ram >= RAM_REQUIRED:
            record = {"gpu_physical_index": 1, "gpu_free_bytes": free,
                      "ram_available_bytes": ram, "required_gpu_free_bytes": GPU_FREE_REQUIRED,
                      "required_ram_bytes": RAM_REQUIRED, "cpu_threads": CPU_THREADS,
                      "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                      "memory_fraction": GPU_FRACTION,
                      "wait_seconds": time.time() - started}
            progress("resources_ready", resources=record)
            return record
        progress("waiting_for_resources", resources={
            "gpu_physical_index": 1, "gpu_free_bytes": free, "ram_available_bytes": ram,
            "required_gpu_free_bytes": GPU_FREE_REQUIRED, "required_ram_bytes": RAM_REQUIRED,
            "cpu_threads": CPU_THREADS, "cuda_visible_devices": "1",
            "memory_fraction": GPU_FRACTION})
        time.sleep(30)
    raise TimeoutError("GPU1/RAM resource wait exceeded six-hour budget")


def load_diagnose(path, evidence_path):
    evidence_dir = str(evidence_path.parent.resolve())
    sys.path.insert(0, evidence_dir)
    evidence_spec = importlib.util.spec_from_file_location("evidence", str(evidence_path.resolve()))
    if evidence_spec is None or evidence_spec.loader is None:
        raise ImportError("Cannot load frozen evidence.py")
    evidence = importlib.util.module_from_spec(evidence_spec)
    sys.modules["evidence"] = evidence
    evidence_spec.loader.exec_module(evidence)
    spec = importlib.util.spec_from_file_location("frozen_dual_witness_diagnose",
                                                  str(path.resolve()))
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load frozen diagnose.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded_evidence = sys.modules.get("evidence")
    if loaded_evidence is None or Path(loaded_evidence.__file__).resolve() != evidence_path.resolve():
        raise RuntimeError("The frozen evidence dependency was imported from another path")
    required = ("internal_split", "comparisons", "auc", "paired", "key")
    if any(not hasattr(module, name) for name in required):
        raise RuntimeError("Frozen diagnose.py lacks required API")
    return module


def load_local(name):
    spec = importlib.util.spec_from_file_location("review_signal_" + name, str(ROOT / (name + ".py")))
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load local " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_train_csr(path, n_items):
    import numpy as np
    from scipy import sparse
    users, items = [], []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"userID", "itemID", "x_label"}
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError("Interactions lack required columns")
        max_user = -1
        for row in reader:
            user = int(row["userID"])
            if user < 0:
                raise ValueError("Negative user ID")
            max_user = max(max_user, user)
            if int(float(row["x_label"])) != 0:
                continue
            item = int(row["itemID"])
            if user < 0 or item < 0 or item >= n_items:
                raise ValueError("TRAIN ID outside frozen item range")
            users.append(user)
            items.append(item)
            max_user = max(max_user, user)
    if max_user < 0:
        raise ValueError("No TRAIN interactions")
    rows = np.ones(len(users), dtype=np.float32)
    train = sparse.csr_matrix((rows, (users, items)),
                              shape=(max_user + 1, n_items), dtype=np.float32)
    train.sum_duplicates()
    train.eliminate_zeros()
    if train.nnz != len(users) or (train.data != 1).any():
        raise ValueError("TRAIN interactions must be unique binary edges")
    return train


def clean_records_and_donors(records):
    import numpy as np
    core = load_local("screen_core")
    records = list(records)
    texts = [core.review_text(row) for row in records]
    donors = core.same_item_donors(records)
    keep = donors >= 0
    return records, texts, donors.astype(np.int64), keep


def encode_texts(texts, weights, vendor_root, deadline):
    import numpy as np
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    import torch
    sys.path.insert(0, str(vendor_root.resolve()))
    import clip
    if Path(clip.__file__).resolve().parent != (vendor_root / "clip").resolve():
        raise RuntimeError("CLIP was imported outside the protected vendor directory")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required and only physical GPU1 is allowed")
    torch.set_num_threads(CPU_THREADS)
    torch.cuda.set_per_process_memory_fraction(GPU_FRACTION, device=0)
    model, _ = clip.load(str(weights), device="cuda", jit=False)
    model.eval()
    model.requires_grad_(False)
    tokenizer = clip.clip._tokenizer
    sot = tokenizer.encoder["<|startoftext|>"]
    eot = tokenizer.encoder["<|endoftext|>"]
    result = np.zeros((len(texts), 512), dtype=np.float32)
    batch_size = 128
    for start in range(0, len(texts), batch_size):
        if time.time() >= deadline:
            raise TimeoutError("Review encoding exceeded six-hour budget")
        stop = min(start + batch_size, len(texts))
        batch = torch.zeros((stop - start, 77), dtype=torch.long)
        for row, text in enumerate(texts[start:stop]):
            tokens = tokenizer.encode(text)[:75]
            batch[row, :len(tokens) + 2] = torch.tensor([sot] + tokens + [eot])
        with torch.no_grad():
            vectors = model.encode_text(batch.cuda()).float()
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
        result[start:stop] = vectors.cpu().numpy()
        update_memory_peaks(torch)
        progress("encoding_reviews", counts={"encoded_texts": stop, "total_texts": len(texts)})
    del model
    torch.cuda.empty_cache()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite encoded review features")
    return result


def update_memory_peaks(torch):
    global _PEAK_ALLOCATED, _PEAK_RESERVED
    _PEAK_ALLOCATED = max(_PEAK_ALLOCATED, int(torch.cuda.max_memory_allocated(0)))
    _PEAK_RESERVED = max(_PEAK_RESERVED, int(torch.cuda.max_memory_reserved(0)))


def _finite_number(value, lower=-1, upper=1):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite numeric metric")
    value = float(value)
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError("Metric is nonfinite or outside its domain")
    return value


def decide(coverage, pairs, half_deltas):
    """Apply the preregistered gate to already computed fixed statistics."""
    coverage = _finite_number(coverage, 0, 1)
    controls = ("same_item_donor", "item_content")
    if set(pairs) != set(controls) or set(half_deltas) != set(controls):
        raise ValueError("Both fixed controls are required")
    cleaned_pairs, cleaned_halves = {}, {}
    for name in controls:
        row = pairs[name]
        if not isinstance(row, dict) or set(row) != {"mean_delta", "user_bootstrap_95ci"}:
            raise ValueError("Invalid paired statistic")
        mean = _finite_number(row["mean_delta"])
        interval = row["user_bootstrap_95ci"]
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("Expected a two-bound interval")
        interval = [_finite_number(value) for value in interval]
        if interval[0] > interval[1]:
            raise ValueError("Reversed confidence interval")
        halves = half_deltas[name]
        if not isinstance(halves, (list, tuple)) or len(halves) != 2:
            raise ValueError("Both frozen hash halves are required")
        cleaned_halves[name] = [_finite_number(value) for value in halves]
        cleaned_pairs[name] = {"mean_delta": mean, "user_bootstrap_95ci": interval}
    passed = (coverage >= .80 and
              all(row["mean_delta"] >= .005 and row["user_bootstrap_95ci"][0] > 0
                  for row in cleaned_pairs.values()) and
              all(value > 0 for values in cleaned_halves.values() for value in values))
    return {"status": "keep" if passed else "discard", "passed": bool(passed),
            "coverage": coverage, "paired_vs_controls": cleaned_pairs,
            "user_hash_half_deltas": cleaned_halves,
            "formal_training_authorized_by_this_screen": False,
            "novelty_established": False, "publication_evidence": False}


def comparison_statistics(diagnose, per_user, users):
    import numpy as np
    if set(per_user) != {"own", "same_item_donor", "item_content"}:
        raise ValueError("All three profile arms are required")
    if not all(array.shape == (len(users),) and np.isfinite(array).all()
               and ((array >= 0) & (array <= 1)).all() for array in per_user.values()):
        raise ValueError("Invalid or incomplete AUC arrays")
    half = np.asarray([diagnose.key("dual-witness-half:v1:", int(user))[0] % 2
                       for user in users])
    if not all(np.any(half == value) for value in (0, 1)):
        raise ValueError("Both fixed hash halves must be nonempty")
    pairs, halves = {}, {}
    for name in ("same_item_donor", "item_content"):
        delta = per_user["own"] - per_user[name]
        pairs[name] = diagnose.paired(delta)
        halves[name] = [float(delta[half == value].mean()) for value in (0, 1)]
    return pairs, halves, half


def write_results(decision, metrics):
    lines = ["arm\tmean_auc\tnonzero_profile_coverage\tstatus"]
    for name in ("own", "same_item_donor", "item_content"):
        if decision["status"] == "crash" or name not in metrics:
            lines.append(name + "\tn/a\tn/a\tcrash")
        else:
            row = metrics[name]
            lines.append("{}\t{:.10f}\t{:.10f}\t{}".format(
                name, row["mean_auc"], row["nonzero_profile_coverage"], decision["status"]))
    atomic_text(RESULTS, "\n".join(lines) + "\n")


def main():
    global _DEADLINE
    started = time.time()
    _DEADLINE = started + MAX_SECONDS
    pre_hashes = {}
    post_hashes = {}
    decision = {"status": "crash", "passed": False, "reason": "not_completed"}
    metrics = {}
    input_records = []
    timer_installed = False
    source_revision = None
    audit_hash = None
    resources = None
    encoding_settings = {"encoder": "CLIP ViT-B/16", "frozen": True, "batch_size": 128,
                         "context_tokens": 77, "content_bpe_tokens": 75,
                         "text_fields": ["summary", "reviewText"], "html_cleaned": True,
                         "whitespace_cleaned": True, "unit_normalized": True,
                         "no_grad": True, "item_bank_max_chunks": 8}
    masking_settings = {"keep_rule": "donors >= 0", "same_mask_all_three_arms": True,
                        "all_diagnostic_users_retained": True, "zero_profiles_retained": True,
                        "negative_comparisons": 16, "ties_credit": 0.5,
                        "donor_matching": "different TRAIN user, same item, closest cleaned wordcount",
                        "donor_reuse_allowed": True, "degree_preserving_permutation": False}

    def timeout_handler(signum, frame):
        raise TimeoutError("six-hour experiment deadline exceeded")

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = str(CPU_THREADS)
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
            timer_installed = True
        progress("auditing_inputs", status="running", started_unix=started,
                 deadline_unix=_DEADLINE, counts={})
        CACHE.mkdir(exist_ok=False)
        import numpy as np
        revision = json.loads((ROOT / "source_revision.json").read_text(encoding="utf-8"))
        source_revision = revision["commit"]
        audit_hash = sha256(AUDIT)
        manifest = json.loads(AUDIT.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("heldout_target_hash") != TARGET_SHA256:
            raise ValueError("A completed audit with the frozen target digest is required")
        input_records = manifest_entries(manifest)
        pre_hashes = verify_entries(input_records)
        inputs = choose_inputs(manifest)
        global_bank = np.load(inputs["global_bank"], mmap_mode="r", allow_pickle=False)
        if global_bank.ndim != 3 or global_bank.shape[0] != EXPECTED_ITEMS or global_bank.shape[1] < 2 or global_bank.shape[2] != 512:
            raise ValueError("global_bank must be [7050, >=2, 512]")
        global_bank = np.asarray(global_bank, dtype=np.float32)
        if not np.isfinite(global_bank).all():
            raise ValueError("global_bank contains nonfinite values")
        target_vectors = global_bank.sum(axis=1)
        target_vectors = target_vectors / np.maximum(np.linalg.norm(target_vectors, axis=1, keepdims=True), 1e-8)
        item_text_vectors = global_bank[:, 1]
        n_items = int(global_bank.shape[0])
        train = read_train_csr(inputs["interactions"], n_items)
        progress("building_internal_split", counts={"train_users": train.shape[0],
                 "train_edges": int(train.nnz), "items": n_items})
        diagnose = load_diagnose(inputs["diagnose"], inputs["evidence"])
        support, users, targets = diagnose.internal_split(train)
        comparisons = diagnose.comparisons(train, users, targets)
        if len(users) != EXPECTED_USERS or comparisons.shape != (EXPECTED_USERS, 17):
            raise ValueError("Frozen diagnostic cardinality changed")
        if np.any(np.asarray(support[users, targets]).reshape(-1) != 0):
            raise RuntimeError("Internal targets survived support")
        target_pairs = sorted(zip(users.tolist(), targets.tolist()))
        target_hash = hashlib.sha256(json.dumps(target_pairs, separators=(",", ":")).encode("ascii")).hexdigest()
        if target_hash != manifest["heldout_target_hash"]:
            raise RuntimeError("Internal target digest changed")
        records, matching_audit = load_records(inputs)
        target_set = set(target_pairs)
        if any((int(row["userID"]), int(row["itemID"])) in target_set for row in records):
            raise RuntimeError("Internal target review entered support records")
        if matching_audit["matched_support_edges"] != matching_audit["expected_support_edges"]:
            raise RuntimeError("Support review identity coverage differs from the frozen audit")
        masking_settings["all_internal_targets_absent_from_support_and_records"] = True
        cleaned, cleaned_texts, donors, keep = clean_records_and_donors(records)
        if len(cleaned) != len(donors) or len(cleaned) != len(keep):
            raise RuntimeError("Donor mask length mismatch")
        progress("matching_review_controls", counts={
            "diagnostic_users": int(len(users)), "support_records": len(cleaned),
            "kept_records": int(keep.sum()), "donor_pairs": int(keep.sum())})
        needed = np.flatnonzero(keep)
        needed_indices = np.unique(np.concatenate([needed, donors[needed]]))
        texts = [cleaned_texts[int(index)] for index in needed_indices]
        if not len(needed):
            raise RuntimeError("No matched review edges")
        resources = wait_resources()
        # Importing torch/CLIP occurs only after resource and integrity checks.
        encoded = encode_texts(texts, inputs["clip_weights"], inputs["vendor_root"], _DEADLINE)
        text_by_index = {int(index): encoded[row] for row, index in enumerate(needed_indices)}
        own_rows, donor_rows, item_rows, profile_users = [], [], [], []
        for index in needed:
            user = int(cleaned[int(index)]["userID"])
            item = int(cleaned[int(index)]["itemID"])
            donor = int(donors[int(index)])
            own_rows.append(text_by_index[int(index)])
            donor_rows.append(text_by_index[donor])
            item_rows.append(item_text_vectors[item])
            profile_users.append(user)
        core = load_local("screen_core")
        own_profile = core.profile(np.asarray(own_rows), profile_users, train.shape[0])
        donor_profile = core.profile(np.asarray(donor_rows), profile_users, train.shape[0])
        item_profile = core.profile(np.asarray(item_rows), profile_users, train.shape[0])
        coverage_mask = ((np.linalg.norm(own_profile[users], axis=1) > 0) &
                         (np.linalg.norm(donor_profile[users], axis=1) > 0) &
                         (np.linalg.norm(item_profile[users], axis=1) > 0))
        coverage = float(coverage_mask.mean())
        scores = {}
        per_user = {}
        arm_coverage = {}
        for name, profiles in (("own", own_profile), ("same_item_donor", donor_profile),
                               ("item_content", item_profile)):
            scores[name] = core.score_rows(profiles, target_vectors, users, comparisons)
            arm_coverage[name] = float((np.linalg.norm(profiles[users], axis=1) > 0).mean())
            per_user[name] = diagnose.auc(scores[name])
            if not np.isfinite(per_user[name]).all():
                raise ValueError("Nonfinite AUC for " + name)
        progress("computing_fixed_statistics", counts={"diagnostic_users": len(users)})
        pairs, half_deltas, halves = comparison_statistics(diagnose, per_user, users)
        decision = decide(coverage, pairs, half_deltas)
        metrics = {name: {"mean_auc": float(per_user[name].mean()),
                          "nonzero_profile_coverage": arm_coverage[name]}
                   for name in per_user}
        np.savez_compressed(CACHE / "per_user.npz", user_ids=users, targets=targets,
                            comparisons=comparisons, user_hash_half=halves,
                            matched_nonzero_profile=coverage_mask, **per_user)
        atomic_json(CACHE / "matching_audit.json", matching_audit)
        post_hashes = verify_entries(input_records)
        if pre_hashes != post_hashes or sha256(AUDIT) != audit_hash:
            raise RuntimeError("Protected inputs changed during experiment")
        elapsed = time.time() - started
        record = {"status": "completed", "decision": decision, "metrics": metrics,
                  "coverage": coverage, "diagnostic_users": int(len(users)),
                  "support_records": len(cleaned), "kept_records": int(keep.sum()),
                  "source_hashes_before": pre_hashes, "source_hashes_after": post_hashes,
                  "matching_audit": matching_audit, "elapsed_seconds": elapsed,
                  "source_revision": source_revision, "input_audit_sha256": audit_hash,
                  "heldout_target_hash": target_hash, "encoding_settings": encoding_settings,
                  "masking_settings": masking_settings, "resources": resources,
                  "time_budget_seconds": MAX_SECONDS, "deadline_exceeded": False,
                  "peak_gpu_allocated_bytes": _PEAK_ALLOCATED,
                  "peak_gpu_reserved_bytes": _PEAK_RESERVED,
                  "validation_evaluated": False, "test_evaluated": False,
                  "training_started": False, "raw_review_text_exported": False}
        atomic_json(DIAGNOSIS, record)
        write_results(decision, metrics)
        progress("completed", status="completed", counts={"diagnostic_users": len(users), "encoded_texts": len(texts),
                 "kept_records": int(keep.sum())}, decision=decision,
                 peak_gpu_allocated_bytes=_PEAK_ALLOCATED,
                 peak_gpu_reserved_bytes=_PEAK_RESERVED)
    except BaseException as error:
        if timer_installed:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if "torch" in sys.modules:
            try:
                update_memory_peaks(sys.modules["torch"])
            except Exception:
                pass
        integrity_error = None
        try:
            if input_records:
                post_hashes = verify_entries(input_records)
            if audit_hash is not None and sha256(AUDIT) != audit_hash:
                raise RuntimeError("Input audit changed")
        except Exception as check_error:
            integrity_error = repr(check_error)
        elapsed = time.time() - started
        decision = {"status": "crash", "passed": False, "reason": "technical_failure",
                    "formal_training_authorized_by_this_screen": False,
                    "novelty_established": False, "publication_evidence": False}
        record = {"status": "technical_failure", "decision": decision, "error": repr(error),
                  "traceback": traceback.format_exc(), "elapsed_seconds": elapsed,
                  "source_hashes_before": pre_hashes,
                  "source_hashes_after": post_hashes, "integrity_error": integrity_error,
                  "source_revision": source_revision, "input_audit_sha256": audit_hash,
                  "encoding_settings": encoding_settings, "masking_settings": masking_settings,
                  "resources": resources, "time_budget_seconds": MAX_SECONDS,
                  "deadline_exceeded": isinstance(error, TimeoutError),
                  "validation_evaluated": False, "test_evaluated": False,
                  "training_started": False, "raw_review_text_exported": False,
                  "peak_gpu_allocated_bytes": _PEAK_ALLOCATED,
                  "peak_gpu_reserved_bytes": _PEAK_RESERVED}
        atomic_json(DIAGNOSIS, record)
        write_results(decision, {})
        progress("failed", status="technical_failure", error=repr(error), peak_gpu_allocated_bytes=_PEAK_ALLOCATED,
                 peak_gpu_reserved_bytes=_PEAK_RESERVED)
        raise
    finally:
        if timer_installed:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if "torch" in sys.modules and hasattr(sys.modules["torch"], "cuda"):
            try:
                update_memory_peaks(sys.modules["torch"])
            except Exception:
                pass


def load_records(inputs):
    mapping = load_local("mapping")
    records, audit = mapping.load_support_reviews(inputs["interactions"], inputs["item_mapping"],
                                          inputs["reviews"], strict=False)
    return records, audit


if __name__ == "__main__":
    main()
