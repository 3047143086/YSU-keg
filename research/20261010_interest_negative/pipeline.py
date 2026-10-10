"""Fixed-budget queue for the interest-negative MSCA pilot.

This queue starts only new children from the immutable deployed bundle.  It does
not retry a failed arm, select an epoch or arm post hoc, or construct test data.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
QUEUE_SECONDS = 48 * 3600
JOB_SECONDS = 8 * 3600
SMOKE_SECONDS = 1800
PREPARATION_SECONDS = 1800
DATASETS = ("baby", "sports")
MODES = ("uniform", "semantic", "shuffled")
GPU_INDEX = "1"
CPU_THREADS = 4
MIN_GPU_FREE = 14 * 1024**3
MIN_RAM = 12 * 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending." + str(os.getpid()))
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(str(pending), str(path))


def _available_ram() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ValueError):
        return 0
    return 0


def resource_snapshot() -> dict:
    """Read admission resources without initializing CUDA."""
    affinity = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                else (os.cpu_count() or 0))
    row = {"gpu_index": 1, "gpu_free_bytes": None, "gpu_uuid": None,
           "gpu_compute_processes": [], "ram_available_bytes": _available_ram(),
           "gpu_process_query_ok": False,
           "cpu_affinity_count": affinity, "cpu_threads": CPU_THREADS}
    try:
        query = subprocess.run(
            ["nvidia-smi", "-i", GPU_INDEX,
             "--query-gpu=index,uuid,memory.free",
             "--format=csv,noheader,nounits"], capture_output=True,
            text=True, timeout=20, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return row
    fields = [part.strip() for part in query.stdout.strip().split(",")]
    if query.returncode == 0 and len(fields) == 3 and fields[0] == GPU_INDEX:
        row["gpu_uuid"] = fields[1]
        try:
            row["gpu_free_bytes"] = int(fields[2]) * 1024**2
        except ValueError:
            pass
        try:
            apps = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                 "--format=csv,noheader,nounits"], capture_output=True,
                text=True, timeout=20, check=False)
            row["gpu_compute_processes"] = [line.strip() for line in apps.stdout.splitlines()
                                              if line.strip().startswith(row["gpu_uuid"] + ",")]
            row["gpu_process_query_ok"] = apps.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return row


def resources_admit(row: dict, *, require_gpu: bool = True) -> tuple[bool, str]:
    if row.get("ram_available_bytes", 0) < MIN_RAM:
        return False, "host RAM below 12 GiB"
    if row.get("cpu_affinity_count", 0) < CPU_THREADS:
        return False, "CPU affinity below four CPUs"
    if require_gpu:
        if row.get("gpu_free_bytes") is None:
            return False, "GPU1 memory query unavailable"
        if row["gpu_free_bytes"] < MIN_GPU_FREE:
            return False, "GPU1 free memory below 14 GiB"
        if not row.get("gpu_process_query_ok", False):
            return False, "GPU1 process query unavailable"
        if row.get("gpu_compute_processes"):
            return False, "GPU1 has compute processes"
    return True, "admitted"


def _arm_name(dataset: str, mode: str, smoke: bool = False) -> str:
    return f"{dataset}_{mode}_s2026_{'smoke' if smoke else 'formal'}"


def _finite_metrics(result: dict) -> bool:
    metrics = result.get("metrics", {})
    required = ("recall@10", "recall@20", "ndcg@10", "ndcg@20")
    return result.get("status") == "completed" and result.get("smoke", False) is False and all(
        type(metrics.get(name)) in (int, float) and math.isfinite(float(metrics[name]))
        and 0 <= float(metrics[name]) <= 1
        for name in required)


def gate(rows: dict[str, dict]) -> dict:
    """Apply the frozen semantic-vs-uniform/shuffled conjunction."""
    semantic = rows.get("semantic", {})
    comparisons = {}
    passed = bool(_finite_metrics(semantic))
    for reference in ("uniform", "shuffled"):
        candidate = semantic.get("metrics", {})
        baseline = rows.get(reference, {}).get("metrics", {})
        valid = (_finite_metrics(semantic) and _finite_metrics(rows.get(reference, {}))
                 and baseline.get("recall@20", 0) > 0 and baseline.get("ndcg@20", 0) > 0)
        values = {"relative_recall20": candidate["recall@20"] / baseline["recall@20"] - 1 if valid else None,
                  "relative_ndcg20": candidate["ndcg@20"] / baseline["ndcg@20"] - 1 if valid else None,
                  "delta_recall10": candidate["recall@10"] - baseline["recall@10"] if valid else None,
                  "delta_ndcg10": candidate["ndcg@10"] - baseline["ndcg@10"] if valid else None,
                  "reference_completed": _finite_metrics(rows.get(reference, {}))}
        values["pass"] = (valid
                          and values["relative_recall20"] >= .005
                          and values["relative_ndcg20"] >= .005
                          and values["delta_recall10"] >= -.0002
                          and values["delta_ndcg10"] >= -.0002)
        comparisons[reference] = values
        passed = passed and bool(values["pass"])
    return {"status": "pass" if passed else "fail", "passed": passed,
            "comparisons": comparisons}


def _owned_child(pid, command) -> bool:
    """Strict identity for this Popen's child and its new session only."""
    path = Path("/proc") / str(pid)
    try:
        args = [part.decode() for part in (path / "cmdline").read_bytes().split(b"\0") if part]
        stat = (path / "stat").read_text()
        parent_pid = int(stat[stat.rfind(")") + 2:].split()[1])
        return (args == list(command) and path.stat().st_uid == os.getuid()
                and parent_pid == os.getpid() and os.getpgid(pid) == pid)
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return False


def _kill_owned(process: subprocess.Popen, command) -> None:
    if process.poll() is not None:
        return
    if not _owned_child(process.pid, command):
        raise RuntimeError("refusing to stop a process with unmatched owned-child identity")
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is not None:
            return
        if not _owned_child(process.pid, command):
            raise RuntimeError("owned-child identity changed before SIGKILL")
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("owned child did not exit after SIGKILL") from exc


def build_command(dataset, mode, output, assets, *, smoke=False, prepare=False):
    command = [sys.executable, "-u", str(ROOT / "train.py"),
               "--dataset", dataset, "--assets", str(Path(assets).resolve())]
    if prepare:
        command.append("--prepare-assets")
    else:
        command += ["--mode", mode, "--output", str(Path(output).resolve())]
        if smoke:
            command.append("--smoke")
    return command


def verify_frozen(root=ROOT):
    """Independently guard source, isolated carrier and original input bytes."""
    root = Path(root)
    deployment = json.loads((root / "deployment.json").read_text())
    for name in ("carrier_manifest", "source_protection"):
        if sha256(root / (name + ".json")) != deployment[name + "_sha256"]:
            raise RuntimeError("protected manifest drift: " + name)
    expected = {str(root / name): digest for name, digest in deployment["bundle_sha256"].items()}
    manifest = json.loads((root / "carrier_manifest.json").read_text())
    expected.update({str(root / "carrier/src" / name): digest
                     for name, digest in manifest["carrier_sha256"].items()})
    for name, row in manifest["inputs"].items():
        if row["exists"]:
            expected[str(root / "carrier/data" / name)] = row["sha256"]
            expected[row["path"]] = row["sha256"]
    original = json.loads((root / "source_protection.json").read_text())
    license_row = original.get("license")
    if not isinstance(license_row, dict) or not license_row.get("exists"):
        raise RuntimeError("original source license protection is missing")
    expected[license_row["path"]] = license_row["sha256"]
    source_root = Path(original["source_root"])
    original_names = {str(path.relative_to(source_root / "src"))
                      for path in (source_root / "src").rglob("*") if path.is_file()}
    if original_names != set(original["source_sha256"]):
        raise RuntimeError("original source inventory changed")
    expected.update({str(source_root / "src" / name): digest
                     for name, digest in original["source_sha256"].items()})
    for row in original["inputs"].values():
        if row["exists"]:
            expected[row["path"]] = row["sha256"]
    verified_real = {}
    for path, digest in expected.items():
        resolved = str(Path(path).resolve())
        if resolved not in verified_real:
            verified_real[resolved] = sha256(path)
        if verified_real[resolved] != digest:
            raise RuntimeError("protected bytes changed: " + path)
    return {"protected_paths_count": len(expected), "protected_paths_match": True,
            "deployment_sha256": sha256(root / "deployment.json")}


def validate_preparation(assets, dataset):
    assets = Path(assets)
    receipt = json.loads((assets / "preparation.json").read_text())
    if (receipt.get("status") != "completed" or receipt.get("dataset") != dataset
            or receipt.get("cuda_initialized") is not False
            or not math.isfinite(float(receipt.get("elapsed_seconds", float("nan"))))):
        raise RuntimeError("invalid asset preparation receipt")
    for filename, key in (("arrays.npz", "array_sha256"), ("diagnostics.json", "diagnostics_sha256")):
        if sha256(assets / filename) != receipt.get(key):
            raise RuntimeError("prepared group asset changed: " + filename)
    return receipt


def validate_result(row, dataset, mode, smoke, output, prepared):
    """Completed results must carry numerical/provenance evidence, including smoke."""
    formal_check = dict(row, smoke=False)
    if (not _finite_metrics(formal_check) or row.get("dataset") != dataset
            or row.get("mode") != mode or row.get("seed") != 2026
            or row.get("smoke") is not smoke or row.get("test_loader_constructed") is not False
            or row.get("test_evaluated") is not False
            or row.get("protected_paths_match") is not True):
        raise RuntimeError("completed result violates frozen protocol")
    epochs = row.get("completed_epochs")
    updates = row.get("finite_optimizer_updates")
    if (type(epochs) is not int or epochs < 1 or epochs > (1 if smoke else 300)
            or type(updates) is not int or updates < epochs):
        raise RuntimeError("missing full finite epochs/optimizer updates")
    order = row.get("positive_epoch_hashes")
    initial = row.get("initial_state_sha256")
    if (not isinstance(order, list) or len(order) != epochs
            or not all(isinstance(value, str) and len(value) == 64 for value in order)
            or not isinstance(initial, str) or len(initial) != 64):
        raise RuntimeError("missing initialization/positive-order hashes")
    for key, expected in (("group_assets_sha256", prepared["array_sha256"]),
                          ("group_diagnostics_sha256", prepared["diagnostics_sha256"])):
        if row.get(key) != expected:
            raise RuntimeError("formal/smoke group assets changed")
    output = Path(output)
    for filename, key in (("best.pt", "checkpoint_sha256"),
                          ("per_user_validation.npz", "per_user_sha256")):
        if sha256(output / filename) != row.get(key):
            raise RuntimeError("completed result artifact changed: " + filename)
    import numpy as np
    with np.load(output / "per_user_validation.npz", allow_pickle=False) as saved:
        users = saved["user_ids"]
        lengths = saved["positive_lengths"]
        if (users.ndim != 1 or users.dtype.kind not in "iu" or users.size == 0
                or lengths.shape != users.shape or lengths.dtype.kind not in "iu"
                or np.any(lengths <= 0) or np.unique(users).size != users.size):
            raise RuntimeError("invalid validation user/positive population")
        population = hashlib.sha256()
        population.update(users.astype("<i8").tobytes())
        population.update(lengths.astype("<i8").tobytes())
        for metric in ("recall@10", "recall@20", "ndcg@10", "ndcg@20"):
            values = saved[metric.replace("@", "_")]
            if (values.shape != users.shape or not np.all(np.isfinite(values))
                    or abs(float(values.mean()) - row["metrics"][metric]) > 1e-12):
                raise RuntimeError("per-user validation arrays disagree with result")
    config = json.loads((output / "config.json").read_text())
    config.pop("checkpoint_dir", None)
    config.pop("epochs", None)  # engineering smoke differs only in allowed epoch ceiling.
    row = dict(row, output=str(output), validation_population_sha256=population.hexdigest(),
               validation_users=int(users.size), validation_positives=int(lengths.sum()),
               comparable_config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
               protected_paths=json.loads((output / "protected_paths.json").read_text()))
    if not math.isfinite(float(row.get("elapsed_seconds", float("nan")))):
        raise RuntimeError("nonfinite/missing child elapsed time")
    return row


def verify_comparability(rows):
    """Require shared initialization, populations, fixed assets and positive order."""
    rows = list(rows)
    if not rows:
        raise RuntimeError("no completed arms for comparability")
    first = rows[0]
    for row in rows[1:]:
        for key in ("initial_state_sha256", "group_assets_sha256", "group_diagnostics_sha256",
                    "validation_population_sha256", "validation_users", "validation_positives",
                    "comparable_config_sha256", "protected_paths"):
            if row.get(key) != first.get(key) or key not in row or key not in first:
                raise RuntimeError("fresh arms differ on " + key)
        left, right = first["positive_epoch_hashes"], row["positive_epoch_hashes"]
        overlap = min(len(left), len(right))
        if overlap < 1 or left[:overlap] != right[:overlap]:
            raise RuntimeError("positive training streams differ on overlapping epochs")
    return {"passed": True, "arms": len(rows),
            "overlap_epochs": min(len(row["positive_epoch_hashes"]) for row in rows),
            "same_initialization_assets_population_positive_order": True}


class Queue:
    def __init__(self):
        self.started = time.monotonic()
        self.deadline = self.started + QUEUE_SECONDS
        self.state = {"status": "starting", "queue_pid": os.getpid(),
                      "completed": [], "current_job": None,
                      "jobs": [], "completed_formal_arms": 0,
                      "test_evaluated": False, "started_unix": time.time(),
                      "deadline_unix": time.time() + QUEUE_SECONDS}

    @property
    def status_path(self):
        return ROOT / "pipeline_status.json"

    def update(self, **fields):
        self.state.update(fields, updated_unix=time.time())
        write_json(self.status_path, self.state)
        print(json.dumps(fields, sort_keys=True), flush=True)

    def remaining(self):
        remain = self.deadline - time.monotonic()
        if remain <= 0:
            raise TimeoutError("48-hour interest-negative queue budget exceeded")
        return remain

    def wait_resources(self, *, require_gpu=True):
        while True:
            self.remaining()
            snapshot = resource_snapshot()
            ok, reason = resources_admit(snapshot, require_gpu=require_gpu)
            self.update(stage="resource_admission" if ok else "waiting_resources",
                        resources=snapshot, resource_reason=reason)
            if ok:
                return snapshot
            time.sleep(min(30, self.remaining()))

    def run_child(self, dataset, mode, output, assets, *, smoke=False,
                  prepare=False):
        self.state["requested_job"] = {"dataset": dataset, "mode": mode,
                                        "smoke": smoke, "prepare": prepare}
        before = verify_frozen()
        self.wait_resources(require_gpu=not prepare)
        if verify_frozen() != before:
            raise RuntimeError("frozen deployment changed during resource wait")
        output = Path(output)
        if not prepare and output.exists():
            raise RuntimeError("refusing to reuse existing output: " + str(output))
        if prepare and Path(assets).exists():
            raise RuntimeError("refusing to reuse prepared TRAIN assets")
        log_path = ROOT / "logs" / ((dataset + "_prepare") if prepare else
                                     _arm_name(dataset, mode, smoke))
        log_path = log_path.with_suffix(".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = build_command(dataset, mode, output, assets, smoke=smoke, prepare=prepare)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if prepare else GPU_INDEX,
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                   OPENBLAS_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4",
                   NUMEXPR_MAX_THREADS="4", PYTHONUNBUFFERED="1",
                   PYTHONPATH=str(ROOT / "carrier/src") + os.pathsep + str(ROOT))
        started = time.monotonic()
        timeout = min(PREPARATION_SECONDS if prepare else
                      (SMOKE_SECONDS if smoke else JOB_SECONDS), self.remaining())
        with log_path.open("xb") as stream:
            process = subprocess.Popen(command, cwd=str(ROOT), env=env,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            self.update(stage="running", current_job={"name": log_path.stem,
                        "pid": process.pid, "command": command,
                        "started_unix": time.time(), "timeout_seconds": timeout,
                        "output": str(output), "prepare": prepare})
            try:
                while process.poll() is None:
                    remaining = min(started + timeout - time.monotonic(), self.remaining())
                    if remaining <= 0:
                        raise TimeoutError(log_path.stem + " exceeded budget")
                    try:
                        process.wait(timeout=min(10, remaining))
                    except subprocess.TimeoutExpired:
                        self.update(stage="running", current_job={"name": log_path.stem,
                                    "pid": process.pid, "command": command,
                                    "elapsed_seconds": time.monotonic() - started,
                                    "timeout_seconds": timeout, "output": str(output),
                                    "prepare": prepare})
            except BaseException:
                _kill_owned(process, command)
                raise
            finally:
                child = {"name": log_path.stem,
                            "pid": process.pid, "exit_code": process.returncode,
                            "elapsed_seconds": time.monotonic() - started,
                            "log": str(log_path), "prepare": prepare, "smoke": smoke,
                            "command": command}
                self.state["jobs"].append(child)
                self.update(current_job=None, last_child=child)
        verify_frozen()
        if process.returncode:
            raise RuntimeError(log_path.stem + " failed; inspect " + str(log_path))
        if prepare:
            return validate_preparation(assets, dataset)
        result_path = output / "result.json"
        if not result_path.is_file():
            raise RuntimeError("completed child omitted result.json")
        result = json.loads(result_path.read_text())
        if not _finite_metrics(dict(result, smoke=False)):
            raise RuntimeError("child result is missing finite completed metrics")
        result["output"] = str(output)
        return result

    def write_results(self, rows):
        lines = ["dataset\tmode\tstatus\tcompletion\tcontrol\trecall@10\tndcg@10\trecall@20\tndcg@20\tseconds\tfinite_updates\tepochs"]
        for dataset in DATASETS:
            for mode in MODES:
                row = rows.get((dataset, mode))
                decision_path = ROOT / (dataset + "_gate.json")
                decision = json.loads(decision_path.read_text()) if decision_path.is_file() else None
                if row is None:
                    requested = self.state.get("requested_job", {})
                    failed = (self.state.get("status") == "technical_failure"
                              and requested.get("dataset") == dataset and requested.get("mode") == mode
                              and not requested.get("smoke") and not requested.get("prepare"))
                    values = [dataset, mode, "crash" if failed else "pending", "missing", int(mode != "semantic")] + ["n/a"] * 7
                else:
                    metrics = row.get("metrics", {})
                    status = "keep" if mode != "semantic" else (
                        ("keep" if decision["passed"] else "discard") if decision else "pending")
                    values = [dataset, mode, status, row.get("status", "missing"), int(mode != "semantic")]
                    values += [str(metrics.get(k, "n/a")) for k in
                               ("recall@10", "ndcg@10", "recall@20", "ndcg@20")]
                    values.append(str(row.get("elapsed_seconds", "n/a")))
                    values += [str(row.get("finite_optimizer_updates", "n/a")), str(row.get("completed_epochs", "n/a"))]
                lines.append("\t".join(map(str, values)))
        pending = ROOT / ("results.tsv.pending." + str(os.getpid()))
        pending.write_text("\n".join(lines) + "\n")
        os.replace(str(pending), str(ROOT / "results.tsv"))

    def report(self, rows):
        gates = {dataset: json.loads((ROOT / (dataset + "_gate.json")).read_text())
                 for dataset in DATASETS if (ROOT / (dataset + "_gate.json")).is_file()}
        complete = len(rows)
        report = {"status": self.state["status"], "completed_formal_arms": complete,
                  "formal_arm_target": 6, "seed": 2026, "gates": gates,
                  "source_revision": json.loads((ROOT / "source_revision.json").read_text()),
                  "jobs": self.state["jobs"], "elapsed_queue_seconds": time.monotonic() - self.started,
                  "test_evaluated": False, "publication_evidence_established": False,
                  "novelty_established": False,
                  "results_tsv_sha256": sha256(ROOT / "results.tsv")}
        write_json(ROOT / "research_report.json", report)
        text = ["# Interest-negative pilot report", "", "Status: " + report["status"],
                "", f"Completed formal arms: {complete}/6; seed 2026. Engineering smokes are separate.",
                "", "Validation decisions use the complete two-reference conjunction. Test was not evaluated.",
                "A two-domain pilot pass requires independent confirmation and does not establish publication evidence or novelty.", ""]
        for dataset, decision in gates.items():
            text += [dataset + ": " + decision["status"], ""]
        text += ["See results.tsv for all six fixed arm rows and pipeline_status.json for process receipts.", ""]
        (ROOT / "REPORT.md").write_text("\n".join(text))

    def run(self):
        runs = ROOT / "runs"
        if runs.exists() or (ROOT / "pipeline_status.json").exists():
            raise RuntimeError("existing queue state; refusing automatic restart")
        runs.mkdir()
        self.update(status="running", stage="starting", time_budget_seconds=QUEUE_SECONDS)
        rows = {}
        self.write_results(rows)
        try:
            # Baby is the preregistered screening domain.  Sports is never
            # prepared or launched unless the entire fresh Baby gate passes.
            dataset = "baby"
            assets = ROOT / "groups" / dataset
            self.update(stage="frozen_input_check", dataset=dataset,
                        integrity=verify_frozen())
            prep = self.run_child(dataset, "uniform", ROOT / ("unused_prepare_" + dataset),
                                  assets, prepare=True)
            prep = validate_preparation(assets, dataset)
            self.update(stage="assets_prepared", dataset=dataset, preparation=prep)
            for mode in ("uniform", "semantic"):
                output = runs / _arm_name(dataset, mode, True)
                smoke = self.run_child(dataset, mode, output, assets, smoke=True)
                smoke = validate_result(smoke, dataset, mode, True, output, prep)
                self.update(stage="smoke_completed", dataset=dataset, mode=mode,
                            smoke_result=smoke)
                verify_frozen()
            formal_rows = []
            for mode in MODES:
                output = runs / _arm_name(dataset, mode, False)
                row = self.run_child(dataset, mode, output, assets)
                row = validate_result(row, dataset, mode, False, output, prep)
                formal_rows.append(row)
                rows[(dataset, mode)] = row
                self.state["completed"].append({"dataset": dataset, "mode": mode,
                                                 "status": row["status"]})
                self.write_results(rows)
                self.update(stage="formal_arm_completed", completed=self.state["completed"],
                            last_result=row, integrity=verify_frozen())
            comparability = verify_comparability(formal_rows)
            decision = gate({mode: rows[(dataset, mode)] for mode in MODES})
            decision["comparability"] = comparability
            write_json(ROOT / (dataset + "_gate.json"), decision)
            self.write_results(rows)
            self.update(stage="domain_gate", dataset=dataset, gate=decision)
            if not decision["passed"]:
                self.update(status="completed_fail", stage="baby_gate_failed",
                            finished_unix=time.time(), test_evaluated=False)
                self.report(rows)
                return

            # Baby passed all fixed comparisons; now prepare and run Sports.
            dataset = "sports"
            assets = ROOT / "groups" / dataset
            self.update(stage="frozen_input_check", dataset=dataset,
                        integrity=verify_frozen())
            prep = self.run_child(dataset, "uniform", ROOT / ("unused_prepare_" + dataset),
                                  assets, prepare=True)
            prep = validate_preparation(assets, dataset)
            self.update(stage="assets_prepared", dataset=dataset, preparation=prep)
            formal_rows = []
            for mode in MODES:
                output = runs / _arm_name(dataset, mode, False)
                row = self.run_child(dataset, mode, output, assets)
                row = validate_result(row, dataset, mode, False, output, prep)
                formal_rows.append(row)
                rows[(dataset, mode)] = row
                self.state["completed"].append({"dataset": dataset, "mode": mode,
                                                 "status": row["status"]})
                self.write_results(rows)
                self.update(stage="formal_arm_completed", completed=self.state["completed"],
                            last_result=row, integrity=verify_frozen())
            comparability = verify_comparability(formal_rows)
            decision = gate({mode: rows[(dataset, mode)] for mode in MODES})
            decision["comparability"] = comparability
            write_json(ROOT / (dataset + "_gate.json"), decision)
            self.write_results(rows)
            self.update(stage="domain_gate", dataset=dataset, gate=decision)
            if not decision["passed"]:
                self.update(status="completed_fail", stage="sports_gate_failed",
                            finished_unix=time.time(), test_evaluated=False)
                self.write_results(rows)
                self.report(rows)
                return
        except BaseException as exc:
            self.update(status="technical_failure", stage="child_failure",
                        error=repr(exc), traceback=traceback.format_exc(),
                        finished_unix=time.time(), test_evaluated=False)
            try:
                self.write_results(rows)
                self.report(rows)
            except Exception as report_exc:
                self.update(report_error=repr(report_exc))
            raise
        self.update(status="pilot_passed_needs_confirmation", stage="done",
                    finished_unix=time.time(), test_evaluated=False)
        self.write_results(rows)
        self.report(rows)


def main():
    Queue().run()


if __name__ == "__main__":
    main()
