"""Immutable audit, test, and single-process manager for review residual."""
import argparse
import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path


LOCAL = Path(__file__).resolve().parent
REMOTE = "/data/sunzixuan/review_residual_20261009_v1"
PYTHON = "/data/sunzixuan/miniconda3/envs/MMRec/bin/python"
STAGE1 = "/data/sunzixuan/rave_20260906_stage1"
DUAL = "/data/sunzixuan/dual_witness_20261009_v1"
PREVIOUS = "/data/sunzixuan/review_signal_20261009_v1"
CARRIER = "/data/sunzixuan/candidate_gate_20261009_v1"
BASELINE = "/data/sunzixuan/bridge_official"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending." + str(os.getpid()))
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _previous_deployment():
    return json.loads((LOCAL.parent / "20261009_review_signal" / "deployment.json").read_text())


def audit_expectations():
    previous = json.loads((LOCAL.parent / "20261009_review_signal" / "input_audit.json").read_text())
    deployment = _previous_deployment()
    files = {name: dict(row) for name, row in previous["files"].items()}
    dependencies = {name: dict(row) for name, row in previous["dependencies"].items()}
    old_sha = deployment["bundle_sha256"]
    dependencies.update({
        "review_signal_experiment": {"path": PREVIOUS + "/experiment.py", "sha256": old_sha["experiment.py"]},
        "review_signal_mapping": {"path": PREVIOUS + "/mapping.py", "sha256": old_sha["mapping.py"]},
        "review_signal_screen_core": {"path": PREVIOUS + "/screen_core.py", "sha256": old_sha["screen_core.py"]},
        "review_signal_manage": {"path": PREVIOUS + "/manage.py", "sha256": old_sha["manage.py"]},
        "review_signal_input_audit": {"path": PREVIOUS + "/input_audit.json", "sha256": old_sha["input_audit.json"]},
    })
    return {
        "files": files,
        "dependencies": dependencies,
        "baseline_carrier": previous["baseline_carrier"],
        "identity_matching": previous["identity_matching"],
        "heldout_target_hash": previous["heldout_target_hash"],
        "carrier_root": CARRIER,
        "carrier_source_revision": json.loads(
            (LOCAL.parent / "20261009_candidate_gate" / "source_revision.json").read_text())["commit"],
    }


def deployment_payload():
    required = {"manage.py", "experiment.py", "residual_core.py", "test_residual_core.py",
                "test_experiment.py", "PROTOCOL.md", "program.md", "input_audit.json"}
    if not (LOCAL / "input_audit.json").is_file():
        raise RuntimeError("Run the read-only audit action before deployment")
    names = sorted(required | {path.name for path in LOCAL.glob("*.py")})
    payload = {name: base64.b64encode((LOCAL / name).read_bytes()).decode("ascii") for name in names}
    git = ["git", "-c", "safe.directory=" + LOCAL.as_posix()]
    commit = subprocess.check_output(git + ["rev-parse", "HEAD"], cwd=str(LOCAL), text=True).strip()
    dirty = subprocess.check_output(git + ["status", "--porcelain", "--untracked-files=all"],
                                    cwd=str(LOCAL), text=True).strip()
    if dirty:
        raise RuntimeError("Commit the isolated bundle before deployment")
    payload["source_revision.json"] = base64.b64encode(json.dumps(
        {"commit": commit, "local_directory": LOCAL.as_posix(), "working_tree_clean": True},
        sort_keys=True).encode("utf-8")).decode("ascii")
    baseline = LOCAL.parent / "20260910_baseline" / "remote_audit.txt"
    if not baseline.is_file():
        raise RuntimeError("Frozen baseline audit is missing")
    first = baseline.read_text(encoding="utf-8").lstrip()
    record, _ = json.JSONDecoder().raw_decode(first)
    payload["baseline_audit.json"] = base64.b64encode(json.dumps(record, sort_keys=True).encode("utf-8")).decode("ascii")
    return payload


def remote_source(action, payload=None, expected=None):
    configuration = repr({"root": REMOTE, "action": action, "payload": payload or {},
                          "expected": expected, "python": PYTHON, "stage1": STAGE1,
                          "dual": DUAL, "carrier": CARRIER, "baseline": BASELINE})
    return "configuration = " + configuration + r'''
import base64, hashlib, json, os, subprocess, sys, time
from pathlib import Path

root = Path(configuration["root"])
action = configuration["action"]
python = configuration["python"]
env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
           OPENBLAS_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4", PYTHONUNBUFFERED="1",
           PYTHONPATH=os.pathsep.join([str(root), configuration["dual"],
                                       configuration["stage1"] + "/vendor"]))
env.pop("CUDA_VISIBLE_DEVICES", None)
test_env = dict(env, CUDA_VISIBLE_DEVICES="")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def atomic(path, content):
    temporary = path.with_name(path.name + ".pending." + str(os.getpid()))
    with temporary.open("xb") as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)

def atomic_json(path, value):
    atomic(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())

def resources():
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1); mem[key] = int(value.strip().split()[0]) * 1024
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return {"available_ram_bytes": mem.get("MemAvailable", 0), "available_cpu_count": cpus,
            "required_ram_bytes": 12 * 1024 ** 3, "cpu_threads": 4,
            "gpu_selection_owner": "experiment.py", "tests_cuda_visible_devices": ""}

def require_resources():
    value = resources()
    if value["available_cpu_count"] is None or value["available_cpu_count"] < 4:
        raise RuntimeError("At least four available CPUs are required")
    if value["available_ram_bytes"] < value["required_ram_bytes"]:
        raise RuntimeError("At least 12 GiB available RAM is required")
    return value

def verify_rows(rows):
    for name, row in rows.items():
        path = Path(row["path"])
        if not path.is_file() or sha(path) != row["sha256"]:
            raise RuntimeError("Frozen input changed: " + name)
        if "bytes" in row and path.stat().st_size != row["bytes"]:
            raise RuntimeError("Frozen input size changed: " + name)

def check_inputs():
    manifest = json.loads((root / "input_audit.json").read_text())
    if manifest.get("status") != "completed": raise RuntimeError("Completed audit required")
    verify_rows(manifest["files"]); verify_rows(manifest["dependencies"])
    verify_rows(manifest["baseline_carrier"])
    if len(manifest["baseline_carrier"]) != 35: raise RuntimeError("Expected 35 carrier files")
    baseline = json.loads((root / "baseline_audit.json").read_text())
    original = Path(configuration["baseline"])
    for relative, row in baseline["files"].items():
        if sha(original / relative) != row["sha256"]: raise RuntimeError("Baseline source changed")
    for info in baseline["checkpoints"].values():
        for row in info["inputs"].values():
            path = Path(row["path"])
            if path.is_file() != row["exists"] or (row["exists"] and sha(path) != row["sha256"]):
                raise RuntimeError("Baseline input changed: " + str(path))

def check_bundle():
    manifest = json.loads((root / "deployment.json").read_text())
    for name, digest in manifest["bundle_sha256"].items():
        if sha(root / name) != digest: raise RuntimeError("Deployment drift: " + name)
    return manifest

if action == "audit":
    expected = configuration["expected"]
    verify_rows(expected["files"]); verify_rows(expected["dependencies"])
    verify_rows(expected["baseline_carrier"])
    carrier = Path(expected["carrier_root"])
    revision = carrier / "source_revision.json"
    deployment = json.loads((carrier / "deployment.json").read_text())
    if sha(revision) != deployment["bundle_sha256"]["source_revision.json"]:
        raise RuntimeError("Carrier revision receipt changed")
    if json.loads(revision.read_text())["commit"] != expected["carrier_source_revision"]:
        raise RuntimeError("Carrier source revision changed")
    output = {"status": "completed", "checked_unix": time.time(), "files": {},
              "dependencies": {}, "baseline_carrier": {},
              "identity_matching": expected["identity_matching"],
              "heldout_target_hash": expected["heldout_target_hash"],
              "carrier_manifest": {"path": str(carrier / "deployment.json"), "sha256": sha(carrier / "deployment.json")},
              "encoding_or_training_started": False, "review_text_exported": False,
              "reviewer_identity_exported": False}
    for group in ("files", "dependencies", "baseline_carrier"):
        output[group] = {name: dict(row, bytes=Path(row["path"]).stat().st_size)
                         for name, row in expected[group].items()}
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(0)

if action in ("deploy", "verify", "launch"):
    if (root / "launch.json").exists() or (root / "launch.lock").exists():
        raise RuntimeError("This bundle is already launched")
    root.mkdir(exist_ok=True)
    hashes = {name: hashlib.sha256(base64.b64decode(value)).hexdigest()
              for name, value in configuration["payload"].items()}
    manifest_path = root / "deployment.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("bundle_sha256") != hashes:
        raise RuntimeError("Immutable bundle differs")
    for name, encoded in configuration["payload"].items():
        path = root / name; content = base64.b64decode(encoded)
        if path.exists() and sha(path) != hashes[name]: raise RuntimeError("Existing file differs: " + name)
        if not path.exists(): atomic(path, content)
    check_inputs()
    if not manifest_path.exists():
        atomic_json(manifest_path, {"bundle_sha256": hashes, "created_unix": time.time(),
            "root": str(root), "cpu_threads": 4, "gpu_selection_owner": "experiment.py",
            "experiment_budget_seconds": 21600, "baseline_carrier_files_checked": 35})
    if action == "deploy":
        print(json.dumps({"deployed": str(root), "files": len(hashes)}, indent=2)); raise SystemExit(0)

if action in ("test", "verify", "launch"):
    if (root / "launch.json").exists() or (root / "launch.lock").exists(): raise RuntimeError("Already launched")
    check_bundle(); check_inputs(); resource_record = require_resources()
    runner = """import json,sys,unittest
suite=unittest.defaultTestLoader.discover(sys.argv[1],pattern='test_*.py')
count=suite.countTestCases()
if count == 0: raise RuntimeError('No tests discovered')
result=unittest.TextTestRunner(verbosity=2).run(suite)
print('REVIEW_RESIDUAL_TEST_SUMMARY='+json.dumps({'tests':count,'successful':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped)}))
sys.exit(0 if result.wasSuccessful() else 1)
"""
    tested = subprocess.run([python, "-u", "-c", runner, str(root)], cwd=str(root), env=test_env,
                            capture_output=True, text=True, timeout=360)
    summary = next((json.loads(line.split("=", 1)[1]) for line in tested.stdout.splitlines()
                    if line.startswith("REVIEW_RESIDUAL_TEST_SUMMARY=")), None)
    receipt = {"exit_code": tested.returncode, "checked_unix": time.time(),
               "deployment_sha256": sha(root / "deployment.json"), "test_summary": summary,
               "resources": resource_record, "cuda_visible_devices": "",
               "stdout": tested.stdout, "stderr": tested.stderr}
    atomic_json(root / "test_receipt.json", receipt)
    print(tested.stdout); print(tested.stderr)
    if tested.returncode or not summary or not summary.get("successful"):
        raise RuntimeError("Tests failed; launch refused")

if action in ("start", "launch"):
    if (root / "launch.json").exists() or (root / "launch.lock").exists(): raise RuntimeError("Already launched")
    check_bundle(); check_inputs(); tested = json.loads((root / "test_receipt.json").read_text())
    if tested["exit_code"] or tested["deployment_sha256"] != sha(root / "deployment.json"):
        raise RuntimeError("Current bundle needs passing tests")
    resource_record = require_resources()
    with (root / "launch.lock").open("x") as stream: stream.write(str(time.time()))
    command = [python, "-u", str(root / "experiment.py")]
    with (root / "pipeline.log").open("xb") as log:
        process = subprocess.Popen(command, cwd=str(root), stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True, env=env)
    atomic_json(root / "launch.json", {"pid": process.pid, "command": command, "root": str(root),
        "launched_unix": time.time(), "resources": resource_record,
        "deployment_sha256": sha(root / "deployment.json")})
    print(json.dumps({"pid": process.pid, "command": command}, indent=2))
elif action == "status":
    record = {"root": str(root), "checked_unix": time.time(), "artifacts": {}, "logs": {}, "resources": resources()}
    for name in ("launch.json", "deployment.json", "source_revision.json", "test_receipt.json",
                 "input_audit.json", "progress.json", "diagnosis.json"):
        path = root / name
        if path.is_file(): record["artifacts"][name] = json.loads(path.read_text())
    results = root / "results.tsv"
    if results.is_file(): record["results_tsv"] = results.read_text()
    log = root / "pipeline.log"
    if log.is_file(): record["logs"]["pipeline.log"] = log.read_bytes()[-4000:].decode("utf-8", "replace")
    receipt = record["artifacts"].get("launch.json", {}); pid = receipt.get("pid")
    proc = Path("/proc/" + str(pid)); alive = bool(pid and proc.is_dir()); matches = False
    if alive:
        try: matches = str(root / "experiment.py") in (proc / "cmdline").read_text().split("\x00")
        except FileNotFoundError: alive = False
    record["process_alive"] = alive and matches; record["process_command_matches"] = matches
    print(json.dumps(record, indent=2, allow_nan=False))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="verify",
                        choices=("audit", "deploy", "test", "verify", "start", "launch", "status"))
    args = parser.parse_args()
    payload = deployment_payload() if args.action in ("deploy", "verify", "launch") else {}
    expected = audit_expectations() if args.action == "audit" else None
    import sys
    sys.path.insert(0, str(LOCAL.parent / "20260905"))
    from remote_session import Session
    session = Session("213")
    try:
        code, output = session.run_python(remote_source(args.action, payload, expected), PYTHON,
                                          360 if args.action != "status" else 180)
        atomic_text(LOCAL / (args.action + "_receipt.txt"), output + "\n")
        if code:
            print(output)
            raise SystemExit(code)
        if args.action == "audit":
            record, _ = json.JSONDecoder().raw_decode(output.lstrip())
            path = LOCAL / "input_audit.json"
            if path.exists():
                frozen = json.loads(path.read_text())
                for key in ("files", "dependencies", "baseline_carrier", "heldout_target_hash"):
                    if record[key] != frozen[key]: raise RuntimeError("Existing audit changed: " + key)
                print(json.dumps({"audit_unchanged": True, "checked_unix": record["checked_unix"]}, indent=2))
            else:
                atomic_text(path, json.dumps(record, indent=2, sort_keys=True) + "\n")
                print(json.dumps(record, indent=2))
            return
        if args.action == "status":
            print(output); return
        print(output)
    finally:
        session.close()


if __name__ == "__main__":
    main()
