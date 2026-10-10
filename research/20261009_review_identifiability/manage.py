"""Freeze, test and launch one CPU-only support-cache expression audit."""
import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


LOCAL = Path(__file__).resolve().parent
REMOTE = "/data/sunzixuan/review_identifiability_20261009_v1"
PREVIOUS = "/data/sunzixuan/review_residual_20261009_v1"
PYTHON = "/data/sunzixuan/miniconda3/envs/MMRec/bin/python"
BUNDLE_FILES = ("PROTOCOL.md", "program.md", "core.py", "experiment.py",
                "manage.py", "test_core.py", "test_manage.py", "input_audit.json")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    pending = path.with_name(path.name + ".pending." + str(os.getpid()))
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    pending.replace(path)


def audit_expectations():
    prior = LOCAL.parent / "20261009_review_residual"
    diagnosis = json.loads((prior / "diagnosis.json").read_text())
    deployment = json.loads((prior / "deployment.json").read_text())
    audit = json.loads((prior / "input_audit.json").read_text())
    revision = json.loads((prior / "source_revision.json").read_text())
    if diagnosis["status"] != "completed" or diagnosis["source_revision"] != revision["commit"]:
        raise RuntimeError("A completed, frozen F25 diagnosis is required")
    files = {
        "review_embeddings": {"path": PREVIOUS + "/cache/review_embeddings.npy",
                              "sha256": diagnosis["encoding_cache_sha256"]["review_embeddings.npy"]},
        "review_index": {"path": PREVIOUS + "/cache/review_index.npz",
                         "sha256": diagnosis["encoding_cache_sha256"]["review_index.npz"]},
        "global_bank": dict(audit["files"]["global_bank"]),
        "residual_core": {"path": PREVIOUS + "/residual_core.py",
                          "sha256": deployment["bundle_sha256"]["residual_core.py"]},
        "predecessor_deployment": {"path": PREVIOUS + "/deployment.json",
                                   "sha256": diagnosis["deployment_manifest_sha256"]},
        "predecessor_input_audit": {"path": PREVIOUS + "/input_audit.json",
                                   "sha256": diagnosis["input_audit_sha256"]},
        "predecessor_revision": {"path": PREVIOUS + "/source_revision.json",
                                 "sha256": deployment["bundle_sha256"]["source_revision.json"]},
    }
    # Freeze the remote diagnosis bytes after verifying its entire content
    # against the prior result, rather than assuming a local JSON reformat is identical.
    return {"files": files, "predecessor_diagnosis": diagnosis,
            "predecessor_commit": revision["commit"], "predecessor_root": PREVIOUS}


def deployment_payload():
    if not (LOCAL / "input_audit.json").is_file():
        raise RuntimeError("Run the read-only input audit before deployment")
    git = ["git", "-c", "safe.directory=" + LOCAL.as_posix()]
    commit = subprocess.check_output(git + ["rev-parse", "HEAD"], cwd=str(LOCAL), text=True).strip()
    if subprocess.check_output(git + ["status", "--porcelain", "--untracked-files=all"],
                               cwd=str(LOCAL), text=True).strip():
        raise RuntimeError("Commit the isolated, clean bundle before deployment")
    payload = {name: base64.b64encode((LOCAL / name).read_bytes()).decode("ascii")
               for name in BUNDLE_FILES}
    payload["source_revision.json"] = base64.b64encode(json.dumps(
        {"commit": commit, "working_tree_clean": True, "local_directory": LOCAL.as_posix()},
        sort_keys=True).encode()).decode("ascii")
    return payload


def remote_source(action, payload=None, expected=None):
    config = repr({"root": REMOTE, "python": PYTHON, "action": action,
                   "payload": payload or {}, "expected": expected})
    return "configuration = " + config + r'''
import base64, hashlib, json, os, subprocess, time
from pathlib import Path
root = Path(configuration["root"])
action = configuration["action"]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4",
           MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4",
           PYTHONUNBUFFERED="1", PYTHONPATH=str(root))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def atomic(path, content):
    pending = path.with_name(path.name + ".pending." + str(os.getpid()))
    with pending.open("xb") as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    pending.replace(path)

def atomic_json(path, value):
    atomic(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode())

def verify_rows(rows):
    for name, row in rows.items():
        path = Path(row["path"])
        if not path.is_file() or sha(path) != row["sha256"]:
            raise RuntimeError("Frozen input changed: " + name)
        if "bytes" in row and path.stat().st_size != row["bytes"]:
            raise RuntimeError("Frozen input size changed: " + name)

def check_inputs():
    manifest = json.loads((root/"input_audit.json").read_text())
    if manifest["status"] != "completed": raise RuntimeError("Completed input audit required")
    verify_rows(manifest["files"])
    return manifest

def check_bundle():
    manifest = json.loads((root/"deployment.json").read_text())
    for name, digest in manifest["bundle_sha256"].items():
        if sha(root/name) != digest: raise RuntimeError("Immutable bundle changed: " + name)
    return manifest

def resources():
    available = next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    cpus = len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else os.cpu_count()
    return {"available_ram_bytes": available, "available_cpu_count": cpus,
            "cpu_threads": 4, "required_ram_bytes": 12*1024**3, "gpu_disabled": True}

def require_resources():
    record = resources()
    if record["available_cpu_count"] < 4 or record["available_ram_bytes"] < record["required_ram_bytes"]:
        raise RuntimeError("Four CPUs and 12 GiB available host RAM are required")
    return record

def unlaunched():
    if any((root/name).exists() for name in ("launch.json","launch.lock","diagnosis.json","cache")):
        raise RuntimeError("Audit already launched or has partial execution artifacts")

if action == "status":
    record = {"root":str(root), "checked_unix":time.time(), "artifacts":{}, "resources":resources()}
    for name in ("input_audit.json","deployment.json","source_revision.json","test_receipt.json",
                 "launch.json","progress.json","diagnosis.json"):
        if (root/name).is_file(): record["artifacts"][name] = json.loads((root/name).read_text())
    if (root/"results.tsv").is_file(): record["results_tsv"]=(root/"results.tsv").read_text()
    if (root/"pipeline.log").is_file():
        record["log_tail"]=(root/"pipeline.log").read_bytes()[-5000:].decode("utf-8","replace")
    pid=record["artifacts"].get("launch.json",{}).get("pid")
    proc=Path("/proc/"+str(pid)); matches=False
    if pid and proc.is_dir():
        try: matches=str(root/"experiment.py") in (proc/"cmdline").read_text().split("\x00")
        except FileNotFoundError: pass
    record["process_alive"]=matches; record["process_command_matches"]=matches
    print(json.dumps(record,allow_nan=False)); raise SystemExit(0)

if action == "audit":
    expected=configuration["expected"]; verify_rows(expected["files"])
    previous=Path(expected["predecessor_root"])
    diagnosis_path=previous/"diagnosis.json"
    if json.loads(diagnosis_path.read_text()) != expected["predecessor_diagnosis"]:
        raise RuntimeError("Predecessor diagnosis content changed")
    revision=json.loads((previous/"source_revision.json").read_text())
    if revision["commit"] != expected["predecessor_commit"]:
        raise RuntimeError("Predecessor revision changed")
    files=dict(expected["files"], predecessor_diagnosis={"path":str(diagnosis_path),"sha256":sha(diagnosis_path)})
    output={"status":"completed","checked_unix":time.time(),"predecessor_commit":revision["commit"],
            "files":{name:dict(row,bytes=Path(row["path"]).stat().st_size) for name,row in files.items()},
            "raw_reviews_read":False,"interactions_read":False,"gpu_used":False,
            "encoding_or_training_started":False,"validation_evaluated":False,"test_evaluated":False}
    print(json.dumps(output,allow_nan=False)); raise SystemExit(0)

if action in ("deploy","verify"):
    unlaunched(); root.mkdir(exist_ok=True)
    hashes={name:hashlib.sha256(base64.b64decode(value)).hexdigest()
            for name,value in configuration["payload"].items()}
    deployment=root/"deployment.json"
    if deployment.exists() and json.loads(deployment.read_text())["bundle_sha256"] != hashes:
        raise RuntimeError("A different immutable bundle already exists")
    for name,encoded in configuration["payload"].items():
        path=root/name
        if path.exists() and sha(path)!=hashes[name]: raise RuntimeError("Existing payload differs: "+name)
        if not path.exists(): atomic(path,base64.b64decode(encoded))
    check_inputs()
    if not deployment.exists():
        atomic_json(deployment,{"root":str(root),"bundle_sha256":hashes,"created_unix":time.time(),
            "cpu_threads":4,"gpu_disabled":True,"experiment_budget_seconds":3600})
    if action=="deploy":
        print(json.dumps({"deployed":str(root),"files":len(hashes)})); raise SystemExit(0)

if action in ("test","verify"):
    unlaunched(); check_bundle(); check_inputs(); resource_record=require_resources()
    runner="""import json,sys,unittest
suite=unittest.defaultTestLoader.discover(sys.argv[1],pattern='test_*.py')
count=suite.countTestCases()
if count==0: raise RuntimeError('No tests discovered')
result=unittest.TextTestRunner(verbosity=2).run(suite)
summary={'tests':count,'successful':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped)}
print('IDENTIFIABILITY_TEST_SUMMARY='+json.dumps(summary))
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
"""
    tested=subprocess.run([configuration["python"],"-u","-c",runner,str(root)],cwd=str(root),env=env,
                          capture_output=True,text=True,timeout=180)
    summary=next((json.loads(line.split("=",1)[1]) for line in tested.stdout.splitlines()
                  if line.startswith("IDENTIFIABILITY_TEST_SUMMARY=")),None)
    check_bundle(); check_inputs()
    receipt={"exit_code":tested.returncode,"checked_unix":time.time(),"test_summary":summary,
             "deployment_sha256":sha(root/"deployment.json"),"resources":resource_record,
             "cuda_visible_devices":"","stdout":tested.stdout,"stderr":tested.stderr}
    atomic_json(root/"test_receipt.json",receipt)
    print(json.dumps(receipt,allow_nan=False))
    if tested.returncode or not summary or not summary["successful"] or summary["skipped"]:
        raise RuntimeError("Actual-environment tests failed; launch refused")

if action == "start":
    unlaunched(); check_bundle(); check_inputs()
    receipt=json.loads((root/"test_receipt.json").read_text()); summary=receipt["test_summary"]
    if (receipt["exit_code"] or receipt["deployment_sha256"] != sha(root/"deployment.json")
            or not summary or not summary["successful"] or summary["skipped"]):
        raise RuntimeError("Passing tests for this exact deployment are required")
    resource_record=require_resources()
    with (root/"launch.lock").open("x") as stream: stream.write(str(time.time()))
    command=[configuration["python"],"-u",str(root/"experiment.py")]
    with (root/"pipeline.log").open("xb") as log:
        process=subprocess.Popen(command,cwd=str(root),env=env,stdout=log,stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL,start_new_session=True)
    launch={"pid":process.pid,"command":command,"root":str(root),"launched_unix":time.time(),
            "resources":resource_record,"deployment_sha256":sha(root/"deployment.json")}
    atomic_json(root/"launch.json",launch); print(json.dumps(launch,allow_nan=False))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "audit", "deploy", "verify", "test", "start"))
    args = parser.parse_args()
    payload = deployment_payload() if args.action in ("deploy", "verify") else None
    expected = audit_expectations() if args.action == "audit" else None
    sys.path.insert(0, str(LOCAL.parent / "20260905"))
    from remote_session import Session
    session = Session("213")
    try:
        code, output = session.run_python(remote_source(args.action, payload, expected), PYTHON, 240)
    finally:
        session.close()
    (LOCAL / (args.action + "_receipt.txt")).write_text(output+"\n", encoding="utf-8")
    if code:
        print(output); raise SystemExit(code)
    record, _ = json.JSONDecoder().raw_decode(output.lstrip())
    if args.action == "audit":
        destination = LOCAL / "input_audit.json"
        if destination.exists():
            existing = json.loads(destination.read_text())
            if any(existing[key] != record[key] for key in ("files", "predecessor_commit", "status")):
                raise RuntimeError("The frozen local input audit changed")
        else:
            atomic_json(destination, record)
        atomic_json(LOCAL / "input_audit_receipt.json", record)
        print(json.dumps({"audited_files":len(record["files"]),"predecessor_commit":record["predecessor_commit"]}))
    elif args.action == "status":
        atomic_json(LOCAL / "latest_status.json", record)
        for name, value in record["artifacts"].items():
            if name != "input_audit.json": atomic_json(LOCAL / name, value)
        if "results_tsv" in record: (LOCAL/"results.tsv").write_text(record["results_tsv"],encoding="utf-8")
        print(json.dumps({"process_alive":record["process_alive"],"root":REMOTE,
            "artifacts":list(record["artifacts"]),"diagnosis":record["artifacts"].get("diagnosis.json"),
            "progress":record["artifacts"].get("progress.json")},allow_nan=False))
    else:
        if args.action in ("test", "verify", "start"):
            destination = "test_receipt.json" if args.action in ("test", "verify") else "launch.json"
            atomic_json(LOCAL / destination, record)
        print(json.dumps(record,allow_nan=False))


if __name__ == "__main__":
    main()
