"""Run the preregistered CPU mechanism diagnostics in a bounded queue."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/data/sunzixuan/bridge_official/data")
PREVIOUS = Path("/data/sunzixuan/candidate_gate_20261009_v1")
DEADLINE = time.time() + 8 * 3600
STATE = {
    "status": "starting",
    "stage": "initializing",
    "started_unix": time.time(),
    "datasets": {},
}


def update(**fields):
    STATE.update(fields)
    STATE["updated_unix"] = time.time()
    pending = ROOT / "pipeline_status.pending.json"
    pending.write_text(json.dumps(STATE, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(ROOT / "pipeline_status.json")
    print(json.dumps(fields, allow_nan=False), flush=True)


def predecessor_state():
    path = PREVIOUS / "pipeline_status.json"
    state = json.loads(path.read_text()) if path.is_file() else {}
    active = state.get("active_job") or {}
    launch_path = PREVIOUS / "launch.json"
    launch = json.loads(launch_path.read_text()) if launch_path.is_file() else {}
    alive = False
    for pid, expected in ((launch.get("pid"), "run_pipeline.py"), (active.get("pid"), "")):
        if not pid:
            continue
        try:
            command = Path("/proc/" + str(pid) + "/cmdline").read_bytes().decode(errors="replace")
            alive = alive or (str(PREVIOUS) in command and expected in command)
        except FileNotFoundError:
            pass
    return state, alive


def wait_for_predecessor():
    while time.time() < DEADLINE:
        state, alive = predecessor_state()
        if not alive:
            if state.get("status") != "closed_validation_gate":
                raise RuntimeError("Unexpected predecessor status: " + str(state.get("status")))
            update(stage="predecessor_closed", predecessor={"root": str(PREVIOUS),
                   "status": state.get("status"), "process_alive": False})
            return
        update(stage="waiting_for_previous_queue", predecessor={"root": str(PREVIOUS),
               "status": state.get("status"), "process_alive": True,
               "active_job": state.get("active_job")})
        time.sleep(30)
    raise TimeoutError("predecessor wait exceeded eight-hour queue budget")


def available_ram_bytes():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0]) * 1024
    return values.get("MemAvailable", 0)


def terminate_child(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def run_dataset(dataset):
    if available_ram_bytes() < 12 * 1024 ** 3:
        raise RuntimeError("less than 12 GiB host RAM is available")
    output = ROOT / "diagnostics" / dataset
    if output.exists():
        raise RuntimeError("diagnostic output already exists: " + str(output))
    output.parent.mkdir(exist_ok=True)
    log_path = ROOT / "logs" / (dataset + ".log")
    command = [sys.executable, "-u", str(ROOT / "diagnose.py"), "--dataset", dataset,
               "--data-root", str(DATA_ROOT), "--output", str(output)]
    started = time.monotonic()
    update(stage="diagnostic_running", active_job={"dataset": dataset, "command": command})
    with log_path.open("xb") as log:
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True,
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4",
                                            MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                                            PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT)))
        expires = min(time.time() + 2 * 3600, DEADLINE)
        try:
            while process.poll() is None:
                if time.time() >= expires:
                    raise TimeoutError(dataset + " diagnostic exceeded two-hour limit")
                update(active_job={"dataset": dataset, "pid": process.pid,
                                   "elapsed_seconds": time.monotonic() - started,
                                   "ram_available_bytes": available_ram_bytes()})
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            terminate_child(process)
            raise
    if process.returncode:
        raise RuntimeError(dataset + " diagnostic failed; see " + str(log_path))
    result = json.loads((output / "diagnosis.json").read_text())
    tsv_path = ROOT / "results.tsv"
    with tsv_path.open("a") as stream:
        if tsv_path.stat().st_size == 0:
            stream.write("dataset\tarm\tmean_auc\tpositive_coverage\tstatus\tseconds\n")
        for mode, arm in result["arms"].items():
            stream.write("{}\t{}\t{:.10f}\t{:.10f}\t{}\t{:.3f}\n".format(
                dataset, mode, arm["mean_auc"], arm["positive_nonzero_coverage"],
                result["decision"]["status"], arm["seconds"]))
    update(active_job=None, datasets={**STATE["datasets"], dataset: result})
    return result


def main():
    ROOT.joinpath("logs").mkdir(exist_ok=True)
    update(status="running", stage="waiting_for_predecessor")
    wait_for_predecessor()
    baby = run_dataset("baby")
    if baby["decision"]["status"] != "keep":
        update(status="closed_mechanism_gate", stage="baby_discard", reason="baby diagnostic failed")
        return
    sports = run_dataset("sports")
    if sports["decision"]["status"] != "keep":
        update(status="closed_mechanism_gate", stage="sports_discard", reason="sports diagnostic failed")
        return
    update(status="diagnostic_passed_needs_formal_preregistration",
           stage="diagnostics_complete", formal_training_started=False,
           publication_evidence=False)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        update(status="technical_failure", stage="failed", error=repr(error))
        raise
