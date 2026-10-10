"""Launch one protected Baby baseline on host 213, independent of this PC."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE = "/data/sunzixuan/baseline_confirmation_20261010_v1"
PYTHON = "/data/sunzixuan/miniconda3/envs/MMRec/bin/python"
CARRIER = "/data/sunzixuan/interest_negative_20261010_v1"

SUPERVISOR = r'''
import json, os, subprocess, time, traceback
from pathlib import Path
root = Path("/data/sunzixuan/baseline_confirmation_20261010_v1")
python = "/data/sunzixuan/miniconda3/envs/MMRec/bin/python"
carrier = Path("/data/sunzixuan/interest_negative_20261010_v1")
started = time.monotonic()
def write(name, value):
    path = root / name
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(str(pending), str(path))
cmd = [python, "-u", str(carrier / "train.py"), "--dataset", "baby",
       "--mode", "uniform", "--assets", str(carrier / "groups/baby"),
       "--output", str(root / "run")]
try:
    free = int(subprocess.check_output(["nvidia-smi", "-i", "1",
        "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
    if free < 14336:
        raise RuntimeError("GPU1 has less than 14 GiB free")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "4"
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 4:
        raise RuntimeError("less than four available CPU cores")
    os.sched_setaffinity(0, set(allowed[:4]))
    with (root / "train.log").open("wb") as log:
        child = subprocess.Popen(cmd, cwd=str(carrier), env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
        write("launch.json", {"supervisor_pid": os.getpid(), "child_pid": child.pid,
              "command": cmd, "started_unix": time.time(), "status": "training",
              "physical_gpu": 1, "timeout_seconds": 28800,
              "source_commit": "e1775dad7b57f2b1ffae08609d720e8122eccfa9"})
        try:
            code = child.wait(timeout=28800)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            code = 124
    write("queue_result.json", {"status": "completed" if code == 0 else "technical_failure",
          "exit_code": code, "elapsed_seconds": time.monotonic() - started,
          "innovation_gain_claimed": False})
except Exception as error:
    write("queue_result.json", {"status": "technical_failure", "error": str(error),
          "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started,
          "innovation_gain_claimed": False})
'''


def main():
    repo = HERE.parents[1] / "YSU-keg"
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    local_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source = """import json, subprocess\nfrom pathlib import Path\n"""
    source += f"root=Path({REMOTE!r})\n"
    source += "if root.exists(): raise RuntimeError('refusing duplicate launch')\nroot.mkdir()\n"
    source += f"(root/'supervisor.py').write_text({SUPERVISOR!r})\n"
    manifest = {'public_protocol_commit': commit, 'launcher_sha256': local_sha,
                'budget_seconds': 28800, 'kind': 'baseline_confirmation'}
    source += f"(root/'protocol_manifest.json').write_text(json.dumps({manifest!r},indent=2))\n"
    source += f"log=(root/'supervisor.log').open('wb')\np=subprocess.Popen([{PYTHON!r},'-u',str(root/'supervisor.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\n"
    source += "print('LAUNCH_RESULT='+json.dumps({'supervisor_pid':p.pid,'root':str(root),'detached':True}))\n"
    sys.path.insert(0, str(HERE.parent / "20260905"))
    from remote_session import Session
    session = Session("213")
    try:
        code, output = session.run_python(source, PYTHON, 60)
        if code:
            raise RuntimeError(output[-3000:])
        marker = "LAUNCH_RESULT="
        value, _ = json.JSONDecoder().raw_decode(output[output.index(marker) + len(marker):])
        (HERE / "launch_receipt.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(value))
    finally:
        session.close()


if __name__ == "__main__":
    main()
