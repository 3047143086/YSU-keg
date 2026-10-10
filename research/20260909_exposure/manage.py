"""Deploy, test, launch, and inspect the remote temporal screen."""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL.parent / "20260905"))
from remote_session import Session

REMOTE = "/data/sunzixuan/temporal_20260909_screen_v2"
PYTHON = "/data/sunzixuan/miniconda3/envs/MMRec/bin/python"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("deploy", "test", "start", "status", "inspect"))
    args = parser.parse_args()
    files = [p for p in LOCAL.iterdir() if p.suffix == ".py" or p.name in ("experiment.json", "PROTOCOL.md")]
    payload = {p.name: base64.b64encode(p.read_bytes()).decode("ascii") for p in files}
    remote_source = r'''
import base64, hashlib, json, os, shutil, subprocess, time
from pathlib import Path
root = Path(%r)
action = %r
payload = %r
python = %r
if action == 'inspect':
    record = {'root': str(root), 'root_exists': root.exists(),
              'python_exists': Path(python).is_file(),
              'inputs': {name: Path(path).is_file() for name, path in {
                  'interactions': '/data/sunzixuan/expirement/LOBSTER/data/baby/baby.inter',
                  'feature_bank': '/data/sunzixuan/rcmqr_20260908_screen_v2/features/global_bank.npy'}.items()}}
    record['gpu'] = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15).stdout
    print(json.dumps(record, indent=2))
elif action == 'deploy':
    root.mkdir(exist_ok=True)
    if (root / 'pipeline_launch.json').exists():
        raise RuntimeError('refusing to mutate a launched experiment')
    hashes = {}
    for name, encoded in payload.items():
        if Path(name).name != name:
            raise RuntimeError('invalid payload path')
        content = base64.b64decode(encoded)
        destination = root / name
        if destination.exists() and destination.read_bytes() != content:
            raise RuntimeError('remote file differs; refuse overwrite after deployment')
        destination.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    (root / 'deployment_manifest.json').write_text(json.dumps(hashes, indent=2))
    print(json.dumps({'deployed': str(root), 'hashes': hashes}, indent=2))
elif action == 'test':
    result = subprocess.run([python, '-u', '-m', 'unittest', 'discover', '-s', str(root), '-p', 'test_*.py', '-v'],
                            cwd=str(root), capture_output=True, text=True, timeout=180)
    print(result.stdout); print(result.stderr)
    if result.returncode: raise RuntimeError('remote tests failed')
    print('REMOTE_TESTS_PASSED')
elif action == 'start':
    receipt = root / 'pipeline_launch.json'
    if receipt.exists(): raise RuntimeError('launch receipt already exists')
    required = ('temporal_train.py', 'test_temporal.py', 'run_pipeline.py', 'experiment.json')
    if any(not (root / name).is_file() for name in required): raise RuntimeError('deployment incomplete')
    environment = dict(os.environ, PYTHONUNBUFFERED='1')
    command = [python, '-u', str(root / 'run_pipeline.py')]
    with (root / 'pipeline.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=str(root), env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    value = {'pid': process.pid, 'command': command, 'root': str(root), 'launched_unix': time.time()}
    receipt.write_text(json.dumps(value, indent=2)); print(json.dumps(value, indent=2))
elif action == 'status':
    record = {'root': str(root), 'checked_unix': time.time()}
    for name in ('pipeline_launch.json', 'pipeline_status.json', 'summary.json'):
        path = root / name
        if path.exists(): record[name] = json.loads(path.read_text())
    record['logs'] = {}
    for path in [root / 'pipeline.log'] + sorted(root.glob('**/stdout.log')):
        if path.is_file():
            with path.open('rb') as handle:
                handle.seek(max(0, path.stat().st_size - 3000))
                record['logs'][str(path.relative_to(root))] = handle.read().decode('utf-8', errors='replace')
    if (root / 'RESULTS.md').exists(): record['results_markdown'] = (root / 'RESULTS.md').read_text()
    print(json.dumps(record, indent=2))
''' % (REMOTE, args.action, payload if args.action == "deploy" else {}, PYTHON)
    session = Session("213")
    try:
        code, output = session.run_python(remote_source, PYTHON, 240 if args.action == "test" else 120)
        (LOCAL / (args.action + "_receipt.txt")).write_text(output + "\n", encoding="utf-8")
        print(output)
        raise SystemExit(code)
    finally:
        session.close()


if __name__ == "__main__":
    main()
