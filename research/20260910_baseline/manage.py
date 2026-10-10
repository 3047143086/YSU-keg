"""Deploy an immutable validation confirmation bundle or read its status."""
import argparse
import base64
import json
import sys
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL.parent / '20260905'))
from remote_session import Session

REMOTE = '/data/sunzixuan/bridge_baseline_confirm_20260910_v2'
PYTHON = '/data/sunzixuan/miniconda3/envs/MMRec/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('start', 'status'))
    args = parser.parse_args()
    payload = {}
    if args.action == 'start':
        for name in ('confirm.py', 'run_confirmation.py'):
            payload[name] = base64.b64encode((LOCAL / name).read_bytes()).decode('ascii')
        audit, _ = json.JSONDecoder().raw_decode((LOCAL / 'remote_audit.txt').read_text().lstrip())
        payload['input_manifest.json'] = base64.b64encode(json.dumps(audit).encode()).decode('ascii')
        for dataset in ('baby', 'sports'):
            payload[dataset + '_historical.json'] = base64.b64encode(
                (LOCAL.parent / '20260905' / (dataset + '_baseline.json')).read_bytes()).decode('ascii')
    source = '''
import base64, hashlib, json, os, subprocess, time
from pathlib import Path
root = Path(%r)
action = %r
payload = %r
python = %r
if action == 'start':
    root.mkdir(exist_ok=False)
    hashes = {}
    for name, content in payload.items():
        raw = base64.b64decode(content)
        with (root / name).open('xb') as stream:
            stream.write(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    with (root / 'deployment.json').open('x') as stream:
        json.dump(hashes, stream, indent=2)
    command = [python, '-u', str(root / 'run_confirmation.py')]
    with (root / 'worker.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=str(root), stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'))
    receipt = {'pid': process.pid, 'command': command, 'root': str(root), 'launched_unix': time.time()}
    with (root / 'launch.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))
else:
    record = {'root': str(root), 'checked_unix': time.time(), 'artifacts': {}, 'logs': {}}
    for name in ('launch.json', 'worker_start.json', 'completion.json', 'deployment.json',
                 'baby_confirmation.json', 'sports_confirmation.json'):
        path = root / name
        if path.is_file():
            record['artifacts'][name] = json.loads(path.read_text())
    for name in ('worker.log', 'baby.log', 'sports.log'):
        path = root / name
        if path.is_file():
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 2500))
                record['logs'][name] = stream.read().decode('utf-8', errors='replace')
    launch = record['artifacts'].get('launch.json', {})
    pid = launch.get('pid')
    record['process_alive'] = bool(pid and Path('/proc/' + str(pid)).is_dir())
    print(json.dumps(record, indent=2))
''' % (REMOTE, args.action, payload, PYTHON)
    session = Session('213')
    try:
        code, output = session.run_python(source, PYTHON, 120)
        (LOCAL / (args.action + '_receipt.txt')).write_text(output + '\n', encoding='utf-8')
        if code == 0 and args.action == 'status':
            record, _ = json.JSONDecoder().raw_decode(output.lstrip())
            for name, value in record['artifacts'].items():
                (LOCAL / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
            print(json.dumps({
                'root': record['root'], 'process_alive': record['process_alive'],
                'completion': record['artifacts'].get('completion.json'),
                'metrics': {name: row['metrics'] for name, row in record['artifacts'].items()
                            if name.endswith('_confirmation.json')}, 'logs': record['logs']
            }, indent=2))
        else:
            print(output)
        raise SystemExit(code)
    finally:
        session.close()


if __name__ == '__main__':
    main()
