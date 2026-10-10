"""Deploy and inspect the isolated pairing screen using the existing SSH helper."""

import argparse
import base64
import json
import sys
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL.parent / '20260905'))
from remote_session import Session

REMOTE = '/data/sunzixuan/hpair_20260908_screen_v1'
PYTHON = '/data/sunzixuan/miniconda3/envs/MMRec/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('inspect', 'deploy', 'test', 'start', 'status'))
    parser.add_argument('--brief', action='store_true')
    args = parser.parse_args()
    selected = [p for p in LOCAL.iterdir() if
                (p.suffix == '.py' and p.name != 'manage.py')
                or p.name in ('experiment.json', 'PROTOCOL.md')]
    payload = {p.name: base64.b64encode(p.read_bytes()).decode('ascii') for p in selected}
    source = '''
import base64, hashlib, json, os, shutil, subprocess, time
from pathlib import Path
root = Path(%r)
action = %r
payload = %r
python = %r
if action == 'inspect':
    old = Path('/data/sunzixuan/rcmqr_20260908_screen_v2')
    record = {'checked_unix': time.time(), 'root_exists': root.exists(),
              'disk_free_gb': shutil.disk_usage('/data/sunzixuan').free / 1e9,
              'prepared': {}, 'python_exists': Path(python).is_file()}
    for seed in (2026, 2027, 2028):
        directory = old / 'prepared' / str(seed)
        record['prepared'][str(seed)] = {name: (directory / name).is_file() for name in
            ('base_scores.npy', 'support_mask.npy', 'targets.npy', 'roles.npy', 'history.npy')}
        ready = old / 'shared' / str(seed) / 'ready.json'
        record['prepared'][str(seed)]['shared_ready'] = json.loads(ready.read_text()) if ready.exists() else None
    record['gpu'] = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15).stdout
    print(json.dumps(record, indent=2))
elif action == 'deploy':
    root.mkdir(exist_ok=True)
    if (root / 'pipeline_launch.json').exists():
        raise RuntimeError('Refusing to mutate a launched experiment')
    hashes = {}
    for name, encoded in payload.items():
        if Path(name).name != name:
            raise RuntimeError('Invalid payload path')
        destination = root / name
        content = base64.b64decode(encoded)
        if destination.exists() and destination.read_bytes() != content:
            archive = root / 'deployment_history'
            archive.mkdir(exist_ok=True)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            prior = archive / (name + '.' + digest)
            if not prior.exists():
                prior.write_bytes(destination.read_bytes())
        destination.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    (root / 'deployment_manifest.json').write_text(json.dumps(hashes, indent=2))
    print(json.dumps({'deployed': str(root), 'hashes': hashes}, indent=2))
elif action == 'test':
    command = [python, '-u', '-m', 'unittest', 'discover', '-s', str(root), '-p', 'test_*.py', '-v']
    result = subprocess.run(command, cwd=str(root), capture_output=True, text=True, timeout=180)
    print(result.stdout)
    print(result.stderr)
    if result.returncode:
        raise RuntimeError('Remote unit tests failed')
    print('REMOTE_TESTS_PASSED')
elif action == 'start':
    receipt_path = root / 'pipeline_launch.json'
    if receipt_path.exists():
        raise RuntimeError('Launch receipt already exists; inspect status, do not double launch')
    required = ['train_pairing.py', 'test_pairing.py', 'run_pipeline.py', 'experiment.json']
    if any(not (root / name).is_file() for name in required):
        raise RuntimeError('Deployment incomplete')
    environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    command = [python, '-u', str(root / 'run_pipeline.py')]
    with (root / 'pipeline.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=str(root), env=environment, stdout=log,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    receipt = {'pid': process.pid, 'command': command, 'root': str(root), 'launched_unix': time.time()}
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
elif action == 'status':
    record = {'root': str(root), 'checked_unix': time.time()}
    for name in ('pipeline_launch.json', 'pipeline_status.json', 'summary.json'):
        path = root / name
        if path.exists():
            record[name] = json.loads(path.read_text())
    receipt = record.get('pipeline_launch.json')
    if receipt:
        proc = Path('/proc') / str(receipt['pid'])
        receipt['process_exists'] = proc.exists()
        if proc.exists():
            receipt['process_state'] = (proc / 'stat').read_text().split(') ', 1)[1].split()[0]
            receipt['command_matches'] = str(root / 'run_pipeline.py') in (proc / 'cmdline').read_text()
    record['gpu'] = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15).stdout
    record['logs'] = {}
    for path in [root / 'pipeline.log'] + sorted(root.glob('**/stdout.log')):
        if path.is_file():
            with path.open('rb') as handle:
                handle.seek(max(0, path.stat().st_size - 3000))
                record['logs'][str(path.relative_to(root))] = handle.read().decode('utf-8', errors='replace')
    record['progress'] = {str(p.relative_to(root)): json.loads(p.read_text()) for p in root.glob('**/progress.json')}
    if (root / 'RESULTS.md').exists():
        record['results_markdown'] = (root / 'RESULTS.md').read_text()
    print(json.dumps(record, indent=2))
''' % (REMOTE, args.action, payload if args.action == 'deploy' else {}, PYTHON)
    session = Session('213')
    try:
        code, output = session.run_python(source, PYTHON, 240 if args.action == 'test' else 120)
        (LOCAL / (args.action + '_receipt.txt')).write_text(output + '\n', encoding='utf-8')
        if args.action in ('inspect', 'status') and not code:
            record, _ = json.JSONDecoder().raw_decode(output.lstrip())
            (LOCAL / ('latest_' + args.action + '.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')
            if 'results_markdown' in record:
                (LOCAL / 'REMOTE_RESULTS.md').write_text(record['results_markdown'], encoding='utf-8')
            if args.brief:
                record.pop('logs', None)
                record.pop('results_markdown', None)
                summary = record.get('summary.json', {})
                summary.pop('seeds', None)
            print(json.dumps(record, indent=2))
        else:
            print(output)
        raise SystemExit(code)
    finally:
        session.close()


if __name__ == '__main__':
    main()
