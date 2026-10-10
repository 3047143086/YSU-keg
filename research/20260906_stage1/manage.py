"""Deploy only this experiment's scripts and manage its isolated remote jobs."""

import argparse
import base64
import json
import sys
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL.parent / '20260905'))
from remote_session import Session

REMOTE = '/data/sunzixuan/rave_20260906_stage1'
PYTHON = '/data/sunzixuan/miniconda3/envs/MMRec/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'deploy', 'start', 'status', 'test'])
    parser.add_argument('--brief', action='store_true')
    args = parser.parse_args()
    payload = {path.name: base64.b64encode(path.read_bytes()).decode('ascii')
               for path in LOCAL.iterdir() if (path.suffix == '.py' and path.name != 'manage.py')
               or path.name in ('experiment.json', 'PROTOCOL.md')}
    source = '''
import base64, hashlib, json, os, subprocess
from pathlib import Path
root = Path(%r)
action = %r
payload = %r
if action == 'prepare':
    root.mkdir(exist_ok=False)
elif not root.exists():
    raise RuntimeError('Experiment root does not exist')
if action in ('prepare', 'deploy'):
    for name, encoded in payload.items():
        destination = root / name
        if destination.exists() and destination.read_bytes() != base64.b64decode(encoded):
            if (root / 'pipeline_launch.json').exists():
                raise RuntimeError('Refusing to change scripts after pipeline launch: ' + name)
            archive = root / 'deployment_history'
            archive.mkdir(exist_ok=True)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            prior = archive / (name + '.' + digest)
            if not prior.exists():
                prior.write_bytes(destination.read_bytes())
            destination.write_bytes(base64.b64decode(encoded))
        if not destination.exists():
            destination.write_bytes(base64.b64decode(encoded))
    print('DEPLOYED', sorted(payload))
if action in ('prepare', 'start'):
    script = 'prepare_assets.py' if action == 'prepare' else 'run_pipeline.py'
    job_name = 'preparation' if action == 'prepare' else 'pipeline'
    receipt_path = root / (job_name + '_launch.json')
    if receipt_path.exists():
        raise RuntimeError('Job was already launched; inspect status before any retry')
    environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    command = [%r, '-u', str(root / script)]
    with (root / (job_name + '.log')).open('xb') as log:
        process = subprocess.Popen(command, cwd=str(root), env=environment, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'pid': process.pid, 'command': command, 'root': str(root)}
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print('LAUNCHED', json.dumps(receipt))
if action == 'test':
    result = subprocess.run([%r, '-m', 'unittest', '-v', 'test_stage1.py'], cwd=str(root),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    print(result.stdout)
    if result.returncode:
        raise RuntimeError('Remote tests failed')
if action == 'status':
    record = {}
    for path in root.glob('*launch.json'):
        receipt = json.loads(path.read_text())
        process = Path('/proc') / str(receipt['pid'])
        receipt['process_exists'] = process.exists()
        if process.exists():
            receipt['process_state'] = (process / 'stat').read_text().split(') ', 1)[1].split()[0]
        record[path.name] = receipt
    for name in ('download_progress.json', 'assets_ready.json', 'assets_failed.json', 'encoder_manifest.json',
                 'feature_manifest.json', 'extraction_progress.json', 'pipeline_status.json', 'summary.json'):
        path = root / name
        if path.exists():
            record[name] = json.loads(path.read_text())
    for name in ('preparation.log', 'pipeline.log', 'extract_features.log',
                 'smoke_global.log', 'smoke_local.log', 'smoke_shuffled.log'):
        path = root / name
        if path.exists():
            record[name] = path.read_text(errors='replace')[-5000:]
    jobs = root / 'runs'
    if jobs.exists():
        record['runs'] = {}
        for directory in sorted(jobs.iterdir()):
            progress = directory / 'progress.json'
            record['runs'][directory.name] = json.loads(progress.read_text()) if progress.exists() else {}
            if (directory / 'stdout.log').exists():
                record['runs'][directory.name]['stdout_tail'] = (directory / 'stdout.log').read_text(errors='replace')[-1800:]
    print(json.dumps(record, indent=2))
''' % (REMOTE, args.action, payload if args.action in ('prepare', 'deploy') else {}, PYTHON, PYTHON)
    session = Session('213')
    try:
        code, output = session.run_python(source, PYTHON, 240 if args.action == 'test' else 90)
        if not args.brief or args.action != 'status' or code:
            print(output)
        (LOCAL / (args.action + '_receipt.txt')).write_text(output + '\n', encoding='utf-8')
        if args.action == 'status' and not code:
            record, _ = json.JSONDecoder().raw_decode(output.lstrip())
            (LOCAL / 'latest_status.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
            if args.brief:
                brief = {key: value for key, value in record.items()
                         if not key.endswith('.log') and key not in
                         ('encoder_manifest.json', 'feature_manifest.json', 'download_progress.json')}
                if 'feature_manifest.json' in record:
                    features = record['feature_manifest.json']
                    brief['features'] = {key: features[key] for key in
                        ('items', 'missing_image_items', 'missing_text_items', 'elapsed_seconds')}
                if 'runs' in brief:
                    brief['runs'] = {name: {key: value for key, value in run.items() if key != 'stdout_tail'}
                                     for name, run in brief['runs'].items()}
                if 'summary.json' in brief:
                    full = brief['summary.json']
                    brief['summary.json'] = {key: full[key] for key in
                        ('all_complete', 'aggregate', 'decision')}
                print(json.dumps(brief, indent=2))
        raise SystemExit(code)
    finally:
        session.close()


if __name__ == '__main__':
    main()
