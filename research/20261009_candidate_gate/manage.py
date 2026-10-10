"""Deploy an isolated study, run checks, launch the bounded queue, and fetch status."""
import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL.parent / '20260905'))
from remote_session import Session

REMOTE = '/data/sunzixuan/candidate_gate_20261009_v1'
PYTHON = '/data/sunzixuan/miniconda3/envs/MMRec/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('deploy', 'test', 'verify', 'start', 'launch', 'status'))
    parser.add_argument('--brief', action='store_true')
    args = parser.parse_args()
    payload = {}
    if args.action in ('deploy', 'verify', 'launch'):
        required = ('train.py', 'proposal.py', 'run_pipeline.py',
                    'test_train_protocol.py', 'test_proposal.py',
                    'lineage.py', 'reporting.py', 'test_reporting.py', 'test_lineage.py', 'test_pipeline.py',
                    'PROTOCOL.md', 'README.md', 'program.md',
                    'guarded_train.py', 'numerics.py', 'test_numerics.py')
        payload = {name: base64.b64encode((LOCAL / name).read_bytes()).decode('ascii') for name in required}
        git = ['git', '-c', 'safe.directory=' + LOCAL.as_posix()]
        commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], cwd=str(LOCAL), text=True).strip()
        dirty = subprocess.check_output(git + ['status', '--porcelain', '--untracked-files=all'],
                                        cwd=str(LOCAL), text=True).strip()
        if dirty:
            raise RuntimeError('Commit this isolated hypothesis before deploying')
        metadata = {'commit': commit, 'local_directory': LOCAL.as_posix(), 'working_tree_clean': True}
        payload['source_revision.json'] = base64.b64encode(json.dumps(metadata).encode()).decode('ascii')
        audit, _ = json.JSONDecoder().raw_decode(
            (LOCAL.parent / '20260910_baseline/remote_audit.txt').read_text().lstrip())
        payload['baseline_audit.json'] = base64.b64encode(json.dumps(audit).encode()).decode('ascii')
    source = '''
import base64, hashlib, json, os, shutil, subprocess, time
from pathlib import Path
root = Path(%r)
action = %r
payload = %r
python = %r
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
if action in ('deploy', 'verify', 'launch'):
    if (root / 'launch.json').exists(): raise RuntimeError('Launched bundle is immutable')
    root.mkdir(exist_ok=True)
    hashes = {}
    for name, encoded in payload.items():
        content = base64.b64decode(encoded)
        path = root / name
        if path.exists() and path.read_bytes() != content:
            archive = root / 'deployment_history'
            archive.mkdir(exist_ok=True)
            prior = archive / (name + '.' + sha(path))
            if not prior.exists(): shutil.copy2(path, prior)
        path.write_bytes(content)
        hashes[name] = sha(path)
    source_root = Path('/data/sunzixuan/bridge_official')
    audit = json.loads((root / 'baseline_audit.json').read_text())
    for relative, row in audit['files'].items():
        if sha(source_root / relative) != row['sha256']:
            raise RuntimeError('Baseline changed: ' + relative)
    for dataset, data in audit['checkpoints'].items():
        for row in data['inputs'].values():
            path = Path(row['path'])
            if path.is_file() != row['exists'] or (row['exists'] and sha(path) != row['sha256']):
                raise RuntimeError('Input changed: ' + str(path))
    snapshot = root / 'carrier'
    snapshot.mkdir(exist_ok=True)
    carrier_hashes = {}
    for path in sorted((source_root / 'src').rglob('*')):
        relative = path.relative_to(source_root / 'src')
        if any(part in ('saved_models','saved','log','logs','__pycache__') for part in relative.parts): continue
        if not path.is_file() or path.suffix not in ('.py','.yaml','.yml'): continue
        destination = snapshot / 'src' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha(destination) != sha(path):
            raise RuntimeError('Existing carrier snapshot differs: ' + str(relative))
        if not destination.exists(): shutil.copy2(path, destination)
        carrier_hashes[str(relative)] = sha(destination)
    if not (snapshot / 'data').exists(): (snapshot / 'data').symlink_to(source_root / 'data', target_is_directory=True)
    manifest = {'bundle_sha256': hashes, 'carrier_sha256': carrier_hashes,
                'source_root': str(source_root), 'created_unix': time.time()}
    (root / 'deployment.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'deployed': str(root), 'files': len(hashes), 'carrier_files': len(carrier_hashes)}, indent=2))
if action in ('test', 'verify', 'launch'):
    if (root / 'launch.json').exists(): raise RuntimeError('Launched bundle is immutable')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', PYTHONPATH=str(root / 'carrier/src') + os.pathsep + str(root))
    result = subprocess.run([python, '-u', '-m', 'unittest', 'discover', '-s', str(root), '-p', 'test_*.py', '-v'],
        cwd=str(root), env=environment, capture_output=True, text=True, timeout=240)
    print(result.stdout); print(result.stderr)
    receipt = {'exit_code': result.returncode, 'checked_unix': time.time(),
               'deployment_sha256': sha(root / 'deployment.json'), 'stdout': result.stdout, 'stderr': result.stderr}
    (root / 'test_receipt.json').write_text(json.dumps(receipt, indent=2))
    if result.returncode: raise RuntimeError('Unit tests failed; do not launch')
if action in ('start', 'launch'):
    if (root / 'launch.json').exists(): raise RuntimeError('Already launched; inspect status')
    tested = json.loads((root / 'test_receipt.json').read_text())
    if tested['exit_code'] or tested['deployment_sha256'] != sha(root / 'deployment.json'):
        raise RuntimeError('Current bundle needs passing tests')
    deployment = json.loads((root / 'deployment.json').read_text())
    for name, value in deployment['bundle_sha256'].items():
        if sha(root / name) != value: raise RuntimeError('Deployment drift: ' + name)
    for name, value in deployment['carrier_sha256'].items():
        if sha(root / 'carrier/src' / name) != value: raise RuntimeError('Carrier drift: ' + name)
    command = [python, '-u', str(root / 'run_pipeline.py')]
    with (root / 'pipeline.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=str(root), stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
            env=dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
    receipt = {'pid': process.pid, 'command': command, 'root': str(root), 'launched_unix': time.time()}
    with (root / 'launch.json').open('x') as stream: json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))
elif action == 'status':
    record = {'root': str(root), 'checked_unix': time.time(), 'artifacts': {}, 'jobs': {}, 'logs': {}}
    for name in ('launch.json','pipeline_status.json','baby_screen.json','sports_screen.json','test_receipt.json',
                 'reuse_receipt.json','research_report.json','source_revision.json',
                 'baby_mechanism.json','sports_mechanism.json'):
        if (root / name).is_file(): record['artifacts'][name] = json.loads((root / name).read_text())
    for dataset, receipt in record['artifacts'].get('reuse_receipt.json', {}).get('datasets', {}).items():
        path = Path(receipt['source_run'])
        record['jobs']['runs/' + dataset + '_baseline_s2026'] = {
            'result': json.loads((path / 'result.json').read_text()), 'reused_from': str(path)}
    if (root / 'results.tsv').is_file(): record['results_tsv'] = (root / 'results.tsv').read_text()
    for path in sorted(root.glob('cache/*/prepare.json')):
        record['artifacts'][str(path.relative_to(root))] = json.loads(path.read_text())
    for directory in ('smoke','runs'):
        for path in sorted((root / directory).glob('*')):
            row = {}
            if (path / 'result.json').is_file(): row['result'] = json.loads((path / 'result.json').read_text())
            progress = path / 'progress.jsonl'
            if progress.is_file():
                with progress.open('rb') as stream:
                    stream.seek(max(0, progress.stat().st_size - 10000))
                    lines = stream.read().decode('utf-8', errors='replace').splitlines()
                if lines: row['latest_epoch'] = json.loads(lines[-1])
            record['jobs'][str(path.relative_to(root))] = row
    for path in [root / 'pipeline.log'] + sorted((root / 'logs').glob('*.log')):
        if path.is_file():
            with path.open('rb') as stream:
                stream.seek(max(0,path.stat().st_size-1800))
                record['logs'][str(path.relative_to(root))] = stream.read().decode('utf-8', errors='replace')
    pid = record['artifacts'].get('launch.json',{}).get('pid')
    record['process_alive'] = bool(pid and Path('/proc/' + str(pid)).is_dir())
    if record['process_alive']:
        record['process_command_matches'] = str(root / 'run_pipeline.py') in Path('/proc/' + str(pid) + '/cmdline').read_text()
        record['process_alive'] = record['process_command_matches']
    record['gpu'] = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20).stdout
    print(json.dumps(record, indent=2))
''' % (REMOTE, args.action, payload, PYTHON)
    session = Session('213')
    try:
        code, output = session.run_python(source, PYTHON, 360 if args.action in ('test', 'verify', 'launch') else 180)
        (LOCAL / (args.action + '_receipt.txt')).write_text(output + '\n', encoding='utf-8')
        if not code and args.action == 'status':
            record, _ = json.JSONDecoder().raw_decode(output.lstrip())
            (LOCAL / 'latest_status.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
            if 'results_tsv' in record:
                (LOCAL / 'results.tsv').write_text(record['results_tsv'], encoding='utf-8')
            for name, value in record['artifacts'].items():
                if '/' not in name:
                    (LOCAL / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
            state = record['artifacts'].get('pipeline_status.json', {})
            brief_state = {key: value for key, value in state.items() if key not in ('diagnostic', 'screens', 'traceback')}
            diagnostic = state.get('diagnostic', {})
            if args.brief:
                jobs = {name: {'status': row.get('result', {}).get('status', 'running_or_incomplete'),
                    'epoch': row.get('latest_epoch', {}).get('epoch'),
                    'best_score': row.get('latest_epoch', {}).get('best_score'),
                    'metrics': row.get('result', {}).get('metrics'),
                    'reused_from': row.get('reused_from')} for name, row in record['jobs'].items()}
                print(json.dumps({'root': record['root'], 'checked_unix': record['checked_unix'],
                    'process_alive': record['process_alive'], 'status': state.get('status'),
                    'stage': state.get('stage'), 'error': state.get('error'), 'jobs': jobs,
                    'screens': state.get('screens'), 'active_job': state.get('active_job'),
                    'predecessor': state.get('predecessor'), 'gpu': record['gpu']}, indent=2))
                raise SystemExit(code)
            print(json.dumps({'root': record['root'], 'process_alive': record['process_alive'],
                  'state': brief_state, 'mechanism_gate': diagnostic.get('mechanism_gate'),
                  'baseline_parity': diagnostic.get('baseline_parity'), 'jobs': record['jobs'], 'gpu': record['gpu'],
                  'current_log': {key: value for key, value in record['logs'].items()
                                  if key == 'pipeline.log' or state.get('stage', '') in key}}, indent=2))
        else:
            print(output)
        raise SystemExit(code)
    finally:
        session.close()


if __name__ == '__main__':
    main()
