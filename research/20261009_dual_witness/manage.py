"""Deploy, verify, and inspect the isolated CPU diagnostic queue."""
import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path


LOCAL = Path(__file__).resolve().parent
REMOTE = '/data/sunzixuan/dual_witness_20261009_v1'
PYTHON = '/data/sunzixuan/miniconda3/envs/MMRec/bin/python'


def atomic_text(path, text):
    temporary = path.with_name(path.name + '.pending.' + str(os.getpid()))
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def deployment_payload():
    required = ('manage.py', 'diagnose.py', 'evidence.py', 'run_pipeline.py',
                'test_evidence.py', 'test_diagnose.py', 'PROTOCOL.md', 'program.md')
    names = set(required)
    names.update(path.name for path in LOCAL.glob('test_*.py'))
    if (LOCAL / 'README.md').is_file():
        names.add('README.md')
    payload = {name: base64.b64encode((LOCAL / name).read_bytes()).decode('ascii')
               for name in sorted(names)}
    git = ['git', '-c', 'safe.directory=' + LOCAL.as_posix()]
    commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], cwd=str(LOCAL), text=True).strip()
    dirty = subprocess.check_output(git + ['status', '--porcelain', '--untracked-files=all'],
                                    cwd=str(LOCAL), text=True).strip()
    if dirty:
        raise RuntimeError('Commit the isolated diagnostic before deploying')
    revision = {'commit': commit, 'local_directory': LOCAL.as_posix(), 'working_tree_clean': True}
    payload['source_revision.json'] = base64.b64encode(json.dumps(revision).encode()).decode('ascii')
    audit, _ = json.JSONDecoder().raw_decode(
        (LOCAL.parent / '20260910_baseline/remote_audit.txt').read_text().lstrip())
    payload['baseline_audit.json'] = base64.b64encode(json.dumps(audit).encode()).decode('ascii')
    return payload


def remote_source(action, payload):
    configuration = repr({'root': REMOTE, 'action': action, 'payload': payload, 'python': PYTHON})
    return 'configuration = ' + configuration + '\n' + r'''
import base64, hashlib, json, os, shutil, subprocess, time
from pathlib import Path

root = Path(configuration['root'])
action = configuration['action']
python = configuration['python']
environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
                   MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4',
                   PYTHONPATH=str(root), PYTHONUNBUFFERED='1')

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()

def atomic_text(path, value):
    temporary = path.with_name(path.name + '.pending.' + str(os.getpid()))
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + '\n')

def resources():
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        memory[key] = int(value.strip().split()[0]) * 1024
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    return {'available_ram_bytes': memory.get('MemAvailable', 0), 'available_cpu_count': cpus,
            'required_ram_bytes': 12 * 1024 ** 3, 'cpu_threads': 4, 'gpu_used': False,
            'cuda_visible_devices': environment['CUDA_VISIBLE_DEVICES']}

def require_resources():
    record = resources()
    if record['available_cpu_count'] is None or record['available_cpu_count'] < 4:
        raise RuntimeError('At least four available CPU cores are required')
    if record['available_ram_bytes'] < record['required_ram_bytes']:
        raise RuntimeError('Less than 12 GiB host RAM is available')
    return record

def check_bundle():
    deployment = json.loads((root / 'deployment.json').read_text())
    for name, expected in deployment['bundle_sha256'].items():
        if sha(root / name) != expected: raise RuntimeError('Deployment drift: ' + name)
    return deployment

def check_inputs():
    audit = json.loads((root / 'baseline_audit.json').read_text())
    source = Path('/data/sunzixuan/bridge_official')
    for relative, record in audit['files'].items():
        if sha(source / relative) != record['sha256']:
            raise RuntimeError('Baseline changed: ' + relative)
    for dataset, info in audit['checkpoints'].items():
        for record in info['inputs'].values():
            path = Path(record['path'])
            if path.is_file() != record['exists'] or (record['exists'] and sha(path) != record['sha256']):
                raise RuntimeError('Input changed: ' + str(path))

if action in ('deploy', 'verify', 'launch'):
    if (root / 'launch.json').exists() or (root / 'launch.lock').exists():
        raise RuntimeError('Launched bundle is immutable; inspect status')
    root.mkdir(exist_ok=True)
    hashes = {}
    for name, encoded in configuration['payload'].items():
        content = base64.b64decode(encoded)
        path = root / name
        if path.is_file() and path.read_bytes() != content:
            archive = root / 'deployment_history'
            archive.mkdir(exist_ok=True)
            prior = archive / (name + '.' + sha(path))
            if not prior.exists(): shutil.copy2(path, prior)
        atomic_text(path, content.decode('utf-8'))
        hashes[name] = sha(path)
    check_inputs()
    atomic_json(root / 'deployment.json', {'bundle_sha256': hashes, 'created_unix': time.time(),
                'root': str(root), 'cpu_threads': 4, 'gpu_used': False,
                'per_dataset_timeout_seconds': 2 * 3600, 'queue_timeout_seconds': 8 * 3600})
    print(json.dumps({'deployed': str(root), 'files': len(hashes)}, indent=2))

if action in ('test', 'verify', 'launch'):
    if (root / 'launch.json').exists() or (root / 'launch.lock').exists():
        raise RuntimeError('Launched bundle is immutable; inspect status')
    check_bundle()
    resource_record = require_resources()
    runner = """import json, sys, unittest
suite = unittest.defaultTestLoader.discover(sys.argv[1], pattern='test_*.py')
count = suite.countTestCases()
core = unittest.defaultTestLoader.loadTestsFromName('test_evidence').countTestCases()
if core < 13:
    raise RuntimeError('The original thirteen evidence tests must be present')
result = unittest.TextTestRunner(verbosity=2).run(suite)
print('DUAL_WITNESS_TEST_SUMMARY=' + json.dumps({'tests': count, 'evidence_tests': core,
    'successful': result.wasSuccessful(), 'failures': len(result.failures), 'errors': len(result.errors)}))
sys.exit(0 if result.wasSuccessful() else 1)
"""
    try:
        tested = subprocess.run([python, '-u', '-c', runner, str(root)], cwd=str(root),
            env=environment, capture_output=True, text=True, timeout=240)
        stdout, stderr, code = tested.stdout, tested.stderr, tested.returncode
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or b''
        stderr = error.stderr or b''
        stdout = stdout.decode('utf-8', errors='replace') if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode('utf-8', errors='replace') if isinstance(stderr, bytes) else stderr
        code = 124
    summary = None
    for line in stdout.splitlines():
        if line.startswith('DUAL_WITNESS_TEST_SUMMARY='):
            summary = json.loads(line.split('=', 1)[1])
    if code == 0 and (not summary or not summary.get('successful')): code = 1
    atomic_json(root / 'test_receipt.json', {'exit_code': code, 'checked_unix': time.time(),
                'deployment_sha256': sha(root / 'deployment.json'), 'test_summary': summary,
                'resources': resource_record, 'stdout': stdout, 'stderr': stderr})
    print(stdout)
    print(stderr)
    if code: raise RuntimeError('Diagnostic tests failed; do not launch')

if action in ('start', 'launch'):
    if (root / 'launch.json').exists() or (root / 'launch.lock').exists():
        raise RuntimeError('Already launched; inspect status and preserve the existing process')
    check_bundle()
    tested = json.loads((root / 'test_receipt.json').read_text())
    if tested['exit_code'] or tested['deployment_sha256'] != sha(root / 'deployment.json'):
        raise RuntimeError('Current bundle needs passing tests')
    resource_record = require_resources()
    check_inputs()
    with (root / 'launch.lock').open('x') as stream: stream.write(str(time.time()))
    command = [python, '-u', str(root / 'run_pipeline.py')]
    with (root / 'pipeline.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=str(root), stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True, env=environment)
    receipt = {'pid': process.pid, 'command': command, 'root': str(root),
               'launched_unix': time.time(), 'resources': resource_record,
               'deployment_sha256': sha(root / 'deployment.json')}
    atomic_json(root / 'launch.json', receipt)
    print(json.dumps(receipt, indent=2))

elif action == 'status':
    record = {'root': str(root), 'checked_unix': time.time(), 'artifacts': {},
              'diagnostics': {}, 'logs': {}, 'resources': resources()}
    for name in ('launch.json', 'deployment.json', 'pipeline_status.json',
                 'source_revision.json', 'test_receipt.json'):
        path = root / name
        if path.is_file(): record['artifacts'][name] = json.loads(path.read_text())
    for dataset in ('baby', 'sports'):
        output = root / 'diagnostics' / dataset
        row = {'output_exists': output.is_dir()}
        for name in ('diagnosis.json', 'failure.json', 'progress.json'):
            path = output / name
            if path.is_file(): row[name] = json.loads(path.read_text())
        record['diagnostics'][dataset] = row
    log_paths = [root / 'pipeline.log'] + sorted((root / 'logs').glob('*'))
    for path in log_paths:
        if not path.is_file(): continue
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 3000))
            record['logs'][str(path.relative_to(root))] = stream.read().decode('utf-8', errors='replace')
    receipt = record['artifacts'].get('launch.json', {})
    pid = receipt.get('pid')
    proc = Path('/proc/' + str(pid))
    alive = bool(pid and proc.is_dir())
    matches = False
    if alive:
        try: matches = str(root / 'run_pipeline.py') in (proc / 'cmdline').read_text().split('\x00')
        except FileNotFoundError: alive = False
    record['process_alive'] = alive and matches
    record['process_command_matches'] = matches
    print(json.dumps(record, indent=2, allow_nan=False))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('deploy', 'test', 'verify', 'start', 'launch', 'status'))
    parser.add_argument('--brief', action='store_true')
    args = parser.parse_args()
    payload = deployment_payload() if args.action in ('deploy', 'verify', 'launch') else {}
    sys.path.insert(0, str(LOCAL.parent / '20260905'))
    from remote_session import Session
    session = Session('213')
    try:
        code, output = session.run_python(remote_source(args.action, payload), PYTHON,
            360 if args.action in ('test', 'verify', 'launch') else 180)
        atomic_text(LOCAL / (args.action + '_receipt.txt'), output + '\n')
        if code or args.action != 'status':
            print(output)
            raise SystemExit(code)
        record, _ = json.JSONDecoder().raw_decode(output.lstrip())
        atomic_text(LOCAL / 'latest_status.json', json.dumps(record, indent=2) + '\n')
        for name in ('launch.json', 'deployment.json', 'pipeline_status.json',
                     'source_revision.json', 'test_receipt.json'):
            if name in record['artifacts']:
                atomic_text(LOCAL / name, json.dumps(record['artifacts'][name], indent=2) + '\n')
        diagnoses = {dataset: row['diagnosis.json'] for dataset, row in record['diagnostics'].items()
                     if 'diagnosis.json' in row}
        if diagnoses:
            atomic_text(LOCAL / 'diagnosis.json', json.dumps(diagnoses, indent=2) + '\n')
        if args.brief:
            state = record['artifacts'].get('pipeline_status.json', {})
            print(json.dumps({'root': record['root'], 'checked_unix': record['checked_unix'],
                'process_alive': record['process_alive'], 'status': state.get('status'),
                'stage': state.get('stage'), 'active_job': state.get('active_job'),
                'error': state.get('error'), 'decisions': {name: row.get('decision')
                                                       for name, row in diagnoses.items()},
                'resources': record['resources']}, indent=2))
        else:
            print(json.dumps(record, indent=2))
    finally:
        session.close()


if __name__ == '__main__':
    main()
