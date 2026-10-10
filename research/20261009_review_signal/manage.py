"""Freeze inputs, verify an isolated bundle, and manage its single experiment."""
import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path


LOCAL = Path(__file__).resolve().parent
REMOTE = '/data/sunzixuan/review_signal_20261009_v1'
PYTHON = '/data/sunzixuan/miniconda3/envs/MMRec/bin/python'
STAGE1 = '/data/sunzixuan/rave_20260906_stage1'
DUAL = '/data/sunzixuan/dual_witness_20261009_v1'
CARRIER = '/data/sunzixuan/candidate_gate_20261009_v1'


def atomic_text(path, value):
    temporary = path.with_name(path.name + '.pending.' + str(os.getpid()))
    temporary.write_text(value, encoding='utf-8')
    temporary.replace(path)


def audit_expectations():
    identity = json.loads((LOCAL / 'audit_receipt.json').read_text())
    if identity.get('status') != 'completed' or not identity['matching']['strict_identity_passed']:
        raise RuntimeError('A completed strict TRAIN identity audit is required')
    stage = json.loads((LOCAL.parent / '20260906_stage1/latest_status.json').read_text())
    features = stage['feature_manifest.json']
    encoder = stage['encoder_manifest.json']
    files = {name: dict(row) for name, row in identity['inputs'].items()}
    files['global_bank'] = {'path': STAGE1 + '/features/global_bank.npy',
                            'sha256': features['sha256']['global_bank.npy']}
    files['clip_weights'] = {'path': STAGE1 + '/weights/ViT-B-16.pt',
                             'sha256': encoder['weight_sha256']}
    for name, digest in encoder['source_hashes'].items():
        files['clip_source_' + name] = {'path': STAGE1 + '/vendor/clip/' + name, 'sha256': digest}
    dual = json.loads((LOCAL.parent / '20261009_dual_witness/deployment.json').read_text())
    dependencies = {
        'dual_witness_diagnose': {'path': DUAL + '/diagnose.py',
                                  'sha256': dual['bundle_sha256']['diagnose.py']},
        'dual_witness_evidence': {'path': DUAL + '/evidence.py',
                                  'sha256': dual['bundle_sha256']['evidence.py']},
    }
    revision = json.loads((LOCAL.parent / '20261009_candidate_gate/source_revision.json').read_text())
    return {'files': files, 'dependencies': dependencies, 'identity_matching': identity['matching'],
            'heldout_target_hash': identity['heldout_target_hash'],
            'carrier_root': CARRIER, 'carrier_source_revision': revision['commit']}


def deployment_payload():
    required = ('manage.py', 'experiment.py', 'mapping.py', 'screen_core.py',
                'PROTOCOL.md', 'program.md', 'input_audit.json')
    names = set(required)
    names.update(path.name for path in LOCAL.glob('*.py'))
    payload = {name: base64.b64encode((LOCAL / name).read_bytes()).decode('ascii')
               for name in sorted(names)}
    git = ['git', '-c', 'safe.directory=' + LOCAL.as_posix()]
    commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], cwd=str(LOCAL), text=True).strip()
    dirty = subprocess.check_output(git + ['status', '--porcelain', '--untracked-files=all'],
                                    cwd=str(LOCAL), text=True).strip()
    if dirty:
        raise RuntimeError('Commit the isolated hypothesis before deploying')
    revision = {'commit': commit, 'local_directory': LOCAL.as_posix(), 'working_tree_clean': True}
    payload['source_revision.json'] = base64.b64encode(json.dumps(revision).encode()).decode('ascii')
    baseline, _ = json.JSONDecoder().raw_decode(
        (LOCAL.parent / '20260910_baseline/remote_audit.txt').read_text().lstrip())
    payload['baseline_audit.json'] = base64.b64encode(json.dumps(baseline).encode()).decode('ascii')
    return payload


def remote_source(action, payload=None, expected=None):
    configuration = repr({'root': REMOTE, 'action': action, 'payload': payload or {},
                          'expected': expected, 'python': PYTHON, 'stage1': STAGE1, 'dual': DUAL})
    return 'configuration = ' + configuration + '\n' + r'''
import base64, hashlib, json, os, subprocess, time
from pathlib import Path

root = Path(configuration['root'])
action = configuration['action']
python = configuration['python']
environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4',
                   PYTHONPATH=os.pathsep.join([str(root), configuration['dual'],
                                               configuration['stage1'] + '/vendor']),
                   PYTHONUNBUFFERED='1')
environment.pop('CUDA_VISIBLE_DEVICES', None)
test_environment = dict(environment, CUDA_VISIBLE_DEVICES='')

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()

def atomic_bytes(path, content):
    temporary = path.with_name(path.name + '.pending.' + str(os.getpid()))
    with temporary.open('xb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8'))

def resources():
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        memory[key] = int(value.strip().split()[0]) * 1024
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    return {'available_ram_bytes': memory.get('MemAvailable', 0), 'available_cpu_count': cpus,
            'required_ram_bytes': 12 * 1024 ** 3, 'cpu_threads': 4,
            'gpu_selection_owner': 'experiment.py', 'tests_cuda_visible_devices': ''}

def require_resources():
    record = resources()
    if record['available_cpu_count'] is None or record['available_cpu_count'] < 4:
        raise RuntimeError('At least four available CPU cores are required')
    if record['available_ram_bytes'] < record['required_ram_bytes']:
        raise RuntimeError('Less than 12 GiB host RAM is available')
    return record

def verify_rows(rows):
    for name, record in rows.items():
        path = Path(record['path'])
        if not path.is_file() or sha(path) != record['sha256']:
            raise RuntimeError('Frozen input or dependency changed: ' + name + ' (' + str(path) + ')')
        if 'bytes' in record and path.stat().st_size != record['bytes']:
            raise RuntimeError('Frozen input size changed: ' + name)

def check_inputs():
    inputs = json.loads((root / 'input_audit.json').read_text())
    if inputs.get('status') != 'completed': raise RuntimeError('Completed input audit is required')
    verify_rows(inputs['files'])
    verify_rows(inputs['dependencies'])
    if len(inputs['baseline_carrier']) != 35:
        raise RuntimeError('Expected thirty-five frozen carrier source files')
    verify_rows(inputs['baseline_carrier'])
    baseline = json.loads((root / 'baseline_audit.json').read_text())
    original = Path('/data/sunzixuan/bridge_official')
    for relative, record in baseline['files'].items():
        if sha(original / relative) != record['sha256']:
            raise RuntimeError('Baseline changed: ' + relative)
    for dataset, info in baseline['checkpoints'].items():
        for record in info['inputs'].values():
            path = Path(record['path'])
            if path.is_file() != record['exists'] or (record['exists'] and sha(path) != record['sha256']):
                raise RuntimeError('Baseline input changed: ' + str(path))

def check_bundle():
    deployment = json.loads((root / 'deployment.json').read_text())
    for name, expected in deployment['bundle_sha256'].items():
        if sha(root / name) != expected: raise RuntimeError('Deployment drift: ' + name)
    return deployment

if action == 'audit':
    expected = configuration['expected']
    verify_rows(expected['files'])
    verify_rows(expected['dependencies'])
    clip_root = Path(configuration['stage1']) / 'vendor/clip'
    expected_code = {Path(row['path']).name for row in expected['files'].values()
                     if Path(row['path']).parent == clip_root and Path(row['path']).suffix == '.py'}
    if {path.name for path in clip_root.glob('*.py')} != expected_code:
        raise RuntimeError('CLIP source inventory differs from the recorded encoder')
    carrier_root = Path(expected['carrier_root'])
    carrier = json.loads((carrier_root / 'deployment.json').read_text())
    revision_path = carrier_root / 'source_revision.json'
    if sha(revision_path) != carrier['bundle_sha256']['source_revision.json']:
        raise RuntimeError('Carrier revision receipt changed')
    if json.loads(revision_path.read_text())['commit'] != expected['carrier_source_revision']:
        raise RuntimeError('Carrier commit differs from the completed experiment')
    baseline_carrier = {name: {'path': '/data/sunzixuan/bridge_official/src/' + name, 'sha256': digest}
                        for name, digest in carrier['carrier_sha256'].items()}
    if len(baseline_carrier) != 35: raise RuntimeError('Expected thirty-five recorded carrier files')
    verify_rows(baseline_carrier)
    files = {name: dict(row, bytes=Path(row['path']).stat().st_size)
             for name, row in expected['files'].items()}
    dependencies = {name: dict(row, bytes=Path(row['path']).stat().st_size)
                    for name, row in expected['dependencies'].items()}
    print(json.dumps({'status': 'completed', 'checked_unix': time.time(), 'files': files,
        'dependencies': dependencies, 'baseline_carrier': baseline_carrier,
        'identity_matching': expected['identity_matching'], 'heldout_target_hash': expected['heldout_target_hash'],
        'carrier_manifest': {'path': str(carrier_root / 'deployment.json'),
                             'sha256': sha(carrier_root / 'deployment.json')},
        'encoding_or_training_started': False, 'review_text_exported': False,
        'reviewer_identity_exported': False}, indent=2, allow_nan=False))

if action in ('deploy', 'verify', 'launch'):
    if (root / 'launch.json').exists() or (root / 'launch.lock').exists():
        raise RuntimeError('Launched bundle is immutable; inspect status')
    root.mkdir(exist_ok=True)
    hashes = {}
    for name, encoded in configuration['payload'].items():
        content = base64.b64decode(encoded)
        path = root / name
        hashes[name] = hashlib.sha256(content).hexdigest()
    existing_manifest = root / 'deployment.json'
    if existing_manifest.is_file():
        existing = json.loads(existing_manifest.read_text())
        if existing.get('bundle_sha256') != hashes:
            raise RuntimeError('Existing deployment bundle differs; preserve immutable bundle')
    for name, encoded in configuration['payload'].items():
        content = base64.b64decode(encoded)
        path = root / name
        if path.is_file() and sha(path) != hashes[name]:
            raise RuntimeError('Existing deployed file differs from immutable bundle: ' + name)
        if not path.is_file(): atomic_bytes(path, content)
    check_inputs()
    if not existing_manifest.is_file():
        atomic_json(root / 'deployment.json', {'bundle_sha256': hashes, 'created_unix': time.time(),
                    'root': str(root), 'cpu_threads': 4, 'gpu_selection_owner': 'experiment.py',
                    'experiment_budget_seconds': 6 * 3600, 'baseline_carrier_files_checked': 35})
    print(json.dumps({'deployed': str(root), 'files': len(hashes), 'baseline_carrier_files_checked': 35}, indent=2))

if action in ('test', 'verify', 'launch'):
    if (root / 'launch.json').exists() or (root / 'launch.lock').exists():
        raise RuntimeError('Launched bundle is immutable; inspect status')
    check_bundle()
    check_inputs()
    resource_record = require_resources()
    runner = """import json, sys, unittest
suite = unittest.defaultTestLoader.discover(sys.argv[1], pattern='test_*.py')
count = suite.countTestCases()
if count == 0: raise RuntimeError('No unit tests were discovered')
result = unittest.TextTestRunner(verbosity=2).run(suite)
print('REVIEW_SIGNAL_TEST_SUMMARY=' + json.dumps({'tests': count, 'successful': result.wasSuccessful(),
    'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped)}))
sys.exit(0 if result.wasSuccessful() else 1)
"""
    try:
        tested = subprocess.run([python, '-u', '-c', runner, str(root)], cwd=str(root),
            env=test_environment, capture_output=True, text=True, timeout=240)
        stdout, stderr, code = tested.stdout, tested.stderr, tested.returncode
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or b'', error.stderr or b''
        stdout = stdout.decode('utf-8', errors='replace') if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode('utf-8', errors='replace') if isinstance(stderr, bytes) else stderr
        code = 124
    summary = None
    for line in stdout.splitlines():
        if line.startswith('REVIEW_SIGNAL_TEST_SUMMARY='):
            summary = json.loads(line.split('=', 1)[1])
    if code == 0 and (not summary or not summary.get('successful')): code = 1
    atomic_json(root / 'test_receipt.json', {'exit_code': code, 'checked_unix': time.time(),
                'deployment_sha256': sha(root / 'deployment.json'), 'test_summary': summary,
                'resources': resource_record, 'cuda_visible_devices': '', 'stdout': stdout, 'stderr': stderr})
    print(stdout)
    print(stderr)
    if code: raise RuntimeError('Tests failed; do not launch')

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
    command = [python, '-u', str(root / 'experiment.py')]
    with (root / 'pipeline.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=str(root), stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True, env=environment)
    receipt = {'pid': process.pid, 'command': command, 'root': str(root),
               'launched_unix': time.time(), 'resources': resource_record,
               'deployment_sha256': sha(root / 'deployment.json')}
    atomic_json(root / 'launch.json', receipt)
    print(json.dumps(receipt, indent=2))

elif action == 'status':
    record = {'root': str(root), 'checked_unix': time.time(), 'artifacts': {}, 'logs': {}, 'resources': resources()}
    for name in ('launch.json', 'deployment.json', 'source_revision.json', 'test_receipt.json',
                 'input_audit.json', 'progress.json', 'diagnosis.json', 'failure.json'):
        path = root / name
        if path.is_file(): record['artifacts'][name] = json.loads(path.read_text())
    path = root / 'results.tsv'
    if path.is_file(): record['results_tsv'] = path.read_text()
    path = root / 'pipeline.log'
    if path.is_file():
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 4000))
            record['logs']['pipeline.log'] = stream.read().decode('utf-8', errors='replace')
    receipt = record['artifacts'].get('launch.json', {})
    pid = receipt.get('pid')
    proc = Path('/proc/' + str(pid))
    alive, matches = bool(pid and proc.is_dir()), False
    if alive:
        try: matches = str(root / 'experiment.py') in (proc / 'cmdline').read_text().split('\x00')
        except FileNotFoundError: alive = False
    record['process_alive'] = alive and matches
    record['process_command_matches'] = matches
    print(json.dumps(record, indent=2, allow_nan=False))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='verify',
                        choices=('audit', 'deploy', 'test', 'verify', 'start', 'launch', 'status'))
    parser.add_argument('--brief', action='store_true')
    args = parser.parse_args()
    payload = deployment_payload() if args.action in ('deploy', 'verify', 'launch') else {}
    expected = audit_expectations() if args.action == 'audit' else None
    sys.path.insert(0, str(LOCAL.parent / '20260905'))
    from remote_session import Session
    session = Session('213')
    try:
        code, output = session.run_python(remote_source(args.action, payload, expected), PYTHON,
            360 if args.action in ('audit', 'deploy', 'test', 'verify', 'launch') else 180)
        atomic_text(LOCAL / (args.action + '_receipt.txt'), output + '\n')
        if code:
            print(output)
            raise SystemExit(code)
        if args.action not in ('audit', 'status'):
            print(output)
            return
        record, _ = json.JSONDecoder().raw_decode(output.lstrip())
        if args.action == 'audit':
            path = LOCAL / 'input_audit.json'
            if path.exists():
                frozen = json.loads(path.read_text())
                for key in ('files', 'dependencies', 'baseline_carrier', 'heldout_target_hash'):
                    if record[key] != frozen[key]:
                        raise RuntimeError('Existing audit changed; preserve the frozen audit and investigate: ' + key)
                print(json.dumps({'audit_unchanged': True, 'checked_unix': record['checked_unix']}, indent=2))
            else:
                atomic_text(path, json.dumps(record, indent=2) + '\n')
                print(json.dumps(record, indent=2))
            return
        atomic_text(LOCAL / 'latest_status.json', json.dumps(record, indent=2) + '\n')
        for name in ('launch.json', 'deployment.json', 'source_revision.json', 'test_receipt.json',
                     'progress.json', 'diagnosis.json'):
            if name in record['artifacts']:
                atomic_text(LOCAL / name, json.dumps(record['artifacts'][name], indent=2) + '\n')
        if 'results_tsv' in record:
            atomic_text(LOCAL / 'results.tsv', record['results_tsv'])
        if args.brief:
            progress = record['artifacts'].get('progress.json', {})
            diagnosis = record['artifacts'].get('diagnosis.json', {})
            print(json.dumps({'root': record['root'], 'checked_unix': record['checked_unix'],
                'process_alive': record['process_alive'], 'progress': progress,
                'diagnosis_status': diagnosis.get('status'), 'decision': diagnosis.get('decision'),
                'resources': record['resources']}, indent=2))
        else:
            print(json.dumps(record, indent=2))
    finally:
        session.close()


if __name__ == '__main__':
    main()
