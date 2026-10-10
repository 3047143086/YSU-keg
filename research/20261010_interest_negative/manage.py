"""Isolated MSCA deployment; only ``launch`` may start the new experiment.

All SSH output is collected through the existing credential-redacting Session.
Preparation, deployment and status use stdlib only and never initialize CUDA.
"""
import argparse
import base64
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
WORKSPACE = LOCAL.parents[1]
sys.path.insert(0, str(LOCAL.parent / '20260905'))
from remote_session import Session

REMOTE = '/data/sunzixuan/interest_negative_20261010_v1'
SOURCE = '/data/sunzixuan/MSCA-main'
PYTHON = '/data/sunzixuan/miniconda3/envs/MMRec/bin/python'
EXPECTED_SOURCE = {
    'models/msca.py': '066aafcf2e5373b20b2fdf9b3660302b8dfcaf580710d3ab864fac4f19e2360b',
    'common/trainer.py': '9520a7a712bb29ec0cb7c6319e362a0f3a89d33cd9ebfa7acd0afe9b48051e39',
    'configs/model/MSCA.yaml': 'a95df9ba3adb0f0657dc16e7c9231938c35d6c830b4e24da081040f97435f927',
}


def local_revision():
    git = ['git', '-c', 'safe.directory=' + LOCAL.as_posix()]
    commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], cwd=str(LOCAL), text=True).strip()
    dirty = subprocess.check_output(git + ['status', '--porcelain', '--untracked-files=all'],
                                    cwd=str(LOCAL), text=True).strip()
    if dirty:
        raise RuntimeError('Commit this isolated experiment before deploying or launching')
    return {'commit': commit, 'vcs': 'git', 'local_directory': LOCAL.as_posix(), 'working_tree_clean': True}


def bundle(revision):
    required = ('train.py', 'pipeline.py', 'PROTOCOL.md')
    for name in required:
        if not (LOCAL / name).is_file():
            raise RuntimeError('Missing experiment source: ' + name)
    excluded = {'launch.json', 'deployment.json', 'carrier_manifest.json',
                'source_protection.json', 'source_revision.json', 'test_receipt.json',
                'pipeline_status.json', 'research_report.json', 'latest_status.json'}
    files = {p.name: p.read_bytes() for p in sorted(LOCAL.iterdir()) if p.is_file()
             and (p.suffix in ('.py', '.md', '.yaml', '.yml', '.json', '.toml') or p.name == 'LICENSE')
             and p.name not in excluded and not p.name.endswith('_receipt.json')}
    files['source_revision.json'] = (json.dumps(revision, indent=2) + '\n').encode('utf-8')
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(content), 0o644, 0
            archive.addfile(info, io.BytesIO(content))
    return base64.b64encode(stream.getvalue()).decode('ascii'), {
        name: hashlib.sha256(content).hexdigest() for name, content in files.items()}


REMOTE_IMPLEMENTATION = r'''
import base64, hashlib, importlib.metadata, io, json, os, shutil, subprocess, tarfile, time
from pathlib import Path

root = Path(request['root'])
original = Path(request['source'])
action = request['action']
python = request['python']

def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def load(path):
    return json.loads(path.read_text())

def write_json(path, value):
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(str(temporary), str(path))

def owned_jobs():
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            args = (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if any(token in args for token in ('/run_pipeline.py', '/pipeline.py', '/guarded_train.py',
                   '/run_queue.py', '/train.py', '/run_pravi_experiment.py', '/diagnose_msca_gradients.py')):
                found.append({'pid': int(path.name), 'command': args.strip()})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return found

def gpu1():
    command = ['nvidia-smi', '-i', '1', '--query-gpu=index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu',
               '--format=csv,noheader,nounits']
    process = subprocess.run(command, capture_output=True, text=True, timeout=20)
    row = {'index': 1, 'exit_code': process.returncode, 'stdout': process.stdout.strip()}
    if process.returncode == 0:
        parts = [part.strip() for part in process.stdout.strip().split(',')]
        if len(parts) == 7 and parts[0] == '1':
            row.update(uuid=parts[1], name=parts[2], memory_total_mib=int(parts[3]),
                       memory_free_mib=int(parts[4]), memory_used_mib=int(parts[5]), utilization=int(parts[6]))
    if row.get('uuid'):
        query = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
        row['compute_processes'] = [line.strip() for line in query.stdout.splitlines()
                                    if line.startswith(row['uuid'] + ',')]
    return row

def inventory():
    packages = {}
    for name in ('torch', 'numpy', 'scipy', 'torch-geometric', 'pandas'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'checked_unix': time.time(), 'owned_research_processes': owned_jobs(),
            'gpu1': gpu1(), 'packages': packages, 'cuda_initialized': False}

def source_snapshot():
    src = original / 'src'
    files = {str(path.relative_to(src)): sha(path) for path in sorted(src.rglob('*')) if path.is_file()}
    for name, expected in request['expected_source'].items():
        if files.get(name) != expected:
            raise RuntimeError('Official MSCA source differs from inspected version: ' + name)
    inputs = {}
    for dataset in ('baby', 'sports'):
        data_root = original / 'data' / dataset
        if not data_root.is_dir():
            raise RuntimeError('Missing official MSCA dataset directory: ' + str(data_root))
        for path in sorted(data_root.iterdir()):
            if not path.is_file():
                continue
            name = path.name
            inputs[dataset + '/' + name] = {'path': str(path), 'exists': True,
                'sha256': sha(path), 'bytes': path.stat().st_size,
                'role': 'frozen_graph_cache' if name.endswith('.pt') else
                        ('frozen_split' if name.endswith('.inter') else 'frozen_feature')}
    license_path = original / 'LICENSE'
    license_record = {'path': str(license_path), 'exists': license_path.is_file(),
                      'sha256': sha(license_path) if license_path.is_file() else None,
                      'bytes': license_path.stat().st_size if license_path.is_file() else 0}
    return {'source_root': str(original), 'source_sha256': files, 'license': license_record, 'inputs': inputs}

def check_original(record):
    actual_names = {str(path.relative_to(original / 'src')) for path in (original / 'src').rglob('*')
                    if path.is_file()}
    if actual_names != set(record['source_sha256']):
        raise RuntimeError('Official source file inventory changed')
    for name, expected in record['source_sha256'].items():
        if sha(original / 'src' / name) != expected:
            raise RuntimeError('Original source changed: ' + name)
    license_record = record.get('license', {'exists': False})
    license_path = Path(license_record.get('path', str(original / 'LICENSE')))
    if license_path.is_file() != license_record.get('exists', False):
        raise RuntimeError('Official license changed')
    if license_record.get('exists') and sha(license_path) != license_record['sha256']:
        raise RuntimeError('Official license changed')
    for name, row in record['inputs'].items():
        path = Path(row['path'])
        if path.is_file() != row['exists'] or (row['exists'] and sha(path) != row['sha256']):
            raise RuntimeError('Original input changed: ' + name)

def check_carrier(manifest):
    for name, expected in manifest['carrier_sha256'].items():
        if sha(root / 'carrier/src' / name) != expected:
            raise RuntimeError('Carrier source changed: ' + name)
    for name, row in manifest['inputs'].items():
        path = root / 'carrier/data' / name
        if path.is_file() != row['exists'] or (row['exists'] and sha(path) != row['sha256']):
            raise RuntimeError('Carrier input/cache changed: ' + name)

def prepare():
    if (root / 'launch.json').exists():
        raise RuntimeError('Launched experiment is immutable; use status')
    preflight = inventory()
    if preflight['owned_research_processes']:
        raise RuntimeError('Other owned experiment is active: ' + json.dumps(preflight['owned_research_processes']))
    if not root.is_dir():
        root.mkdir()
    protected = root / 'source_protection.json'
    if protected.exists():
        record = load(protected)
        expected_inputs = {str(dataset) + '/' + path.name for dataset in ('baby', 'sports')
                           for path in sorted((original / 'data' / dataset).iterdir()) if path.is_file()}
        if set(record.get('inputs', {})) != expected_inputs or 'license' not in record:
            record = source_snapshot()
            record['created_unix'] = time.time()
            write_json(protected, record)
        else:
            check_original(record)
    else:
        record = source_snapshot()
        record['created_unix'] = time.time()
        write_json(protected, record)
    snapshot = root / 'carrier/src'
    if not snapshot.exists():
        shutil.copytree(str(original / 'src'), str(snapshot))
    for name, expected in record['source_sha256'].items():
        if sha(snapshot / name) != expected:
            raise RuntimeError('Existing carrier is not original MSCA: ' + name)
    license_record = record.get('license', {'exists': False})
    if license_record.get('exists'):
        destination = root / 'carrier/LICENSE'
        if not destination.exists():
            shutil.copy2(license_record['path'], str(destination))
        if sha(destination) != license_record['sha256']:
            raise RuntimeError('Existing carrier LICENSE differs from official source')
    # Existing graph caches are private copies; huge immutable features/splits are read-only uses of source links.
    for name, row in record['inputs'].items():
        if not row['exists']:
            continue
        destination = root / 'carrier/data' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            if name.endswith(('image_adj_10_True.pt', 'text_adj_10_True.pt')):
                shutil.copy2(row['path'], str(destination))
            else:
                destination.symlink_to(row['path'])
        if sha(destination) != row['sha256']:
            raise RuntimeError('Isolated input differs: ' + name)
    manifest = {'source_root': str(original), 'carrier_root': str(root / 'carrier'),
                'data_path': str(root / 'carrier/data') + '/', 'carrier_sha256': record['source_sha256'],
                'license': license_record,
                'inputs': record['inputs'], 'source_protection_sha256': sha(protected),
                'graph_cache_policy': 'copy existing graph caches; no original directory writes',
                'gpu_id': 1, 'cpu_threads': 4, 'created_unix': record['created_unix']}
    write_json(root / 'carrier_manifest.json', manifest)
    return {'prepared': str(root), 'source_files': len(manifest['carrier_sha256']),
            'carrier_manifest_sha256': sha(root / 'carrier_manifest.json'),
            'source_protection_sha256': sha(protected), 'data_path': manifest['data_path'],
            'preflight': preflight}

def deploy():
    if (root / 'launch.json').exists():
        raise RuntimeError('Launched bundle is immutable')
    if not (root / 'carrier_manifest.json').is_file():
        prepare()
    check_original(load(root / 'source_protection.json'))
    check_carrier(load(root / 'carrier_manifest.json'))
    if owned_jobs():
        raise RuntimeError('An owned experiment is active; do not deploy over it')
    if (root / 'deployment.json').is_file():
        obsolete = set(load(root / 'deployment.json')['bundle_sha256']) - set(request['bundle_sha256'])
        if obsolete:
            raise RuntimeError('Deployment would leave obsolete source files: ' + ', '.join(sorted(obsolete)))
    payload = base64.b64decode(request['payload'])
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        members = archive.getmembers()
        actual_names = {member.name for member in members}
        if actual_names != set(request['bundle_sha256']) or len(members) != len(actual_names):
            raise RuntimeError('Invalid source archive inventory')
        for member in members:
            if not member.isfile() or Path(member.name).name != member.name or member.name in ('.', '..'):
                raise RuntimeError('Unsafe archive member')
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != request['bundle_sha256'][member.name]:
                raise RuntimeError('Source archive hash mismatch: ' + member.name)
            destination = root / member.name
            if destination.exists() and destination.read_bytes() != data:
                history = root / 'deployment_history'
                history.mkdir(exist_ok=True)
                previous = history / (member.name + '.' + sha(destination))
                if not previous.exists():
                    shutil.copy2(str(destination), str(previous))
            destination.write_bytes(data)
    manifest = load(root / 'carrier_manifest.json')
    deployment = {'bundle_sha256': request['bundle_sha256'], 'carrier_sha256': manifest['carrier_sha256'],
                  'inputs': manifest['inputs'], 'source_root': str(original),
                  'carrier_manifest_sha256': sha(root / 'carrier_manifest.json'),
                  'source_protection_sha256': sha(root / 'source_protection.json'), 'created_unix': time.time()}
    write_json(root / 'deployment.json', deployment)
    return {'deployed': str(root), 'bundle_files': len(request['bundle_sha256']),
            'deployment_sha256': sha(root / 'deployment.json'), 'carrier_files': len(manifest['carrier_sha256'])}

def launch():
    if (root / 'launch.json').exists():
        raise RuntimeError('Already launched; inspect status instead of relaunching')
    if (root / 'pipeline.log').exists():
        raise RuntimeError('A previous launch log exists; inspect status before retrying')
    deployment = load(root / 'deployment.json')
    if load(root / 'source_revision.json')['commit'] != request['local_revision']['commit']:
        raise RuntimeError('Local committed revision differs from deployed revision; deploy first')
    for name, expected in deployment['bundle_sha256'].items():
        if sha(root / name) != expected:
            raise RuntimeError('Bundle drift: ' + name)
    for name in ('source_protection', 'carrier_manifest'):
        if sha(root / (name + '.json')) != deployment[name + '_sha256']:
            raise RuntimeError('Protected manifest drift: ' + name)
    check_original(load(root / 'source_protection.json'))
    check_carrier(load(root / 'carrier_manifest.json'))
    preflight = inventory()
    if preflight['owned_research_processes']:
        raise RuntimeError('An owned experiment is active: ' + json.dumps(preflight['owned_research_processes']))
    if preflight['gpu1'].get('memory_free_mib', 0) < 14 * 1024:
        raise RuntimeError('Physical GPU1 has less than 14GiB free; do not use GPU0')
    threads = {'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
               'NUMEXPR_NUM_THREADS': '4', 'NUMEXPR_MAX_THREADS': '4'}
    test_environment = dict(os.environ, CUDA_VISIBLE_DEVICES='',
                            PYTHONPATH=str(root / 'carrier/src') + os.pathsep + str(root), **threads)
    tested = subprocess.run([python, '-u', '-m', 'unittest', 'discover', '-s', str(root), '-p', 'test_*.py', '-v'],
                            cwd=str(root), env=test_environment, capture_output=True, text=True, timeout=240)
    receipt = {'exit_code': tested.returncode, 'checked_unix': time.time(),
               'deployment_sha256': sha(root / 'deployment.json'), 'stdout': tested.stdout, 'stderr': tested.stderr}
    write_json(root / 'test_receipt.json', receipt)
    if tested.returncode:
        raise RuntimeError('CPU tests failed; experiment not launched: ' + tested.stderr[-3000:])
    if 'Ran 0 tests' in tested.stderr:
        raise RuntimeError('No CPU tests were discovered; experiment not launched')
    # Recheck source/input hashes after tests and before starting any GPU process.
    check_original(load(root / 'source_protection.json'))
    check_carrier(load(root / 'carrier_manifest.json'))
    if owned_jobs():
        raise RuntimeError('An owned experiment appeared during tests; do not launch')
    final_gpu = gpu1()
    if final_gpu.get('memory_free_mib', 0) < 14 * 1024:
        raise RuntimeError('GPU1 no longer has 14GiB free')
    command = [python, '-u', str(root / 'pipeline.py')]
    with (root / 'pipeline.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=str(root), stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='1', PYTHONPATH=str(root / 'carrier/src') + os.pathsep + str(root), **threads))
    receipt = {'pid': process.pid, 'command': command, 'root': str(root), 'launched_unix': time.time(),
               'physical_gpu': 1, 'cpu_threads': 4, 'deployment_sha256': sha(root / 'deployment.json'),
               'preflight': preflight, 'final_gpu1': final_gpu}
    with (root / 'launch.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    return receipt

def tail(path, length):
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - length))
        return stream.read().decode('utf-8', errors='replace')

def status():
    record = {'root': str(root), 'checked_unix': time.time(), 'artifacts': {}, 'jobs': {}, 'logs': {},
              'artifact_sha256': {}, 'gpu1': gpu1(), 'owned_research_processes': owned_jobs()}
    if not root.is_dir():
        record['exists'] = False
        return record
    record['exists'] = True
    for path in sorted(root.glob('*.json')):
        if path.stat().st_size > 2 * 1024 * 1024:
            continue
        try:
            record['artifacts'][path.name] = load(path)
            record['artifact_sha256'][path.name] = sha(path)
        except json.JSONDecodeError:
            record.setdefault('partial_json', []).append(path.name)
    for name in ('results.tsv', 'report.md', 'REPORT.md', 'research_report.md'):
        path = root / name
        if path.is_file():
            record.setdefault('text_artifacts', {})[name] = tail(path, 64000)
            record['artifact_sha256'][name] = sha(path)
    for directory in ('smoke', 'runs'):
        parent = root / directory
        if not parent.is_dir():
            continue
        for path in sorted(parent.iterdir()):
            if not path.is_dir():
                continue
            row = {'artifact_sha256': {}}
            for name in ('result.json', 'failure.json', 'config.json', 'input_manifest.json',
                         'progress.json', 'environment.json'):
                source = path / name
                if source.is_file():
                    try:
                        row[name[:-5]] = load(source)
                        row['artifact_sha256'][name] = sha(source)
                    except json.JSONDecodeError:
                        row.setdefault('partial_json', []).append(name)
            source = path / 'protected_paths.json'
            if source.is_file():
                metadata = load(source)
                row['protected_paths'] = {'count': len(metadata), 'sha256': sha(source)}
                row['artifact_sha256']['protected_paths.json'] = sha(source)
            for name in ('epochs.jsonl', 'progress.jsonl', 'progress.ndjson'):
                progress = path / name
                if progress.is_file():
                    lines = tail(progress, 16000).splitlines()
                    for line in reversed(lines):
                        try:
                            row['latest_epoch'] = json.loads(line)
                            break
                        except json.JSONDecodeError:
                            continue
                    row['progress_bytes'] = progress.stat().st_size
            result = row.get('result', {})
            if result.get('checkpoint_sha256') and (path / 'best.pt').is_file():
                row['checkpoint_hash_matches'] = sha(path / 'best.pt') == result['checkpoint_sha256']
            if result.get('per_user_sha256') and (path / 'per_user_validation.npz').is_file():
                row['per_user_hash_matches'] = sha(path / 'per_user_validation.npz') == result['per_user_sha256']
            record['jobs'][str(path.relative_to(root))] = row
    logs = [root / 'pipeline.log']
    logs.extend(sorted((root / 'logs').glob('*.log')))
    logs.extend(sorted((root / 'logs').glob('*.stdout')))
    for path in logs:
        if path.is_file():
            record['logs'][str(path.relative_to(root))] = tail(path, 4000)
    pid = record['artifacts'].get('launch.json', {}).get('pid')
    proc = Path('/proc/' + str(pid)) if pid else None
    record['process_alive'] = False
    if proc and proc.is_dir():
        try:
            command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
            matches = str(root / 'pipeline.py') in command and proc.stat().st_uid == os.getuid()
            record.update(process_alive=matches, process_command_matches=matches, process_command=command)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return record

if action == 'prepare':
    record = prepare()
elif action == 'deploy':
    record = deploy()
elif action == 'launch':
    record = launch()
else:
    record = status()
print(json.dumps(record, indent=2, allow_nan=False))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'deploy', 'status', 'launch'))
    parser.add_argument('--brief', action='store_true')
    args = parser.parse_args()
    request = {'root': REMOTE, 'source': SOURCE, 'python': PYTHON, 'action': args.action,
               'expected_source': EXPECTED_SOURCE}
    if args.action == 'deploy':
        request['payload'], request['bundle_sha256'] = bundle(local_revision())
    elif args.action == 'launch':
        # Deployment hashes are checked remotely; this also forbids launching uncommitted local work.
        request['local_revision'] = local_revision()
    source = 'request = ' + repr(request) + '\n' + REMOTE_IMPLEMENTATION
    session = Session('213')
    try:
        code, output = session.run_python(source, PYTHON, 420 if args.action == 'launch' else 240)
    finally:
        session.close()
    destination = LOCAL / 'status'
    destination.mkdir(exist_ok=True)
    stamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    (destination / (stamp + '_' + args.action + '.txt')).write_text(output + '\n', encoding='utf-8')
    if code:
        print(output)
        raise SystemExit(code)
    record, _ = json.JSONDecoder().raw_decode(output.lstrip())
    (destination / (stamp + '_' + args.action + '.json')).write_text(
        json.dumps(record, indent=2) + '\n', encoding='utf-8')
    if args.action == 'status':
        (destination / 'latest_status.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        for name, value in record.get('artifacts', {}).items():
            (destination / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
        for name, value in record.get('text_artifacts', {}).items():
            (destination / name).write_text(value, encoding='utf-8')
    if args.brief and args.action == 'status':
        state = record.get('artifacts', {}).get('pipeline_status.json', {})
        print(json.dumps({'root': record['root'], 'checked_unix': record['checked_unix'],
            'process_alive': record.get('process_alive'), 'status': state.get('status'),
            'stage': state.get('stage'), 'error': state.get('error'), 'gpu1': record['gpu1'],
            'jobs': {name: {'status': row.get('result', {}).get('status'),
                           'progress': row.get('progress'), 'latest_epoch': row.get('latest_epoch'),
                           'metrics': row.get('result', {}).get('metrics')}
                     for name, row in record.get('jobs', {}).items()}}, indent=2))
    else:
        print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
