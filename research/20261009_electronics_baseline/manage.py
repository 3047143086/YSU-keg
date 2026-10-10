"""Deploy, test, start, and inspect the isolated Electronics baseline queue."""
import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
REMOTE = "/data/sunzixuan/electronics_baseline_20261009_v1"
PYTHON = "/data/sunzixuan/miniconda3/envs/MMRec/bin/python"
PAYLOAD_NAMES = ("PROTOCOL.md", "input_audit.json", "manage.py", "graph_prepare.py", "run_queue.py",
                 "train.py", "guarded_train.py", "numerics.py", "test_graph_prepare.py",
                 "test_train_protocol.py", "test_source_lineage.py")


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def source_revision():
    hashes = {name: hashlib.sha256((LOCAL / name).read_bytes()).hexdigest() for name in PAYLOAD_NAMES}
    git = ['git', '-c', 'safe.directory=' + LOCAL.as_posix()]
    commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], cwd=str(LOCAL), text=True).strip()
    dirty = subprocess.check_output(git + ['status', '--porcelain', '--untracked-files=all'], cwd=str(LOCAL), text=True).strip()
    if dirty:
        raise RuntimeError('Commit the isolated baseline bundle before deployment')
    return {"commit": commit, "source_files_sha256": hashes,
            "source_directory": str(LOCAL), "working_tree_clean": True}


def payload():
    result = {name: base64.b64encode((LOCAL / name).read_bytes()).decode("ascii") for name in PAYLOAD_NAMES}
    result["source_revision.json"] = base64.b64encode(json.dumps(source_revision()).encode()).decode()
    return result


def remote_source(action, data):
    return r'''
import base64, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
configuration = %r
root = Path(configuration['root'])
action = configuration['action']
python = configuration['python']
payload = configuration['payload']
carrier_source = Path('/data/sunzixuan/candidate_gate_20261009_v1/carrier/src')
recovered = Path('/data/sunzixuan/bridge_extra_assets_20261009/elec')

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): digest.update(block)
    return digest.hexdigest()

def atomic_json(path, value):
    pending = path.with_name(path.name + '.pending.' + str(os.getpid()))
    with pending.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    pending.replace(path)

if action == 'deploy':
    if (root / 'launch.json').exists(): raise RuntimeError('Queue already launched; immutable')
    if root.exists() and any(root.iterdir()): raise RuntimeError('Nonempty target refuses overwrite')
    root.mkdir(parents=True, exist_ok=True)
    for name, encoded in payload.items():
        content = base64.b64decode(encoded)
        path = root / name
        path.write_bytes(content)
    carrier = root / 'carrier'
    (carrier / 'src').mkdir(parents=True, exist_ok=True)
    expected = json.loads((root / 'input_audit.json').read_text())
    if len(expected['carrier_files']) != 35: raise RuntimeError('Expected exactly thirty-five carrier files')
    for name, row in expected['carrier_files'].items():
        source = Path(row['path'])
        if sha(source) != row['sha256']: raise RuntimeError('Frozen source changed: ' + name)
        destination = carrier / 'src' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    data = carrier / 'data' / 'elec'
    data.mkdir(parents=True, exist_ok=True)
    for name in ('elec.inter', 'image_feat.npy', 'text_feat.npy'):
        source = recovered / name
        if not source.is_file(): raise FileNotFoundError(str(source))
        (data / name).symlink_to(source)
    carrier_hashes = {}
    for path in sorted((carrier / 'src').rglob('*')):
        if path.is_file(): carrier_hashes[str(path.relative_to(carrier / 'src'))] = sha(path)
    for name, row in expected['carrier_files'].items():
        if carrier_hashes.get(name) != row['sha256']: raise RuntimeError('Carrier hash mismatch: ' + name)
    bundle = {name: sha(root / name) for name in payload}
    atomic_json(root / 'deployment.json', {'bundle_sha256': bundle, 'carrier_sha256': carrier_hashes,
        'carrier_source': str(carrier_source), 'recovered_data': str(recovered),
        'data_links': {name: str(data / name) for name in ('elec.inter','image_feat.npy','text_feat.npy')},
        'created_unix': time.time(), 'test_evaluated': False})
    print(json.dumps({'status': 'deployed', 'root': str(root), 'carrier_files': len(carrier_hashes)}, indent=2))

if action in ('test', 'verify', 'start'):
    deployment = json.loads((root / 'deployment.json').read_text())
    for name, digest in deployment['bundle_sha256'].items():
        if sha(root / name) != digest: raise RuntimeError('Bundle drift: ' + name)
    for name, digest in deployment['carrier_sha256'].items():
        if sha(root / 'carrier' / 'src' / name) != digest: raise RuntimeError('Carrier drift: ' + name)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4',
                       PYTHONPATH=str(root / 'carrier' / 'src') + os.pathsep + str(root))
    if action in ('test', 'verify'):
        tested = subprocess.run([python, '-u', '-m', 'unittest', 'discover', '-s', str(root),
                                 '-p', 'test_*.py', '-v'], cwd=str(root), env=environment,
                                capture_output=True, text=True, timeout=300)
        print(tested.stdout); print(tested.stderr)
        atomic_json(root / 'test_receipt.json', {'exit_code': tested.returncode, 'checked_unix': time.time(),
            'deployment_sha256': sha(root / 'deployment.json'), 'stdout': tested.stdout,
            'stderr': tested.stderr, 'cuda_visible_devices': ''})
        if tested.returncode: raise RuntimeError('Actual MMRec tests failed')
    if action == 'start':
        tested = json.loads((root / 'test_receipt.json').read_text())
        if tested['exit_code'] or tested['deployment_sha256'] != sha(root / 'deployment.json'):
            raise RuntimeError('Current bundle lacks passing tests')
        if (root / 'launch.json').exists(): raise RuntimeError('Already launched')
        with (root / 'launch.lock').open('x') as lock: lock.write(str(time.time()))
        log = (root / 'queue.log').open('x')
        process = subprocess.Popen([python, '-u', str(root / 'run_queue.py')], cwd=str(root), env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
        atomic_json(root / 'launch.json', {'pid': process.pid, 'command': [python, '-u', str(root / 'run_queue.py')],
                     'root': str(root), 'launched_unix': time.time(), 'deployment_sha256': sha(root / 'deployment.json')})
        print(json.dumps({'status': 'started', 'pid': process.pid, 'root': str(root)}, indent=2))

if action == 'status':
    record = {'root': str(root), 'checked_unix': time.time(), 'artifacts': {}, 'logs': {}}
    for name in ('deployment.json','source_revision.json','input_audit.json','test_receipt.json',
                 'launch.json','pipeline_status.json','graph_manifest.json','results.tsv'):
        path = root / name
        if path.is_file():
            try: record['artifacts'][name] = json.loads(path.read_text())
            except Exception: record['artifacts'][name] = path.read_text()
    log = root / 'queue.log'
    if log.is_file():
        record['logs']['queue.log'] = log.read_text(errors='replace')[-12000:]
    launch = record['artifacts'].get('launch.json', {})
    pid = launch.get('pid')
    record['process_alive'] = bool(pid and Path('/proc/' + str(pid)).is_dir())
    if record['process_alive']:
        record['process_command_matches'] = str(root / 'run_queue.py') in Path('/proc/' + str(pid) + '/cmdline').read_text()
        record['process_alive'] = record['process_command_matches']
    print(json.dumps(record, indent=2, allow_nan=False))
''' % {'root': REMOTE, 'action': action, 'payload': data, 'python': PYTHON}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('deploy', 'test', 'verify', 'start', 'status'))
    args = parser.parse_args()
    sys.path.insert(0, str(LOCAL.parent / '20260905'))
    from remote_session import Session
    session = Session('213')
    try:
        code, output = session.run_python(remote_source(args.action, payload() if args.action == 'deploy' else {}), PYTHON, 600)
        print(output)
        if code: raise SystemExit(code)
    finally:
        session.close()


if __name__ == '__main__':
    main()
