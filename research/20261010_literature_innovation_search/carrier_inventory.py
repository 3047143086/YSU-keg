"""Bounded read-only inventory; never initializes CUDA or starts training."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research/20260905'))
from remote_session import Session


REMOTE_SOURCE = r'''
import hashlib, importlib.metadata, json, os, platform, subprocess, time
from pathlib import Path

def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()

record = {'checked_unix': time.time(), 'host': '213', 'read_only': True,
          'cuda_initialized': False, 'training_started': False,
          'python_version': platform.python_version(), 'packages': {},
          'owned_research_processes': [], 'roots': {}, 'assets': {}}
for package in ('torch', 'numpy', 'scipy', 'torch-geometric', 'transformers',
                'sentence-transformers', 'open-clip-torch', 'pandas', 'psutil'):
    try:
        record['packages'][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        record['packages'][package] = None
for command, key in ((['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu',
                      '--format=csv,noheader,nounits'], 'gpu'),
                     (['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                       '--format=csv,noheader,nounits'], 'gpu_compute_processes')):
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    record[key] = {'exit_code': result.returncode, 'stdout': result.stdout.strip()}
uid = os.getuid()
for process in Path('/proc').iterdir():
    if not process.name.isdigit() or int(process.name) == os.getpid():
        continue
    try:
        if process.stat().st_uid != uid:
            continue
        args = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', errors='replace')
        if any(token in args for token in ('run_pipeline.py', 'guarded_train.py', 'run_queue.py',
                                           '/data/sunzixuan/research_', '/data/sunzixuan/bridge_')):
            record['owned_research_processes'].append({'pid': int(process.name), 'command': args.strip()})
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
roots = [Path('/data/sunzixuan') / name for name in (
    'bridge_official', 'target_isolated_20261008_autoresearch_v1',
    'decoupled_gate_20261008_recovery_v1', 'candidate_gate_20261009_v1',
    'electronics_baseline_20261009_v1', 'electronics_recovery_20261010_v3',
    'msca_official', 'rave_20260906_stage1')]
core = ('models/bridge.py', 'models/dual_frequency_encoder.py', 'common/trainer.py',
        'utils/dataloader.py', 'utils/topk_evaluator.py', 'configs/model/BRIDGE.yaml')
for root in roots:
    row = {'exists': root.is_dir(), 'core_sha256': {}, 'receipts': {}, 'baselines': {}}
    if root.is_dir():
        src = root / 'src' if (root / 'src').is_dir() else root / 'carrier/src'
        for name in core:
            path = src / name
            if path.is_file():
                row['core_sha256'][name] = sha(path)
        for name in ('launch.json', 'pipeline_status.json', 'source_revision.json'):
            path = root / name
            if path.is_file():
                data = json.loads(path.read_text())
                row['receipts'][name] = {key: data.get(key) for key in (
                    'status', 'stage', 'pid', 'commit', 'error', 'finished_unix') if key in data}
        for dataset in ('baby', 'sports'):
            output = root / 'runs' / (dataset + '_baseline_s2026')
            result = output / 'result.json'
            if result.is_file():
                data = json.loads(result.read_text())
                row['baselines'][dataset] = {'output': str(output), 'result_sha256': sha(result),
                    'result': data, 'checkpoint_hash_matches': (
                        sha(output / 'best.pt') == data.get('checkpoint_sha256')),
                    'per_user_hash_matches': (sha(output / 'per_user_validation.npz') == data.get('per_user_sha256'))}
    record['roots'][str(root)] = row
asset_roots = [Path('/data/sunzixuan/bridge_official/data') / name for name in ('baby', 'sports')]
asset_roots += [Path('/data/sunzixuan/bridge_extra_assets_20261009/elec'),
                Path('/data/sunzixuan/rave_20260906_stage1/weights')]
for root in asset_roots:
    row = {'exists': root.is_dir(), 'files': []}
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if path.is_file():
                row['files'].append({'name': path.name, 'bytes': path.stat().st_size})
    record['assets'][str(root)] = row
for name in ('MSCA', 'BM3', 'FREEDOM', 'LATTICE', 'LightGCN', 'MGCN', 'SGL', 'UltraGCN',
             'SG-URInit', 'MRLib'):
    hits = []
    for base in (Path('/data/sunzixuan'), Path('/data/sunzixuan/bridge_official')):
        if base.is_dir():
            for path in base.iterdir():
                if name.lower() in path.name.lower():
                    hits.append({'path': str(path), 'is_dir': path.is_dir()})
    record.setdefault('baseline_name_candidates', {})[name] = hits
print(json.dumps(record, indent=2, allow_nan=False))
'''

MSCA_SOURCE = r'''
import hashlib, json, os, time
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

base = Path('/data/sunzixuan')
record = {'checked_unix': time.time(), 'read_only': True, 'cuda_initialized': False,
          'candidates': {}, 'experiments213_paths': [], 'max_source_text_bytes': 80000}
roots = [base / name for name in ('MSCA-main', 'PRAVI', 'pravi', 'MMRec', 'MRLib',
                                  'MRLib_pravi', 'Multimodal-Recommendation-Library-main')]
for path in base.iterdir():
    if path.is_dir() and any(token in path.name.lower() for token in ('pravi', 'msca', 'mmrec', 'mrlib')):
        if path not in roots:
            roots.append(path)
for root in roots:
    if not root.is_dir():
        continue
    src_candidates = [root / 'src']
    src_candidates.extend(child / 'src' for child in root.iterdir() if child.is_dir())
    for src in src_candidates:
        if not src.is_dir():
            continue
        row = {'root': str(root), 'src': str(src), 'files': {}, 'datasets': {}, 'experiments': []}
        for name in ('models/msca.py', 'configs/model/MSCA.yaml', 'configs/dataset/baby.yaml',
                     'configs/dataset/sports.yaml', 'configs/overall.yaml', 'utils/quick_start.py',
                     'common/trainer.py', 'utils/dataloader.py', 'utils/configurator.py',
                     'utils/dataset.py', 'utils/utils.py', 'main.py'):
            path = src / name
            if path.is_file():
                data = path.read_bytes()
                row['files'][name] = {'bytes': len(data), 'sha256': sha(path),
                    'text': data.decode('utf-8-sig', errors='replace') if len(data) <= 80000 else None}
        for dataset in ('baby', 'sports', 'elec'):
            data_root = root / 'data' / dataset
            if data_root.is_dir():
                row['datasets'][dataset] = [{'name': p.name, 'bytes': p.stat().st_size}
                    for p in sorted(data_root.iterdir()) if p.is_file()]
        for directory in (root / 'experiments213', src / 'experiments213'):
            if directory.is_dir():
                record['experiments213_paths'].append(str(directory))
                row['experiments'] += [{'name': p.name, 'bytes': p.stat().st_size} for p in sorted(directory.iterdir())
                    if p.is_file() and any(token in p.name.lower() for token in ('msca', 'baseline', 'control'))][:80]
        record['candidates'][str(src)] = row
print(json.dumps(record, indent=2, allow_nan=False))
'''


def main():
    selected = '--msca' in sys.argv
    session = Session('213')
    try:
        code, output = session.run_python(
            MSCA_SOURCE if selected else REMOTE_SOURCE,
            '/data/sunzixuan/miniconda3/envs/MMRec/bin/python', 180)
    finally:
        session.close()
    if code:
        raise RuntimeError('Read-only inventory failed: ' + output[-2000:])
    record, _ = json.JSONDecoder().raw_decode(output.lstrip())
    destination = Path(__file__).with_name('msca_inventory.json' if selected else 'carrier_inventory.json')
    destination.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    if selected:
        print(json.dumps({'receipt': str(destination), 'experiments213_paths': record['experiments213_paths'],
            'candidates': {key: {'files': {name: {'bytes': value['bytes'], 'sha256': value['sha256']}
                                        for name, value in row['files'].items()},
                                'datasets': row['datasets'], 'experiments': row['experiments']}
                          for key, row in record['candidates'].items()}}, indent=2))
        return
    print(json.dumps({'receipt': str(destination), 'gpu': record['gpu'],
                      'owned_research_processes': record['owned_research_processes'],
                      'packages': record['packages'],
                      'roots': {key: {'exists': value['exists'], 'receipts': value['receipts'],
                                      'baselines': list(value['baselines'])}
                                for key, value in record['roots'].items()}}, indent=2))


if __name__ == '__main__':
    main()
