"""Read-only remote receipt for the failed original Electronics baseline."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20260905'))
from remote_session import Session

REMOTE_SOURCE = r'''
import hashlib, json, os, subprocess, time
from pathlib import Path
ROOT = Path('/data/sunzixuan/electronics_baseline_20261009_v1')
RUN = ROOT / 'runs/elec_baseline_s2026'

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def file_row(path, content=False, tail=None):
    path = Path(path)
    if not path.is_file():
        return {'path': str(path), 'exists': False}
    stat = path.stat()
    row = {'path': str(path), 'resolved_path': str(path.resolve()), 'exists': True,
           'bytes': stat.st_size, 'mtime_unix': stat.st_mtime, 'sha256': sha(path)}
    if content or tail is not None:
        text = path.read_text(errors='replace')
        if tail is not None:
            text = text[-tail:]
        try:
            row['content'] = json.loads(text)
        except Exception:
            row['content'] = text
    return row

def process(pid):
    path = Path('/proc', str(pid))
    if not path.is_dir():
        return {'pid': pid, 'exists': False}
    return {'pid': pid, 'exists': True,
            'command': (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'),
            'stat': (path / 'stat').read_text()}

receipt = {'checked_unix': time.time(), 'remote_root': str(ROOT),
           'read_only': True, 'processes': [process(pid) for pid in (2933472, 2936814)],
           'artifacts': {}, 'hash_verification': {}, 'files': []}
for name in ('deployment.json', 'source_revision.json', 'input_audit.json', 'launch.json',
             'pipeline_status.json', 'test_receipt.json', 'results.tsv', 'graph_manifest.json'):
    receipt['artifacts'][name] = file_row(ROOT / name, content=True)
for name in ('failure.json', 'numerics_receipt.json', 'config.json', 'inputs.json', 'result.json',
             'progress.jsonl', 'best.pt', 'best_rng.pt'):
    receipt['artifacts']['runs/elec_baseline_s2026/' + name] = file_row(
        RUN / name, content=name.endswith('.json'), tail=100000 if name.endswith('.jsonl') else None)
for name in ('queue.log', 'logs/elec_baseline_s2026.log', 'logs/prepare_original_mm_graph.log'):
    receipt['artifacts'][name] = file_row(ROOT / name, tail=120000)
for base in (RUN, ROOT / 'carrier/data/elec'):
    if base.is_dir():
        for path in sorted(base.rglob('*')):
            if path.is_file():
                receipt['files'].append({'path': str(path), 'bytes': path.stat().st_size,
                                         'is_symlink': path.is_symlink()})
deployment = json.loads((ROOT / 'deployment.json').read_text())
audit = json.loads((ROOT / 'input_audit.json').read_text())
groups = {
    'bundle': [(ROOT / name, digest) for name, digest in deployment['bundle_sha256'].items()],
    'carrier_deployed': [(ROOT / 'carrier/src' / name, digest) for name, digest in deployment['carrier_sha256'].items()],
    'carrier_source': [(Path(row['path']), row['sha256']) for row in audit['carrier_files'].values()],
    'original_inputs': [(Path(row['path']), row['sha256']) for row in audit['original_inputs'].values()],
}
run_inputs = RUN / 'inputs.json'
if run_inputs.is_file():
    recorded = json.loads(run_inputs.read_text())
    groups['run_inputs'] = [(Path(path), digest) for path, digest in recorded['sha256'].items()]
    receipt['absent_run_inputs'] = [{'path': path, 'still_absent': not Path(path).exists()}
                                   for path in recorded['absent']]
for label, entries in groups.items():
    rows = []
    for path, expected in entries:
        exists = path.is_file()
        actual = sha(path) if exists else None
        rows.append({'path': str(path), 'exists': exists, 'expected_sha256': expected,
                     'actual_sha256': actual, 'matches': actual == expected})
    receipt['hash_verification'][label] = {'count': len(rows), 'all_match': all(row['matches'] for row in rows),
                                          'rows': rows}
receipt['gpu_snapshot'] = {}
for label, command in (
    ('gpus', ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits']),
    ('compute_processes', ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits']),
):
    row = subprocess.run(command, capture_output=True, text=True, timeout=20)
    receipt['gpu_snapshot'][label] = {'exit_code': row.returncode, 'stdout': row.stdout, 'stderr': row.stderr}
receipt['checked_finished_unix'] = time.time()
print(json.dumps(receipt, indent=2, allow_nan=False))
'''

def main():
    session = Session('213')
    try:
        code, output = session.run_python(
            REMOTE_SOURCE, '/data/sunzixuan/miniconda3/envs/MMRec/bin/python', 240)
        if code:
            raise RuntimeError('Remote read-only collector failed: ' + output[-2000:])
        receipt, _ = json.JSONDecoder().raw_decode(output.lstrip())
        (HERE / 'remote_receipt.json').write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
        state = receipt['artifacts']['pipeline_status.json']['content']
        summary = {'status': state.get('status'), 'jobs': state.get('jobs'),
                   'error': state.get('error'), 'completed_seeds': state.get('completed_seeds'),
                   'processes': receipt['processes'], 'gpu_snapshot': receipt['gpu_snapshot'],
                   'hash_groups': {key: {'count': row['count'], 'all_match': row['all_match']}
                                   for key, row in receipt['hash_verification'].items()},
                   'run_files': [row for row in receipt['files'] if '/runs/' in row['path']]}
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        session.close()

if __name__ == '__main__':
    main()
