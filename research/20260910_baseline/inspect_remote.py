"""Read-only verification of the existing BRIDGE baseline artifacts."""
import hashlib
import json
import subprocess
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


root = Path('/data/sunzixuan/bridge_official')
record = {'checked_unix': time.time(), 'root': str(root), 'files': {}, 'checkpoints': {}}
for relative in ('src/models/bridge.py', 'src/models/dual_frequency_encoder.py',
                 'src/configs/model/BRIDGE.yaml', 'src/configs/overall.yaml',
                 'src/diagnose_ecli.py', 'src/utils/ecli.py'):
    path = root / relative
    record['files'][relative] = {'exists': path.is_file()}
    if path.is_file():
        record['files'][relative]['sha256'] = digest(path)
        if path.suffix == '.yaml':
            record['files'][relative]['config'] = path.read_text()
for dataset, name in (('baby', 'bridge_baby_seed2026_screen'),
                      ('sports', 'bridge_sports_s2026_confirm')):
    directory = root / 'src/saved_models/BRIDGE' / dataset / name
    record['checkpoints'][dataset] = {'files': [], 'inputs': {}}
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            if path.is_file():
                row = {'path': str(path), 'size': path.stat().st_size, 'sha256': digest(path)}
                if path.suffix in ('.json', '.yaml', '.yml') and path.stat().st_size < 40000:
                    row['metadata'] = path.read_text()
                record['checkpoints'][dataset]['files'].append(row)
    for name in (dataset + '.inter', 'image_feat.npy', 'text_feat.npy',
                 'mm_adj_10.pt', 'user_graph_dict.npy', 'user_emb.npy'):
        path = root / 'data' / dataset / name
        row = {'exists': path.is_file(), 'path': str(path)}
        if path.is_file():
            row.update(size=path.stat().st_size, sha256=digest(path))
        record['checkpoints'][dataset]['inputs'][name] = row
for name, command in (
    ('gpu', ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits']),
    ('git_commit', ['git', '-C', str(root), 'rev-parse', 'HEAD']),
    ('git_origin', ['git', '-C', str(root), 'remote', 'get-url', 'origin']),
    ('git_status', ['git', '-C', str(root), 'status', '--short'])):
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    record[name] = result.stdout[:6000]
record['memory'] = [line for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith(('MemTotal:', 'MemAvailable:'))]
print(json.dumps(record, indent=2))
