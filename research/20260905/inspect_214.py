"""Inspect the second documented experiment location for missing raw inputs."""

import os
import socket
import subprocess
from pathlib import Path

print('HOST', socket.gethostname())
print(subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv'], text=True))
root = Path('/data/gongshanghui/sun_exp')
if not root.is_dir():
    raise RuntimeError('Documented experiment directory is absent')
print('ROOT_DIRECTORIES', sorted(p.name for p in root.iterdir() if not p.name.startswith('.')))
count = 0
for current, directories, files in os.walk(root):
    depth = len(Path(current).relative_to(root).parts)
    directories[:] = [d for d in directories if not d.startswith('.') and d not in
                      {'saved_models', 'saved', 'log', 'logs', '__pycache__', 'cache', 'node_modules', 'site-packages'}]
    if depth >= 5:
        directories[:] = []
    for filename in files:
        if filename.endswith(('.item', '.json.gz', '.b')) or any(
            term in filename.lower() for term in ('asin', 'mapping', 'meta_', 'clip', 'title', 'elec.inter')
        ):
            path = Path(current) / filename
            print('POTENTIAL_INPUT', str(path), path.stat().st_size)
            count += 1
print('MATCH_COUNT', count)
for path in sorted((root / 'LOBSTER/data').glob('*')):
    print('DATA_ENTRY', path.name, 'dir' if path.is_dir() else path.stat().st_size)
