"""Read-only resource and input check for the proposed training study."""
import json
import subprocess
import time
from pathlib import Path

root = Path('/data/sunzixuan/bridge_official')
result = {'checked_unix': time.time(), 'root': str(root), 'inputs': {}}
for dataset in ('baby', 'sports'):
    result['inputs'][dataset] = {
        name: {'exists': (root / 'data' / dataset / name).is_file(),
               'resolved': str((root / 'data' / dataset / name).resolve())}
        for name in (dataset + '.inter', 'image_feat.npy', 'text_feat.npy', 'mm_adj_10.pt')}
for key, command in (
    ('gpu', ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits']),
    ('gpu_processes', ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory,gpu_uuid', '--format=csv,noheader,nounits'])):
    result[key] = subprocess.run(command, capture_output=True, text=True, timeout=20).stdout
result['memory'] = [line for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith(('MemTotal:', 'MemAvailable:'))]
result['disk'] = subprocess.run(['df', '-h', str(root)], capture_output=True, text=True, timeout=10).stdout
print(json.dumps(result, indent=2))
