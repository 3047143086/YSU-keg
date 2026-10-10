"""Read-only readiness check; never downloads data or touches running jobs."""

import json
import subprocess
from pathlib import Path

import numpy as np
import torch

root = Path('/data/sunzixuan/rave_20260906_stage1')
checks = {}
for name in ('features/global_bank.npy', 'features/image_feat.npy',
             'features/text_feat.npy', 'split/baby.inter', 'split_manifest.json',
             'bridge_src/models/bridge.py', 'bridge_src/diagnose_ecli.py'):
    path = root / name
    checks[name] = {'exists': path.exists(), 'bytes': path.stat().st_size if path.exists() else None}
for name in ('global_bank', 'image_feat', 'text_feat'):
    path = root / 'features' / (name + '.npy')
    if path.exists():
        checks[name] = {'shape': list(np.load(path, mmap_mode='r').shape)}
checks['gpu'] = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
    '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=20).stdout
checks['processes'] = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20).stdout
checks['disk'] = subprocess.run(['df', '-h', str(root)], capture_output=True, text=True, check=True).stdout
checks['torch'] = torch.__version__
checks['cuda_available'] = torch.cuda.is_available()
checks['new_root_exists'] = Path('/data/sunzixuan/rcmqr_20260908_screen_v1').exists()
checks['split_manifest'] = json.loads((root / 'split_manifest.json').read_text())
print(json.dumps(checks, indent=2))
