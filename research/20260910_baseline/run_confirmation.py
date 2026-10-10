"""Run two bounded CPU rescoring jobs, preserving logs and completion state."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent
record = {'started_unix': time.time(), 'status': 'running', 'jobs': []}
with (root / 'worker_start.json').open('x') as stream:
    json.dump(record, stream)
for dataset in ('baby', 'sports'):
    command = [sys.executable, '-u', str(root / 'confirm.py'), '--dataset', dataset,
               '--bridge-root', '/data/sunzixuan/bridge_official',
               '--output', str(root / (dataset + '_confirmation.json'))]
    started = time.time()
    with (root / (dataset + '.log')).open('xb') as log:
        try:
            result = subprocess.run(command, cwd=str(root), stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
                                             MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'), timeout=1800)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    record['jobs'].append({'dataset': dataset, 'exit_code': code, 'seconds': time.time() - started})
    if code:
        record['status'] = 'failed'
        break
else:
    record['status'] = 'completed'
record['finished_unix'] = time.time()
with (root / 'completion.json').open('x') as stream:
    json.dump(record, stream, indent=2)
print(json.dumps(record), flush=True)
raise SystemExit(0 if record['status'] == 'completed' else 1)
