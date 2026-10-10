"""Read-only audit of the queued experiment's real completed controls."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / '20260905'))
from remote_session import Session

SOURCE = r'''
import json, sys, time
from pathlib import Path
root = Path('/data/sunzixuan/decoupled_gate_20261008_v1')
sys.path.insert(0, str(root))
from lineage import SOURCE, read, sha, require, validate_config, validate_result, verify_files
new, old = read(root / 'deployment.json'), read(SOURCE / 'deployment.json')
require(new['carrier_sha256'] == old['carrier_sha256'], 'Carrier manifests differ')
for name, expected in new['carrier_sha256'].items():
    require(sha(root / 'carrier/src' / name) == expected, 'New carrier changed')
    require(sha(SOURCE / 'carrier/src' / name) == expected, 'Source carrier changed')
for name in ('train.py', 'prepare.py', 'evidence.py'):
    require(sha(root / name) == sha(SOURCE / name), 'Frozen module differs: ' + name)
require(sha(root / 'isolation.py') == sha(SOURCE / 'proposal.py'), 'Cache validation differs')
controls = {}
for dataset in ('baby', 'sports'):
    run = SOURCE / 'runs' / (dataset + '_baseline_s2026')
    result = read(run / 'result.json')
    validate_result(result, dataset)
    validate_config(read(run / 'config.json'), dataset)
    verify_files(read(run / 'inputs.json'))
    for name, key in (('best.pt','checkpoint_sha256'),('per_user_validation.npz','per_user_sha256')):
        require(sha(run / name) == result[key], 'Source output drift')
    prepared = read(SOURCE / 'cache' / dataset / 'prepare.json')
    for filename, expected in prepared['input_sha256'].items():
        require(sha(filename) == expected, 'Preparation drift')
    require(sha(prepared['cache']['path']) == prepared['cache']['sha256'], 'Cache drift')
    controls[dataset] = {'completed_baseline_verified': True, 'cache_verified': True,
                         'metrics': result['metrics']}
prior = read(SOURCE / 'pipeline_status.json')
progress = SOURCE / 'runs/sports_history_s2026/progress.jsonl'
last = None
if progress.exists():
    with progress.open('rb') as stream:
        stream.seek(max(0, progress.stat().st_size - 10000))
        lines = stream.read().decode().splitlines()
    if lines: last = json.loads(lines[-1])
print(json.dumps({'checked_unix': time.time(), 'controls': controls,
                  'prior_state': {'status':prior['status'],'stage':prior['stage']},
                  'prior_history_epoch': None if last is None else last['epoch'],
                  'current_state': read(root / 'pipeline_status.json')}, indent=2))
'''

session = Session('213')
try:
    code, output = session.run_python(SOURCE, '/data/sunzixuan/miniconda3/envs/MMRec/bin/python', 240)
    print(output)
    raise SystemExit(code)
finally:
    session.close()
