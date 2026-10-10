"""Read bounded diagnostic progress, with no training or model changes."""
import sys
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL.parent / '20260905'))
from remote_session import Session

version = int(sys.argv[1]) if len(sys.argv) > 1 else 3
source = '''
import json, subprocess
from pathlib import Path
root = Path('/data/sunzixuan/decoupled_gate_20261008_diagnostic_v%d')
record = {'root': str(root)}
for name in ('diagnostic.json', 'fix_validation.json'):
    path = root / name
    if path.is_file():
        value = json.loads(path.read_text())
        record[name] = {key: value for key, value in value.items() if key not in ('steps', 'epochs')}
        record[name]['epochs_completed'] = len(value.get('epochs', []))
path = root / 'progress.jsonl'
if path.is_file():
    lines = path.read_text().splitlines()
    record['last_epoch'] = json.loads(lines[-1]) if lines else None
    record['epoch_count'] = len(lines)
path = root / 'diagnostic.log'
if path.is_file(): record['log_tail'] = path.read_text()[-1600:]
record['gpu'] = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20).stdout
print(json.dumps(record, indent=2))
''' % version
session = Session('213')
try:
    code, output = session.run_python(source, '/data/sunzixuan/miniconda3/envs/MMRec/bin/python', 90)
    print(output)
    raise SystemExit(code)
finally:
    session.close()
