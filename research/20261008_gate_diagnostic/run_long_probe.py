"""Deploy a new bounded probe without editing any previous experiment."""
import base64
import json
import sys
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL.parent / '20260905'))
from remote_session import Session

encoded = base64.b64encode((LOCAL / 'long_probe.py').read_bytes()).decode('ascii')
source = '''
import base64, json, os, subprocess
from pathlib import Path
root = Path('/data/sunzixuan/decoupled_gate_20261008_diagnostic_v3')
root.mkdir(exist_ok=False)
(root / 'long_probe.py').write_bytes(base64.b64decode(%r))
result = subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],
                        capture_output=True,text=True,timeout=20,check=True)
available = []
for line in result.stdout.splitlines():
    index, free = [int(field.strip()) for field in line.split(',')]
    if free >= 14336: available.append((free,index))
if not available: raise RuntimeError('No GPU with 14 GiB free for diagnostic')
gpu = str(max(available)[1])
with (root / 'diagnostic.log').open('xb') as log:
    completed = subprocess.run(['/data/sunzixuan/miniconda3/envs/MMRec/bin/python','-u',str(root / 'long_probe.py')],
        cwd=str(root),stdout=log,stderr=subprocess.STDOUT,timeout=1080,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
path = root / 'diagnostic.json'
if path.exists(): print(path.read_text())
else: print((root / 'diagnostic.log').read_text()[-16000:])
raise SystemExit(completed.returncode)
''' % encoded
session = Session('213')
try:
    code, output = session.run_python(source, '/data/sunzixuan/miniconda3/envs/MMRec/bin/python', 1140)
    (LOCAL / 'long_probe_receipt.txt').write_text(output, encoding='utf-8')
    if not code:
        record, _ = json.JSONDecoder().raw_decode(output.lstrip())
        (LOCAL / 'long_probe.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps({key: value for key, value in record.items() if key != 'epochs'}, indent=2))
    else:
        print(output)
    raise SystemExit(code)
finally:
    session.close()
