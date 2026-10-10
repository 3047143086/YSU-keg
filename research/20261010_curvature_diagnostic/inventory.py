"""Bounded read-only MSCA/GPU1/checkpoint inventory; no torch import."""
import base64
import json
import sys
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL.parent / '20260905'))
from remote_session import Session

REMOTE = r'''
import base64, hashlib, json, os, subprocess, time
from pathlib import Path
root = Path('/data/sunzixuan/interest_negative_20261010_v1')
def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()
files = {}
for name in ('carrier/src/models/msca.py', 'carrier/src/configs/model/MSCA.yaml',
             'carrier/src/configs/dataset/baby.yaml', 'carrier/src/configs/overall.yaml',
             'carrier/src/common/abstract_recommender.py', 'carrier_manifest.json',
             'source_protection.json', 'deployment.json',
             'runs/baby_uniform_s2026_formal/config.json',
             'runs/baby_uniform_s2026_formal/environment.json',
             'runs/baby_uniform_s2026_formal/result.json',
             'runs/baby_uniform_s2026_formal/best.pt'):
    path = root / name
    row = {'path': str(path), 'exists': path.is_file()}
    if path.is_file():
        row.update(sha256=sha(path), bytes=path.stat().st_size)
        if path.suffix != '.pt':
            row['text'] = path.read_text()
        if name.endswith('models/msca.py'):
            row['base64_exact_bytes'] = base64.b64encode(path.read_bytes()).decode('ascii')
    files[name] = row
processes = []
for path in Path('/proc').iterdir():
    if not path.name.isdigit() or int(path.name) == os.getpid():
        continue
    try:
        if path.stat().st_uid != os.getuid():
            continue
        cmd = (path/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        if any(x in cmd for x in ('/pipeline.py','/train.py','/audit.py','/run_queue.py')):
            processes.append({'pid':int(path.name),'command':cmd.strip()})
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
gpu = subprocess.run(['nvidia-smi','-i','1','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu',
                      '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
app = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                      '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
uuid = gpu.stdout.strip().split(',')[1].strip() if gpu.returncode == 0 else ''
print(json.dumps({'checked_unix':time.time(),'cuda_initialized':False,'files':files,
                  'owned_processes':processes,'gpu1':gpu.stdout.strip(),
                  'gpu1_compute_processes':[l for l in app.stdout.splitlines() if l.startswith(uuid+',')]}))
'''

def main():
    compile(REMOTE, '<inventory-remote>', 'exec')
    session = Session('213')
    try:
        code, output = session.run_python(REMOTE, 'python', 120)
    finally:
        session.close()
    if code:
        raise RuntimeError(output)
    value, _ = json.JSONDecoder().raw_decode(output[output.find('{'):])
    (LOCAL / 'carrier_inventory.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    for name, row in value['files'].items():
        if name.endswith('models/msca.py'):
            (LOCAL / 'inspection_msca.py').write_bytes(base64.b64decode(row['base64_exact_bytes']))
    print(json.dumps({'gpu1':value['gpu1'], 'gpu1_compute_processes':value['gpu1_compute_processes'],
                      'owned_processes':value['owned_processes'], 'checkpoint':value['files']['runs/baby_uniform_s2026_formal/best.pt']},indent=2))

if __name__ == '__main__':
    main()
