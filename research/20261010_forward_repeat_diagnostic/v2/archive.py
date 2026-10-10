"""Read-only download of exact small receipt bytes after the diagnostic exits."""
import base64
import hashlib
import json
import sys
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
REMOTE = '/data/sunzixuan/forward_repeat_diagnostic_20261010_v2'
PYTHON = '/data/sunzixuan/miniconda3/envs/MMRec/bin/python'
NAMES = ('diagnostic_config.json', 'tests.json', 'launch.json', 'repeat_progress.json', 'repeat_receipt.json')


def main():
    sys.path.insert(0, str(LOCAL.parents[1]/'20260905'))
    from remote_session import Session
    session = Session('213')
    try:
        source = r'''
import base64,hashlib,json,subprocess
from pathlib import Path
root=Path(%r)
receipt=json.loads((root/'repeat_receipt.json').read_text())
if receipt['status'] not in ('completed_exact','completed_different','technical_failure'):raise RuntimeError('Final diagnostic receipt required')
launch=json.loads((root/'launch.json').read_text());pid=launch['pid'];process=Path('/proc/'+str(pid))
try:alive=process.exists() and str(root/'repeat_diagnostic.py') in (process/'cmdline').read_text()
except FileNotFoundError:alive=False
if alive:raise RuntimeError('Diagnostic process must exit before archive')
rows={}
for name in %r:
 p=root/name
 if p.exists():
  raw=p.read_bytes()
  if len(raw)>2*1024*1024:raise RuntimeError('Small receipts only')
  rows[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'base64':base64.b64encode(raw).decode()}
gpu=subprocess.run(['nvidia-smi','-i','1','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
print(json.dumps({'root':str(root),'pid':pid,'process_alive':alive,'status':receipt['status'],'files':rows,
                  'gpu_query_exit_code':gpu.returncode,'gpu1':gpu.stdout,'compute_apps':apps.stdout}))
''' % (REMOTE, NAMES)
        code, output = session.run_python(source, PYTHON, 180)
        if code:
            raise RuntimeError('Read-only receipt archive unavailable; diagnostic may still be running')
        result = json.JSONDecoder().raw_decode(output.lstrip())[0]
        for name, row in result['files'].items():
            raw = base64.b64decode(row.pop('base64'))
            if hashlib.sha256(raw).hexdigest() != row['sha256'] or len(raw) != row['bytes']:
                raise RuntimeError('Downloaded receipt bytes changed: ' + name)
            (LOCAL/name).write_bytes(raw)
        (LOCAL/'archive_receipt.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(result, indent=2))
    finally:
        session.close()


if __name__ == '__main__':
    main()
