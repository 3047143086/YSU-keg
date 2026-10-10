"""Deploy and inspect one deferred GPU cache audit, independent of baseline."""
import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
REMOTE = '/data/sunzixuan/electronics_eval_cache_gpu_audit_20261010_v1'
PYTHON = '/data/sunzixuan/miniconda3/envs/MMRec/bin/python'
NAMES = ('electronics_gpu_audit.py','gpu_manage.py','ELECTRONICS_GPU_REVIEW.md','eval_cache.py',
         'checkpoint_structure.py','test_eval_cache.py','test_cache_lifecycle.py','test_checkpoint_structure.py')


def remote_source(action, payload):
    return r'''
import base64,hashlib,json,os,subprocess,time
from pathlib import Path
c=%r
root=Path(c['root']); action=c['action']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
if action=='deploy':
 if root.exists() and any(root.iterdir()):raise RuntimeError('Nonempty GPU audit target refuses overwrite')
 root.mkdir(parents=True,exist_ok=True)
 for name,encoded in c['payload'].items():(root/name).write_bytes(base64.b64decode(encoded))
 (root/'deployment.json').write_text(json.dumps({'sha256':{n:sha(root/n) for n in c['payload']},'unix':time.time()},indent=2))
 print(json.dumps({'status':'deployed','root':str(root)}))
if action in ('test','start'):
 deployed=json.loads((root/'deployment.json').read_text())
 for name,h in deployed['sha256'].items():
  if sha(root/name)!=h:raise RuntimeError('GPU audit payload changed: '+name)
 cpuenv=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 if action=='test':
  p=subprocess.run([c['python'],'-m','unittest','discover','-s',str(root),'-p','test_*.py','-v'],cwd=str(root),
                   env=cpuenv,capture_output=True,text=True,timeout=180)
  receipt={'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'cuda_visible_devices':'',
           'deployment_sha256':sha(root/'deployment.json')}
  (root/'tests.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
  if p.returncode:raise RuntimeError('GPU audit CPU lifecycle tests failed')
 else:
  tests=json.loads((root/'tests.json').read_text())
  if tests['exit_code'] or tests['deployment_sha256']!=sha(root/'deployment.json'):raise RuntimeError('Current CPU tests required')
  if (root/'launch.json').exists():raise RuntimeError('GPU audit is single-use')
  environment=dict(cpuenv,CUDA_VISIBLE_DEVICES='1')
  log=(root/'audit.log').open('x')
  process=subprocess.Popen([c['python'],'-u',str(root/'electronics_gpu_audit.py')],cwd=str(root),env=environment,
      stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  (root/'launch.json').write_text(json.dumps({'pid':process.pid,'unix':time.time(),
      'command':[c['python'],'-u',str(root/'electronics_gpu_audit.py')],'time_budget_seconds':3600},indent=2))
  print(json.dumps({'status':'started','pid':process.pid,'root':str(root)}))
if action=='status':
 result={}
 for name in ('deployment.json','source_revision.json','tests.json','launch.json','electronics_gpu_receipt.json'):
  p=root/name
  if p.exists():result[name]=json.loads(p.read_text())
 log=root/'audit.log'
 if log.exists():result['audit.log']=log.read_text(errors='replace')[-12000:]
 pid=result.get('launch.json',{}).get('pid')
 p=Path('/proc/'+str(pid))
 result['process_alive']=p.exists() and str(root/'electronics_gpu_audit.py') in (p/'cmdline').read_text()
 print(json.dumps(result,allow_nan=False))
''' % {'root':REMOTE,'python':PYTHON,'action':action,'payload':payload}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('deploy','test','start','status'))
    args=parser.parse_args()
    payload={}
    if args.action=='deploy':
        dirty=subprocess.check_output(['git','status','--porcelain'],cwd=LOCAL,text=True).strip()
        if dirty:raise RuntimeError('GPU audit preparation must be committed')
        revision={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=LOCAL,text=True).strip(),
                  'sha256':{n:hashlib.sha256((LOCAL/n).read_bytes()).hexdigest() for n in NAMES}}
        payload={n:base64.b64encode((LOCAL/n).read_bytes()).decode() for n in NAMES}
        payload['source_revision.json']=base64.b64encode(json.dumps(revision).encode()).decode()
    sys.path.insert(0,str(LOCAL.parent/'20260905'))
    from remote_session import Session
    s=Session('213')
    try:
        code,out=s.run_python(remote_source(args.action,payload),PYTHON,300)
        if args.action=='status' and not code:
            result=json.JSONDecoder().raw_decode(out.lstrip())[0]
            target=LOCAL/'archives'/'gpu_v1';target.mkdir(parents=True,exist_ok=True)
            for name,value in result.items():
                if name.endswith('.json'):(target/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
                elif name=='audit.log':(target/name).write_text(value,encoding='utf-8')
        print(out)
        if code:raise SystemExit(code)
    finally:
        s.close()


if __name__=='__main__':main()
