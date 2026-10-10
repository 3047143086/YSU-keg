"""Detached serial queue for temporal feasibility runs."""
import json, os, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent; SPEC=json.loads((ROOT/'experiment.json').read_text()); ACTIVE=None; START=time.monotonic(); DEADLINE=START+SPEC['max_pipeline_seconds']
def atomic(p,v):
 t=Path(p).with_suffix('.tmp'); t.write_text(json.dumps(v,indent=2)); t.replace(p)
def status(stage,**kw): atomic(ROOT/'pipeline_status.json',dict(stage=stage,updated_unix=time.time(),elapsed_seconds=time.monotonic()-START,**kw)); print(stage,kw,flush=True)
def gpu():
 out=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
 rows=[]
 for line in out.strip().splitlines():
  i,m,u=map(lambda x:int(x.strip()),line.split(',')); rows.append((i,m,u))
 return next((i for i,m,u in rows if i in SPEC['allowed_gpus'] and m<6000 and u<40),None)
def run(seed,var,smoke=False):
 global ACTIVE
 while True:
  if time.monotonic()>DEADLINE: raise TimeoutError('queue deadline')
  device=gpu()
  if device is not None: break
  status('waiting_for_gpu',next_job=f'{seed}_{var}'); time.sleep(20)
 out=ROOT/('smoke' if smoke else 'results')/str(seed)/var; out.mkdir(parents=True,exist_ok=False)
 cmd=[sys.executable,'-u',str(ROOT/'temporal_train.py'),'--interactions',SPEC['interactions'],'--feature-bank',SPEC['feature_bank'],'--output',str(out),'--seed',str(seed),'--variant',var,'--epochs',str(2 if smoke else SPEC['epochs']),'--batch-size',str(SPEC['batch_size']),'--negatives',str(SPEC['negatives']),'--hidden',str(SPEC['hidden']),'--device','cuda:0','--threads',str(SPEC['threads'])]
 if smoke: cmd.append('--smoke')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(device),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
 with (out/'stdout.log').open('x') as log:
  p=subprocess.Popen(cmd,cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL); ACTIVE=p
  while p.poll() is None:
   status('smoke_running' if smoke else 'training',seed=seed,variant=var,child_pid=p.pid,gpu=device); time.sleep(10)
  ACTIVE=None
 if p.returncode or not (out/'result.json').exists(): raise RuntimeError(f'failed {seed} {var}')
 result=json.loads((out/'result.json').read_text())
 if smoke and (result.get('status')!='smoke_passed' or result.get('test_evaluated')): raise RuntimeError('smoke contract failed')
 return result
def main():
 import fcntl
 with (ROOT/'pipeline.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); status('started')
  subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT),'-p','test_*.py','-v'],check=True)
  for v in SPEC['variants']: run(SPEC['seeds'][0],v,True)
  for s in SPEC['seeds']:
   for v in SPEC['variants']: run(s,v)
  rows=[]
  for s in SPEC['seeds']:
   for v in SPEC['variants']: rows.append((s,v,json.loads((ROOT/'results'/str(s)/v/'result.json').read_text())))
  metrics=['recall@20','ndcg@20','recall@10','ndcg@10','recall@200']; agg={}
  for v in SPEC['variants']:
   vals=[r['test_metrics'] for s,x,r in rows if x==v]; agg[v]={m:float(sum(a[m] for a in vals)/len(vals)) for m in metrics}
  baseline=agg['long_mean']; summary={'completed_jobs':len(rows),'total_jobs':9,'aggregate':agg,'baseline':'long_mean','decision':'feasibility_only; no novelty claim','official_validation_evaluated':False,'official_test_evaluated':False}
  atomic(ROOT/'summary.json',summary); (ROOT/'RESULTS.md').write_text(json.dumps(summary,indent=2)); status('completed',completed_jobs=9,total_jobs=9)
if __name__=='__main__':
 try: main()
 except Exception as e: status('failed',error=str(e),traceback=traceback.format_exc()); raise
