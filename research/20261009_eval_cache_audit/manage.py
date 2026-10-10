"""Isolated CPU equivalence audit; no training or GPU commands."""
import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
REMOTE = '/data/sunzixuan/eval_cache_audit_20261009_v3'
PYTHON = '/data/sunzixuan/miniconda3/envs/MMRec/bin/python'
PAYLOAD = ('eval_cache.py', 'test_eval_cache.py', 'test_cache_lifecycle.py', 'actual_baby_cpu.py',
           'PROTOCOL.md', 'CPU_TEST_PLAN.md', 'program.md', 'manage.py', 'input_audit.json',
           'checkpoint_structure.py', 'test_checkpoint_structure.py')


def source_revision():
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=LOCAL, text=True).strip()
    if dirty:
        raise RuntimeError('Commit the independent audit bundle before deployment')
    return {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=LOCAL, text=True).strip(),
            'files': {n: hashlib.sha256((LOCAL/n).read_bytes()).hexdigest() for n in PAYLOAD}}


def remote_source(action, payload):
    return r'''
import base64, hashlib, json, os, subprocess, sys, time
from pathlib import Path
c = %r
root = Path(c['root'])
action = c['action']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
if action=='probe':
    q=Path('/data/sunzixuan/candidate_gate_20261009_v1')
    b=Path('/data/sunzixuan/target_isolated_20261008_autoresearch_v1')
    print(json.dumps({'baby_checkpoints':[str(p) for p in q.rglob('best.pt') if 'baby' in str(p)],
                      'remote_exists':root.exists(),
                      'baseline_anchors':{str(p):sha(p) for p in
                        [b/'deployment.json']+[b/'runs/baby_baseline_s2026'/n for n in
                         ('inputs.json','result.json','config.json','best.pt')]}}
                     ,allow_nan=False))
if action=='deploy':
    if root.exists() and any(root.iterdir()): raise RuntimeError('Nonempty audit target refuses overwrite')
    root.mkdir(parents=True,exist_ok=True)
    for n, encoded in c['payload'].items(): (root/n).write_bytes(base64.b64decode(encoded))
    manifest={n:sha(root/n) for n in c['payload']}
    (root/'deployment.json').write_text(json.dumps({'sha256':manifest,'unix':time.time()},indent=2))
    print(json.dumps({'status':'deployed','root':str(root)},allow_nan=False))
if action in ('test','run'):
    deployment=json.loads((root/'deployment.json').read_text())
    for n,h in deployment['sha256'].items():
        if sha(root/n)!=h: raise RuntimeError('Audit payload changed: '+n)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
             OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    if action=='test':
        process=subprocess.run([c['python'],'-u','-m','unittest','discover','-s',str(root),'-p','test_*.py','-v'],
                               cwd=str(root),env=env,capture_output=True,text=True,timeout=120)
        receipt={'exit_code':process.returncode,'stdout':process.stdout,'stderr':process.stderr,
                 'cuda_visible_devices':'','deployment_sha256':sha(root/'deployment.json')}
        (root/'tests.json').write_text(json.dumps(receipt,indent=2))
        print(json.dumps(receipt,allow_nan=False))
        if process.returncode: raise RuntimeError('Lifecycle tests failed')
    else:
        tests=json.loads((root/'tests.json').read_text())
        if tests['exit_code'] or tests['deployment_sha256']!=sha(root/'deployment.json'):
            raise RuntimeError('Current bundle lacks passing lifecycle tests')
        if (root/'run_started.json').exists(): raise RuntimeError('Audit is single-use')
        (root/'run_started.json').write_text(json.dumps({'unix':time.time()}))
        process=subprocess.run([c['python'],'-u',str(root/'actual_baby_cpu.py')],cwd=str(root),env=env,
                               capture_output=True,text=True,timeout=1200)
        (root/'audit.log').write_text(process.stdout+'\n'+process.stderr)
        print(json.dumps({'exit_code':process.returncode,'stdout':process.stdout[-5000:],
                          'stderr':process.stderr[-5000:]}))
        if process.returncode: raise RuntimeError('Actual Baby CPU equivalence audit failed')
if action=='status':
    result={}
    for n in ('deployment.json','source_revision.json','tests.json','receipt.json','run_started.json'):
        p=root/n
        if p.exists():result[n]=json.loads(p.read_text())
    log=root/'audit.log'
    if log.exists():result['audit.log']=log.read_text()[-5000:]
    print(json.dumps(result,allow_nan=False))
''' % {'root':REMOTE,'python':PYTHON,'action':action,'payload':payload}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('probe','deploy','test','run','status'))
    args=parser.parse_args()
    data={}
    if args.action=='deploy':
        data={n:base64.b64encode((LOCAL/n).read_bytes()).decode() for n in PAYLOAD}
        data['source_revision.json']=base64.b64encode(json.dumps(source_revision()).encode()).decode()
    sys.path.insert(0,str(LOCAL.parent/'20260905'))
    from remote_session import Session
    session=Session('213')
    try:
        code,output=session.run_python(remote_source(args.action,data),PYTHON,1400 if args.action=='run' else 180)
        print(output)
        if args.action=='status' and code==0:
            result=json.JSONDecoder().raw_decode(output.lstrip())[0]
            for n,value in result.items():
                if n.endswith('.json'): (LOCAL/n).write_text(json.dumps(value,indent=2),encoding='utf-8')
                elif n=='audit.log': (LOCAL/n).write_text(value,encoding='utf-8')
        if code: raise SystemExit(code)
    finally:
        session.close()


if __name__=='__main__': main()
