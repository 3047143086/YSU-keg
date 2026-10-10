"""Reviewable one-launch deployment; anchors/status are read-only on the server."""
import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
REMOTE = '/data/sunzixuan/same_input_integration_audit_20261010_v1'
PREDECESSOR = '/data/sunzixuan/failure_model_gpu_audit_20261010_v1'
PYTHON = '/data/sunzixuan/miniconda3/envs/MMRec/bin/python'
NAMES = ('remote_integration.py', 'repeat_guards.py', 'runtime_checks.py', 'integration_gate.py',
         'method_observer.py', 'warning_observer.py', 'parameter_snapshot.py', 'test_parameter_snapshot.py',
         'test_integration_gate.py', 'test_method_observer.py', 'test_warning_observer.py', 'test_runtime_compatibility.py',
         'manage.py', 'PROTOCOL.md', 'program.md', 'ROOT_REVIEW.md')


def remote_source(configuration):
    return r'''
import base64,hashlib,json,os,subprocess,sys,time
from pathlib import Path
c=%r
root=Path(c['root']);action=c['action']
def sha(p):
 digest=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
 return digest.hexdigest()
if action=='anchors':
 predecessor=Path(c['predecessor'])
 names=('remote_gpu.py','guards.py','audit_gate.py','audit_config.json','gpu_audit_receipt.json')
 rows={name:sha(predecessor/name) for name in names}
 config=json.loads((predecessor/'audit_config.json').read_text())
 receipt=json.loads((predecessor/'gpu_audit_receipt.json').read_text())
 print(json.dumps({'root':str(predecessor),'sha256':rows,'source_revision':config['source_revision'],
                  'receipt_source_revision':receipt['source_revision'],'receipt_status':receipt['status']}))
if action=='deploy':
 root.mkdir(parents=True,exist_ok=False)
 for name,encoded in c['payload'].items():
  content=base64.b64decode(encoded)
  if hashlib.sha256(content).hexdigest()!=c['payload_sha256'][name]:raise RuntimeError('Transfer changed '+name)
  (root/name).write_bytes(content)
 (root/'integration_config.json').write_text(json.dumps({k:c[k] for k in ('root','source_revision','payload_sha256')},indent=2))
 print(json.dumps({'status':'deployed','root':str(root),'source_revision':c['source_revision']}))
if action in ('test','start'):
 config=json.loads((root/'integration_config.json').read_text())
 for name,h in config['payload_sha256'].items():
  if sha(root/name)!=h:raise RuntimeError('Frozen payload changed: '+name)
 cpuenv=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
 if action=='test':
  sys.path.insert(0,str(root))
  from repeat_guards import frozen_helpers
  helper,guard,predecessor=frozen_helpers()
  protected=guard.verify(root,config)
  p=subprocess.run([c['python'],'-m','unittest','discover','-s',str(root),'-p','test_*.py','-v'],
      cwd=str(root),env=cpuenv,capture_output=True,text=True,timeout=180)
  receipt={'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'cuda_visible_devices':'',
           'integration_config_sha256':sha(root/'integration_config.json'),'protected_count':protected['count'],
           'predecessor_anchors_match':predecessor['all_match']}
  with (root/'tests.json').open('x') as f:json.dump(receipt,f,indent=2)
  print(json.dumps(receipt))
  if p.returncode:raise RuntimeError('Actual environment diagnostic tests failed')
 else:
  tests=json.loads((root/'tests.json').read_text())
  if tests['exit_code'] or tests['integration_config_sha256']!=sha(root/'integration_config.json'):raise RuntimeError('Matching passing actual tests required')
  sys.path.insert(0,str(root))
  from repeat_guards import frozen_helpers
  helper,guard,predecessor=frozen_helpers()
  protected=guard.verify(root,config)
  resource_gate=guard.idle_gpu_resources()
  ticket=root/'launch.json'
  with ticket.open('x') as f:json.dump({'status':'reserved','unix':time.time(),'resource_gate':resource_gate},f)
  log=(root/'integration.log').open('x')
  command=[c['python'],'-u',str(root/'remote_integration.py')]
  process=subprocess.Popen(command,cwd=str(root),env=dict(cpuenv,CUDA_VISIBLE_DEVICES='1'),stdout=log,
      stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  launch={'status':'started','pid':process.pid,'unix':time.time(),'command':command,
          'time_budget_seconds':1800,'resource_gate':resource_gate,'source_revision':config['source_revision'],
          'predecessor':predecessor,'protected_count':protected['count']}
  ticket.write_text(json.dumps(launch,indent=2));print(json.dumps(launch))
if action=='status':
 result={}
 for name in ('integration_config.json','tests.json','launch.json','integration_progress.json','integration_receipt.json'):
  p=root/name
  if p.exists():result[name]=json.loads(p.read_text())
 pid=result.get('launch.json',{}).get('pid');p=Path('/proc/'+str(pid))
 try:result['process_alive']=p.exists() and str(root/'remote_integration.py') in (p/'cmdline').read_text()
 except FileNotFoundError:result['process_alive']=False
 log=root/'integration.log'
 if log.exists():result['log_tail']=log.read_text(errors='replace')[-4000:]
 print(json.dumps(result,allow_nan=False))
''' % configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('anchors', 'deploy', 'test', 'start', 'status'))
    args = parser.parse_args()
    config = {'root': REMOTE, 'python': PYTHON, 'action': args.action, 'predecessor': PREDECESSOR}
    if args.action == 'deploy':
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=LOCAL, text=True).strip():
            raise RuntimeError('Commit the reviewed payload before deployment')
        config['source_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=LOCAL, text=True).strip()
        config['payload'] = {name: base64.b64encode((LOCAL/name).read_bytes()).decode() for name in NAMES}
        config['payload_sha256'] = {name: hashlib.sha256((LOCAL/name).read_bytes()).hexdigest() for name in NAMES}
    sys.path.insert(0, str(LOCAL.parent/'20260905'))
    from remote_session import Session
    session = Session('213')
    try:
        code, output = session.run_python(remote_source(config), PYTHON, 300)
        if code:
            (LOCAL/'transport_failure.log').write_text(output, encoding='utf-8')
            raise RuntimeError('Remote command failed; see local transport_failure.log')
        result = json.JSONDecoder().raw_decode(output.lstrip())[0]
        if args.action == 'status':
            for name, value in result.items():
                if name.endswith('.json'):
                    (LOCAL/name).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
            receipt = result.get('integration_receipt.json', {})
            print(json.dumps({'process_alive': result.get('process_alive'), 'status': receipt.get('status'),
                              'elapsed_seconds': receipt.get('elapsed_seconds'), 'error': receipt.get('error'),
                              'full_forward_count': receipt.get('full_forward_count'), 'backward_calls': receipt.get('backward_calls'),
                              'exact_bands': sum(sum(row.get('bands_exact', [])) for row in receipt.get('rows', [])),
                              'participating_parameters': len(receipt.get('gradients', {}))}, indent=2))
        else:
            (LOCAL/(args.action+'_receipt.json')).write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
            print(json.dumps(result, indent=2))
    finally:
        session.close()


if __name__ == '__main__':
    main()
