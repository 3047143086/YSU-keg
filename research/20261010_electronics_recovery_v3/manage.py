"""Manage the isolated v3 only after complete evidence and explicit root review."""
import argparse
import base64
import hashlib
import json
import subprocess
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE = '/data/sunzixuan/electronics_baseline_20261010_recovery_v3'
PYTHON = '/data/sunzixuan/miniconda3/envs/MMRec/bin/python'
from payload_manifest import payload_names


def revision():
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'],
                               cwd=HERE, text=True).strip():
        raise RuntimeError('Commit the isolated recovery preparation before deployment')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()


def remote_source(configuration):
    return r'''
import base64,hashlib,io,json,os,shutil,subprocess,sys,time,unittest
from pathlib import Path
c=%r
root=Path(c['root']); action=c['action']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_json(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
if action=='deploy':
 root.mkdir(parents=True,exist_ok=False)
 for name,encoded in c['payload'].items():
  content=base64.b64decode(encoded)
  if hashlib.sha256(content).hexdigest()!=c['payload_sha256'][name]:raise RuntimeError('Transferred payload drift: '+name)
  path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
 sys.path.insert(0,str(root))
 from prerequisite_gate import V1,verify_frozen
 verify_frozen(root,deployment_required=False)
 audit=json.loads((root/'input_audit.json').read_text())
 carrier=root/'carrier/src';carrier.mkdir(parents=True,exist_ok=False)
 old=json.loads((V1/'deployment.json').read_text())
 for name,h in old['carrier_sha256'].items():
  source=V1/'carrier/src'/name
  if sha(source)!=h:raise RuntimeError('V1 carrier changed: '+name)
  destination=carrier/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
 data=root/'carrier/data/elec';data.mkdir(parents=True,exist_ok=False)
 links={name:row['path'] for name,row in audit['original_inputs'].items()}
 links['mm_adj_10.pt']=str(V1/'carrier/data/elec/mm_adj_10.pt')
 for name,source in links.items():(data/name).symlink_to(source)
 write_json(root/'source_revision.json',{'commit':c['source_revision'],'working_tree_clean':True,
  'payload_sha256':c['payload_sha256']})
 hashes=dict(c['payload_sha256']);hashes['source_revision.json']=sha(root/'source_revision.json')
 deployment={'bundle_sha256':hashes,'carrier_sha256':old['carrier_sha256'],'data_links':links,
  'source_revision':c['source_revision'],'created_unix':time.time(),'eval_cache_integrated':False,
  'training_started':False,'test_evaluated':False}
 write_json(root/'deployment.json',deployment)
 verify_frozen(root)
 print(json.dumps({'status':'prepared_only','root':str(root),'source_revision':c['source_revision'],
  'training_started':False,'eval_cache_integrated':False}))
if action in ('verify','start','status'):
 if not root.exists():
  print(json.dumps({'status':'not_deployed','training_started':False}));sys.exit(0)
 sys.path.insert(0,str(root))
 from prerequisite_gate import inspect_prerequisites,require_prerequisites,verify_frozen
 prerequisite=inspect_prerequisites(json.loads((root/'prerequisites.json').read_text()),base_root=root)
 if action=='verify':
  protected=verify_frozen(root)
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
   OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',PYTHONPATH=str(root))
  source="import io,json,torch,unittest;torch.set_num_threads(4);s=io.StringIO();suite=unittest.defaultTestLoader.discover('.',pattern='test_*.py');r=unittest.TextTestRunner(stream=s,verbosity=2).run(suite);print(json.dumps({'status':'passed' if r.wasSuccessful() and not r.skipped else 'failed','tests':r.testsRun,'skipped':len(r.skipped),'failures':len(r.failures),'errors':len(r.errors),'output':s.getvalue(),'cuda_initialized':torch.cuda.is_initialized(),'torch_version':torch.__version__,'threads':torch.get_num_threads()}))"
  p=subprocess.run([c['python'],'-c',source],cwd=str(root),env=env,capture_output=True,text=True,timeout=300)
  tests=json.loads(p.stdout) if p.returncode==0 else {'status':'failed','stderr':p.stderr,'exit_code':p.returncode}
  tests.update(deployment_sha256=sha(root/'deployment.json'),prerequisites=prerequisite,protected=protected)
  write_json(root/'test_receipt.json',tests)
  print(json.dumps(tests))
  if tests['status']!='passed' or tests.get('cuda_initialized') is not False:raise RuntimeError('Actual MMRec CPU tests failed')
 if action=='start':
  # Require every independently anchored v3 role before reservations or GPU queries.
  require_prerequisites(root);verify_frozen(root)
  tests=json.loads((root/'test_receipt.json').read_text())
  if tests.get('status')!='passed' or tests.get('skipped')!=0 or tests.get('deployment_sha256')!=sha(root/'deployment.json'):
   raise RuntimeError('Matching actual CPU tests required')
  from run_queue import resources
  resource=resources()
  if not(resource['query_succeeded'] and not resource['gpu1_compute_processes'] and
   resource['gpu_free_bytes']>=14*1024**3 and resource['ram_available_bytes']>=12*1024**3 and resource['available_cpu_count']>=4):
   raise RuntimeError('Fresh GPU1/RAM/CPU process snapshot failed')
  write_json(root/'launch.reservation.json',{'unix':time.time(),'source_revision':json.loads((root/'source_revision.json').read_text())['commit']})
  log=(root/'queue.log').open('x')
  command=[c['python'],'-u',str(root/'run_queue.py')]
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
  process=subprocess.Popen(command,cwd=str(root),env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  launch={'pid':process.pid,'command':command,'unix':time.time(),'resource_snapshot':resource,'queue_seconds':72*3600,'job_seconds':24*3600}
  write_json(root/'launch.json',launch);print(json.dumps(launch))
 if action=='status':
  result={'prerequisites':prerequisite,'training_eligible':prerequisite['eligible'],'training_started':False}
  for name in ('source_revision.json','deployment.json','test_receipt.json','launch.json','pipeline_status.json'):
   path=root/name
   if path.exists():result[name]=json.loads(path.read_text())
  launch=result.get('launch.json',{});pid=launch.get('pid')
  try:alive=Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0')[:-1]==[v.encode() for v in launch['command']]
  except (FileNotFoundError,KeyError):alive=False
  result.update(process_alive=alive,training_started=bool(launch))
  print(json.dumps(result))
''' % configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('deploy', 'verify', 'start', 'status'))
    args = parser.parse_args()
    # Fail locally before credentials, a connection or a remote file is touched.
    from prerequisite_gate import require_root_review, require_prerequisites
    require_root_review(HERE, args.action)
    require_prerequisites(HERE)
    config = {'root': REMOTE, 'python': PYTHON, 'action': args.action}
    if args.action == 'deploy':
        config['source_revision'] = revision()
        names = payload_names(HERE)
        config['payload'] = {name: base64.b64encode((HERE / name).read_bytes()).decode() for name in names}
        config['payload_sha256'] = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names}
    sys.path.insert(0, str(HERE.parent / '20260905'))
    from remote_session import Session
    session = Session('213')
    try:
        source=remote_source(config)
        encoded=base64.b64encode(zlib.compress(source.encode('utf-8'),9)).decode('ascii')
        if zlib.decompress(base64.b64decode(encoded)).decode('utf-8') != source:
            raise RuntimeError('Remote source compression changed bytes')
        wrapper='import base64,zlib\nexec(zlib.decompress(base64.b64decode('+repr(encoded)+')))'
        code, output = session.run_python(wrapper, PYTHON, 600)
        print(output)
        if code:
            raise SystemExit(code)
    finally:
        session.close()


if __name__ == '__main__':
    main()
