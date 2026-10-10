"""Run the complete preparation tests in a temporary CPU-only MMRec sandbox."""
import base64
import argparse
import hashlib
import json
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-cpu',action='store_true',required=True,
                        help='Root-authorized temporary CPU tests only; never deploy or train')
    parser.parse_args()
    from payload_manifest import payload_names
    names = tuple(name for name in payload_names(HERE) if name != 'preparation_cpu_receipt.json')
    hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names}
    payload = {name: base64.b64encode((HERE / name).read_bytes()).decode() for name in names}
    source = r'''
import base64,hashlib,io,json,os,sys,tempfile,time,unittest
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
import torch
torch.set_num_threads(4)
payload=%r
expected=%r
started=time.monotonic()
with tempfile.TemporaryDirectory(prefix='recovery_v3_cpu_') as directory:
 root=Path(directory)
 for name,encoded in payload.items():
  content=base64.b64decode(encoded)
  if hashlib.sha256(content).hexdigest()!=expected[name]:raise RuntimeError('CPU transfer changed '+name)
  path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
 sys.path.insert(0,str(root))
 os.chdir(root)
 suite=unittest.defaultTestLoader.discover(str(root),pattern='test_*.py')
 stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
 result={'status':'passed' if r.wasSuccessful() and not r.skipped else 'failed','tests':r.testsRun,
  'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped),'output':stream.getvalue(),
  'cuda_initialized':torch.cuda.is_initialized(),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
  'threads':torch.get_num_threads(),'torch_version':torch.__version__,'elapsed_seconds':time.monotonic()-started,
  'source_files_sha256':expected,'temporary_cpu_sandbox':True,'training_executed':False,
  'action':'prepare_cpu_only','authorization':'explicit root task message; no deployment/training authority',
  'recommendation_metrics_evaluated':False,'official_validation_or_test_used':False,'recovery_deployed':False}
print(json.dumps(result,allow_nan=False))
''' % (payload, hashes)
    sys.path.insert(0, str(HERE.parent / '20260905'))
    from remote_session import Session
    session = Session('213')
    try:
        encoded=base64.b64encode(zlib.compress(source.encode('utf-8'),9)).decode('ascii')
        if zlib.decompress(base64.b64decode(encoded)).decode('utf-8') != source:
            raise RuntimeError('CPU source compression changed bytes')
        wrapper='import base64,zlib\nexec(zlib.decompress(base64.b64decode('+repr(encoded)+')))'
        code, output = session.run_python(wrapper, '/data/sunzixuan/miniconda3/envs/MMRec/bin/python', 300)
        if code:
            raise RuntimeError('Actual CPU test transport failed: ' + output[-4000:])
        result = json.JSONDecoder().raw_decode(output.lstrip())[0]
        with (HERE / 'preparation_cpu_receipt.json').open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(json.dumps({k: v for k, v in result.items() if k != 'output' and k != 'source_files_sha256'}, indent=2))
        if result['status'] != 'passed' or result['cuda_initialized'] is not False:
            raise RuntimeError('Actual CPU preparation tests failed')
    finally:
        session.close()


if __name__ == '__main__':
    main()
