"""Run only the small registered unit/gradient checks in actual MMRec on CPU."""
import base64
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20260905'))
from remote_session import Session


def main():
    payload = {name: base64.b64encode((HERE / name).read_bytes()).decode('ascii')
               for name in ('projector.py', 'test_projector.py')}
    expected = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                for name in ('projector.py', 'test_projector.py', 'PROTOCOL.md', 'program.md')}
    source = r'''
import base64, contextlib, hashlib, io, json, os, sys, time, types, unittest
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import torch
torch.set_num_threads(4)
payload=%r
started=time.time()
for name in ('projector.py','test_projector.py'):
 code=base64.b64decode(payload[name])
 module=types.ModuleType(name[:-3]);sys.modules[module.__name__]=module
 module.__file__='<small_cpu_test_bundle>/' + name
 exec(compile(code,module.__file__,'exec'),module.__dict__)
stream=io.StringIO()
suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules['test_projector'])
with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
receipt={'status':'passed' if result.wasSuccessful() else 'failed','test_count':result.testsRun,
 'torch_version':torch.__version__,'cuda_visible_devices':'','cuda_initialized':torch.cuda.is_initialized(),
 'threads':torch.get_num_threads(),'elapsed_seconds':time.time()-started,'output':stream.getvalue(),
 'source_files_sha256':{name:hashlib.sha256(base64.b64decode(code)).hexdigest() for name,code in payload.items()},
 'replay_loaded':False,'training_executed':False,'source_modified':False}
print(json.dumps(receipt,indent=2,allow_nan=False))
''' % payload
    session = Session('213')
    try:
        code, output = session.run_python(source, '/data/sunzixuan/miniconda3/envs/MMRec/bin/python', 180)
        if code:
            raise RuntimeError(output[-3000:])
        receipt, _ = json.JSONDecoder().raw_decode(output.lstrip())
        receipt['registered_local_files_sha256'] = expected
        (HERE / 'cpu_test_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(receipt, indent=2))
        if receipt['status'] != 'passed':
            raise SystemExit(1)
    finally:
        session.close()


if __name__ == '__main__':
    main()
