"""Execute the reviewed audit in its own immutable remote directory on CPU."""
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE = '/data/sunzixuan/spectral_projector_audit_20261010_v1'
PAYLOAD_NAMES = ('projector.py', 'test_projector.py', 'remote_audit.py', 'replay_cpu.py',
                 'test_cpu_remote.py', 'PROTOCOL.md', 'program.md')
sys.path.insert(0, str(HERE.parent / '20260905'))
from remote_session import Session


def main():
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(HERE), text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'],
                                    cwd=str(HERE), text=True).strip()
    if dirty:
        raise RuntimeError('Audit must be committed and clean before actual replay')
    hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in PAYLOAD_NAMES}
    test = json.loads((HERE / 'cpu_test_receipt.json').read_text())
    if (test['status'] != 'passed' or test['source_files_sha256']['projector.py'] != hashes['projector.py']
            or test['source_files_sha256']['test_projector.py'] != hashes['test_projector.py']
            or test['registered_local_files_sha256']['PROTOCOL.md'] != hashes['PROTOCOL.md']):
        raise RuntimeError('Current reviewed audit lacks matching small CPU test receipt')
    payload = {name: base64.b64encode((HERE / name).read_bytes()).decode('ascii') for name in PAYLOAD_NAMES}
    configuration = {'root': REMOTE, 'source_revision': revision, 'payload_sha256': hashes,
                     'root_review_required': True, 'cpu_only': True}
    source = r'''
import base64, hashlib, json, os, runpy, sys
from pathlib import Path
configuration=%r
payload=%r
root=Path(configuration['root'])
root.mkdir(parents=True,exist_ok=False)
for name,encoded in payload.items():
 content=base64.b64decode(encoded)
 if hashlib.sha256(content).hexdigest()!=configuration['payload_sha256'][name]:
  raise RuntimeError('Payload content drift: '+name)
 (root/name).write_bytes(content)
(root/'audit_config.json').write_text(json.dumps(configuration,indent=2))
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(root))
runpy.run_path(str(root/'remote_audit.py'),run_name='__main__')
''' % (configuration, payload)
    session = Session('213')
    try:
        code, output = session.run_python(source, '/data/sunzixuan/miniconda3/envs/MMRec/bin/python', 1300)
        marker = 'AUDIT_FINAL_RECEIPT_JSON'
        if code or marker not in output:
            (HERE / 'replay_transport_failure.log').write_text(output, encoding='utf-8')
            raise RuntimeError('Audit transport did not return a final receipt; inspect remote before retry')
        receipt, _ = json.JSONDecoder().raw_decode(output.split(marker, 1)[1].lstrip())
        (HERE / 'replay_audit_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        summary = {key: receipt.get(key) for key in ('status', 'source_revision', 'elapsed_seconds',
                   'all_twelve_captures_inspected', 'candidate_mapping_and_finite_gradients_passed',
                   'available_generic_gradient_comparisons', 'available_generic_gradient_comparisons_passed',
                   'cuda_initialized', 'maximum_resident_set_bytes', 'error')}
        summary['captures'] = [{key: row.get(key) for key in ('capture_index', 'exact_forward_bands',
                 'candidate_input_gradient_finite', 'generic_input_gradient_finite',
                 'candidate_mapping_and_finite_gradient_passed', 'error')}
                               for row in receipt['captures']]
        print(json.dumps(summary, indent=2))
    finally:
        session.close()


if __name__ == '__main__':
    main()
