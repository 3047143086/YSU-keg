"""Reuse the frozen CPU manager; add only new cache/provenance input audit."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


LOCAL = Path(__file__).resolve().parent
REMOTE = "/data/sunzixuan/review_bank_alignment_20261010_v1"
BASE_SHA = "74d4d1eb61e5e25cc229a14c91e2b3b404432b9c8e4536fe09ec46e58dd64f4a"
sys.dont_write_bytecode = True
base_paths = (LOCAL.parent / "20261009_review_identifiability" / "manage.py",
              Path("/data/sunzixuan/review_identifiability_20261009_v1/manage.py"))
BASE_PATH = next((path for path in base_paths if path.is_file()), None)
if BASE_PATH is None or hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_SHA:
    raise RuntimeError("Frozen F26 CPU manager is missing or changed")
spec = importlib.util.spec_from_file_location("frozen_cpu_manager", BASE_PATH)
_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_base)
_original_expectations, _original_remote_source = _base.audit_expectations, _base.remote_source
_base.LOCAL, _base.REMOTE = LOCAL, REMOTE
_base.BUNDLE_FILES = ("PROTOCOL.md", "program.md", "core.py", "experiment.py",
                      "manage.py", "test_core.py", "test_manage.py", "input_audit.json")


def audit_expectations():
    expected = _original_expectations()
    residual = LOCAL.parent / "20261009_review_residual"
    expression = LOCAL.parent / "20261009_review_identifiability"
    dual = LOCAL.parent / "20261009_dual_witness"
    previous_audit = json.loads((residual / "input_audit.json").read_text())
    for name in ("dual_witness_diagnose", "dual_witness_evidence"):
        expected["files"][name] = previous_audit["dependencies"][name]
    expected["files"]["cpu_manager"] = {"path":"/data/sunzixuan/review_identifiability_20261009_v1/manage.py",
                                         "sha256":BASE_SHA}
    expected["expression_diagnosis"] = json.loads((expression / "diagnosis.json").read_text())
    expected["dual_diagnosis"] = json.loads((dual / "diagnosis.json").read_text())["baby"]
    expected["extra_paths"] = {
        "f25_per_user":"/data/sunzixuan/review_residual_20261009_v1/cache/per_user.npz",
        "f26_diagnosis":"/data/sunzixuan/review_identifiability_20261009_v1/diagnosis.json",
        "f23_diagnosis":"/data/sunzixuan/dual_witness_20261009_v1/diagnostics/baby/diagnosis.json",
        "f23_per_user":"/data/sunzixuan/dual_witness_20261009_v1/diagnostics/baby/per_user.npz",
    }
    return expected


def remote_source(action, payload=None, expected=None):
    if action != "audit": return _original_remote_source(action, payload, expected)
    return "expected = " + repr(expected) + r'''
import hashlib,json,os,time
from pathlib import Path
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
files=dict(expected['files'])
for name,row in files.items():
    if sha(row['path']) != row['sha256']: raise RuntimeError('Frozen input changed: '+name)
previous=Path(expected['predecessor_root'])
diagnosis_path=previous/'diagnosis.json'
diagnosis=json.loads(diagnosis_path.read_text())
if diagnosis != expected['predecessor_diagnosis']: raise RuntimeError('F25 diagnosis content changed')
if json.loads((previous/'source_revision.json').read_text())['commit']!=expected['predecessor_commit']:
    raise RuntimeError('F25 revision changed')
files['predecessor_diagnosis']={'path':str(diagnosis_path),'sha256':sha(diagnosis_path)}
paths=expected['extra_paths']
expression=json.loads(Path(paths['f26_diagnosis']).read_text())
dual=json.loads(Path(paths['f23_diagnosis']).read_text())
if expression!=expected['expression_diagnosis'] or expression['status']!='completed':
    raise RuntimeError('F26 expression provenance changed')
if dual!=expected['dual_diagnosis'] or dual['status']!='completed':
    raise RuntimeError('F23 control provenance changed')
with np.load(paths['f25_per_user'],allow_pickle=False) as a, np.load(paths['f23_per_user'],allow_pickle=False) as b:
    for name in ('user_ids','targets','comparisons'):
        if not np.array_equal(a[name],b[name]): raise RuntimeError('F23/F25 frozen comparison mismatch')
    if a['user_ids'].shape!=(13447,) or a['comparisons'].shape!=(13447,17):
        raise RuntimeError('Frozen diagnostic population changed')
    for name in ('raw_own','same_item_donor','item_content','residual','residual_donor'):
        value=a[name]
        if value.shape!=(13447,) or not np.isfinite(value).all() or abs(value.mean()-diagnosis['metrics'][name]['mean_auc'])>1e-12:
            raise RuntimeError('F25 cached control disagrees with its frozen diagnosis')
    for name in ('itemknn','userknn','min2','dual','shuffled'):
        if b[name].shape!=(13447,) or not np.isfinite(b[name]).all() or abs(b[name].mean()-dual['arms'][name]['mean_auc'])>1e-12:
            raise RuntimeError('F23 cached control disagrees with its frozen diagnosis')
for name,path in paths.items(): files[name]={'path':path,'sha256':sha(path)}
output={'status':'completed','checked_unix':time.time(),'predecessor_commit':expected['predecessor_commit'],
        'files':{name:dict(row,bytes=Path(row['path']).stat().st_size) for name,row in files.items()},
        'raw_reviews_read':False,'interactions_read':False,'gpu_used':False,
        'encoding_or_training_started':False,'validation_evaluated':False,'test_evaluated':False,
        'diagnostic_users':13447,'cached_support_edges':105032,'full_support_reference_edges':105104,
        'prior_aucs_checked_against_receipts':True,'new_candidate_metrics_computed':False}
print(json.dumps(output,allow_nan=False))
'''


_base.audit_expectations = audit_expectations
_base.remote_source = remote_source


if __name__ == "__main__":
    _base.main()
