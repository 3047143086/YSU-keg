"""Read-only replay and source metadata inspection on the isolated remote run."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20260905'))
from remote_session import Session

SCRIPT = r'''
import hashlib, json, os, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
ROOT=Path('/data/sunzixuan/electronics_baseline_20261009_v1')
RUN=ROOT/'runs/elec_baseline_s2026'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
out={'checked_unix':time.time(),'read_only':True}
for name in ('best.pt','best_rng.pt','failure_replay.pt'):
 p=RUN/name; out[name]={'path':str(p),'exists':p.is_file()}
 if p.is_file():
  out[name].update({'bytes':p.stat().st_size,'sha256':sha(p),'mtime_unix':p.stat().st_mtime})
try:
 import torch
 torch.set_num_threads(4)
 packet=torch.load(str(RUN/'failure_replay.pt'),map_location='cpu')
 out['replay_record']=packet.get('record',{})
 captures=[]
 for cap in packet.get('svd_captures',[]):
  s=cap['singular']; g=cap.get('output_gradients',{})
  gaps=s[:-1]-s[1:]
  row={'shape':cap.get('shape'),'singular_count':int(s.numel()),
       'equal_adjacent_singular_values':int(cap.get('equal_adjacent_singular_values',0)),
       'min_adjacent_gap_recorded':cap.get('min_adjacent_gap'),
       'min_adjacent_gap_recomputed':float(gaps.min().item()) if gaps.numel() else None,
       'max_singular':float(s.max().item()),'min_singular':float(s.min().item()),
       'singular_finite':bool(torch.isfinite(s).all().item()),
       'singular_values':s.tolist(), 'adjacent_gaps':gaps.tolist(),
       'gradient_finite':{str(k):bool(torch.isfinite(v).all().item()) for k,v in g.items()},
       'gradient_shapes':{str(k):list(v.shape) for k,v in g.items()}}
  eq=(gaps==0).nonzero(as_tuple=False).flatten()
  row['equal_gap_indices_first_50']=eq[:50].tolist()
  row['near_zero_gap_count_1e-7']=int((gaps.abs()<=1e-7).sum().item())
  row['near_zero_gap_count_1e-6']=int((gaps.abs()<=1e-6).sum().item())
  row['near_zero_gap_count_1e-5']=int((gaps.abs()<=1e-5).sum().item())
  captures.append(row)
 out['svd_captures']=captures
 del packet
except Exception as e:
 out['replay_load_error']=repr(e)
for name in ('models/dual_frequency_encoder.py','common/encoders.py','models/bridge.py'):
 source=ROOT/'carrier/src'/name
 lines=source.read_text(errors='replace').splitlines()
 hits=[i for i,line in enumerate(lines) if 'svd' in line.lower() or 'frequency_decompose' in line.lower() or 'band' in line.lower()]
 selected=set()
 for i in hits:
  selected.update(range(max(0,i-10),min(len(lines),i+12)))
 out.setdefault('source_svd_context',{})[name]={'path':str(source),'sha256':sha(source),
  'lines':{str(i+1):lines[i] for i in sorted(selected)}}
for name in ('graph_manifest.json','pipeline_status.json','results.tsv'):
 p=ROOT/name
 if p.is_file():
  try:out[name]=json.loads(p.read_text())
  except:out[name]=p.read_text(errors='replace')
out['checked_finished_unix']=time.time()
print(json.dumps(out,indent=2,allow_nan=False))
'''

session = Session('213')
try:
    code, output = session.run_python(SCRIPT, '/data/sunzixuan/miniconda3/envs/MMRec/bin/python', 300)
    if code:
        raise RuntimeError(output[-2000:])
    value, _ = json.JSONDecoder().raw_decode(output.lstrip())
    (HERE / 'replay_metadata.json').write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    keys = ('best.pt', 'best_rng.pt', 'failure_replay.pt', 'replay_record', 'svd_captures', 'replay_load_error', 'graph_manifest.json')
    print(json.dumps({key: value.get(key) for key in keys}, indent=2, ensure_ascii=False))
finally:
    session.close()
