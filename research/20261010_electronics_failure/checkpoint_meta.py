"""Inspect trusted saved checkpoint lineage on CPU without a model forward."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20260905'))
from remote_session import Session

SCRIPT = r'''
import gc, hashlib, json, os, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import torch
torch.set_num_threads(4)
RUN=Path('/data/sunzixuan/electronics_baseline_20261009_v1/runs/elec_baseline_s2026')
root=RUN.parents[1]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
out={'checked_unix':time.time(),'read_only':True,'torch_version':torch.__version__,
 'gpu_forward_executed':False,'model_instantiated':False}
best=torch.load(str(RUN/'best.pt'),map_location='cpu')
recorded=json.loads((RUN/'inputs.json').read_text())
config=json.loads((RUN/'config.json').read_text())
normalized_config=json.loads(json.dumps(best['config'],default=str))
out['checkpoint']={'path':str(RUN/'best.pt'),'bytes':(RUN/'best.pt').stat().st_size,
 'sha256':sha(RUN/'best.pt'),'keys':sorted(best),'epoch':best['epoch'],
 'selection_score':best['selection_score'],'validation':best['validation'],
 'config_matches_effective_run_config':normalized_config==config['effective_config'],
 'inputs_match_run_inputs':best['inputs_sha256']==recorded['sha256'],
 'seed':normalized_config.get('seed'),'dataset':normalized_config.get('dataset'),
 'max_epochs':normalized_config.get('epochs'),'train_batch_size':normalized_config.get('train_batch_size'),
 'learning_rate':normalized_config.get('learning_rate'),'freq_decomp_method':normalized_config.get('freq_decomp_method'),
 'num_freq_bands':normalized_config.get('num_freq_bands'),
 'run_inputs_sha256':sha(RUN/'inputs.json'),'run_config_sha256':sha(RUN/'config.json'),
 'source_revision_sha256':sha(root/'source_revision.json'),
 'source_revision':json.loads((root/'source_revision.json').read_text())['commit'],
 'finite_model_state':{},'finite_optimizer_state':True}
for name,t in best['model'].items():
 if torch.is_tensor(t):
  values=t.coalesce().values() if t.is_sparse else t
  out['checkpoint']['finite_model_state'][name]={'shape':list(t.shape),'dtype':str(t.dtype),
   'is_sparse':bool(t.is_sparse),'finite':bool(torch.isfinite(values).all().item())}
for state in best['optimizer']['state'].values():
 for value in state.values():
  if torch.is_tensor(value) and not torch.isfinite(value).all():
   out['checkpoint']['finite_optimizer_state']=False
out['checkpoint']['all_model_state_finite']=all(row['finite'] for row in out['checkpoint']['finite_model_state'].values())
out['checkpoint']['optimizer_parameter_groups']=len(best['optimizer']['param_groups'])
out['checkpoint']['scheduler']=best['scheduler']
del best;gc.collect()
rng=torch.load(str(RUN/'best_rng.pt'),map_location='cpu')
out['rng_packet']={'path':str(RUN/'best_rng.pt'),'sha256':sha(RUN/'best_rng.pt'),'epoch':rng.get('epoch'),
 'keys':sorted(rng),'all_items_count':len(rng.get('all_items',[])),
 'dataframe_shape':list(rng['dataframe'].shape),'dataframe_columns':list(rng['dataframe'].columns),
 'loader_pr':rng.get('loader_pr'),'loader_inter_pr':rng.get('loader_inter_pr'),
 'test_evaluated':rng.get('test_evaluated'),'rng_components':sorted(rng['rng']),
 'cuda_rng_count':len(rng['rng'].get('cuda',[]))}
out['checked_finished_unix']=time.time()
print(json.dumps(out,indent=2,default=str,allow_nan=False))
'''

session = Session('213')
try:
    code, output = session.run_python(SCRIPT, '/data/sunzixuan/miniconda3/envs/MMRec/bin/python', 300)
    if code:
        raise RuntimeError(output[-2000:])
    value, _ = json.JSONDecoder().raw_decode(output.lstrip())
    (HERE / 'checkpoint_metadata.json').write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    compact = dict(value['checkpoint'])
    compact.pop('finite_model_state')
    print(json.dumps({'checkpoint': compact, 'rng_packet': value['rng_packet']}, indent=2, ensure_ascii=False))
finally:
    session.close()
