"""Frozen, bounded MSCA curvature diagnosis; private batches stay remote."""
import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = '/data/sunzixuan/interest_negative_20261010_v1'
RUN_ROOT = '/data/sunzixuan/curvature_diagnostic_20261010_v1'
PYTHON = '/data/sunzixuan/miniconda3/envs/MMRec/bin/python'
CHECKPOINT_SHA = '6f6ee28429668b9bb7d54d4673252c8751978f22cd24db334d03d7b4234acbd8'
INITIAL_SHA = '4aacd659cec4a99f4e2e1898c681bb44fd302479b71f031af1794af7a5900c40'
BLOCKS = {'image': ('image_trs.', 'gate_v.'), 'text': ('text_trs.', 'gate_t.')}

# This core is shared verbatim by CPU fixtures and the GPU diagnostic.
CORE = r'''
BLOCKS = {'image': ('image_trs.', 'gate_v.'), 'text': ('text_trs.', 'gate_t.')}
def finite(value, label):
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(label)

def component_tensors(model, batch):
    """Observe the exact official calculate_loss call and restore method shadows."""
    original_bpr, original_reg = model.cal_bpr_loss, model.cal_reg_loss
    captured = {'bpr': [], 'regularizer_raw': []}
    prior = {name: model.__dict__.get(name) for name in ('cal_bpr_loss','cal_reg_loss')}
    present = {name: name in model.__dict__ for name in prior}
    def bpr(*args, **kwargs):
        value = original_bpr(*args, **kwargs)
        captured['bpr'].append(value)
        return value
    def regularizer(*args, **kwargs):
        value = original_reg(*args, **kwargs)
        captured['regularizer_raw'].append(value)
        return value
    model.cal_bpr_loss, model.cal_reg_loss = bpr, regularizer
    try:
        full = model.calculate_loss(batch)
    finally:
        for name in prior:
            if present[name]:
                setattr(model, name, prior[name])
            else:
                delattr(model, name)
    if len(captured['bpr']) != 1 or len(captured['regularizer_raw']) != 1:
        raise RuntimeError('unexpected official loss component calls')
    bpr_value, reg_raw = captured['bpr'][0], captured['regularizer_raw'][0]
    reg_weighted = model.reg_weight * reg_raw
    result = {'full': full, 'bpr': bpr_value, 'regularizer': reg_weighted,
              'regularizer_raw': reg_raw, 'contrastive_residual': full-bpr_value-reg_weighted}
    for name, value in result.items():
        finite(value, 'nonfinite component: '+name)
    return result

def losses(model, batch):
    with torch.no_grad():
        return {name: float(value.item()) for name,value in component_tensors(model,batch).items()}

def named_block(model, block):
    result = [(name,p) for name,p in model.named_parameters() if name.startswith(BLOCKS[block])]
    if not result:
        raise RuntimeError('empty block: '+block)
    return result

def l2(values):
    return math.sqrt(sum(float(value.detach().double().square().sum().item()) for value in values))

def cosine(left, right):
    denom = l2(left)*l2(right)
    if denom == 0:
        return None
    return sum(float((a.detach().double()*b.detach().double()).sum().item())
               for a,b in zip(left,right))/denom

def normalize(values):
    norm = l2(values)
    if not math.isfinite(norm) or norm == 0:
        raise RuntimeError('zero/nonfinite vector')
    for value in values:
        finite(value,'nonfinite direction')
    return [value/norm for value in values]

def private_gaussian(params, seed):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    values = [torch.randn(p.shape, generator=generator, device='cpu', dtype=p.dtype).to(p.device)
              for _,p in params]
    return normalize(values)

def clone_state(model):
    return {name:value.detach().clone() for name,value in model.state_dict().items()}

def exact_state(model, before):
    current=model.state_dict()
    return set(current)==set(before) and all(torch.equal(current[name],value) for name,value in before.items())

def set_delta(params, originals, deltas, sign):
    with torch.no_grad():
        for (_,param),original,delta in zip(params,originals,deltas):
            param.copy_(original + sign*delta)

def restore(params, originals):
    with torch.no_grad():
        for (_,param),original in zip(params,originals):
            param.copy_(original)

def measurement(zero, repeat, plus, minus, restored, delta_sq):
    result={}
    if not math.isfinite(delta_sq) or delta_sq<=0:
        raise RuntimeError('zero/nonfinite perturbation radius')
    for name in zero:
        vals=(zero[name],repeat[name],plus[name],minus[name],restored[name])
        if not all(math.isfinite(value) for value in vals):
            raise FloatingPointError('nonfinite scalar '+name)
        excess=.5*(plus[name]+minus[name])-zero[name]
        floor=max(1e-7,1e-5*max(1.,abs(zero[name]),abs(plus[name]),abs(minus[name])))
        result[name]={'L0':zero[name],'Lrepeat':repeat[name],'Lplus':plus[name],
                      'Lminus':minus[name],'Lrestored':restored[name],
                      'E':excess,'K':2.*excess/delta_sq,'floor_E':floor,
                      'floor_K':2.*floor/max(delta_sq,1e-30),
                      'repeat_jitter':abs(repeat[name]-zero[name]),
                      'restoration_jitter':abs(restored[name]-zero[name])}
    return result

def probe(model,batch,params,direction,before,zero,repeat,radius):
    originals=[p.detach().clone() for _,p in params]
    deltas=[value*radius for value in direction]
    delta_sq=l2(deltas)**2
    try:
        set_delta(params,originals,deltas,+1.)
        plus=losses(model,batch)
        restore(params,originals)
        if not exact_state(model,before):
            raise RuntimeError('state mismatch after plus restoration')
        set_delta(params,originals,deltas,-1.)
        minus=losses(model,batch)
    finally:
        restore(params,originals)
    if not exact_state(model,before):
        raise RuntimeError('state mismatch after minus restoration')
    restored=losses(model,batch)
    values=measurement(zero,repeat,plus,minus,restored,delta_sq)
    valid=all(q['repeat_jitter']<=q['floor_E'] and q['restoration_jitter']<=q['floor_E']
              for q in values.values())
    return values,delta_sq,valid

def direction_winner(image,text):
    for candidate,other,name in ((image,text,'image'),(text,image,'text')):
        passes=[]
        for component in ('bpr','full'):
            for statistic,floor in (('E','floor_E'),('K','floor_K')):
                a,b=candidate['components'][component],other['components'][component]
                passes.append(a[statistic]>a[floor] and
                              a[statistic]-b[statistic]>a[floor]+b[floor])
        if all(passes) and candidate['numerically_valid'] and other['numerically_valid']:
            return name
    return None

def dominance(records):
    decisions=[]
    for state in ('init','uniform_best'):
        for batch_id in range(3):
            for direction in ('full_grad','rand0','rand1','rand2'):
                selected=[r for r in records if r['state']==state and r['batch_id']==batch_id
                          and r['direction']==direction]
                if len(selected)!=2 or {r['block'] for r in selected}!={'image','text'}:
                    raise RuntimeError('missing required comparison')
                by_block={r['block']:r for r in selected}
                decisions.append({'state':state,'batch_id':batch_id,'direction':direction,
                                  'winner':direction_winner(by_block['image'],by_block['text'])})
    counts={}
    winners=[]
    for state in ('init','uniform_best'):
        counts[state]={}
        for block in ('image','text'):
            batch_counts=[sum(d['winner']==block for d in decisions
                              if d['state']==state and d['batch_id']==batch_id) for batch_id in range(3)]
            gradient_wins=sum(d['winner']==block for d in decisions
                              if d['state']==state and d['direction']=='full_grad')
            passed=sum(n>=3 for n in batch_counts)>=2 and gradient_wins>=2
            counts[state][block]={'direction_wins_per_batch':batch_counts,
                                  'full_gradient_wins':gradient_wins,'passed':passed}
            if passed:
                winners.append((state,block))
    valid=all(r['numerically_valid'] for r in records)
    matched=(len(winners)==2 and winners[0][1]==winners[1][1] and valid)
    return {'passed':matched,'dominant_block':winners[0][1] if matched else None,
            'all_probes_numerically_valid':valid,'counts':counts,'directions':decisions,
            'formal_training_started':False,'recommendation_gain_established':False}

def fixed_private_batches(users,positives,csr,batch_size=2048,n_batches=3):
    wanted=batch_size*n_batches
    chosen=np.random.RandomState(20261010).permutation(len(users))[:wanted]
    if chosen.size!=wanted:raise RuntimeError('insufficient TRAIN rows')
    selected_users=users[chosen];selected_pos=positives[chosen]
    universe=np.sort(np.unique(positives))
    rng=np.random.RandomState(20261011)
    negatives=np.empty(wanted,dtype=np.int64)
    for index,user in enumerate(selected_users):
        seen=csr.indices[csr.indptr[user]:csr.indptr[user+1]]
        unseen=universe[~np.isin(universe,seen)]
        if unseen.size==0:raise RuntimeError('no eligible TRAIN-universe negative')
        negatives[index]=unseen[rng.randint(unseen.size)]
    arrays=np.stack([np.stack((selected_users[i*batch_size:(i+1)*batch_size],
                              selected_pos[i*batch_size:(i+1)*batch_size],
                              negatives[i*batch_size:(i+1)*batch_size])) for i in range(n_batches)])
    if any(np.any(csr[a[0],a[2]].A1) or not np.all(csr[a[0],a[1]].A1>0) for a in arrays):
        raise RuntimeError('batch eligibility failed')
    return arrays,chosen,universe
'''

RUNTIME = r'''
import gc, hashlib, json, math, os, random, subprocess, sys, time, traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='1'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='4'
ROOT=Path(REQUEST['original']); RUN=Path(REQUEST['run_root'])
START=time.monotonic()
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):
            h.update(part)
    return h.hexdigest()
def write_json(name,value):
    path=RUN/name; tmp=path.with_name(path.name+'.pending')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(str(tmp),str(path))
def verify(paths):
    for path,expected in paths.items():
        if digest(path)!=expected:
            raise RuntimeError('protected SHA changed: '+path)
def resource_guard():
    available=sorted(os.sched_getaffinity(0))
    if len(available)<4:raise RuntimeError('fewer than four CPUs available')
    os.sched_setaffinity(0,set(available[:4]))
    found=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name)==os.getpid():
            continue
        try:
            if path.stat().st_uid!=os.getuid():
                continue
            cmd=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if any(token in cmd for token in ('/pipeline.py','/train.py','/run_queue.py','/audit.py')):
                found.append({'pid':int(path.name),'command':cmd.strip()})
        except (FileNotFoundError,PermissionError,ProcessLookupError):
            pass
    if found:
        raise RuntimeError('owned research process still active: '+json.dumps(found))
    gpu=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu',
                                 '--format=csv,noheader,nounits'],text=True,timeout=20).strip()
    parts=[p.strip() for p in gpu.split(',')]
    if len(parts)!=5 or parts[0]!='1' or int(parts[2])<14*1024:
        raise RuntimeError('GPU1 free-memory guard failed')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                  '--format=csv,noheader,nounits'],text=True,timeout=20)
    if any(line.startswith(parts[1]+',') for line in apps.splitlines()):
        raise RuntimeError('GPU1 occupied')
    mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    if mem['MemAvailable']<12*1024*1024:
        raise RuntimeError('host-memory guard failed')
    return {'gpu1':gpu,'owned_processes':found,'host_available_kib':mem['MemAvailable'],
            'cpu_affinity':sorted(os.sched_getaffinity(0))}
def state_sha(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def rng_sha():
    h=hashlib.sha256()
    h.update(repr(random.getstate()).encode())
    state=np.random.get_state();h.update(state[0].encode());h.update(state[1].tobytes());h.update(repr(state[2:]).encode())
    h.update(torch.get_rng_state().numpy().tobytes())
    for value in torch.cuda.get_rng_state_all():h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()
def graph_sha(model):
    result={}
    for name in ('norm_R','norm_adj','struct_original_adj','image_original_adj','text_original_adj'):
        value=getattr(model,name).detach()
        h=hashlib.sha256();h.update(str(value.shape).encode());h.update(str(value.layout).encode())
        if value.is_sparse:
            value=value.coalesce();h.update(value.indices().cpu().contiguous().numpy().tobytes())
            h.update(value.values().cpu().contiguous().numpy().tobytes())
        else:h.update(value.cpu().contiguous().numpy().tobytes())
        result[name]=h.hexdigest()
    return result
def deadline():
    if time.monotonic()-START>3600:
        raise TimeoutError('one-hour diagnostic budget exhausted')
def run():
    preflight=resource_guard()
    if RUN.exists():raise RuntimeError('new diagnostic directory already exists; inspect, never relaunch')
    RUN.mkdir();(RUN/'data').mkdir()
    write_json('deployment.json',REQUEST)
    paths=json.loads((ROOT/'runs/baby_uniform_s2026_formal/protected_paths.json').read_text())
    for name,row in REQUEST['bound_files'].items():paths[str(ROOT/name)]=row['sha256']
    verify(paths)
    write_json('manifest_pre.json',{'protected_paths':paths,'count':len(paths),'all_match':True})
    manifest_pre_sha=digest(RUN/'manifest_pre.json')
    import numpy as np_module
    import torch as torch_module
    globals()['np']=np_module;globals()['torch']=torch_module
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.55,device=0)
    os.chdir(ROOT/'carrier/src');sys.path[:0]=[str(ROOT),str(ROOT/'carrier/src')]
    from models.msca import MSCA
    from utils.configurator import Config
    from utils.dataloader import TrainDataLoader
    from utils.dataset import RecDataset
    from utils.utils import init_seed
    config_saved=json.loads((ROOT/'runs/baby_uniform_s2026_formal/config.json').read_text())
    config=Config('MSCA','baby',config_saved.copy(),mg=False)
    if (str(config['device'])!='cuda' or config['inter_splitting_label']!='x_label'
        or config['use_full_sampling'] or not config['use_neg_sampling'] or config['use_neighborhood_loss']):
        raise RuntimeError('unsupported frozen carrier configuration')
    cache_dir=ROOT/'carrier/data/baby'
    for name in ('image_adj_10_True.pt','text_adj_10_True.pt'):
        if not (cache_dir/name).is_file():raise RuntimeError('frozen graph cache missing')
    dataset=RecDataset(config)
    train_df=dataset.df.loc[dataset.df['x_label']==0].drop(columns=['x_label']).copy()
    train_data=dataset.copy(train_df);str(train_data)
    loader=TrainDataLoader(config,train_data,batch_size=2048,shuffle=True)
    csr=loader.inter_matrix(form='csr');csr.sort_indices()
    # Independent private arrays; the original training loader is never iterated.
    arrays,chosen,universe=fixed_private_batches(
        train_df[dataset.uid_field].to_numpy(dtype=np.int64),
        train_df[dataset.iid_field].to_numpy(dtype=np.int64),csr)
    own_positive_excluded=all(not np.any(csr[a[0],a[2]].A1) for a in arrays)
    positives_in_train=all(np.all(csr[a[0],a[1]].A1>0) for a in arrays)
    if not own_positive_excluded or not positives_in_train:raise RuntimeError('batch eligibility failed')
    np.savez_compressed(RUN/'data/batches.npz',batches=arrays)
    receipt={'path':str(RUN/'data/batches.npz'),'sha256':digest(RUN/'data/batches.npz'),
             'bytes':(RUN/'data/batches.npz').stat().st_size,'shape':list(arrays.shape),
             'batch_hashes':[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays],
             'train_rows':len(train_df),'positive_rows_distinct':len(np.unique(chosen)),
             'TRAIN_only':True,'own_train_positive_excluded':own_positive_excluded,
             'positive_in_train':positives_in_train,'negative_universe_size':len(universe),
             'positive_seed':20261010,'negative_seed':20261011}
    write_json('batch_receipt.json',receipt)  # before any model/loss is inspected
    init_seed(2026);loader.pretrain_setup()
    model=MSCA(config,loader).to(config['device']);model.eval()
    init_hash=state_sha(model)
    if init_hash!=REQUEST['initial_sha']:raise RuntimeError('initial state SHA differs: '+init_hash)
    initial=clone_state(model);graphs_before=graph_sha(model)
    checkpoint_path=ROOT/'runs/baby_uniform_s2026_formal/best.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu')
    baseline_receipt=json.loads((ROOT/'runs/baby_uniform_s2026_formal/result.json').read_text())
    if (baseline_receipt['status']!='completed' or baseline_receipt['initial_state_sha256']!=init_hash
        or checkpoint['initial_sha256']!=init_hash or checkpoint['epoch']!=44
        or baseline_receipt['best_epoch']!=44 or baseline_receipt['checkpoint_sha256']!=REQUEST['checkpoint_sha']):
        raise RuntimeError('completed uniform checkpoint certificate differs')
    write_json('environment.json',{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,
               'physical_gpu':1,'cpu_threads':4,'cuda_memory_fraction':.55,'preflight':preflight,
               'initial_state_sha256':init_hash,'source_commit':REQUEST['source_commit']})
    batches=[torch.from_numpy(a).to(config['device']) for a in arrays]
    records=[];states={};stage='probes';write_json('progress.json',{'stage':stage,'completed_probes':0})
    for state in ('init','uniform_best'):
        model.load_state_dict(initial if state=='init' else checkpoint['state_dict'],strict=True)
        before=clone_state(model);rng_before=rng_sha();state_before=state_sha(model)
        states[state]={'state_sha256':state_before,'selection':'fresh_initialization' if state=='init'
                       else 'fixed_validation_selected_uniform_best','checkpoint_epoch':None if state=='init' else 44,
                       'block_names':{b:[n for n,p in named_block(model,b)] for b in BLOCKS}}
        for batch_id,batch in enumerate(batches):
            deadline();zero=losses(model,batch);repeat=losses(model,batch)
            params_all=[p for b in ('image','text') for n,p in named_block(model,b)]
            values=component_tensors(model,batch)
            g_full=torch.autograd.grad(values['full'],params_all,retain_graph=True,allow_unused=False)
            g_bpr=torch.autograd.grad(values['bpr'],params_all,retain_graph=False,allow_unused=False)
            del values
            offset=0
            for block in ('image','text'):
                params=named_block(model,block);count=len(params)
                gf=list(g_full[offset:offset+count]);gb=list(g_bpr[offset:offset+count]);offset+=count
                theta_norm=l2([p for n,p in params]);radius=.01*theta_norm
                if not math.isfinite(theta_norm) or theta_norm==0:raise RuntimeError('zero/nonfinite block norm')
                dirs=[('full_grad',None,normalize(gf))]
                dirs += [('rand'+str(i),20261020+i,private_gaussian(params,20261020+i)) for i in range(3)]
                for direction,seed,unit in dirs:
                    deadline()
                    components,delta_sq,valid=probe(model,batch,params,unit,before,zero,repeat,radius)
                    row={'state':state,'batch_id':batch_id,'block':block,'direction':direction,
                         'seed':seed,'rho':.01,'theta_norm':theta_norm,'direction_norm':l2(unit),
                         'delta_norm':math.sqrt(delta_sq),'delta_norm_sq':delta_sq,
                         'full_gradient_norm':l2(gf),'bpr_gradient_norm':l2(gb),
                         'full_bpr_gradient_alignment':cosine(gf,gb),
                         'direction_full_gradient_alignment':cosine(unit,gf),
                         'direction_bpr_gradient_alignment':cosine(unit,gb),
                         'numerically_valid':valid,'exact_parameter_buffer_restoration':True,
                         'source_commit':REQUEST['source_commit'],'manifest_pre_sha256':manifest_pre_sha,
                         'checkpoint_sha256':REQUEST['checkpoint_sha'],
                         'components':components}
                    records.append(row)
                    write_json('progress.json',{'stage':stage,'state':state,'batch_id':batch_id,
                               'completed_probes':len(records),'elapsed_seconds':time.monotonic()-START})
            del g_full,g_bpr;gc.collect()
            if not exact_state(model,before) or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError('state or gradient slots were modified')
        if rng_sha()!=rng_before:raise RuntimeError('diagnostic consumed global RNG')
        if graph_sha(model)!=graphs_before:raise RuntimeError('plain graph tensors changed')
        states[state].update(exact_state_restored=True,global_rng_unchanged=True,graph_tensors_unchanged=True)
    verify(paths);decision=dominance(records)
    write_json('manifest_post.json',{'protected_paths':paths,'count':len(paths),'all_match':True})
    manifest_post_sha=digest(RUN/'manifest_post.json')
    if manifest_post_sha!=manifest_pre_sha:raise RuntimeError('pre/post manifest bytes differ')
    write_json('states.json',states)
    result={'schema_version':1,'status':'diagnostic_pass' if decision['passed'] else 'diagnostic_inconclusive',
            'dataset':'baby','source_commit':REQUEST['source_commit'],'source_sha256':REQUEST['source_sha256'],
            'protocol_sha256':REQUEST['protocol_sha256'],'checkpoint_sha256':REQUEST['checkpoint_sha'],
            'initial_state_sha256':init_hash,'checkpoint_validation_selected':True,
            'batch_receipt':receipt,'states':states,'records':records,'decision':decision,
            'protected_count':len(paths),'protected_paths_match':True,
            'manifest_pre_sha256':manifest_pre_sha,'manifest_post_sha256':manifest_post_sha,
            'process_pid':os.getpid(),'process_command':sys.argv,
            'optimizer_instantiated':False,'optimizer_steps':0,'validation_loader_constructed':False,
            'test_loader_constructed':False,'test_evaluated':False,'physical_gpu':1,'cpu_threads':4,
            'elapsed_seconds':time.monotonic()-START,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(0)}
    write_json('result.json',result)
    columns=['state','batch_id','block','direction','seed','component','rho','theta_norm',
             'direction_norm','delta_norm','delta_norm_sq','numerically_valid',
             'source_commit','manifest_pre_sha256','checkpoint_sha256',
             'L0','Lrepeat','Lplus','Lminus','Lrestored','E','K','floor_E','floor_K','repeat_jitter','restoration_jitter']
    with (RUN/'probes.tsv').open('w') as stream:
        stream.write('\t'.join(columns)+'\n')
        for row in records:
            for component,values in row['components'].items():
                flat={**row,**values,'component':component}
                stream.write('\t'.join(str(flat[key]) for key in columns)+'\n')
    print('CURVATURE_RESULT='+json.dumps(result,allow_nan=False),flush=True)
try:
    run()
except Exception as error:
    failure={'status':'technical_failure','error_type':type(error).__name__,'error':str(error),
             'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-START,
             'source_commit':REQUEST['source_commit'],'optimizer_steps':0}
    if RUN.is_dir():write_json('failure.json',failure)
    print('CURVATURE_RESULT='+json.dumps(failure),flush=True)
    raise
'''


def remote_source(request):
    source = 'REQUEST=' + repr(request) + '\n' + CORE + '\n' + RUNTIME
    compile(source, '<curvature-remote>', 'exec')
    return source


def git_revision():
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=HERE, text=True).strip()
    if dirty:
        raise RuntimeError('Commit this independent diagnostic before execution')
    return sha


def request():
    inventory = json.loads((HERE / 'carrier_inventory.json').read_text())
    names = ('carrier/src/models/msca.py','carrier_manifest.json','source_protection.json','deployment.json',
             'runs/baby_uniform_s2026_formal/config.json','runs/baby_uniform_s2026_formal/environment.json',
             'runs/baby_uniform_s2026_formal/result.json','runs/baby_uniform_s2026_formal/best.pt')
    result = {'original':ORIGINAL,'run_root':RUN_ROOT,'source_commit':git_revision(),
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol_sha256':hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),
              'checkpoint_sha':CHECKPOINT_SHA,'initial_sha':INITIAL_SHA,
              'bound_files':{name:{'sha256':inventory['files'][name]['sha256'],
                                    'bytes':inventory['files'][name]['bytes']} for name in names}}
    if result['bound_files'][names[-1]]['sha256'] != CHECKPOINT_SHA:
        raise RuntimeError('checkpoint inventory mismatch')
    return result


def invoke(source, timeout):
    sys.path.insert(0,str(HERE.parent/'20260905'))
    from remote_session import Session
    connection=Session('213')
    try:
        return connection.run_python(source,PYTHON,timeout)
    finally:
        connection.close()


def main():
    args=argparse.ArgumentParser()
    args.add_argument('action',choices=('compile','run','fetch'))
    opts=args.parse_args()
    if opts.action=='compile':
        remote_source({'original':ORIGINAL,'run_root':RUN_ROOT})
        compile(supervisor_source('print("fixture")'),'<supervisor>','exec')
        print('Outer and remote syntax PASS; no CUDA import')
        return
    if opts.action=='fetch':
        source='import json\nfrom pathlib import Path\np=Path('+repr(RUN_ROOT)+')\nprint("CURVATURE_RESULT="+json.dumps({name:(p/name).read_text() for name in ("result.json","states.json","probes.tsv","manifest_pre.json","manifest_post.json","batch_receipt.json","environment.json") if (p/name).is_file()}))'
        code,output=invoke(source,120)
        marker='CURVATURE_RESULT=';start=output.find(marker)
        if code or start<0:raise RuntimeError(output)
        values,_=json.JSONDecoder().raw_decode(output[start+len(marker):])
        for name,text in values.items():(HERE/name).write_text(text,encoding='utf-8')
        print('Read-only fetched '+str(len(values))+' public aggregate artifacts')
        return
    req=request();before=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # subprocess.run kills and waits for its direct child on TimeoutExpired.
    # The supervising process creates no queue or daemon and touches no other PID.
    payload=remote_source(req)
    supervisor=supervisor_source(payload)
    code,output=invoke(supervisor,3700)
    marker='CURVATURE_RESULT=';start=output.find(marker)
    if start<0:raise RuntimeError(output)
    value,_=json.JSONDecoder().raw_decode(output[start+len(marker):])
    value['local_source_sha256_post']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    value['local_source_unchanged']=before==value['local_source_sha256_post']
    if not value['local_source_unchanged']:
        value['status']='technical_failure';value['error']='local source changed during diagnostic'
        code=1
    destination='result.json' if code==0 else 'failure.json'
    (HERE/destination).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({key:value.get(key) for key in ('status','error','elapsed_seconds','protected_count','decision')},indent=2))
    if code:raise SystemExit(code)


def supervisor_source(payload):
    # File launch makes the actual child command legible and preserves exact
    # submitted payload bytes separately from the immutable carrier.
    directory=RUN_ROOT+'_source'
    return ('import json,subprocess,sys\nfrom pathlib import Path\n'
            'p=Path('+repr(directory)+')\n'
            'if p.exists():raise RuntimeError("curvature payload already deployed; inspect before retry")\n'
            'p.mkdir()\nscript=p/"audit_payload.py"\nscript.write_text('+repr(payload)+')\n'
            'try:\n result=subprocess.run(['+repr(PYTHON)+',"-u",str(script)],'
            'capture_output=True,text=True,timeout=3600)\n'
            ' print(result.stdout,flush=True)\n print(result.stderr,file=sys.stderr,flush=True)\n'
            ' raise SystemExit(result.returncode)\nexcept subprocess.TimeoutExpired:\n'
            ' print("CURVATURE_RESULT="+json.dumps({"status":"technical_failure",'
            '"error":"fixed one-hour child timeout; subprocess killed and waited","optimizer_steps":0}),flush=True)\n'
            ' raise SystemExit(124)\n')

if __name__=='__main__':
    main()
