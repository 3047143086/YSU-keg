"""Deferred, read-only GPU Electronics cache equivalence audit.

This file is preparation only. It must be copied to a fresh remote audit
target and reviewed before invocation. It never trains, writes the baseline,
or evaluates test data.
"""
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/data/sunzixuan/electronics_baseline_20261009_v1').resolve()
CARRIER = ROOT / 'carrier'
RUN = ROOT / 'runs/elec_baseline_s2026'
CHECKPOINT = RUN / 'best.pt'
AUDIT = Path(__file__).resolve().parent
SELECTED_BATCHES = (0, 1, 1000, 1503, 2000, 3006)
ANCHORS = {
    str(ROOT / 'deployment.json'): '58658242005fb07a58f7b677732bdea65ea0aab1feeed191d5310ad939f421d8',
    str(RUN / 'config.json'): 'a8d27ee50f98cb6b8c89acf0a59c9d42846ffbcfbd85f34b8d864f8b95dcb875',
    str(RUN / 'inputs.json'): '18dd014949737077a3ce63e004e2bddd82d1870f6caf430f43732b35db3d9b2e',
    str(CHECKPOINT): '3f9283407ff51178549c2f78e0f8cf61f895c269164afa6b3b9a0a56d8faacda',
    str(RUN / 'failure.json'): '6bd82cce4407ffe47502de59d886799e2ecffba31d771569d8604be0c5e63a1f',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def protected_inputs():
    for path, expected in ANCHORS.items():
        if digest(path) != expected:
            raise RuntimeError('Electronics baseline anchor changed: ' + path)
    deployment = json.loads((ROOT / 'deployment.json').read_text())
    if len(deployment['carrier_sha256']) != 35:
        raise RuntimeError('Electronics carrier is not the 35-file frozen snapshot')
    for name, expected in deployment['carrier_sha256'].items():
        path = CARRIER / 'src' / name
        if digest(path) != expected:
            raise RuntimeError('Frozen Electronics carrier changed: ' + name)
    graph = json.loads((ROOT / 'pipeline_status.json').read_text())['graph_manifest']
    if graph['graph_sha256'] != '76ccb18835b2d18b2f56e882cf4670b2d7af733951090656ab4be23ef70a9ed2':
        raise RuntimeError('Electronics graph manifest changed')
    if graph['source_graph_source_hash'] != '287c1627a289707685028d1852105efa69b8da5106cb175c2871cacd7904e90c':
        raise RuntimeError('Electronics graph source lineage changed')
    protected = {str(CARRIER / 'src' / n): h for n, h in deployment['carrier_sha256'].items()}
    protected.update(ANCHORS)
    protected.update(json.loads((RUN / 'inputs.json').read_text())['sha256'])
    protected[str(ROOT / 'pipeline_status.json')] = digest(ROOT / 'pipeline_status.json')
    protected.update({str(AUDIT / name): expected for name, expected in
                      json.loads((AUDIT / 'deployment.json').read_text())['sha256'].items()})
    protected[str(AUDIT / 'deployment.json')] = digest(AUDIT / 'deployment.json')
    for path, expected in protected.items():
        if digest(path) != expected:
            raise RuntimeError('Protected GPU audit input changed: ' + path)
    absent = json.loads((RUN / 'inputs.json').read_text()).get('absent', [])
    if any(Path(path).exists() for path in absent):
        raise RuntimeError('Previously absent input appeared')
    return protected, absent


def rng_state():
    cuda = tuple(s.clone() for s in torch.cuda.get_rng_state_all())
    return torch.get_rng_state().clone(), cuda, np.random.get_state(), random.getstate()


def same_rng(a, b):
    return (torch.equal(a[0], b[0]) and len(a[1]) == len(b[1]) and
            all(torch.equal(x, y) for x, y in zip(a[1], b[1])) and
            a[2][0] == b[2][0] and np.array_equal(a[2][1], b[2][1]) and
            a[2][2:] == b[2][2:] and a[3] == b[3])


def state_digest(model):
    h = hashlib.sha256()
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def transient_state(model):
    result = {}
    for name in ('id_rep', 'id_preference', 'v_rep', 'v_preference', 't_rep', 't_preference'):
        value = getattr(model, name, None)
        result[name] = None if value is None else value.detach().clone()
    return result


def original_masked_top50(scores, mask):
    scores = scores.clone()
    scores[mask[0], mask[1]] = -1e10
    return torch.topk(scores, k=50, dim=1).indices.cpu()


def main():
    started = time.monotonic()
    def timeout(signum, frame):
        raise TimeoutError('Electronics GPU audit exceeded fixed one-hour budget')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(3600)
    launch = json.loads((ROOT / 'launch.json').read_text())
    status = json.loads((ROOT / 'pipeline_status.json').read_text())
    if Path('/proc/' + str(launch['pid'])).exists() or status['status'] != 'technical_failure':
        raise RuntimeError('Electronics predecessor must be exited in technical_failure state')
    if Path('/proc/2936814').exists():
        raise RuntimeError('Electronics child must be exited before audit')
    gpu_query = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    free = {int(row.split(',')[0]): int(row.split(',')[1].strip()) * 1024**2
            for row in gpu_query.strip().splitlines()}
    if free.get(1, 0) < 14 * 1024**3:
        raise RuntimeError('GPU 1 lacks 14 GiB free memory for audit')
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
        '--format=csv,noheader,nounits'], text=True)
    uuid_rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid',
        '--format=csv,noheader,nounits'], text=True)
    uuid = {int(row.split(',')[0]): row.split(',')[1].strip() for row in uuid_rows.strip().splitlines()}
    if any(row.strip().endswith(uuid[1]) for row in processes.strip().splitlines()):
        raise RuntimeError('GPU 1 has an active compute process; exclusive audit refused')
    memory = {p[0].rstrip(':'):int(p[1])*1024 for p in
              (line.split() for line in Path('/proc/meminfo').read_text().splitlines())
              if len(p)>1 and p[1].isdigit()}
    if memory['MemAvailable'] < 24 * 1024**3:
        raise RuntimeError('GPU audit requires 24 GiB available RAM')
    protected, absent = protected_inputs()
    if not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('GPU audit requires CUDA_VISIBLE_DEVICES=1 and CUDA')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('GPU audit must expose exactly one logical GPU')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.55, device=0)
    os.chdir(CARRIER / 'src')
    sys.path[:0] = [str(CARRIER / 'src'), str(AUDIT)]
    from models.bridge import BRIDGE
    from models.dual_frequency_encoder import DualFrequencyEncoder
    from utils.configurator import Config
    from utils.dataloader import EvalDataLoader, TrainDataLoader
    from utils.dataset import RecDataset
    from eval_cache import cached_eval_pass
    from checkpoint_structure import align_original_gpu_structure

    packet = torch.load(CHECKPOINT, map_location='cpu')
    frozen_config = json.loads((RUN / 'config.json').read_text())['effective_config']
    checkpoint_config = json.loads(json.dumps(packet['config'], default=str))
    if checkpoint_config != frozen_config:
        raise RuntimeError('Electronics checkpoint config differs from archived config')
    runtime = dict(packet['config'])
    runtime.update(use_gpu=True, gpu_id='1', device=torch.device('cuda'), eval_batch_size=64,
                   save_recommended_topk=False)
    config = Config('BRIDGE', 'elec', runtime)
    runtime_keys = ('use_gpu', 'gpu_id', 'device', 'eval_batch_size', 'save_recommended_topk')
    effective = json.loads(json.dumps(config.final_config_dict, default=str))
    if {k: v for k, v in effective.items() if k not in runtime_keys} != {
            k: v for k, v in frozen_config.items() if k not in runtime_keys}:
        raise RuntimeError('Electronics algorithm configuration changed')
    data = RecDataset(config)
    train_df = data.df.loc[data.df[config['inter_splitting_label']] == 0].drop(
        columns=[config['inter_splitting_label']]).copy()
    valid_df = data.df.loc[data.df[config['inter_splitting_label']] == 1].drop(
        columns=[config['inter_splitting_label']]).copy()
    valid_df = valid_df.loc[valid_df[data.uid_field].isin(train_df[data.uid_field])].copy()
    train_dataset, valid_dataset = data.copy(train_df), data.copy(valid_df)
    str(train_dataset); str(valid_dataset)
    train = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    valid = EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=64)
    train.pretrain_setup()
    model = BRIDGE(config, train).to(config['device']).eval()
    structure = align_original_gpu_structure(
        model, packet['model'], (CARRIER / 'src/models/dual_frequency_encoder.py').read_text())
    del packet
    direct = model.full_sort_predict
    batches = []
    for index, interaction in enumerate(valid):
        if index in SELECTED_BATCHES:
            batches.append((index, interaction))
        if index > SELECTED_BATCHES[-1]:
            break
    if [index for index, _ in batches] != list(SELECTED_BATCHES):
        raise RuntimeError('Selected Electronics validation batches are unavailable')
    before_state, before_rng = state_digest(model), rng_state()
    torch.cuda.reset_peak_memory_stats()
    direct_scores, direct_topk = [], []
    for _, interaction in batches:
        with torch.no_grad():
            scores = direct(interaction)
            direct_scores.append(scores.detach().cpu())
            direct_topk.append(original_masked_top50(scores, interaction[1]))
    direct_transient = transient_state(model)
    after_direct_state, after_direct_rng = state_digest(model), rng_state()
    cached_scores, cached_topk = [], []
    with torch.no_grad(), cached_eval_pass(model, direct, DualFrequencyEncoder) as (cached, cache):
        for _, interaction in batches:
            scores = cached(interaction)
            cached_scores.append(scores.detach().cpu())
            cached_topk.append(original_masked_top50(scores, interaction[1]))
        cache_receipt = cache.receipt()
    cached_transient = transient_state(model)
    after_cached_state, after_cached_rng = state_digest(model), rng_state()
    errors = [float((a - b).abs().max()) for a, b in zip(direct_scores, cached_scores)]
    topk_equal = [bool(torch.equal(a, b)) for a, b in zip(direct_topk, cached_topk)]
    transient_equal = all((a is None and b is None) or (a is not None and torch.equal(a, b))
                          for a, b in zip(direct_transient.values(), cached_transient.values()))
    post = {path: digest(path) for path in protected}
    result = {
        'status': 'completed', 'dataset': 'elec', 'checkpoint_partial_training': True,
        'test_loader_constructed': False, 'test_evaluated': False, 'batches': list(SELECTED_BATCHES),
        'rows': sum(x.shape[0] for x in direct_scores), 'max_abs_error': max(errors),
        'per_batch_max_abs_error': errors, 'masked_top50_exact': all(topk_equal),
        'per_batch_top50_exact': topk_equal, 'state_before_sha256': before_state,
        'state_after_direct_sha256': after_direct_state, 'state_after_cached_sha256': after_cached_state,
        'rng_unchanged_direct': same_rng(before_rng, after_direct_rng),
        'rng_unchanged_cached': same_rng(before_rng, after_cached_rng),
        'transient_forward_attributes_equal': transient_equal, 'cache_cleared_on_exit': cache._outputs is None,
        'original_code_and_globals_unchanged': BRIDGE.full_sort_predict.__code__ is direct.__func__.__code__ and
            BRIDGE.full_sort_predict.__globals__['DualFrequencyEncoder'] is DualFrequencyEncoder,
        'protected_inputs_unchanged': post == protected and not any(Path(path).exists() for path in absent),
        'protected_entries': len(protected), 'protected_inputs_pre_sha256': protected,
        'protected_inputs_post_sha256': post, 'absent_inputs': absent,
        'checkpoint_structure': structure, 'cache': cache_receipt,
        'gpu_name': torch.cuda.get_device_name(0), 'torch_version': torch.__version__,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'cuda_device_count': torch.cuda.device_count(),
        'cpu_threads': torch.get_num_threads(), 'time_budget_seconds': 3600,
        'elapsed_seconds': time.monotonic()-started,
    }
    (AUDIT / 'electronics_gpu_receipt.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    if not (result['max_abs_error'] <= 1e-6 and result['masked_top50_exact'] and
            result['state_before_sha256'] == result['state_after_cached_sha256'] and
            result['rng_unchanged_direct'] and result['rng_unchanged_cached'] and
            result['transient_forward_attributes_equal'] and result['cache_cleared_on_exit'] and
            result['protected_inputs_unchanged'] and result['original_code_and_globals_unchanged'] and
            result['state_before_sha256'] == result['state_after_direct_sha256'] and
            cache_receipt['forward_generations'] == 1 and
            cache_receipt['cache_hits'] == len(SELECTED_BATCHES) - 1):
        result['status'] = 'non_equivalent'
        (AUDIT / 'electronics_gpu_receipt.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        raise RuntimeError('Electronics GPU cache non-equivalence')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
