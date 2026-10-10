"""Run the cache against the frozen Baby BRIDGE implementation and checkpoint."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/data/sunzixuan/target_isolated_20261008_autoresearch_v1').resolve()
CARRIER = ROOT / 'carrier'
CHECKPOINT = ROOT / 'runs/baby_baseline_s2026/best.pt'
AUDIT = Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.chdir(CARRIER / 'src')
sys.path[:0] = [str(CARRIER / 'src'), str(Path(__file__).resolve().parent)]
from models.bridge import BRIDGE
from models.dual_frequency_encoder import DualFrequencyEncoder
from utils.configurator import Config
from utils.dataloader import EvalDataLoader, TrainDataLoader
from utils.dataset import RecDataset
from eval_cache import cached_eval_pass
from checkpoint_structure import align_original_gpu_structure


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def state_digest(model):
    h = hashlib.sha256()
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def transient_state(model):
    return {name: getattr(model, name).detach().clone() for name in
            ('id_rep', 'id_preference', 'v_rep', 'v_preference', 't_rep', 't_preference')}


def rng_state():
    import random
    return (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())


def same_rng(left, right):
    return bool(torch.equal(left[0], right[0]) and left[1][0] == right[1][0]
                and np.array_equal(left[1][1], right[1][1]) and left[1][2:] == right[1][2:]
                and left[2] == right[2])


def original_masked_top50(scores, mask):
    scores = scores.clone()
    scores[mask[0], mask[1]] = -1e10
    return torch.topk(scores, k=50, dim=1).indices


def protected_inputs():
    baseline = ROOT / 'runs/baby_baseline_s2026'
    declared = json.loads((AUDIT / 'input_audit.json').read_text())
    anchors = declared['baseline_anchors_sha256']
    for path, expected in anchors.items():
        if digest(path) != expected:
            raise RuntimeError('Pre-registered baseline manifest changed: ' + path)
    deployment = json.loads((ROOT / 'deployment.json').read_text())
    if len(deployment['carrier_sha256']) != 35:
        raise RuntimeError('Expected exactly 35 frozen carrier files')
    protected = {str(CARRIER / 'src' / name): expected
                 for name, expected in deployment['carrier_sha256'].items()}
    protected.update(anchors)
    for name, expected in declared['frozen_source'].items():
        path = str(CARRIER / 'src/models' / name)
        if protected.get(path) != expected or digest(path) != expected:
            raise RuntimeError('Pre-registered original core source changed: ' + name)
    inputs = json.loads((baseline / 'inputs.json').read_text())
    protected.update(inputs['sha256'])
    result = json.loads((baseline / 'result.json').read_text())
    if result['model'] != 'BRIDGE' or result['test_evaluated'] or result['test_loader_constructed']:
        raise RuntimeError('Expected a completed validation-only original baseline')
    protected[str(CHECKPOINT)] = result['checkpoint_sha256']
    audit_deployment = json.loads((AUDIT / 'deployment.json').read_text())
    protected.update({str(AUDIT / name): expected
                      for name, expected in audit_deployment['sha256'].items()})
    protected[str(AUDIT / 'deployment.json')] = digest(AUDIT / 'deployment.json')
    for path, expected in protected.items():
        if digest(path) != expected:
            raise RuntimeError('Protected input changed: ' + path)
    absent = inputs.get('absent', [])
    if any(Path(path).exists() for path in absent):
        raise RuntimeError('Previously absent baseline input appeared')
    return protected, absent


def main():
    started = time.monotonic()
    memory = {parts[0].rstrip(':'): int(parts[1]) * 1024
              for parts in (line.split() for line in Path('/proc/meminfo').read_text().splitlines())
              if len(parts) > 1 and parts[1].isdigit()}
    if memory['MemAvailable'] < 12 * 1024 ** 3:
        raise RuntimeError('CPU audit requires at least 12 GiB available RAM')
    protected, absent = protected_inputs()
    torch.set_num_threads(4)
    packet = torch.load(CHECKPOINT, map_location='cpu')
    checkpoint_config = json.loads(json.dumps(packet['config'], default=str))
    frozen_config = json.loads((CHECKPOINT.parent / 'config.json').read_text())['effective_config']
    if checkpoint_config != frozen_config:
        raise RuntimeError('Checkpoint configuration differs from frozen completed baseline')
    runtime_keys = ('use_gpu', 'device', 'eval_batch_size', 'save_recommended_topk')
    runtime_config = dict(packet['config'])
    runtime_config.update(use_gpu=False, device=torch.device('cpu'), eval_batch_size=64,
                          save_recommended_topk=False)
    config = Config('BRIDGE', 'baby', runtime_config)
    effective = json.loads(json.dumps(config.final_config_dict, default=str))
    if {k:v for k,v in effective.items() if k not in runtime_keys} != {
            k:v for k,v in frozen_config.items() if k not in runtime_keys}:
        raise RuntimeError('Algorithm configuration changed during CPU reconstruction')
    dataset = RecDataset(config)
    train_df = dataset.df.loc[dataset.df[config['inter_splitting_label']] == 0].drop(
        columns=[config['inter_splitting_label']]).copy()
    valid_df = dataset.df.loc[dataset.df[config['inter_splitting_label']] == 1].drop(
        columns=[config['inter_splitting_label']]).copy()
    valid_df = valid_df.loc[valid_df[dataset.uid_field].isin(train_df[dataset.uid_field])].copy()
    train_dataset, valid_dataset = dataset.copy(train_df), dataset.copy(valid_df)
    str(train_dataset); str(valid_dataset)
    train = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    valid = EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=64)
    train.pretrain_setup()
    model = BRIDGE(config, train).eval()
    structure = align_original_gpu_structure(model, packet['model'],
                  (CARRIER / 'src/models/dual_frequency_encoder.py').read_text())
    del packet
    direct = model.full_sort_predict
    original_code = BRIDGE.full_sort_predict.__code__
    original_globals_encoder = BRIDGE.full_sort_predict.__globals__['DualFrequencyEncoder']
    # Materialize these fixed interactions before RNG snapshots. No target
    # metric calculation, test loader, or train-loader iteration occurs.
    batches = []
    for idx, interaction in enumerate(valid):
        if idx >= 4: break
        batches.append(interaction)
    if len(batches) != 4:
        raise RuntimeError('Expected exactly four frozen validation batches')
    before_state = state_digest(model)
    before_rng = rng_state()
    direct_scores = []
    direct_started = time.monotonic()
    for interaction in batches:
        with torch.no_grad():
            direct_scores.append(direct(interaction).detach().clone())
    direct_seconds = time.monotonic() - direct_started
    direct_transient = transient_state(model)
    after_direct_state = state_digest(model)
    after_direct_rng = rng_state()
    if before_state != after_direct_state or not same_rng(before_rng, after_direct_rng):
        raise RuntimeError('direct forward changed model state or CPU RNG')

    cached_scores = []
    cache_receipt = None
    cached_started = time.monotonic()
    with torch.no_grad(), cached_eval_pass(model, direct, DualFrequencyEncoder) as (cached, cache):
        for interaction in batches:
            cached_scores.append(cached(interaction).detach().clone())
        cache_receipt = cache.receipt()
    cached_seconds = time.monotonic() - cached_started
    cached_transient = transient_state(model)
    after_cached_state = state_digest(model)
    after_cached_rng = rng_state()
    transient_equal = all(torch.equal(direct_transient[n], cached_transient[n])
                          for n in direct_transient)
    errors = [float((a - b).abs().max()) for a, b in zip(direct_scores, cached_scores)]
    topk_equal = []
    for interaction, direct_score, cached_score in zip(batches, direct_scores, cached_scores):
        mask = interaction[1]
        topk_equal.append(bool(torch.equal(original_masked_top50(direct_score, mask),
                                           original_masked_top50(cached_score, mask))))
    post = {path: digest(path) for path in protected}
    guards_passed = post == protected and not any(Path(path).exists() for path in absent)
    result = {
        'status': 'completed', 'device': 'cpu', 'torch_version': torch.__version__,
        'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': digest(CHECKPOINT),
        'checkpoint_structure': structure,
        'batches': len(batches), 'rows': sum(x.shape[0] for x in direct_scores),
        'max_abs_error': max(errors), 'per_batch_max_abs_error': errors,
        'masked_top50_exact': all(topk_equal), 'per_batch_top50_exact': topk_equal,
        'state_before_sha256': before_state, 'state_after_direct_sha256': after_direct_state,
        'state_after_cached_sha256': after_cached_state,
        'rng_unchanged_direct': same_rng(before_rng, after_direct_rng),
        'rng_unchanged_cached': same_rng(before_rng, after_cached_rng),
        'transient_forward_attributes_equal': transient_equal,
        'cache_cleared_on_exit': cache._outputs is None,
        'original_code_and_globals_unchanged': BRIDGE.full_sort_predict.__code__ is original_code and
            BRIDGE.full_sort_predict.__globals__['DualFrequencyEncoder'] is original_globals_encoder,
        'protected_inputs_pre_sha256': protected, 'protected_inputs_post_sha256': post,
        'protected_inputs_unchanged': guards_passed, 'absent_inputs': absent,
        'frozen_carrier_files': 35, 'cuda_initialized': torch.cuda.is_initialized(),
        'ram_available_before_bytes': memory['MemAvailable'],
        'configuration_equal_except_declared_runtime_keys': True,
        'runtime_override_keys': list(runtime_keys),
        'algorithm_config_sha256': hashlib.sha256(json.dumps({k:v for k,v in frozen_config.items()
            if k not in runtime_keys}, sort_keys=True).encode()).hexdigest(),
        'direct_seconds': direct_seconds, 'cached_seconds': cached_seconds,
        'elapsed_seconds': time.monotonic() - started,
        'cache': cache_receipt, 'test_evaluated': False,
        'source_hashes': {
            'bridge.py': digest(CARRIER / 'src/models/bridge.py'),
            'dual_frequency_encoder.py': digest(CARRIER / 'src/models/dual_frequency_encoder.py'),
        },
    }
    equivalent = (result['max_abs_error'] <= 1e-6 and result['masked_top50_exact']
        and guards_passed and before_state == after_cached_state and transient_equal
        and same_rng(before_rng, after_cached_rng) and result['cache_cleared_on_exit']
        and result['original_code_and_globals_unchanged'] and not result['cuda_initialized']
        and cache_receipt['forward_generations'] == 1 and cache_receipt['cache_hits'] == 3)
    if not equivalent:
        result['status'] = 'non_equivalent'
    (AUDIT / 'receipt.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if not k.startswith('protected_inputs_')}, allow_nan=False))
    if not equivalent:
        raise RuntimeError('CPU original BRIDGE cache non-equivalence')


if __name__ == '__main__':
    main()
