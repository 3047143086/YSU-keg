"""Two fixed GPU probes, with no optimizer construction or parameter update."""
import gc
import importlib.util
import json
import os
import random
import resource
import signal
import sys
import time
import traceback
from pathlib import Path
from guards import BASELINE, CPU_AUDIT, RUN, idle_gpu_resources, verify
from audit_gate import archived_candidate_passed, model_candidate_passed

ROOT = Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '4'
SECONDS = 30 * 60


def atomic_json(path, value):
    pending = path.with_name(path.name + '.pending.' + str(os.getpid()))
    with pending.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def load_source(name, path):
    specification = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def tensor_hash(value):
    import hashlib
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def compare(new, old, relative_limit=5e-5, absolute_scale=5e-4):
    import torch
    if new.shape != old.shape:
        return {'passed': False, 'shape_matches': False}
    new, old = new.detach().cpu(), old.detach().cpu()
    difference = new.double() - old.double()
    error = float(torch.linalg.vector_norm(difference).item())
    norm = float(torch.linalg.vector_norm(old.double()).item())
    maximum = float(difference.abs().max().item())
    relative = error / norm if norm else (0.0 if error == 0 else None)
    limit = absolute_scale * max(1.0, float(old.abs().max().item()))
    return {'exact': torch.equal(new, old), 'relative_frobenius_error': relative,
            'maximum_absolute_error': maximum, 'maximum_absolute_limit': limit,
            'passed': relative is not None and relative <= relative_limit and maximum <= limit}


def state_hashes(model):
    return {name: tensor_hash(value) for name, value in model.state_dict().items()}


def restore_rng(packet):
    import numpy as np
    import torch
    random.setstate(packet['python'])
    np.random.set_state(packet['numpy'])
    torch.set_rng_state(packet['torch'])
    torch.cuda.set_rng_state_all(packet['cuda'])


def phase_archived(captures, projector):
    import torch
    rows = []
    for index, capture in enumerate(captures):
        started = time.monotonic()
        value = capture['input']
        row = {'capture_index': index, 'input_shape': list(value.shape), 'input_dtype': str(value.dtype),
               'original_tensor_sha256': tensor_hash(value), 'probe_seed': 2026,
               'probe_type': 'independent_dL_dBand', 'reconstructs_training_gradient': False}
        observed, original_svd = [], torch.linalg.svd
        def observe(*arguments, **keywords):
            result = original_svd(*arguments, **keywords)
            observed.append(result[1].detach().cpu().clone())
            return result
        try:
            if value.dtype != torch.float32:
                raise RuntimeError('Archived float32 input required without promotion')
            original = value.to('cuda:0').requires_grad_()
            candidate = value.to('cuda:0').clone().requires_grad_()
            torch.linalg.svd = observe
            try:
                old = projector.original_svd_bands(original, 3)
                new = projector.projector_bands(candidate, 3)
            finally:
                torch.linalg.svd = original_svd
            row.update(archived_cuda_singular_values=capture['singular'].tolist(),
                       current_original_cuda_singular_values=observed[0].tolist(),
                       current_candidate_cuda_singular_values=observed[1].tolist(),
                       archived_singular_values_exact=torch.equal(observed[0], capture['singular']),
                       exact_forward_bands=[torch.equal(left, right) for left, right in zip(old, new)],
                       boundary_gaps=projector.check_boundary_gaps(observed[1], (22, 21, 21)))
            generator = torch.Generator(device='cpu').manual_seed(2026)
            gradients = [torch.randn(output.shape, dtype=torch.float32, generator=generator).to('cuda:0')
                         for output in old]
            sum((output * weight).sum() for output, weight in zip(old, gradients)).backward()
            sum((output * weight).sum() for output, weight in zip(new, gradients)).backward()
            generic_finite = bool(torch.isfinite(original.grad).all().item())
            candidate_finite = bool(torch.isfinite(candidate.grad).all().item())
            row.update(generic_gradient_finite=generic_finite, candidate_gradient_finite=candidate_finite,
                       generic_nonfinite_elements=int((~torch.isfinite(original.grad)).sum().item()))
            row['generic_comparison'] = (compare(candidate.grad, original.grad, 1e-3, 1e-3)
                                         if generic_finite and candidate_finite else
                                         {'passed': False, 'status': 'unavailable_nonfinite_reference',
                                          'equivalence_claimed': False})
            row['candidate_passed'] = archived_candidate_passed(row)
        except Exception as error:
            if 'out of memory' in str(error).lower():
                raise
            row.update(candidate_passed=False, error=repr(error), traceback=traceback.format_exc())
            if observed:
                row['current_original_cuda_singular_values'] = observed[0].tolist()
        finally:
            torch.linalg.svd = original_svd
            row['elapsed_seconds'] = time.monotonic() - started
        rows.append(row)
        atomic_json(ROOT / 'archived_input_progress.json', {'rows': rows})
        print(json.dumps({'phase': 'archived_inputs', 'capture': index, 'passed': row['candidate_passed']}), flush=True)
        # The next capture never keeps the prior CUDA graph or upstream probe.
        for name in ('original', 'candidate', 'old', 'new', 'gradients'):
            if name in locals():
                del locals()[name]
        gc.collect()
        torch.cuda.empty_cache()
    return {'all_twelve_inspected': len(rows) == 12, 'rows': rows,
            'candidate_passed': len(rows) == 12 and all(row['candidate_passed'] for row in rows)}


def construct_model(packet):
    import torch
    source = BASELINE / 'carrier/src'
    os.chdir(source)
    sys.path.insert(0, str(source))
    frozen = load_source('_frozen_electronics_train', BASELINE / 'train.py')
    from models.bridge import BRIDGE
    from utils.configurator import Config
    from utils.dataloader import EvalDataLoader, TrainDataLoader
    from utils.dataset import RecDataset
    from utils.utils import init_seed
    config = Config('BRIDGE', 'elec', {'use_gpu': True, 'gpu_id': '1', 'cuda_memory_fraction': .55,
                    'seed': [2026], 'epochs': 300, 'stopping_step': 20, 'eval_step': 1,
                    'eval_batch_size': 64, 'save_recommended_topk': False, 'req_training': True,
                    'valid_metric': 'Recall@20'})
    frozen.scalarize(config)
    expected = json.loads((RUN / 'config.json').read_text())['effective_config']
    if json.loads(json.dumps(config.final_config_dict, default=str)) != expected:
        raise RuntimeError('Reconstructed effective configuration differs from original run')
    init_seed(2026)
    dataset = RecDataset(config)
    train_dataset, valid_dataset = frozen.split_train_valid(dataset, config)
    train = TrainDataLoader(config, train_dataset, batch_size=2048, shuffle=True)
    valid = EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=64)
    train.pretrain_setup()
    model = BRIDGE(config, train).to('cuda:0')
    frozen.assert_baseline(model)
    aliases = {}
    for prefix in ('id', 'v', 't'):
        alias, original = prefix + '_preference', prefix + '_gcn.preference'
        if alias in packet['model']:
            if original not in packet['model'] or not torch.equal(packet['model'][alias], packet['model'][original]):
                raise RuntimeError('Saved preference alias differs from its GCN parameter')
            setattr(model, alias, getattr(model, prefix + '_gcn').preference)
            aliases[alias] = original
    model.load_state_dict(packet['model'], strict=True)
    model.train()
    del valid
    return model, {'config_matches': True, 'strict_load': True, 'preference_aliases': aliases,
                   'seed': 2026, 'dataset': 'elec', 'test_loader_constructed': False,
                   'optimizer_constructed': False, 'train_interactions': len(train_dataset)}


def model_run(model, packet, projector, candidate):
    import torch
    from models.dual_frequency_encoder import FrequencyDecompositionModule
    original_method = FrequencyDecompositionModule.frequency_decompose_svd
    calls = []
    def observed(module, value):
        outputs = (projector.projector_bands(value, module.M) if candidate
                   else original_method(module, value))
        calls.append({'input': value.detach().cpu().clone(),
                      'outputs': [output.detach().cpu().clone() for output in outputs]})
        return outputs
    FrequencyDecompositionModule.frequency_decompose_svd = observed
    try:
        model.zero_grad(set_to_none=True)
        restore_rng(packet['rng'])
        interaction = packet['interaction']
        interaction = (interaction.to('cuda:0') if torch.is_tensor(interaction)
                       else [value.to('cuda:0') for value in interaction])
        losses = model.calculate_loss(interaction)
        components = losses if isinstance(losses, tuple) else (losses,)
        if not all(torch.isfinite(value).all() for value in components):
            raise FloatingPointError('Nonfinite loss before any audit backward')
        loss_values = [float(value.detach().cpu().item()) for value in components]
        sum(components).backward()
        gradients = {name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters()
                     if parameter.grad is not None}
        finite = {name: bool(torch.isfinite(gradient).all().item()) for name, gradient in gradients.items()}
        return {'calls': calls, 'loss_values': loss_values, 'gradients': gradients,
                'gradient_finite': finite, 'all_gradients_finite': all(finite.values()),
                'nonfinite_gradient_parameters': [name for name, okay in finite.items() if not okay]}
    finally:
        FrequencyDecompositionModule.frequency_decompose_svd = original_method


def phase_model(packet, projector):
    import torch
    model, construction = construct_model(packet)
    before = state_hashes(model)
    original = model_run(model, packet, projector, False)
    model.zero_grad(set_to_none=True)
    for name in ('id_rep', 'v_rep', 't_rep', 'result_embed'):
        value = getattr(model, name, None)
        if torch.is_tensor(value) and not isinstance(value, torch.nn.Parameter):
            setattr(model, name, None)
    gc.collect()
    torch.cuda.empty_cache()
    candidate = model_run(model, packet, projector, True)
    after = state_hashes(model)
    rows = []
    if len(original['calls']) != 12 or len(candidate['calls']) != 12:
        raise RuntimeError('Full original and candidate model must each produce twelve calls')
    for index, (old, new, archived) in enumerate(zip(original['calls'], candidate['calls'], packet['svd_captures'])):
        rows.append({'capture_index': index,
                     'original_input_sha256': tensor_hash(old['input']),
                     'candidate_input_sha256': tensor_hash(new['input']),
                     'archived_input_sha256': tensor_hash(archived['input']),
                     'original_input_matches_archive': torch.equal(old['input'], archived['input']),
                     'candidate_input_matches_archive': torch.equal(new['input'], archived['input']),
                     'candidate_vs_original_input': compare(new['input'], old['input']),
                     'candidate_vs_original_bands': [compare(right, left)
                                                     for left, right in zip(old['outputs'], new['outputs'])]})
    loss_match = len(original['loss_values']) == len(candidate['loss_values']) and all(
        abs(new - old) <= 5e-6 + 5e-5 * abs(old)
        for old, new in zip(original['loss_values'], candidate['loss_values']))
    gradients = {}
    participating_match = set(original['gradients']) == set(candidate['gradients'])
    if original['all_gradients_finite'] and candidate['all_gradients_finite']:
        if not participating_match:
            raise RuntimeError('Participating parameter sets differ between finite-gradient probes')
        gradients = {name: compare(candidate['gradients'][name], original['gradients'][name], 1e-3, 1e-3)
                     for name in original['gradients']}
    archive_exact = all(row['original_input_matches_archive'] for row in rows)
    state_unchanged = before == after
    result = {'construction': construction, 'rows': rows, 'state_unchanged': state_unchanged,
              'state_sha256_before': before, 'state_sha256_after': after,
              'original_loss_values': original['loss_values'], 'candidate_loss_values': candidate['loss_values'],
              'loss_matches': loss_match, 'candidate_all_gradients_finite': candidate['all_gradients_finite'],
              'candidate_gradient_finite': candidate['gradient_finite'],
              'participating_parameters_match': participating_match,
              'participating_parameter_count': len(candidate['gradients']),
              'original_all_gradients_finite': original['all_gradients_finite'],
              'original_nonfinite_parameters': original['nonfinite_gradient_parameters'],
              'exact_archive_inputs_reproduced': archive_exact,
              'exact_historical_crash_reproduced': archive_exact and not original['all_gradients_finite'],
              'finite_original_gradient_comparisons': gradients,
              'generic_gradient_equivalence_available': original['all_gradients_finite'],
              'generic_gradient_equivalence_passed': bool(gradients) and all(row['passed'] for row in gradients.values())}
    result['candidate_integration_passed'] = model_candidate_passed(result)
    return result


def main():
    started = time.time()
    configuration = json.loads((ROOT / 'audit_config.json').read_text())
    receipt = {'status': 'running', 'source_revision': configuration['source_revision'],
               'pid': os.getpid(), 'started_unix': started, 'time_budget_seconds': SECONDS,
               'optimizer_updates': 0, 'optimizer_constructed': False, 'training_queue_started': False,
               'test_evaluated': False, 'test_loader_constructed': False, 'baseline_replaced': False}
    def timeout(_signal, _frame):
        raise TimeoutError('Thirty-minute GPU audit budget exhausted')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(SECONDS)
    try:
        receipt['resources_before'] = idle_gpu_resources()
        receipt['protected_before'] = verify(ROOT, configuration)
        import torch
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.55, device=0)
        receipt['torch_version'] = torch.__version__
        receipt['physical_gpu_index'] = 1
        projector = load_source('_frozen_projector_audit', CPU_AUDIT / 'projector.py')
        packet = torch.load(str(RUN / 'failure_replay.pt'), map_location='cpu')
        packet.pop('optimizer', None)
        if len(packet['svd_captures']) != 12:
            raise RuntimeError('All twelve archived captures required')
        receipt['phase_archived'] = phase_archived(packet['svd_captures'], projector)
        atomic_json(ROOT / 'gpu_progress.json', receipt)
        receipt['phase_model'] = phase_model(packet, projector)
        receipt['protected_after'] = verify(ROOT, configuration)
        receipt['status'] = ('completed_pass' if receipt['phase_archived']['candidate_passed']
                             and receipt['phase_model']['candidate_integration_passed'] else 'completed_fail')
        receipt['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(0)
        receipt['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(0)
    except BaseException as error:
        receipt.update(status='technical_failure', error=repr(error), traceback=traceback.format_exc())
        try:
            receipt['protected_after_failure'] = verify(ROOT, configuration)
        except BaseException as guard_error:
            receipt['post_failure_guard_error'] = repr(guard_error)
    finally:
        signal.alarm(0)
        receipt.update(finished_unix=time.time(), elapsed_seconds=time.time() - started,
                       maximum_resident_set_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        atomic_json(ROOT / 'gpu_audit_receipt.json', receipt)
        print('GPU_AUDIT_FINAL_RECEIPT_JSON', flush=True)
        print(json.dumps(receipt, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
