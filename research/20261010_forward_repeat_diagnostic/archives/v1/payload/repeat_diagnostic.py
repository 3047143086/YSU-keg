"""Exactly two observed original loss forwards; no backward or model update."""
import gc
import hashlib
import json
import os
import pickle
import random
import resource
import signal
import time
import traceback
from pathlib import Path

from repeat_guards import frozen_helpers, predecessor_guard, sha
from repeat_summary import summarize

ROOT = Path(__file__).resolve().parent
SECONDS = 30 * 60
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '4'


def atomic_json(path, value):
    pending = path.with_name(path.name + '.pending.' + str(os.getpid()))
    with pending.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def rng_hashes(helper, packet=None):
    import numpy as np
    import torch
    values = packet if packet is not None else {
        'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}
    return {'python': hashlib.sha256(pickle.dumps(values['python'], protocol=4)).hexdigest(),
            'numpy': hashlib.sha256(pickle.dumps(values['numpy'], protocol=4)).hexdigest(),
            'torch': helper.tensor_hash(values['torch']),
            'cuda': [helper.tensor_hash(value) for value in values['cuda']]}


def backend_settings():
    import torch
    return {'deterministic_algorithms_enabled': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only_enabled': (torch.is_deterministic_algorithms_warn_only_enabled()
                                                if hasattr(torch, 'is_deterministic_algorithms_warn_only_enabled') else None),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_enabled': torch.backends.cudnn.enabled,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'float32_matmul_precision': (torch.get_float32_matmul_precision()
                                        if hasattr(torch, 'get_float32_matmul_precision') else None),
            'cudnn_version': torch.backends.cudnn.version(), 'cuda_version': torch.version.cuda,
            'environment': {name: os.environ.get(name) for name in (
                'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE', 'CUDA_VISIBLE_DEVICES',
                'CUDA_LAUNCH_BLOCKING', 'PYTHONHASHSEED')}}


def state_snapshot(helper, model):
    import torch
    state = model.state_dict()
    aliases = {prefix: getattr(model, prefix + '_preference', None) is getattr(model, prefix + '_gcn').preference
               for prefix in ('id', 'v', 't')}
    if not all(aliases.values()):
        raise RuntimeError('Registered preference aliases lost Parameter identity')
    gradients = [name for name, parameter in model.named_parameters(remove_duplicate=False) if parameter.grad is not None]
    if gradients:
        raise RuntimeError('A forward-only diagnostic must have no parameter gradients')
    if any(not torch.isfinite(value).all().item() for value in state.values()):
        raise FloatingPointError('Nonfinite learned state')
    return {'values_sha256': helper.state_hashes(model),
            'schemas': {name: {'shape': list(value.shape), 'dtype': str(value.dtype)} for name, value in state.items()},
            'parameter_identities': {name: id(parameter) for name, parameter in model.named_parameters(remove_duplicate=False)},
            'preference_alias_identity': aliases,
            'module_training_modes': {name: module.training for name, module in model.named_modules()},
            'parameter_gradients_present': gradients}


def interaction_snapshot(helper, interaction):
    import torch
    values = [interaction] if torch.is_tensor(interaction) else list(interaction)
    if not values or any(not torch.is_tensor(value) for value in values):
        raise RuntimeError('Frozen failure interaction must contain tensors only')
    return [{'sha256': helper.tensor_hash(value), 'shape': list(value.shape), 'dtype': str(value.dtype)}
            for value in values]


def cleanup_transients(model):
    import torch
    cleared = []
    for name in ('id_rep', 'v_rep', 't_rep', 'result_embed'):
        value = getattr(model, name, None)
        if torch.is_tensor(value) and not isinstance(value, torch.nn.Parameter):
            cleared.append({'name': name, 'shape': list(value.shape), 'had_grad_fn': value.grad_fn is not None})
            setattr(model, name, None)
    # The local reference to the last cleared tensor must not retain its graph.
    value = None
    collected = gc.collect()
    torch.cuda.empty_cache()
    return {'cleared': cleared, 'garbage_collected_objects': collected, 'cuda_empty_cache_called': True}


def original_forward(helper, model, packet):
    import torch
    from models.dual_frequency_encoder import FrequencyDecompositionModule
    if not model.training or not torch.is_grad_enabled():
        raise RuntimeError('Original training-mode loss forward with autograd enabled is required')
    original_method = FrequencyDecompositionModule.frequency_decompose_svd
    module_names = {id(module): name for name, module in model.named_modules()}
    calls = []

    def observed(module, value):
        outputs = original_method(module, value)
        if len(outputs) != 3:
            raise RuntimeError('Original decomposition must return three bands')
        if not torch.isfinite(value).all().item() or any(not torch.isfinite(output).all().item() for output in outputs):
            raise FloatingPointError('Original SVD input or band is nonfinite')
        calls.append({'module': module_names[id(module)], 'input': value.detach().cpu().clone(),
                      'outputs': [output.detach().cpu().clone() for output in outputs]})
        return outputs

    try:
        FrequencyDecompositionModule.frequency_decompose_svd = observed
        helper.restore_rng(packet['rng'])
        before_rng = rng_hashes(helper)
        if before_rng != rng_hashes(helper, packet['rng']):
            raise RuntimeError('Packet RNG restore differs before original forward')
        interaction = packet['interaction']
        interaction = (interaction.to('cuda:0') if torch.is_tensor(interaction)
                       else [value.to('cuda:0') for value in interaction])
        losses = model.calculate_loss(interaction)
        components = losses if isinstance(losses, tuple) else (losses,)
        if not components or any(not torch.isfinite(value).all().item() or value.numel() != 1 for value in components):
            raise FloatingPointError('Original loss must contain finite scalar components')
        loss_captures = [value.detach().cpu().clone() for value in components]
        after_rng = rng_hashes(helper)
        # No GPU loss/interaction/graph reference is returned; all captures are detached CPU tensors.
        return {'calls': calls, 'losses': loss_captures, 'rng_before': before_rng,
                'rng_after': after_rng, 'loss_values': [float(value.item()) for value in loss_captures]}
    finally:
        FrequencyDecompositionModule.frequency_decompose_svd = original_method


def observation_rows(helper, first, second, archived):
    import torch
    if len(first['calls']) != 12 or len(second['calls']) != 12 or len(archived) != 12:
        raise RuntimeError('Both forwards and the archive must have twelve captures')
    rows = []
    for index, (old, new, saved) in enumerate(zip(first['calls'], second['calls'], archived)):
        rows.append({'capture_index': index, 'first_module': old['module'], 'second_module': new['module'],
                     'input_shape': list(old['input'].shape), 'input_dtype': str(old['input'].dtype),
                     'first_input_sha256': helper.tensor_hash(old['input']),
                     'second_input_sha256': helper.tensor_hash(new['input']),
                     'archived_input_sha256': helper.tensor_hash(saved['input']),
                     'first_input_exact_archive': torch.equal(old['input'], saved['input']),
                     'second_input_exact_archive': torch.equal(new['input'], saved['input']),
                     'input': helper.compare(new['input'], old['input']),
                     'first_band_sha256': [helper.tensor_hash(value) for value in old['outputs']],
                     'second_band_sha256': [helper.tensor_hash(value) for value in new['outputs']],
                     'bands': [helper.compare(right, left) for left, right in zip(old['outputs'], new['outputs'])]})
    if len(first['losses']) != len(second['losses']):
        raise RuntimeError('Original loss component counts differ')
    losses = []
    for index, (old, new) in enumerate(zip(first['losses'], second['losses'])):
        reference, current = float(old.item()), float(new.item())
        absolute = abs(current - reference)
        inherited_limit = 5e-6 + 5e-5 * abs(reference)
        losses.append({'component_index': index, 'first': reference, 'second': current,
                       'first_sha256': helper.tensor_hash(old), 'second_sha256': helper.tensor_hash(new),
                       'exact': torch.equal(old, new), 'absolute_error': absolute,
                       'inherited_absolute_limit': inherited_limit, 'within_inherited_bounds': absolute <= inherited_limit})
    return rows, losses


def main():
    started = time.time()
    configuration = json.loads((ROOT / 'diagnostic_config.json').read_text())
    config_hash = sha(ROOT / 'diagnostic_config.json')
    receipt = {'status': 'running', 'source_revision': configuration['source_revision'], 'pid': os.getpid(),
               'started_unix': started, 'time_budget_seconds': SECONDS, 'forward_count': 0, 'forward_calls_started': 0,
               'backward_calls': 0, 'optimizer_constructed': False, 'optimizer_updates': 0,
               'training_started': False, 'test_loader_constructed': False, 'test_evaluated': False,
               'projector_invoked': False, 'model_cache_used': False, 'solver_override': False,
               'deterministic_setting_changed': False, 'exact_historical_crash_replayed': False,
               'diagnostic_config_sha256_before': config_hash}
    helper = guard = None

    def timeout(_signal, _frame):
        raise TimeoutError('Thirty-minute two-forward diagnostic budget exhausted')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(SECONDS)
    try:
        helper, guard, receipt['predecessor_before'] = frozen_helpers()
        receipt['resources_before'] = guard.idle_gpu_resources()
        receipt['protected_before'] = guard.verify(ROOT, configuration)
        import torch
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.55, device=0)
        receipt['torch_version'] = torch.__version__
        receipt['physical_gpu_index'] = 1
        receipt['backend_before_construction'] = backend_settings()
        packet = torch.load(str(guard.RUN / 'failure_replay.pt'), map_location='cpu')
        packet.pop('optimizer', None)
        model, receipt['construction'] = helper.construct_model(packet)
        receipt['backend_after_construction'] = backend_settings()
        if receipt['backend_before_construction'] != receipt['backend_after_construction']:
            raise RuntimeError('Construction changed deterministic/backend settings')
        receipt['state_before'] = state_snapshot(helper, model)
        receipt['interaction_before'] = interaction_snapshot(helper, packet['interaction'])
        from models.dual_frequency_encoder import FrequencyDecompositionModule
        original_method = FrequencyDecompositionModule.frequency_decompose_svd
        receipt['forward_calls_started'] = 1
        first = original_forward(helper, model, packet)
        receipt['forward_count'] = 1
        receipt['first_capture_count'] = len(first['calls'])
        if receipt['first_capture_count'] != 12:
            raise RuntimeError('First original forward must produce twelve SVD calls')
        if FrequencyDecompositionModule.frequency_decompose_svd is not original_method:
            raise RuntimeError('Original method not restored after first observer')
        receipt['state_after_first'] = state_snapshot(helper, model)
        if receipt['state_after_first'] != receipt['state_before']:
            raise RuntimeError('Model values/identities/modes changed after first forward')
        receipt['first_cleanup'] = cleanup_transients(model)
        receipt['state_before_second'] = state_snapshot(helper, model)
        if receipt['state_before_second'] != receipt['state_before']:
            raise RuntimeError('First graph cleanup changed model state or Parameter identities')
        receipt['first_loss_values'] = first['loss_values']
        receipt['first_rng_before'] = first['rng_before']
        receipt['first_rng_after'] = first['rng_after']
        receipt['backend_after_first'] = backend_settings()
        if receipt['backend_after_first'] != receipt['backend_before_construction']:
            raise RuntimeError('Deterministic/backend settings changed during first original forward')
        receipt['interaction_after_first'] = interaction_snapshot(helper, packet['interaction'])
        if receipt['interaction_after_first'] != receipt['interaction_before']:
            raise RuntimeError('Saved interaction changed during first original forward')
        atomic_json(ROOT / 'repeat_progress.json', receipt)
        receipt['forward_calls_started'] = 2
        second = original_forward(helper, model, packet)
        receipt['forward_count'] = 2
        receipt['second_capture_count'] = len(second['calls'])
        if receipt['second_capture_count'] != 12:
            raise RuntimeError('Second original forward must produce twelve SVD calls')
        if FrequencyDecompositionModule.frequency_decompose_svd is not original_method:
            raise RuntimeError('Original method not restored after second observer')
        receipt['state_after_second'] = state_snapshot(helper, model)
        if receipt['state_after_second'] != receipt['state_before']:
            raise RuntimeError('Model values/identities/modes changed after second forward')
        receipt['second_cleanup'] = cleanup_transients(model)
        receipt['state_after_cleanup'] = state_snapshot(helper, model)
        if receipt['state_after_cleanup'] != receipt['state_before']:
            raise RuntimeError('Second graph cleanup changed model state or Parameter identities')
        receipt['second_loss_values'] = second['loss_values']
        receipt['second_rng_before'] = second['rng_before']
        receipt['second_rng_after'] = second['rng_after']
        if first['rng_before'] != second['rng_before'] or first['rng_after'] != second['rng_after']:
            raise RuntimeError('Restored or post-forward RNG states disagree between original calls')
        receipt['interaction_after'] = interaction_snapshot(helper, packet['interaction'])
        if receipt['interaction_after'] != receipt['interaction_before']:
            raise RuntimeError('Saved interaction changed during original forwards')
        receipt['backend_after'] = backend_settings()
        if receipt['backend_after'] != receipt['backend_before_construction']:
            raise RuntimeError('Deterministic/backend settings changed during original forwards')
        receipt['rows'], receipt['losses'] = observation_rows(helper, first, second, packet['svd_captures'])
        receipt['summary'] = summarize(receipt['rows'], receipt['losses'])
        receipt['predecessor_after'] = predecessor_guard()
        receipt['protected_after'] = guard.verify(ROOT, configuration)
        receipt['diagnostic_config_sha256_after'] = sha(ROOT / 'diagnostic_config.json')
        if receipt['diagnostic_config_sha256_after'] != config_hash:
            raise RuntimeError('Deployed diagnostic configuration changed')
        receipt.update(status=receipt['summary']['status'], state_unchanged=True,
                       parameter_identities_unchanged=True, preference_aliases_unchanged=True,
                       observer_restored=True, interaction_unchanged=True, restored_rng_matches=True,
                       post_forward_rng_matches=True, backend_settings_unchanged=True,
                       candidate_accepted=False, projector_equivalence_established=False)
        receipt['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(0)
        receipt['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(0)
    except BaseException as error:
        receipt.update(status='technical_failure', error=repr(error), traceback=traceback.format_exc())
        try:
            receipt['predecessor_after_failure'] = predecessor_guard()
            if guard is not None:
                receipt['protected_after_failure'] = guard.verify(ROOT, configuration)
            receipt['diagnostic_config_sha256_after_failure'] = sha(ROOT / 'diagnostic_config.json')
        except BaseException as guard_error:
            receipt['post_failure_guard_error'] = repr(guard_error)
    finally:
        signal.alarm(0)
        receipt.update(finished_unix=time.time(), elapsed_seconds=time.time() - started,
                       maximum_resident_set_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        atomic_json(ROOT / 'repeat_receipt.json', receipt)
        print('ORIGINAL_FORWARD_REPEAT_FINAL_RECEIPT_JSON', flush=True)
        print(json.dumps(receipt, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
