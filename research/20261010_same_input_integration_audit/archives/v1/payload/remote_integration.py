"""One same-realized-X candidate loss forward and at most one actual backward."""
import json
import os
import resource
import signal
import time
import traceback
import warnings
from pathlib import Path
from integration_gate import EXPECTED_PARAMETERS, same_input_forward_passed, integration_passed
from method_observer import observed_methods
from parameter_snapshot import all_named_parameters, synthetic_check
from repeat_guards import frozen_helpers, predecessor_guard, load_source, sha
from runtime_checks import backend_settings, cleanup, interaction_snapshot, rng_hashes, state_snapshot, unique_parameter_rows
from warning_observer import observed_warnings, reported_svd_fallback

ROOT = Path(__file__).resolve().parent
SECONDS = 1800
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


def tensor_metadata(value):
    return {'data_ptr': value.data_ptr(), 'storage_data_ptr': value.storage().data_ptr(),
            'storage_offset': value.storage_offset(), 'stride': list(value.stride()),
            'version': value._version, 'device': str(value.device), 'dtype': str(value.dtype),
            'shape': list(value.shape)}


def gradient_observer(helper, events):
    def observed(gradient):
        import torch
        events.append({'finite': bool(torch.isfinite(gradient).all().item()),
                       'element_count': gradient.numel(), 'sha256': helper.tensor_hash(gradient),
                       'shape': list(gradient.shape), 'dtype': str(gradient.dtype)})
        # None preserves the real gradient, including true zero values.
        return None
    return observed


def candidate_forward(helper, model, packet, projector, result, hook_handles, warning_phase):
    import torch
    from models.dual_frequency_encoder import FrequencyDecompositionModule
    if not model.training or not torch.is_grad_enabled():
        raise RuntimeError('Training-mode actual loss graph is required')
    original_method = FrequencyDecompositionModule.frequency_decompose_svd
    original_svd = torch.linalg.svd
    module_names = {id(module): name for name, module in model.named_modules()}
    phase = {'name': None, 'row': None, 'input': None}

    def observe_svd(*arguments, **keywords):
        row, name, value = phase['row'], phase['name'], phase['input']
        if (row is None or name not in ('shadow', 'candidate') or len(arguments) != 1
                or keywords != {'full_matrices': False} or arguments[0].data_ptr() != value.data_ptr()
                or arguments[0].dtype != value.dtype or arguments[0].device != value.device):
            raise ValueError('Unexpected SVD input/solver/argument contract')
        event = {'capture_index': row['capture_index'], 'phase': name, 'full_matrices': False,
                 'driver_specified': False, 'input_data_ptr': arguments[0].data_ptr(), 'successful': False}
        row['svd_events'].append(event)
        try:
            outputs = original_svd(*arguments, **keywords)
            event['successful'] = True
            row['successful_' + name + '_svd_calls'] += 1
            return outputs
        except BaseException as error:
            row['svd_failures'] += 1
            event['error'] = repr(error)
            raise

    def same_input(module, value):
        row = {'capture_index': len(result['rows']), 'module': module_names[id(module)],
               'input_shape': list(value.shape), 'input_dtype': str(value.dtype), 'bands_requested': module.M,
               'successful_shadow_svd_calls': 0, 'successful_candidate_svd_calls': 0, 'svd_failures': 0,
               'svd_events': [], 'candidate_output_gradient_events': [[], [], []], 'actual_input_gradient_events': []}
        result['rows'].append(row)
        if value.dtype != torch.float32 or value.device != torch.device('cuda:0') or not value.requires_grad:
            raise RuntimeError('Actual float32 CUDA input with upstream autograd is required')
        detached = value.detach()
        before = tensor_metadata(value)
        row['input_metadata_before'] = before
        row['input_sha256_before'] = helper.tensor_hash(value)
        row['same_storage_shadow_input'] = (detached.data_ptr() == value.data_ptr()
            and detached.storage().data_ptr() == value.storage().data_ptr()
            and detached.storage_offset() == value.storage_offset() and detached.stride() == value.stride())
        shadow_rng = rng_hashes(helper)
        warning_phase['name'] = 'shadow_call_' + str(row['capture_index'])
        phase.update(name='shadow', row=row, input=detached)
        with torch.no_grad():
            old = original_method(module, detached)
        row['shadow_rng_unchanged'] = rng_hashes(helper) == shadow_rng
        row['input_metadata_after_shadow'] = tensor_metadata(value)
        row['input_sha256_after_shadow'] = helper.tensor_hash(value)
        candidate_rng = rng_hashes(helper)
        warning_phase['name'] = 'candidate_call_' + str(row['capture_index'])
        phase.update(name='candidate', row=row, input=value)
        new = projector.projector_bands(value, module.M)
        row['candidate_rng_unchanged'] = rng_hashes(helper) == candidate_rng
        phase.update(name=None, row=None, input=None)
        warning_phase['name'] = 'loss_forward_other_operations'
        row['input_metadata_after_candidate'] = tensor_metadata(value)
        row['input_sha256_after_candidate'] = helper.tensor_hash(value)
        row['input_unchanged'] = row['input_sha256_before'] == row['input_sha256_after_shadow'] == row['input_sha256_after_candidate']
        row['input_version_storage_unchanged'] = before == row['input_metadata_after_shadow'] == row['input_metadata_after_candidate']
        if len(old) != 3 or len(new) != 3:
            raise RuntimeError('Exactly three original and candidate bands required')
        row['shadow_outputs_detached'] = all(not output.requires_grad and output.grad_fn is None for output in old)
        row['outputs_finite'] = all(bool(torch.isfinite(output).all().item()) for output in tuple(old) + tuple(new))
        row['bands_exact'] = [torch.equal(left, right) for left, right in zip(old, new)]
        row['original_band_sha256'] = [helper.tensor_hash(output) for output in old]
        row['candidate_band_sha256'] = [helper.tensor_hash(output) for output in new]
        row['descriptive_band_errors'] = [helper.compare(right, left) for left, right in zip(old, new)]
        row['candidate_grad_fn_names'] = [type(output.grad_fn).__name__ for output in new]
        row['candidate_outputs_custom_grad_fn'] = [output.requires_grad and isinstance(output.grad_fn, projector._ProjectorBands._backward_cls)
                                                    for output in new]
        for index, output in enumerate(new):
            if output.requires_grad:
                hook_handles.append(output.register_hook(gradient_observer(helper, row['candidate_output_gradient_events'][index])))
        hook_handles.append(value.register_hook(gradient_observer(helper, row['actual_input_gradient_events'])))
        # Only the real candidate outputs survive on the loss graph.
        return new

    helper.restore_rng(packet['rng'])
    result['rng_before_forward'] = rng_hashes(helper)
    result['restored_rng_matches'] = result['rng_before_forward'] == rng_hashes(helper, packet['rng'])
    if not result['restored_rng_matches']:
        raise RuntimeError('Saved packet RNG restoration failed')
    result['full_forward_calls_started'] = 1
    with observed_methods(((FrequencyDecompositionModule, 'frequency_decompose_svd', same_input),
                           (torch.linalg, 'svd', observe_svd))):
        interaction = packet['interaction']
        interaction = interaction.to('cuda:0') if torch.is_tensor(interaction) else [value.to('cuda:0') for value in interaction]
        losses = model.calculate_loss(interaction)
    result['full_forward_count'] = 1
    result['observer_restored'] = FrequencyDecompositionModule.frequency_decompose_svd is original_method
    result['svd_identity_restored'] = torch.linalg.svd is original_svd
    result['rng_after_forward'] = rng_hashes(helper)
    result['forward_rng_unchanged'] = result['rng_after_forward'] == result['rng_before_forward']
    components = losses if isinstance(losses, tuple) else (losses,)
    result['loss_component_count'] = len(components)
    valid = bool(components) and all(torch.is_tensor(value) and value.numel() == 1 and bool(torch.isfinite(value).all().item()) for value in components)
    total = sum(components) if valid else None
    result['actual_loss_finite'] = valid and bool(torch.isfinite(total).all().item())
    result['loss_values'] = [float(value.detach().cpu().item()) for value in components] if valid else None
    result['total_loss_value'] = float(total.detach().cpu().item()) if result['actual_loss_finite'] else None
    return total, original_method, original_svd


def main():
    started = time.time()
    configuration = json.loads((ROOT / 'integration_config.json').read_text())
    config_hash = sha(ROOT / 'integration_config.json')
    result = {'status': 'running', 'source_revision': configuration['source_revision'], 'pid': os.getpid(),
              'started_unix': started, 'time_budget_seconds': SECONDS, 'full_forward_count': 0,
              'full_forward_calls_started': 0, 'backward_calls': 0, 'rows': [], 'gradients': {},
              'optimizer_constructed': False, 'optimizer_updates': 0, 'training_started': False,
              'test_evaluated': False, 'cache_used': False, 'solver_override': False,
              'historical_crash_replayed': False, 'generic_vjp_equivalence_claimed': False,
              'publication_gain_claimed': False, 'integration_config_sha256_before': config_hash}
    hook_handles = []
    helper = guard = None
    result['warnings'] = []
    warning_phase = {'name': 'pre_torch_import'}
    original_warning_method = warnings.showwarning
    warning_context = observed_warnings(result['warnings'], warning_phase)

    def timeout(_signal, _frame):
        raise TimeoutError('Thirty-minute single-forward integration budget exhausted')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(SECONDS)
    warning_context.__enter__()
    try:
        helper, guard, result['old_evidence_before'] = frozen_helpers()
        result['protected_before'] = guard.verify(ROOT, configuration)
        # Check idle resources after the final hash read, directly before CUDA initialization.
        result['resources_before'] = guard.idle_gpu_resources()
        import torch
        torch.set_num_threads(4)
        result['actual_torch_synthetic_parameter_check'] = synthetic_check()
        warning_phase['name'] = 'cuda_initialization'
        torch.cuda.set_per_process_memory_fraction(.55, device=0)
        result['torch_version'] = torch.__version__
        result['backend_before'] = backend_settings()
        projector = load_source('_same_input_frozen_projector', guard.CPU_AUDIT / 'projector.py')
        packet = torch.load(str(guard.RUN / 'failure_replay.pt'), map_location='cpu')
        packet.pop('optimizer', None)
        warning_phase['name'] = 'model_construction'
        model, result['construction'] = helper.construct_model(packet)
        if backend_settings() != result['backend_before']:
            raise RuntimeError('Construction changed backend settings')
        result['state_before'] = state_snapshot(helper, model)
        result['interaction_before'] = interaction_snapshot(helper, packet['interaction'])
        if any(value.grad is not None for _, value in all_named_parameters(model)):
            raise RuntimeError('No prior gradient may enter the actual backward audit')
        warning_phase['name'] = 'loss_forward'
        total, original_method, original_svd = candidate_forward(helper, model, packet, projector, result, hook_handles, warning_phase)
        result['state_after_forward'] = state_snapshot(helper, model)
        if result['state_after_forward'] != result['state_before'] or backend_settings() != result['backend_before']:
            raise RuntimeError('Forward changed learned state/identities/modes/backend settings')
        if (not result['observer_restored'] or not result['svd_identity_restored'] or not result['forward_rng_unchanged']
                or interaction_snapshot(helper, packet['interaction']) != result['interaction_before']):
            raise RuntimeError('Method/RNG/interaction guard failed before backward')
        atomic_json(ROOT / 'integration_progress.json', result)
        result['no_reported_svd_fallback'] = not reported_svd_fallback(result['warnings'])
        if same_input_forward_passed(result) and result['no_reported_svd_fallback']:
            result['backward_calls'] = 1
            warning_phase['name'] = 'actual_loss_backward'
            total.backward()
            result['gradients'] = {name: {'finite': bool(torch.isfinite(value.grad).all().item()),
                'element_count': value.grad.numel(), 'sha256': helper.tensor_hash(value.grad), 'shape': list(value.grad.shape)}
                for name, value in unique_parameter_rows(model) if value.grad is not None}
        result['rng_after_backward'] = rng_hashes(helper)
        result['backward_rng_unchanged'] = result['rng_after_backward'] == result['rng_after_forward']
        from models.dual_frequency_encoder import FrequencyDecompositionModule
        result['observer_restored'] = result['observer_restored'] and FrequencyDecompositionModule.frequency_decompose_svd is original_method
        result['svd_identity_restored'] = result['svd_identity_restored'] and torch.linalg.svd is original_svd
        result['state_after_backward'] = state_snapshot(helper, model)
        if result['state_after_backward'] != result['state_before']:
            raise RuntimeError('Backward changed learned state/identities/modes')
        result['state_unchanged'] = result['parameter_identities_unchanged'] = result['preference_aliases_unchanged'] = True
        result['registered_parameter_coverage_complete'] = True
        result['all_alias_gradients_finite'] = all(value.grad is None or bool(torch.isfinite(value.grad).all().item()) for _, value in all_named_parameters(model))
        result['participating_parameter_names_match_original'] = set(result['gradients']) == set(EXPECTED_PARAMETERS)
        result['interaction_after'] = interaction_snapshot(helper, packet['interaction'])
        result['interaction_unchanged'] = result['interaction_after'] == result['interaction_before']
        result['backend_after'] = backend_settings()
        result['backend_settings_unchanged'] = result['backend_after'] == result['backend_before']
        for row in result['rows']:
            row['unused_candidate_output_indices'] = [index for index, events in enumerate(row['candidate_output_gradient_events']) if not events]
            row['actual_input_gradient_observed'] = bool(row['actual_input_gradient_events'])
        # Release the scalar loss graph even when exactness failed and no backward ran.
        total = None
        result['transients_cleared'] = cleanup(model)
        warning_phase['name'] = 'post_forward_backward_guards'
        if state_snapshot(helper, model) != result['state_before']:
            raise RuntimeError('Graph cleanup changed learned state or aliases')
        result['old_evidence_after'] = predecessor_guard()
        result['protected_after'] = guard.verify(ROOT, configuration)
        result['protected_hashes_unchanged'] = result['protected_before']['all_match'] and result['protected_after']['all_match']
        result['old_receipts_unchanged'] = result['old_evidence_before']['all_match'] and result['old_evidence_after']['all_match']
        result['integration_config_sha256_after'] = sha(ROOT / 'integration_config.json')
        result['deployment_config_unchanged'] = result['integration_config_sha256_after'] == config_hash
        if not all(result.get(flag) for flag in ('interaction_unchanged', 'backward_rng_unchanged', 'backend_settings_unchanged',
                                               'observer_restored', 'svd_identity_restored', 'deployment_config_unchanged')):
            raise RuntimeError('Final interaction/RNG/backend/method/config guard failed')
        result['no_reported_svd_fallback'] = not reported_svd_fallback(result['warnings'])
        result['status'] = 'completed_pass' if integration_passed(result) else 'completed_fail'
        result['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(0)
        result['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(0)
    except BaseException as error:
        result.update(status='technical_failure', error=repr(error), traceback=traceback.format_exc())
        try:
            result['old_evidence_after_failure'] = predecessor_guard()
            if guard is not None:
                result['protected_after_failure'] = guard.verify(ROOT, configuration)
            result['integration_config_sha256_after_failure'] = sha(ROOT / 'integration_config.json')
        except BaseException as guard_error:
            result['post_failure_guard_error'] = repr(guard_error)
    finally:
        signal.alarm(0)
        warning_context.__exit__(None, None, None)
        try:
            for handle in hook_handles:
                handle.remove()
        except BaseException as error:
            result.update(status='technical_failure', hook_cleanup_error=repr(error))
        result['warning_observer_restored'] = warnings.showwarning is original_warning_method
        result['no_reported_svd_fallback'] = not reported_svd_fallback(result['warnings'])
        if not result['warning_observer_restored']:
            result.update(status='technical_failure', error='Warning observer identity not restored')
        elif result['status'] == 'completed_pass' and not result['no_reported_svd_fallback']:
            result['status'] = 'completed_fail'
        result.update(finished_unix=time.time(), elapsed_seconds=time.time() - started,
                      maximum_resident_set_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        atomic_json(ROOT / 'integration_receipt.json', result)
        print('SAME_INPUT_INTEGRATION_FINAL_RECEIPT_JSON', flush=True)
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
