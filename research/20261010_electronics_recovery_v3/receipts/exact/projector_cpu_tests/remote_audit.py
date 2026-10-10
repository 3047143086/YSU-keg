"""Fixed bounded CPU audit of all twelve archived finite SVD inputs."""
import gc
import hashlib
import json
import os
import resource
import signal
import time
import traceback
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import torch
from projector import (check_boundary_gaps, original_svd_bands, projector_bands,
                       split_sizes)

ROOT = Path(__file__).resolve().parent
BASELINE = Path('/data/sunzixuan/electronics_baseline_20261009_v1')
RUN = BASELINE / 'runs/elec_baseline_s2026'
SECONDS = 20 * 60
FORWARD_RELATIVE = 5e-5
FORWARD_MAX_ABSOLUTE_SCALE = 5e-4
GRADIENT_RELATIVE = 1e-3
GRADIENT_MAX_ABSOLUTE_SCALE = 1e-3
FIXED_HASHES = {
    'deployment.json': '58658242005fb07a58f7b677732bdea65ea0aab1feeed191d5310ad939f421d8',
    'input_audit.json': '109caa06a4eb24526438bda5d71ffc51e1d9d7d4c40d75525daa5ea32000c5d2',
    'source_revision.json': 'b4b9ec9c645ad98a7ec8913bc7976d2277d993a076658194f81a8fef16bde201',
    'runs/elec_baseline_s2026/inputs.json': '18dd014949737077a3ce63e004e2bddd82d1870f6caf430f43732b35db3d9b2e',
    'runs/elec_baseline_s2026/config.json': 'a8d27ee50f98cb6b8c89acf0a59c9d42846ffbcfbd85f34b8d864f8b95dcb875',
    'runs/elec_baseline_s2026/failure.json': '6bd82cce4407ffe47502de59d886799e2ecffba31d771569d8604be0c5e63a1f',
    'runs/elec_baseline_s2026/best.pt': '3f9283407ff51178549c2f78e0f8cf61f895c269164afa6b3b9a0a56d8faacda',
    'runs/elec_baseline_s2026/failure_replay.pt': '7b6f12731715c0315018f6b3c903b39a8829cc7727035e18eb92eeaa4bd3deb7',
}


class ResourceGateFailure(RuntimeError):
    pass


class CaptureAuditFailure(RuntimeError):
    def __init__(self, original, row):
        super().__init__(repr(original))
        self.row = row


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    pending = path.with_name(path.name + '.pending.' + str(os.getpid()))
    with pending.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def resources():
    available = 0
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1]) * 1024
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    row = {'ram_available_bytes': available, 'available_cpu_count': cpus,
           'threads': torch.get_num_threads(), 'cuda_initialized': torch.cuda.is_initialized(),
           'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    if available < 12 * 1024 ** 3 or cpus < 4 or row['cuda_initialized']:
        raise ResourceGateFailure('CPU audit resource gate failed: ' + repr(row))
    return row


def verify_protected(configuration):
    rows = []
    def check(path, expected, group):
        actual = sha(path)
        rows.append({'path': str(path), 'group': group, 'expected_sha256': expected,
                     'actual_sha256': actual, 'matches': actual == expected})
        if actual != expected:
            raise RuntimeError('Protected hash changed: ' + str(path))
    for name, expected in FIXED_HASHES.items():
        check(BASELINE / name, expected, 'fixed_receipts_and_packets')
    deployment = json.loads((BASELINE / 'deployment.json').read_text())
    audit = json.loads((BASELINE / 'input_audit.json').read_text())
    if len(deployment['carrier_sha256']) != 35 or len(audit['carrier_files']) != 35:
        raise RuntimeError('Expected exactly 35 protected carrier files')
    for name, digest in deployment['bundle_sha256'].items():
        check(BASELINE / name, digest, 'baseline_bundle')
    for name, digest in deployment['carrier_sha256'].items():
        check(BASELINE / 'carrier/src' / name, digest, 'deployed_carrier')
    for row in audit['carrier_files'].values():
        check(row['path'], row['sha256'], 'original_carrier')
    for row in audit['original_inputs'].values():
        check(row['path'], row['sha256'], 'original_inputs')
    run_inputs = json.loads((RUN / 'inputs.json').read_text())
    for path, digest in run_inputs['sha256'].items():
        check(path, digest, 'run_inputs')
    absent = [{'path': path, 'still_absent': not Path(path).exists()} for path in run_inputs['absent']]
    if not all(row['still_absent'] for row in absent):
        raise RuntimeError('An absent optional input appeared')
    for name, digest in configuration['payload_sha256'].items():
        check(ROOT / name, digest, 'audit_payload')
    return {'all_match': True, 'rows': rows, 'absent_inputs': absent,
            'total_files_checked': len(rows)}


def tensor_sha(value):
    array = value.detach().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def errors(candidate, reference, relative_limit, absolute_scale):
    difference = candidate.detach().double() - reference.detach().double()
    fro_difference = float(torch.linalg.vector_norm(difference).item())
    fro_reference = float(torch.linalg.vector_norm(reference.detach().double()).item())
    maximum = float(difference.abs().max().item())
    reference_maximum = float(reference.detach().abs().max().item())
    relative = (fro_difference / fro_reference if fro_reference else
                (0.0 if fro_difference == 0 else None))
    limit = absolute_scale * max(1.0, reference_maximum)
    return {'relative_frobenius_error': relative, 'absolute_frobenius_error': fro_difference,
            'reference_frobenius_norm': fro_reference, 'maximum_absolute_error': maximum,
            'maximum_absolute_limit': limit, 'reference_maximum_absolute': reference_maximum,
            'passed': relative is not None and relative <= relative_limit and maximum <= limit}


def inspect_capture(index, capture):
    started = time.monotonic()
    value = capture['input']
    if value.device.type != 'cpu' or value.dtype != torch.float32 or not torch.isfinite(value).all():
        raise RuntimeError('Archived input must remain original finite CPU float32')
    row = {'capture_index': index, 'shape': list(value.shape), 'dtype': str(value.dtype),
           'input_tensor_sha256': tensor_sha(value), 'input_cast_before_svd': False,
           'resources': resources(), 'archived_cuda_singular_values': capture['singular'].tolist(),
           'gradient_probe_seed': 2026, 'gradient_probe_type': 'independent_dL_dBand',
           'reconstructs_training_gradient': False}
    calls = []
    original_svd = torch.linalg.svd
    def observed_svd(*arguments, **keywords):
        result = original_svd(*arguments, **keywords)
        calls.append((result[1].detach().clone(), result[2].detach().clone()))
        return result
    torch.linalg.svd = observed_svd
    try:
        reference_input = value.detach().clone().requires_grad_()
        candidate_input = value.detach().clone().requires_grad_()
        reference = original_svd_bands(reference_input, 3)
        candidate = projector_bands(candidate_input, 3)
    except Exception as error:
        if calls:
            row['cpu_original_singular_values'] = calls[0][0].tolist()
            if len(calls) > 1:
                row['cpu_candidate_singular_values'] = calls[1][0].tolist()
            values = calls[-1][0]
            eigenvalues = values.double().square()
            row['cpu_boundary_gaps_before_failure'] = [
                {'boundary_index': boundary,
                 'eigenvalue_gap': float((eigenvalues[boundary - 1] - eigenvalues[boundary]).item())}
                for boundary in (22, 43)]
        raise CaptureAuditFailure(error, row) from error
    finally:
        torch.linalg.svd = original_svd
    if len(calls) != 2:
        raise RuntimeError('Expected one original SVD and one candidate SVD')
    singular, vh = calls[1]
    row.update(cpu_original_singular_values=calls[0][0].tolist(),
               cpu_candidate_singular_values=singular.tolist(),
               cpu_singular_values_identical=torch.equal(singular, calls[0][0]),
               boundary_gaps=check_boundary_gaps(singular, split_sizes(value.shape[1], 3)),
               minimum_adjacent_singular_gap=float((singular[:-1] - singular[1:]).min().item()),
               exact_forward_bands=[torch.equal(new, old) for new, old in zip(candidate, reference)])
    start, relations = 0, []
    with torch.no_grad():
        for size, old in zip(split_sizes(value.shape[1], 3), reference):
            band = vh[start:start + size].T
            direct = value @ (band @ band.T)
            relations.append(errors(direct, old, FORWARD_RELATIVE, FORWARD_MAX_ABSOLUTE_SCALE))
            start += size
        row['direct_XP_diagnostic'] = relations
        row['sum_bands_diagnostic'] = errors(sum(candidate), value, FORWARD_RELATIVE,
                                             FORWARD_MAX_ABSOLUTE_SCALE)
    generator = torch.Generator(device='cpu').manual_seed(2026)
    gradients = [torch.randn(old.shape, dtype=value.dtype, generator=generator) for old in reference]
    sum((old * gradient).sum() for old, gradient in zip(reference, gradients)).backward()
    sum((new * gradient).sum() for new, gradient in zip(candidate, gradients)).backward()
    generic_finite = bool(torch.isfinite(reference_input.grad).all().item())
    candidate_finite = bool(torch.isfinite(candidate_input.grad).all().item())
    row.update(generic_input_gradient_finite=generic_finite,
               candidate_input_gradient_finite=candidate_finite,
               candidate_gradient_dtype=str(candidate_input.grad.dtype),
               generic_nonfinite_elements=int((~torch.isfinite(reference_input.grad)).sum().item()))
    if generic_finite and candidate_finite:
        comparison = errors(candidate_input.grad, reference_input.grad,
                            GRADIENT_RELATIVE, GRADIENT_MAX_ABSOLUTE_SCALE)
        comparison['status'] = 'compared_finite_reference'
        row['generic_gradient_comparison'] = comparison
    else:
        row['generic_gradient_comparison'] = {'status': 'failed_unavailable_nonfinite_reference'
                                              if not generic_finite else 'failed_nonfinite_candidate',
                                              'passed': False, 'metrics': None,
                                              'equivalence_claimed': False}
    row['candidate_mapping_and_finite_gradient_passed'] = (
        all(row['exact_forward_bands']) and all(item['passed'] for item in relations)
        and row['sum_bands_diagnostic']['passed'] and candidate_finite)
    row['elapsed_seconds'] = time.monotonic() - started
    return row


def main():
    configuration = json.loads((ROOT / 'audit_config.json').read_text())
    torch.set_num_threads(4)
    started = time.time()
    receipt = {'status': 'running', 'started_unix': started, 'pid': os.getpid(),
               'cpu_only': True, 'training_executed': False, 'test_evaluated': False,
               'source_revision': configuration['source_revision'], 'time_budget_seconds': SECONDS,
               'torch_version': torch.__version__, 'captures': [], 'baseline_replaced': False,
               'registered_protocol_sha256': configuration['payload_sha256']['PROTOCOL.md']}
    def timeout(_signal, _frame):
        raise TimeoutError('Twenty-minute CPU replay audit budget exhausted')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(SECONDS)
    try:
        receipt['resources_before'] = resources()
        receipt['protected_before'] = verify_protected(configuration)
        packet = torch.load(str(RUN / 'failure_replay.pt'), map_location='cpu')
        captures = packet['svd_captures']
        receipt['original_failure_record'] = packet['record']
        if len(captures) != 12:
            raise RuntimeError('Expected all twelve original SVD captures')
        del packet
        gc.collect()
        for index, capture in enumerate(captures):
            try:
                row = inspect_capture(index, capture)
            except (TimeoutError, MemoryError, ResourceGateFailure):
                raise
            except Exception as error:
                row = dict(getattr(error, 'row', {}))
                row.update(capture_index=index, status='technical_failure', error=repr(error),
                           traceback=traceback.format_exc(), shape=list(capture['input'].shape),
                           candidate_mapping_and_finite_gradient_passed=False,
                           input_tensor_sha256=tensor_sha(capture['input']))
            receipt['captures'].append(row)
            receipt['last_completed_capture_index'] = index
            atomic_json(ROOT / 'audit_progress.json', receipt)
            print(json.dumps({'capture_index': index, 'passed': row['candidate_mapping_and_finite_gradient_passed'],
                              'error': row.get('error')}, allow_nan=False), flush=True)
            gc.collect()
        receipt['protected_after'] = verify_protected(configuration)
        receipt['all_twelve_captures_inspected'] = len(receipt['captures']) == 12
        receipt['candidate_mapping_and_finite_gradients_passed'] = all(
            row['candidate_mapping_and_finite_gradient_passed'] for row in receipt['captures'])
        available = [row['generic_gradient_comparison'] for row in receipt['captures']
                     if row.get('generic_gradient_comparison', {}).get('status') == 'compared_finite_reference']
        receipt['available_generic_gradient_comparisons'] = len(available)
        receipt['unavailable_or_failed_capture_comparisons'] = 12 - len(available)
        receipt['available_generic_gradient_comparisons_passed'] = bool(available) and all(
            row['passed'] for row in available)
        receipt['status'] = ('completed_pass' if receipt['candidate_mapping_and_finite_gradients_passed']
                             and receipt['available_generic_gradient_comparisons_passed'] else 'completed_fail')
    except BaseException as error:
        receipt.update(status='technical_failure', error=repr(error), traceback=traceback.format_exc())
        try:
            receipt['protected_after_failure'] = verify_protected(configuration)
        except BaseException as guard_error:
            receipt['post_failure_guard_error'] = repr(guard_error)
    finally:
        signal.alarm(0)
        receipt.update(finished_unix=time.time(), elapsed_seconds=time.time() - started,
                       cuda_initialized=torch.cuda.is_initialized(),
                       maximum_resident_set_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        atomic_json(ROOT / 'replay_audit_receipt.json', receipt)
        print('AUDIT_FINAL_RECEIPT_JSON', flush=True)
        print(json.dumps(receipt, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
