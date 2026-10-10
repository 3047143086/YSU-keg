"""Read only the saved v3 failure packet on CPU; never recompute an SVD."""
import hashlib
import json
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '4'

ROOT = Path('/data/sunzixuan/electronics_baseline_20261010_recovery_v3')
RUN = ROOT / 'runs/elec_projector_recovery_s2026'
PROJECTOR_SHA = 'e44eb739482f3fe0448e1c9fdb066e36c4d6ea22d05a0c25a32734f29a856ab6'
FAILURE_SHA = '390b9a80dbeaa291e344f7fe8658ce07a508627d3e379b2e77ea79913a7d989b'
POST_SHA = 'b6e36489cc08365c66e2cd828acc82f5c79fe2b0d70ce1f0ff09a13c0c218fb3'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()


def main():
    started = time.time()
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    if available < 12 * 1024 ** 3 or len(os.sched_getaffinity(0)) < 4:
        raise RuntimeError('Read-only CPU diagnostic requires 12 GiB available RAM and four CPU cores')
    post = json.loads((RUN / 'post_run_guard.json').read_text())
    assert post['all_match'] and post['v1_preserved'] and sha(RUN / 'post_run_guard.json') == POST_SHA
    assert sha(ROOT / 'projector.py') == PROJECTOR_SHA and sha(RUN / 'failure.json') == FAILURE_SHA
    protected_before = {name: sha(name) for name in post['sha256']}
    assert protected_before == post['sha256'], 'Protected bytes drifted before diagnostic'
    packet_path = RUN / 'failure_replay.pt'
    packet_stat = packet_path.stat()
    packet_sha_before = sha(packet_path)
    import torch
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    packet = torch.load(str(packet_path), map_location='cpu')
    failure = json.loads((RUN / 'failure.json').read_text())
    assert packet['record'] == failure
    rows = []
    for index, capture in enumerate(packet['svd_captures']):
        value, singular = capture['input'], capture['singular']
        row = {'capture_index': index, 'saved_keys': sorted(capture), 'shape': list(value.shape),
               'input_dtype': str(value.dtype), 'input_device': str(value.device),
               'input_finite': bool(torch.isfinite(value).all()), 'input_sha256': tensor_hash(value),
               'gradient_kind': capture.get('gradient_kind'),
               'svd_grad_mode_enabled': capture.get('svd_grad_mode_enabled'),
               'generic_output_gradients_available': capture.get('generic_output_gradients_available'),
               'saved_output_gradient_names': sorted(capture.get('output_gradients', {})),
               'actual_input_gradient_observed': capture.get('actual_input_gradient_observed'),
               'actual_input_gradient_present': capture.get('actual_input_gradient') is not None,
               'reported_warnings': capture.get('reported_warnings'), 'solver_error': capture.get('solver_error')}
        gradient = capture.get('actual_input_gradient')
        if gradient is not None:
            row['actual_input_gradient'] = {'dtype': str(gradient.dtype), 'shape': list(gradient.shape),
                                           'finite': bool(torch.isfinite(gradient).all())}
        if singular is not None:
            assert singular.ndim == 1 and len(singular) == 64 and singular.dtype == torch.float32
            eigenvalues = singular.double().square()
            scale = eigenvalues.max()
            threshold = 32.0 * torch.finfo(singular.dtype).eps * scale
            gaps = singular[:-1] - singular[1:]
            boundaries = []
            for boundary in (22, 43):
                gap = eigenvalues[boundary - 1] - eigenvalues[boundary]
                boundaries.append({'boundary_index': boundary,
                    'zero_based_singular_indices': [boundary - 1, boundary],
                    'singular_values': singular[boundary - 1:boundary + 1].tolist(),
                    'singular_values_exactly_equal': bool(singular[boundary - 1] == singular[boundary]),
                    'singular_value_gap': float((singular[boundary - 1] - singular[boundary]).item()),
                    'eigenvalues': eigenvalues[boundary - 1:boundary + 1].tolist(),
                    'eigenvalue_gap': float(gap.item()), 'gap_strictly_positive': bool(gap > 0),
                    'gap_finite': bool(torch.isfinite(gap)),
                    'relative_eigenvalue_gap': float((gap / scale).item()) if scale > 0 else None,
                    'required_eigenvalue_gap': float(threshold.item()),
                    'gap_over_required_gap': float((gap / threshold).item()) if threshold > 0 else None,
                    'frozen_guard_passes': bool(torch.isfinite(gap) and gap > threshold)})
            row.update(singular_dtype=str(singular.dtype), singular_finite=bool(torch.isfinite(singular).all()),
                singular_values=singular.tolist(), eigenvalues=eigenvalues.tolist(),
                maximum_eigenvalue=float(scale.item()), float32_epsilon=torch.finfo(singular.dtype).eps,
                boundary_eps_multiplier=32.0, required_eigenvalue_gap=float(threshold.item()),
                equal_adjacent_singular_values=int((gaps == 0).sum()),
                minimum_singular_value_gap=float(gaps.min().item()), boundary_rows=boundaries)
        rows.append(row)
    tensor_states = [value for value in packet['model'].values() if torch.is_tensor(value)]
    optimizer_values = [value for state in packet['optimizer']['state'].values() for value in state.values()
                        if torch.is_tensor(value)]
    model_finite = all(bool(torch.isfinite(value).all()) for value in tensor_states)
    optimizer_finite = all(bool(torch.isfinite(value).all()) for value in optimizer_values)
    protected_after = {name: sha(name) for name in post['sha256']}
    packet_sha_after = sha(packet_path)
    assert protected_after == protected_before == post['sha256']
    assert packet_sha_before == packet_sha_after and packet_path.stat().st_size == packet_stat.st_size
    assert not torch.cuda.is_initialized()
    finished = time.time()
    result = {'status': 'completed_readonly_diagnostic', 'elapsed_seconds': finished - started,
        'remote_started_unix': started, 'remote_finished_unix': finished,
        'remote_started_utc': datetime.fromtimestamp(started, timezone.utc).isoformat(),
        'remote_finished_utc': datetime.fromtimestamp(finished, timezone.utc).isoformat(),
        'clock_offset_measured': False,
        'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'cuda_initialized': torch.cuda.is_initialized(),
        'torch_version': torch.__version__, 'cpu_threads': torch.get_num_threads(),
        'available_ram_bytes_before': available, 'maximum_resident_set_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'svd_recomputed': False, 'model_replayed': False, 'backward_executed': False,
        'training_executed': False, 'threshold_modified': False, 'packet_keys': sorted(packet),
        'packet_path': str(packet_path), 'packet_bytes': packet_stat.st_size,
        'packet_sha256_before': packet_sha_before, 'packet_sha256_after': packet_sha_after,
        'failure_json_sha256': FAILURE_SHA, 'failure_record': failure, 'model_state_tensor_count': len(tensor_states),
        'model_state_finite': model_finite, 'optimizer_state_tensor_count': len(optimizer_values),
        'optimizer_state_finite': optimizer_finite, 'saved_capture_count': len(rows), 'captures': rows,
        'frozen_projector_sha256': PROJECTOR_SHA, 'post_run_guard_sha256': POST_SHA,
        'protected_before': protected_before, 'protected_after': protected_after,
        'protected_distinct_paths': len(protected_before), 'all_protected_hashes_match': True,
        'numerics_receipt_exists': (RUN / 'numerics_receipt.json').exists(),
        'complete_epoch_progress_bytes': (RUN / 'progress.jsonl').stat().st_size}
    print('RECOVERY_BOUNDARY_DIAGNOSTIC_JSON', flush=True)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
