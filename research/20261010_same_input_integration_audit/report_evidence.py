"""Reproduce a compact evidence artifact from exact archived remote receipts."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    archive_root = ROOT / 'archives/v1/receipts'
    archive = json.loads((archive_root / 'archive_receipt.json').read_text())
    if archive['process_alive']:
        raise RuntimeError('Exited process required for final evidence')
    for name, expected in archive['files'].items():
        if hashlib.sha256((archive_root / name).read_bytes()).hexdigest() != expected['sha256']:
            raise RuntimeError('Raw archived receipt SHA mismatch: ' + name)
    receipt = json.loads((archive_root / 'integration_receipt.json').read_text())
    configuration = json.loads((archive_root / 'integration_config.json').read_text())
    tests = json.loads((archive_root / 'tests.json').read_text())
    if receipt['source_revision'] != configuration['source_revision']:
        raise RuntimeError('Receipt/deployment source revision mismatch')
    for name, expected in configuration['payload_sha256'].items():
        if hashlib.sha256((ROOT / 'archives/v1/payload' / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen archived payload SHA mismatch: ' + name)
    # Recompute with the actual archived gate, independent of later workspace edits.
    gate_path = ROOT / 'archives/v1/payload/integration_gate.py'
    spec = importlib.util.spec_from_file_location('_archived_same_input_gate', gate_path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    computed_pass = gate.integration_passed(receipt)
    if receipt['status'] == 'completed_pass' and not computed_pass:
        raise RuntimeError('A claimed completed pass does not satisfy the frozen conjunction')
    names = ('status', 'source_revision', 'pid', 'elapsed_seconds', 'full_forward_count',
        'full_forward_calls_started', 'backward_calls', 'actual_loss_finite', 'loss_component_count',
        'loss_values', 'total_loss_value', 'rows', 'gradients', 'state_unchanged',
        'parameter_identities_unchanged', 'preference_aliases_unchanged', 'observer_restored',
        'svd_identity_restored', 'interaction_unchanged', 'restored_rng_matches', 'forward_rng_unchanged',
        'backward_rng_unchanged', 'backend_settings_unchanged', 'backend_before', 'backend_after',
        'protected_hashes_unchanged', 'old_receipts_unchanged', 'deployment_config_unchanged',
        'registered_parameter_coverage_complete', 'all_alias_gradients_finite', 'no_reported_svd_fallback',
        'warnings', 'warning_observer_restored', 'optimizer_constructed', 'optimizer_updates', 'training_started',
        'cache_used', 'solver_override', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes',
        'maximum_resident_set_bytes', 'historical_crash_replayed', 'generic_vjp_equivalence_claimed',
        'publication_gain_claimed', 'error', 'transients_cleared')
    evidence = {name: receipt.get(name) for name in names}
    evidence['recomputed_frozen_gate_passed'] = computed_pass
    evidence['deployed_payload_sha256'] = configuration['payload_sha256']
    evidence['actual_cpu_tests'] = {'exit_code': tests['exit_code'], 'cuda_visible_devices': tests['cuda_visible_devices'],
                                    'integration_config_sha256': tests['integration_config_sha256']}
    test_count = re.search(r'Ran (\d+) tests? in ', tests['stderr'])
    evidence['actual_cpu_tests']['tests_run'] = int(test_count.group(1)) if test_count else None
    evidence['raw_remote_receipts'] = archive['files']
    for name in ('protected_before', 'protected_after', 'protected_after_failure'):
        value = receipt.get(name, {})
        evidence[name] = {'all_match': value.get('all_match'), 'count': value.get('count')}
    for name in ('old_evidence_before', 'old_evidence_after'):
        evidence[name] = receipt.get(name)
    state = receipt.get('state_before', {})
    evidence['registered_parameter_coverage'] = state.get('parameter_state_coverage')
    evidence['requires_grad_by_path'] = state.get('parameter_requires_grad')
    evidence['process_alive_at_archive'] = archive['process_alive']
    evidence['gpu1_at_archive'] = archive['gpu1']
    evidence['compute_apps_at_archive'] = archive['compute_apps']
    (ROOT / 'INTEGRATION_EVIDENCE.json').write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'status': receipt['status'], 'gate_recomputed': computed_pass, 'source_revision': receipt['source_revision'],
        'full_forward_count': receipt['full_forward_count'], 'backward_calls': receipt['backward_calls'],
        'exact_bands': sum(sum(row.get('bands_exact', [])) for row in receipt['rows']),
        'participating_parameters': len(receipt['gradients']), 'elapsed_seconds': receipt['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()
