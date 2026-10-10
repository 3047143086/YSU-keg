"""Build a small public diagnostic artifact from exact archived receipt bytes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    directory = ROOT / 'archives/v2/receipts'
    archive = json.loads((directory / 'archive_receipt.json').read_text())
    if archive['process_alive']:
        raise RuntimeError('An exited diagnostic process is required')
    for name, expected in archive['files'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected['sha256']:
            raise RuntimeError('Local archived raw receipt changed: ' + name)
    receipt = json.loads((directory / 'repeat_receipt.json').read_text())
    evidence = {name: receipt.get(name) for name in (
        'status', 'source_revision', 'pid', 'elapsed_seconds', 'forward_count', 'forward_calls_started',
        'backward_calls', 'optimizer_updates', 'optimizer_constructed', 'construction',
        'actual_torch_synthetic_parameter_check', 'state_unchanged', 'parameter_identities_unchanged',
        'preference_aliases_unchanged', 'observer_restored', 'interaction_unchanged', 'restored_rng_matches',
        'post_forward_rng_matches', 'backend_settings_unchanged', 'backend_before_construction',
        'first_loss_values', 'second_loss_values', 'summary', 'rows', 'losses',
        'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes', 'maximum_resident_set_bytes', 'error')}
    for ordinal in ('first', 'second'):
        before, after = receipt.get(ordinal + '_rng_before'), receipt.get(ordinal + '_rng_after')
        evidence[ordinal + '_rng_state_unchanged'] = (before == after if before is not None and after is not None else None)
    for name in ('protected_before', 'protected_after', 'protected_after_failure'):
        guard = receipt.get(name, {})
        evidence[name] = {'all_match': guard.get('all_match'), 'count': guard.get('count')}
    evidence['raw_remote_receipts'] = archive['files']
    evidence['process_alive_at_archive'] = archive['process_alive']
    evidence['gpu1_at_archive'] = archive['gpu1']
    evidence['compute_apps_at_archive'] = archive['compute_apps']
    (ROOT / 'v2/DIAGNOSTIC_EVIDENCE.json').write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({name: evidence[name] for name in ('status', 'source_revision', 'elapsed_seconds',
                                                      'forward_count', 'summary', 'error')}, indent=2))


if __name__ == '__main__':
    main()
