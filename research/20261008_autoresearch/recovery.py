"""Audit reuse of one immutable completed arm; never reuse interrupted training."""
import csv
import hashlib
import json
import math
import time
from pathlib import Path

SOURCE = '/data/sunzixuan/target_isolated_20260911_soft_v1'
ALGORITHM_FILES = ('train.py', 'proposal.py', 'evidence.py', 'prepare.py')
RANKING_METRICS = ('recall@10', 'ndcg@10', 'recall@20', 'ndcg@20')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_result(result):
    require(result.get('status') == 'completed', 'Only completed results may be reused')
    require(result.get('dataset') == 'baby' and result.get('seed') == 2026,
            'Recovery result must be Baby seed 2026')
    require(result.get('model') == 'TargetIsolatedBRIDGE', 'Unexpected recovered model')
    for field, expected in (('test_evaluated', False), ('test_loader_constructed', False),
                            ('inputs_unchanged', True), ('fresh_initialization', True)):
        require(result.get(field) is expected, 'Invalid result protocol: ' + field)
    require(result.get('selection_metric') == 'rounded_4_validation_recall@20',
            'Checkpoint selection protocol differs')
    for key in RANKING_METRICS:
        value = result.get('metrics', {}).get(key)
        require(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1,
                'Invalid recovered metric: ' + key)


def check_config(config):
    arguments = config['arguments']
    expected = {'dataset': 'baby', 'seed': 2026, 'max_epochs': 300, 'device': 'cuda',
                'cuda_memory_fraction': .55, 'train_batch_size': None, 'learning_rate': None,
                'smoke': False, 'plugin_module': 'proposal', 'plugin_class': 'TargetIsolatedBRIDGE'}
    for key, value in expected.items():
        require(key in arguments and arguments[key] == value, 'Run arguments differ: ' + key)
    require(config.get('budget_overridden') is False, 'Cannot reuse a shortened run')
    effective = config['effective_config']
    for key, value in {'evidence_mode': 'soft_edge', 'isolation_strength': .5,
                       'stopping_step': 20, 'eval_step': 1, 'eval_batch_size': 64,
                       'train_batch_size': 2048}.items():
        require(effective.get(key) == value, 'Recovered configuration differs: ' + key)


def recover_edge(destination, source=SOURCE):
    """Verify data, code, config, outputs and lineage before returning old output path."""
    destination, source = Path(destination), Path(source)
    require(destination.resolve() != source.resolve(), 'Recovery must use a new directory')
    source_state = read(source / 'pipeline_status.json')
    require(source_state['status'] == 'technical_failure', 'Unexpected source pipeline state')
    launch = read(source / 'launch.json')
    command_line = Path('/proc') / str(launch['pid']) / 'cmdline'
    if command_line.is_file():
        require(str(source / 'run_pipeline.py').encode() not in command_line.read_bytes(),
                'Source pipeline is still active')
    run = source / 'runs/baby_edge_s2026'
    result = read(run / 'result.json')
    check_result(result)
    check_config(read(run / 'config.json'))
    manifest = read(run / 'inputs.json')
    for filename, expected in manifest['sha256'].items():
        require(sha(filename) == expected, 'Recovered input changed: ' + filename)
    for filename in manifest.get('absent', []):
        require(not Path(filename).exists(), 'Previously absent input appeared: ' + filename)
    for filename in ALGORITHM_FILES:
        require(sha(source / filename) == sha(destination / filename),
                'Algorithm source differs: ' + filename)
    old_carrier = read(source / 'deployment.json')['carrier_sha256']
    new_carrier = read(destination / 'deployment.json')['carrier_sha256']
    require(old_carrier == new_carrier, 'Carrier model/config snapshot differs')
    for filename, expected in new_carrier.items():
        require(sha(destination / 'carrier/src' / filename) == expected,
                'Recovery carrier changed: ' + filename)
    require(sha(source / 'cache/baby/cache.npz') == sha(destination / 'cache/baby/cache.npz'),
            'Prepared evidence cache differs')
    for filename, key in (('best.pt', 'checkpoint_sha256'),
                          ('per_user_validation.npz', 'per_user_sha256')):
        require(sha(run / filename) == result[key], 'Recovered output changed: ' + filename)
    receipt = {'checked_unix': time.time(), 'source_run': str(run),
               'source_status': source_state['status'], 'source_failed_stage': source_state['stage'],
               'result_sha256': sha(run / 'result.json'),
               'config_sha256': sha(run / 'config.json'), 'checks_passed': True,
               'reused_arms': ['baby_edge_s2026'], 'retry_limit_per_missing_arm': 1,
               'test_evaluated': False, 'result': result}
    with (destination / 'recovery_receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    return run


def finish_autoresearch(root, state):
    """Persist the bounded recovery decision separately from the scientific verdict."""
    root = Path(root)
    directory = root / 'autoresearch/loop-recovery'
    directory.mkdir(parents=True, exist_ok=True)
    completed = sum(1 for path in (root / 'runs').glob('*/result.json')
                    if read(path).get('status') == 'completed')
    if (root / 'recovery_receipt.json').is_file():
        completed += 1
    finished = state.get('status') in (
        'closed_validation_gate', 'pilot_passed_needs_multiseed_and_multimodal_attribution')
    finished = finished or (state.get('status') == 'closed_mechanism_gate' and completed >= 3)
    outcome = 'keep' if finished and completed >= 3 else 'crash'
    stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    with (directory / 'loop-results.tsv').open('w', newline='', encoding='utf-8') as stream:
        stream.write('# metric_direction: higher_is_better\n')
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['iteration', 'timestamp', 'commit', 'metric', 'delta', 'guard',
                         'guard-metric', 'status', 'description'])
        writer.writerow([0, stamp, '-', 1, 0, 'pass', '-', 'baseline',
                         'Source: one completed soft arm; baseline technical failure'])
        writer.writerow([1, stamp, '-', completed, completed - 1,
                         'pass' if finished else 'fail', '-', outcome,
                         'One bounded recovery; scientific verdict stored in screens'])
    handoff = {'version': '2.1.0', 'source': 'loop', 'timestamp': stamp,
               'status': 'COMPLETE' if finished else 'ERROR',
               'results_tsv': str(directory / 'loop-results.tsv'),
               'findings': [{'pipeline_status': state.get('status'), 'completed_arms': completed,
                             'scientific_screens': state.get('screens', {}),
                             'technical_error': state.get('error')}],
               'config': {'goal': 'Complete the interrupted fixed-protocol experiment once',
                          'scope': 'run_pipeline.py recovery.py',
                          'metric': 'completed formal arms, excluding smoke runs',
                          'direction': 'higher_is_better', 'iterations': 1}}
    with (directory / 'handoff.json').open('w', encoding='utf-8') as stream:
        json.dump(handoff, stream, indent=2, allow_nan=False)
