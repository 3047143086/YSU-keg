"""Verify immutable completed controls and training-only cache provenance."""
import hashlib
import json
import math
import time
from pathlib import Path

SOURCE = Path('/data/sunzixuan/target_isolated_20261008_autoresearch_v1')
METRICS = ('recall@10', 'ndcg@10', 'recall@20', 'ndcg@20')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def process_matches(pid, script):
    path = Path('/proc') / str(pid) / 'cmdline'
    try:
        return str(script).encode() in path.read_bytes()
    except FileNotFoundError:
        return False


def predecessor_state(source=SOURCE):
    source = Path(source)
    launch = read(source / 'launch.json')
    state = read(source / 'pipeline_status.json')
    return state, process_matches(launch['pid'], source / 'run_pipeline.py')


def validate_result(result, dataset):
    require(result.get('status') == 'completed', 'A completed baseline is required')
    require(result.get('dataset') == dataset and result.get('seed') == 2026,
            'Baseline dataset or seed differs')
    require(result.get('model') == 'BRIDGE', 'Only original BRIDGE controls may be reused')
    for key, value in (('test_evaluated', False), ('test_loader_constructed', False),
                       ('inputs_unchanged', True), ('fresh_initialization', True)):
        require(result.get(key) is value, 'Baseline protocol differs: ' + key)
    require(result.get('selection_metric') == 'rounded_4_validation_recall@20',
            'Baseline selection protocol differs')
    for key in METRICS:
        value = result.get('metrics', {}).get(key)
        require(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1,
                'Invalid baseline metric: ' + key)


def validate_config(config, dataset):
    arguments = config['arguments']
    required = {'dataset': dataset, 'seed': 2026, 'max_epochs': 300, 'device': 'cuda',
                'cuda_memory_fraction': .55, 'train_batch_size': None, 'learning_rate': None,
                'smoke': False, 'plugin_module': None, 'plugin_class': None, 'plugin_config': None}
    for key, value in required.items():
        require(key in arguments and arguments[key] == value, 'Baseline arguments differ: ' + key)
    require(config.get('budget_overridden') is False, 'Baseline budget was overridden')
    settings = config['effective_config']
    for key, value in {'stopping_step': 20, 'eval_step': 1, 'eval_batch_size': 64,
                       'train_batch_size': 2048, 'feature_ablation': 'all',
                       'use_behavior_correction': True, 'use_topk_correction': True,
                       'use_behavior_gate': True, 'use_calibration_gate': True}.items():
        require(settings.get(key) == value, 'Baseline configuration differs: ' + key)


def verify_files(manifest):
    for filename, expected in manifest['sha256'].items():
        require(sha(filename) == expected, 'Source input changed: ' + filename)
    for filename in manifest.get('absent', []):
        require(not Path(filename).exists(), 'Previously absent input appeared: ' + filename)


def reuse(destination, dataset, source=SOURCE):
    destination, source = Path(destination), Path(source)
    state, alive = predecessor_state(source)
    require(not alive and state['status'] == 'closed_validation_gate',
            'The prior scientific comparison must finish before the next trial')
    require(read(source / 'sports_screen.json')['passed'] is False,
            'This iteration is authorized by the unsuccessful Sports screen')
    require(dataset in ('baby', 'sports'), 'Unsupported dataset')
    require(source.resolve() != destination.resolve(), 'Use an isolated destination')
    old_deployment, deployment = read(source / 'deployment.json'), read(destination / 'deployment.json')
    require(old_deployment['carrier_sha256'] == deployment['carrier_sha256'],
            'Source and destination carrier snapshots differ')
    for name, expected in deployment['carrier_sha256'].items():
        require(sha(source / 'carrier/src' / name) == expected,
                'Source carrier drift: ' + name)
        require(sha(destination / 'carrier/src' / name) == expected,
                'Destination carrier drift: ' + name)
    for name in ('train.py', 'prepare.py', 'evidence.py'):
        require(sha(source / name) == sha(destination / name), 'Frozen module differs: ' + name)
    require(sha(source / 'proposal.py') == sha(destination / 'isolation.py'),
            'The original cache validation module differs')

    run = source / 'runs' / (dataset + '_baseline_s2026')
    result, config = read(run / 'result.json'), read(run / 'config.json')
    validate_result(result, dataset)
    validate_config(config, dataset)
    verify_files(read(run / 'inputs.json'))
    for name, key in (('best.pt', 'checkpoint_sha256'),
                      ('per_user_validation.npz', 'per_user_sha256')):
        require(sha(run / name) == result[key], 'Baseline output changed: ' + name)

    preparation = read(source / 'cache' / dataset / 'prepare.json')
    require(preparation.get('formal_training_allowed') is True, 'Source mechanism diagnostic failed')
    require(preparation.get('official_test_evaluated') is False
            and preparation.get('official_validation_evaluated') is False
            and preparation.get('held_out_labels_used_for_diagnostic_or_negative_filtering') is False,
            'Source diagnostic used held-out labels')
    for filename, expected in preparation['input_sha256'].items():
        require(sha(filename) == expected, 'Source preparation input changed: ' + filename)
    cache_path = Path(preparation['cache']['path'])
    require(cache_path.resolve() == (source / 'cache' / dataset / 'cache.npz').resolve(),
            'Unexpected cache path')
    require(sha(cache_path) == preparation['cache']['sha256'], 'Source cache drift')
    receipt = {'checked_unix': time.time(), 'source_run': str(run), 'checks_passed': True,
               'result_sha256': sha(run / 'result.json'), 'config_sha256': sha(run / 'config.json'),
               'cache': preparation['cache'], 'prepare_sha256': sha(source / 'cache' / dataset / 'prepare.json'),
               'result': result, 'test_evaluated': False}
    output = destination / 'reuse_receipt.json'
    receipts = read(output) if output.exists() else {'datasets': {}}
    require(dataset not in receipts['datasets'], 'Already reused this baseline')
    receipts['datasets'][dataset] = receipt
    with (destination / 'reuse_receipt.pending.json').open('w', encoding='utf-8') as stream:
        json.dump(receipts, stream, indent=2, allow_nan=False)
    (destination / 'reuse_receipt.pending.json').replace(output)
    return run, preparation
