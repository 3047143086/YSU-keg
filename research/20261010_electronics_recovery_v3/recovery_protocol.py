"""Pure protocol validation, independent of Torch and recommendation data."""
import argparse
import json
from pathlib import Path

SEEDS = tuple(range(2026, 2031))
QUEUE_SECONDS = 72 * 3600
JOB_SECONDS = 24 * 3600
REMOTE = Path('/data/sunzixuan/electronics_baseline_20261010_recovery_v3')
PLUGIN_KEYS = frozenset(('spectral_projector_file', 'spectral_projector_sha256'))


def run_command(root, python, seed):
    root = Path(root)
    if seed not in SEEDS:
        raise ValueError('The complete frozen seed order is required')
    return [str(python), '-u', str(root / 'guarded_train.py'), '--bridge-root', str(root / 'carrier'),
            '--dataset', 'elec', '--seed', str(seed), '--output',
            str(root / 'runs' / ('elec_projector_recovery_s' + str(seed))),
            '--device', 'cuda', '--gpu-id', '1', '--cuda-memory-fraction', '.55',
            '--plugin-module', 'proposal', '--plugin-class', 'ProjectorRecoveryBRIDGE',
            '--plugin-config', str(root / 'plugin_config.json')]


def parse_frozen_arguments(arguments, root):
    parser = argparse.ArgumentParser()
    parser.add_argument('--bridge-root', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--gpu-id', required=True)
    parser.add_argument('--cuda-memory-fraction', type=float, required=True)
    parser.add_argument('--plugin-module', required=True)
    parser.add_argument('--plugin-class', required=True)
    parser.add_argument('--plugin-config', type=Path, required=True)
    args = parser.parse_args(arguments)
    root = Path(root).resolve()
    if (args.seed not in SEEDS or args.dataset != 'elec' or args.device != 'cuda'
            or args.gpu_id != '1' or args.cuda_memory_fraction != .55
            or args.plugin_module != 'proposal' or args.plugin_class != 'ProjectorRecoveryBRIDGE'
            or args.bridge_root.resolve() != root / 'carrier'
            or args.plugin_config.resolve() != root / 'plugin_config.json'
            or args.output.resolve() != root / 'runs' / ('elec_projector_recovery_s' + str(args.seed))):
        parser.error('Arguments do not match the frozen fresh numerical-recovery job')
    return args


def assert_algorithm_config(config, reference):
    actual = json.loads(json.dumps(config, default=str))
    expected = json.loads(json.dumps(reference, default=str))
    if actual.get('seed') not in SEEDS:
        raise RuntimeError('Unexpected training seed')
    if set(actual) - set(expected) != PLUGIN_KEYS:
        raise RuntimeError('Only the two registered numerical plugin keys may be added')
    actual = {k: v for k, v in actual.items() if k not in PLUGIN_KEYS and k != 'seed'}
    expected = {k: v for k, v in expected.items() if k != 'seed'}
    if actual != expected:
        differing = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
        raise RuntimeError('Original Electronics algorithm configuration changed: ' + ', '.join(differing))


def validate_result(row, seed):
    if (row.get('status') != 'completed' or row.get('dataset') != 'elec' or row.get('seed') != seed
            or row.get('model') != 'ProjectorRecoveryBRIDGE' or row.get('budget_overridden', False)
            or row.get('test_evaluated') is not False or row.get('test_loader_constructed') is not False
            or row.get('inputs_unchanged') is not True or row.get('fresh_initialization') is not True
            or not 1 <= row.get('epochs_completed', 0) <= 300):
        raise RuntimeError('Unexpected recovery result protocol')


def numerical_receipt(steps, counts):
    return {'finite_gradient_steps_completed': steps, 'nonfinite_steps_skipped': 0,
            'solver_changed': False, 'optimizer_changed': False,
            'backward_implementation_changed': True,
            'forward_arithmetic': 'original torch.linalg.svd; U_band @ diag(s_band) @ Vh_band',
            'backward_implementation': 'reviewed band-projector first VJP; float64 accumulation',
            'frequency_calls': counts['calls'], 'eval_cache_integrated': False,
            'wall_time_budget_changed_from_v1': True, 'job_seconds': JOB_SECONDS,
            'queue_seconds': QUEUE_SECONDS, 'test_evaluated': False}
