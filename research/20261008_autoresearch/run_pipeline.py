"""Bounded, validation-only experiment queue with preregistered stop rules."""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from recovery import recover_edge, finish_autoresearch

ROOT = Path(__file__).resolve().parent
DEADLINE = time.time() + 48 * 3600
STATE = {'started_unix': time.time(), 'status': 'running', 'jobs': [], 'screens': {}}


def update(**fields):
    STATE.update(fields, updated_unix=time.time())
    pending = ROOT / 'pipeline_status.pending.json'
    pending.write_text(json.dumps(STATE, indent=2), encoding='utf-8')
    os.replace(pending, ROOT / 'pipeline_status.json')
    print(json.dumps(fields), flush=True)


def execute(name, command, timeout):
    remaining = DEADLINE - time.time()
    if remaining <= 0:
        raise TimeoutError('Pipeline reached its 48-hour budget')
    update(stage=name, command=command)
    started = time.time()
    log_path = ROOT / 'logs' / (name + '.log')
    with log_path.open('xb') as log:
        result = subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, timeout=min(timeout, remaining),
                                env=dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                                         OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1',
                                         PYTHONPATH=str(ROOT / 'carrier/src') + os.pathsep + str(ROOT)))
    STATE['jobs'].append({'name': name, 'exit_code': result.returncode,
                          'seconds': time.time() - started, 'log': str(log_path)})
    update()
    if result.returncode:
        raise RuntimeError(name + ' failed; see ' + str(log_path))


def available_gpu():
    while time.time() < DEADLINE:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
        candidates = []
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                index, free = [int(x.strip()) for x in line.split(',')]
                if free >= 14336:
                    candidates.append((free, index))
        if candidates:
            # Run only one of this pipeline's jobs at once, even if both GPUs fit.
            return max(candidates)[1]
        update(stage='waiting_for_14GiB_free_GPU', gpu=result.stdout.strip())
        time.sleep(30)
    raise TimeoutError('GPU resource wait exceeded the pipeline budget')


def train(dataset, arm, smoke):
    gpu = available_gpu()
    name = '{}_{}_{}'.format(dataset, arm, 'smoke' if smoke else 's2026')
    output = ROOT / ('smoke' if smoke else 'runs') / name
    command = [sys.executable, '-u', str(ROOT / 'train.py'), '--bridge-root', str(ROOT / 'carrier'),
               '--dataset', dataset, '--seed', '2026', '--output', str(output),
               '--device', 'cuda', '--gpu-id', str(gpu)]
    if arm != 'baseline':
        command += ['--plugin-module', 'proposal', '--plugin-class', 'TargetIsolatedBRIDGE',
                    '--plugin-config', str(ROOT / 'configs' / (dataset + '_' + arm + '.json'))]
    if smoke:
        command.append('--smoke')
    execute(name, command, 1800 if smoke else 8 * 3600)
    return output


def screen(dataset, outputs):
    import numpy as np
    results = {arm: json.loads((path / 'result.json').read_text()) for arm, path in outputs.items()}
    base, edge, history = [results[arm]['metrics'] for arm in ('baseline', 'edge', 'history')]
    relative = {key: edge[key] / base[key] - 1 for key in ('recall@20', 'ndcg@20')}
    guard = {key: edge[key] - base[key] for key in ('recall@10', 'ndcg@10')}
    stronger = {key: edge[key] - history[key] for key in ('recall@20', 'ndcg@20')}
    passed = all(v >= .005 for v in relative.values()) and all(v >= -.0002 for v in guard.values()) \
        and all(v > 0 for v in stronger.values())
    bootstrap = {}
    with np.load(outputs['edge'] / 'per_user_validation.npz') as proposed:
        for control in ('baseline', 'history'):
            with np.load(outputs[control] / 'per_user_validation.npz') as reference:
                if not np.array_equal(proposed['user_ids'], reference['user_ids']) or not np.array_equal(
                        proposed['positive_lengths'], reference['positive_lengths']):
                    raise RuntimeError('Paired validation user alignment failed')
                for key in ('recall_10', 'ndcg_10', 'recall_20', 'ndcg_20'):
                    delta = proposed[key] - reference[key]
                    rng = np.random.default_rng(2026)
                    draws = np.array([delta[rng.integers(len(delta), size=len(delta))].mean()
                                      for _ in range(2000)])
                    bootstrap[control + ':' + key] = {'mean_delta': float(delta.mean()),
                        'user_bootstrap_95ci': np.quantile(draws, [.025, .975]).tolist()}
    record = {'dataset': dataset, 'passed': passed, 'seed': 2026,
              'metrics': {arm: row['metrics'] for arm, row in results.items()},
              'relative_vs_baseline': relative, 'shortlist_absolute_delta': guard,
              'absolute_vs_history': stronger, 'paired_exploratory_not_independent_confirmation': bootstrap,
              'test_evaluated': False}
    with (ROOT / (dataset + '_screen.json')).open('x') as stream:
        json.dump(record, stream, indent=2)
    STATE['screens'][dataset] = record
    update()
    return passed


def main():
    (ROOT / 'logs').mkdir(exist_ok=False)
    (ROOT / 'configs').mkdir(exist_ok=False)
    tested = json.loads((ROOT / 'test_receipt.json').read_text())
    if tested['exit_code'] != 0:
        raise RuntimeError('Passing tests are required before preparation')
    update(stage='tests_already_passed_preparing')
    for dataset in ('baby', 'sports'):
        cache_dir = ROOT / 'cache' / dataset
        execute(dataset + '_prepare', [sys.executable, '-u', str(ROOT / 'prepare.py'),
                '--bridge-root', str(ROOT / 'carrier'), '--dataset', dataset,
                '--output-dir', str(cache_dir)], 1800)
        prepared = json.loads((cache_dir / 'prepare.json').read_text())
        update(diagnostic=prepared)
        if not prepared['formal_training_allowed']:
            update(status='closed_mechanism_gate', stage=dataset, finished_unix=time.time())
            return
        for arm in ('history', 'edge'):
            with (ROOT / 'configs' / (dataset + '_' + arm + '.json')).open('x') as stream:
                config = {'evidence_mode': ('soft_edge' if arm == 'edge' else arm),
                          'evidence_cache': str(cache_dir / 'cache.npz'),
                          'evidence_cache_sha256': prepared['cache']['sha256']}
                if arm == 'edge':
                    config['isolation_strength'] = 0.5
                json.dump(config, stream, indent=2)
        outputs = {}
        if dataset == 'baby':
            outputs['edge'] = recover_edge(ROOT)
            update(reused={'baby_edge_s2026': str(outputs['edge'])})
        for arm in ('baseline', 'history', 'edge'):
            train(dataset, arm, smoke=True)
        for arm in ('edge', 'baseline', 'history'):
            if arm not in outputs:
                outputs[arm] = train(dataset, arm, smoke=False)
        if not screen(dataset, outputs):
            update(status='closed_validation_gate', stage=dataset, finished_unix=time.time())
            return
    update(status='pilot_passed_needs_multiseed_and_multimodal_attribution', stage='done', finished_unix=time.time())


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        update(status='technical_failure', error=str(error), traceback=traceback.format_exc(),
               finished_unix=time.time())
        raise
    finally:
        finish_autoresearch(ROOT, STATE)
