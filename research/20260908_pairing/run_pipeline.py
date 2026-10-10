"""Single-GPU, bounded queue; source artifacts are strictly read-only."""

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT / 'experiment.json').read_text())
STARTED = time.monotonic()
DEADLINE = STARTED + SPEC['max_pipeline_seconds']
ACTIVE = None
METRICS = ('recall@10', 'ndcg@10', 'recall@20', 'ndcg@20', 'recall@200')


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def status(stage, **details):
    record = dict(stage=stage, pid=os.getpid(), updated_unix=time.time(),
                  elapsed_seconds=time.monotonic() - STARTED, **details)
    atomic_json(ROOT / 'pipeline_status.json', record)
    print(json.dumps(record), flush=True)


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def prepare():
    source = Path(SPEC['source_root'])
    manifest = {'source_root': str(source), 'source_sha256': {},
                'official_validation_evaluated': False, 'official_test_evaluated': False,
                'historical_internal_development_reused': True}
    paths = [source / 'features' / 'global_bank.npy']
    for seed in SPEC['seeds']:
        paths += [source / 'prepared' / str(seed) / name for name in
                  ('base_scores.npy', 'support_mask.npy', 'targets.npy', 'roles.npy', 'history.npy')]
        paths += [source / 'shared' / str(seed) / name for name in
                  ('relevance_scores.npy', 'ready.json')]
    for path in paths:
        if not path.is_file():
            raise RuntimeError('Missing prepared input: ' + str(path))
        manifest['source_sha256'][str(path)] = digest(path)
    atomic_json(ROOT / 'input_manifest.json', manifest)
    status('prepared', source=str(source), inputs_hashed=len(paths))


def available_gpu():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=15)
    rows = []
    selected = None
    for line in result.stdout.strip().splitlines():
        gpu, memory, utilization = [int(part.strip()) for part in line.split(',')]
        rows.append({'gpu': gpu, 'memory_used_mb': memory, 'utilization': utilization})
        if (gpu in SPEC['allowed_gpus'] and memory <= SPEC['gpu_max_used_mb_before_launch']
                and utilization <= SPEC['gpu_max_utilization_before_launch']):
            selected = gpu
    return selected, rows


def stop_owned(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)


def collect_summary():
    record = {'updated_unix': time.time(), 'seeds': {}, 'aggregate': {},
              'official_test_evaluated': False, 'official_validation_evaluated': False,
              'historical_internal_development_reused': True,
              'interpretation': 'Exploratory screen, not confirmation of novelty or improvement.'}
    completed = 0
    for seed in SPEC['seeds']:
        entries = {}
        for variant in SPEC['variants']:
            directory = ROOT / 'results' / str(seed) / variant
            entry = {'status': 'pending'}
            for name in ('progress.json', 'result.json', 'failure.json'):
                if (directory / name).exists():
                    value = json.loads((directory / name).read_text())
                    entry = dict(status='failed', error=value) if name == 'failure.json' else value
            if entry.get('status') == 'completed':
                completed += 1
            entries[variant] = entry
        record['seeds'][str(seed)] = entries
    record.update(completed_jobs=completed, total_jobs=len(SPEC['seeds']) * len(SPEC['variants']))
    record['all_complete'] = completed == record['total_jobs']
    for variant in SPEC['variants']:
        rows = [s[variant] for s in record['seeds'].values() if s[variant].get('status') == 'completed']
        if rows:
            record['aggregate'][variant] = {'completed_seeds': len(rows), 'metrics': {
                name: {'mean': float(np.mean([r['dev_metrics'][name] for r in rows])),
                       'std': float(np.std([r['dev_metrics'][name] for r in rows], ddof=1)) if len(rows) > 1 else None}
                for name in METRICS}}
    baseline_rows = [s['additive'] for s in record['seeds'].values() if s['additive'].get('status') == 'completed']
    if baseline_rows:
        record['aggregate']['frozen_baseline'] = {'completed_seeds': len(baseline_rows), 'metrics': {
            name: {'mean': float(np.mean([r['baseline_metrics'][name] for r in baseline_rows]))} for name in METRICS}}
    comparisons = {}
    for control in SPEC['variants'] + ['frozen_baseline']:
        if control == SPEC['primary_variant']:
            continue
        pairs = []
        for entries in record['seeds'].values():
            candidate = entries[SPEC['primary_variant']]
            if candidate.get('status') != 'completed':
                continue
            if control == 'frozen_baseline':
                pairs.append((candidate['dev_metrics'], candidate['baseline_metrics']))
            elif entries[control].get('status') == 'completed':
                pairs.append((candidate['dev_metrics'], entries[control]['dev_metrics']))
        if pairs:
            comparisons[control] = {'paired_seeds': len(pairs), 'metrics': {}}
            for name in METRICS:
                left = np.array([p[0][name] for p in pairs])
                right = np.array([p[1][name] for p in pairs])
                comparisons[control]['metrics'][name] = {
                    'per_seed_delta': (left - right).tolist(),
                    'relative_mean_gain': float((left.mean() - right.mean()) / max(right.mean(), 1e-12))}
    record['paired_cov_vs_controls'] = comparisons
    record['decision'] = 'incomplete; no conclusion'
    if record['all_complete']:
        passed = True
        for control in SPEC['primary_controls'] + ['frozen_baseline']:
            for name in SPEC['primary_metrics']:
                value = comparisons[control]['metrics'][name]
                passed &= (value['relative_mean_gain'] >= SPEC['minimum_relative_gain']
                           and min(value['per_seed_delta']) > 0)
            for name in ('recall@10', 'ndcg@10'):
                passed &= comparisons[control]['metrics'][name]['relative_mean_gain'] >= 0
        record['decision'] = ('screen_passed; fresh confirmation and published baselines still required' if passed
                              else 'screen_not_passed; do not claim reliable pairing benefit')
        # Seeds share users. Resample users after averaging seed deltas, not 3N independent rows.
        bootstrap = {}
        for control in SPEC['primary_controls']:
            deltas = []
            reference_users = None
            for seed in SPEC['seeds']:
                with np.load(ROOT / 'results' / str(seed) / 'paired_cov' / 'per_user.npz') as candidate, \
                     np.load(ROOT / 'results' / str(seed) / control / 'per_user.npz') as other:
                    if not np.array_equal(candidate['users'], other['users']):
                        raise ValueError('Per-user comparisons have mismatched users')
                    if reference_users is not None and not np.array_equal(reference_users, candidate['users']):
                        raise ValueError('Seed-wise user identities differ')
                    reference_users = candidate['users'].copy()
                    deltas.append(candidate['metrics'] - other['metrics'])
            mean_delta = np.mean(deltas, axis=0)
            rng = np.random.RandomState(20260908)
            samples = np.array([mean_delta[rng.randint(0, len(mean_delta), len(mean_delta))].mean(axis=0)
                                for _ in range(2000)])
            lo, hi = np.quantile(samples, [0.025, 0.975], axis=0)
            bootstrap[control] = {name: {'delta': float(mean_delta[:, i].mean()), 'ci95': [float(lo[i]), float(hi[i])]}
                                  for i, name in enumerate(METRICS)}
        record['user_clustered_bootstrap_unadjusted_exploratory'] = bootstrap
        confidence_passed = all(bootstrap[control][name]['ci95'][0] > 0
            for control in SPEC['primary_controls'] for name in SPEC['primary_metrics'])
        record['confidence_screen_passed'] = confidence_passed
        if not confidence_passed:
            record['decision'] = 'screen_not_passed; no reliably positive pairing benefit over all primary controls'
    atomic_json(ROOT / 'summary.json', record)
    lines = ['# History Pairing Screen', '', 'Exploratory Baby internal development, not official test.', '',
             '| Variant | Seeds | R20 | N20 | R200 |', '| --- | --- | --- | --- | --- |']
    for variant, row in record['aggregate'].items():
        metrics = row['metrics']
        lines.append('| %s | %d/3 | %.8f | %.8f | %.8f |' % (variant, row['completed_seeds'],
            metrics['recall@20']['mean'], metrics['ndcg@20']['mean'], metrics['recall@200']['mean']))
    lines += ['', '%d / %d jobs completed.' % (completed, record['total_jobs']), '', record['decision'], '',
              'All variants use the same support-only history, held training targets, and frozen full-catalog baseline.',
              'The learned branch changes the actual full-catalog ranking. Calibration selects epochs and scale.',
              'Internal development has already been used for earlier screens; this is not an untouched confirmation set.']
    temporary = ROOT / 'RESULTS.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    temporary.replace(ROOT / 'RESULTS.md')
    return record


def run_job(seed, variant, smoke=False):
    global ACTIVE
    while True:
        if time.monotonic() >= DEADLINE:
            raise TimeoutError('Pipeline limit reached before launch')
        gpu, resources = available_gpu()
        if gpu is not None:
            break
        status('waiting_for_gpu', next_job='%s_%s' % (seed, variant), resources=resources)
        collect_summary()
        time.sleep(20)
    output_root = ROOT / 'smoke' if smoke else ROOT
    output = output_root / 'results' / str(seed) / variant
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-u', str(ROOT / 'train_pairing.py'), '--root', str(output_root),
               '--source-root', SPEC['source_root'], '--seed', str(seed), '--variant', variant,
               '--epochs', str(2 if smoke else SPEC['epochs']), '--patience', str(SPEC['patience']),
               '--batch-size', str(SPEC['batch_size']), '--device', 'cuda:0']
    if smoke:
        command.append('--smoke')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    began = time.monotonic()
    limit = min(600 if smoke else SPEC['max_run_seconds'], DEADLINE - began)
    with (output / 'stdout.log').open('x') as log:
        process = subprocess.Popen(command, cwd=str(ROOT), env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        ACTIVE = process
        atomic_json(output / 'launch.json', {'pid': process.pid, 'gpu': gpu, 'command': command,
                                            'started_unix': time.time(), 'max_seconds': limit})
        while process.poll() is None:
            status('smoke_running' if smoke else 'training', seed=seed, variant=variant,
                   child_pid=process.pid, gpu=gpu, output=str(output))
            if not smoke:
                collect_summary()
            if time.monotonic() - began > limit:
                stop_owned(process)
                atomic_json(output / 'failure.json', {'status': 'failed', 'error': 'Time limit exceeded'})
                break
            time.sleep(10)
        ACTIVE = None
    if process.returncode or not (output / 'result.json').is_file():
        if not (output / 'failure.json').exists():
            atomic_json(output / 'failure.json', {'status': 'failed', 'exit_code': process.returncode})
        return False
    result = json.loads((output / 'result.json').read_text())
    if smoke and (result.get('status') != 'smoke_passed' or result.get('dev_evaluated') is not False):
        raise RuntimeError('Smoke run violated no-development-evaluation contract')
    if not smoke and result.get('status') != 'completed':
        raise RuntimeError('Formal run did not produce a completed result')
    return True


def main():
    import fcntl
    with (ROOT / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / 'input_manifest.json').exists():
            raise RuntimeError('Refusing to silently restart an existing experiment')
        status('checking_inputs')
        prepare()
        command = [sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT), '-p', 'test_*.py', '-v']
        result = subprocess.run(command, timeout=180)
        if result.returncode:
            raise RuntimeError('Unit tests failed')
        for variant in SPEC['variants']:
            if not run_job(SPEC['seeds'][0], variant, smoke=True):
                raise RuntimeError('Real-data smoke failed: ' + variant)
        for seed in SPEC['seeds']:
            for variant in SPEC['variants']:
                run_job(seed, variant)
        record = collect_summary()
        status('completed' if record['all_complete'] else 'finished_with_failed_runs',
               completed_jobs=record['completed_jobs'], total_jobs=record['total_jobs'])


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error), traceback=traceback.format_exc())
        collect_summary()
        raise
    finally:
        if ACTIVE is not None:
            stop_owned(ACTIVE)
