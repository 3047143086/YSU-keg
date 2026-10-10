"""Bounded, detached queue for the training-internal retrieval screen."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT / 'experiment.json').read_text())
STARTED = time.monotonic()
DEADLINE = STARTED + SPEC['max_pipeline_seconds']
ACTIVE = None


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def status(stage, **details):
    record = dict(stage=stage, pid=os.getpid(), updated_unix=time.time(),
                  elapsed_seconds=time.monotonic() - STARTED, **details)
    atomic_json(ROOT / 'pipeline_status.json', record)
    print(json.dumps(record), flush=True)


def prepare():
    source = Path(SPEC['source_stage1'])
    if not (source / 'features').is_dir() or not (source / 'bridge_src').is_dir():
        raise RuntimeError('Missing existing features or source snapshot')
    (ROOT / 'features').symlink_to(source / 'features', target_is_directory=True)
    snapshot = ROOT / 'bridge_src'
    snapshot.mkdir()
    for path in (source / 'bridge_src').rglob('*'):
        relative = path.relative_to(source / 'bridge_src')
        if (path.suffix not in ('.py', '.yaml', '.yml') or not path.is_file()
                or any(part in ('saved_models', 'logs', 'log', '__pycache__', '.git') for part in relative.parts)):
            continue
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    split = ROOT / 'split'
    split.mkdir()
    shutil.copy2(source / 'split' / 'baby.inter', split / 'baby.inter')
    manifest = json.loads((source / 'split_manifest.json').read_text())
    if digest(split / 'baby.inter') != manifest['split_sha256']:
        raise RuntimeError('Source split hash mismatch')
    frame = pd.read_csv(split / 'baby.inter', sep='\t')
    if not set(frame.x_label.unique()).issubset({0, 1}):
        raise RuntimeError('Official test labels present in internal split')
    support = frame[frame.x_label == 0]
    held = frame[frame.x_label == 1]
    if held.userID.duplicated().any() or frame.duplicated(['userID', 'itemID']).any():
        raise RuntimeError('This screen requires unique pairs and one held target per user')
    if not set(held.userID).issubset(set(support.userID)):
        raise RuntimeError('Held user missing support history')
    users = np.sort(held.userID.to_numpy(dtype=np.int64))
    rng = np.random.RandomState(SPEC['role_seed'])
    shuffled = rng.permutation(users)
    roles = np.full(manifest['users'], -1, dtype=np.int8)
    first = int(len(users) * SPEC['role_fractions'][0])
    second = first + int(len(users) * SPEC['role_fractions'][1])
    for role, selected in enumerate((shuffled[:first], shuffled[first:second], shuffled[second:])):
        roles[selected] = role
    np.save(ROOT / 'roles.npy', roles)
    manifest.update(role_seed=SPEC['role_seed'], role_counts={str(role): int((roles == role).sum())
                    for role in range(3)}, role_sha256=digest(ROOT / 'roles.npy'),
                    role_names=['retriever_training', 'calibration', 'internal_development'],
                    selection_metric='role1 calibration only',
                    historical_stage1_development_reused=True,
                    note='Internal mechanism screen; not a fresh publication confirmation set.')
    atomic_json(ROOT / 'split_manifest.json', manifest)
    provenance = {}
    for directory in (ROOT,):
        for path in sorted(directory.rglob('*.py')):
            if '__pycache__' not in path.parts and path.is_file():
                provenance[str(path)] = digest(path)
    for name in ('global_bank.npy', 'image_feat.npy', 'text_feat.npy'):
        provenance[str(source / 'features' / name)] = digest(source / 'features' / name)
    atomic_json(ROOT / 'source_manifest.json', provenance)
    status('prepared', role_counts=manifest['role_counts'])


def available_gpu():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=15)
    rows = []
    for line in result.stdout.strip().splitlines():
        gpu, memory, utilization = [int(part.strip()) for part in line.split(',')]
        rows.append({'gpu': gpu, 'memory_used_mb': memory, 'utilization': utilization})
        if (gpu in SPEC['allowed_gpus'] and memory <= SPEC['gpu_max_used_mb_before_launch']
                and utilization <= SPEC['gpu_max_utilization_before_launch']):
            return gpu, rows
    return None, rows


def stop_owned(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)


def collect_summary():
    record = {'stage': SPEC['purpose'], 'updated_unix': time.time(), 'seeds': {},
              'official_test_evaluated': False, 'official_validation_evaluated': False,
              'interpretation': 'Internal screen, not proof of improvement or novelty.'}
    complete = 0
    for seed in SPEC['seeds']:
        entries = {}
        directories = [('base', ROOT / 'prepared' / str(seed))]
        directories += [(variant, ROOT / 'runs' / str(seed) / variant) for variant in SPEC['variants']]
        for variant, directory in directories:
            entry = {'status': 'pending'}
            for name in ('progress.json', 'result.json', 'failure.json'):
                path = directory / name
                if path.exists():
                    if name == 'failure.json':
                        entry = {'status': 'failed', 'error': json.loads(path.read_text())}
                    else:
                        entry = json.loads(path.read_text())
            if entry.get('status') == 'completed':
                complete += 1
            entries[variant] = entry
        record['seeds'][str(seed)] = entries
    record['completed_jobs'] = complete
    record['total_jobs'] = len(SPEC['seeds']) * (len(SPEC['variants']) + 1)
    record['all_complete'] = complete == record['total_jobs']
    aggregate = {}
    metrics = ('recall@10', 'ndcg@10', 'recall@20', 'ndcg@20', 'recall@200')
    for variant in SPEC['variants']:
        rows = [entries[variant] for entries in record['seeds'].values()
                if entries[variant].get('status') == 'completed']
        if not rows:
            continue
        aggregate[variant] = {'completed_seeds': len(rows), 'metrics': {}}
        for metric in metrics:
            values = np.array([row['dev_metrics'][metric] for row in rows])
            aggregate[variant]['metrics'][metric] = {'mean': float(values.mean()),
                'std': float(values.std(ddof=1)) if len(values) > 1 else None}
        for name in ('candidate_recovered_rate', 'candidate_displaced_rate', 'candidate_net_delta'):
            aggregate[variant][name] = float(np.mean([row[name] for row in rows]))
    independent_rows = [entries['independent'] for entries in record['seeds'].values()
                        if entries['independent'].get('status') == 'completed']
    for name in ('raw_base_baseline', 'pipeline_baseline', 'full_catalog_fusion_baseline'):
        if independent_rows:
            aggregate[name] = {'completed_seeds': len(independent_rows), 'metrics': {},
                               'candidate_net_delta': None}
            for metric in metrics:
                values = np.array([row[name][metric] for row in independent_rows])
                aggregate[name]['metrics'][metric] = {'mean': float(values.mean()),
                    'std': float(values.std(ddof=1)) if len(values) > 1 else None}
    record['aggregate'] = aggregate
    comparisons = {}
    for control in SPEC['variants']:
        if control == 'conditional':
            continue
        paired = [(entries['conditional'], entries[control]) for entries in record['seeds'].values()
                  if entries['conditional'].get('status') == 'completed'
                  and entries[control].get('status') == 'completed']
        if paired:
            comparisons[control] = {'paired_seeds': len(paired), 'metrics': {}}
            for metric in metrics:
                left = np.array([a['dev_metrics'][metric] for a, _ in paired])
                right = np.array([b['dev_metrics'][metric] for _, b in paired])
                comparisons[control]['metrics'][metric] = {
                    'conditional_minus_control': (left - right).tolist(),
                    'relative_mean_gain': float((left.mean() - right.mean()) / max(right.mean(), 1e-12))}
    record['conditional_vs_controls'] = comparisons
    fusion_pairs = [(entries['conditional']['dev_metrics'], entries['independent']['full_catalog_fusion_baseline'])
                    for entries in record['seeds'].values()
                    if entries['conditional'].get('status') == 'completed'
                    and entries['independent'].get('status') == 'completed']
    if fusion_pairs:
        comparisons['full_catalog_fusion'] = {'paired_seeds': len(fusion_pairs), 'metrics': {}}
        for metric in metrics:
            left = np.array([a[metric] for a, _ in fusion_pairs])
            right = np.array([b[metric] for _, b in fusion_pairs])
            comparisons['full_catalog_fusion']['metrics'][metric] = {
                'conditional_minus_control': (left - right).tolist(),
                'relative_mean_gain': float((left.mean() - right.mean()) / max(right.mean(), 1e-12))}
    record['decision'] = 'incomplete; no performance conclusion'
    if record['all_complete']:
        success = True
        for comparison in comparisons.values():
            for metric in ('recall@20', 'ndcg@20'):
                values = comparison['metrics'][metric]
                success &= (values['relative_mean_gain'] >= 0.02
                            and min(values['conditional_minus_control']) > 0)
            for metric in ('recall@10', 'ndcg@10'):
                success &= comparison['metrics'][metric]['relative_mean_gain'] >= 0
        record['decision'] = ('screen_passed; independent confirmation and broader baselines required'
                              if success else 'screen_not_passed; no reliable new mechanism established')
    atomic_json(ROOT / 'summary.json', record)
    lines = ['# Retrieval Screen Status', '',
             'Training-internal Baby screen. Official validation/test are not evaluated.', '',
             '| Seed | Job | Status |', '| --- | --- | --- |']
    for seed, entries in record['seeds'].items():
        for variant, entry in entries.items():
            lines.append('| %s | %s | %s |' % (seed, variant, entry.get('status', 'unknown')))
    lines += ['', '## Internal Development Metrics', '',
              '| Variant | Seeds | R20 | N20 | Candidate R200 | Net Candidate Gain |',
              '| --- | --- | --- | --- | --- | --- |']
    for variant, row in aggregate.items():
        values = row['metrics']
        net = row['candidate_net_delta']
        lines.append('| %s | %d/3 | %.8f | %.8f | %.8f | %s |' % (
            variant, row['completed_seeds'], values['recall@20']['mean'],
            values['ndcg@20']['mean'], values['recall@200']['mean'],
            '%+.8f' % net if net is not None else 'n/a'))
    lines += ['', record['decision'], '',
              'Primary comparisons are conditional vs hard/independent/shuffled controls with identical labels.',
              'Raw base has fewer training labels; gains over it alone are not an equal-label comparison.']
    lines += ['', 'Full metrics and selected configurations: summary.json and each result.json.',
              'Do not compare these internal metrics directly to old official-validation results.']
    temporary = ROOT / 'RESULTS.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    temporary.replace(ROOT / 'RESULTS.md')
    return record


def run_job(name, script, output, arguments, gpu_needed=True, limit=None):
    global ACTIVE
    gpu = None
    while gpu_needed:
        if time.monotonic() >= DEADLINE:
            raise TimeoutError('Pipeline deadline reached while waiting for GPU')
        gpu, resources = available_gpu()
        if gpu is not None:
            break
        status('waiting_for_gpu', next_job=name, resources=resources)
        collect_summary()
        time.sleep(20)
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-u', str(ROOT / script), *arguments]
    if gpu_needed:
        command.extend(['--gpu-id', str(gpu), '--output', str(output)])
    environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                       PYTHONUNBUFFERED='1')
    began = time.monotonic()
    timeout = min(limit or SPEC['max_run_seconds'], max(1, DEADLINE - began))
    with (output / 'stdout.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=str(ROOT), env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        ACTIVE = process
        atomic_json(output / 'launch.json', {'pid': process.pid, 'command': command, 'gpu': gpu,
                    'started_unix': time.time(), 'max_seconds': timeout})
        while process.poll() is None:
            status('running', job=name, child_pid=process.pid, gpu=gpu, output=str(output))
            collect_summary()
            if time.monotonic() - began > timeout:
                stop_owned(process)
                atomic_json(output / 'failure.json', {'error': 'run or pipeline time limit exceeded'})
                break
            time.sleep(15)
        ACTIVE = None
    if process.returncode:
        if not (output / 'failure.json').exists():
            atomic_json(output / 'failure.json', {'exit_code': process.returncode})
        return False
    if gpu_needed and not (output / 'result.json').exists():
        atomic_json(output / 'failure.json', {'error': 'Zero exit without result.json'})
        return False
    return True


def main():
    import fcntl
    with (ROOT / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / 'roles.npy').exists():
            raise RuntimeError('Refusing to silently restart an already prepared experiment')
        prepare()
        if not run_job('unit_tests', 'run_tests.py', ROOT / 'unit_tests', [], False, 180):
            raise RuntimeError('Unit tests failed')
        if not run_job('base_smoke', 'train_base.py', ROOT / 'smoke' / 'base',
                       ['--seed', str(SPEC['seeds'][0]), '--smoke'], limit=600):
            raise RuntimeError('Base smoke failed')
        for seed in SPEC['seeds']:
            if not run_job('base_s%d' % seed, 'train_base.py', ROOT / 'prepared' / str(seed),
                           ['--seed', str(seed)]):
                continue
            if seed == SPEC['seeds'][0]:
                for variant in SPEC['variants']:
                    if not run_job('smoke_' + variant, 'train_retrieval.py',
                        ROOT / 'smoke' / variant,
                        ['--seed', str(seed), '--variant', variant, '--smoke'], limit=600):
                        raise RuntimeError('Retrieval smoke failed: ' + variant)
            for variant in SPEC['variants']:
                if time.monotonic() >= DEADLINE:
                    raise TimeoutError('Pipeline wall-clock limit reached')
                successful = run_job('%s_s%d' % (variant, seed), 'train_retrieval.py',
                    ROOT / 'runs' / str(seed) / variant, ['--seed', str(seed), '--variant', variant])
                if variant == 'independent' and not successful:
                    break
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
