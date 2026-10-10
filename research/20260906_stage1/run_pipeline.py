"""Detached, bounded queue for feature extraction and nine stage-one runs."""

import collections
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

from stage1_core import atomic_json, internal_holdout, sha256


ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT / 'experiment.json').read_text())
STARTED = time.monotonic()
DEADLINE = STARTED + SPEC['max_pipeline_seconds']
ACTIVE = {}


def status(stage, **details):
    record = {'stage': stage, 'pid': os.getpid(), 'updated_unix': time.time(),
              'elapsed_seconds': time.monotonic() - STARTED, **details}
    atomic_json(ROOT / 'pipeline_status.json', record)
    print(json.dumps(record), flush=True)


def idle_gpus():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True,
                            check=True, timeout=15)
    available = []
    for line in result.stdout.strip().splitlines():
        index, memory, utilization = [int(value.strip()) for value in line.split(',')]
        if memory < 500 and utilization == 0 and index not in ACTIVE:
            available.append(index)
    return available


def wait_gpu():
    while time.monotonic() < DEADLINE:
        available = idle_gpus()
        if available:
            return available[0]
        status('waiting_for_idle_gpu')
        time.sleep(20)
    raise TimeoutError('No idle GPU became available within the pipeline limit')


def run_step(name, command, timeout):
    log_path = ROOT / (name + '.log')
    with log_path.open('x', encoding='utf-8') as log:
        status(name, command=command, log=str(log_path))
        subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=min(timeout, max(1, DEADLINE - time.monotonic())))


def prepare_split_and_snapshot():
    source = Path('/data/sunzixuan/bridge_official/src')
    snapshot = ROOT / 'bridge_src'
    snapshot.mkdir(exist_ok=False)
    provenance = {}
    # Only source/config files are needed; never copy checkpoints or old outputs.
    for path in source.rglob('*'):
        relative = path.relative_to(source)
        if any(part in ('saved_models', 'saved', 'logs', 'log', '__pycache__', 'recommend_topk', '.git')
               for part in relative.parts) or path.suffix not in ('.py', '.yaml', '.yml'):
            continue
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        provenance[str(relative)] = sha256(destination)
    atomic_json(ROOT / 'source_manifest.json', provenance)
    data_path = Path('/data/sunzixuan/expirement/LOBSTER/data/baby/baby.inter')
    original = pd.read_csv(data_path, sep='\t', usecols=['userID', 'itemID', 'x_label'])
    train = original.loc[original.x_label == 0, ['userID', 'itemID']].copy().reset_index(drop=True)
    if train.duplicated(['userID', 'itemID']).any():
        raise RuntimeError('Duplicate training pairs require explicit handling before leave-one-out')
    held = internal_holdout(train.userID.to_numpy(), SPEC['split_seed'])
    train['x_label'] = 0
    train.loc[held, 'x_label'] = 1
    support = train.loc[train.x_label == 0]
    dev = train.loc[train.x_label == 1]
    if not set(dev.userID).issubset(set(support.userID)):
        raise RuntimeError('Internal split created a user with no history')
    directory = ROOT / 'split'
    directory.mkdir(exist_ok=False)
    train.to_csv(directory / 'baby.inter', sep='\t', index=False)
    np.save(directory / 'held_training_row_indices.npy', held)
    manifest = {'source_inter_sha256': sha256(data_path),
                'split_sha256': sha256(directory / 'baby.inter'), 'split_seed': SPEC['split_seed'],
                'source_training_interactions': len(train), 'support_interactions': len(support),
                'development_interactions': len(dev), 'development_users': int(dev.userID.nunique()),
                'items': len(np.load(ROOT / 'features/image_feat.npy', mmap_mode='r')),
                'users': int(train.userID.max()) + 1,
                'development_items_absent_in_support': int((~dev.itemID.isin(support.itemID)).sum()),
                'official_validation_included': False, 'official_test_included': False,
                'policy_training': False, 'selection_metric': 'internal development NDCG@20'}
    atomic_json(ROOT / 'split_manifest.json', manifest)


def summary():
    results = {}
    all_complete = True
    for variant in SPEC['variants']:
        results[variant] = []
        for seed in SPEC['seeds']:
            directory = ROOT / 'runs' / ('%s_s%d' % (variant, seed))
            result_path = directory / 'result.json'
            if result_path.exists():
                record = json.loads(result_path.read_text())
                record['seed'] = seed
            else:
                all_complete = False
                record = {'seed': seed, 'status': 'pending_or_running'}
                if (directory / 'failure.json').exists():
                    record = {'seed': seed, 'status': 'failed',
                              'failure': json.loads((directory / 'failure.json').read_text())}
            if record.get('status') != 'completed':
                all_complete = False
            results[variant].append(record)
    aggregate = {}
    for variant, records in results.items():
        completed = [row for row in records if row.get('status') == 'completed']
        if completed:
            aggregate[variant] = {'completed_seeds': len(completed), 'metrics': {}}
            for metric in ('recall@10', 'ndcg@10', 'recall@20', 'ndcg@20'):
                values = np.array([row['best']['metrics'][metric] for row in completed])
                aggregate[variant]['metrics'][metric] = {'mean': float(values.mean()),
                    'std': float(values.std(ddof=1)) if len(values) > 1 else None}
    decision = 'incomplete; do not interpret as a performance conclusion'
    comparisons = {}
    if all_complete:
        passes = True
        for metric in ('recall@20', 'ndcg@20'):
            baseline = np.array([row['best']['metrics'][metric] for row in results['global']])
            local = np.array([row['best']['metrics'][metric] for row in results['local']])
            shuffled = np.array([row['best']['metrics'][metric] for row in results['shuffled']])
            relative = float((local.mean() - baseline.mean()) / max(baseline.mean(), 1e-12))
            comparisons[metric] = {'local_vs_global_relative_gain': relative,
                'paired_seed_differences': (local - baseline).tolist(),
                'local_vs_shuffled_difference': float((local - shuffled).mean())}
            passes &= relative >= SPEC['success_relative_gain'] and np.all(local > baseline) and local.mean() > shuffled.mean()
        for metric in ('recall@10', 'ndcg@10'):
            passes &= aggregate['local']['metrics'][metric]['mean'] >= aggregate['global']['metrics'][metric]['mean']
        decision = ('screen_passed; independent confirmation still required' if passes else
                    'screen_not_passed; do not claim a reliable improvement or launch the full policy automatically')
    record = {'stage': 'information_screen_not_full_RAVE', 'all_complete': all_complete,
              'updated_unix': time.time(), 'results': results, 'aggregate': aggregate,
              'comparisons': comparisons, 'decision': decision,
              'evaluation': 'training-internal development set, also used for epoch selection',
              'not_publication_evidence': True, 'official_test_evaluated': False}
    atomic_json(ROOT / 'summary.json', record)
    lines = ['# Stage-One Results', '', 'This is a development screen, not a final RAVE result.', '',
             '| Variant | Seeds Complete | Recall@20 | NDCG@20 |', '| --- | --- | --- | --- |']
    for variant in SPEC['variants']:
        row = aggregate.get(variant)
        if row:
            lines.append('| %s | %d/3 | %.8f | %.8f |' % (variant, row['completed_seeds'],
                         row['metrics']['recall@20']['mean'], row['metrics']['ndcg@20']['mean']))
        else:
            lines.append('| %s | 0/3 | pending | pending |' % variant)
    lines.extend(['', decision, '', 'Errors and per-run status are recorded in summary.json and runs/.'])
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return all_complete


def stop_owned_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=20)


def train_queue():
    queue = collections.deque((variant, seed) for seed in SPEC['seeds'] for variant in SPEC['variants'])
    while queue or ACTIVE:
        now = time.monotonic()
        if now > DEADLINE:
            raise TimeoutError('Pipeline exceeded configured elapsed-time limit')
        for gpu, job in list(ACTIVE.items()):
            process = job['process']
            code = process.poll()
            if code is None and now - job['started'] > SPEC['max_run_seconds']:
                stop_owned_process(process)
                atomic_json(job['directory'] / 'failure.json', {'error': 'run time limit exceeded'})
                code = process.returncode
            if code is not None:
                job['log'].close()
                if code and not (job['directory'] / 'failure.json').exists():
                    atomic_json(job['directory'] / 'failure.json', {'exit_code': code})
                elif code == 0 and not (job['directory'] / 'result.json').exists():
                    atomic_json(job['directory'] / 'failure.json', {'error': 'process exited without result'})
                del ACTIVE[gpu]
        if queue:
            for gpu in idle_gpus():
                if not queue:
                    break
                variant, seed = queue.popleft()
                directory = ROOT / 'runs' / ('%s_s%d' % (variant, seed))
                directory.mkdir(parents=True, exist_ok=False)
                command = [sys.executable, '-u', str(ROOT / 'train_stage1.py'), '--variant', variant,
                           '--seed', str(seed), '--gpu-id', str(gpu), '--output', str(directory)]
                log = (directory / 'stdout.log').open('x', encoding='utf-8')
                process = subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
                atomic_json(directory / 'launch.json', {'pid': process.pid, 'gpu': gpu, 'command': command})
                ACTIVE[gpu] = {'process': process, 'started': time.monotonic(), 'directory': directory, 'log': log}
        summary()
        status('training', queued=list(queue), active=[{'gpu': gpu, 'pid': job['process'].pid,
               'run': job['directory'].name} for gpu, job in ACTIVE.items()])
        if queue or ACTIVE:
            time.sleep(20)


def main():
    import fcntl
    with (ROOT / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (ROOT / 'assets_ready.json').exists():
            if (ROOT / 'assets_failed.json').exists():
                raise RuntimeError((ROOT / 'assets_failed.json').read_text())
            if time.monotonic() > DEADLINE:
                raise TimeoutError('Preparation did not complete in time')
            status('waiting_for_assets')
            time.sleep(20)
        run_step('unit_tests', [sys.executable, '-m', 'unittest', '-v', 'test_stage1.py'], 180)
        gpu = wait_gpu()
        run_step('extract_features', [sys.executable, '-u', str(ROOT / 'extract_features.py'),
                                     '--gpu-id', str(gpu)], 3600)
        prepare_split_and_snapshot()
        for variant in SPEC['variants']:
            gpu = wait_gpu()
            run_step('smoke_' + variant, [sys.executable, '-u', str(ROOT / 'train_stage1.py'),
                '--variant', variant, '--seed', '2026', '--gpu-id', str(gpu),
                '--output', str(ROOT / 'smoke' / variant), '--smoke'], 300)
        train_queue()
        complete = summary()
        status('completed' if complete else 'finished_with_failed_runs')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=str(exc), traceback=traceback.format_exc())
        summary()
        raise
    finally:
        for job in ACTIVE.values():
            stop_owned_process(job['process'])
            job['log'].close()
