"""One bounded candidate-negative gate hypothesis with a joint-loss control."""
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from lineage import PREVIOUS, predecessor_state, reuse
from reporting import decide, finish

ROOT = Path(__file__).resolve().parent
DEADLINE = time.time() + 48 * 3600
STATE = {'started_unix': time.time(), 'status': 'running', 'jobs': [], 'screens': {}}
OUTPUTS = {}


def update(**fields):
    STATE.update(fields, updated_unix=time.time())
    pending = ROOT / 'pipeline_status.pending.json'
    pending.write_text(json.dumps(STATE, indent=2, allow_nan=False), encoding='utf-8')
    pending.replace(ROOT / 'pipeline_status.json')
    if fields:
        print(json.dumps(fields, allow_nan=False), flush=True)


def sample_memory(pid):
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                             '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=15)
    values = []
    if not result.returncode:
        for line in result.stdout.splitlines():
            fields = [value.strip() for value in line.split(',')]
            if len(fields) == 2 and fields[0] == str(pid) and fields[1].isdigit():
                values.append(int(fields[1]))
    return sum(values) if values else None


def terminate_own_child(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def execute(name, command, timeout):
    remaining = DEADLINE - time.time()
    if remaining <= 0:
        raise TimeoutError('Pipeline reached its 48-hour budget')
    update(stage=name, command=command)
    started, peak, process = time.monotonic(), None, None
    expires = started + min(timeout, remaining)
    log_path = ROOT / 'logs' / (name + '.log')
    with log_path.open('xb') as log:
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True,
                                   env=dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                                            OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1',
                                            PYTHONPATH=str(ROOT / 'carrier/src') + os.pathsep + str(ROOT)))
        next_sample = started
        try:
            while process.poll() is None:
                remaining = expires - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(name + ' exceeded its declared timeout')
                if time.monotonic() >= next_sample:
                    current = sample_memory(process.pid)
                    if current is not None:
                        peak = current if peak is None else max(peak, current)
                    update(active_job={'name': name, 'pid': process.pid,
                                       'elapsed_seconds': time.monotonic() - started,
                                       'peak_sampled_gpu_memory_mib': peak})
                    next_sample = time.monotonic() + 30
                try:
                    process.wait(timeout=min(10, remaining))
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            terminate_own_child(process)
            raise
        finally:
            STATE['jobs'].append({'name': name, 'exit_code': process.poll(),
                                  'seconds': time.monotonic() - started, 'log': str(log_path),
                                  'peak_sampled_gpu_memory_mib': peak})
            update(active_job=None)
    if process.returncode:
        raise RuntimeError(name + ' failed; see ' + str(log_path))


def wait_for_predecessor():
    while time.time() < DEADLINE:
        state, alive = predecessor_state()
        if not alive:
            if state.get('status') != 'closed_validation_gate':
                raise RuntimeError('Unexpected predecessor outcome: ' + str(state.get('status')))
            update(predecessor={'root': str(PREVIOUS), 'status': state['status'], 'process_alive': False})
            return
        update(stage='waiting_for_previous_queue',
               predecessor={'root': str(PREVIOUS), 'status': state.get('status'),
                            'stage': state.get('stage'), 'process_alive': True})
        time.sleep(30)
    raise TimeoutError('Previous queue did not finish within the 48-hour budget')


def available_gpu():
    while time.time() < DEADLINE:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free',
                                 '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=20)
        candidates = []
        if not result.returncode:
            for line in result.stdout.strip().splitlines():
                index, free = [int(value.strip()) for value in line.split(',')]
                if free >= 14336:
                    candidates.append((free, index))
        if candidates:
            return max(candidates)[1]
        update(stage='waiting_for_14GiB_free_GPU', gpu=result.stdout.strip())
        time.sleep(30)
    raise TimeoutError('GPU wait exceeded the pipeline budget')


def mechanism_gate(dataset, output):
    row = json.loads((output / 'progress.jsonl').read_text().splitlines()[-1])
    values = row['train_diagnostics']
    # This training-only check tests supervision coverage, not performance.
    passed = (values['random_negative_in_scope_rate'] <= .10
              and values['eligible_pair_rate'] >= .02
              and values['mean_valid_candidate_negatives_eligible'] >= 20)
    record = {'dataset': dataset, 'passed': passed, 'train_diagnostics': values,
              'validation_metrics_used_for_mechanism_gate': False, 'test_evaluated': False,
              'criteria': {'max_random_negative_in_scope_rate': .10,
                           'min_eligible_pair_rate': .02, 'min_mean_valid_negatives': 20}}
    with (ROOT / (dataset + '_mechanism.json')).open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    update(mechanism=record)
    return passed


def train(dataset, arm, smoke=False):
    name = '{}_{}_{}'.format(dataset, arm, 'smoke' if smoke else 's2026')
    output = ROOT / ('smoke' if smoke else 'runs') / name
    command = [sys.executable, '-u', str(ROOT / 'guarded_train.py'), '--bridge-root', str(ROOT / 'carrier'),
               '--dataset', dataset, '--seed', '2026', '--output', str(output),
               '--device', 'cuda', '--gpu-id', str(available_gpu()),
               '--plugin-module', 'proposal', '--plugin-class', 'CandidateGateBRIDGE',
               '--plugin-config', str(ROOT / 'configs' / (dataset + '_' + arm + '.json'))]
    if smoke:
        command.append('--smoke')
    execute(name, command, 1800 if smoke else 8 * 3600)
    return output


def screen(dataset, outputs):
    import numpy as np
    rows = {arm: json.loads((path / 'result.json').read_text()) for arm, path in outputs.items()}
    metrics = {arm: row['metrics'] for arm, row in rows.items()}
    decision = decide(metrics)
    if decision['status'] not in ('keep', 'discard'):
        raise RuntimeError('A complete comparison is required for screening')
    bootstrap = {}
    with np.load(outputs['candidate'] / 'per_user_validation.npz') as candidate:
        for control in ('baseline', 'joint'):
            with np.load(outputs[control] / 'per_user_validation.npz') as reference:
                if not all(np.array_equal(candidate[field], reference[field])
                           for field in ('user_ids', 'positive_lengths')):
                    raise RuntimeError('Paired validation users differ')
                for key in ('recall_10', 'ndcg_10', 'recall_20', 'ndcg_20'):
                    delta = candidate[key] - reference[key]
                    rng = np.random.default_rng(2026)
                    draws = [delta[rng.integers(len(delta), size=len(delta))].mean()
                             for _ in range(2000)]
                    bootstrap[control + ':' + key] = {
                        'mean_delta': float(delta.mean()),
                        'user_bootstrap_95ci': np.quantile(draws, [.025, .975]).tolist()}
    record = dict(decision, dataset=dataset, passed=decision['status'] == 'keep', seed=2026,
                  metrics=metrics, paired_exploratory_not_independent_confirmation=bootstrap,
                  test_evaluated=False)
    with (ROOT / (dataset + '_screen.json')).open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    STATE['screens'][dataset] = record
    update()
    return record['passed']


def main():
    (ROOT / 'logs').mkdir(exist_ok=False)
    (ROOT / 'configs').mkdir(exist_ok=False)
    if json.loads((ROOT / 'test_receipt.json').read_text())['exit_code'] != 0:
        raise RuntimeError('Passing actual-environment tests are required')
    wait_for_predecessor()
    for dataset in ('baby', 'sports'):
        update(stage=dataset + '_verifying_baseline')
        baseline = reuse(ROOT, dataset)
        OUTPUTS[dataset] = {'baseline': baseline}
        for arm in ('candidate', 'joint'):
            config = {'gate_supervision': arm}
            with (ROOT / 'configs' / (dataset + '_' + arm + '.json')).open('x', encoding='utf-8') as stream:
                json.dump(config, stream, indent=2, allow_nan=False)
            smoke = train(dataset, arm, smoke=True)
            if arm == 'candidate' and not mechanism_gate(dataset, smoke):
                update(status='closed_mechanism_gate', stage=dataset, finished_unix=time.time())
                return
        for arm in ('candidate', 'joint'):
            OUTPUTS[dataset][arm] = train(dataset, arm)
            finish(ROOT, STATE, OUTPUTS, revision())
        if not screen(dataset, OUTPUTS[dataset]):
            update(status='closed_validation_gate', stage=dataset, finished_unix=time.time())
            return
    update(status='pilot_passed_needs_multiseed_and_multimodal_attribution',
           stage='done', finished_unix=time.time())


def revision():
    return json.loads((ROOT / 'source_revision.json').read_text())['commit']


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        update(status='technical_failure', error=str(error), traceback=traceback.format_exc(),
               finished_unix=time.time())
        raise
    finally:
        finish(ROOT, STATE, OUTPUTS, revision())
