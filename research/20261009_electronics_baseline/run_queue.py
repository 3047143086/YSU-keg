"""Bounded five-seed original Electronics queue behind the residual diagnostic."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREVIOUS = Path('/data/sunzixuan/review_residual_20261009_v1')
SEEDS = tuple(range(2026, 2031))
QUEUE_SECONDS = 72 * 3600
JOB_SECONDS = 6 * 3600
STATE = {'pid': os.getpid(), 'status': 'starting', 'jobs': [], 'completed_seeds': []}
DEADLINE = None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def update(**fields):
    STATE.update(fields, updated_unix=time.time())
    temporary = ROOT / ('pipeline_status.pending.' + str(os.getpid()) + '.json')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(STATE, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(ROOT / 'pipeline_status.json')
    print(json.dumps(fields, allow_nan=False), flush=True)


def time_remaining():
    remaining = DEADLINE - time.time()
    if remaining <= 0:
        raise TimeoutError('Electronics queue exceeded its 72-hour budget')
    return remaining


def process_matches(pid, script):
    try:
        return str(script).encode() in Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return False


def wait_for_predecessor():
    while True:
        time_remaining()
        launch_path = PREVIOUS / 'launch.json'
        diagnosis_path = PREVIOUS / 'diagnosis.json'
        if launch_path.is_file():
            launch = json.loads(launch_path.read_text())
            alive = process_matches(launch['pid'], PREVIOUS / 'experiment.py')
            if not alive and diagnosis_path.is_file():
                diagnosis = json.loads(diagnosis_path.read_text())
                if diagnosis.get('status') != 'completed':
                    raise RuntimeError('Residual predecessor technical failure: ' + str(diagnosis.get('status')))
                row = {'root': str(PREVIOUS), 'pid': launch['pid'], 'process_alive': False,
                       'status': diagnosis['status'], 'launch_sha256': sha256(launch_path),
                       'diagnosis_sha256': sha256(diagnosis_path), 'metrics_used_for_baseline': False}
                update(predecessor=row)
                return
            update(stage='waiting_for_residual_predecessor', predecessor={'root': str(PREVIOUS),
                   'pid': launch['pid'], 'process_alive': alive, 'final_receipt_exists': diagnosis_path.is_file()})
        else:
            update(stage='waiting_for_residual_launch_receipt', predecessor={'root': str(PREVIOUS),
                   'launch_receipt_exists': False})
        time.sleep(min(30, time_remaining()))


def available_ram():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return 0


def resources():
    query = subprocess.run(['nvidia-smi', '-i', '1', '--query-gpu=memory.free',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
    free = int(query.stdout.strip()) * 1024 ** 2 if query.returncode == 0 and query.stdout.strip().isdigit() else None
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    return {'gpu_physical_index': 1, 'gpu_free_bytes': free, 'ram_available_bytes': available_ram(),
            'available_cpu_count': cpus, 'cuda_memory_fraction': .55, 'cpu_threads': 4}


def wait_for_resources():
    while True:
        time_remaining()
        row = resources()
        if (row['gpu_free_bytes'] is not None and row['gpu_free_bytes'] >= 14 * 1024 ** 3
                and row['ram_available_bytes'] >= 12 * 1024 ** 3
                and row['available_cpu_count'] is not None and row['available_cpu_count'] >= 4):
            update(resources=row)
            return
        update(stage='waiting_for_GPU1_RAM_CPU', resources=row)
        time.sleep(min(30, time_remaining()))


def verify_frozen():
    deployment = json.loads((ROOT / 'deployment.json').read_text())
    for name, digest in deployment['bundle_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError('Bundle changed: ' + name)
    for name, digest in deployment['carrier_sha256'].items():
        if sha256(ROOT / 'carrier/src' / name) != digest:
            raise RuntimeError('Carrier changed: ' + name)
    audit = json.loads((ROOT / 'input_audit.json').read_text())
    for row in audit['original_inputs'].values():
        if sha256(row['path']) != row['sha256']:
            raise RuntimeError('Original Electronics input changed: ' + row['path'])
    return audit


def sample_gpu(pid):
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
    values = []
    for line in query.stdout.splitlines() if query.returncode == 0 else ():
        fields = [value.strip() for value in line.split(',')]
        if len(fields) == 2 and fields[0] == str(pid) and fields[1].isdigit():
            values.append(int(fields[1]))
    return sum(values) if values else None


def execute(name, command, gpu):
    timeout = min(JOB_SECONDS, time_remaining())
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='1' if gpu else '', OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4',
                       PYTHONPATH=str(ROOT / 'carrier/src') + os.pathsep + str(ROOT), PYTHONUNBUFFERED='1')
    started, process, peak = time.time(), None, None
    log_path = ROOT / 'logs' / (name + '.log')
    with log_path.open('xb') as log:
        process = subprocess.Popen(command, cwd=str(ROOT), env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        update(stage=name, active_job={'name': name, 'pid': process.pid, 'command': command,
               'started_unix': started, 'timeout_seconds': timeout, 'gpu_used': gpu})
        try:
            next_sample = 0
            while process.poll() is None:
                remaining = started + timeout - time.time()
                if remaining <= 0:
                    raise TimeoutError(name + ' exceeded its six-hour job budget')
                if time.time() >= next_sample:
                    current = sample_gpu(process.pid) if gpu else None
                    if current is not None:
                        peak = max(peak or 0, current)
                    update(active_job={'name': name, 'pid': process.pid, 'command': command,
                           'elapsed_seconds': time.time() - started, 'gpu_used': gpu,
                           'peak_sampled_gpu_memory_mib': peak})
                    next_sample = time.time() + 30
                try:
                    process.wait(timeout=min(10, remaining))
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            raise
        finally:
            STATE['jobs'].append({'name': name, 'pid': process.pid, 'exit_code': process.poll(),
                                  'elapsed_seconds': time.time() - started, 'gpu_used': gpu,
                                  'peak_sampled_gpu_memory_mib': peak, 'log_path': str(log_path)})
            update(active_job=None)
    if process.returncode:
        raise RuntimeError(name + ' failed; see ' + str(log_path))


def results_tsv():
    lines = ['seed\tstatus\trecall@10\tndcg@10\trecall@20\tndcg@20\tepochs\tseconds']
    for seed in SEEDS:
        path = ROOT / 'runs' / ('elec_baseline_s' + str(seed)) / 'result.json'
        if path.is_file():
            row = json.loads(path.read_text())
            values = [str(seed), row['status']] + [str(row['metrics'][name]) for name in
                      ('recall@10', 'ndcg@10', 'recall@20', 'ndcg@20')]
            values += [str(row['epochs_completed']), str(row['elapsed_seconds'])]
        else:
            status = 'crash' if STATE.get('status') == 'technical_failure' and STATE.get('current_seed') == seed else 'pending'
            values = [str(seed), status] + ['n/a'] * 6
        lines.append('\t'.join(values))
    temporary = ROOT / ('results.tsv.pending.' + str(os.getpid()))
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    temporary.replace(ROOT / 'results.tsv')


def main():
    global DEADLINE
    started = time.time()
    DEADLINE = started + QUEUE_SECONDS
    update(status='running', started_unix=started, deadline_unix=DEADLINE,
           time_budget_seconds=QUEUE_SECONDS, seed_order=list(SEEDS), test_evaluated=False)
    (ROOT / 'logs').mkdir(exist_ok=False)
    (ROOT / 'runs').mkdir(exist_ok=False)
    test = json.loads((ROOT / 'test_receipt.json').read_text())
    if test['exit_code'] or test['deployment_sha256'] != sha256(ROOT / 'deployment.json'):
        raise RuntimeError('Passing tests for the current bundle are required')
    audit = verify_frozen()
    wait_for_predecessor()
    wait_for_resources()
    data = ROOT / 'carrier/data/elec'
    execute('prepare_original_mm_graph', [sys.executable, '-u', str(ROOT / 'graph_prepare.py'),
            '--data-root', str(data), '--output', str(data / 'mm_adj_10.pt'),
            '--carrier-root', str(ROOT / 'carrier'), '--source-graph-hash',
            audit['carrier_files']['models/dual_frequency_encoder.py']['sha256']], gpu=False)
    graph = json.loads((data / 'graph_manifest.json').read_text())
    update(graph_manifest=graph)
    for seed in SEEDS:
        verify_frozen()
        if sha256(data / 'mm_adj_10.pt') != graph['graph_sha256']:
            raise RuntimeError('Frozen Electronics graph changed')
        wait_for_resources()
        update(current_seed=seed)
        output = ROOT / 'runs' / ('elec_baseline_s' + str(seed))
        command = [sys.executable, '-u', str(ROOT / 'guarded_train.py'), '--bridge-root', str(ROOT / 'carrier'),
                   '--dataset', 'elec', '--seed', str(seed), '--output', str(output),
                   '--device', 'cuda', '--gpu-id', '1', '--cuda-memory-fraction', '.55']
        execute('elec_baseline_s' + str(seed), command, gpu=True)
        row = json.loads((output / 'result.json').read_text())
        if (row['status'] != 'completed' or row['seed'] != seed or row['dataset'] != 'elec'
                or row['test_evaluated'] or row['test_loader_constructed'] or not row['inputs_unchanged']):
            raise RuntimeError('Unexpected Electronics baseline result protocol')
        STATE['completed_seeds'].append(seed)
        update(last_completed_result=row)
        results_tsv()
    verify_frozen()
    update(status='completed', stage='done', finished_unix=time.time(),
           source_revision=json.loads((ROOT / 'source_revision.json').read_text())['commit'])


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        update(status='technical_failure', error=repr(error), traceback=traceback.format_exc(), finished_unix=time.time())
        raise
    finally:
        results_tsv()
