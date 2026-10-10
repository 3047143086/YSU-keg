"""Single-use v3 queue requiring independent evidence and explicit root approval."""
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from prerequisite_gate import PrerequisiteError, require_prerequisites, require_root_review, sha, verify_frozen
from recovery_protocol import JOB_SECONDS, QUEUE_SECONDS, SEEDS, run_command, validate_result

ROOT = Path(__file__).resolve().parent
STATE = {'pid': os.getpid(), 'status': 'preparing', 'jobs': [], 'completed_seeds': [],
         'seed_states': {str(seed): 'not_started' for seed in SEEDS}, 'test_evaluated': False,
         'test_loader_constructed': False, 'eval_cache_integrated': False,
         'wall_time_budget_changed_from_v1': True, 'job_seconds': JOB_SECONDS,
         'queue_seconds': QUEUE_SECONDS, 'backward_implementation_changed': True}
DEADLINE = None


def atomic_json(path, row):
    path = Path(path)
    pending = path.with_name(path.name + '.pending.' + str(os.getpid()))
    with pending.open('x', encoding='utf-8') as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def update(**fields):
    STATE.update(fields, updated_unix=time.time())
    atomic_json(ROOT / 'pipeline_status.json', STATE)
    print(json.dumps(fields, allow_nan=False), flush=True)


def remaining():
    value = DEADLINE - time.monotonic()
    if value <= 0:
        raise TimeoutError('72-hour queue budget exhausted, including waits')
    return value


def exact_process(pid, command):
    try:
        actual = Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0')
        return actual[:-1] == [str(v).encode() for v in command]
    except FileNotFoundError:
        return False


def resources():
    gpu = subprocess.run(['nvidia-smi', '-i', '1', '--query-gpu=uuid,memory.free',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                           '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
    fields = [part.strip() for part in gpu.stdout.strip().split(',')]
    valid = gpu.returncode == apps.returncode == 0 and len(fields) == 2 and fields[1].isdigit()
    active = []
    if valid:
        for line in apps.stdout.splitlines():
            parts = [part.strip() for part in line.split(',')]
            if len(parts) == 2 and parts[1] == fields[0]:
                pid = int(parts[0]) if parts[0].isdigit() else None
                try:
                    command = Path('/proc', str(pid), 'cmdline').read_bytes().decode(errors='replace').split('\0')[:-1]
                except FileNotFoundError:
                    command = []
                active.append({'pid': pid, 'command': command})
    ram = next((int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:')), 0)
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    return {'physical_gpu': 1, 'gpu_free_bytes': int(fields[1]) * 1024 ** 2 if valid else None,
            'gpu1_compute_processes': active, 'query_succeeded': valid, 'ram_available_bytes': ram,
            'available_cpu_count': cpus, 'cpu_threads': 4, 'cuda_memory_fraction': .55}


def wait_resources():
    while True:
        remaining()
        # Every wait refreshes identities; no unrelated process is terminated.
        row = resources()
        if (row['query_succeeded'] and not row['gpu1_compute_processes']
                and row['gpu_free_bytes'] >= 14 * 1024 ** 3
                and row['ram_available_bytes'] >= 12 * 1024 ** 3
                and row['available_cpu_count'] >= 4):
            return row
        update(stage='waiting_for_exclusive_GPU1_RAM_CPU', resources=row)
        time.sleep(min(30, remaining()))


def sample(pid):
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
    mib = None
    for line in gpu.stdout.splitlines() if gpu.returncode == 0 else ():
        fields = [part.strip() for part in line.split(',')]
        if len(fields) == 2 and fields[0] == str(pid) and fields[1].isdigit():
            mib = int(fields[1])
    try:
        rss = next((int(line.split()[1]) for line in Path('/proc', str(pid), 'status').read_text().splitlines()
                    if line.startswith('VmHWM:')), None)
    except FileNotFoundError:
        rss = None
    return mib, rss


def execute(seed):
    timeout = min(JOB_SECONDS, remaining())
    command = run_command(ROOT, sys.executable, seed)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4', PYTHONUNBUFFERED='1',
                       PYTHONPATH=str(ROOT / 'carrier/src') + os.pathsep + str(ROOT))
    started, peak_gpu, peak_rss = time.monotonic(), None, None
    with (ROOT / 'logs' / ('seed' + str(seed) + '.log')).open('xb') as log:
        # Large prerequisite/source hashes finish first; refresh directly before spawn.
        row = wait_resources()
        timeout = min(JOB_SECONDS, remaining())
        update(resources=row)
        process = subprocess.Popen(command, cwd=str(ROOT), env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        STATE['seed_states'][str(seed)] = 'running'
        update(stage='training', current_seed=seed,
               active_job={'pid': process.pid, 'command': command, 'timeout_seconds': timeout})
        try:
            next_sample = 0
            while process.poll() is None:
                left = min(timeout - (time.monotonic() - started), remaining())
                if left <= 0:
                    raise TimeoutError('Seed job wall-time limit exhausted')
                if time.monotonic() >= next_sample:
                    if not exact_process(process.pid, command) and process.poll() is None:
                        raise RuntimeError('Owned training process command changed')
                    gpu, rss = sample(process.pid)
                    if gpu is not None:
                        peak_gpu = max(peak_gpu or 0, gpu)
                    if rss is not None:
                        peak_rss = max(peak_rss or 0, rss)
                    update(active_job={'pid': process.pid, 'command': command,
                           'elapsed_seconds': time.monotonic() - started,
                           'peak_sampled_gpu_mib': peak_gpu, 'peak_host_rss_kib': peak_rss})
                    next_sample = time.monotonic() + 30
                try:
                    process.wait(timeout=min(10, left))
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            if process.poll() is None:
                if not exact_process(process.pid, command):
                    raise RuntimeError('Refuse to signal a process whose command identity changed')
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if not exact_process(process.pid, command):
                        raise RuntimeError('Refuse to SIGKILL a changed process identity')
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            raise
        finally:
            STATE['jobs'].append({'seed': seed, 'pid': process.pid, 'command': command,
                                  'exit_code': process.poll(), 'elapsed_seconds': time.monotonic() - started,
                                  'peak_sampled_gpu_mib': peak_gpu, 'peak_host_rss_kib': peak_rss})
            update(active_job=None)
    if process.returncode:
        raise RuntimeError('Seed failed; queue stops without retry or fallback')


def write_results():
    lines = ['seed\tstatus\trecall@10\tndcg@10\trecall@20\tndcg@20\tepochs\tseconds']
    for seed in SEEDS:
        output = ROOT / 'runs' / ('elec_projector_recovery_s' + str(seed))
        state = STATE['seed_states'][str(seed)]
        path = output / 'result.json'
        if state == 'completed' and path.is_file():
            row = json.loads(path.read_text())
            validate_result(row, seed)
            fields = [str(seed), state] + [str(row['metrics'][key]) for key in
                      ('recall@10', 'ndcg@10', 'recall@20', 'ndcg@20')]
            fields += [str(row['epochs_completed']), str(row['elapsed_seconds'])]
        else:
            fields = [str(seed), state] + ['n/a'] * 6
        lines.append('\t'.join(fields))
    pending = ROOT / ('results.tsv.pending.' + str(os.getpid()))
    with pending.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    pending.replace(ROOT / 'results.tsv')


def main():
    global DEADLINE
    # Acquiring the Linux lock is local to this queue; it does not touch GPU0.
    import fcntl
    with (ROOT / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_root_review(ROOT, 'start')
        require_prerequisites(ROOT)
        verify_frozen(ROOT)
        tests = json.loads((ROOT / 'test_receipt.json').read_text())
        if (tests.get('status') != 'passed' or tests.get('skipped') != 0
                or tests.get('deployment_sha256') != sha(ROOT / 'deployment.json')):
            raise PrerequisiteError('Actual MMRec tests with no skips must pass for this deployment')
        with (ROOT / 'execution.reservation.json').open('x') as ticket:
            json.dump({'pid': os.getpid(), 'unix': time.time(), 'single_use': True}, ticket)
        DEADLINE = time.monotonic() + QUEUE_SECONDS
        update(status='running', started_unix=time.time(), seed_order=list(SEEDS))
        (ROOT / 'logs').mkdir(exist_ok=False)
        (ROOT / 'runs').mkdir(exist_ok=False)
        for seed in SEEDS:
            require_prerequisites(ROOT)
            verify_frozen(ROOT)
            row = wait_resources()
            require_prerequisites(ROOT)
            verify_frozen(ROOT)
            update(resources=row)
            execute(seed)
            output = ROOT / 'runs' / ('elec_projector_recovery_s' + str(seed))
            validate_result(json.loads((output / 'result.json').read_text()), seed)
            guard = json.loads((output / 'post_run_guard.json').read_text())
            numeric = json.loads((output / 'numerics_receipt.json').read_text())
            if (guard.get('all_match') is not True or numeric.get('nonfinite_steps_skipped') != 0
                    or numeric.get('backward_implementation_changed') is not True
                    or numeric.get('eval_cache_integrated') is not False):
                raise RuntimeError('Job numerical/protected-state receipt failed')
            verify_frozen(ROOT)
            STATE['seed_states'][str(seed)] = 'completed'
            STATE['completed_seeds'].append(seed)
            update(last_completed_seed=seed)
            write_results()
        update(status='completed', finished_unix=time.time())


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        state = 'not_started_prerequisite_failure' if isinstance(error, PrerequisiteError) else (
            'budget_exhausted' if isinstance(error, TimeoutError) else 'technical_failure')
        seed = STATE.get('current_seed')
        if seed is not None and STATE['seed_states'][str(seed)] == 'running':
            STATE['seed_states'][str(seed)] = 'incomplete_timeout' if isinstance(error, TimeoutError) else 'crash'
        update(status=state, error=repr(error), traceback=traceback.format_exc(), finished_unix=time.time())
        raise
    finally:
        write_results()
