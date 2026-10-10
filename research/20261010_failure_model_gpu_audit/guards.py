"""Hash and resource guards without importing Torch or initializing CUDA."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

BASELINE = Path('/data/sunzixuan/electronics_baseline_20261009_v1')
RUN = BASELINE / 'runs/elec_baseline_s2026'
CPU_AUDIT = Path('/data/sunzixuan/spectral_projector_audit_20261010_v1')
CRITICAL = {
    'deployment.json': '58658242005fb07a58f7b677732bdea65ea0aab1feeed191d5310ad939f421d8',
    'input_audit.json': '109caa06a4eb24526438bda5d71ffc51e1d9d7d4c40d75525daa5ea32000c5d2',
    'source_revision.json': 'b4b9ec9c645ad98a7ec8913bc7976d2277d993a076658194f81a8fef16bde201',
    'runs/elec_baseline_s2026/inputs.json': '18dd014949737077a3ce63e004e2bddd82d1870f6caf430f43732b35db3d9b2e',
    'runs/elec_baseline_s2026/config.json': 'a8d27ee50f98cb6b8c89acf0a59c9d42846ffbcfbd85f34b8d864f8b95dcb875',
    'runs/elec_baseline_s2026/failure.json': '6bd82cce4407ffe47502de59d886799e2ecffba31d771569d8604be0c5e63a1f',
    'runs/elec_baseline_s2026/best.pt': '3f9283407ff51178549c2f78e0f8cf61f895c269164afa6b3b9a0a56d8faacda',
    'runs/elec_baseline_s2026/failure_replay.pt': '7b6f12731715c0315018f6b3c903b39a8829cc7727035e18eb92eeaa4bd3deb7',
}
PROJECTOR_SHA256 = 'e44eb739482f3fe0448e1c9fdb066e36c4d6ea22d05a0c25a32734f29a856ab6'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify(root, configuration):
    rows = []
    def check(path, expected):
        actual = sha(path)
        rows.append({'path': str(path), 'expected_sha256': expected, 'actual_sha256': actual,
                     'matches': actual == expected})
        if actual != expected:
            raise RuntimeError('Protected input/source hash changed: ' + str(path))
    for name, expected in CRITICAL.items():
        check(BASELINE / name, expected)
    deployment = json.loads((BASELINE / 'deployment.json').read_text())
    audit = json.loads((BASELINE / 'input_audit.json').read_text())
    if len(deployment['carrier_sha256']) != 35 or len(audit['carrier_files']) != 35:
        raise RuntimeError('Expected 35 frozen carrier files')
    for name, expected in deployment['bundle_sha256'].items():
        check(BASELINE / name, expected)
    for name, expected in deployment['carrier_sha256'].items():
        check(BASELINE / 'carrier/src' / name, expected)
    for row in audit['carrier_files'].values():
        check(row['path'], row['sha256'])
    for row in audit['original_inputs'].values():
        check(row['path'], row['sha256'])
    inputs = json.loads((RUN / 'inputs.json').read_text())
    for path, expected in inputs['sha256'].items():
        check(path, expected)
    absent = [{'path': path, 'still_absent': not Path(path).exists()} for path in inputs['absent']]
    if not all(row['still_absent'] for row in absent):
        raise RuntimeError('Absent optional baseline input appeared')
    check(CPU_AUDIT / 'projector.py', PROJECTOR_SHA256)
    cpu = json.loads((CPU_AUDIT / 'replay_audit_receipt.json').read_text())
    if cpu['status'] != 'completed_pass' or cpu['source_revision'] != 'a12c750dbba12205ae049cc143a123b59c6b766f':
        raise RuntimeError('Frozen passing CPU audit receipt required')
    for name, expected in configuration['payload_sha256'].items():
        check(root / name, expected)
    return {'all_match': True, 'count': len(rows), 'rows': rows, 'absent_inputs': absent,
            'cpu_audit_receipt_sha256': sha(CPU_AUDIT / 'replay_audit_receipt.json')}


def idle_gpu_resources():
    commands = {
        'gpu': ['nvidia-smi', '-i', '1', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
        'processes': ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                      '--format=csv,noheader,nounits'],
    }
    query = {key: subprocess.run(command, capture_output=True, text=True, timeout=20)
             for key, command in commands.items()}
    if any(result.returncode for result in query.values()):
        raise RuntimeError('GPU resource query failed')
    uuid, free = [field.strip() for field in query['gpu'].stdout.strip().split(',')]
    occupied = [line for line in query['processes'].stdout.splitlines() if line.split(',')[0].strip() == uuid]
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    cpus = len(os.sched_getaffinity(0))
    row = {'physical_gpu_index': 1, 'uuid': uuid, 'free_gpu_bytes': int(free) * 1024 ** 2,
           'gpu_compute_processes': occupied, 'ram_available_bytes': available,
           'available_cpu_count': cpus}
    if occupied or row['free_gpu_bytes'] < 14 * 1024 ** 3 or available < 12 * 1024 ** 3 or cpus < 4:
        raise RuntimeError('GPU 1 idle/resource gate failed: ' + repr(row))
    return row
