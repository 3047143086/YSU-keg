"""Fail-closed, independent numerical prerequisite and protected-source guards."""
import hashlib
import json
from pathlib import Path

PROJECTOR_SHA256 = 'e44eb739482f3fe0448e1c9fdb066e36c4d6ea22d05a0c25a32734f29a856ab6'
V1 = Path('/data/sunzixuan/electronics_baseline_20261009_v1')
GRAPH_SHA256 = '76ccb18835b2d18b2f56e882cf4670b2d7af733951090656ab4be23ef70a9ed2'
ANCHORS = {
    'deployment.json': '58658242005fb07a58f7b677732bdea65ea0aab1feeed191d5310ad939f421d8',
    'input_audit.json': '109caa06a4eb24526438bda5d71ffc51e1d9d7d4c40d75525daa5ea32000c5d2',
    'source_revision.json': 'b4b9ec9c645ad98a7ec8913bc7976d2277d993a076658194f81a8fef16bde201',
    'runs/elec_baseline_s2026/inputs.json': '18dd014949737077a3ce63e004e2bddd82d1870f6caf430f43732b35db3d9b2e',
    'runs/elec_baseline_s2026/config.json': 'a8d27ee50f98cb6b8c89acf0a59c9d42846ffbcfbd85f34b8d864f8b95dcb875',
    'runs/elec_baseline_s2026/failure.json': '6bd82cce4407ffe47502de59d886799e2ecffba31d771569d8604be0c5e63a1f',
    'runs/elec_baseline_s2026/best.pt': '3f9283407ff51178549c2f78e0f8cf61f895c269164afa6b3b9a0a56d8faacda',
    'runs/elec_baseline_s2026/failure_replay.pt': '7b6f12731715c0315018f6b3c903b39a8829cc7727035e18eb92eeaa4bd3deb7',
}


class PrerequisiteError(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise PrerequisiteError(message)


def check_cpu(receipt):
    require(receipt.get('status') == 'completed_pass', 'CPU projector audit did not pass')
    for key in ('all_twelve_captures_inspected', 'candidate_mapping_and_finite_gradients_passed',
                'available_generic_gradient_comparisons_passed'):
        require(receipt.get(key) is True, 'CPU prerequisite missing/failed: ' + key)
    require(receipt.get('training_executed') is False and receipt.get('test_evaluated') is False,
            'CPU audit scope changed')
    rows = receipt.get('captures', [])
    require(len(rows) == 12 and [r.get('capture_index') for r in rows] == list(range(12)),
            'CPU audit must inspect exactly all twelve captures')
    for row in rows:
        require(row.get('exact_forward_bands') == [True] * 3 and
                row.get('candidate_input_gradient_finite') is True and
                row.get('candidate_mapping_and_finite_gradient_passed') is True,
                'CPU capture failed')
        if row.get('generic_input_gradient_finite') is True:
            require(row.get('generic_gradient_comparison', {}).get('passed') is True,
                    'Available CPU generic-gradient comparison failed')
    for phase in ('protected_before', 'protected_after'):
        require(receipt.get(phase, {}).get('all_match') is True, 'CPU protected hashes failed')


def check_gpu(receipt):
    # Never turn an overall failure into a pass by selecting the archived phase.
    require(receipt.get('status') == 'completed_pass', 'True-model GPU projector audit did not pass')
    require(receipt.get('optimizer_updates') == 0 and receipt.get('optimizer_constructed') is False
            and receipt.get('training_queue_started') is False and receipt.get('test_evaluated') is False
            and receipt.get('test_loader_constructed') is False, 'GPU audit scope changed')
    for phase in ('protected_before', 'protected_after'):
        require(receipt.get(phase, {}).get('all_match') is True, 'GPU protected hashes failed')
    archived = receipt.get('phase_archived', {})
    require(archived.get('candidate_passed') is True, 'Archived GPU phase failed')
    rows = archived.get('rows', [])
    require(len(rows) == 12 and [r.get('capture_index') for r in rows] == list(range(12)),
            'Archived GPU phase must inspect all twelve captures')
    for row in rows:
        require(row.get('exact_forward_bands') == [True] * 3 and
                row.get('candidate_gradient_finite') is True, 'Archived GPU capture failed')
        if row.get('generic_gradient_finite') is True:
            require(row.get('generic_comparison', {}).get('passed') is True,
                    'Available archived GPU gradient comparison failed')
    model = receipt.get('phase_model', {})
    for key in ('candidate_integration_passed', 'candidate_all_gradients_finite',
                'participating_parameters_match', 'state_unchanged', 'loss_matches'):
        require(model.get(key) is True, 'True-model GPU prerequisite failed: ' + key)
    require(model.get('state_sha256_before') and
            model.get('state_sha256_before') == model.get('state_sha256_after'),
            'GPU model state changed')
    flags = model.get('candidate_gradient_finite', {})
    require(len(flags) == model.get('participating_parameter_count') and len(flags) > 0
            and all(value is True for value in flags.values()), 'GPU per-parameter gradients missing/failed')
    rows = model.get('rows', [])
    require(len(rows) == 12 and [r.get('capture_index') for r in rows] == list(range(12)),
            'True model must inspect all twelve calls')
    for row in rows:
        require(row.get('candidate_vs_original_input', {}).get('passed') is True,
                'True-model input comparison failed')
        bands = row.get('candidate_vs_original_bands', [])
        require(len(bands) == 3 and all(b.get('passed') is True for b in bands),
                'True-model spectral-band comparison failed')
    require(isinstance(model.get('exact_historical_crash_reproduced'), bool),
            'Historical crash reproduction status must be explicit')
    if model.get('original_all_gradients_finite') is True:
        require(model.get('generic_gradient_equivalence_passed') is True,
                'Available true-model finite reference gradient comparison failed')


def inspect_prerequisites(manifest, loader=None, hasher=None):
    loader = loader or (lambda p: json.loads(Path(p).read_text()))
    hasher = hasher or sha
    result = {'eligible': False, 'checks': {}, 'eval_cache_integrated': False}
    require(manifest.get('schema_version') == 1 and manifest.get('excluded_eval_cache') is True,
            'Only the registered uncached numerical recovery is allowed')
    require(manifest.get('projector_sha256') == PROJECTOR_SHA256, 'Projector source version changed')
    entries = manifest.get('entries', {})
    require(set(entries) == {'spectral_cpu', 'spectral_gpu', 'excluded_cache_gpu'},
            'Exactly the two projector prerequisites and negative cache evidence are required')
    errors = []
    for role, entry in entries.items():
        try:
            require(hasher(entry['path']) == entry['sha256'], role + ' receipt SHA drift')
            require(hasher(entry['source_config_path']) == entry['source_config_sha256'],
                    role + ' source configuration SHA drift')
            source = loader(entry['source_config_path'])
            receipt = loader(entry['path'])
            if role.startswith('spectral_'):
                require(source.get('source_revision') == entry['source_revision'] and
                        receipt.get('source_revision') == entry['source_revision'], role + ' source revision changed')
                require(source.get('payload_sha256') == entry['payload_sha256'], role + ' payload changed')
                for name, expected in source['payload_sha256'].items():
                    require(hasher(Path(entry['source_config_path']).parent / name) == expected,
                            role + ' reviewed audit source changed: ' + name)
                if role == 'spectral_cpu':
                    require(source['payload_sha256'].get('projector.py') == PROJECTOR_SHA256,
                            'CPU projector source drift')
                    check_cpu(receipt)
                else:
                    check_gpu(receipt)
            else:
                require(receipt.get('status') == 'non_equivalent', 'Cache exclusion evidence is missing')
            result['checks'][role] = {'passed': True, 'receipt_status': receipt.get('status')}
        except (OSError, KeyError, ValueError, PrerequisiteError) as error:
            result['checks'][role] = {'passed': False, 'error': str(error)}
            errors.append(role + ': ' + str(error))
    result['errors'] = errors
    # Preparation cannot be made eligible merely by editing the failed receipt.
    result['eligible'] = not errors and manifest.get('ready_for_training') is True
    if not errors and not result['eligible']:
        result['errors'].append('This committed preparation is marked ready_for_training=false')
    return result


def require_prerequisites(root):
    result = inspect_prerequisites(json.loads((Path(root) / 'prerequisites.json').read_text()))
    require(result['eligible'], 'Recovery launch denied: ' + '; '.join(result['errors']))
    return result


def verify_frozen(root, deployment_required=True):
    root = Path(root)
    checked = {}
    def check(path, expected):
        actual = sha(path)
        require(actual == expected, 'Protected source/input changed: ' + str(path))
        checked[str(path)] = actual
    for name, expected in ANCHORS.items():
        check(V1 / name, expected)
    old = json.loads((V1 / 'deployment.json').read_text())
    launch = json.loads((V1 / 'launch.json').read_text())
    try:
        command = Path('/proc', str(launch['pid']), 'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        command = []
    require(str(V1 / 'run_queue.py').encode() not in command, 'Original v1 queue is still active')
    require(json.loads((V1 / 'pipeline_status.json').read_text()).get('status') == 'technical_failure',
            'Original v1 failure state changed')
    audit = json.loads((root / 'input_audit.json').read_text())
    require(len(old['carrier_sha256']) == len(audit['carrier_files']) == 35,
            'Exactly thirty-five carrier source/config files are protected')
    for name, expected in old['bundle_sha256'].items():
        check(V1 / name, expected)
    for name, expected in old['carrier_sha256'].items():
        check(V1 / 'carrier/src' / name, expected)
        check(audit['carrier_files'][name]['path'], expected)
        if deployment_required:
            check(root / 'carrier/src' / name, expected)
    for row in audit['original_inputs'].values():
        check(row['path'], row['sha256'])
    run_inputs = json.loads((V1 / 'runs/elec_baseline_s2026/inputs.json').read_text())
    for name, expected in run_inputs['sha256'].items():
        check(name, expected)
    require(not any(Path(p).exists() for p in run_inputs['absent']), 'Previously absent input appeared')
    check(V1 / 'carrier/data/elec/mm_adj_10.pt', GRAPH_SHA256)
    if deployment_required:
        deployment = json.loads((root / 'deployment.json').read_text())
        require(deployment['carrier_sha256'] == old['carrier_sha256'], 'Recovery carrier manifest changed')
        for name, expected in deployment['bundle_sha256'].items():
            check(root / name, expected)
        check(root / 'carrier/data/elec/mm_adj_10.pt', GRAPH_SHA256)
        for name, row in audit['original_inputs'].items():
            check(root / 'carrier/data/elec' / name, row['sha256'])
        require(not (root / 'carrier/data/elec/user_emb.npy').exists() and
                not (root / 'carrier/data/elec/user_graph_dict.npy').exists(),
                'Recovery optional user inputs must remain absent')
    check(root / 'projector.py', PROJECTOR_SHA256)
    return {'all_match': True, 'sha256': checked, 'graph_sha256': GRAPH_SHA256,
            'v1_preserved': True, 'test_evaluated': False}
