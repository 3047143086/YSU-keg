"""Fail-closed, independent numerical prerequisite and protected-source guards."""
import hashlib
import json
import math
import re
import os
from same_input_gate import EXPECTED_PARAMETERS, integration_passed
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


ROLES = frozenset(('projector_cpu_tests', 'spectral_cpu', 'old_gpu_archived',
                   'original_repeat', 'same_input_gpu', 'same_input_cpu_tests', 'excluded_cache_gpu'))
SOURCE_REVISIONS = {
    'projector_cpu_tests': 'a12c750dbba12205ae049cc143a123b59c6b766f',
    'spectral_cpu': 'a12c750dbba12205ae049cc143a123b59c6b766f',
    'old_gpu_archived': 'fd42959e0a97f24169196851cd6d291477b31c59',
    'original_repeat': 'dd55a8d1de4a72226847fbe1f629745184ea2d5f',
    'same_input_gpu': 'e99e57b42f52f9e6494dd527e0c2eab6ce6d4fc6',
    'same_input_cpu_tests': 'e99e57b42f52f9e6494dd527e0c2eab6ce6d4fc6',
}
RAW_RECEIPT_SHA256 = {
    'old_gpu_archived': 'a840a3a1d952a8300d44951af77d8068b28770057a0ba3ca1dc833e0c21b7eb0',
    'original_repeat': '651f5d066d03433a21926565d885050a14ecb63faae8cd5b0966aac20f565540',
    'excluded_cache_gpu': '3df214b97ee633e71e4c8f616150ed8abe7fd29ecc7f78cabe85843d3d882d59',
    'same_input_gpu': '87251d388816ec3d395f477a096a7f67e1ea6e25cfaf84116b517c6a36c64f2c',
    'same_input_cpu_tests': '691fb0304b200c488186c3c63525ecf210da931a8a713989c3b91fc8ee13b57b',
}
SAME_INPUT_CONFIG_SHA256 = '9d217792556b38d685d79cad426a521d78bedfc4e28529e0040ed1774a5df84e'


def is_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def check_math_tests(receipt):
    require(receipt.get('status') == 'passed' and receipt.get('test_count') == 13,
            'All thirteen original mathematical tests must pass')
    require(receipt.get('cuda_initialized') is False and receipt.get('cuda_visible_devices') == ''
            and receipt.get('threads') == 4 and receipt.get('training_executed') is False,
            'Mathematical-test scope changed')
    require(receipt.get('source_files_sha256', {}).get('projector.py') == PROJECTOR_SHA256,
            'Mathematical tests used another projector')
    require(receipt.get('source_files_sha256', {}).get('test_projector.py') ==
            'ffb1de4e9e90e356557f99e95dfec23c5ca582b27ef560d94b58f8e087098ea4',
            'Original mathematical tests changed')


def check_old_gpu_archived(receipt):
    # The complete old receipt remains failed; its archived phase is distinct evidence.
    require(receipt.get('status') == 'completed_fail', 'Old full-model GPU failure must remain failed')
    require(receipt.get('phase_model', {}).get('candidate_integration_passed') is False
            and receipt['phase_model'].get('exact_historical_crash_reproduced') is False,
            'Old failed integration/replay cannot be reclassified')
    archived = receipt.get('phase_archived', {})
    rows = archived.get('rows', [])
    require(archived.get('candidate_passed') is True and len(rows) == 12
            and [row.get('capture_index') for row in rows] == list(range(12)),
            'Complete archived GPU evidence required')
    for row in rows:
        require(row.get('exact_forward_bands') == [True] * 3
                and row.get('candidate_gradient_finite') is True, 'Archived GPU capture failed')
        if row.get('generic_gradient_finite') is True:
            require(row.get('generic_comparison', {}).get('passed') is True,
                    'Available archived generic-gradient comparison failed')
        else:
            require(row.get('generic_comparison', {}).get('passed') is not True,
                    'Unavailable generic gradient cannot become an equivalence pass')
    require(rows[0].get('generic_gradient_finite') is False
            and rows[0].get('generic_nonfinite_elements', 0) > 0,
            'Original independent-probe failure evidence required')
    for phase in ('protected_before', 'protected_after'):
        require(receipt.get(phase, {}).get('all_match') is True, 'Old GPU protected hashes failed')


def check_repeat(receipt):
    require(receipt.get('status') == 'completed_different' and receipt.get('forward_count') == 2,
            'Original-original difference receipt required')
    require(receipt.get('summary', {}).get('bands', {}).get('nonexact_count') == 27
            and receipt['summary']['bands'].get('exceeds_inherited_bounds_count') == 13,
            'Frozen original-repeat result changed')
    require(receipt.get('backward_calls') == 0 and receipt.get('optimizer_updates') == 0,
            'Original-repeat scope changed')
    for phase in ('protected_before', 'protected_after'):
        require(receipt.get(phase, {}).get('all_match') is True, 'Original-repeat hashes failed')


def check_same_input(receipt, config_sha256):
    require(receipt.get('status') == 'completed_pass', 'Actual same-input integration did not pass')
    require(integration_passed(receipt), 'Same-input 12/36/27 gradient and lifecycle conjunction failed')
    require(receipt.get('full_forward_calls_started') == 1 and receipt.get('warning_observer_restored') is True,
            'Full-forward or warning observer lifecycle failed')
    require(receipt.get('participating_parameter_names_match_original') is True,
            'Exact original parameter participation set required')
    for field in ('optimizer_constructed', 'training_started', 'test_evaluated', 'cache_used',
                  'solver_override', 'historical_crash_replayed', 'generic_vjp_equivalence_claimed',
                  'publication_gain_claimed'):
        require(receipt.get(field) is False, 'Same-input audit scope changed: ' + field)
    require(receipt.get('optimizer_updates') == 0, 'Same-input audit performed an update')
    require(is_sha(config_sha256) and receipt.get('integration_config_sha256_before') == config_sha256
            and receipt.get('integration_config_sha256_after') == config_sha256,
            'Same-input deployment config hashes disagree')
    values = receipt.get('loss_values', [])
    require(len(values) == receipt.get('loss_component_count') and bool(values)
            and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) for value in values)
            and isinstance(receipt.get('total_loss_value'), (int, float))
            and math.isfinite(receipt['total_loss_value']), 'Actual scalar loss evidence missing/nonfinite')
    warnings = receipt.get('warnings')
    require(isinstance(warnings, list), 'Reported warning evidence missing')
    for row in warnings:
        message = row.get('message', '').lower()
        require(not (any(word in message for word in ('svd', 'gesvd', 'cusolver'))
                     and any(word in message for word in ('converg', 'fallback', 'fall back', 'recomput', 'failed'))),
                'Reported SVD fallback/nonconvergence warning forbids recovery')
    for phase in ('protected_before', 'protected_after', 'old_evidence_before', 'old_evidence_after'):
        require(receipt.get(phase, {}).get('all_match') is True, 'Same-input source/lineage hashes failed')
    for row in receipt['rows']:
        hashes = [row.get('input_sha256_' + phase) for phase in ('before', 'after_shadow', 'after_candidate')]
        require(all(is_sha(value) for value in hashes) and len(set(hashes)) == 1,
                'Same-input input hashes disagree')
        metadata = [row.get('input_metadata_' + phase) for phase in ('before', 'after_shadow', 'after_candidate')]
        require(metadata[0] and metadata[0] == metadata[1] == metadata[2], 'Input storage/version changed')
        old, new = row.get('original_band_sha256', []), row.get('candidate_band_sha256', [])
        require(len(old) == len(new) == 3 and all(is_sha(value) for value in old + new) and old == new,
                'Exact band hash evidence missing or inconsistent')
        events = row.get('svd_events', [])
        require(len(events) == 2 and [event.get('phase') for event in events] == ['shadow', 'candidate']
                and all(event.get('capture_index') == row['capture_index']
                        and event.get('successful') is True and event.get('full_matrices') is False
                        and event.get('driver_specified') is False and 'error' not in event for event in events),
                'All twenty-four original solver invocations required')


def check_same_input_cpu_tests(receipt, config_sha256):
    require(receipt.get('exit_code') == 0 and receipt.get('cuda_visible_devices') == ''
            and receipt.get('protected_count') == 129
            and receipt.get('predecessor_anchors_match') is True,
            'Actual same-input CPU transport/scope/anchors failed')
    require(config_sha256 == SAME_INPUT_CONFIG_SHA256
            and receipt.get('integration_config_sha256') == config_sha256,
            'Actual same-input CPU tests used another source config')
    output = receipt.get('stdout', '') + receipt.get('stderr', '')
    require(len(re.findall(r'^test_.*\.\.\. ok\s*$', output, re.MULTILINE)) == 31
            and re.search(r'\nRan 31 tests in [^\n]+\n\nOK\s*$', output) is not None
            and not re.search(r'\.\.\. skipped|^FAILED|^ERROR:|unexpected success', output, re.MULTILINE),
            'All actual thirty-one same-input CPU tests with zero skips required')


def inspect_prerequisites(manifest, loader=None, hasher=None, base_root=None, use_local=False):
    loader = loader or (lambda path: json.loads(Path(path).read_text()))
    hasher = hasher or sha
    result = {'eligible': False, 'checks': {}, 'eval_cache_integrated': False, 'errors': []}
    try:
        require(manifest.get('schema_version') == 3 and manifest.get('excluded_eval_cache') is True,
                'Only the separately registered uncached v3 is allowed')
        require(manifest.get('projector_sha256') == PROJECTOR_SHA256, 'Projector source version changed')
        require(set(manifest.get('entries', {})) == ROLES, 'All seven independent evidence roles required')
    except PrerequisiteError as error:
        result['errors'].append(str(error))
        return result
    for role, entry in manifest['entries'].items():
        try:
            def path_for(field):
                local_field = 'local_path' if field == 'path' else 'local_source_config_path'
                path = Path(entry[local_field] if use_local else entry[field])
                return Path(base_root) / path if base_root is not None and not path.is_absolute() else path
            receipt_path, config_path = path_for('path'), path_for('source_config_path')
            require(not entry.get('anchors_pending') and is_sha(entry.get('sha256'))
                    and is_sha(entry.get('source_config_sha256')) and bool(entry.get('payload_sha256')),
                    role + ' actual receipt/config/payload anchors missing')
            if role in RAW_RECEIPT_SHA256:
                require(entry['sha256'] == RAW_RECEIPT_SHA256[role], role + ' frozen raw evidence changed')
            if role in ('same_input_gpu', 'same_input_cpu_tests'):
                require(entry['source_config_sha256'] == SAME_INPUT_CONFIG_SHA256, 'Frozen same-input config changed')
            require(hasher(receipt_path) == entry['sha256'], role + ' receipt SHA drift')
            require(hasher(config_path) == entry['source_config_sha256'], role + ' config SHA drift')
            config, receipt = loader(config_path), loader(receipt_path)
            if role in SOURCE_REVISIONS:
                require(config.get('source_revision') == entry.get('source_revision') == SOURCE_REVISIONS[role],
                        role + ' reviewed source revision changed')
                if role not in ('projector_cpu_tests', 'same_input_cpu_tests'):
                    require(receipt.get('source_revision') == entry['source_revision'], role + ' receipt source changed')
            payload = config.get('payload_sha256', config.get('bundle_sha256', config.get('sha256')))
            require(payload == entry['payload_sha256'], role + ' payload manifest changed')
            for name, expected in entry['payload_sha256'].items():
                require(not Path(name).is_absolute() and '..' not in Path(name).parts and is_sha(expected),
                        'Unsafe or incomplete audit payload')
                require(hasher(config_path.parent / name) == expected,
                        role + ' reviewed audit source drift: ' + name)
            if role == 'projector_cpu_tests':
                check_math_tests(receipt)
            elif role == 'spectral_cpu':
                require(config['payload_sha256'].get('projector.py') == PROJECTOR_SHA256, 'CPU projector source drift')
                check_cpu(receipt)
            elif role == 'old_gpu_archived':
                check_old_gpu_archived(receipt)
            elif role == 'original_repeat':
                check_repeat(receipt)
            elif role == 'same_input_gpu':
                check_same_input(receipt, entry['source_config_sha256'])
            elif role == 'same_input_cpu_tests':
                check_same_input_cpu_tests(receipt, entry['source_config_sha256'])
            else:
                require(receipt.get('status') == 'non_equivalent', 'Cache exclusion evidence missing')
            result['checks'][role] = {'passed': True, 'receipt_status': receipt.get('status')}
        except (OSError, KeyError, ValueError, TypeError, PrerequisiteError) as error:
            result['checks'][role] = {'passed': False, 'error': str(error)}
            result['errors'].append(role + ': ' + str(error))
    result['eligible'] = not result['errors'] and manifest.get('ready_for_training') is True
    if not result['errors'] and not result['eligible']:
        result['errors'].append('New preparation still has ready_for_training=false')
    return result


def require_root_review(root, action):
    root = Path(root)
    review = json.loads((root / 'root_review.json').read_text())
    require(review.get('status') == 'approved' and review.get('remote_actions_authorized') is True,
            'Root review and explicit remote authorization are pending')
    require(review.get('reviewed_by') == '/root' and bool(review.get('reviewed_payload_sha256')),
            'Matching independent root review required')
    from payload_manifest import payload_names
    require(set(review['reviewed_payload_sha256']) == set(payload_names(root)) - {'root_review.json'},
            'Root review must bind the complete preparation payload, including actual CPU receipt and prerequisites')
    for name, expected in review['reviewed_payload_sha256'].items():
        require(is_sha(expected) and sha(root / name) == expected, 'Root-reviewed preparation source drift: ' + name)
    if action == 'start':
        require(review.get('training_authorized') is True, 'Root training authorization is pending')
    return review


def require_prerequisites(root):
    result = inspect_prerequisites(json.loads((Path(root) / 'prerequisites.json').read_text()),
                                  base_root=root, use_local=os.name == 'nt')
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
