"""Anchor immutable predecessor helpers before importing their source."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from integration_gate import EXPECTED_PARAMETERS

PREDECESSOR = Path('/data/sunzixuan/failure_model_gpu_audit_20261010_v1')
PREDECESSOR_REVISION = 'fd42959e0a97f24169196851cd6d291477b31c59'
ANCHORS = {
    'remote_gpu.py': '2b395f5aeddd6e3983a563e1787c1fa27b0ba366f9ead59d4d5eff7a4ae409f8',
    'guards.py': 'ceb9c4c46c6287318b9cbcef682967620bfed8947e9bc75242ef789a362d1a46',
    'audit_gate.py': '96d6fd782602867b9b2a7c618bbbd789b96bebbb55afb657863a7afa3b539b40',
    'audit_config.json': '42364aa8cb84e44d243fc58fbdd89420fbe7c2a68804897a625ceceaa5460cdd',
    'gpu_audit_receipt.json': 'a840a3a1d952a8300d44951af77d8068b28770057a0ba3ca1dc833e0c21b7eb0',
}
REPEAT_ROOT = Path('/data/sunzixuan/forward_repeat_diagnostic_20261010_v2')
REPEAT_ANCHORS = {
    'diagnostic_config.json': 'f5b433ab7962c890c3707b084ffd9c0f52f88adc930f8f23583f6b114a7b1526',
    'repeat_receipt.json': '651f5d066d03433a21926565d885050a14ecb63faae8cd5b0966aac20f565540',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def predecessor_guard():
    rows = []
    for name, expected in ANCHORS.items():
        actual = sha(PREDECESSOR / name)
        rows.append({'path': str(PREDECESSOR / name), 'expected_sha256': expected,
                     'actual_sha256': actual, 'matches': expected == actual})
        if actual != expected:
            raise RuntimeError('Frozen predecessor anchor changed: ' + name)
    config = json.loads((PREDECESSOR / 'audit_config.json').read_text())
    receipt = json.loads((PREDECESSOR / 'gpu_audit_receipt.json').read_text())
    if config['source_revision'] != PREDECESSOR_REVISION or receipt['source_revision'] != PREDECESSOR_REVISION:
        raise RuntimeError('Unexpected predecessor revision')
    if receipt['status'] != 'completed_fail':
        raise RuntimeError('Frozen failed predecessor receipt required')
    for name, expected in config['payload_sha256'].items():
        if sha(PREDECESSOR / name) != expected:
            raise RuntimeError('Frozen predecessor payload changed: ' + name)
    phase = receipt['phase_model']
    if (not phase['original_all_gradients_finite'] or phase['participating_parameter_count'] != 27
            or not phase['participating_parameters_match']
            or set(phase['finite_original_gradient_comparisons']) != set(EXPECTED_PARAMETERS)):
        raise RuntimeError('Frozen original participating-parameter reference changed')
    archived = receipt['phase_archived']
    if (len(archived['rows']) != 12 or not archived['candidate_passed']
            or archived['rows'][0]['generic_gradient_finite']
            or not archived['rows'][0]['candidate_gradient_finite']):
        raise RuntimeError('Distinct frozen archived-input GPU evidence required')
    repeat_rows = []
    for name, expected in REPEAT_ANCHORS.items():
        actual = sha(REPEAT_ROOT / name)
        if actual != expected:
            raise RuntimeError('Frozen original-repeat raw receipt changed: ' + name)
        repeat_rows.append({'path': str(REPEAT_ROOT/name), 'expected_sha256': expected,
                            'actual_sha256': actual, 'matches': True})
    repeat_config = json.loads((REPEAT_ROOT / 'diagnostic_config.json').read_text())
    repeat_receipt = json.loads((REPEAT_ROOT / 'repeat_receipt.json').read_text())
    if (repeat_config['source_revision'] != 'dd55a8d1de4a72226847fbe1f629745184ea2d5f'
            or repeat_receipt['source_revision'] != repeat_config['source_revision']
            or repeat_receipt['status'] != 'completed_different' or repeat_receipt['forward_count'] != 2):
        raise RuntimeError('Frozen original-repeat lineage differs')
    for name, expected in repeat_config['payload_sha256'].items():
        if sha(REPEAT_ROOT / name) != expected:
            raise RuntimeError('Frozen original-repeat payload changed: ' + name)
    return {'all_match': True, 'anchor_rows': rows, 'source_revision': PREDECESSOR_REVISION,
            'status': receipt['status'], 'payload_count': len(config['payload_sha256']),
            'repeat_anchor_rows': repeat_rows, 'repeat_payload_count': len(repeat_config['payload_sha256']),
            'original_reference_parameters': sorted(EXPECTED_PARAMETERS),
            'archived_input_evidence': {'all_twelve_candidate_passed': True,
                'generic_capture0_gradient_finite': False, 'projector_capture0_gradient_finite': True,
                'generic_capture0_nonfinite_elements': archived['rows'][0]['generic_nonfinite_elements'],
                'probe_is_actual_minibatch_gradient': False}}


def load_source(name, path):
    if name in sys.modules:
        raise RuntimeError('Module name already occupied before frozen import: ' + name)
    specification = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def frozen_helpers():
    evidence = predecessor_guard()
    guard = load_source('guards', PREDECESSOR / 'guards.py')
    load_source('audit_gate', PREDECESSOR / 'audit_gate.py')
    helper = load_source('_frozen_failure_gpu_helpers', PREDECESSOR / 'remote_gpu.py')
    return helper, guard, evidence
