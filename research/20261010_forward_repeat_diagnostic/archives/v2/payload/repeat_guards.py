"""Anchor immutable predecessor helpers before importing their source."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

PREDECESSOR = Path('/data/sunzixuan/failure_model_gpu_audit_20261010_v1')
PREDECESSOR_REVISION = 'fd42959e0a97f24169196851cd6d291477b31c59'
ANCHORS = {
    'remote_gpu.py': '2b395f5aeddd6e3983a563e1787c1fa27b0ba366f9ead59d4d5eff7a4ae409f8',
    'guards.py': 'ceb9c4c46c6287318b9cbcef682967620bfed8947e9bc75242ef789a362d1a46',
    'audit_gate.py': '96d6fd782602867b9b2a7c618bbbd789b96bebbb55afb657863a7afa3b539b40',
    'audit_config.json': '42364aa8cb84e44d243fc58fbdd89420fbe7c2a68804897a625ceceaa5460cdd',
    'gpu_audit_receipt.json': 'a840a3a1d952a8300d44951af77d8068b28770057a0ba3ca1dc833e0c21b7eb0',
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
    return {'all_match': True, 'anchor_rows': rows, 'source_revision': PREDECESSOR_REVISION,
            'status': receipt['status'], 'payload_count': len(config['payload_sha256'])}


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
