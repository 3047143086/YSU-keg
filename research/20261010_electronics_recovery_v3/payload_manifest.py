"""The complete, independently reviewed recovery preparation payload."""
from pathlib import Path

ROOT_NAMES = ('PROTOCOL.md', 'program.md', 'README.md', 'BUDGET_ASSESSMENT.md',
    'budget_evidence.json', 'v1_reference_config.json', 'input_audit.json', 'prerequisites.json',
    'plugin_config.json', 'train.py', 'numerics.py', 'guarded_train.py', 'proposal.py',
    'integration.py', 'projector.py', 'prerequisite_gate.py', 'recovery_protocol.py',
    'run_queue.py', 'manage.py', 'cpu_tests.py', 'archive_evidence.py',
    'preparation_cpu_receipt.json', 'root_review.json', 'same_input_gate.py',
    'recovery_guard.py', 'payload_manifest.py', 'SOURCE_LINEAGE.json',
    'test_recovery_guard.py', 'test_projector.py', 'test_train_protocol.py',
    'test_recovery.py', 'test_integration.py', 'test_numerics.py')


def payload_names(root):
    root = Path(root)
    receipts = tuple(sorted(path.relative_to(root).as_posix()
                            for path in (root / 'receipts').rglob('*') if path.is_file()))
    return ROOT_NAMES + receipts
