"""Reject model changes and extra dataset-runner edits before deployment."""
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class SourceLineageTests(unittest.TestCase):
    def test_runner_has_only_the_registered_dataset_choice_change(self):
        audit = json.loads((ROOT / 'input_audit.json').read_text())
        content = (ROOT / 'train.py').read_bytes()
        before = b"choices=('baby', 'sports', 'elec')"
        after = b"choices=('baby', 'sports')"
        self.assertEqual(content.count(before), 1)
        restored = content.replace(before, after)
        self.assertEqual(hashlib.sha256(restored).hexdigest(), audit['runner_files']['train.py']['sha256'])

    def test_numerical_observers_are_byte_identical(self):
        audit = json.loads((ROOT / 'input_audit.json').read_text())
        for name in ('guarded_train.py', 'numerics.py'):
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                             audit['runner_files'][name]['sha256'])


if __name__ == '__main__':
    unittest.main()
