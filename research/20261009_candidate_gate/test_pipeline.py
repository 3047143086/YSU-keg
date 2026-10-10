"""Keep the training-coverage gate separate from validation selection."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_pipeline


class MechanismTests(unittest.TestCase):
    def evaluate(self, diagnostics, validation=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'smoke'
            output.mkdir()
            row = {'train_diagnostics': diagnostics, 'validation': validation}
            (output / 'progress.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
            with mock.patch.object(run_pipeline, 'ROOT', root), mock.patch.object(run_pipeline, 'update'):
                passed = run_pipeline.mechanism_gate('baby', output)
            record = json.loads((root / 'baby_mechanism.json').read_text())
            self.assertIs(record['passed'], passed)
            self.assertIs(record['test_evaluated'], False)
            self.assertIs(record['validation_metrics_used_for_mechanism_gate'], False)
            return passed

    def test_exact_boundary_passes_even_with_zero_validation_recall(self):
        self.assertTrue(self.evaluate({'random_negative_in_scope_rate': .10,
                                      'eligible_pair_rate': .02,
                                      'mean_valid_candidate_negatives_eligible': 20},
                                     {'recall@20': 0.0}))

    def test_each_coverage_failure_stops_even_with_high_validation_recall(self):
        passing = {'random_negative_in_scope_rate': .10, 'eligible_pair_rate': .02,
                   'mean_valid_candidate_negatives_eligible': 20}
        for key, value in (('random_negative_in_scope_rate', .100001),
                           ('eligible_pair_rate', .019999),
                           ('mean_valid_candidate_negatives_eligible', 19.999)):
            with self.subTest(field=key):
                values = dict(passing, **{key: value})
                self.assertFalse(self.evaluate(values, {'recall@20': 1.0}))

    def test_missing_diagnostics_are_a_technical_error(self):
        with self.assertRaises(KeyError):
            self.evaluate({})


if __name__ == '__main__':
    unittest.main()
