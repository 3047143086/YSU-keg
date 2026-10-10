"""Queue contracts and paired aggregation, without remote or GPU access."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import run_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def test_real_data_smoke_command_cannot_evaluate_development(self):
        with tempfile.TemporaryDirectory(prefix='pairing_queue_test_') as directory:
            root = Path(directory)
            captured = []

            class Completed:
                pid = 123
                returncode = 0

                def poll(self):
                    return 0

            def launch(command, **kwargs):
                captured.append(command)
                output = root / 'smoke' / 'results' / '2026' / 'paired_cov'
                (output / 'result.json').write_text(json.dumps({
                    'status': 'smoke_passed', 'dev_evaluated': False}))
                return Completed()

            with patch.object(pipeline, 'ROOT', root), \
                 patch.object(pipeline, 'available_gpu', return_value=(1, [])), \
                 patch.object(pipeline.subprocess, 'Popen', side_effect=launch):
                self.assertTrue(pipeline.run_job(2026, 'paired_cov', smoke=True))
            self.assertIn('--smoke', captured[0])
            self.assertEqual(captured[0][captured[0].index('--device') + 1], 'cuda:0')

    def test_aggregate_shared_users_and_zero_gain_rejection(self):
        with tempfile.TemporaryDirectory(prefix='pairing_aggregate_test_') as directory:
            root = Path(directory)
            baseline = np.tile(np.array([.06, .03, .09, .04, .31]), (24, 1))
            for seed in pipeline.SPEC['seeds']:
                for variant in pipeline.SPEC['variants']:
                    output = root / 'results' / str(seed) / variant
                    output.mkdir(parents=True)
                    means = dict(zip(pipeline.METRICS, baseline.mean(axis=0).tolist()))
                    (output / 'result.json').write_text(json.dumps({
                        'status': 'completed', 'dev_metrics': means, 'baseline_metrics': means}))
                    np.savez_compressed(output / 'per_user.npz', users=np.arange(24),
                        targets=np.arange(24), metrics=baseline, baseline=baseline,
                        metric_names=np.asarray(pipeline.METRICS))
            with patch.object(pipeline, 'ROOT', root):
                summary = pipeline.collect_summary()
            self.assertEqual(summary['completed_jobs'], 15)
            self.assertTrue(summary['all_complete'])
            self.assertFalse(summary['confidence_screen_passed'])
            self.assertIn('screen_not_passed', summary['decision'])
            interval = summary['user_clustered_bootstrap_unadjusted_exploratory']['shuffled_cov']['ndcg@20']
            self.assertEqual(interval['ci95'], [0.0, 0.0])
            self.assertTrue((root / 'RESULTS.md').is_file())


if __name__ == '__main__':
    unittest.main()
