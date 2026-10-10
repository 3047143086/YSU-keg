import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reporting import REMOTE_ROOT, TERMINAL_SUCCESS, decide, finish


class ScientificDecisionTests(unittest.TestCase):
    def metrics(self):
        return {'baseline': {'recall@20': .1, 'ndcg@20': .05, 'recall@10': .07, 'ndcg@10': .04},
                'joint': {'recall@20': .1002, 'ndcg@20': .0501, 'recall@10': .07, 'ndcg@10': .04},
                'candidate': {'recall@20': .101, 'ndcg@20': .051, 'recall@10': .07, 'ndcg@10': .04}}

    def test_requires_every_primary_shortlist_and_matched_control_gate(self):
        self.assertEqual(decide(self.metrics())['status'], 'keep')
        for key, value in (('recall@20', .1004), ('ndcg@20', .05),
                           ('recall@10', .0697), ('ndcg@10', .0397)):
            metrics = self.metrics()
            metrics['candidate'][key] = value
            self.assertEqual(decide(metrics)['status'], 'discard')
        for key in ('recall@20', 'ndcg@20'):
            metrics = self.metrics()
            metrics['joint'][key] = metrics['candidate'][key]
            self.assertEqual(decide(metrics)['status'], 'discard')

    def test_joint_comparison_and_exploratory_attribution_are_reported(self):
        decision = decide(self.metrics())
        self.assertIn('absolute_vs_joint', decision)
        self.assertNotIn('absolute_vs_history', decision)
        self.assertEqual(decision['evidence_level'], 'single-seed exploratory validation')
        self.assertIn('gate only', decision['tested_mechanism'])

    def test_missing_control_is_pending_and_invalid_metrics_fail(self):
        metrics = self.metrics()
        del metrics['joint']
        self.assertEqual(decide(metrics), {'status': 'pending', 'missing': ['joint']})
        for value in (float('nan'), float('inf'), -1., 1.1, True, '0.05', None):
            metrics = self.metrics()
            metrics['baseline']['ndcg@20'] = value
            with self.assertRaises(ValueError):
                decide(metrics)
        metrics = self.metrics()
        metrics['baseline']['recall@20'] = 0
        with self.assertRaises(ValueError):
            decide(metrics)

    def test_invalid_present_metrics_fail_even_with_missing_control(self):
        metrics = self.metrics()
        del metrics['joint']
        metrics['candidate']['recall@20'] = float('nan')
        with self.assertRaises(ValueError):
            decide(metrics)
        metrics = self.metrics()
        metrics['history'] = metrics.pop('joint')
        with self.assertRaises(ValueError):
            decide(metrics)

    def write_outputs(self, root, datasets=('baby',)):
        outputs = {}
        for dataset in datasets:
            outputs[dataset] = {}
            for arm, values in self.metrics().items():
                output = root / (dataset + '_' + arm)
                output.mkdir()
                (output / 'result.json').write_text(json.dumps({
                    'status': 'completed', 'test_evaluated': False,
                    'metrics': values}), encoding='utf-8')
                outputs[dataset][arm] = output
        return outputs

    def read_rows(self, root):
        with (root / 'results.tsv').open(encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream, delimiter='\t'))

    def test_finish_records_controls_kept_candidate_memory_and_formal_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = self.write_outputs(root, ('baby', 'sports'))
            state = {'status': TERMINAL_SUCCESS, 'stage': 'done', 'jobs': [
                {'name': 'baby_candidate_s2026', 'peak_sampled_gpu_memory_mib': 3072},
                {'name': 'baby_candidate_smoke', 'peak_sampled_gpu_memory_mib': 8192}]}
            with patch('reporting.time.time', return_value=123.):
                report = finish(root, state, outputs, 'abc123')
            self.assertTrue(report['pilot_success'])
            self.assertEqual(report['new_completed_formal_arms'], 4)
            self.assertFalse(report['novelty_established'])
            self.assertFalse(report['generalizable_performance_claim'])
            self.assertFalse(report['test_evaluated'])
            self.assertEqual(json.loads((root / 'research_report.json').read_text()), report)
            rows = self.read_rows(root)
            self.assertEqual(sum(row['status'] == 'keep' for row in rows), 2)
            self.assertEqual(sum(row['status'] == 'control' for row in rows), 4)
            candidate = next(row for row in rows if row['description'] == 'baby candidate seed 2026')
            self.assertEqual(candidate['sampled_peak_gpu_gib'], '3.000000')
            baseline = next(row for row in rows if row['description'] == 'baby baseline seed 2026')
            self.assertEqual(baseline['sampled_peak_gpu_gib'], 'NA')

    def test_finish_discard_and_crash_preserve_completed_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = self.write_outputs(root)
            candidate = outputs['baby']['candidate'] / 'result.json'
            result = json.loads(candidate.read_text())
            result['metrics']['recall@20'] = .1
            candidate.write_text(json.dumps(result), encoding='utf-8')
            report = finish(root, {'status': 'technical_failure', 'error': 'bad gradient'}, outputs)
            self.assertFalse(report['pilot_success'])
            self.assertEqual(report['decisions']['baby']['status'], 'discard')
            self.assertEqual(report['decisions']['sports']['status'], 'pending')
            self.assertEqual(report['new_completed_formal_arms'], 2)
            self.assertEqual([row['status'] for row in self.read_rows(root)],
                             ['control', 'control', 'discard', 'crash'])

    def test_all_metric_gates_cannot_override_technical_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = self.write_outputs(root, ('baby', 'sports'))
            report = finish(root, {'status': 'technical_failure', 'error': 'late failure'}, outputs)
            self.assertEqual(report['decisions']['baby']['status'], 'keep')
            self.assertEqual(report['decisions']['sports']['status'], 'keep')
            self.assertFalse(report['pilot_success'])
            self.assertEqual(report['pipeline_status'], 'technical_failure')

    def test_success_requires_the_declared_terminal_pipeline_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = self.write_outputs(root, ('baby', 'sports'))
            report = finish(root, {'status': 'closed_validation_gate'}, outputs)
            self.assertFalse(report['pilot_success'])

    def test_partial_report_cannot_claim_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = self.write_outputs(root)
            del outputs['baby']['joint']
            report = finish(root, {'status': 'running'}, outputs)
            self.assertFalse(report['pilot_success'])
            self.assertEqual(report['decisions']['baby']['missing'], ['joint'])
            self.assertEqual(report['new_completed_formal_arms'], 1)
            self.assertIn('pending', [row['status'] for row in self.read_rows(root)])

    def test_finish_rejects_unfinished_test_evaluated_and_invalid_results(self):
        for change in ({'status': 'running'}, {'test_evaluated': True},
                       {'metrics': {'recall@20': float('nan')}}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                outputs = self.write_outputs(root)
                candidate = outputs['baby']['candidate'] / 'result.json'
                result = json.loads(candidate.read_text())
                result.update(change)
                candidate.write_text(json.dumps(result), encoding='utf-8')
                with self.assertRaises((ValueError, RuntimeError)):
                    finish(root, {'status': 'running'}, outputs)
                self.assertFalse((root / 'results.tsv').exists())

    def test_snapshot_root_is_the_new_immutable_experiment(self):
        self.assertEqual(REMOTE_ROOT, '/data/sunzixuan/candidate_gate_20261009_v1')


if __name__ == '__main__':
    unittest.main()
