"""Reject incomplete/altered evidence and keep recovery distinct from method success."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from recovery import check_result, check_config, finish_autoresearch, recover_edge, sha, RANKING_METRICS
from unittest.mock import patch


class RecoveryTests(unittest.TestCase):
    def valid_result(self):
        return {'status': 'completed', 'dataset': 'baby', 'seed': 2026,
                'model': 'TargetIsolatedBRIDGE', 'test_evaluated': False,
                'test_loader_constructed': False, 'inputs_unchanged': True,
                'fresh_initialization': True, 'selection_metric': 'rounded_4_validation_recall@20',
                'metrics': {key: .1 for key in RANKING_METRICS}}

    def test_reuse_rejects_incomplete_smoke_wrong_seed_and_test_exposure(self):
        check_result(self.valid_result())
        for key, value in [('status', 'running'), ('seed', 2027), ('dataset', 'sports'),
                           ('test_evaluated', True), ('inputs_unchanged', False),
                           ('test_loader_constructed', True), ('fresh_initialization', False)]:
            with self.subTest(key=key):
                row = self.valid_result()
                row[key] = value
                with self.assertRaises(RuntimeError):
                    check_result(row)
        for value in (float('nan'), float('inf'), -1, 2, None):
            row = self.valid_result()
            row['metrics']['recall@20'] = value
            with self.assertRaises(RuntimeError):
                check_result(row)

    def test_reuse_requires_full_frozen_training_budget_and_alpha(self):
        config = {'arguments': {'dataset': 'baby', 'seed': 2026, 'max_epochs': 300,
                  'device': 'cuda', 'cuda_memory_fraction': .55, 'train_batch_size': None,
                  'learning_rate': None, 'smoke': False, 'plugin_module': 'proposal',
                  'plugin_class': 'TargetIsolatedBRIDGE'}, 'budget_overridden': False,
                  'effective_config': {'evidence_mode': 'soft_edge', 'isolation_strength': .5,
                  'stopping_step': 20, 'eval_step': 1, 'eval_batch_size': 64,
                  'train_batch_size': 2048}}
        check_config(config)
        for section, key, value in [('arguments', 'max_epochs', 1), ('arguments', 'smoke', True),
                                    ('effective_config', 'isolation_strength', .25),
                                    ('effective_config', 'stopping_step', 12)]:
            row = copy.deepcopy(config)
            row[section][key] = value
            with self.assertRaises(RuntimeError):
                check_config(row)

    def test_negative_scientific_gate_is_successful_recovery_but_crash_is_not(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'recovery_receipt.json').write_text('{}')
            for arm in ('baseline', 'history'):
                path = root / 'runs' / ('baby_' + arm + '_s2026')
                path.mkdir(parents=True)
                (path / 'result.json').write_text(json.dumps({'status': 'completed'}))
            state = {'status': 'closed_validation_gate', 'screens': {'baby': {'passed': False}}}
            finish_autoresearch(root, state)
            handoff = json.loads((root / 'autoresearch/loop-recovery/handoff.json').read_text())
            self.assertEqual(handoff['status'], 'COMPLETE')
            self.assertEqual(handoff['findings'][0]['completed_arms'], 3)
            self.assertFalse(handoff['findings'][0]['scientific_screens']['baby']['passed'])
            finish_autoresearch(root, {'status': 'technical_failure', 'error': 'non-finite loss'})
            handoff = json.loads((root / 'autoresearch/loop-recovery/handoff.json').read_text())
            self.assertEqual(handoff['status'], 'ERROR')

    def test_recovery_validates_actual_artifact_hashes_and_never_reuses_failed_arm(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old, new = root / 'source', root / 'recovery'
            run = old / 'runs/baby_edge_s2026'
            run.mkdir(parents=True)
            for directory in (old, new):
                (directory / 'carrier/src').mkdir(parents=True)
                (directory / 'cache/baby').mkdir(parents=True)
                for name in ('train.py', 'proposal.py', 'evidence.py', 'prepare.py'):
                    (directory / name).write_text('frozen algorithm')
                (directory / 'carrier/src/model.py').write_text('frozen model')
                (directory / 'cache/baby/cache.npz').write_bytes(b'frozen evidence')
                (directory / 'deployment.json').write_text(json.dumps({'carrier_sha256': {
                    'model.py': sha(directory / 'carrier/src/model.py')}}))
            (old / 'pipeline_status.json').write_text(json.dumps({
                'status': 'technical_failure', 'stage': 'baby_baseline_s2026'}))
            (old / 'launch.json').write_text('{"pid": 999999999}')
            for name in ('best.pt', 'per_user_validation.npz'):
                (run / name).write_bytes(b'completed artifact')
            result = self.valid_result()
            result.update(checkpoint_sha256=sha(run / 'best.pt'),
                          per_user_sha256=sha(run / 'per_user_validation.npz'))
            (run / 'result.json').write_text(json.dumps(result))
            (run / 'config.json').write_text('{}')
            (run / 'inputs.json').write_text(json.dumps({'sha256': {
                str(old / 'train.py'): sha(old / 'train.py')}, 'absent': [str(old / 'optional')]}))
            with patch('recovery.check_config'):
                self.assertEqual(recover_edge(new, old), run)
                receipt = json.loads((new / 'recovery_receipt.json').read_text())
                self.assertEqual(receipt['reused_arms'], ['baby_edge_s2026'])
                self.assertTrue(receipt['checks_passed'])
                (run / 'best.pt').write_bytes(b'changed after completion')
                with self.assertRaisesRegex(RuntimeError, 'Recovered output changed'):
                    recover_edge(new, old)
                (old / 'optional').write_text('new input')
                with self.assertRaisesRegex(RuntimeError, 'Previously absent input appeared'):
                    recover_edge(new, old)


if __name__ == '__main__':
    unittest.main()
