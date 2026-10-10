"""End-to-end synthetic checks, never counted as real experiment results."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch needed for integration')
class IntegrationTests(unittest.TestCase):
    def test_all_variants_finish_with_shared_ranker(self):
        script = Path(__file__).with_name('train_retrieval.py')
        with tempfile.TemporaryDirectory(prefix='retrieval_screen_test_') as directory:
            root = Path(directory)
            prepared = root / 'prepared' / '2026'
            prepared.mkdir(parents=True)
            (root / 'features').mkdir()
            rng = np.random.RandomState(72)
            n_users, n_items = 30, 300
            features = rng.normal(size=(n_items, 2, 512)).astype(np.float32)
            features /= np.linalg.norm(features, axis=-1, keepdims=True)
            base = rng.normal(size=(n_users, n_items)).astype(np.float32)
            mask = np.zeros_like(base, dtype=bool)
            mask[np.arange(n_users), np.arange(n_users)] = True
            hidden = base.copy()
            hidden[mask] = -np.inf
            top = np.argsort(-hidden, axis=1, kind='stable')[:, :200]
            np.save(root / 'features' / 'global_bank.npy', features)
            arrays = {'base_scores': base, 'base_top200': top,
                      'base_user': rng.normal(size=(n_users, 64)).astype(np.float32),
                      'history': features[np.arange(n_users)].reshape(n_users, 1024),
                      'support_mask': mask, 'targets': np.arange(n_users) + 230,
                      'roles': np.repeat(np.arange(3), 10)}
            for name, value in arrays.items():
                np.save(prepared / (name + '.npy'), value)
            (root / 'experiment.json').write_text(json.dumps({'epochs': 1, 'batch_size': 10,
                'eval_every': 1, 'eval_batch_size': 10, 'patience_evaluations': 1}), encoding='utf-8')
            ranker_hash = None
            for variant in ('independent', 'hard', 'clear', 'conditional', 'shuffled', 'base_user', 'feedback'):
                output = root / 'runs' / variant
                command = [sys.executable, str(script), '--root', str(root), '--seed', '2026',
                           '--gpu-id', '-1', '--output', str(output), '--variant', variant]
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='4',
                                   MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
                result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=50)
                self.assertEqual(result.returncode, 0, msg=variant + '\n' + result.stdout[-3000:] + result.stderr[-3000:])
                record = json.loads((output / 'result.json').read_text())
                self.assertEqual(record['status'], 'completed')
                self.assertFalse(record['dev_used_for_training_or_selection'])
                self.assertEqual(record['candidate_budget'], 200)
                self.assertIn('full_catalog_fusion_baseline', record)
                if ranker_hash is None:
                    ranker_hash = record['common_ranker_sha256']
                self.assertEqual(record['common_ranker_sha256'], ranker_hash)
                with np.load(output / 'dev_per_user.npz') as saved:
                    self.assertEqual(saved['candidates'].shape, (10, 200))
                    self.assertTrue(np.isfinite(saved['metrics']).all())


if __name__ == '__main__':
    unittest.main()
