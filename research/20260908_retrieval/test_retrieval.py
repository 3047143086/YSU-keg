"""Small CPU checks; model tests additionally run when PyTorch is installed."""

import importlib.util
import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np

from train_retrieval import (METRIC_NAMES, Prepared, fixed_budget_candidates,
                             make_model, matched_context_permutation,
                             metric_arrays, standardize_scores)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.top = np.tile(np.arange(200), (3, 1))
        self.scores = np.tile(np.arange(300, dtype=np.float32), (3, 1))
        self.mask = np.zeros_like(self.scores, dtype=bool)
        self.mask[:, 299] = True

    def test_exact_budget_and_support_exclusion(self):
        for budget in (20, 50):
            actual = fixed_budget_candidates(self.top, self.scores, self.mask, budget)
            self.assertEqual(actual.shape, (3, 200))
            self.assertTrue(np.all(actual[:, :200 - budget] == self.top[:, :200 - budget]))
            self.assertTrue(np.all(actual[:, 200 - budget:] >= 200))
            self.assertFalse(np.any(actual == 299))
            self.assertTrue(all(len(np.unique(row)) == 200 for row in actual))

    def test_impossible_budget_raises(self):
        self.mask[:, 200:] = True
        with self.assertRaises(ValueError):
            fixed_budget_candidates(self.top, self.scores, self.mask, 20)

    def test_metrics_and_net_coverage(self):
        actual = fixed_budget_candidates(self.top, self.scores, self.mask, 20)
        targets = np.asarray([298, 199, 0])
        raw = metric_arrays(self.top, self.scores, targets)
        new = metric_arrays(actual, self.scores, targets)
        self.assertEqual(new.shape[1], len(METRIC_NAMES))
        self.assertAlmostEqual(new[0, 1], 1.0)
        self.assertAlmostEqual(new[0, 3], 1.0)
        self.assertEqual(new[:, -1].tolist(), [1.0, 0.0, 1.0])
        self.assertEqual((new[:, -1] - raw[:, -1]).tolist(), [1.0, -1.0, 0.0])

    def test_standardization_handles_history_and_constant_rows(self):
        values = self.scores.copy()
        values[0, 20] = -np.inf
        values[1] = 2.0
        normalized = standardize_scores(values)
        self.assertTrue(np.isneginf(normalized[0, 20]))
        self.assertFalse(np.isnan(normalized).any())
        self.assertAlmostEqual(float(normalized[0, np.isfinite(normalized[0])].mean()), 0.0, places=5)
        self.assertTrue(np.all(normalized[1] == 0.0))

    def test_context_derangement_is_reproducible_and_bijective(self):
        degree = np.repeat(np.arange(10), 13)
        popularity = np.linspace(0, 1, len(degree))
        permutation = matched_context_permutation(degree, popularity)
        self.assertTrue(np.all(permutation != np.arange(len(degree))))
        self.assertEqual(len(np.unique(permutation)), len(degree))
        self.assertTrue(np.array_equal(permutation, matched_context_permutation(degree, popularity)))

    def test_prepared_interface_and_target_leak_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared" / "2026"
            prepared.mkdir(parents=True)
            (root / "features").mkdir()
            rng = np.random.default_rng(5)
            np.save(root / "features" / "global_bank.npy", rng.normal(size=(300, 2, 512)).astype(np.float32))
            np.save(prepared / "base_scores.npy", self.scores)
            np.save(prepared / "base_top200.npy", self.top)
            np.save(prepared / "base_user.npy", np.ones((3, 64), dtype=np.float32))
            np.save(prepared / "history.npy", np.ones((3, 1024), dtype=np.float32))
            np.save(prepared / "support_mask.npy", self.mask)
            np.save(prepared / "targets.npy", np.asarray([298, 298, 298]))
            np.save(prepared / "roles.npy", np.asarray([0, 1, 2]))
            data = Prepared(root, 2026)
            self.assertEqual(data.context.shape, (3, 1024))
            self.assertEqual(data.variant_context("base_user").shape, (3, 1024))
            self.assertTrue(np.isneginf(data.base_rows([0])[0, 299]))
            self.assertTrue(np.all(data.variant_context("hard") == data.profiles))
            del data
            np.save(prepared / "targets.npy", np.asarray([299, 298, 298]))
            with self.assertRaises(ValueError):
                Prepared(root, 2026)
            gc.collect()

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable locally")
    def test_model_capacity_and_finite_gradient(self):
        import torch

        features = np.random.default_rng(9).normal(size=(300, 2, 512)).astype(np.float32)
        first = make_model(torch, features)
        second = make_model(torch, features)
        self.assertEqual(sum(p.numel() for p in first.parameters()), sum(p.numel() for p in second.parameters()))
        history = torch.randn(3, 1024)
        scores = first(history, history)
        self.assertEqual(scores.shape, (3, 300))
        torch.nn.functional.cross_entropy(scores, torch.tensor([1, 2, 3])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in first.parameters()))


if __name__ == "__main__":
    unittest.main()
