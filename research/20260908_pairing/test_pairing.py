"""Protocol tests, including a bounded synthetic CPU integration smoke."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from train_pairing import (VARIANTS, Prepared, combine_scores, history_statistics,
                           main, make_model, metric_arrays, orthogonal_projection,
                           rms_transform, split_roles, standardize_scores, validate_support)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class PairingProtocolTests(unittest.TestCase):
    def test_covariance_identity(self):
        rng = np.random.default_rng(3)
        visual, textual = rng.normal(size=(2, 7, 4)).astype(np.float32)
        result = history_statistics(visual, textual)
        expected = ((visual - visual.mean(axis=0)).T @ (textual - textual.mean(axis=0))) / 7
        np.testing.assert_allclose(result["paired_cov"].reshape(4, 4), expected, atol=1e-7)
        np.testing.assert_allclose(result["paired_moment"],
                                   result["marginal_tensor"] + result["paired_cov"], atol=1e-7)

    def test_permutation_preserves_marginals_not_pairing(self):
        visual = np.asarray([[0, 1], [2, 4], [8, 3]], dtype=np.float32)
        textual = np.asarray([[3, 8], [2, 1], [7, 9]], dtype=np.float32)
        original = history_statistics(visual, textual)
        shuffled = history_statistics(visual, textual, np.asarray([2, 0, 1]))
        for key in ("visual_mean", "textual_mean", "marginal_tensor"):
            np.testing.assert_allclose(original[key], shuffled[key], atol=1e-6)
        self.assertFalse(np.allclose(original["paired_cov"], shuffled["paired_cov"]))
        np.testing.assert_array_equal(np.sort(textual, axis=0), np.sort(textual[[2, 0, 1]], axis=0))

    def test_uniform_permutation_expectation_is_zero(self):
        import itertools
        visual = np.asarray([[0, 1], [2, 4], [8, 3]], dtype=np.float32)
        textual = np.asarray([[3, 8], [2, 1], [7, 9]], dtype=np.float32)
        values = [history_statistics(visual, textual, permutation)["paired_cov"]
                  for permutation in itertools.permutations(range(3))]
        np.testing.assert_allclose(np.mean(values, axis=0), 0, atol=2e-6)

    def test_single_history_is_exact_zero(self):
        values = history_statistics(np.asarray([[0.314, 0.7]]), np.asarray([[0.617, 0.1]]))
        np.testing.assert_array_equal(values["paired_cov"], np.zeros(4))

    def test_seeded_projection_and_statistics_are_deterministic(self):
        first = orthogonal_projection(32, 8, 42)
        second = orthogonal_projection(32, 8, 42)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first.T @ first, np.eye(8), atol=2e-7)
        rng = np.random.default_rng(7)
        visual, textual = rng.normal(size=(2, 5, 8)).astype(np.float32)
        permutation = np.random.default_rng(19).permutation(5)
        for key, value in history_statistics(visual, textual, permutation).items():
            np.testing.assert_array_equal(value, history_statistics(visual, textual, permutation)[key])

    def test_heldout_exclusion_and_role_boundaries(self):
        targets = np.asarray([2, 1, 0])
        support = np.eye(3, dtype=bool)
        support[1] = [True, False, False]
        support[2] = [False, True, False]
        validate_support(support, targets)
        groups = split_roles(targets, np.asarray([0, 1, 2]))
        for role, group in enumerate(groups):
            np.testing.assert_array_equal(group, [role])
        support[0, 2] = True
        with self.assertRaisesRegex(ValueError, "leaked"):
            validate_support(support, targets)

    def test_invalid_permutation_and_zero_rms_are_safe(self):
        with self.assertRaises(ValueError):
            history_statistics(np.ones((3, 4)), np.ones((3, 4)), [0, 0, 2])
        scaled, rms = rms_transform(np.zeros((5, 4), dtype=np.float32), np.asarray([0, 1]))
        self.assertTrue(np.isfinite(scaled).all())
        np.testing.assert_array_equal(scaled, 0)
        np.testing.assert_array_equal(rms, 0)

    def test_score_residual_genuinely_changes_top20(self):
        baseline = -np.arange(220, dtype=np.float32)[None, :]
        residual = np.zeros_like(baseline)
        residual[0, 100] = 120
        combined = combine_scores(baseline, residual, 1)
        self.assertEqual(np.argsort(-combined[0])[0], 100)
        before = metric_arrays(baseline, [100])
        after = metric_arrays(combined, [100])
        self.assertEqual(before[0, 2], 0)
        self.assertEqual(after[0, 2], 1)
        np.testing.assert_array_equal(combine_scores(baseline, residual, 0), baseline)

    def test_metrics_match_stable_sort_and_support_mask(self):
        rng = np.random.default_rng(3)
        scores = rng.integers(0, 5, size=(12, 230)).astype(np.float32)
        targets = rng.integers(3, 230, size=12)
        scores[:, :3] = -np.inf
        metrics = metric_arrays(scores, targets)
        order = np.argsort(-scores, axis=1, kind="stable")
        rank = np.argmax(order == targets[:, None], axis=1)
        np.testing.assert_array_equal(metrics[:, 0], rank < 10)
        np.testing.assert_array_equal(metrics[:, 2], rank < 20)
        np.testing.assert_array_equal(metrics[:, 4], rank < 200)
        np.testing.assert_allclose(metrics[:, 3], (rank < 20) / np.log2(rank + 2))
        normalized = standardize_scores(scores)
        self.assertTrue(np.isneginf(normalized[:, :3]).all())
        np.testing.assert_allclose(normalized[:, 3:].mean(axis=1), 0, atol=1e-6)
        np.testing.assert_allclose(normalized[:, 3:].std(axis=1), 1, atol=1e-6)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable in local environment")
class PairingTorchTests(unittest.TestCase):
    def test_finite_gradients_and_tensor_parameter_matching(self):
        import torch
        torch.set_num_threads(1)
        rng = np.random.default_rng(10)
        users = rng.normal(size=(6, 1024)).astype(np.float32)
        items = rng.normal(size=(240, 1024)).astype(np.float32)
        parameters = []
        initial = []
        for variant in VARIANTS[1:]:
            torch.manual_seed(2026)
            model = make_model(torch, users, items, variant)
            parameters.append(sum(p.numel() for p in model.parameters()))
            initial.append([p.detach().clone() for p in model.parameters()])
            scores = model(torch.arange(6))
            loss = torch.nn.functional.cross_entropy(scores, torch.arange(6))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))
        self.assertEqual(len(set(parameters)), 1)
        self.assertTrue(all(torch.equal(a, b) for group in initial[1:] for a, b in zip(initial[0], group)))

    def test_all_arms_synthetic_smoke(self):
        rng = np.random.default_rng(80)
        with tempfile.TemporaryDirectory(prefix="pairing_smoke_") as temporary:
            source = Path(temporary) / "source"
            root = Path(temporary) / "run"
            seed = 2026
            prepared = source / "prepared" / str(seed)
            shared = source / "shared" / str(seed)
            features_dir = source / "features"
            for directory in (prepared, shared, features_dir):
                directory.mkdir(parents=True)
            users, items = 30, 240
            targets = np.arange(users, dtype=np.int64)
            support = np.zeros((users, items), dtype=bool)
            for user in range(users):
                support[user, 100 + user:103 + user] = True
            base = rng.normal(size=(users, items)).astype(np.float32)
            relevance = rng.normal(size=(users, items)).astype(np.float32)
            relevance[support] = -np.inf
            np.save(prepared / "base_scores.npy", base)
            np.save(prepared / "support_mask.npy", support)
            np.save(prepared / "targets.npy", targets)
            np.save(prepared / "roles.npy", np.arange(users, dtype=np.int64) % 3)
            np.save(features_dir / "global_bank.npy", rng.normal(size=(items, 2, 512)).astype(np.float32))
            np.save(shared / "relevance_scores.npy", standardize_scores(relevance))
            (shared / "ready.json").write_text(json.dumps({"seed": seed, "alpha": 0.25,
                                                            "model_sha256": "synthetic"}), encoding="utf-8")
            original_targets = targets.copy()
            for variant in VARIANTS:
                data = Prepared(source, seed, variant)
                np.testing.assert_array_equal(data.targets, original_targets)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = main(["--root", str(root), "--source-root", str(source),
                                   "--seed", str(seed), "--variant", variant,
                                   "--device", "cpu", "--threads", "1", "--batch-size", "8", "--smoke"])
                self.assertEqual(result["status"], "smoke_passed")
                self.assertFalse(result["dev_evaluated"])
                self.assertFalse((root / "results" / str(seed) / variant / "per_user.npz").exists())
            final_root = Path(temporary) / "complete_run"
            with contextlib.redirect_stdout(io.StringIO()):
                final = main(["--root", str(final_root), "--source-root", str(source),
                              "--seed", str(seed), "--variant", "paired_cov", "--epochs", "1",
                              "--device", "cpu", "--threads", "1", "--batch-size", "8"])
            self.assertEqual(final["status"], "completed")
            self.assertEqual(final["development_evaluations"], 1)
            self.assertFalse(final["dev_used_for_training_or_selection"])
            destination = final_root / "results" / str(seed) / "paired_cov"
            self.assertTrue((destination / "selection_locked.json").exists())
            with np.load(destination / "per_user.npz") as report:
                expected_users = np.flatnonzero(np.arange(users) % 3 == 2)
                np.testing.assert_array_equal(report["users"], expected_users)
                np.testing.assert_array_equal(report["targets"], targets[expected_users])
                self.assertEqual(report["top20"].shape, (10, 20))
                self.assertFalse(np.take_along_axis(support[expected_users], report["top20"], axis=1).any())
                if final["selected_scale"] == 0:
                    np.testing.assert_array_equal(report["top20"], report["baseline_top20"])


if __name__ == "__main__":
    unittest.main()
