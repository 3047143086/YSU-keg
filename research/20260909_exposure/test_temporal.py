import tempfile
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from temporal_train import (VARIANTS, build_training_examples, chronological_split,
                            context_for, metric_rows, summary_vector, train, METRICS)


class TemporalProtocolTests(unittest.TestCase):
    def test_strict_split_and_no_future_target_in_context(self):
        rows = [(4, 10, 1), (4, 11, 2), (4, 12, 3), (4, 13, 4),
                (7, 20, 1), (7, 21, 2), (7, 22, 3)]
        train, calibration, test = chronological_split(rows)
        self.assertEqual(train[4], (10, 11))
        self.assertEqual(calibration[4], 12)
        self.assertEqual(test[4], 13)
        self.assertNotIn(calibration[4], context_for(4, train, calibration, "calibration"))
        self.assertNotIn(test[4], context_for(4, train, calibration, "test"))
        self.assertGreater(4, 0)

    def test_training_contexts_are_prefixes_only(self):
        features = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
        train = {1: (0, 1, 2, 3)}
        for variant in VARIANTS:
            examples = build_training_examples(train, features, variant)
            self.assertEqual([x[2] for x in examples], [1, 2, 3])
            for _, _, target, history in examples:
                self.assertNotIn(target, history)

    def test_item_ties_use_item_id_order(self):
        scores = np.zeros((1, 205), dtype=np.float32)
        rows = metric_rows(scores, [0], [set()])
        self.assertEqual(rows[0, 0], 1.0)
        self.assertEqual(rows[0, 2], 1.0)

    def test_duplicate_targets_fail_instead_of_leaking(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            chronological_split([(1, 2, 1), (1, 2, 2), (1, 3, 3)])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
    def test_formal_cpu_run_writes_metrics(self):
        with tempfile.TemporaryDirectory(prefix="temporal_screen_test_") as directory:
            root = Path(directory)
            interaction_file = root / "baby.inter"
            interaction_file.write_text(
                "userID\titemID\ttimestamp\n"
                "1\t0\t1\n1\t1\t2\n1\t2\t3\n1\t3\t4\n"
                "2\t4\t1\n2\t5\t2\n2\t6\t3\n2\t7\t4\n",
                encoding="utf-8")
            feature_file = root / "global_bank.npy"
            np.save(feature_file, np.random.default_rng(8).normal(size=(8, 2, 6)).astype(np.float32))
            args = SimpleNamespace(interactions=interaction_file, feature_bank=feature_file,
                output=root / "out", seed=2026, variant="learned_two_scale", epochs=1,
                batch_size=2, negatives=2, hidden=4, learning_rate=0.01,
                weight_decay=0.0, threads=1, device="cpu", smoke=False)
            result = train(args)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(set(result["test_metrics"]), set(METRICS))
            self.assertTrue((root / "out" / "result.json").is_file())
            with np.load(root / "out" / "per_user.npz") as report:
                self.assertEqual(report["top20"].shape, (2, 8))


if __name__ == "__main__":
    unittest.main()
