"""Exercise fixed decisions without importing the predecessor or an encoder."""
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location("residual_executor", Path(__file__).with_name("experiment.py"))
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.pairs = {name: {"mean_delta": .006, "user_bootstrap_95ci": [.001, .011]}
                      for name in experiment.CONTROLS}
        self.halves = {name: [.004, .008] for name in experiment.CONTROLS}

    def test_passes_at_coverage_and_delta_boundary(self):
        for row in self.pairs.values():
            row["mean_delta"] = .005
        result = experiment.decide(.80, self.pairs, self.halves)
        self.assertEqual(result["status"], "keep")
        self.assertTrue(result["passed"])
        for name in ("formal_training_authorized_by_this_screen", "novelty_established", "publication_evidence"):
            self.assertFalse(result[name])

    def test_coverage_below_boundary_discards(self):
        self.assertFalse(experiment.decide(.799999, self.pairs, self.halves)["passed"])

    def test_every_control_must_pass_delta(self):
        for name in experiment.CONTROLS:
            pairs = copy.deepcopy(self.pairs)
            pairs[name]["mean_delta"] = .004999
            self.assertFalse(experiment.decide(.90, pairs, self.halves)["passed"])

    def test_every_lower_bound_must_be_strictly_positive(self):
        for name in experiment.CONTROLS:
            pairs = copy.deepcopy(self.pairs)
            pairs[name]["user_bootstrap_95ci"][0] = 0
            self.assertFalse(experiment.decide(.90, pairs, self.halves)["passed"])

    def test_every_control_and_hash_half_must_be_positive(self):
        for name in experiment.CONTROLS:
            for half in (0, 1):
                halves = copy.deepcopy(self.halves)
                halves[name][half] = 0
                self.assertFalse(experiment.decide(.90, self.pairs, halves)["passed"])

    def test_missing_or_extra_control_is_an_error(self):
        for names in (experiment.CONTROLS[:-1], experiment.CONTROLS + ("unexpected",)):
            pairs = {name: self.pairs.get(name, self.pairs["raw_own"]) for name in names}
            with self.assertRaises(ValueError):
                experiment.decide(.90, pairs, self.halves)

    def test_missing_half_is_an_error(self):
        self.halves["residual_donor"] = [.01]
        with self.assertRaises(ValueError):
            experiment.decide(.90, self.pairs, self.halves)

    def test_nonfinite_or_out_of_domain_is_an_error(self):
        for coverage in (float("nan"), float("inf"), -.01, 1.01, True):
            with self.assertRaises(ValueError):
                experiment.decide(coverage, self.pairs, self.halves)
        self.pairs["residual_donor"]["mean_delta"] = float("nan")
        with self.assertRaises(ValueError):
            experiment.decide(.90, self.pairs, self.halves)

    def test_reversed_interval_is_an_error(self):
        self.pairs["item_content"]["user_bootstrap_95ci"] = [.02, .01]
        with self.assertRaises(ValueError):
            experiment.decide(.90, self.pairs, self.halves)

    def test_import_and_decision_leave_encoder_unloaded(self):
        experiment.decide(.90, self.pairs, self.halves)
        for name in ("torch", "clip", "remote_session", "frozen_review_signal_executor"):
            self.assertNotIn(name, sys.modules)

    def test_predecessor_hash_is_checked_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "untrusted.py"
            source.write_text("raise AssertionError('this module must not execute')\n")
            manifest = {"dependencies": {"review_signal_experiment":
                        {"path": str(source), "sha256": experiment.sha256(source)}}}
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                experiment.load_helpers(manifest, 0)

    def test_bundle_guard_rejects_core_mutation(self):
        original_root = experiment.ROOT
        try:
            with tempfile.TemporaryDirectory() as directory:
                experiment.ROOT = Path(directory)
                for name in experiment.PROTECTED_BUNDLE_FILES:
                    (experiment.ROOT / name).write_text(name, encoding="utf-8")
                hashes = {name: experiment.sha256(experiment.ROOT / name)
                          for name in experiment.PROTECTED_BUNDLE_FILES}
                (experiment.ROOT / "deployment.json").write_text(
                    json.dumps({"bundle_sha256": hashes}), encoding="utf-8")
                self.assertEqual(experiment.verify_bundle(), hashes)
                (experiment.ROOT / "residual_core.py").write_text("modified", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "bundle changed"):
                    experiment.verify_bundle()
        finally:
            experiment.ROOT = original_root

    def test_bundle_guard_rejects_missing_input_audit_hash(self):
        original_root = experiment.ROOT
        try:
            with tempfile.TemporaryDirectory() as directory:
                experiment.ROOT = Path(directory)
                (experiment.ROOT / "deployment.json").write_text(
                    json.dumps({"bundle_sha256": {}}), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "lacks protected bundle hashes"):
                    experiment.verify_bundle()
        finally:
            experiment.ROOT = original_root


if __name__ == "__main__":
    unittest.main()
