"""Pure gate tests; importing the executor does not load an encoder or connect."""
import copy
import importlib.util
import sys
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location("review_signal_executor", Path(__file__).with_name("experiment.py"))
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.pairs = {name: {"mean_delta": .006, "user_bootstrap_95ci": [.001, .011]}
                      for name in ("same_item_donor", "item_content")}
        self.halves = {name: [.004, .008] for name in self.pairs}

    def test_fixed_gate_passes_at_coverage_and_delta_boundary(self):
        for row in self.pairs.values():
            row["mean_delta"] = .005
        result = experiment.decide(.80, self.pairs, self.halves)
        self.assertEqual(result["status"], "keep")
        self.assertTrue(result["passed"])
        self.assertFalse(result["formal_training_authorized_by_this_screen"])
        self.assertFalse(result["novelty_established"])
        self.assertFalse(result["publication_evidence"])

    def test_coverage_below_boundary_discards(self):
        self.assertEqual(experiment.decide(.799999, self.pairs, self.halves)["status"], "discard")

    def test_each_control_must_pass_delta(self):
        for name in self.pairs:
            pairs = copy.deepcopy(self.pairs)
            pairs[name]["mean_delta"] = .004999
            self.assertFalse(experiment.decide(.90, pairs, self.halves)["passed"])

    def test_each_confidence_lower_bound_must_be_strictly_positive(self):
        for name in self.pairs:
            pairs = copy.deepcopy(self.pairs)
            pairs[name]["user_bootstrap_95ci"][0] = 0
            self.assertFalse(experiment.decide(.90, pairs, self.halves)["passed"])

    def test_each_control_and_half_must_have_positive_delta(self):
        for name in self.halves:
            for half in (0, 1):
                halves = copy.deepcopy(self.halves)
                halves[name][half] = 0
                self.assertFalse(experiment.decide(.90, self.pairs, halves)["passed"])

    def test_missing_control_is_an_error(self):
        with self.assertRaises(ValueError):
            experiment.decide(.90, {"item_content": self.pairs["item_content"]}, self.halves)

    def test_missing_half_is_an_error(self):
        self.halves["item_content"] = [.01]
        with self.assertRaises(ValueError):
            experiment.decide(.90, self.pairs, self.halves)

    def test_nonfinite_or_invalid_domain_is_an_error(self):
        for coverage in (float("nan"), float("inf"), -.01, 1.01, True):
            with self.assertRaises(ValueError):
                experiment.decide(coverage, self.pairs, self.halves)
        self.pairs["item_content"]["mean_delta"] = float("nan")
        with self.assertRaises(ValueError):
            experiment.decide(.90, self.pairs, self.halves)

    def test_reversed_interval_is_an_error(self):
        self.pairs["item_content"]["user_bootstrap_95ci"] = [.02, .01]
        with self.assertRaises(ValueError):
            experiment.decide(.90, self.pairs, self.halves)

    def test_import_and_decision_leave_encoder_unloaded(self):
        experiment.decide(.90, self.pairs, self.halves)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("clip", sys.modules)
        self.assertNotIn("remote_session", sys.modules)


if __name__ == "__main__":
    unittest.main()
