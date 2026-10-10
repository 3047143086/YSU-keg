"""Guard training-only dispatch, artifact identity, and original-arm gradients.

Run with the carrier src directory on PYTHONPATH, or pass --bridge-root PATH.
Tests use the actual BRIDGE methods with a tiny initialized nn.Module carrier;
full-dataset model construction is covered by the separate integration smoke.
"""

import argparse
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--bridge-root")
arguments, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining
if arguments.bridge_root:
    sys.path.insert(0, str(Path(arguments.bridge_root) / "src"))
else:
    local_carrier = Path(__file__).resolve().parents[2] / "baselines" / "BRIDGE-official" / "src"
    if local_carrier.is_dir():
        sys.path.insert(0, str(local_carrier))

import numpy as np
import torch
from scipy import sparse

from evidence import build_counterfactual_cache
from proposal import TargetIsolatedBRIDGE, _verify_training_matrix
from models.bridge import BRIDGE


TRAIN = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0]], dtype=np.float32)


class Dataset:
    def __init__(self, train=TRAIN):
        self.train = sparse.csr_matrix(train)

    def inter_matrix(self, form="csr"):
        return self.train.asformat(form)


def fake_base_init(model, config, dataset):
    torch.nn.Module.__init__(model)
    model.n_users, model.n_items = dataset.train.shape
    model.behavior_sim_method = "cosine"
    model.behavior_aggregation = "sum_sqrt"
    model.behavior_score_norm = "z"
    for name in ("use_behavior_proto", "use_private_residual", "use_residual_gate", "use_activity_simplex", "use_user_graph"):
        setattr(model, name, False)
    model.msr_weight = model.private_residual_weight = 0.0
    model.use_behavior_correction = True
    model.behavior_score_matrix = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]])
    model.coefficient = torch.nn.Parameter(torch.tensor(0.3))


def pair_loss(model, interaction):
    users, positive, negative = interaction
    pos = model._pair_behavior_scores(users, positive)
    neg = model._pair_behavior_scores(users, negative)
    return torch.nn.functional.softplus(-model.coefficient * (pos - neg)).mean()


class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = build_counterfactual_cache(sparse.csr_matrix(TRAIN), topk=2)
        self.path = Path(self.directory.name) / "evidence.npz"
        self.cache.save(self.path)
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.interaction = (torch.tensor([0, 1]), torch.tensor([1, 0]), torch.tensor([2, 3]))

    def make_model(self, mode="edge", **updates):
        config = {"evidence_mode": mode, "evidence_cache": str(self.path),
                  "evidence_cache_sha256": self.digest}
        config.update(updates)
        with mock.patch.object(BRIDGE, "__init__", fake_base_init):
            return TargetIsolatedBRIDGE(config, Dataset())

    def test_both_arms_use_event_specific_evidence_and_restore_original_dispatch(self):
        for mode in ("edge", "history"):
            model = self.make_model(mode)
            expected = self.cache.get_pair(*[v.numpy() for v in self.interaction], mode=mode)
            expected_loss = torch.nn.functional.softplus(
                -model.coefficient * torch.tensor(expected[0] - expected[1])).mean()
            with mock.patch.object(BRIDGE, "calculate_loss", pair_loss):
                actual_loss = model.calculate_loss(self.interaction)
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
            self.assertIsNone(model._evidence_context)
            torch.testing.assert_close(model._pair_behavior_scores(self.interaction[0], self.interaction[1]),
                                       model.behavior_score_matrix[self.interaction[0], self.interaction[1]])
            report = model.train_diagnostics(reset=True)
            self.assertEqual(report["count"], 2)
            self.assertEqual(report["batches"], 1)
            self.assertAlmostEqual(report["mean_adjusted_positive_z"], float(expected[0].mean()), places=6)
            self.assertEqual(model.train_diagnostics()["count"], 0)

    def test_soft_edge_interpolates_exactly_between_original_and_edge(self):
        model = self.make_model("soft_edge", isolation_strength=0.5)
        edge = self.cache.get_pair(*[v.numpy() for v in self.interaction], mode="edge")
        original = [model.behavior_score_matrix[self.interaction[0], items].numpy()
                    for items in self.interaction[1:]]
        expected = [base + 0.5 * (value - base) for value, base in zip(edge, original)]
        with mock.patch.object(BRIDGE, "calculate_loss", pair_loss):
            model.calculate_loss(self.interaction)
        actual = model._evidence_statistics
        self.assertAlmostEqual(actual["adjusted_positive_z"] / 2.0, float(expected[0].sum()) / 2.0, places=6)
        self.assertAlmostEqual(actual["adjusted_negative_z"] / 2.0, float(expected[1].sum()) / 2.0, places=6)

    def test_isolation_strength_rejects_invalid_values(self):
        for value in (-0.01, 1.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.make_model("soft_edge", isolation_strength=value)

    def test_original_arm_preserves_loss_gradient_and_random_state(self):
        model = self.make_model("original", evidence_cache="missing", evidence_cache_sha256="")
        reference = BRIDGE.__new__(BRIDGE)
        fake_base_init(reference, {}, Dataset())
        state = torch.get_rng_state().clone()
        numpy_state = np.random.get_state()
        with mock.patch.object(BRIDGE, "calculate_loss", pair_loss):
            loss = model.calculate_loss(self.interaction)
            expected = reference.calculate_loss(self.interaction)
        loss.backward()
        expected.backward()
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        torch.testing.assert_close(model.coefficient.grad, reference.coefficient.grad, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        current_numpy_state = np.random.get_state()
        self.assertEqual(numpy_state[0], current_numpy_state[0])
        np.testing.assert_array_equal(numpy_state[1], current_numpy_state[1])
        self.assertEqual(numpy_state[2:], current_numpy_state[2:])
        self.assertIsNone(model._evidence_cache)

    def test_dispatch_mismatch_and_exception_always_clear_context(self):
        def wrong_order(model, interaction):
            return model._pair_behavior_scores(interaction[0], interaction[2])

        def failure(model, interaction):
            model._pair_behavior_scores(interaction[0], interaction[1])
            raise ArithmeticError("forced downstream error")

        for function, error in ((wrong_order, RuntimeError), (failure, ArithmeticError)):
            model = self.make_model()
            with mock.patch.object(BRIDGE, "calculate_loss", function):
                with self.assertRaises(error):
                    model.calculate_loss(self.interaction)
            self.assertIsNone(model._evidence_context)
            self.assertEqual(model.train_diagnostics()["count"], 0)

    def test_missing_or_extra_parent_queries_fail(self):
        def missing(model, interaction):
            return model._pair_behavior_scores(interaction[0], interaction[1])

        def extra(model, interaction):
            pair_loss(model, interaction)
            return model._pair_behavior_scores(interaction[0], interaction[1])

        for function in (missing, extra):
            model = self.make_model()
            with mock.patch.object(BRIDGE, "calculate_loss", function):
                with self.assertRaises(RuntimeError):
                    model.calculate_loss(self.interaction)
            self.assertIsNone(model._evidence_context)

    def test_cache_hash_mismatch_and_relative_path_fail(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.make_model(evidence_cache_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "absolute"):
            self.make_model(evidence_cache="evidence.npz")

    def test_changed_training_ids_and_edge_metadata_fail(self):
        changed = TRAIN.copy()
        changed[0, 1], changed[0, 2] = 0, 1
        with self.assertRaisesRegex(ValueError, "training matrix differs"):
            _verify_training_matrix(self.cache, Dataset(changed), 3, 4)
        self.cache.edge_items = self.cache.edge_items.copy()
        self.cache.edge_items[0] = 3
        with self.assertRaisesRegex(ValueError, "training edge IDs"):
            _verify_training_matrix(self.cache, Dataset(), 3, 4)

    def test_inference_and_parameter_schema_are_inherited(self):
        self.assertIs(TargetIsolatedBRIDGE.full_sort_predict, BRIDGE.full_sort_predict)
        original = self.make_model("original")
        adjusted = self.make_model("edge")
        self.assertEqual(list(original.state_dict()), list(adjusted.state_dict()))
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, adjusted.state_dict()[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
