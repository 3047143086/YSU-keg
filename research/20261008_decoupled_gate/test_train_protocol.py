"""Executable protocol checks, requiring no local Torch installation."""
import unittest
from types import SimpleNamespace

from train import Selection, scalarize, training_loop


class ProtocolTest(unittest.TestCase):
    def run_fake(self, recalls, patience=20, losses=None, diagnostics=None):
        events, saved, progress = [], [], []
        valid = object()
        values = iter(recalls)
        losses = iter(losses or [1.0] * len(recalls))

        class Model:
            def pre_epoch_processing(self):
                events.append('pre')

            def post_epoch_processing(self):
                events.append('post')

        class FakeTrainer:
            model = Model()
            optimizer = SimpleNamespace(param_groups=[{'lr': 0.0001}])
            lr_scheduler = SimpleNamespace(step=lambda: events.append('scheduler'))

            def fit(self, *args, **kwargs):
                raise AssertionError('Legacy fit may expose the test set')

            def _train_epoch(self, train, epoch, loss_func):
                events.append('train')
                return next(losses), []

            def evaluate(self, data):
                if data is not valid:
                    raise AssertionError('Only the validation loader may be evaluated')
                events.append('validation')
                return {'recall@20': next(values), 'ndcg@20': 0.05}

        if diagnostics is not None:
            FakeTrainer.model.train_diagnostics = lambda: diagnostics
        selection, epochs = training_loop(
            FakeTrainer(), object(), valid, len(recalls), patience, None,
            lambda epoch, score, metrics: saved.append((epoch, score)), progress.append)
        return selection, epochs, saved, progress, events

    def test_near_ties_keep_earlier_checkpoint_and_stop_only_after_patience(self):
        selection, epochs, saved, progress, events = self.run_fake(
            [0.10001, 0.10004, 0.099, 0.10003, 0.8], patience=2)
        self.assertEqual(epochs, 4)
        self.assertEqual(saved, [(0, 0.1)])
        self.assertEqual(selection.best_epoch, 0)
        self.assertEqual([row['early_stop'] for row in progress], [False, False, False, True])
        self.assertEqual(events, ['pre', 'train', 'scheduler', 'post', 'validation'] * 4)

    def test_real_improvement_resets_patience_and_checkpoint_is_validation_selected(self):
        selection, epochs, saved, progress, _ = self.run_fake(
            [0.1, 0.099, 0.10006, 0.10008, 0.09, 0.08], patience=2)
        self.assertEqual(epochs, 6)
        self.assertEqual(saved, [(0, 0.1), (2, 0.1001)])
        self.assertEqual(selection.best_epoch, 2)
        self.assertEqual(progress[2]['bad_epochs'], 0)

    def test_nonfinite_losses_and_validation_never_finish_successfully(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                with self.assertRaises(FloatingPointError):
                    self.run_fake([0.1], losses=[value])
                with self.assertRaises(FloatingPointError):
                    self.run_fake([value])

    def test_all_loss_components_checked(self):
        with self.assertRaises(FloatingPointError):
            self.run_fake([0.1], losses=[(1.0, float('inf'))])

    def test_training_diagnostics_are_recorded_as_finite_json(self):
        diagnostics = {'positive_mean': 0.5, 'counts': [3, 5]}
        _, _, _, progress, _ = self.run_fake([0.1], diagnostics=diagnostics)
        self.assertEqual(progress[0]['train_diagnostics'], diagnostics)
        with self.assertRaises(ValueError):
            self.run_fake([0.1], diagnostics={'margin': float('nan')})
        with self.assertRaises(TypeError):
            self.run_fake([0.1], diagnostics={'tensor': object()})

    def test_scalarize_preserves_real_lists_and_refuses_silent_parameter_selection(self):
        config = {'hyper_parameters': ['seed', 'embedding_size'], 'seed': [2026],
                  'embedding_size': [64], 'topk': [5, 10, 20, 50]}
        scalarize(config)
        self.assertEqual(config['seed'], 2026)
        self.assertEqual(config['embedding_size'], 64)
        self.assertEqual(config['topk'], [5, 10, 20, 50])
        for value in ([], [64, 128]):
            with self.assertRaises(ValueError):
                scalarize({'hyper_parameters': ['embedding_size'], 'embedding_size': value})

    def test_invalid_selection_never_updates_state(self):
        selection = Selection()
        selection.update(0.1, 0)
        with self.assertRaises(FloatingPointError):
            selection.update(float('nan'), 1)
        self.assertEqual(selection.best_epoch, 0)
        self.assertEqual(selection.bad_epochs, 0)


if __name__ == '__main__':
    unittest.main()
