import threading
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from eval_cache import EvalCacheInvalidated, EvalForwardCache, cached_eval_pass


class DualFrequencyEncoder:
    calls = 0
    consume_rng = False

    @staticmethod
    def forward(model):
        DualFrequencyEncoder.calls += 1
        if DualFrequencyEncoder.consume_rng:
            torch.rand(1)
        return (model.user + 0, model.item + 0, model.user.sum() * 0,
                [model.user + 0], [model.item + 0])


@unittest.skipUnless(torch is not None, "actual MMRec environment provides PyTorch")
class LifecycleTests(unittest.TestCase):
    def setUp(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.register_buffer('marker', torch.tensor(0.0))
                self.user = torch.tensor([[1., 0.], [0., 1.]])
                self.item = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
                self.flag = False

            def full_sort_predict(self, interaction):
                users = interaction[0]
                users_all, items, _, _, _ = DualFrequencyEncoder.forward(self)
                return torch.matmul(users_all[users], items.transpose(0, 1))

        DualFrequencyEncoder.calls = 0
        DualFrequencyEncoder.consume_rng = False
        self.Encoder, self.Model = DualFrequencyEncoder, Model
        self.users = (torch.tensor([0, 1]),)

    def _mutation_rejected(self, mutation):
        model = self.Model().eval()
        with torch.no_grad(), cached_eval_pass(model, model.full_sort_predict, self.Encoder) as (cached, cache):
            cached(self.users)
            mutation(model, cache)
            with self.assertRaises(EvalCacheInvalidated):
                cached(self.users)

    def test_load_state_dict_invalidates(self):
        self._mutation_rejected(lambda m, c: m.load_state_dict(m.state_dict()))

    def test_buffer_change_invalidates(self):
        self._mutation_rejected(lambda m, c: m.marker.add_(1))

    def test_plain_tensor_change_invalidates(self):
        self._mutation_rejected(lambda m, c: m.item.add_(1))

    def test_scalar_change_invalidates(self):
        self._mutation_rejected(lambda m, c: setattr(m, 'flag', True))

    def test_training_transition_invalidates(self):
        self._mutation_rejected(lambda m, c: m.train())

    def test_output_change_invalidates(self):
        self._mutation_rejected(lambda m, c: c._outputs[1].add_(1))

    def test_wrong_model_and_thread_rejected(self):
        model, other = self.Model().eval(), self.Model().eval()
        with torch.no_grad(), cached_eval_pass(model, model.full_sort_predict, self.Encoder) as (cached, cache):
            with self.assertRaises(EvalCacheInvalidated):
                cache.forward(other, self.Encoder)
            failures = []
            def foreign():
                with torch.no_grad():
                    try:
                        cache.forward(model, self.Encoder)
                    except EvalCacheInvalidated:
                        failures.append(True)
            thread = threading.Thread(target=foreign)
            thread.start(); thread.join()
            self.assertEqual(failures, [True])

    def test_concurrent_same_model_pass_rejected(self):
        model = self.Model().eval()
        with torch.no_grad(), cached_eval_pass(model, model.full_sort_predict, self.Encoder):
            with self.assertRaises(EvalCacheInvalidated):
                with cached_eval_pass(model, model.full_sort_predict, self.Encoder):
                    pass

    def test_closed_and_reentered_cache_rejected(self):
        model = self.Model().eval()
        with torch.no_grad():
            cache = EvalForwardCache(model)
            with cache:
                cache.forward(model, self.Encoder)
            self.assertIsNone(cache._outputs)
            with self.assertRaises(EvalCacheInvalidated):
                cache.forward(model, self.Encoder)
            with self.assertRaises(EvalCacheInvalidated):
                with cache:
                    pass

    def test_exception_cleans_and_new_pass_recomputes(self):
        model = self.Model().eval()
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                with cached_eval_pass(model, model.full_sort_predict, self.Encoder) as (cached, cache):
                    cached(self.users)
                    raise RuntimeError('fixture failure')
            self.assertIsNone(cache._outputs)
            with cached_eval_pass(model, model.full_sort_predict, self.Encoder) as (cached, _):
                cached(self.users)
            self.assertEqual(self.Encoder.calls, 2)

    def test_rng_consuming_encoder_rejected(self):
        model = self.Model().eval()
        before = torch.get_rng_state().clone()
        try:
            self.Encoder.consume_rng = True
            with torch.no_grad(), cached_eval_pass(model, model.full_sort_predict, self.Encoder) as (cached, cache):
                with self.assertRaises(EvalCacheInvalidated):
                    cached(self.users)
                self.assertIsNone(cache._outputs)
        finally:
            torch.set_rng_state(before)

    def test_external_rng_change_between_batches_rejected(self):
        model = self.Model().eval()
        before = torch.get_rng_state().clone()
        try:
            with torch.no_grad(), cached_eval_pass(model, model.full_sort_predict, self.Encoder) as (cached, _):
                cached(self.users)
                torch.rand(1)
                with self.assertRaises(EvalCacheInvalidated):
                    cached(self.users)
        finally:
            torch.set_rng_state(before)

    def test_original_method_global_and_code_unchanged(self):
        model = self.Model().eval()
        original = self.Model.full_sort_predict
        code, globals_id = original.__code__, id(original.__globals__)
        encoder = original.__globals__['DualFrequencyEncoder']
        with torch.no_grad(), cached_eval_pass(model, original, self.Encoder) as (cached, _):
            cached(self.users)
            self.assertIs(cached.__func__.__code__, code)
        self.assertIs(self.Model.full_sort_predict, original)
        self.assertEqual(id(original.__globals__), globals_id)
        self.assertIs(original.__globals__['DualFrequencyEncoder'], encoder)

    def test_mode_guard_receipt_states_round_trip_limitation(self):
        model = self.Model().eval()
        with torch.no_grad(), cached_eval_pass(model, model.full_sort_predict, self.Encoder) as (cached, cache):
            cached(self.users)
            model.train(); model.eval()
            # A round trip without forward/state updates leaves no observable
            # token; this is explicitly prohibited by the exclusive-pass scope.
            cached(self.users)
            self.assertEqual(cache.receipt()['mode_guard'],
                'current_mode_only; intermediate_mode_transitions_prohibited')


if __name__ == '__main__':
    unittest.main()
