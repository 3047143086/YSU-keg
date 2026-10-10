import unittest

try:
    import torch
except ModuleNotFoundError:  # local workstation intentionally has no Torch
    torch = None

from eval_cache import EvalCacheInvalidated, cached_eval_pass, masked_topk


class DualFrequencyEncoder:
    calls = 0

    @staticmethod
    def forward(model):
        DualFrequencyEncoder.calls += 1
        return (model.user, model.item, model.user * 0,
                [model.user], [model.item])


@unittest.skipUnless(torch is not None, "actual MMRec environment provides PyTorch")
class EvalCacheTests(unittest.TestCase):
    def setUp(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.user = torch.tensor([[1., 0.], [0., 1.]])
                self.item = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])

            def full_sort_predict(self, interaction):
                users = interaction[0]
                _, items, _, _, _ = DualFrequencyEncoder.forward(self)
                return torch.matmul(self.user[users], items.transpose(0, 1))

        DualFrequencyEncoder.calls = 0
        self.Encoder, self.Model = DualFrequencyEncoder, Model

    def test_single_generation_and_exact_masked_topk(self):
        model = self.Model().eval()
        direct = model.full_sort_predict
        users = (torch.tensor([0, 1]),)
        expected = direct(users)
        with torch.no_grad(), cached_eval_pass(model, model.full_sort_predict, self.Encoder) as pair:
            cached, cache = pair
            actual1 = cached(users)
            actual2 = cached(users)
            self.assertTrue(torch.equal(expected, actual1))
            self.assertTrue(torch.equal(actual1, actual2))
            self.assertEqual(cache.misses, 1)
            self.assertEqual(cache.hits, 1)
            positives = [[0], [1]]
            self.assertTrue(torch.equal(masked_topk(expected, positives, 2),
                                        masked_topk(actual2, positives, 2)))
        self.assertEqual(self.Encoder.calls, 2)  # direct + one cached generation

    def test_training_or_grad_context_is_rejected(self):
        model = self.Model()
        with self.assertRaises(EvalCacheInvalidated):
            with cached_eval_pass(model, model.full_sort_predict, self.Encoder):
                pass
        model.eval()
        with self.assertRaises(EvalCacheInvalidated):
            with cached_eval_pass(model, model.full_sort_predict, self.Encoder):
                model.full_sort_predict((torch.tensor([0]),))

    def test_parameter_change_invalidates_without_reuse(self):
        model = self.Model().eval()
        with torch.no_grad(), cached_eval_pass(model, model.full_sort_predict, self.Encoder) as pair:
            cached, _ = pair
            cached((torch.tensor([0]),))
            with torch.no_grad():
                model.weight.add_(1.)
            with self.assertRaises(EvalCacheInvalidated):
                cached((torch.tensor([0]),))

    def test_rng_is_unchanged_by_cache(self):
        model = self.Model().eval()
        torch.manual_seed(2026)
        before = torch.get_rng_state().clone()
        with torch.no_grad(), cached_eval_pass(model, model.full_sort_predict, self.Encoder) as pair:
            cached, _ = pair
            cached((torch.tensor([0]),))
            cached((torch.tensor([1]),))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
