import unittest
try:
    import torch
except ModuleNotFoundError:
    torch = None
from checkpoint_structure import align_original_gpu_structure


@unittest.skipUnless(torch is not None, 'MMRec environment provides Torch')
class StructureTests(unittest.TestCase):
    def fixture(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.result_embed = torch.nn.Parameter(torch.ones(2))
                for name in ('id', 'v', 't'):
                    gcn = torch.nn.Module()
                    gcn.preference = torch.nn.Parameter(torch.tensor([1., 2.]))
                    setattr(self, name + '_gcn', gcn)
        model = Model()
        state = {k:v.clone() for k,v in model.state_dict().items() if k != 'result_embed'}
        for name in ('id', 'v', 't'):
            state[name + '_preference'] = state[name + '_gcn.preference'].clone()
        return model, state

    def test_strict_alias_and_state_coverage(self):
        model, state = self.fixture()
        receipt = align_original_gpu_structure(model, state, 'self.result_embed = value')
        self.assertTrue(receipt['strict_load'])
        self.assertEqual(set(model.state_dict()), set(state))
        self.assertIs(model.id_preference, model.id_gcn.preference)
        self.assertNotIn('result_embed', model._parameters)
        self.assertTrue(all(torch.equal(model.state_dict()[k], state[k]) for k in state))

    def test_alias_mismatch_rejected(self):
        model, state = self.fixture()
        state['id_preference'][0] += 1
        with self.assertRaisesRegex(RuntimeError, 'alias disagrees'):
            align_original_gpu_structure(model, state, 'self.result_embed = value')

    def test_additional_key_mismatch_rejected(self):
        model, state = self.fixture()
        state['unknown'] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, 'mismatch differs'):
            align_original_gpu_structure(model, state, 'self.result_embed = value')

    def test_used_result_embed_rejected(self):
        model, state = self.fixture()
        with self.assertRaisesRegex(RuntimeError, 'construction-only'):
            align_original_gpu_structure(model, state, 'self.result_embed = value\nx = self.result_embed')

    def test_plain_tensor_gpu_registration_shape_supported(self):
        model, state = self.fixture()
        value = model.result_embed.detach()
        delattr(model, 'result_embed')
        model.result_embed = value
        result = align_original_gpu_structure(model, state, 'self.result_embed = value')
        self.assertTrue(result['original_gpu_registration_shape_preserved'])
        self.assertEqual(set(model.state_dict()), set(state))
