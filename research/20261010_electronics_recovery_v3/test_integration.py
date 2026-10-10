"""Instance-local contract checks without Torch or carrier imports."""
import unittest
from types import SimpleNamespace
from integration import install_projector


class FrequencyDecompositionModule:
    decomp_method, M = 'svd', 3
    def frequency_decompose_svd(self, representation):
        return ['original']


class Model:
    def __init__(self):
        self.FeqDecomposeOperator = FrequencyDecompositionModule()
        self.parameter = object()
    def named_parameters(self):
        return [('fixed', self.parameter)]
    def state_dict(self):
        return {'fixed': self.parameter}


class IntegrationTests(unittest.TestCase):
    def test_parameters_and_other_instance_are_preserved(self):
        model, untouched = Model(), Model()
        original_class_method = FrequencyDecompositionModule.frequency_decompose_svd
        value = SimpleNamespace(shape=(12, 6))
        outputs = [SimpleNamespace(shape=value.shape) for _ in range(3)]
        parameter = model.parameter
        counts = install_projector(model, lambda rep, bands: tuple(outputs))
        self.assertEqual(model.FeqDecomposeOperator.frequency_decompose_svd(value), outputs)
        self.assertEqual(counts['calls'], 1)
        self.assertIs(model.parameter, parameter)
        self.assertIs(FrequencyDecompositionModule.frequency_decompose_svd, original_class_method)
        self.assertEqual(untouched.FeqDecomposeOperator.frequency_decompose_svd(value), ['original'])

    def test_repeated_install_and_changed_band_contract_are_rejected(self):
        value = SimpleNamespace(shape=(12, 6))
        model = Model()
        install_projector(model, lambda rep, bands: ())
        with self.assertRaises(RuntimeError):
            model.FeqDecomposeOperator.frequency_decompose_svd(value)
        with self.assertRaises(RuntimeError):
            install_projector(model, lambda rep, bands: ())
        model = Model()
        model.FeqDecomposeOperator.M = 2
        with self.assertRaises(ValueError):
            install_projector(model, lambda rep, bands: ())


if __name__ == '__main__':
    unittest.main()
