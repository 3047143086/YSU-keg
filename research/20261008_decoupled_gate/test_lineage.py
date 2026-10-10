import copy
import unittest

from lineage import validate_config, validate_result


class LineageTests(unittest.TestCase):
    def result(self):
        return {'status': 'completed', 'dataset': 'baby', 'seed': 2026, 'model': 'BRIDGE',
                'test_evaluated': False, 'test_loader_constructed': False,
                'inputs_unchanged': True, 'fresh_initialization': True,
                'selection_metric': 'rounded_4_validation_recall@20',
                'metrics': {'recall@10': .07, 'ndcg@10': .04, 'recall@20': .1, 'ndcg@20': .05}}

    def config(self):
        return {'budget_overridden': False,
                'arguments': {'dataset': 'baby', 'seed': 2026, 'max_epochs': 300, 'device': 'cuda',
                              'cuda_memory_fraction': .55, 'train_batch_size': None,
                              'learning_rate': None, 'smoke': False, 'plugin_module': None,
                              'plugin_class': None, 'plugin_config': None},
                'effective_config': {'stopping_step': 20, 'eval_step': 1, 'eval_batch_size': 64,
                                     'train_batch_size': 2048, 'feature_ablation': 'all',
                                     'use_behavior_correction': True, 'use_topk_correction': True,
                                     'use_behavior_gate': True, 'use_calibration_gate': True}}

    def test_never_reuse_interrupted_test_selected_or_non_original_results(self):
        validate_result(self.result(), 'baby')
        for key, value in (('status', 'running'), ('seed', 2027), ('model', 'DecoupledGateBRIDGE'),
                           ('test_evaluated', True), ('test_loader_constructed', True),
                           ('fresh_initialization', False), ('inputs_unchanged', False)):
            result = self.result()
            result[key] = value
            with self.assertRaises(RuntimeError):
                validate_result(result, 'baby')
        result = self.result()
        result['metrics']['recall@20'] = float('nan')
        with self.assertRaises(RuntimeError):
            validate_result(result, 'baby')

    def test_full_frozen_budget_and_no_plugin_are_required(self):
        validate_config(self.config(), 'baby')
        for key, value in (('max_epochs', 1), ('smoke', True), ('plugin_module', 'proposal'),
                           ('learning_rate', .01), ('train_batch_size', 1024)):
            config = copy.deepcopy(self.config())
            config['arguments'][key] = value
            with self.assertRaises(RuntimeError):
                validate_config(config, 'baby')
        config = self.config()
        config['effective_config']['stopping_step'] = 12
        with self.assertRaises(RuntimeError):
            validate_config(config, 'baby')


if __name__ == '__main__':
    unittest.main()
