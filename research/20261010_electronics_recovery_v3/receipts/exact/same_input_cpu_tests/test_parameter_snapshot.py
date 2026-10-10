"""Actual CPU Torch checks for alias paths, buffers and old API compatibility."""
import unittest

import torch
from parameter_snapshot import all_named_parameters, parameter_coverage, synthetic_check


class ParameterSnapshotTests(unittest.TestCase):
    def test_actual_torch_shared_parameter_and_child_aliases(self):
        result = synthetic_check()
        self.assertTrue(result['state_unchanged'])
        self.assertEqual(result['coverage']['registered_path_count'], 4)
        self.assertEqual(result['coverage']['unique_parameter_object_count'], 1)
        self.assertEqual(result['coverage']['buffer_paths_excluded'], ['child.buffer', 'child_alias.buffer'])

    def test_api_with_no_remove_duplicate_keyword_is_not_called(self):
        class OldModel(torch.nn.Module):
            def named_parameters(self):
                raise AssertionError('Version-sensitive public enumeration must not be called')
        model = OldModel()
        model.weight = torch.nn.Parameter(torch.zeros(3))
        model.alias = model.weight
        self.assertEqual(set(dict(all_named_parameters(model))), {'weight', 'alias'})
        self.assertTrue(parameter_coverage(model)['all_parameter_state_paths_covered'])

    def test_none_parameter_and_child_are_skipped(self):
        model = torch.nn.Module()
        model.register_parameter('empty', None)
        model.add_module('empty_child', None)
        self.assertEqual(all_named_parameters(model), [])
        self.assertEqual(parameter_coverage(model)['registered_path_count'], 0)

    def test_registration_cycle_is_rejected(self):
        model = torch.nn.Module()
        model.add_module('loop', model)
        with self.assertRaises(ValueError): all_named_parameters(model)

    def test_separate_objects_have_separate_identities(self):
        model = torch.nn.Linear(3, 2)
        result = parameter_coverage(model)
        self.assertEqual(result['registered_path_count'], 2)
        self.assertEqual(result['unique_parameter_object_count'], 2)


if __name__ == '__main__':
    unittest.main()
