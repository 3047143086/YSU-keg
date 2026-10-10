"""No partial, approximate, unused or nonfinite evidence can pass integration."""
import copy
import unittest
from integration_gate import EXPECTED_PARAMETERS, EXPECTED_SHAPES, integration_passed, same_input_forward_passed


def evidence():
    row_flags = ('input_unchanged', 'same_storage_shadow_input', 'shadow_rng_unchanged', 'candidate_rng_unchanged',
                 'shadow_outputs_detached', 'outputs_finite', 'input_version_storage_unchanged')
    rows = []
    for index, shape in enumerate(EXPECTED_SHAPES):
        row = {'capture_index': index, 'module': 'FeqDecomposeOperator', 'input_shape': list(shape),
               'input_dtype': 'torch.float32', 'bands_requested': 3, 'successful_shadow_svd_calls': 1,
               'successful_candidate_svd_calls': 1, 'svd_failures': 0, 'bands_exact': [True]*3,
               'candidate_outputs_custom_grad_fn': [True]*3,
               'candidate_output_gradient_events': [[{'finite': True, 'element_count': 4}], [], []],
               'actual_input_gradient_events': [{'finite': True, 'element_count': 4}]}
        row.update({flag: True for flag in row_flags})
        rows.append(row)
    result = {'rows': rows, 'full_forward_count': 1, 'actual_loss_finite': True, 'loss_component_count': 1,
              'backward_calls': 1, 'gradients': {name: {'finite': True, 'element_count': 1} for name in EXPECTED_PARAMETERS}}
    result.update({flag: True for flag in ('state_unchanged','parameter_identities_unchanged','preference_aliases_unchanged',
        'observer_restored','svd_identity_restored','interaction_unchanged','restored_rng_matches','forward_rng_unchanged',
        'backward_rng_unchanged','backend_settings_unchanged','protected_hashes_unchanged','old_receipts_unchanged',
        'deployment_config_unchanged','registered_parameter_coverage_complete','all_alias_gradients_finite',
        'no_reported_svd_fallback')})
    return result


class IntegrationGateTests(unittest.TestCase):
    def test_complete_evidence_passes(self): self.assertTrue(integration_passed(evidence()))
    def test_partial_inputs_reject(self):
        result=evidence(); result['rows'].pop(); self.assertFalse(integration_passed(result))
    def test_partial_bands_reject(self):
        result=evidence(); result['rows'][0]['bands_exact'].pop(); self.assertFalse(integration_passed(result))
    def test_approximate_band_cannot_replace_exact(self):
        result=evidence(); result['rows'][0]['bands_exact'][1]=False; self.assertFalse(integration_passed(result))
    def test_swapped_call_order_rejects(self):
        result=evidence(); result['rows'][0]['capture_index']=1; self.assertFalse(integration_passed(result))
    def test_gram_fallback_after_svd_exception_rejects(self):
        result=evidence(); result['rows'][0]['svd_failures']=1; self.assertFalse(integration_passed(result))
    def test_missing_svd_rejects(self):
        result=evidence(); result['rows'][0]['successful_shadow_svd_calls']=0; self.assertFalse(integration_passed(result))
    def test_changed_input_version_rejects(self):
        result=evidence(); result['rows'][0]['input_version_storage_unchanged']=False; self.assertFalse(integration_passed(result))
    def test_second_full_forward_rejects(self):
        result=evidence(); result['full_forward_count']=2; self.assertFalse(integration_passed(result))
    def test_no_backward_rejects(self):
        result=evidence(); result['backward_calls']=0; self.assertFalse(integration_passed(result))
    def test_missing_or_extra_gradient_rejects(self):
        result=evidence(); result['gradients'].pop(EXPECTED_PARAMETERS[0]); self.assertFalse(integration_passed(result))
        result=evidence(); result['gradients']['extra']={'finite':True,'element_count':1}; self.assertFalse(integration_passed(result))
    def test_nonfinite_parameter_input_or_output_gradient_rejects(self):
        for kind in ('parameter','input','output'):
            result=evidence()
            target = (result['gradients'][EXPECTED_PARAMETERS[0]] if kind=='parameter' else
                result['rows'][0]['actual_input_gradient_events'][0] if kind=='input' else
                result['rows'][0]['candidate_output_gradient_events'][0][0])
            target['finite']=False
            with self.subTest(kind=kind): self.assertFalse(integration_passed(result))
    def test_unused_entire_decomposition_call_rejects(self):
        result=evidence(); result['rows'][0]['candidate_output_gradient_events']=[[],[],[]]; self.assertFalse(integration_passed(result))
    def test_unobserved_input_gradient_rejects(self):
        result=evidence(); result['rows'][0]['actual_input_gradient_events']=[]; self.assertFalse(integration_passed(result))
    def test_detached_candidate_output_rejects(self):
        result=evidence(); result['rows'][0]['candidate_outputs_custom_grad_fn'][0]=False; self.assertFalse(integration_passed(result))
    def test_every_lifecycle_guard_is_required(self):
        original=evidence()
        guards=[name for name,value in original.items() if value is True and name != 'actual_loss_finite']
        for flag in guards:
            result=copy.deepcopy(original); result.pop(flag)
            with self.subTest(flag=flag): self.assertFalse(integration_passed(result))
    def test_nonfinite_loss_prevents_backward_gate(self):
        result=evidence(); result['actual_loss_finite']=False; self.assertFalse(same_input_forward_passed(result))


if __name__ == '__main__': unittest.main()
