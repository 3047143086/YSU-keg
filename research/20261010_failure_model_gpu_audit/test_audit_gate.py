import copy
import unittest
from audit_gate import archived_candidate_passed, model_candidate_passed


class AuditGateTests(unittest.TestCase):
    def test_finite_generic_requires_comparison_pass(self):
        row = dict(exact_forward_bands=[True]*3, candidate_gradient_finite=True,
                   generic_gradient_finite=True, generic_comparison={'passed': False})
        self.assertFalse(archived_candidate_passed(row))
        row['generic_comparison']['passed'] = True
        self.assertTrue(archived_candidate_passed(row))

    def test_nonfinite_generic_is_unavailable_not_equivalence(self):
        row = dict(exact_forward_bands=[True]*3, candidate_gradient_finite=True,
                   generic_gradient_finite=False, generic_comparison={'passed': False})
        self.assertTrue(archived_candidate_passed(row))
        row['candidate_gradient_finite'] = False
        self.assertFalse(archived_candidate_passed(row))

    def test_missing_band_never_passes(self):
        self.assertFalse(archived_candidate_passed(dict(exact_forward_bands=[], candidate_gradient_finite=True)))

    def passing_model(self):
        return dict(state_unchanged=True, loss_matches=True, candidate_all_gradients_finite=True,
                    participating_parameters_match=True, participating_parameter_count=3,
                    original_all_gradients_finite=False, generic_gradient_equivalence_passed=False,
                    rows=[dict(candidate_vs_original_input={'passed': True},
                               candidate_vs_original_bands=[{'passed': True} for _ in range(3)])
                          for _ in range(12)])

    def test_full_model_rejects_empty_or_different_parameters(self):
        for key, bad in [('participating_parameter_count', 0), ('participating_parameters_match', False)]:
            result = self.passing_model()
            result[key] = bad
            self.assertFalse(model_candidate_passed(result))

    def test_full_model_rejects_any_missing_capture_or_mismatch(self):
        result = self.passing_model()
        self.assertTrue(model_candidate_passed(result))
        for bad in ('capture', 'band', 'input', 'state', 'loss'):
            probe = copy.deepcopy(result)
            if bad == 'capture': probe['rows'].pop()
            elif bad == 'band': probe['rows'][7]['candidate_vs_original_bands'][2]['passed'] = False
            elif bad == 'input': probe['rows'][5]['candidate_vs_original_input']['passed'] = False
            elif bad == 'state': probe['state_unchanged'] = False
            else: probe['loss_matches'] = False
            self.assertFalse(model_candidate_passed(probe))

    def test_finite_full_gradient_requires_equivalence(self):
        result = self.passing_model()
        result['original_all_gradients_finite'] = True
        self.assertFalse(model_candidate_passed(result))
        result['generic_gradient_equivalence_passed'] = True
        self.assertTrue(model_candidate_passed(result))


if __name__ == '__main__': unittest.main()
