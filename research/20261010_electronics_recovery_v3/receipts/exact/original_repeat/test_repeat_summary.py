"""Prevent incomplete or nonfinite evidence from becoming a completed diagnosis."""
import copy
import unittest

from repeat_summary import summarize


def evidence():
    comparison = {'exact': True, 'passed': True, 'relative_frobenius_error': 0.0, 'maximum_absolute_error': 0.0}
    rows = [{'capture_index': index, 'first_module': 'frequency', 'second_module': 'frequency',
             'input': copy.deepcopy(comparison), 'bands': [copy.deepcopy(comparison) for _ in range(3)]}
            for index in range(12)]
    losses = [{'exact': True, 'within_inherited_bounds': True, 'absolute_error': 0.0}]
    return rows, losses


class RepeatSummaryTests(unittest.TestCase):
    def test_exact_is_diagnosis_without_integration_approval(self):
        result = summarize(*evidence())
        self.assertEqual(result['status'], 'completed_exact')
        self.assertEqual(result['inputs']['exact_count'], 12)
        self.assertEqual(result['bands']['exact_count'], 36)
        for flag in ('candidate_accepted', 'training_authorized', 'exact_historical_crash_replayed',
                     'projector_equivalence_established'):
            self.assertFalse(result[flag])

    def test_nonexact_below_threshold_still_different(self):
        rows, losses = evidence()
        rows[3]['bands'][1].update(exact=False, relative_frobenius_error=1e-8, maximum_absolute_error=1e-7)
        result = summarize(rows, losses)
        self.assertEqual(result['status'], 'completed_different')
        self.assertEqual(result['bands']['nonexact_count'], 1)
        self.assertEqual(result['bands']['exceeds_inherited_bounds_count'], 0)

    def test_inherited_failure_is_retained(self):
        rows, losses = evidence()
        rows[0]['input'].update(exact=False, passed=False, relative_frobenius_error=6e-5, maximum_absolute_error=0.001)
        result = summarize(rows, losses)
        self.assertEqual(result['inputs']['exceeds_inherited_bounds_count'], 1)
        self.assertEqual(result['inputs']['relative_frobenius_error']['maximum'], 6e-5)

    def test_loss_only_difference_is_visible(self):
        rows, losses = evidence()
        losses[0].update(exact=False, absolute_error=1e-9)
        self.assertEqual(summarize(rows, losses)['status'], 'completed_different')

    def test_missing_capture_rejected(self):
        rows, losses = evidence()
        with self.assertRaises(ValueError): summarize(rows[:-1], losses)

    def test_duplicate_capture_rejected(self):
        rows, losses = evidence()
        rows[5]['capture_index'] = 4
        with self.assertRaises(ValueError): summarize(rows, losses)

    def test_missing_band_rejected(self):
        rows, losses = evidence()
        rows[11]['bands'].pop()
        with self.assertRaises(ValueError): summarize(rows, losses)

    def test_module_order_mismatch_rejected(self):
        rows, losses = evidence()
        rows[0]['second_module'] = 'different_module'
        with self.assertRaises(ValueError): summarize(rows, losses)

    def test_undefined_relative_error_cannot_pass(self):
        rows, losses = evidence()
        rows[0]['input']['relative_frobenius_error'] = None
        with self.assertRaises(ValueError): summarize(rows, losses)

    def test_zero_reference_nonzero_error_is_reported(self):
        rows, losses = evidence()
        rows[0]['input'].update(exact=False, passed=False, relative_frobenius_error=None, maximum_absolute_error=1e-7)
        result = summarize(rows, losses)
        self.assertEqual(result['status'], 'completed_different')
        self.assertEqual(result['inputs']['relative_frobenius_error']['undefined_count'], 1)
        self.assertEqual(result['inputs']['exceeds_inherited_bounds_count'], 1)

    def test_nonfinite_loss_rejected(self):
        rows, losses = evidence()
        losses[0]['absolute_error'] = float('nan')
        with self.assertRaises(ValueError): summarize(rows, losses)

    def test_missing_exactness_rejected(self):
        rows, losses = evidence()
        del rows[0]['bands'][0]['exact']
        with self.assertRaises(ValueError): summarize(rows, losses)


if __name__ == '__main__':
    unittest.main()
