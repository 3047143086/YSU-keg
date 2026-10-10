import unittest
import warnings
from warning_observer import observed_warnings, reported_svd_fallback


class WarningObserverTests(unittest.TestCase):
    def test_reported_pytorch_nonconvergence_fallback_rejected(self):
        rows=[{'message':'During the SVD execution, batches 0 failed to converge. A more accurate method will be used to calculate the SVD as a fallback.'}]
        self.assertTrue(reported_svd_fallback(rows))

    def test_ordinary_warning_is_not_solver_failure(self):
        self.assertFalse(reported_svd_fallback([{'message':'Storage API is deprecated'}]))

    def test_reporting_and_identity_preserved_on_exception(self):
        rows=[];phase={'name':'candidate'}
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('always')
            original=warnings.showwarning
            with self.assertRaises(RuntimeError):
                with observed_warnings(rows,phase):
                    warnings.warn('ordinary warning',UserWarning)
                    raise RuntimeError('probe failed')
            self.assertIs(warnings.showwarning,original)
        self.assertEqual(len(emitted),1); self.assertEqual(len(rows),1); self.assertEqual(rows[0]['phase'],'candidate')


if __name__ == '__main__': unittest.main()
