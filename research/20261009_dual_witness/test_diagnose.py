import unittest

import numpy as np
from scipy import sparse

from diagnose import auc, comparisons, decide, internal_split, item_scores


class DiagnosticTests(unittest.TestCase):
    def test_item_scores_use_original_orientation_and_row_pairs(self):
        support = sparse.csr_matrix([[1., 0., 1.], [0., 1., 0.]])
        weights = sparse.csr_matrix([[0., .2, .3], [.4, 0., .5], [.6, .7, 0.]])
        users = np.asarray([1, 0, 1])
        items = np.asarray([[0, 2], [2, 0], [1, 0]])
        expected = (support @ weights.T).toarray()[users[:, None], items]
        np.testing.assert_array_equal(item_scores(weights, support, users, items, 1), expected)

    def test_targets_removed_globally_and_all_train_positives_excluded(self):
        train = sparse.csr_matrix(np.asarray([[1, 1, 1, 1, 0, 0],
                                               [0, 1, 1, 1, 1, 0],
                                               [0, 0, 0, 0, 0, 1]], dtype=float))
        support, users, targets = internal_split(train)
        self.assertEqual(users.tolist(), [0, 1])
        self.assertEqual(support.nnz, train.nnz - 2)
        self.assertTrue(np.all(support[users, targets].A1 == 0))
        selected = comparisons(train, users, targets)
        self.assertEqual(selected.shape, (2, 17))
        self.assertTrue(np.array_equal(selected[:, 0], targets))
        for row, user in enumerate(users):
            self.assertTrue(np.all(train[user, selected[row, 1:]].toarray() == 0))
        np.testing.assert_array_equal(selected, comparisons(train, users, targets))

    def test_auc_ties_are_half_wins_and_nonfinite_scores_rejected(self):
        scores = np.zeros((3, 17))
        scores[0, 0] = 1
        scores[2, 0] = -1
        np.testing.assert_array_equal(auc(scores), [1, .5, 0])
        scores[1, 2] = np.nan
        with self.assertRaises(ValueError):
            auc(scores)

    def test_decision_gate_rejects_low_coverage_without_adjusting_thresholds(self):
        base = np.linspace(0.45, 0.55, 32)
        per_user = {name: base.copy() for name in ('dual', 'itemknn', 'userknn', 'min2', 'shuffled')}
        per_user['dual'] = base + 0.02
        result = decide(per_user, 0.249, np.arange(base.size))
        self.assertEqual(result['status'], 'discard')
        self.assertFalse(result['passed'])
        self.assertFalse(result['formal_training_authorized_by_this_screen'])
        self.assertFalse(result['publication_evidence'])


if __name__ == '__main__':
    unittest.main()
