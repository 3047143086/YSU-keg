import unittest

import numpy as np
import torch

from stage1_core import EvidenceMatcher, internal_holdout, leave_one_out_profile, matched_permutation
from extract_features import clean_text


class StageOneTests(unittest.TestCase):
    def test_internal_split_preserves_history_and_is_reproducible(self):
        users = np.array([3, 1, 3, 2, 1, 3, 9])
        selected = internal_holdout(users, 51)
        np.testing.assert_array_equal(selected, internal_holdout(users, 51))
        self.assertEqual(set(users[selected]), {1, 3})
        support = np.delete(users, selected)
        self.assertTrue(set(users[selected]) <= set(support))

    def test_shuffle_preserves_missingness(self):
        mask = np.array([[1, 1], [1, 0], [1, 1], [1, 0], [0, 0]], dtype=bool)
        permutation = matched_permutation(mask, 11)
        np.testing.assert_array_equal(mask[permutation], mask)
        np.testing.assert_array_equal(np.sort(permutation), np.arange(len(mask)))

    def test_profile_excludes_positive_for_both_alternatives(self):
        features = torch.eye(3)
        sums = torch.tensor([[1., 1., 0.], [0., 0., 1.]])
        counts = torch.tensor([2., 1.])
        profile = leave_one_out_profile(sums, counts, features, torch.tensor([0, 1]), torch.tensor([0, 2]))
        torch.testing.assert_close(profile, torch.tensor([[0., 1., 0.], [0., 0., 0.]]))

    def test_matching_shapes_gradients_and_equal_parameters(self):
        rng = np.random.RandomState(4)
        global_bank = rng.randn(6, 2, 8).astype('float32')
        local_bank = rng.randn(6, 5, 8).astype('float32')
        global_mask = np.ones((6, 2), dtype=bool)
        local_mask = np.ones((6, 5), dtype=bool)
        global_bank[0] = 0
        local_bank[0] = 0
        global_mask[0] = False
        local_mask[0] = False
        counts = []
        for variant in ('global', 'local', 'shuffled'):
            torch.manual_seed(123)
            matcher = EvidenceMatcher(global_bank, global_mask, local_bank, local_mask, variant)
            profile = torch.randn(2, 16)
            scores = matcher(profile, torch.tensor([[0, 1, 2], [3, 4, 5]]))
            self.assertEqual(tuple(scores.shape), (2, 3))
            self.assertTrue(torch.isfinite(scores).all())
            pair = matcher(profile, torch.tensor([0, 4]))
            pair.sum().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in matcher.parameters()))
            self.assertGreater(float(matcher.scorer[-1].weight.grad.abs().sum()), 0)
            counts.append(sum(p.numel() for p in matcher.parameters()))
        self.assertEqual(len(set(counts)), 1)

    def test_metadata_cleaning(self):
        self.assertEqual(clean_text('A &amp; B <b>red</b>'), 'A & B red')
        self.assertEqual(clean_text(None), '')
        self.assertEqual(clean_text(['small', 'cotton']), 'small cotton')


if __name__ == '__main__':
    unittest.main()
