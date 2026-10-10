"""CPU-only protocol tests; the remote base smoke checks BRIDGE integration."""

import unittest

import numpy as np

from train_base import ranking_metrics, validate_and_profile


class BaseProtocolTests(unittest.TestCase):
    def setUp(self):
        self.features = np.zeros((205, 2, 512), dtype=np.float32)
        self.features[1, 0, 0] = 1.0
        self.features[2, 1, 0] = 1.0
        self.features[3, 0, 1] = 1.0
        self.roles = np.array([0, 1, 2, -1], dtype=np.int8)
        self.support_users = np.array([0, 0, 1, 2])
        self.support_items = np.array([1, 2, 2, 3])
        self.target_users = np.array([0, 1, 2])
        self.target_items = np.array([3, 4, 5])

    def profiles(self, **overrides):
        fields = dict(support_users=self.support_users, support_items=self.support_items,
                      target_users=self.target_users, target_items=self.target_items,
                      roles=self.roles, features=self.features, n_users=4, n_items=205)
        fields.update(overrides)
        return validate_and_profile(**fields)

    def test_support_only_normalized_profile(self):
        mask, targets, history = self.profiles()
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(targets.tolist(), [3, 4, 5, -1])
        self.assertFalse(mask[self.target_users, self.target_items].any())
        self.assertEqual(history.shape, (4, 1024))
        self.assertEqual(history[0, 1], 0.0)
        np.testing.assert_allclose(history[0, [0, 512]], [2 ** -0.5] * 2, rtol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(history[:3], axis=1), 1.0, rtol=1e-6)
        self.assertTrue((history[3] == 0).all())

    def test_leaked_target_rejected(self):
        with self.assertRaisesRegex(ValueError, 'leaked'):
            self.profiles(target_items=np.array([1, 4, 5]))

    def test_duplicate_targets_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            self.profiles(target_users=np.array([0, 0, 2]))

    def test_role_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'disagree'):
            self.profiles(roles=np.array([0, 1, 2, 0]))

    def test_duplicate_support_pairs_do_not_reweight_history(self):
        _, _, expected = self.profiles()
        _, _, repeated = self.profiles(support_users=np.array([0, 0, 0, 1, 2]),
                                        support_items=np.array([1, 1, 2, 2, 3]))
        np.testing.assert_array_equal(expected, repeated)

    def test_single_target_ranking_metrics(self):
        ranking = np.tile(np.arange(20), (3, 1))
        metrics, values = ranking_metrics(ranking, np.array([0, 19, 25]))
        self.assertAlmostEqual(metrics['recall@10'], 1 / 3)
        self.assertAlmostEqual(metrics['recall@20'], 2 / 3)
        self.assertAlmostEqual(metrics['ndcg@20'], (1 + 1 / np.log2(21)) / 3)
        self.assertEqual(values.shape, (3, 4))


if __name__ == '__main__':
    unittest.main()
