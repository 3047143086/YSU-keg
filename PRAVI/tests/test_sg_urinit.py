from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.msca_sg import MSCA_SG
from utils.sg_urinit import (
    build_sg_user_profile,
    inverse_item_degree_operator,
    profile_cache_signature,
)


class SGURInitTest(unittest.TestCase):
    def setUp(self):
        self.interactions = sp.coo_matrix(
            ([1, 1, 1], ([0, 0, 1], [0, 1, 0])), shape=(2, 3)
        )

    def test_inverse_item_degree_matches_official_formula(self):
        operator = inverse_item_degree_operator(self.interactions)
        np.testing.assert_allclose(
            operator.toarray(),
            np.asarray([[0.5, 1.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32),
        )

    def test_zero_cluster_weight_is_exact_local_aggregation(self):
        features = np.asarray(
            [[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]], dtype=np.float32
        )
        profile = build_sg_user_profile(
            self.interactions, features, num_clusters=2, cluster_weight=0.0
        )
        expected = inverse_item_degree_operator(self.interactions) @ features
        np.testing.assert_allclose(profile, expected)

    def test_local_global_profile_is_deterministic(self):
        features = np.asarray(
            [[0.0, 0.0], [0.1, 0.0], [4.0, 4.0]], dtype=np.float32
        )
        first = build_sg_user_profile(
            self.interactions, features, num_clusters=2, cluster_weight=0.1
        )
        second = build_sg_user_profile(
            self.interactions, features, num_clusters=2, cluster_weight=0.1
        )
        np.testing.assert_allclose(first, second)
        self.assertEqual(first.shape, (2, 2))

    def test_cache_signature_tracks_graph_and_features(self):
        features = np.arange(12, dtype=np.float32).reshape(3, 4)
        signature = profile_cache_signature(self.interactions, features)
        self.assertEqual(signature, profile_cache_signature(self.interactions, features))
        changed = features.copy()
        changed[0, 0] += 1.0
        self.assertNotEqual(
            signature, profile_cache_signature(self.interactions, changed)
        )

    def test_semantic_bpr_prefers_larger_positive_margin(self):
        users = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        positive = users.clone()
        negative = -users
        good = MSCA_SG.semantic_bpr_loss(users, positive, negative)
        bad = MSCA_SG.semantic_bpr_loss(users, negative, positive)
        self.assertLess(float(good), float(bad))


if __name__ == "__main__":
    unittest.main()
