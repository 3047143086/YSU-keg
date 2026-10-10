import unittest

import numpy as np

from screen_core import normalize, profile, review_text, same_item_donors, score_rows


class ScreenCoreTests(unittest.TestCase):
    def test_cleaning_is_shared_for_donor_length_and_encoding(self):
        record = {'summary': '<b>Good</b>', 'reviewText': ' for   travel &amp; sleep'}
        self.assertEqual(review_text(record), 'Good for travel & sleep')

    def test_donor_preserves_item_excludes_owner_and_prefers_word_length(self):
        records = [
            {'userID': 0, 'itemID': 3, 'reviewText': 'one two'},
            {'userID': 1, 'itemID': 3, 'reviewText': 'one two three four'},
            {'userID': 2, 'itemID': 3, 'reviewText': 'one two three'},
            {'userID': 3, 'itemID': 8, 'reviewText': 'one two'},
            {'userID': 4, 'itemID': 3, 'reviewText': ''},
        ]
        got = same_item_donors(records)
        self.assertEqual(got[0], 2)
        self.assertEqual(got[3], -1)
        self.assertEqual(got[4], -1)
        for own, donor in enumerate(got):
            if donor >= 0:
                self.assertEqual(records[own]['itemID'], records[donor]['itemID'])
                self.assertNotEqual(records[own]['userID'], records[donor]['userID'])
        np.testing.assert_array_equal(got, same_item_donors(records))

    def test_zero_profiles_remain_zero_and_row_scores_match_literal(self):
        vectors = np.asarray([[1., 0.], [0., 1.], [1., 0.]], dtype=np.float32)
        pooled = profile(vectors, np.asarray([0, 0, 1]), 3)
        self.assertTrue(np.all(pooled[2] == 0))
        items = np.asarray([[1, 2], [2, 0]])
        users = np.asarray([0, 2])
        got = score_rows(pooled, vectors, users, items)
        expected = np.asarray([[pooled[u] @ vectors[i] for i in row]
                               for u, row in zip(users, items)])
        np.testing.assert_allclose(got, expected)
        with self.assertRaises(ValueError):
            normalize(np.asarray([[np.nan, 1.]]))


if __name__ == '__main__':
    unittest.main()
