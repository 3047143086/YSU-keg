import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from mapping import (compute_holdout_pairs, fingerprint, load_support_reviews,
                     match_support_identities, read_train_events)


class MappingTests(unittest.TestCase):
    def make_inputs(self, directory):
        root = Path(directory)
        inter = root / 'baby.inter'
        with inter.open('w', newline='', encoding='utf8') as stream:
            writer = csv.writer(stream, delimiter='\t')
            writer.writerow(['userID', 'itemID', 'rating', 'timestamp', 'x_label'])
            rows = [
                [0, 0, 5, 100, 0], [0, 1, 4, 200, 0], [0, 2, 5, 300, 0], [0, 3, 5, 400, 0],
                [1, 0, 5, 110, 0], [1, 1, 4, 210, 0], [1, 2, 5, 310, 0], [1, 3, 4, 410, 0],
                [2, 4, 5, 500, 0], [2, 5, 5, 600, 0], [2, 6, 5, 700, 0], [2, 7, 5, 800, 0],
                [0, 4, 5, 900, 1], [1, 5, 4, 901, 2],
            ]
            writer.writerows(rows)
        mapping = root / 'item_id_to_asin.json'
        mapping.write_text(json.dumps({str(i): 'A' + str(i) for i in range(8)}), encoding='utf8')
        reviews = root / 'reviews.json.gz'
        rows = []
        for user, item, rating, timestamp in [
                (10, 0, 5, 100), (10, 1, 4, 200), (10, 2, 5, 300), (10, 3, 5, 400),
                (20, 0, 5, 110), (20, 1, 4, 210), (20, 2, 5, 310), (20, 3, 4, 410),
                (30, 4, 5, 500), (30, 5, 5, 600), (30, 6, 5, 700), (30, 7, 5, 800),
        ]:
            rows.append({'reviewerID': 'U' + str(user), 'asin': 'A' + str(item),
                         'overall': rating, 'unixReviewTime': timestamp,
                         'reviewText': 'review %s %s' % (user, item), 'summary': 'summary'})
        with gzip.open(reviews, 'wt', encoding='utf8') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')
        return inter, mapping, reviews

    def test_frozen_targets_and_support_exclude_all_target_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            inter, mapping, reviews = self.make_inputs(directory)
            events = read_train_events(inter, mapping)
            targets = compute_holdout_pairs(events)
            self.assertEqual(len(targets), 3)
            records, audit = load_support_reviews(inter, mapping, reviews)
            self.assertEqual(audit['original_train_edges'], 12)
            self.assertEqual(audit['globally_removed_target_edges'], 3)
            self.assertEqual(audit['expected_support_edges'], 9)
            self.assertEqual(len(records), 9)
            self.assertTrue(audit['all_targets_absent_from_export'])
            self.assertEqual(audit['official_validation_reviews_used'], False)
            self.assertTrue(all('reviewerID' not in record for record in records))
            self.assertTrue(all('rating' not in record for record in records))

    def test_validation_and_test_rows_never_create_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            inter, mapping, reviews = self.make_inputs(directory)
            with gzip.open(reviews, 'at', encoding='utf8') as stream:
                stream.write(json.dumps({'reviewerID': 'U0', 'asin': 'A4', 'overall': 5,
                                         'unixReviewTime': 900, 'reviewText': 'heldout',
                                         'summary': 'heldout'}) + '\n')
            records, audit = load_support_reviews(inter, mapping, reviews)
            self.assertEqual(len(records), 9)
            self.assertFalse(any(row['itemID'] == 4 and row['userID'] == 0 for row in records))
            self.assertFalse(audit['official_validation_reviews_used'])
            self.assertFalse(audit['official_test_reviews_used'])

    def test_strict_mode_rejects_missing_support_review(self):
        with tempfile.TemporaryDirectory() as directory:
            inter, mapping, reviews = self.make_inputs(directory)
            source = Path(reviews)
            rows = []
            with gzip.open(source, 'rt', encoding='utf8') as stream:
                rows = [json.loads(line) for line in stream]
            targets = compute_holdout_pairs(read_train_events(inter, mapping))
            remove_index = next(index for index, row in enumerate(rows)
                                if ({'U10': 0, 'U20': 1, 'U30': 2}[row['reviewerID']],
                                    int(row['asin'][1:])) not in targets)
            with gzip.open(source, 'wt', encoding='utf8') as stream:
                for index, row in enumerate(rows):
                    if index == remove_index:
                        continue
                    stream.write(json.dumps(row) + '\n')
            with self.assertRaises(ValueError):
                load_support_reviews(inter, mapping, reviews, strict=True)

    def test_ambiguous_owner_is_preserved_without_guessing(self):
        identity = fingerprint('A0', 100, 5)
        event = {'userID': 0, 'itemID': 0, 'timestamp': 100, 'identity': identity}
        rows = [{'reviewerID': owner, 'asin': 'A0', 'unixReviewTime': 100, 'overall': 5}
                for owner in ('U0', 'U1')]
        aligned, audit = match_support_identities([event], rows)
        self.assertFalse(aligned)
        self.assertEqual(audit['ambiguous_support_users'], 1)
        self.assertEqual(audit['ambiguous_support_fingerprints'], 1)

    def test_duplicate_raw_owner_event_is_not_arbitrarily_selected(self):
        identity = fingerprint('A0', 100, 5)
        event = {'userID': 0, 'itemID': 0, 'timestamp': 100, 'identity': identity}
        raw = {'reviewerID': 'U0', 'asin': 'A0', 'unixReviewTime': 100, 'overall': 5}
        aligned, audit = match_support_identities([event], [raw, dict(raw)])
        self.assertFalse(aligned)
        self.assertEqual(audit['duplicate_raw_event_users'], 1)
        self.assertEqual(audit['raw_duplicate_owner_fingerprint_records'], 1)

    def test_two_encoded_users_cannot_map_to_one_raw_reviewer(self):
        events = [
            {'userID': 0, 'itemID': 0, 'timestamp': 100, 'identity': fingerprint('A0', 100, 5)},
            {'userID': 1, 'itemID': 1, 'timestamp': 200, 'identity': fingerprint('A1', 200, 5)},
        ]
        rows = [{'reviewerID': 'U0', 'asin': asin, 'unixReviewTime': time, 'overall': 5}
                for asin, time in (('A0', 100), ('A1', 200))]
        aligned, audit = match_support_identities(events, rows)
        self.assertFalse(aligned)
        self.assertEqual(audit['owner_collision_support_users'], 2)


if __name__ == '__main__':
    unittest.main()
