"""Small fixtures for relation boundaries, strict TRAIN selection and endpoints."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from audit_catalog import REMOTE

NS = {}
exec(compile(REMOTE.split('\nrecord =')[0], 'catalog_remote_fixture', 'exec'), NS)


class CatalogFixtures(unittest.TestCase):
    def test_full_signature_equality_partial_overlap_is_different(self):
        relation = NS['relation']
        a = {'brand': 'a', 'categories': (('root', 'x'),)}
        b = {'brand': 'b', 'categories': (('root', 'x'), ('root', 'y'))}
        self.assertEqual(relation(a, b), 'neither')
        b['brand'] = 'a'
        self.assertEqual(relation(a, b), 'same-brand_diff-category')
        b['categories'] = a['categories']
        self.assertIsNone(relation(a, b))
        b['brand'] = 'b'
        self.assertEqual(relation(a, b), 'same-category_diff-brand')

    def test_normalization_preserves_full_paths(self):
        self.assertEqual(NS['categories']([['  ＲＯＯＴ ', 'X  Y'], ['root', 'x y']]),
                         (('root', 'x y'),))
        self.assertEqual(NS['categories']([['root'], ['root', None]]), (('root',),))
        self.assertEqual(NS['norm'](' Ａ  B '), 'a b')

    def test_unique_one_partner_and_item_disjoint_partitions(self):
        items = set(range(100))
        metadata = {item: {'brand': str(item), 'categories': ((str(item),),)} for item in items}
        summary = NS['pair_summary'](items, metadata, 'baby')
        for name, tag in (('fitting', 'fit'), ('heldout', 'holdout')):
            selected = sorted(item for item in items if NS['partition'](item, 'baby') == tag)
            expected = []
            for left, right in zip(selected[::2], selected[1::2]):
                expected.append([hashlib.sha256(('baby:' + str(left)).encode()).hexdigest(),
                                 hashlib.sha256(('baby:' + str(right)).encode()).hexdigest()])
            row = summary['relations']['neither'][name]
            self.assertEqual(row['unique_matching_pair_count'], len(selected) // 2)
            self.assertEqual(row['endpoint_sha256_digest'], hashlib.sha256(
                json.dumps(expected, separators=(',', ':')).encode()).hexdigest())
        fit = {item for item in items if NS['partition'](item, 'baby') == 'fit'}
        held = items - fit
        self.assertFalse(fit & held)
        self.assertEqual(fit | held, items)

    def test_only_split_zero_items_and_degree_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.inter'
            path.write_text('itemID\tx_label\n1\t0\n1\t0\n2\t0\ninvalid-target\t1\nother-target\t2\n')
            row = NS['read_train'](path)
            self.assertEqual(row['train_items'], {1, 2})
            self.assertEqual(dict(row['train_item_degrees']), {1: 2, 2: 1})
            path.write_text('itemID\n1\n')
            with self.assertRaises(ValueError):
                NS['read_train'](path)

    def test_null_permutation_is_deterministic_with_degree_bins(self):
        metadata = {item: {'brand': str(item), 'categories': ((str(item),),)} for item in range(100)}
        degree = {item: 1 if item % 2 == 0 else 8 for item in range(100)}
        first = NS['null_label_summary'](set(metadata), metadata, 'baby', degree)
        second = NS['null_label_summary'](set(metadata), metadata, 'baby', degree)
        self.assertEqual(first, second)
        for row in first['partitions'].values():
            self.assertEqual(set(row['degree_bin_counts']), {'0', '3'})
            self.assertGreaterEqual(row['joint_label_changed_rate'], .5)


if __name__ == '__main__':
    unittest.main()
