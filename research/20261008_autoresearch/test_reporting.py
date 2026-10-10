"""A primary metric gain alone must not turn a failed guard into acceptance."""
import unittest
from karpathy_results import decide


class ReportingTests(unittest.TestCase):
    def rows(self):
        return {'runs/baby_{}_s2026'.format(arm): {'metrics': metrics} for arm, metrics in {
            'baseline': {'recall@20': .1, 'ndcg@20': .05, 'recall@10': .07, 'ndcg@10': .04},
            'history': {'recall@20': .1001, 'ndcg@20': .0501},
            'edge': {'recall@20': .102, 'ndcg@20': .051, 'recall@10': .07, 'ndcg@10': .04}
        }.items()}

    def test_missing_control_stays_pending(self):
        rows = self.rows()
        del rows['runs/baby_history_s2026']
        self.assertEqual(decide(rows, 'baby')['status'], 'pending')

    def test_every_guard_is_required_for_keep(self):
        self.assertEqual(decide(self.rows(), 'baby')['status'], 'keep')
        for key, value in [('recall@10', .069), ('ndcg@20', .05), ('recall@20', .1001)]:
            rows = self.rows()
            rows['runs/baby_edge_s2026']['metrics'][key] = value
            self.assertEqual(decide(rows, 'baby')['status'], 'discard')


if __name__ == '__main__':
    unittest.main()
