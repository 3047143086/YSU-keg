import unittest

from reporting import decide


class ScientificDecisionTests(unittest.TestCase):
    def metrics(self):
        return {'baseline': {'recall@20': .1, 'ndcg@20': .05, 'recall@10': .07, 'ndcg@10': .04},
                'history': {'recall@20': .1002, 'ndcg@20': .0501, 'recall@10': .07, 'ndcg@10': .04},
                'edge': {'recall@20': .101, 'ndcg@20': .051, 'recall@10': .07, 'ndcg@10': .04}}

    def test_requires_every_primary_shortlist_and_matched_control_gate(self):
        self.assertEqual(decide(self.metrics())['status'], 'keep')
        for key, value in (('recall@20', .1004), ('ndcg@20', .05),
                           ('recall@10', .0697), ('ndcg@10', .0397)):
            metrics = self.metrics()
            metrics['edge'][key] = value
            self.assertEqual(decide(metrics)['status'], 'discard')
        metrics = self.metrics()
        metrics['history']['recall@20'] = .101
        self.assertEqual(decide(metrics)['status'], 'discard')

    def test_missing_control_is_pending_and_invalid_metrics_fail(self):
        metrics = self.metrics()
        del metrics['history']
        self.assertEqual(decide(metrics)['status'], 'pending')
        for value in (float('nan'), float('inf'), -1., 1.1):
            metrics = self.metrics()
            metrics['baseline']['ndcg@20'] = value
            with self.assertRaises(ValueError):
                decide(metrics)
        metrics = self.metrics()
        metrics['baseline']['recall@20'] = 0
        with self.assertRaises(ValueError):
            decide(metrics)


if __name__ == '__main__':
    unittest.main()
