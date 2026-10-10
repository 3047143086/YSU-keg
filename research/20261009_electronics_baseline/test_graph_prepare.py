import unittest

try:
    import torch
except ImportError:
    torch = None

from graph_prepare import source_graph


@unittest.skipIf(torch is None, "MMRec environment supplies torch")
class GraphTests(unittest.TestCase):
    def setUp(self):
        from models.dual_frequency_encoder import DualFrequencyEncoder
        self.encoder = DualFrequencyEncoder

    def test_chunked_indices_equal_dense_for_ties(self):
        values = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.], [1., 0.]])
        dense, dense_graph = source_graph(self.encoder, values, 99, 2)
        chunked, chunked_graph = source_graph(self.encoder, values, 2, 2)
        self.assertTrue(torch.equal(dense, chunked))
        self.assertTrue(torch.equal(dense_graph.to_dense(), chunked_graph.to_dense()))

    def test_chunked_and_dense_random_graphs_equal(self):
        values = torch.randn(19, 11, generator=torch.Generator().manual_seed(2026))
        dense, dense_graph = source_graph(self.encoder, values, 99, 4)
        chunked, chunked_graph = source_graph(self.encoder, values, 3, 4)
        self.assertTrue(torch.equal(dense, chunked))
        self.assertTrue(torch.equal(dense_graph.to_dense(), chunked_graph.to_dense()))

    def test_zero_vectors_are_finite_and_shape_preserved(self):
        values = torch.zeros((4, 3), dtype=torch.float32)
        values[0, 0] = 1
        indices, graph = source_graph(self.encoder, values, 2, 2)
        self.assertEqual(tuple(graph.shape), (4, 4))
        self.assertTrue(torch.isfinite(graph._values()).all())


if __name__ == "__main__":
    unittest.main()
