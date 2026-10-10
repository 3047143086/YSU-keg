"""Scientific checks of the formula, including the original failure geometry."""
import unittest
import torch
from projector import original_svd_bands, projector_bands, split_sizes


def diagonal_tall(values, extra_rows=2):
    matrix = torch.zeros((len(values) + extra_rows, len(values)), dtype=torch.float64)
    matrix[:len(values)] = torch.diag(torch.tensor(values, dtype=torch.float64))
    return matrix


def weighted_loss(outputs, generator):
    weights = [torch.randn(output.shape, dtype=output.dtype, generator=generator) for output in outputs]
    return sum((weight * output).sum() for weight, output in zip(weights, outputs))


class ProjectorTests(unittest.TestCase):
    def test_original_band_split_is_preserved(self):
        self.assertEqual(split_sizes(64, 3), (22, 21, 21))

    def test_double_gradcheck_distinct_singular_values(self):
        matrix = diagonal_tall([7., 6., 4., 3., 2., 1.]).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda x: projector_bands(x, 3), (matrix,),
                                               eps=1e-6, atol=1e-7, rtol=1e-5))

    def test_double_gradcheck_repeated_within_each_band(self):
        matrix = diagonal_tall([7., 7., 4., 4., 1., 1.]).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda x: projector_bands(x, 3), (matrix,),
                                               eps=1e-6, atol=1e-7, rtol=1e-5))

    def test_double_gradcheck_rank_deficiency_inside_last_band(self):
        matrix = diagonal_tall([7., 6., 4., 3., 0., 0.]).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda x: projector_bands(x, 3), (matrix,),
                                               eps=1e-6, atol=1e-7, rtol=1e-5))

    def test_double_gradcheck_rotated_internal_degeneracy(self):
        generator = torch.Generator().manual_seed(2026)
        left, _ = torch.linalg.qr(torch.randn((9, 6), dtype=torch.float64, generator=generator))
        right, _ = torch.linalg.qr(torch.randn((6, 6), dtype=torch.float64, generator=generator))
        singular = torch.diag(torch.tensor([7., 7., 4., 4., 1., 1.], dtype=torch.float64))
        matrix = (left @ singular @ right.T).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda x: projector_bands(x, 3), (matrix,),
                                               eps=1e-6, atol=1e-7, rtol=1e-5))

    def test_nondegenerate_forward_and_gradient_match_original(self):
        generator = torch.Generator().manual_seed(2026)
        left, _ = torch.linalg.qr(torch.randn((12, 6), dtype=torch.float64, generator=generator))
        right, _ = torch.linalg.qr(torch.randn((6, 6), dtype=torch.float64, generator=generator))
        source = left @ torch.diag(torch.tensor([9., 7., 5., 3., 2., .5], dtype=torch.float64)) @ right.T
        original = source.clone().requires_grad_()
        candidate = source.clone().requires_grad_()
        old_outputs, new_outputs = original_svd_bands(original, 3), projector_bands(candidate, 3)
        for old, new in zip(old_outputs, new_outputs):
            self.assertTrue(torch.equal(new, old))
            torch.testing.assert_close(new, old, atol=1e-10, rtol=1e-10)
        generator.manual_seed(17)
        weighted_loss(old_outputs, generator).backward()
        generator.manual_seed(17)
        weighted_loss(new_outputs, generator).backward()
        torch.testing.assert_close(candidate.grad, original.grad, atol=1e-9, rtol=1e-8)

    def test_repeated_within_band_gradient_is_finite_for_arbitrary_loss(self):
        matrix = diagonal_tall([7., 7., 4., 4., 1., 1.]).requires_grad_()
        outputs = projector_bands(matrix, 3)
        weighted_loss(outputs, torch.Generator().manual_seed(19)).backward()
        self.assertTrue(torch.isfinite(matrix.grad).all())
        torch.testing.assert_close(sum(outputs), matrix, atol=1e-10, rtol=1e-10)

    def test_boundary_degeneracy_is_rejected(self):
        matrix = diagonal_tall([7., 5., 5., 3., 2., 1.]).requires_grad_()
        with self.assertRaisesRegex(FloatingPointError, 'boundary at index 2'):
            projector_bands(matrix, 3)

    def test_unresolved_near_boundary_is_rejected_without_jitter(self):
        matrix = diagonal_tall([7., 5. + 1e-15, 5., 3., 2., 1.]).requires_grad_()
        with self.assertRaises(FloatingPointError):
            projector_bands(matrix, 3)

    def test_single_band_identity_gradient(self):
        matrix = diagonal_tall([3., 3., 0., 0.]).requires_grad_()
        outputs = projector_bands(matrix, 1)
        outputs[0].sum().backward()
        torch.testing.assert_close(outputs[0], matrix, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(matrix.grad, torch.ones_like(matrix), atol=1e-10, rtol=1e-10)

    def test_zero_unused_band_gradients_are_supported(self):
        matrix = diagonal_tall([7., 6., 4., 3., 2., 1.]).requires_grad_()
        projector_bands(matrix, 3)[1].square().sum().backward()
        expected = torch.zeros_like(matrix)
        expected[2, 2], expected[3, 3] = 8., 6.
        torch.testing.assert_close(matrix.grad, expected, atol=1e-10, rtol=1e-10)

    def test_float32_forward_gradient_and_dtype(self):
        original = diagonal_tall([7., 6., 4., 3., 2., 1.]).float().requires_grad_()
        candidate = original.detach().clone().requires_grad_()
        old, new = original_svd_bands(original, 3), projector_bands(candidate, 3)
        for old_band, new_band in zip(old, new):
            torch.testing.assert_close(new_band, old_band, atol=1e-5, rtol=5e-5)
        weighted_loss(old, torch.Generator().manual_seed(21)).backward()
        weighted_loss(new, torch.Generator().manual_seed(21)).backward()
        self.assertEqual(candidate.grad.dtype, torch.float32)
        torch.testing.assert_close(candidate.grad, original.grad, atol=1e-4, rtol=1e-3)

    def test_nonfinite_upstream_gradients_are_rejected(self):
        matrix = diagonal_tall([7., 6., 4., 3., 2., 1.]).requires_grad_()
        output = projector_bands(matrix, 3)[0]
        with self.assertRaises(FloatingPointError):
            output.backward(torch.full_like(output, float('nan')))


if __name__ == '__main__':
    unittest.main()
