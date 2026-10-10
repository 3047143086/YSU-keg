"""Exact spectral-band mapping with a projector-level first derivative.

This is an infrastructure audit candidate, not a training integration. The
forward still obtains the original SVD. It only avoids differentiating the
unidentifiable singular-vector basis within a whole retained frequency band.
"""
import torch
from torch.autograd.function import once_differentiable

BOUNDARY_EPS_MULTIPLIER = 32.0


def split_sizes(feature_dim, bands):
    if isinstance(bands, bool) or not isinstance(bands, int) or not 1 <= bands <= feature_dim:
        raise ValueError('Require an integer band count between one and feature dimension')
    base, remainder = divmod(feature_dim, bands)
    return tuple(base + (index < remainder) for index in range(bands))


def validate_input(value):
    if value.ndim != 2 or value.shape[0] < value.shape[1] or value.shape[1] == 0:
        raise ValueError('Audit supports nonempty real tall matrices only')
    if value.dtype not in (torch.float32, torch.float64) or not torch.isfinite(value).all():
        raise ValueError('Require finite float32 or float64 input')


def check_boundary_gaps(singular, sizes):
    """Reject unresolved band boundaries instead of changing or regularizing them."""
    eigenvalues = singular.double().square()
    scale = eigenvalues.max()
    threshold = BOUNDARY_EPS_MULTIPLIER * torch.finfo(singular.dtype).eps * scale
    rows, end = [], 0
    for size in sizes[:-1]:
        end += size
        gap = eigenvalues[end - 1] - eigenvalues[end]
        rows.append({'boundary_index': end, 'eigenvalue_gap': float(gap.item()),
                     'relative_eigenvalue_gap': float((gap / scale).item()) if scale > 0 else 0.0,
                     'required_eigenvalue_gap': float(threshold.item())})
        if not torch.isfinite(gap) or gap <= threshold:
            raise FloatingPointError('Unresolved spectral band boundary at index ' + str(end))
    return rows


def original_svd_bands(value, bands):
    """Frozen-source formula used as a comparison, with no fallback substitution."""
    validate_input(value)
    u, s, vh = torch.linalg.svd(value, full_matrices=False)
    sizes, start, outputs = split_sizes(value.shape[1], bands), 0, []
    for size in sizes:
        end = start + size
        outputs.append(u[:, start:end] @ torch.diag(s[start:end]) @ vh[start:end])
        start = end
    return tuple(outputs)


class _ProjectorBands(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, bands):
        validate_input(value)
        sizes = split_sizes(value.shape[1], bands)
        u, singular, vh = torch.linalg.svd(value, full_matrices=False)
        check_boundary_gaps(singular, sizes)
        v = vh.transpose(0, 1)
        # Preserve the original forward dtype, SVD and arithmetic order.
        # X @ P is the derivative's identity, not a different numeric forward.
        outputs, start = [], 0
        for size in sizes:
            end = start + size
            outputs.append(u[:, start:end] @ torch.diag(singular[start:end]) @ vh[start:end])
            start = end
        ctx.save_for_backward(value, singular, v)
        ctx.sizes = sizes
        ctx.set_materialize_grads(False)
        return tuple(outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *output_gradients):
        value, singular, v = ctx.saved_tensors
        # Accumulate the F x F spectral derivative in float64, then cast the
        # complete first derivative back to the original input dtype. This is
        # numerical precision, not denominator clipping or regularization.
        x, basis_all = value.double(), v.double()
        eigenvalues = singular.double().square()
        denominator = eigenvalues[:, None] - eigenvalues[None, :]
        direct = torch.zeros_like(x)
        gram_gradient = torch.zeros_like(denominator)
        start = 0
        for size, gradient in zip(ctx.sizes, output_gradients):
            end = start + size
            if gradient is not None:
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError('Nonfinite upstream spectral-band gradient')
                g = gradient.double()
                band_basis = basis_all[:, start:end]
                direct += (g @ band_basis) @ band_basis.transpose(0, 1)
                h = x.transpose(0, 1) @ g
                h = 0.5 * (h + h.transpose(0, 1))
                eigen_h = basis_all.transpose(0, 1) @ h @ basis_all
                selected = torch.zeros(value.shape[1], dtype=torch.bool, device=value.device)
                selected[start:end] = True
                across = selected[:, None] != selected[None, :]
                coefficient = torch.zeros_like(denominator)
                difference = selected.to(torch.float64)[:, None] - selected.to(torch.float64)[None, :]
                # Only cross-band entries divide. A repeated eigenvalue within
                # this band or its complement is never evaluated as 0 / 0.
                coefficient[across] = difference[across] / denominator[across]
                gram_gradient += coefficient * eigen_h
            start = end
        gram_gradient = basis_all @ gram_gradient @ basis_all.transpose(0, 1)
        input_gradient = direct + x @ (gram_gradient + gram_gradient.transpose(0, 1))
        if not torch.isfinite(input_gradient).all():
            raise FloatingPointError('Nonfinite projector first derivative')
        return input_gradient.to(value.dtype), None


def projector_bands(value, bands=3):
    """Return band reconstructions; the custom function supplies first-order VJP."""
    return _ProjectorBands.apply(value, bands)
