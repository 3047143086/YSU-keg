# Fixed-feature AMR precheck

Frozen 2026-10-10 on branch `precheck-frozen`. This is a one-shot, TRAIN-only
source feasibility check after the independent MSCA curvature diagnostic was
inconclusive. It is not recommendation training and cannot select a checkpoint
or hyperparameter.

The carrier is official MSCA `models/msca.py`, SHA256
`066aafcf2e5373b20b2fdf9b3660302b8dfcaf580710d3ab864fac4f19e2360b`. Use Baby
only, seed 2026, and exactly three fixed 2,048-row TRAIN batches from the
existing private curvature archive. Public output contains only batch hashes,
shapes, counts, and aggregate values; item IDs remain private.

The intervention is one-step image-feature perturbation. For each unique image
row appearing as a positive or negative in a batch, compute the official BPR
gradient with respect to that feature row and use

```text
delta_i = .01 * ||x_i||_2 * grad_i / ||grad_i||_2
```

The random null uses the same rows and exactly the same per-row delta norm with
a private normalized Gaussian direction. The clean, targeted, and random paths
use identical user/positive/negative tensors and the unchanged official BPR
formula. No optimizer, validation/test loader, checkpoint selection, graph
rebuild, or feature-array write is allowed.

Implement perturbations functionally by cloning the image feature embedding and
replacing only selected rows with `index_copy`; do not use `.data` mutation.
The official `image_original_adj`, `text_original_adj`, `norm_R`, and
`norm_adj` remain fixed. A clean loss computed through the wrapper must equal
the original MSCA loss within the predeclared float32 tolerance. After every
view, parameters, buffers, graphs, global RNG, source files, and feature/cache
hashes must restore exactly.

The precheck is eligible only if all rows have finite nonzero feature norms and
finite nonzero attack gradients, clean parity passes, and targeted BPR loss
increase exceeds random loss increase on at least two of three batches. A
failed row or parity check is `blocked`; a targeted-vs-random failure is
`diagnostic_fail`. No epsilon, modality, row subset, graph policy, or null may
be changed after seeing results.

This mechanism is already established by AMR (2018), BSAM (TKDE 2025) on
Amazon Baby/Sports, and VARMAT (AAAI 2026). Passing supports only a bounded
transfer precondition. It does not authorize formal training or a new-method
claim.
