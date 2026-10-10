# MSCA block-curvature diagnostic protocol

Protocol freeze: 2026-10-10. This is a read-only diagnostic for a
MASAM-motivated hypothesis. It is not a training protocol and cannot select a
model, radius, branch, or checkpoint. No optimizer step, validation evaluation,
test evaluation, recommendation, or GPU launch is authorized by this file.

## Question and prior-art boundary

The test asks whether MSCA's image and text transform/gate blocks have a
stable difference in *symmetric local loss curvature* at the same relative
parameter radius. This is narrower than asking whether a one-sided perturbed
loss is larger: a one-sided difference is confounded by first-order gradient
magnitude. It is also different from F10, which measured gradient direction and
work alignment and found no material conflict.

This is a transfer diagnostic only. MASAM (ICLR 2026) uses convergence speed and
gradient alignment to allocate sharpness-aware perturbations across modalities,
but its public code targets classification datasets and has no MSCA Amazon
ranking implementation. BSAM (TKDE 2025) already evaluates sharpness-aware
multimodal BPR recommendation on Amazon Baby and Sports, including random
8:1:1 splits and SAM/ASAM comparisons. A positive result cannot be called a
new SAM, MASAM, or recommendation algorithm.

## Frozen states and blocks

Use Baby only, because F28's uniform Baby arm completed and the conditional
Sports gate stayed closed. No Sports state or batch is part of this diagnostic.
Use exactly two states, in this order:

1. `init`: a fresh official MSCA initialization at seed 2026,
   before any optimizer step.
2. `uniform_best`: the already-produced uniform MSCA baseline checkpoint whose
   epoch was selected by the ordinary validation Recall@20 rule. This state is
   explicitly **validation-selected** and is not a pure TRAIN-only state. The
   diagnostic must not change or reselect that checkpoint.

The uniform Baby checkpoint has SHA256
`6f6ee28429668b9bb7d54d4673252c8751978f22cd24db334d03d7b4234acbd8`,
402,966,577 bytes, and zero-based selected epoch 44. Bind `init` to the initial
parameter-state digest from that completed uniform arm's receipt. If either
state digest does not match its recorded artifact, stop as a technical failure.

For both states, perturb exactly these parameter blocks:

* image block: parameters under `image_trs` and `gate_v`;
* text block: parameters under `text_trs` and `gate_t`.

Do not include `image_embedding`, `text_embedding`, `item_id_embedding`,
`user_embedding`, `query_common`, graph caches, or any shared parameter in a
block. This restriction makes the comparison auditable but means the diagnostic
does not establish dominance of the large feature tables. A missing, empty, or
zero-norm block is a diagnostic failure; do not invent a fallback block.

## Data, batches, and random state

Use exactly three deterministic 2,048-row TRAIN minibatches from the official
split, identical for both blocks and both states. Select positives with a
private `np.RandomState(20261010)` permutation of TRAIN rows and negatives with
private `np.RandomState(20261011)`. Save arrays to the private remote
`data/batches.npz` before any loss is inspected; public receipts contain only
digests, counts, shapes, and own-positive-exclusion assertions. No validation or test row may enter a
batch. Uniform negatives are drawn from the unique global TRAIN-observed item
union, excluding all TRAIN positives for the corresponding user. Keep the
three selected positive orders fixed; do not resample negatives during the
diagnostic.

Use private Torch CPU generators, reset identically for each state/batch/block,
with seeds 20261020, 20261021, and 20261022 for `rand0`, `rand1`, and `rand2`.
They must not consume the training RNG stream. For each state, batch, and block,
construct:

* `full_grad`: the gradient of the exact full MSCA training objective with
  respect to that block;
* `rand0`, `rand1`, `rand2`: independent private normalized Gaussian directions
  with the same block shape.

The gradient and each random direction are normalized to unit norm before the
radius is applied, so their block deltas have exactly the same norm. A zero
gradient is recorded as invalid; no epsilon direction is substituted.

## Perturbation and measurements

The only radius is `rho = 0.01`, interpreted as a relative Frobenius radius for
each block. For a block vector `theta_m` and unit direction `d`, set

```text
delta_m = rho * ||theta_m||_2 * d / ||d||_2.
```

The same relative rule is used for gradient and all random directions. Evaluate
the unperturbed batch loss `L0`, then `L(theta + delta)` and
`L(theta - delta)` with a temporary, exact state substitution. The symmetric
excess for every reported component `q` is

```text
E_q = 0.5 * (q_plus + q_minus) - q_zero
K_q = 2 * E_q / ||delta_m||_2^2.
```

Record signed `E_q`, signed normalized `K_q`, all three raw losses, the block
norm, delta norm, and direction norm. Report these components separately:

* `bpr`: the official BPR term;
* `full`: the exact objective passed to the official loss, including its
  contrastive terms and weighted regularizer;
* `regularizer`: the weighted regularizer contribution, plus the unweighted parameter
  squared norm for auditability.

The regularizer has its own parameter-scale curvature. It must be reported and
must not be used to replace BPR or full loss. If BPR and full (or raw `E` and
normalized `K`) disagree on the winning block, the probe is inconclusive; do
not choose the favorable scale after seeing metrics.

Before reading probe values, freeze the numeric validity floor:

```text
loss_scale = max(1, abs(q_zero), abs(q_plus), abs(q_minus))
floor_q = max(1e-7, 1e-5 * loss_scale)
floor_K = 2 * floor_q / max(||delta_m||_2^2, 1e-30)
```

All values must be finite. A comparison is positive only when the candidate's
`E_q` exceeds its own `floor_q` and exceeds the other block's `E_q` by more
than the sum of both blocks' floors; apply the analogous conditions to `K_q`
and `floor_K`. The floor formula is fixed for every state, batch, direction,
and component. If repeated or restored unperturbed loss differs by more than
this floor, mark the batch invalid rather than relaxing the floor.

## Restoration and source protection

Take a complete source/input SHA256 manifest before and after the run. At
minimum it covers the official MSCA source (including
`models/msca.py`, SHA256
`066aafcf2e5373b20b2fdf9b3660302b8dfcaf580710d3ab864fac4f19e2360b`;
the complete manifest is authoritative), split interactions, image/text feature
files, and graph-cache files. The audit must fail if any protected hash changes.

After every `+delta` and `-delta`, restore every parameter and buffer and verify
bitwise equality with the saved state (`torch.equal`, not a tolerance). Verify
optimizer state is untouched; the diagnostic must not instantiate an optimizer.
Record restoration failures and abort the diagnostic. A final source/input
manifest must match the pre-run manifest exactly.

## Fixed dominance gate

The gate is evaluated before any recommendation is discussed and cannot be
changed after metrics are known. For each state, batch, and direction, a block
is a **valid winner** only if it wins all four comparisons: raw BPR `E`,
normalized BPR `K`, raw full `E`, and normalized full `K`, each positive above
its fixed floor and greater than the other block by the sum of the corresponding
floors. The `regularizer` values are a diagnostic decomposition and never supply
a missing win. Three TRAIN batches are a descriptive screening sample, not
independent evidence for statistical significance. Directional finite
differences do not estimate a maximal Hessian eigenvalue or worst-case sharpness.

The same block must satisfy all of these conditions:

1. In each state, at least two of the three batches have a valid winner in at
   least three of four directions (`full_grad` plus the three random directions).
2. In each state, the `full_grad` direction must itself produce a valid winner
   in at least two of the three batches.
3. The winning block must be the same in `init` and `uniform_best`.
4. Every required probe must be finite, restored exactly, and hash-clean.

Failure of any condition is `diagnostic_fail` or `diagnostic_inconclusive`; it
does not justify a branch choice, a SAM update, or a performance claim. Passing
only establishes a preregistered curvature precondition for a separately
reviewed transfer pilot. It does not select `rho`, infer user preference, or
establish paper-level novelty.

## Resource and output limits

Use physical GPU 1 only, with `CUDA_VISIBLE_DEVICES=1`, at most one hour wall
time, and four CPU threads (`OMP`, `MKL`, OpenBLAS, NumExpr, and Torch). CPU
unit tests may hide CUDA. Do not touch GPU 0 or start an old queue. No new model,
metadata, review, API, or feature extraction may be downloaded.

Write a receipt, `probes.tsv`, `states.json`, and pre/post manifest. The TSV
must include state, batch, block, direction, seed, rho, parameter norm, delta
norm, `L0/Lplus/Lminus` for BPR/full/regularizer, signed `E` and `K` for each component,
validity floor, restoration result, and source/input manifest IDs. A separate
summary records the fixed gate outcome, invalid-probe reasons, and whether the
state was validation-selected. No recommendation report may be generated by
this diagnostic itself.
