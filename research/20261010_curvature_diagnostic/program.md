# Curvature diagnostic program

This program description is frozen with `PROTOCOL.md`. It describes the
bounded audit that the implementation must perform; it is not a launcher for
formal training. The audit owner supplies the Python implementation and tests.

## Inputs

The runner reads only the isolated official MSCA carrier and existing assets:
the protected source/config manifest, Baby split-zero TRAIN interactions, image/text
feature arrays, graph caches, and the recorded completed uniform-baseline best
checkpoint. The `init` state is constructed afresh under seed 2026 before any
optimizer step. The uniform checkpoint's original selection receipt must be
copied into the diagnostic manifest. It was selected by validation Recall@20,
so `uniform_best` is labeled validation-selected throughout the report. The
program does not search checkpoints or construct a test loader. Sports is not
audited because no official Sports uniform state was produced in F28.

## Deterministic procedure

1. Verify source, input, split, feature, and graph-cache SHA256 manifests.
   Verify the checkpoint receipt corresponds to the official uniform arm.
   Record runtime, GPU, and thread settings.
2. Run CPU implementation fixtures with CUDA hidden. Stop on failure. Fixtures
   should cover normalization, symmetric excess, fixed decisions, restoration,
   and private batch/direction determinism.
3. Bind physical GPU 1, set four CPU threads, construct the official MSCA
   model, and capture `init`. Load the recorded checkpoint as `uniform_best`
   without ranking evaluation. No optimizer may be instantiated.
4. Select three fixed 2,048-row TRAIN batches once with private NumPy seeds
   20261010 (positive permutation) and 20261011 (uniform unseen negatives).
   Save IDs only in remote private `data/batches.npz`; publish its digest,
   counts, shape, and own-positive-exclusion assertions. Reuse the identical
   arrays in both states and blocks. Assert that no validation/test interaction
   is used.
5. For each state, batch, block, and direction, evaluate unperturbed BPR, full
   objective, and weighted regularizer. Record a contrastive residual if
   feasible, but never use it as a replacement decision metric.
6. Construct one full-loss gradient direction for the current block and three
   private normalized Gaussian directions from reset Torch CPU seeds
   20261020/20261021/20261022 (`rand0`/`rand1`/`rand2`). Apply the fixed 0.01
   relative Frobenius radius. Block prefixes are exactly `image_trs.` +
   `gate_v.` and `text_trs.` + `gate_t.`.
7. Evaluate `+delta` and `-delta`, restore the saved parameters exactly, and
   evaluate the unperturbed loss again. Record losses, signed excess, delta
   norm squared, normalized curvature, gradient alignment, and jitter.
8. Apply the fixed conjunctive gate in `PROTOCOL.md` after every probe is
   present. A raw/normalized or BPR/full disagreement is inconclusive. The
   regularizer is a reported decomposition, not a decision substitute.
9. Recompute source/input hashes, compare them with the original manifest, and
   emit one terminal status: `diagnostic_pass`, `diagnostic_fail`,
   `diagnostic_inconclusive`, or `technical_failure`.

## Schema contract

Schema version `1` uses these stable identities and measurements:

```text
state: init | uniform_best
batch_id: 0 | 1 | 2
block: image | text
parameter_prefixes: [image_trs., gate_v.] or [text_trs., gate_t.]
direction: full_grad | rand0 | rand1 | rand2
direction_seed: integer
rho: 0.01
norm_theta, norm_delta, norm_delta_sq: finite floats
component: bpr | full | regularizer | cl_residual
L0, Lrepeat, Lplus, Lminus, Lrestored: finite floats
E, K, floor, valid: finite/boolean
gradient_alignment: finite float or null
restore_exact: boolean
source_manifest_before, source_manifest_after: SHA256 IDs
input_manifest_before, input_manifest_after: SHA256 IDs
```

`Lrepeat` measures repeated unperturbed jitter; `Lrestored` measures the loss
after exact parameter restoration. `regularizer` is the weighted objective
contribution; the summary also records the unweighted squared parameter norm.
The dataset, state receipt, batch ID digest, seed, selected epoch, and whether
the state was validation-selected are part of the top-level manifest. Never
infer these fields from a filename alone.

## Failure handling and limits

Any NaN/Inf, zero block norm, zero full gradient, missing state, batch leakage,
source/input hash change, non-exact restoration, or one-hour timeout is terminal.
Preserve the first failure reason. Do not retry with another seed, direction,
radius, batch, block, state, or precision. A technical failure cannot become a
negative scientific metric or a recommendation. Scientific failure of the
fixed gate permits no formal training arm.

Use physical GPU 1 only and four CPU threads. No new model, API, metadata,
review, or feature encoding may be downloaded. The audit does not implement
APS, MDPS, SAM, AMR, a new loss, an optimizer, a reranker, or an automatic formal
queue. A pass is a precondition for a separately registered transfer pilot;
it is not a Recall@20 gain, user-preference observation, or novelty claim.
