# Original failure-model GPU infrastructure audit

This audit follows the committed spectral-projector CPU audit at
`a12c750dbba12205ae049cc143a123b59c6b766f`. Its thirteen small CPU tests and all
twelve archived CPU input checks passed. The CPU recomputation separated the
original CUDA repeated singular pair, so that result did not reproduce the
CUDA failure. This protocol is fixed before any new GPU probe results.

All original baseline v1 files and the CPU audit payload remain immutable.
This directory owns only its audit code and receipts. It performs forward and
backward checks without an optimizer update, training queue, validation loop,
test loader, or test evaluation. A passing audit is infrastructure evidence,
not an innovation contribution or completed baseline.

## Resources, lineage and execution

Root reviews the committed code before execution and coordinates exclusive
use of physical GPU 1 after the cache audit has exited. GPU 0 is never selected.
Before CUDA initialization, query physical GPU 1 and require no compute
process, >=14 GiB free GPU memory, >=12 GiB available RAM and >=4 CPU cores.
Use CUDA_VISIBLE_DEVICES=1, logical cuda:0, per-process memory fraction .55,
and four CPU threads. Budget is thirty minutes including both probe phases;
any failure is recorded and the audit exits without an update or fallback.

Before and after execution verify all original v1 critical receipts, bundle,
35 deployed/source carrier files, three original inputs, nineteen run inputs,
optional absences, archived failure packet and early checkpoint. Also verify
the frozen CPU projector source and this committed payload. Fresh initialization
uses the exact original dataset/loader/configuration/seed 2026 construction
order, with current configuration checked against the original JSON. The
saved model is strict-loaded. Top-level preference aliases created by original
forward are reconstructed only after verifying equality to their saved GCN
preference tensors. No missing learned weight may be ignored.

## Phase A: all archived inputs on the original GPU backend

For each of the twelve original float32 SVD input tensors, retain float32
through CUDA SVD and boundary checks. Compare the frozen original
U_b@diag(s_b)@Vh_b formula and the committed custom function's identical forward
arithmetic. Record all singular values, band-boundary gaps and exact output
equality. Independent dL/dBand probes use a fresh CPU generator with seed 2026
for each capture, then transfer those float32 weights to GPU 1. They are not
the original training gradients. Require finite candidate input gradients.
Where generic gradients are finite, compare VJP errors using the CPU protocol's
relative Frobenius <=1e-3 and maxabs <=1e-3*max(1,maxabs(reference)). Generic
nonfinite gradients are failed/unavailable comparisons, never matched results.

## Phase B: full saved model and same failed minibatch

Reconstruct the original model from the frozen carrier/data with the exact
initial seed and loader order. Load the finite model from failure_replay.pt.
Use its saved interaction unchanged and restore Python, NumPy, CPU Torch and
CUDA RNG immediately before each calculate_loss call. Capture all twelve
SVD inputs and band outputs for the original and custom derivative runs. No
optimizer is constructed or called. The model stays in training mode so this
checks the actual loss path, without performing training updates.

For exact historical replay, all twelve original-run input tensors must match
the archived tensors elementwise and the original model backward must reproduce
nonfinite gradients. Otherwise report a failure-model integration probe, and
state explicitly that the historical crash was not exactly reproduced.
Saved singular values alone or approximate input agreement are insufficient to
claim exact replay. CUDA aggregation may produce different floating-point
rounding; preserve the comparison instead of substituting archived internal
inputs or forcing another backend.

For current original-versus-candidate integration, every input and each of 36
band outputs must have relative Frobenius error <=5e-5 and maximum absolute
error <=5e-4*max(1,maxabs(reference)); record exact equality separately. Loss
components use rtol 5e-5, atol 5e-6. Candidate gradients for every participating
learned parameter must be finite. If the current generic full-model gradient
is finite, require matching participating parameter sets and relative Frobenius
<=1e-3 and maxabs <=1e-3*max(1,maxabs(reference)) per parameter; otherwise report
the generic gradient comparison unavailable. Learned state_dict values must
remain exactly unchanged before/after both runs. No threshold may be relaxed
after observing any probe result.

Record exact archive fidelity, forward/loss equivalence, candidate full-gradient
validity, generic-gradient availability and full-gradient equivalence as
separate outcomes. Do not collapse them into an unsupported claim that v1
training has recovered. Any later baseline queue requires its own protocol,
commit and root review.
