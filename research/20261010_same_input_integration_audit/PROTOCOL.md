# Same-realized-input projector integration audit

The original-original v2 diagnostic found different inputs in nine of twelve
calls and different outputs in twenty-seven of thirty-six bands under identical
state, interaction and restored RNG. Thirteen bands exceeded the old numerical
comparison bounds. Its raw receipt
`651f5d066d03433a21926565d885050a14ecb63faae8cd5b0966aac20f565540`
is lineage evidence. The independent original-versus-projector full-forward
audit `fd42959e0a97f24169196851cd6d291477b31c59` remains failed and immutable.
This separately versioned audit addresses a narrower same-input question and
does not relax that old gate or replace its receipt.

## Scope and execution

Only this new directory is editable. Prepare and commit the code, protocol and
tests, then obtain root review before any deployment, remote test or GPU run.
The preparation state is not a numerical result or readiness change. After
review, deploy one immutable `/data/sunzixuan/same_input_integration_audit_20261010_v1`
and run actual MMRec CPU tests with CUDA excluded. A single exclusive launch
ticket permits exactly one candidate full `BRIDGE.calculate_loss` forward and
one backward of its actual scalar loss. No second full original forward, no
optimizer construction/update, evaluation, training queue, cache, solver change,
precision-setting change or deterministic/backend-setting change is permitted.
No rerun to find a pass, no explicit/application fallback and no reported internal
SVD fallback are accepted. Solver internals are not instrumented.

Require idle physical GPU1 with >=14 GiB free VRAM, >=12 GiB available RAM and
>=4 allowed CPU cores after input verification immediately before launch and
again before CUDA initialization. Use CUDA_VISIBLE_DEVICES=1/logical cuda:0,
four CPU threads and .55 process memory fraction. GPU0 is excluded. Total
process budget is30 minutes, including hashes, reconstruction, one forward and
one backward. Exceptions, timeout or guard violations are technical failures
with receipts, never passes. Never kill another process to obtain resources.

## Immutable input and algorithm lineage

Anchor the old GPU helper source, guards, gate, raw deployment config and raw
failed final receipt before importing them. Anchor the completed original-repeat
v2 raw config/receipt and every deployed v2 payload file. Reuse the old full
baseline hash guard: critical v1 receipts, bundle, all35 carrier/source files,
original data, graphs, nineteen run inputs, optional absences, archived failure
packet and finite early checkpoint. Anchor this newly deployed payload/config
before and after execution and on failures. The frozen projector source SHA256
is `e44eb739482f3fe0448e1c9fdb066e36c4d6ea22d05a0c25a32734f29a856ab6`;
load that exact file without edits, added damping, changed precision or solver.

Use unchanged frozen `construct_model`: original seed2026, dataset/loader order,
effective configuration equality, baseline assertions and strict failure-packet
loading. Reconstruct all saved preference aliases only after equality to their
canonical GCN parameter values, bind the same Parameter objects and retain all
registered paths. Drop saved optimizer state immediately; no optimizer object
is constructed. Keep original training mode and autograd enabled. Use the same
saved interaction, with pre/post tensor hashes. Restore saved Python, NumPy,
CPU Torch and CUDA RNG immediately before the single loss call; verify exact
restoration and require post-forward/backward RNG to remain unchanged.

Observe and compare deterministic algorithms/warn flags, cuDNN flags, TF32
flags and relevant environment variables before/after construction and the
forward/backward. No such flag is changed. Four CPU threads and .55 memory
fraction are declared resource settings, not algorithm changes.

## Same-X shadow and candidate output path

Temporarily wrap only `FrequencyDecompositionModule.frequency_decompose_svd`
and observe `torch.linalg.svd`, always restoring both identities in finally.
For each actual realized input X passed by the full model, first call the saved
original decomposition with X.detach() under no_grad. Then invoke the unchanged
projector with the original X carrying its actual upstream autograd graph.
Both receive exactly the same float32 tensor values; no archived internal input
is inserted, cloned into another forward, or substituted for X. Hash X before
and after the two decompositions to reject mutation. Require the detached shadow
to share X's storage and no shadow output to require grad. Return only the
candidate outputs to the actual loss path; original shadow outputs are used
for observation and promptly released.
Record X's tensor/storage data pointers, offset, stride and version before/after
each phase; require unchanged metadata as well as hashes so mutate-then-restore
cannot silently pass. Observe each phase's RNG without restoring it to hide
observer consumption. Record each SVD's call index/phase/argument contract and
successful or exceptional completion.

Require exactly twelve calls, in frozen call order, each with three bands.
Calls0–2 and6–8 have shape[192403,64], calls3–5 and9–11 shape[63001,64]. Every
call uses the frozen FeqDecomposeOperator module, M=3, float32. Require successful
unchanged SVD invocation once in shadow and once in candidate for each call,
twenty-four successful SVDs total. The original method has a Gram fallback
after SVD RuntimeError; observing any SVD failure rejects this audit even if
the original method returns fallback outputs. No observer changes a solver,
argument, precision or decomposition result. A fresh-process warning observer
runs before Torch import through construction, forward and backward, preserving
ordinary warning reporting and original filters. Reject reported SVD
nonconvergence/fallback warnings. PyTorch1.12.1 can report gesvdj nonconvergence
and internally recompute with gesvd while the Python call succeeds; successful
Python calls alone do not exclude this path. See the
[versioned PyTorch source](https://github.com/pytorch/pytorch/blob/v1.12.1/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp#L747).
This observes reported fallbacks and does not certify unknown solver internals.
Record original/candidate input
and band hashes, exact equality and descriptive errors. Each of all36 bands
must be exactly torch.equal; no approximate tolerance can replace this gate.

Require all finite scalar actual loss components and a finite scalar sum.
Only after complete twelve-call/thirty-six-band exactness and lifecycle guards
pass, call sum(loss_components).backward() once. It must return finite gradients
for a nonempty set of exactly27 independent parameters, with names matching
the frozen original GPU reference participating set. Canonical names use the
first registered path per object, retaining root preference alias names exactly
as the old original reference. Check coverage against state_dict(keep_vars=True),
excluding buffers, and inspect all alias paths so a nonfinite gradient cannot
hide behind deduplication. Missing/extra names, empty or nonfinite gradients
reject the gate. No gradient clipping, nan replacement, optimizer or step occurs.
Record every candidate output's actual custom grad_fn and requires_grad, then
use read-only hooks to observe gradients entering those36 returned outputs and
the12 actual X inputs. Require each of all12 decomposition calls to receive at
least one output gradient from the actual loss and an input gradient; every
observed output/input gradient must be finite and nonempty. An unused output is
explicitly marked unused, never called a finite zero. Hooks return no replacement
gradient. This verifies every projector call participates in the real loss path,
rather than inferring that fact solely from the27 parameter gradients.

Require learned state values/schemas, registered Parameter paths and object
identities, preference aliases and module training modes unchanged before/after
forward and backward. Gradients may newly exist after backward and are inspected
separately; they are never part of learned-state equality. Original method and
SVD function identities must be restored before backward and remain unchanged.
Release ordinary transient graph tensors after recording; never clear a
Parameter alias or change a protected file.
Record requires_grad for every registered Parameter path before/after as part
of the same immutable state guard, so a changed freeze status cannot be hidden.

## Outcomes and limits

The old GPU receipt's archived-input phase is retained as distinct evidence:
all12 candidate probes passed on archived inputs, while generic capture0 had
12,313,792 nonfinite gradient elements and the projector probe was finite.
Those were independent dL/dBand probes, not the current actual minibatch
gradients. Do not rerun them or conflate them with this full-model integration.

`completed_pass` requires all36 same-X bands exact, finite actual loss, the
exact27 participating parameter names with finite gradients, complete counts
and every state/identity/method/RNG/input/hash/backend guard. `completed_fail`
preserves a failed scientific gate; `technical_failure` preserves an exception
or guard failure. Partial counts never pass. A pass would establish same-X
forward integration and finite first-order derivatives for this realized
minibatch only. It would not establish historical crash replay, generic VJP
equivalence, convergence, recommendation gain, publication readiness or clear
the old failed gate. No outcome automatically launches recovery training.
Extra shadow SVDs, CPU hashing and synchronization instrument the execution and
alter its allocator/scheduling history. The audit is restricted to the realized
minibatch in this instrumented process; it does not reproduce an unobserved
training trajectory or the exact historical optimizer/backward crash.
