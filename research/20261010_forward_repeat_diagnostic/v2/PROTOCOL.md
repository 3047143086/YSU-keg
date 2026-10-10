# Two original BRIDGE forwards: numerical repeatability diagnostic

This is a numerical infrastructure diagnostic under the Karpathy autoresearch
`program.md` workflow. The preceding immutable GPU audit deployed revision
`fd42959e0a97f24169196851cd6d291477b31c59` and completed with a failed full-model
original-versus-projector forward gate. Its original and projector full forwards
had slightly different upstream SVD inputs. This diagnostic measures whether
two original full forwards also differ. It is not an algorithm experiment,
projector integration approval, recovered training, or a historical crash replay.

Only this new directory is editable. The original BRIDGE carrier, original
Electronics v1 bundle, projector audit, failure-model audit and their receipts
are read-only. No global result document or export repository is edited here.

## Fixed execution

Version v1 ended with a technical failure before any measured forward: its
public `named_parameters(remove_duplicate=False)` call was incompatible with
the actual model/Torch API. It has no repeatability result and is retained
unchanged in `../archives/v1`. This v2 changes only Parameter-path enumeration
and adds actual CPU Torch alias-coverage tests. It keeps all scientific gates,
two-forward scope, GPU selection, budget, lineage and cleanup unchanged.

Root reviews a committed payload before deployment or GPU execution. Deployment
is to the new immutable `/data/sunzixuan/forward_repeat_diagnostic_20261010_v2`.
Run CPU-only structural tests in the actual MMRec interpreter before a launch.
There is exactly one launch ticket and exactly two original `calculate_loss`
calls, without backward, optimizer creation/update, validation/test evaluation,
training queue, model cache, projector invocation, solver selection or backend
configuration change. No third attempt, rerun to find a pass, or fallback exists.

Require physical GPU 1 to be idle with >=14 GiB free VRAM, >=12 GiB available
host RAM and >=4 allowed CPU cores immediately before launching and before CUDA
initialization. Physical GPU 0 is excluded. Use logical `cuda:0` through
`CUDA_VISIBLE_DEVICES=1`, four CPU threads and a .55 process memory fraction.
The total process budget is 30 minutes, including construction, both forwards,
hashing and cleanup. An exception, timeout or changed input is a technical
failure with a final receipt, never a numerical pass. No process is killed to
obtain resources.

## Frozen lineage and reconstruction

Before importing the original audit helpers verify that its `remote_gpu.py`,
`guards.py`, `audit_gate.py`, deployed `audit_config.json` and final failed receipt
match declared SHA256 anchors. Require the deployed source revision above and
the archived receipt `completed_fail`. Reuse the original `guards.verify` to
verify all v1 critical receipts, bundle, 35 carrier/source files, original data,
nineteen run inputs, graph files, optional absences, finite early checkpoint,
failure packet, CPU audit lineage and this new committed payload. Repeat all
checks after the forwards and on failure. The CPU projector source is hashed
only as a lineage dependency; no projector function is imported or executed.

Use the unchanged frozen audit `construct_model` helper: original dataset and
loader construction order, scalarized effective config equality, seed 2026,
GPU float32 architecture, original baseline assertions and strict loading of
the failure packet model. Reconstruct preference aliases before measuring
state; require saved alias values equal their canonical GCN keys and each bound
attribute be the very same Parameter object. Keep the model in training mode
and autograd enabled to exercise the original training loss forward; do not
call backward. Drop the saved optimizer state immediately after loading the
packet; no optimizer object is created. The saved failure interaction remains
unchanged, and its tensor hashes are checked before and after both calls.

## Two forwards and observations

Restore the packet's Python, NumPy, CPU Torch and CUDA RNG immediately before
each call using the unchanged frozen `restore_rng`. Verify the restored RNG
hashes exactly against the packet and compare the two post-forward RNG states.
Read deterministic-algorithm/warn flags, cuDNN flags, TF32 flags, CUDA/cuDNN
versions and relevant environment variables before and after construction and
forwards; do not set any of those flags or change the environment to force a
numerical outcome. The original constructor's initialization changes RNG, so
its packet restore precedes both measured calls.

Temporarily observe the original `frequency_decompose_svd` method. The observer
calls the saved original method once with unchanged arguments and returns its
original outputs. It records a detached CPU clone of the input and all three
outputs. Always restore the method in `finally`, including exceptions. Require
twelve calls and three bands per call for each forward. Preserve call order
and module names to reject misaligned comparisons. No archived internal tensor
is substituted into the forward. Hash the captures and compare them with the
archived inputs only to report archive fidelity, not as a repeatability gate.

After the first call, only detached CPU captures and scalar losses survive.
Release graph-carrying transient attributes `id_rep`, `v_rep`, `t_rep`, and
`result_embed` when they are ordinary tensors, then collect Python garbage and
empty the CUDA allocator cache. Never clear or rebind a Parameter. Record this
cleanup, then restore exactly the same packet RNG and perform the second call.
Use identical cleanup after the second call. This controlled two-call sequence
does not reproduce the old optimizer/backward/allocator history, and two calls
cannot estimate a general probability of non-repeatability.

Hash all `state_dict` values and record all named Parameter identities,
including aliases, before, after the first call and after the second. Require
exact value equality, identical keys/dtypes/shapes, identical Parameter objects,
alias identity, unchanged module training modes, no parameter gradients and
unchanged deterministic/backend settings. State is verified, never reloaded
between the two calls to hide a mutation.
Enumerate parameters read-only through each module's `_parameters` and `_modules`
registrations, retaining every alias path and rejecting registration cycles.
Require exact name/object coverage against `state_dict(keep_vars=True)` Parameter
entries, excluding buffers. Actual MMRec CPU tests use genuine shared Parameters
and shared child Modules. Repeat a CPU synthetic alias check before any frozen
model construction; validate the real strict-loaded model's parameter-state
coverage before each measured forward and afterward.

## Fixed comparisons and interpretation

Report exact elementwise equality for 12 inputs, 36 bands and every loss
component. Reuse the frozen audit's `compare` defaults unchanged: relative
Frobenius <=5e-5 and maximum absolute error <=5e-4*max(1,maxabs(first)). Loss
components retain rtol 5e-5 and atol 5e-6. These inherited bounds are diagnostic
reference columns, not permission to change a failed integration gate. Report
every row, per-family min/median/max errors, counts of exact/non-exact values,
and counts exceeding the inherited bounds; no threshold or solver is searched.
If a reference tensor has zero norm and a nonzero difference, the unchanged
comparison has undefined relative error and fails its inherited bounds. Retain
that row and report its undefined count rather than replacing it with zero.

`completed_exact` means all observed inputs, bands and losses were exactly
equal in this one two-forward diagnostic. `completed_different` means at least
one observed input, band or loss differed with all lineage/state/lifecycle
guards intact. Both are completed diagnoses, not candidate acceptance. If two
original forwards differ, the previous separate-forward comparison cannot
attribute all differences solely to the projector. If they agree here, that
does not prove projector causality, historical crash replay or equivalence.
No outcome clears the earlier failed GPU gate or starts training. Any further
numerical recovery requires its own fixed protocol, version and root review.
