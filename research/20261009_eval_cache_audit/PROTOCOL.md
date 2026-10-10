# BRIDGE eval forward cache equivalence audit

Declared 2026-10-09 (Beijing time). This is a CPU equivalence audit for an
evaluation-time implementation aid. It is not a model change, ranking method,
training run, or new metric.

The frozen carrier calls `DualFrequencyEncoder.forward(self)` from
`BRIDGE.full_sort_predict` for every evaluation batch. The source definitions
are retained unchanged (carrier `bridge.py` SHA256
`8356fd38cec389f71905d1dde2026ed721ef1747089e7bf802763aa857de4056`, encoder
`dual_frequency_encoder.py` SHA256
`287c1627a289707685028d1852105efa69b8da5106cb175c2871cacd7904e90c`). The
wrapper creates a fresh function object with the original
`full_sort_predict.__code__` and a copied globals dictionary. Only the
`DualFrequencyEncoder` name resolves to a pass-local forward shim. The carrier
class, method source, model parameters, scoring operations, masking and
evaluation loop remain untouched.

The cache is valid only inside one explicit `eval + torch.no_grad()` pass for
one model object. The first call computes and records the exact five forward
outputs; subsequent batches reuse those references. The cache is discarded at
pass exit. It refuses training mode or enabled gradients, rejects a different
model, and invalidates on parameter/buffer version changes. A new epoch,
checkpoint load, or training phase must create a new pass/cache. The scope
is exclusive: callers may not call train()/eval() transitions inside it.
The wrapper checks current mode at every access; it does not observe a
train()->eval() round trip with no parameter update or forward between them.
Such a round trip is outside the supported scope, not a detected transition.
Cache generation and hit counts are recorded. CPU and CUDA RNG states are
snapshotted around the first call and every hit; any change is a hard
non-equivalence failure.

The acceptance test loads the completed original Baby seed-2026 baseline
checkpoint from target_isolated_20261008_autoresearch_v1 and the exact frozen
BRIDGE carrier. It materializes the first four original validation batches
(64 users per batch), then compares direct and cached full-sort scores with
absolute tolerance 1e-6 and zero relative tolerance. Masked Top-50 IDs must
match item by item after the original `scores[mask[0],mask[1]]=-1e10` mask and
unchanged torch.topk call. All 35 frozen source files, the baseline inputs,
checkpoint and absence declarations are hash-checked before and after.
Model parameter/buffer state, transient forward attributes and Torch CPU,
NumPy and Python RNG states must match. CUDA may not be initialized. One
cache generation and exactly three hits are required, and pass exit must
clear its output references. A failure is recorded as `non_equivalent` and
the wrapper must not be used for Electronics. Mock lifecycle tests are
necessary checks but do not establish this actual model equivalence.

Before scores, input_audit.json pins SHA256 of the archived baseline
deployment, inputs, result, effective configuration and checkpoint. Both
core source SHAs are verified explicitly. Construct the exact checkpoint
configuration and assert equality with the archived effective configuration;
only use_gpu/device and evaluation batch/output runtime keys may change for
this CPU audit. Algorithm settings must be equal. Require at least 12 GiB
available RAM before loading the checkpoint/model. Hash-check the deployed
audit payload before and after as well as the frozen scientific inputs.

The first v1 harness exited before model construction because it addressed
bundle/src instead of bundle/carrier/src. Preserve its deployment, lifecycle
receipt and log as a technical failure; v2 corrects the path and runs this
declared acceptance test. No v1 payload is overwritten.

The v2 harness constructed the original CPU model, then failed strict
checkpoint loading before scoring. The original GPU constructor converts
the construction-only result_embed Parameter to a plain Tensor via .to(cuda);
the CPU no-op .to(cpu) leaves it registered. Original encoder forward also
registers id_preference/v_preference/t_preference as aliases of the respective
GCN preference Parameters; the trained checkpoint contains those alias keys.
V3 preserves v2 and explicitly mimics the original GPU registration shape
on the independent CPU audit model. Require each checkpoint alias to equal
its canonical gcn.preference tensor, bind exactly the same Parameter object,
require result_embed to have one source Store and no source Loads, and keep
load_state_dict(strict=True). Require exact checkpoint state-key coverage
and equality of every checkpoint tensor after load. No algorithm weight is
discarded, and no frozen source is edited. This is CPU checkpoint compatibility,
not a model change or a GPU equivalence claim.

This audit does not claim GPU bitwise equivalence or complete Electronics
validation. It does not touch the active Electronics baseline process, its
graph cache, or the 35 frozen carrier source files. No new ranking metrics are
computed.
