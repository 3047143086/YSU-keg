# Compatibility-only v2 review before execution

V1 failed before any measured forward due to its public named-parameter API.
This v2 changes only read-only Parameter enumeration and adds actual CPU Torch
coverage tests. V1 code and raw receipts are preserved at `../archives/v1`.

`parameter_snapshot.py` recursively reads `_parameters` and `_modules`, keeps
all alias paths, skips None registrations and rejects module cycles. It checks
exact path/object coverage against Parameter entries of state_dict(keep_vars=True),
with buffers excluded. It never calls the incompatible public enumerator.
Actual CPU tests cover shared Parameters, shared child-module paths, an old
public API signature, buffer exclusion, None entries and cycles. The synthetic
CPU check also runs before the frozen BRIDGE construction; the real model's
coverage is checked on every state snapshot.

Prepared scope: exactly two original `calculate_loss` calls in train mode with
autograd enabled, no backward and no optimizer. Uses the frozen predecessor's
construction, RNG restore, tensor comparison and learned-state hashing helpers.
The original decomposition observer is restored in every `finally` path.

Predeclared failure packet SHA256:
`7b6f12731715c0315018f6b3c903b39a8829cc7727035e18eb92eeaa4bd3deb7`.
Predecessor revision: `fd42959e0a97f24169196851cd6d291477b31c59`.
Raw remote JSON hashes (not local reserialized JSON): audit config
`42364aa8cb84e44d243fc58fbdd89420fbe7c2a68804897a625ceceaa5460cdd`,
failed receipt `a840a3a1d952a8300d44951af77d8068b28770057a0ba3ca1dc833e0c21b7eb0`.
The local read-only `manage.py anchors` query independently confirmed all five
anchors and both source revisions before this preparation freeze.

Receipt distinguishes exact/nonexact observations from inherited threshold
columns, and always denies projector equivalence, historical crash replay and
candidate acceptance. Deterministic/backend settings are observed, never set.
State values, schema, all Parameter identities including aliases, modes,
gradients, interaction, packet RNG and backend settings are guarded per call.
CPU captures survive graph cleanup. No state reload hides mutation.
The cache audit's six transient forward attributes comprise three graph-carrying
representations and three preference aliases. The aliases are Parameters after
strict reconstruction and must remain registered and identical; only ordinary
representation tensors and the unused ordinary `result_embed` are cleared.

Local Python 3.13 structural/error-summary tests: 12/12 passed. Syntax compilation
also passed. Local Torch alias tests are unavailable because this local Python
has no Torch. Actual MMRec must pass all 17 tests with CUDA excluded before
launch. These tests are not actual-model or GPU numerical evidence.

Commands after root review, from the workspace root:

```powershell
python research/20261010_forward_repeat_diagnostic/v2/manage.py deploy
python research/20261010_forward_repeat_diagnostic/v2/manage.py test
python research/20261010_forward_repeat_diagnostic/v2/manage.py start
python research/20261010_forward_repeat_diagnostic/v2/manage.py status
python research/20261010_forward_repeat_diagnostic/v2/archive.py
```

Launch obtains a fresh resource gate, checks all protected inputs and the
matching passing test receipt, then creates an exclusive launch ticket. The
process repeats its resource/hash guards before CUDA initialization. Do not
use `start` until root has coordinated exclusive GPU1 availability.

No remote deployment, CPU environment test or GPU launch has occurred at this
preparation checkpoint. Root review remains required before those steps.
