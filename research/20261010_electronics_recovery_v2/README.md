# Electronics numerical recovery v2: frozen preparation, not eligible to run

This directory prepares a fresh validation-only five-seed baseline using the
original SVD forward arithmetic and the reviewed spectral-projector first VJP.
It is infrastructure recovery, not a recommendation innovation. The solver,
optimizer, parameter space, learning rate, maximum300 epochs, patience20 and
strict rounded-four-decimal Recall20 selection remain original. The backward
implementation changes and is labeled explicitly in every recovery receipt.

**No recovery deployment or training has occurred.** The frozen true-model
GPU audit is completed_fail and prerequisites.json has ready_for_training=false.
Changing an overall status flag cannot rescue failed individual model bands.
Missing receipts, SHA drift, source/configuration changes and nonpassing
numerical guards all deny launch before a GPU query or loader construction.
Resolving this failure requires a separately preregistered audit version; do
not edit the archived receipt or its hash to make this preparation eligible.

The CPU projector audit passed all12 captures. Its archived GPU-input phase
also passed, but the true-model GPU integration phase failed6 of12 calls under
the frozen spectral-band tolerance. The old historical crash was not exactly
reproduced. Separately, Electronics evaluation caching was non-equivalent
(maximum score error0.00026226;3/6 maskedTop50 differences). The cache is
excluded; train.py, Trainer.evaluate and final_validation retain original
uncached code. The two negative receipts remain archived and hashed.
The local receipt files are semantic JSON exports; receipts/export_manifest.json
records their local hashes separately from the exact original remote-file hashes.
Launch gates check the original remote files and their reviewed source manifests.

The proposed per-seed wall-time bound is24h, an explicit change from v1's6h;
the total queue remains72h including waits. The old epoch mean1506.1855s
implies the minimum22 completed epochs before patience can stop take9.20h.
24h suggests about57 epochs, not guaranteed convergence or300epochs. Five jobs
at the limit require120h; a72h queue can remain incomplete. Pending, crash
and timeout seeds remain visible and are not a completed baseline. See
BUDGET_ASSESSMENT.md and the retained timing evidence.

The prepared harness binds one model instance, checks algorithm configuration
against the v1 archive, protects all35 original carrier source/config files,
the three original assets and original graph SHA, and uses fresh fixed seeds
2026–2030 without retry/resume/fallback. Queue and launch locks are single use;
GPU1 process identities are refreshed and unrelated processes are never killed.
Resources are GPU1 only, CUDA fraction0.55,4 CPU threads, at least14GiB free
GPU RAM and12GiB host RAM. No test loader is constructed or evaluated.

Validation:34/34 tests passed in actual MMRec on CPU with zero skips/errors/
failures,4threads, CUDA hidden and uninitialized (0.500972s). These tests check
the first derivative, finite-step protection, failure-receipt conjunction,
argument/configuration mutation rejection, original-source lineage and
instance-local integration. They used a temporary CPU sandbox, no data loaders
or recommendation metrics. The queue and true model have not run; CPU unit
tests do not override the failed GPU integration audit. All implementation
files and generated remote-manager source also parse successfully.

Root review is required before any use of this preparation. The current
frozen prerequisite result forbids start regardless of that review.
