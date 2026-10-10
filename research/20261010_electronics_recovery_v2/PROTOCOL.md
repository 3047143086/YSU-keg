# Electronics baseline numerical recovery: preparation

Declared 2026-10-10 before recovery training or recommendation metrics.
This protocol is a draft until its audited implementation and prerequisites
are committed. No deployment or training is authorized by a missing receipt.
User authorization to continue experiments is already present; technical
gates are checked by the active agent without asking the user again.

Preserve the original Electronics v1 technical failure and all its artifacts.
The v1 queue completed 0/5 seeds. Its partial checkpoint is a diagnostic input,
not a finished baseline and not an initialization for the new five-seed runs.
The new numerical implementation must be named explicitly in any result table.

## One numerical recovery and prerequisites

Use the independently reviewed spectral projector autograd implementation.
Keep the original torch.linalg.svd, dtype, frequency split and exact original
U_band @ diag(s_band) @ Vh_band forward arithmetic. Replace only its first
derivative with the mathematically equivalent band-projector VJP. Do not add
jitter, choose a different solver, detach the basis, clip gradients, skip
updates or regularize singular-value denominators. Unresolved cross-band
boundaries must fail before an optimizer update. Internal repeated singular
values may use the defined band mapping derivative.

Before any training, require all meaningful small-matrix derivative tests,
the complete twelve-input CPU failure replay audit, and a separately declared
actual BRIDGE GPU audit. The GPU audit must compare the original and repaired
model at the same finite failure-packet state, same interaction and RNG when
reconstructible, check exact forward values and losses, compare gradients
where the original has finite gradients, and establish finite repaired
gradients. If original input replay differs due to missing transient state,
report that limitation and do not claim exact reproduction of the old failure.
No optimizer update occurs during this equivalence audit.

The optional evaluation cache is explicitly excluded. Its Electronics GPU
audit failed the fixed score and masked-Top50 gates: maximum absolute score
difference 0.00026226 and three of six batches with different Top50 IDs.
The Baby CPU pass does not override that failure. Trainer.evaluate and final
validation use the original uncached paths, and no cache code is deployed.
Archive the negative GPU receipt as exclusion evidence, not a passing gate.
A passing projector audit permits preparing the new recovery queue; it does
not establish convergence or an algorithm contribution.

## Protected inputs and training

Snapshot all thirty-five frozen v1 carrier source/configuration files without
editing any of them. Bind the reviewed numerical method to the recovery model
instance only, retaining identical parameter objects and state keys. Use the
three audited original Electronics assets and the v1 graph cache with its
recorded SHA256, linked individually into the independent recovery directory.
Verify source, input, graph, implementation and dependency hashes before and
after each job. Preserve the original sparse behavior backend and scoring.

Run fresh seeds 2026,2027,2028,2029,2030 in that order, with no seed selection
and no automatic retry. Keep original Electronics hyperparameters, Adam,
learning-rate schedule, batch2048, validation batch64, maximum300 epochs,
stopping_step20 and strict rounded-four-decimal validation Recall20 selection.
Keep the earlier checkpoint on ties. Save configuration, RNG/loader order,
model/optimizer/scheduler, progress, finite-step counts, resource use,
per-user final validation arrays and every crash. Do not construct a test
loader or evaluate test. No new mechanism is selected by these baseline runs.

Only physical GPU1, one own GPU process at a time, four CPU threads,
per-process CUDA memory fraction0.55, available GPU RAM>=14GiB and host
RAM>=12GiB. Refresh process identities and resources before each GPU job;
never use GPU0 or terminate unrelated tasks. Queue budget72h including waits;
each training seed at most24h. This changes v1's six-hour wall-time limit,
while leaving its epoch, optimizer and selection rules unchanged. Original
v1 completed nine epochs in about3.8h; six hours is insufficient even for the
earliest22-completed-epoch patience stop (initial best plus21 bad epochs).
A24h seed limit permits about57 epochs at that
observed rate and does not guarantee convergence or300 epochs. Five seeds at
the limit cannot all fit in72h. Queue exhaustion preserves incomplete states
and is not a completed baseline. See BUDGET_ASSESSMENT.md for measured times.
Stop the queue on numerical or source failure,
not on an unfavorable validation score, and preserve all queued seed states.

Remote root: /data/sunzixuan/electronics_baseline_20261010_recovery_v2.
Freeze a dedicated branch and committed payload before deployment. Use a
single-use launch receipt and exact process-command verification. Root review
and actual environment tests precede the detached bounded queue launch.
The backward implementation changed; the solver, optimizer and parameter
space did not. Receipts must state those facts without the ambiguous claim
that all learning arithmetic is unchanged. Missing or nonpassing prerequisite
receipts fail closed, and the preparation commit does not authorize a launch.
Report numerical recovery separately from innovative methods and publication
confirmation; the latter still follows result/PUBLICATION_PROTOCOL.md.
