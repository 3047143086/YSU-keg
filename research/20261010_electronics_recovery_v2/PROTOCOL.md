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

An optional single-pass evaluation cache is separate infrastructure. Integrate
it only after its actual original-BRIDGE Baby CPU and Electronics GPU audits
pass fixed full-sort score and exact masked Top50 checks, state/RNG/hash
guards and cache lifecycle tests. A cache is valid for one model in one
exclusive eval/no_grad pass with no training transitions or checkpoint loads.
Train loss and optimizer steps never use this cache. Measure its wall time;
it is not a recommendation improvement.

The two numerical/cache changes must retain separate audit receipts and
attribution. A passing audit permits preparing the new recovery queue; it
does not establish convergence or an algorithm contribution.

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
each training seed at most6h. Stop the queue on numerical or source failure,
not on an unfavorable validation score, and preserve all queued seed states.

Remote root: /data/sunzixuan/electronics_baseline_20261010_recovery_v2.
Freeze a dedicated branch and committed payload before deployment. Use a
single-use launch receipt and exact process-command verification. Root review
and actual environment tests precede the detached bounded queue launch.
Report numerical recovery separately from innovative methods and publication
confirmation; the latter still follows result/PUBLICATION_PROTOCOL.md.
