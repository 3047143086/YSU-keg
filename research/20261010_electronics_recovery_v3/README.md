# Electronics numerical recovery v3: reviewed preparation

This is a new bounded numerical baseline feasibility preparation, using the
original uncached SVD forward and frozen band-projector first VJP. It changes
backward precision/implementation; solver, optimizer, parameter space, batch2048,
eval64, maximum300 epochs, patience20 and validation selection remain original.
The 35 frozen carrier files, train.py, numerics.py and projector.py retain their
original byte hashes. This is infrastructure recovery, not an innovation.

The independent same-realized-input audit passed: one candidate full forward,
one loss backward, 12/12 inputs, 36/36 exact shadow bands, 36 finite output hooks,
12 finite input hooks and 27 finite participating parameter gradients. Its
31/31 actual CPU tests also passed without skips. Full source/configuration/raw
anchors are in prerequisites.json and receipts/exact. The old13 mathematical
tests, archived12 CPU/GPU probes, original-repeat differences and cache negative
receipt are independently required. The old full-model audit remains failed,
its historical crash remains unreplicated, and v2 remains ineligible. A one-batch
pass does not establish long-run stability, v1 trajectory equivalence or gain.

RecoveryTrainingGuard records actual SVD inputs even during custom forward's
implicit no-grad. It retains singular=None on an unsolved failure and explicitly
marks unavailable generic output gradients. Nonfinite intermediate/parameter
gradients, reported solver fallback and original solver exceptions stop without
retry. Observers add no solver call or optimizer change and release each batch's
hooks/captures. Only post-update finite Adam/parameter state counts as a finite
step. Diagnostic snapshots add memory/time overhead; abrupt termination may
prevent saving a full packet. Silent unreported library fallback is not ruled out.

The budget is24h per seed and72h total including waits, explicitly changed from
v1's6h per seed. The old mean1506.185479s per completed epoch implies the earliest
22-epoch patience stop needs9.20h;24h suggests about57 epochs. Five maximum jobs
need120h, so all five seeds may not finish. Fixed fresh seeds2026-2030, no
resume/retry/fallback/test; incomplete seeds remain visible. Future controls must
use the same numerical fix and matched rules. See PROTOCOL.md and budget evidence.

Root reviewed candidate8094f25 and all prerequisite/source hashes after the
actual CPU pass. Readiness and conditional deployment/start authority are frozen
in root_review.json. Start requires exact deployed hashes, actual deployed CPU
tests with zero skips and fresh exclusive GPU1/RAM/CPU checks; approval alone
does not indicate deployment or training. Actual MMRec CPU preparation tests ran only in a temporary sandbox,
with CUDA hidden/uninitialized and no data, recommendation metrics or recovery
deployment. Actual MMRec preparation passed50/50 tests with zero skips/errors/
failures,4threads,Torch1.12.1+cu113 in0.592412s. The first CPU attempt had one
error because frozen input validation preceded the SVD observer; that failure
is preserved in receipts/preparation_cpu_attempt1_failed.json. The corrected
instance entry records this input without a solve, and successful calls reuse
one capture. Full output and exact transferred source hashes are retained in
preparation_cpu_receipt.json. Completion facts were then added to this README
and SOURCE_LINEAGE; tested execution files remain byte-identical. The deployed
preparation must also pass its own actual MMRec tests before any queue launch.
