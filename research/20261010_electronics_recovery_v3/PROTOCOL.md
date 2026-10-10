# Electronics numerical baseline recovery v3

Prepared before any v3 deployment, training or recommendation metric. This is
a separately registered bounded baseline feasibility run, not an innovation.
It uses the Karpathy autoresearch workflow with immutable inputs, one numerical
change, fixed budgets, reproducible receipts and honest failure states. Root
review of candidate8094f25 and conditional deployment/start authorization are
recorded in root_review.json. Approval is not a launch or training result.

## Numerical change and evidence

Preserve the frozen original SVD solver, float32 input, three frequency bands,
and U_band @ diag(s_band) @ Vh_band forward arithmetic. Use the frozen
projector.py e44eb739482f3fe0448e1c9fdb066e36c4d6ea22d05a0c25a32734f29a856ab6:
the band-projector first VJP with float64 accumulation and explicit cross-band
domain checks. The backward implementation and its precision change. No jitter,
solver override, clipping, denominator regularization, detached basis, skipped
update, resume, retry or fallback is allowed. Higher derivatives are unsupported.
An unresolved cross-band boundary stops the job before its optimizer update.
This does not claim identical v1 gradient arithmetic, trajectory or convergence.

Seven mandatory, independently hashed receipt roles comprise the conjunction:

1. Original 13/13 small mathematical CPU tests, with no skipped tests.
2. Original 12-input CPU archive audit: all 36 bands exact, finite candidate
   gradients and every available finite generic-gradient comparison passing.
3. Complete old GPU receipt a840a3a1...: status must remain completed_fail.
   Its archived 12-input phase passes; the separate full-model phase failed.
   Capture0's original generic gradient is nonfinite, and cannot establish
   gradient equivalence. The historical crash was not exactly reconstructed.
4. Original-original repeat raw receipt 651f5d066...: completed_different,
   27/36 bands nonexact and 13 exceeding the inherited bounds. It demonstrates
   independent-forward upstream variation; it does not rescue the old gate.
5. Independent same-input raw GPU receipt 87251d388816ec3d395f477a096a7f67e1ea6e25cfaf84116b517c6a36c64f2c.
   One actual candidate full forward and one actual loss backward passed with
   all 12 realized inputs, all 36 bands torch.equal to detached original
   shadows on the same realized X, all 36 output hooks and all 12 input hooks
   reached once with finite gradients, and all 27 canonical participating
   parameter gradients finite. Exactly 24 solver calls (one original shadow
   and one candidate per input) succeeded with unchanged solver arguments.
   Method/parameter/state/alias/RNG/backend/interaction/input/version identities,
   old evidence and configuration remain guarded. There were 129 hash checks
   over 112 distinct protected paths; no optimizer, test or training occurred.
6. Actual same-input CPU test raw receipt 691fb0304b200c488186c3c63525ecf210da931a8a713989c3b91fc8ee13b57b:
   exit_code0, 31/31 individual passes and zero skips, CUDA hidden, predecessor
   anchors matching and configuration SHA9d217792556b38d685d79cad426a521d78bedfc4e28529e0040ed1774a5df84e.
   Both same-input roles bind all 16 deployed source-file hashes and reviewed
   source revision e99e57b42f52f9e6494dd527e0c2eab6ce6d4fc6.
7. Complete Electronics cache negative receipt 3df214b9...: non_equivalent,
   score error0.00026226 and 3/6 masked-Top50 differences. The cache is excluded.

prerequisites.json contains exact original paths, raw/config SHA256 and payload
SHA256 plus byte-identical local copies. Semantic exports are labeled separately
and never substituted for raw anchors. Local checks use exact archived bytes;
authorized remote checks re-read the original remote evidence and sources.
All seven roles must pass their own role-specific checks. Setting a status flag,
dropping failed rows, approximate band equality, partial12/36/27 coverage,
missing gradient reach, source/hash/RNG/identity drift or reported SVD fallback
cannot satisfy the conjunction. The new one-batch pass does not overturn the
old completed_fail or make the separately frozen v2 eligible.

## Training and observation

Fresh seeds2026,2027,2028,2029,2030 run in that fixed order without selecting
seeds. The complete original 35 carrier source/configuration files, train.py
and numerics.py remain byte-identical. Preserve the original three Electronics
assets, original v1 graph SHA76ccb18835b2d18b2f56e882cf4670b2d7af733951090656ab4be23ef70a9ed2,
dependency versions, sparse backend, parameter objects/state keys, Adam, LR
schedule, batch2048, validation batch64, maximum300 epochs and patience20.
Selection uses strict rounded-four-decimal validation Recall20, keeping the
earlier checkpoint on ties. Trainer.evaluate and final_validation remain the
original uncached paths. Never construct or evaluate a test loader.

The separate RecoveryTrainingGuard observes the one actual original solver
call, including its custom-forward no-grad context. Before checking/solving,
it clones the actual training input; singular=None means none were obtained.
It saves returned singular values and actual dL/dX if backward reaches X.
Generic dL/dU,S,Vh are explicitly unavailable inside the custom forward, never
fabricated. No observer adds a second solve. It blocks nonfinite intermediate
or parameter gradients before Adam and checks parameter/Adam-state finiteness
after an update; only then increments finite_steps_completed. Captures/hooks
release at each next loss, successful step and close. These diagnostic clones
consume additional memory and time but do not alter tensors returned to training.

Solver RuntimeError, nonfinite input, unresolved domain, reported SVD
nonconvergence/fallback or backward RuntimeError stops without retry and saves
the actual packet when execution permits. Ordinary warnings retain existing
filter/reporting behavior. The observer detects reported warnings only; it does
not prove the absence of silent internal library fallback. Unknown runtime
phases do not claim a pre-update block. Timeouts, SIGTERM/SIGKILL or fatal OOM
may prevent complete packet serialization: retain the queue's incomplete state
and available logs/checkpoints without claiming a complete replay packet.

Save configuration, RNG/loader order, original model/optimizer/scheduler
checkpoints, finite-step counts, progress, resource use, per-user final validation
arrays and every failure. Verify original source/input/graph/dependency hashes
before and after each seed. A later comparison method must use the same recovery
fix and matched epoch/selection/budget rules. Comparing to crashed v1 as if it
were a complete baseline is prohibited.

## Bounded execution and authorization

Only physical GPU1, one own process at a time; CUDA memory fraction0.55,
4CPU threads, at least14GiB free GPU RAM and12GiB available host RAM. Refresh
GPU1 process/resource identity after large hashes immediately before spawning;
the child checks again before Torch/CUDA initialization. GPU0 and unrelated
tasks are untouched. Queue and launch locks/reservations are single use.

Each seed has at most24h; the queue has72h total including waits. The job limit
explicitly changes v1's6h wall-time budget; epoch/LR/patience/selection remain
unchanged. The old combined mean1506.185479s implies minimum22 completed epochs
before patience stopping take9.20h, and24h permits about57 epochs. Five maximum
jobs require120h, so72h does not guarantee all five completed seeds. Stop on
numerical/source failure or budget exhaustion, not an unfavorable validation
score. Preserve not_started, crash and incomplete_timeout states; incomplete
or favorable subsets are not a completed five-seed baseline.

Remote root: /data/sunzixuan/electronics_baseline_20261010_recovery_v3.
The committed candidate must include all gate/manager/runner/guard sources,
all prerequisite manifests and exact raw receipts, source lineage and actual
CPU preparation receipt. Root must approve and freeze the complete SHA256 payload
before deployment. A single frozen approval may conditionally authorize training
only after the deployed payload matches every SHA, its actual MMRec CPU tests
pass without skips, the original job is inactive, GPU1/RAM/CPU checks pass and
single-use locks are available. Root then executes start separately; no approval
record is mutated after deployment, because its bytes are protected too. Current
root_review.json records conditional approval and ready_for_training=true.
At approval freeze no v3 deployment or training had occurred. This supports a bounded numerical feasibility attempt
only; no recommendation gain, novel mechanism or publication readiness has been
established. Publication claims follow result/PUBLICATION_PROTOCOL.md.
