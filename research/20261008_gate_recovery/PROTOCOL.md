# Instrumented recovery of decoupled counterfactual gate supervision

Recovery declared on 2026-10-08 after the original candidate stopped with a
nonfinite loss following epoch 38. The bounded checkpoint probe has not yet
established its root cause. This is one technical recovery attempt, with the
same seed, scientific hypothesis and metric gates; it is not another candidate.
No numerical solver, gradient, learning rate, data or evaluation change is
permitted in this recovery. train.py, proposal.py, isolation.py, prepare.py,
evidence.py and the carrier remain byte-identical to the failed bundle.
guarded_train.py observes every optimizer update and saves the first failed
batch, model/optimizer state, RNG states and SVD inputs/output gradients before
invalid gradients can reach Adam. Healthy updates must equal the unguarded
updates in tests. Nonfinite arithmetic stops the queue; no sanitization,
skipped steps or automatic retries. Alongside the selected checkpoint, save
its RNG and dataloader order for future diagnostic replay.

Run only after the failed queue and bounded diagnostic have exited. Freshly
initialize both candidate arms; do not use diagnostic-trained weights or
interrupted candidate checkpoints for formal comparisons. Existing original
BRIDGE baselines may be reused after unchanged algorithm/input checks. A
future solver change requires a new protocol with freshly trained controls.

Declared 2026-10-08 before running this candidate. This is one new, bounded
hypothesis, not a parameter search to rescue uniform target isolation.

## Trigger and falsifiable hypothesis

The completed uniform soft-isolation and baseline runs show:

| Dataset | Baseline R@20 | Soft isolation R@20 | Baseline N@20 | Soft isolation N@20 |
| --- | ---: | ---: | ---: | ---: |
| Baby | 0.10546504793740588 | 0.1062309444219961 | 0.04910749907453286 | 0.04946499163588207 |
| Sports | 0.1195296505011454 | 0.11877571907205982 | 0.05647686319160367 | 0.055922382090289116 |

Sports fails the existing primary gate regardless of the unfinished history
control. Preserve that control and its final screening record. Do not change
the old bundle or its decision rule.

The training-only diagnostic establishes self-support in behavior evidence,
not its causal role in the performance regression. Uniform isolation changes
both encoder ranking gradients and calibration-gate gradients. Hypothesis:
isolation is useful for gate supervision, while applying it to encoder ranking
is an unnecessary disturbance. A separate counterfactual gate objective tests
this claim; the present results do not establish it.

## Single algorithm change

Keep the frozen training-only fixed-KNN-support evidence definitions. For
each supervised training pair, construct either edge-deleted or history-only
evidence and interpolate at the already fixed strength alpha=0.5:

`z_cf = z_original + 0.5 * (z_isolated - z_original)`.

Let `b_theta` be the encoder score, `g_phi` the original BRIDGE gate, `w=0.4`
the original behavior weight, `m` the original detached candidate mask, and
`sg` stop-gradient. Two score paths share a single encoder forward:

`s_encoder = b_theta + sg(w * m * g_phi(b_theta, z_original) * z_original)`

`s_gate = sg(b_theta) + w * m * g_phi(sg(b_theta), z_cf) * z_cf`.

The main ranking gradients are from `BPR(s_encoder) + BPR(s_gate)`. The
implementation subtracts `sg(BPR(s_gate))` to keep the reported scalar's
forward value equal to the original main BPR value. The reported scalar is
therefore a surrogate; gate-objective values are logged separately. This is
gradient separation, not a changed weighting coefficient or gradient
projection. Auxiliary encoder losses and original gate regularization remain
exactly as supplied by BRIDGE.

At the SAME fixed parameter state, the encoder gradient equals original
BRIDGE; gate-ranking gradients use the counterfactual path only. Later encoder
updates can differ indirectly because gate parameters evolve differently.
Returning to original evidence must recover original loss and all gradients.
Inference calls the unchanged BRIDGE prediction and scoring functions. No
new learned parameters, features, edges, random draws or negative samples.

## Arms and gates

| Arm | Encoder ranking | Gate-ranking evidence | Validation/inference |
| --- | --- | --- | --- |
| baseline | Original BRIDGE | Original | Original BRIDGE |
| history | Original at fixed parameters | Soft history-only, alpha=0.5 | Original BRIDGE |
| edge | Original at fixed parameters | Soft fixed-support edge deletion, alpha=0.5 | Original BRIDGE |

History strength now matches the candidate strength, addressing the unequal
strength control in the earlier pilot. The old all-parameter soft-isolation
result is reported as an external reference, not silently reused as this arm.

Preserve the existing numerical gates: both validation R@20 and N@20 improve
at least 0.5% relative to baseline; absolute R@10 and N@10 decreases must not
exceed 0.0002; both R@20 and N@20 strictly exceed the matched history control.
Baby gates progression to Sports. Failure closes this candidate without
scanning strength, learning rate, loss weight, seed or thresholds.

## Evaluation and provenance

Use the 2026-09-10 default BRIDGE configuration and existing image/text
features. All evaluator, split, preparation and evidence code is copied
byte-for-byte from the 2026-10-08 recovery. Only training and validation
loaders are constructed. Test labels never enter selection, negative filtering
or diagnostics. Full-sort validation masks training positives.

Fixed seed 2026, fresh initialization for every new arm, maximum 300 epochs,
patience 20, batch 2048, evaluation batch 64, four-decimal validation R@20
strict-improvement checkpoint selection, earlier checkpoint on ties. Final
best-checkpoint validation is recomputed. Input, checkpoint and per-user
artifact hashes must pass. Non-finite values stop as technical failure.

Reuse only completed fresh BRIDGE baselines from
`/data/sunzixuan/target_isolated_20261008_autoresearch_v1` after checking the
effective configuration, data/code hashes, source and destination carrier
hashes, artifacts, seed, full budget, and validation-only protocol. Reuse its
completed training-only evidence caches after matching frozen preparation and
evidence code and all input/cache hashes. Never reuse interrupted candidates.
Wait for that pipeline's matching live process to finish before starting any
new training. An unrelated process with a recycled PID is not the old pipeline.

Unit tests must establish unchanged encoder gradients, counterfactual-only
gate-ranking gradients, original-evidence parity, unchanged inference and
state schema, masked-pair behavior, and restored context after exceptions.
One-epoch integration smokes run before formal arms. Log original and
counterfactual gate-objective values, active candidate counts and gate-credit
differences as diagnostics. They are not performance success criteria.

The decision includes 2000 paired user-bootstrap draws on validation. These
intervals are exploratory because selection and reporting share validation.
Any pilot pass still needs fixed seeds 2026-2030, a third dataset, ID-only and
modality-attribution controls. No publication-level or causal claim yet.

## Bounds and outputs

At most four new formal tasks and four one-epoch smokes. One training task
at a time, requiring 14 GiB free GPU memory with per-process fraction 0.55.
Each formal task has an eight-hour timeout; total queue including dependency
and resource waits is bounded to 48 hours. Timeout terminates this queue's
own child process group. No other job is terminated. Resource logs include
elapsed time and sampled process GPU memory; reused baseline memory is NA.

The immutable server bundle is
`/data/sunzixuan/decoupled_gate_20261008_recovery_v1`. Record scientific keep/discard/crash
in `results.tsv` and detailed gates in `research_report.json`. A candidate may
be kept as a pilot only when both dataset gates pass; it never replaces the
workspace's incumbent baseline solely on this single-seed screen.

## Literature positioning

- [BRIDGE](https://arxiv.org/html/2605.22073) supplies the existing encoder,
  behavior evidence and candidate gate. Those are not new contributions.
- [SpotTarget](https://arxiv.org/abs/2306.00899) studies target-link inclusion
  and target removal; [FakeEdge](https://proceedings.mlr.press/v198/dong22a.html)
  studies the training/inference topology gap. Target removal is not new.
- [KGD](https://arxiv.org/abs/2608.02738) separates parameter ownership for
  streaming transfer; decoupling itself is not new either.

The narrower candidate is analytic target-excluded behavioral supervision
assigned specifically to an existing multimodal candidate calibrator, while
preserving encoder gradients at fixed parameters. Whether that combination
is novel enough and useful requires empirical and broader prior-art evidence.
This round tests the mechanism; it does not declare a proven innovation.
