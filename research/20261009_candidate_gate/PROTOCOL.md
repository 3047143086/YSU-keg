# Candidate-negative supervision assigned to the BRIDGE calibrator

Declared 2026-10-09 before running. One new hypothesis after the completed
decoupled counterfactual gate pilot failed Baby. Do not relax or rescue its
gate. Preserve old results and frozen failed families.

## Trigger and scope

The old candidate improved over baseline but underperformed its matched
history control. Target-isolation supervision is therefore unsupported.
Training diagnostics separately show random negative items enter raw base
Top200 only about 2-3% of the time. The calibrator is applied to raw Top200 at
inference; its negative training scope is consequently sparsely covered.
This is a coverage observation, not evidence that coverage caused a regression.

Hypothesis: assigning ranking against actual candidate negatives to the
existing calibrator helps top-K ordering, while changing encoder ranking to
the same hard-negative objective is an unnecessary disturbance.

F16 CBCR reweighted the existing sampled pair and did not construct candidate
negatives. F17 C2R changed modality routing and inference. This trial changes
the gate's negative comparison set and includes a conventional joint-update
control. It neither reweights their losses nor repeats counterfactual evidence.
No new preference information is claimed. False negatives remain a risk.

## Single intervention and matched control

Use the exact detached raw base Top200 already used by BRIDGE for candidate
masks. Then remove ALL user training-positive items from that set to define
candidate negatives. Never mask history before Top200 or refill the filtered
set, since inference applies correction first and evaluator history-mask later.
Training-history membership uses the provided TRAIN split only. Validation and
test labels never affect the negative set. No new RNG draws or learned weights.

An example is eligible only if its supervised positive is in raw Top200 and
at least one candidate negative remains. For eligible examples, define the
candidate objective as mean BPR against every valid candidate negative. Average
each example over its own valid negatives, then average ALL batch examples
equally. Ineligible examples fall back to their original sampled-pair objective
with the original scope masks. No rows, steps or gradients are skipped.

| Arm | Main encoder ranking gradient | Main gate ranking gradient |
| --- | --- | --- |
| baseline | Original BRIDGE sampled BPR | Original sampled BPR |
| candidate | Original at the same fixed parameters | Candidate-set BPR; original fallback |
| joint | Candidate-set BPR; original fallback | Same candidate-set BPR; original fallback |

Candidate arm separates ownership with stop-gradient on original correction
in the encoder path and on encoder base scores in the gate path. Its main
surrogate keeps the original sampled-BPR forward scalar; actual candidate
objective is logged separately. Joint uses the SAME comparison and no new
coefficient, but allows both encoder/gate gradients to change. It also logs
its actual objective. Auxiliary losses and original sampled gate regularizer
remain unchanged for both arms. At fixed parameters candidate encoder gradients
must equal baseline, and gate gradients must equal an independent candidate
formula. Later encoder trajectories can differ as gate parameters evolve.

Original-evidence behavior scores, weight0.4, Top200, candidate masks and
inference are unchanged. No target deletion, history-only evidence, adapters,
new topology, modality routes, confidence weights or gradient projection.
Only the currently verified dense behavior backend is supported; no silently
different sparse normalization. original mode must recover all baseline
gradients, RNG state, parameter schema and prediction dispatch.

## Mechanism, scientific gates and budget

Before formal arms, each dataset has one-epoch candidate/joint integration
smokes. Candidate smoke training-only diagnostics must show random-negative
scope rate <=0.10, eligible supervised pair rate >=0.02, and mean remaining
candidate negatives among eligible pairs >=20. These criteria are fixed now;
no validation metric is used for this mechanism gate. Failure stops without
scanning K, weights, hardness, seeds or thresholds. Smoke validation serves
only finite-score and runner integration checking, not model selection.

Preserve the existing scientific thresholds: candidate R@20 and N@20 must
both improve at least 0.5% relative to baseline, R@10/N@10 absolute decreases
must not exceed0.0002, and both primary metrics must strictly exceed joint.
Baby gates progression to Sports. Failed Baby closes this candidate. Joint
alone improving would support conventional candidate training, not the
gate-only ownership hypothesis. Missing/crashed arms are never accepted.

Each fresh formal arm: seed2026, at most300epochs, patience20 (stop after
21 unimproved rounds), training batch2048, evaluation batch64, four CPU
threads. Selection is strict improvement in rounded4-decimal validation R@20,
earlier checkpoint on ties. Recompute full-sort best-checkpoint validation.
No test loader/evaluation or held-out negative filtering. Protected train.py,
carrier, supplied split, graph and existing image/text features are unchanged.
Original baselines may be reused from the completed October8 original-BRIDGE
controls only after configuration, budget, code/data/input/output hashes match.
Do not reuse interrupted models or preceding candidate/history controls.

At most four formal arms and four one-epoch smokes, one training task at a
time. Require14GiB free GPU, fraction0.55, per-formal timeout8hours, queue
including resource wait48hours. Wait for the preceding queue to exit. Do not
stop others' jobs. Existing finite-gradient/optimizer guards save first-failure
packets and checkpoint RNG/dataloader state, and never sanitize or skip.
Technical failure stops with no automatic numerical retry or solver change.

Use results.tsv keep/discard/crash and paired validation-user bootstrap2000
draws. Intervals remain exploratory because selection shares validation.
Any two-dataset pilot pass still requires seeds2026-2030, Electronics,
ID-only/modality attribution and efficiency evidence. This bounded pilot
cannot establish publication-level innovation or causal improvement.

## Provenance and prior art

Immutable remote bundle: `/data/sunzixuan/candidate_gate_20261009_v1`.
Local source branch: `autoresearch/oct9-candidate-gate`.
Preserve `/data/sunzixuan/decoupled_gate_20261008_recovery_v1`, whose completed
Baby edge/history comparison rejected target-isolation gate supervision.

Hard-negative sampling and its relationship to top-K objectives are established:
[Shi et al., WWW2023](https://arxiv.org/abs/2302.03472). False negatives and
BPR robustness are established concerns:
[Hard-BPR](https://arxiv.org/abs/2403.19276). Their methods are not new claims
here and Hard-BPR is not introduced as another simultaneous intervention.
The narrower unproven mechanism is candidate-set supervision assigned to an
existing multimodal calibrator with the encoder objective held at baseline,
tested against conventional joint parameter ownership. Further prior-art
review is necessary even if the pilot passes.
