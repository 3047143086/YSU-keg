# Failed Innovation Experiment Ledger

Last updated: 2026-10-10

This file is the memory guard for subsequent innovation work. An entry marked
`FREEZE` must not be repeated under a new name unless a new hypothesis directly
addresses the documented failure mechanism. Results are validation-selected;
formula-level reconstructions are explicitly distinguished from official code.

## Decision rule

- Screen on validation Recall@20 and NDCG@20; never select on test.
- A one-seed or one-dataset gain is not publishable evidence.
- Continue only when both primary metrics improve without a material regression
  in Recall@10 or NDCG@10.
- After a screening failure, do not launch a parameter grid to rescue it.
- Full evidence requires at least Baby, Sports, and Electronics, five fixed
  seeds, mean/std, paired significance, efficiency, and activity-stratified
  results.

## Frozen reference

Official MSCA, Baby, seed 2026:

| Split | R@10 | R@20 | N@10 | N@20 |
|---|---:|---:|---:|---:|
| Validation | 0.0675 | 0.1043 | 0.0369 | 0.0462 |
| Test | 0.0689 | 0.1054 | 0.0375 | 0.0469 |

The test row is a historical baseline measurement. New mechanisms remain
validation-only until frozen.

## Confirmed failures

### F01 - CCSR user-side simplex routing (`FREEZE`)

- Carrier/data: BRIDGE, Sports, seed 2026.
- Control validation R@20/N@20: `0.1195/0.0565`.
- CCSR validation R@20/N@20: `0.1179/0.0553`.
- Control test R@20/N@20: `0.1203/0.0570`.
- CCSR test R@20/N@20: `0.1196/0.0565`.
- Finding: enforcing positive normalized modality mass and calibrating its
  temperature did not improve ranking. The original unconstrained route was
  not the limiting factor.
- Do not repeat: simplex/softmax reparameterization of existing modality gates
  without adding a new preference signal.

### F02 - DyPER + HARM dynamic pseudo-edge refinement (`FREEZE`)

| Dataset | Variant | Test R@20 | Test N@20 |
|---|---|---:|---:|
| Baby | SiMGR control | 0.0956 | 0.0448 |
| Baby | DyPER + HARM | 0.0981 | 0.0456 |
| Sports | SiMGR control | 0.1136 | 0.0519 |
| Sports | DyPER + HARM | 0.1129 | 0.0516 |

- Finding: the Baby improvement did not transfer to Sports. Dynamic edges can
  amplify model errors and dataset-specific neighborhood bias.
- Do not claim: the old paper drafts describing consistent three-dataset gains
  are superseded by these controlled results.
- Do not repeat: periodic pseudo-positive or model-mined pseudo-negative graph
  replacement as the main contribution.

### F03 - RGCE reliability-gated consensus edges (`FREEZE`)

| Dataset | Variant | Test R@20 | Test N@20 |
|---|---|---:|---:|
| Baby | SiMGR control | 0.0956 | 0.0448 |
| Baby | RGCE | 0.0801 | 0.0374 |
| Sports | SiMGR control | 0.1136 | 0.0519 |
| Sports | RGCE | 0.1099 | 0.0497 |

- Finding: end-to-end graph gates still wrote prediction errors back into
  topology. Baby dropped roughly 16%; Sports also regressed and encountered a
  late NaN after its best checkpoint.
- Do not repeat: learned gates on pseudo-edge adjacency, even when bounded or
  initialized to preserve the base graph.

### F04 - CF-Beta-LCB virtual interaction family (`FREEZE`)

- Carrier/data: MSCA, Baby, seed 2026.
- Control validation R@20: `0.1043`.
- Best first-order Beta-LCB response (`beta=8`) validation R@20: `0.1041`.
- Other tested forms included additive augmentation, collaborative/semantic
  propagation, degree-preserving swaps, cycle transport, net-utility/profile-
  drift transport, and conservative response scales. None passed the frozen
  validation gate.
- Finding: candidate precision was too low for topology-level intervention;
  mass conservation and uncertainty bounds reduced damage but did not create a
  useful new signal.
- Do not repeat: any method whose core action is inserting, replacing, or
  transporting inferred user-item edges from the same fixed modality kNN
  graphs.

### F05 - VI-MMRec formula reconstruction on MSCA (`FREEZE`)

- These are formula-level reconstructions, not official VI-MMRec results.
- Baby/seed-2026 validation R@20: VI-MMRec-O `0.0915`, VI-MMRec-S `0.0967`,
  versus MSCA `0.1043`.
- Finding: wholesale virtual-interaction replacement is incompatible with this
  carrier/split and substantially weaker than the official MSCA path.
- Do not repeat: tuning the reconstruction and presenting it as an official
  baseline. Obtain official code for paper-level comparison.

### F06 - Reliability-conditioned alignment/routing (`FREEZE`)

- Tested interventions: CL sample weighting, fusion-view scaling, attention
  priors, their combination, and shared/private consensus residual alignment.
- All Baby/seed-2026 screens failed to exceed the official MSCA validation
  gate; exact logs are retained under `experiments213/adaptive_*` and
  `experiments213/consensus_*` on server 213.
- Finding: reweighting existing MSCA semantic views changes emphasis but does
  not add independent preference evidence.
- Do not repeat: another confidence function applied only as a weight on the
  same CL or fusion terms.

### F07 - SDA structure-aware modality adapters (`FREEZE`)

| Variant | Valid R@10 | Valid R@20 | Valid N@10 | Valid N@20 |
|---|---:|---:|---:|---:|
| MSCA | 0.0675 | 0.1043 | 0.0369 | 0.0462 |
| SDA-Struct | 0.0671 | 0.1037 | 0.0365 | 0.0458 |
| SDA-Gate | 0.0679 | 0.1038 | 0.0368 | 0.0458 |

- Finding: low-rank feature adaptation and agreement-gated shared content did
  not improve the primary metrics. SDA-Gate moved only R@10 upward while both
  R@20 and N@20 fell.
- Do not repeat: adding feature adapters or cross-modal alignment losses to the
  same pretrained VGG/text features without a new user-conditioned signal.

### F08 - LCHR leave-one-out calibrated history residual (`FREEZE`)

| Variant | Valid R@10 | Valid R@20 | Valid N@10 | Valid N@20 |
|---|---:|---:|---:|---:|
| MSCA | 0.0675 | 0.1043 | 0.0369 | 0.0462 |
| LCHR | 0.0678 | 0.1038 | 0.0369 | 0.0461 |

- Best LCHR checkpoint: epoch 45 on Baby/seed 2026.
- Finding: explicit leave-one-out history matching raised only R@10. Both
  primary R@20 and N@20 missed the frozen gate, and larger learned residual
  scales later in training coincided with worse validation metrics.
- Do not repeat: rescaling or temperature-searching this same score residual;
  the evidence indicates signal mismatch rather than insufficient capacity.

### F09 - UECA user-equalized collision-free alignment (`FREEZE`)

| Variant | Valid R@10 | Valid R@20 | Valid N@10 | Valid N@20 |
|---|---:|---:|---:|---:|
| MSCA | 0.0675 | 0.1043 | 0.0369 | 0.0462 |
| UECA | 0.0669 | 0.1035 | 0.0366 | 0.0459 |

- Best UECA checkpoint: epoch 45 on Baby/seed 2026; training stopped at epoch
  66.
- Finding: removing repeated contrastive entities and exactly correcting user
  minibatch-inclusion probability reduced all four ranking metrics. Sampling
  imbalance exists, but correcting it is not aligned with top-k utility.
- Do not repeat: deduplication, inverse-frequency, inverse-inclusion, or other
  user-equalizing reweighting of the same MSCA contrastive anchors.

### F10 - RAPA ranking-anchored projected alignment (`FREEZE`)

- Diagnostic: official MSCA, Baby, seed 2026, 12 epochs, 36 probes.
- All six alignment terms had zero conflicts at the preregistered cosine
  threshold `< -0.05`; the summed alignment gradient also had zero conflicts.
- Summed alignment median/minimum cosine: `0.0147/-0.0074`; median weighted
  norm ratio versus BPR: `9.5240`; negative-work fraction: `0.0283`.
- The largest term-level negative-work fraction was item-structure (`0.3522`),
  but its minimum cosine was only `-0.0180`, so it never met the material
  conflict definition.
- Finding: auxiliary gradients can dominate BPR in magnitude, but the repeated,
  material opposition required by projection is absent. The preregistered gate
  failed, so implementing PCGrad/RAPA would be post-hoc mechanism search.
- Do not repeat: gradient projection or gradient-surgery variants justified by
  alleged BPR/alignment conflict on this carrier. Full results are recorded in
  `../PRAVI/RAPA_EXPERIMENT.md`.

### F11 - Rating-residual multimodal anchor (`FREEZE`)

- Diagnostic: Baby, seed 2026, 20,559 validation events and 16 deterministic
  unobserved comparisons per event.
- Polarity coverage passed (`0.8751`), but high-rating polarity AUC was only
  `0.5086` versus `0.5036` after ratings were shuffled within each user.
- Exposure-only fused AUC was `0.5992`; the preregistered fixed combination fell
  to `0.5963`, a paired delta of `-0.00284` with 95% CI
  `[-0.00416, -0.00152]`.
- Finding: user-centered star-rating residuals weakly identify disliked items
  but do not transfer enough positive semantic preference to the standard
  implicit-ranking target. Four of five preregistered checks failed.
- Do not repeat: explicit-rating polarity residuals, rating-scale tuning, or
  rating-conditioned semantic gates on this implicit benchmark.
- Full protocol and results: `../PRAVI/RATING_SIGNAL_EXPERIMENT.md`.

### F12 - C3E degree-conditioned collaborative evidence calibration (`FREEZE`)

| Signal | Validation pairwise AUC |
|---|---:|
| Inverse-degree exposure profile | 0.5995 |
| Raw candidate Top-3 cosine | 0.6225 |
| Global cross-fitted calibration | **0.6264** |
| Degree-conditioned C3E | 0.6121 |

- Carrier/data: frozen image/text features, Baby, seed 2026; 20,559 validation
  events and 16 deterministic unobserved comparisons per event.
- C3E versus exposure: `+0.01262`, paired 95% CI
  `[0.00959, 0.01565]`; this missed the preregistered `+0.0200` gate.
- C3E versus raw candidate Top-3: `-0.01039`, paired 95% CI
  `[-0.01231, -0.00847]`.
- Both cross-fitting folds improved over exposure, but neither rescued the
  central mechanism. Popularity-cell conditioning damaged the useful global
  semantic ordering instead of correcting hubness.
- Do not repeat: degree-bin pairs, degree-conditioned calibration tables, or a
  shrinkage/bin/Top-q search on these same fixed features.
- Observation only: global cross-fitted calibration achieved the highest AUC
  (`0.6264`), but its `+0.00386` aggregate gain over raw Top-3 was below the
  frozen `+0.0050` requirement and lacks a stored paired CI. It is not a pass.
- Full protocol and results: `../PRAVI/C3E_EXPERIMENT.md`; raw JSON is retained
  as `experiments213/c3e_baby_s2026.json` on server 213.

### F13 - SG-URInit semantic user-profile adapter (`FREEZE`)

- Carrier/data: official MSCA, Baby, seed 2026; fixed SG-URInit settings
  `K=4`, cluster weight `0.1`, semantic-BPR weight `0.001`.
- Validation result: R@10 `0.0668`, R@20 `0.1031`, N@10 `0.0365`, N@20
  `0.0457`; official MSCA is `0.0675/0.1043/0.0369/0.0462`.
- Finding: inverse-degree local plus cluster-level semantic user profiles did
  not clear either primary validation metric. The adapter is a useful recent
  baseline reference, but it is not a supported improvement on this carrier.
- Do not repeat: another direct mean/profile initialization or semantic-BPR
  auxiliary term over the same fixed image/text features without an explicit
  multi-interest or counterfactual signal.
- Full protocol and log: `../PRAVI/MSCA_SG_EXPERIMENT.md` and
  `experiments213/msca_sg_baby_s2026_gpu1.stdout` on server 213.

### F14 - DCRL disagreement-conditioned conjunctive ranking loss (`FREEZE`)

- Carrier/data: official MSCA, Baby, seed 2026; fixed DCRL settings weight
  `0.02`, margin `0.10`, temperature `0.20`, minimum conflict weight `0.10`.
- Validation result: R@10 `0.0667`, R@20 `0.1031`, N@10 `0.0363`, N@20
  `0.0455`; official MSCA is `0.0675/0.1043/0.0369/0.0462`.
- Finding: requiring the positive to win in both modality views, with
  disagreement-weighted softplus penalties, did not improve the fused ranker.
  It missed both primary gates and regressed every tracked metric slightly.
- Do not repeat: DCRL or a weight/margin/temperature sweep of the same
  disagreement-conditioned pairwise loss on the fixed MSCA views.
- Full protocol and log: `../PRAVI/DCRL_EXPERIMENT.md` and
  `experiments213/msca_dcrl_baby_s2026_gpu1.stdout` on server 213.

### F15 - JCEF joint cross-fitted evidence fusion (`FREEZE`)

- Carrier/data: frozen image/text evidence diagnostic, Baby, seed 2026;
  two user-level folds and 16 deterministic negatives per validation event.
- JCEF pairwise AUC: `0.62784` (95% CI `[0.62355, 0.63212]`), below the
  preregistered `0.6314` threshold.
- Paired deltas: versus equal global calibration `+0.00147`
  (95% CI `[+0.00028, +0.00265]`), versus learned additive fusion `+0.00088`
  (95% CI `[+0.00034, +0.00143]`); both missed their required margins.
- Finding: agreement/conflict features contain a weak positive signal, but not
  enough independent evidence to justify a model residual. Fold deltas were
  positive yet small (`+0.00060`, `+0.00233`).
- Do not repeat: JCEF ridge/bin/Top-q/interaction-term sweeps or direct
  integration of this same calibrated evidence on MSCA.
- Full protocol and JSON: `../PRAVI/JCEF_EXPERIMENT.md` and
  `experiments213/jcef_baby_s2026.json` on server 213.

### F16 - BRIDGE-CBCR confidence-bounded candidate repair (`FREEZE`)

- Carrier/data: official BRIDGE, Baby, seed 2026, matched split and features.
- Validation R@20/N@20: CBCR `0.1056/0.0491`, matched BRIDGE
  `0.1056/0.0492`; R@10 was `0.0704` versus `0.0709`.
- Finding: emphasizing uncertain or incorrectly ranked top-K pairs did not
  improve Recall and slightly reduced NDCG. The correction-budget loss added
  no useful independent preference evidence.
- Do not repeat: CBCR loss-weight, temperature, or guard-weight searches on
  the same BRIDGE correction branch.
- Logs: `/data/sunzixuan/bridge_official/experiments_cbcr/` on server 213.

### F17 - BRIDGE-C2R candidate-conditional reliability routing (`FREEZE`)

- Baby/seed-2026 screen: C2R validation R@20/N@20 `0.1066/0.0496`
  versus matched BRIDGE `0.1056/0.0492`.
- Independent Sports/seed-2026 confirmation: C2R validation
  R@10/R@20/N@10/N@20 `0.0827/0.1191/0.0472/0.0565` versus matched BRIDGE
  `0.0834/0.1198/0.0474/0.0567`.
- Sports test R@20/N@20 moved slightly upward (`+0.0005/+0.0003`), but test
  cannot override the preregistered validation failure.
- Finding: the small Baby gain did not transfer. Candidate-wise routing of the
  same ID/image/text score blocks changes emphasis but supplies no independent
  preference evidence.
- Do not repeat: C2R route-strength, temperature, balance-loss, or hidden-size
  searches. The failed cross-dataset confirmation closes this family.
- Logs: `/data/sunzixuan/bridge_official/experiments_c2r/` and
  `/data/sunzixuan/bridge_official/experiments_c2r_confirm/` on server 213.

### F18 - ECLI episodic cross-modal late interaction (`FREEZE`)

- Signal: equal fusion of each candidate's Top-3 image/text matches against the
  user's individual training items; unlike a history mean, this preserves
  multiple interests.
- Scope: parameter-free local reordering inside BRIDGE's top-200 candidates.
  No graph edges, encoder features, or baseline parameters are changed.
- Rounded Baby/seed-2026 screen: BRIDGE R@10/R@20/N@10/N@20
  `0.0709/0.1056/0.0404/0.0492`; ECLI
  `0.0710/0.1058/0.0405/0.0493`. The directional gate passed, but the gains
  (`+0.0001/+0.0002/+0.0001/+0.0001`) are too small to trust at four-decimal
  precision.
- Exact Baby deltas: R@10 `+0.0001029`, R@20 `+0.0002657`, N@10
  `+0.0000267`, N@20 `+0.0000616`. Direction is positive, but the paired 95%
  CIs for R@20 `[-0.0004243, +0.0009557]` and N@20
  `[-0.0001289, +0.0002521]` cross zero, so this is only a screen pass.
- Independent Sports/seed-2026 exact deltas were R@10 `-0.0002729`, R@20
  `+0.0000843`, N@10 `-0.0000409`, and N@20 `+0.0000496`. The R@10 regression
  exceeded the preregistered `0.0002` guard, and every paired 95% confidence
  interval crossed zero.
- Finding: candidate-wise Top-3 late interaction has a weak, directionally
  useful R@20/N@20 signal on both datasets, but its effect is too small and it
  trades away short-list recall. It cannot support a paper contribution.
- Do not repeat: Top-q, candidate-K, clipping, modality-weight, or residual-gap
  searches for this same candidate-to-history correction.
- Outputs: `/data/sunzixuan/bridge_official/experiments_ecli/`.
- Protocol: `../baselines/BRIDGE-official/ECLI_EXPERIMENT.md`.

### F19 - TACE temporal-affinity calibrated evidence (`FREEZE`)

- Signal: replace BRIDGE's equally weighted co-user ItemKNN relation with a
  training-only temporal affinity. Each user contributes
  `1 / max(distinct-timestamp-rank distance, 1)` to an item pair; the remaining
  candidate scope, cosine normalization, Top-K pruning, learned gate, and
  correction weight are inherited unchanged from the checkpoint.
- Baby/seed-2026 exact validation baseline R@10/R@20/N@10/N@20 was
  `0.0708605/0.1055593/0.0404249/0.0492366`; TACE reached
  `0.0698889/0.1055919/0.0386836/0.0477486`.
- Exact deltas were R@10 `-0.0009716`, R@20 `+0.0000326`, N@10
  `-0.0017413`, and N@20 `-0.0014880`. The paired 95% CI for N@20 was
  `[-0.0024030, -0.0005730]`, establishing a statistically clear ranking
  quality regression rather than an inconclusive tie.
- Finding: temporal proximity slightly shifts hits toward deeper ranks but
  damages top-rank ordering. The random full-sort benchmark split also does
  not support a next-item interpretation of timestamps.
- Do not repeat: temporal-rank kernels, decay-rate searches, temporal/uniform
  ItemKNN mixtures, or post-hoc correction-strength tuning on this split.
- Output: `/data/sunzixuan/bridge_official/experiments_tace/` on server 213.
- Protocol: `../baselines/BRIDGE-official/TACE_EXPERIMENT.md`.

### F20 - Uniform soft target-interaction isolation (`FREEZE`)

| Dataset | Arm | Validation R@20 | Validation N@20 |
| --- | --- | ---: | ---: |
| Baby | Fresh matched BRIDGE | 0.10546504793740588 | 0.04910749907453286 |
| Baby | Soft isolation, alpha=0.5 | 0.1062309444219961 | 0.04946499163588207 |
| Sports | Fresh matched BRIDGE | 0.1195296505011454 | 0.05647686319160367 |
| Sports | Soft isolation, alpha=0.5 | 0.11877571907205982 | 0.055922382090289116 |

- Fixed seed 2026; validation-only selection and no test evaluation. Both
  baseline and candidate runs in this table are complete. The Sports history
  control has now completed at R@20/N@20 0.1194870116691083/0.05673559134689602.
  The completed six-arm queue closed as `closed_validation_gate`; its process
  is no longer running.
- Baby passed the prespecified pilot gate with relative R@20/N@20 gains
  of 0.7262%/0.7280%. Its exploratory paired confidence intervals cross zero.
- Sports relative R@20/N@20 declines are approximately 0.6307%/0.9818%;
  R@10/N@10 absolute deltas are -0.00052976/-0.00048785, also beyond the
  -0.0002 guards. The cross-dataset hypothesis therefore fails.
- Training-only self-support diagnostics passed on both datasets, but their
  presence does not establish that removal improves recommendation. Uniform
  isolation changes both encoder and gate supervision. That coupled change
  leaves the regression mechanism unresolved.
- Do not repeat: an alpha, loss-weight, learning-rate, threshold or seed search
  for uniform isolation. Preserve the incumbent BRIDGE configuration.
- Source records: `../research/20261008_autoresearch/latest_status.json` and
  `/data/sunzixuan/target_isolated_20261008_autoresearch_v1`.

### F21 - Decoupled counterfactual gate supervision (`FREEZE`)

Decoupled counterfactual gate supervision tests a distinct attribution question:
apply the target-excluded training evidence only to gate-ranking gradients,
while preserving original encoder gradients at the same fixed parameter state.
The gate still indirectly affects later encoder updates. Its original
regularizer and the complete inference path remain unchanged.

Edge-deleted and history-only arms both use alpha=0.5, so the control matches
the candidate's strength. This addresses the unequal-strength history control
in the old pilot. It is not a claim that decoupling is novel or already useful.

The isolated candidate is committed as `6b2a927`; all 31 actual-environment
tests and both Baby integration smokes passed. Formal Baby edge training
stopped after completed epoch 38 (zero-based) with a non-finite training loss.
Queue PID 2540694 is no longer running; the outcome is `technical_failure`,
not a scientific rejection or performance pass. Neither new formal arm has a
completed result, and Sports was not started. Preserve this failure; no blind
numerical retry or parameter scan was launched.

A separate diagnostic initialized from the finite epoch-38 checkpoint tested
58 training batches, including SVD outputs/input gradients and parameter
gradients. It did not reproduce the failure. No RNG state was saved by the
original runner, so this is not an exact failed-batch replay and does not
establish a root cause or a numerical fix. Details:
`../research/20261008_gate_diagnostic/NUMERICS.md`.

An expanded isolated probe subsequently completed 40 additional epochs and
2320 training batches without failure; original checkpoint hashes were
unchanged. The original failure remains unconfirmed. A single fresh observed
recovery is declared in `research/20261008_gate_recovery/PROTOCOL.md` with the
same algorithm, seed, budget and scientific gates. Its gradient guards block
bad updates and save failed-batch replay packets; they are not a solver fix.
All 35 actual-environment checks passed. Preserve the original failure record.
Recovery PID 2567421 has exited; both Baby smokes and both formal arms completed.
The observation code did not establish the original NaN root cause.
The recovery later completed both Baby formal arms without a numerical failure:
edge R@20/N@20=0.10631188/0.04943678 and matched history
R@20/N@20=0.10661481/0.04960990. Edge improved over baseline by 0.80295% and
0.67054%, but fell below history on both primary metrics and missed the R@10
shortlist guard by 0.000000198. The Baby screen is `discard`; Sports was not
started by the declared gate. This is a scientific rejection, not a crash.

Protocol: `../research/20261008_decoupled_gate/PROTOCOL.md`.
Remote directory: `/data/sunzixuan/decoupled_gate_20261008_v1`.
Earlier frozen families remain closed.

### F22 - Candidate-negative supervision assigned only to the gate (`FREEZE`)

Candidate-negative supervision assigned only to the existing BRIDGE gate.
Use raw base Top200, then exclude all TRAIN-positive items without refilling.
Eligible rows compare their positive to the mean over remaining candidates;
other rows retain original sampled BPR. Encoder gradients remain original at
fixed parameters. The matched control uses the same candidate objective for
both encoder and gate. This changes the comparison set rather than retrying
counterfactual evidence, reweighting sampled pairs, or changing modality routes.
No new preference signal or established novelty is claimed.

The protocol was declared before running. All 43 actual-environment tests and
both Baby integration smokes passed. The full four-arm formal queue completed
as `closed_validation_gate`; queue PID 2772330 has exited. Isolated branch
commit f46078a. Training-only coverage
passed: random negative scope 0.0278361, eligible-pair rate 0.343076, mean valid
candidate negatives per eligible row 196.5646. This is mechanism feasibility,
not a performance pass. Training-only supervision coverage gates formal
training, and Baby validation gates Sports. Existing performance
thresholds are unchanged. No automatic parameter scan or numerical retry.
Baby candidate/joint formal arms completed 151/41 epochs under identical
patience rules. Candidate R@20/N@20 = 0.1063847359463199/0.04939945866349945;
joint = 0.09463964294897696/0.04476829403374747. Candidate gains over baseline
are 0.87203%/0.59453%; R@10 delta -0.000157342 passes the shortlist guard,
and both primaries beat joint. Baby screen is keep, permitting Sports. Both
primary paired bootstrap intervals versus baseline still cross zero.
Sports candidate completed 130 epochs, with zero-based best epoch 108 (not the
last epoch 129). Exact R@20/N@20/R@10/N@10 are
0.11915554453332632/0.05623051899353304/0.08245836035035253/0.04690931057948191.
Relative R@20/N@20 changes versus matched baseline are
-0.3129817%/-0.4361860%; R@10/N@10 absolute deltas
-0.0005534353/-0.0002771326 also fail the -0.0002 shortlist guards.
Sports joint completed 48 epochs, with zero-based best epoch 26, at
R@20/N@20=0.10867385118251109/0.05144774661129747. Both primary metrics beat
joint but not the original baseline, so Sports is `discard` and the overall
pilot is not a success. Preserve the candidate-gate family as `FREEZE`;
do not rescue it by scanning gate strength, negative scope, or training
parameters. No test evaluation occurred (`test_evaluated=false`).
Protocol: `../research/20261009_candidate_gate/PROTOCOL.md`.
Remote directory: `/data/sunzixuan/candidate_gate_20261009_v1`.

### F23 - Dual-witness collaborative evidence (`FREEZE`)

Dual-witness collaborative evidence is an unproven TRAIN-structure hypothesis,
not a new observed preference or established novelty. Split item columns by a
fixed label-free SHA256 rule into disjoint A/B views. The same co-user must
overlap both support-history views; its weight is the geometric mean of the
two view cosines, with self removed and fixed Top100. Compare against ordinary
ItemKNN, UserKNN, a minimum-overlap-two rule and degree-matched shuffled B.

Only a CPU mechanism screen is declared. For every TRAIN-degree>=4 user,
select one deterministic internal target and remove ALL selected target edges
globally before computing evidence. Compare each target against 16 fixed
unobserved samples excluding all that user's original TRAIN positives, with
AUC ties worth 0.5. Require positive coverage>=25%, AUC gain>=0.005 and a
positive bootstrap lower bound versus each simple control, gain>=0.005 versus
shuffled, and positive control deltas in both fixed user-hash halves. No
validation or test labels are used; no formal training is launched by this
screen. Baby must pass before the same Sports diagnostic is permitted.

The isolated directory is `../research/20261009_dual_witness`, branch
`autoresearch/oct9-dual-witness`, immutable commit `3b5b026`. The first remote
verification passed 18/19 tests; an ItemKNN indexing incompatibility with the
existing SciPy version was fixed, and all 19 tests then passed. The CPU
diagnostic queue launched as PID `2874627` at 2026-10-09 18:03:55 Beijing
time, with four worker threads, no GPU, and the declared host-memory gate
satisfied. The Baby diagnostic completed in 14.904509139014408 seconds over
13,447 users. Its original TRAIN graph has 118,551 edges; globally removing
all 13,447 internal targets leaves 105,104 support edges.

| Arm | Mean AUC | Candidate minus control | Paired user bootstrap 95% CI |
| --- | ---: | ---: | --- |
| ItemKNN | 0.6782762883914628 | -0.15308479586524876 | [-0.1576053208894177, -0.14869714202052503] |
| UserKNN | 0.59070331672492 | -0.06551182419870603 | [-0.0691907349222875, -0.061859987357774975] |
| min2 | 0.5381962519521083 | -0.013004759425894252 | [-0.014608393136015469, -0.011547533836543467] |
| dual | 0.525191492526214 | 0 | n/a |
| shuffled B | 0.5130047594258943 | +0.012186733100319773 | [0.010217718171339332, 0.014239932698743208] |

Positive nonzero coverage is 0.05793113705659255, far below the frozen 0.25
gate. The confidence intervals against every correct simple control are
strictly negative, as are both fixed user-hash-half mean deltas. Improvement
over the mismatched B control does not establish useful additional evidence.
The queue closed as `closed_mechanism_gate` / `baby_discard`; PID `2874627`
has exited. Sports was not started. No formal model training, validation, or
test evaluation occurred. This is a scientific mechanism rejection, not a
crash. Freeze this family; do not scan item partitions, K, target eligibility,
normalization or weight to rescue the rejected mechanism.
Budget: four CPU threads, >=12 GiB available host RAM, no GPU, two hours per
dataset and eight hours for the whole bounded queue. Old processes and frozen
families remain preserved.
Protocol: `../research/20261009_dual_witness/PROTOCOL.md`.
Remote directory: `/data/sunzixuan/dual_witness_20261009_v1`.
Source result: `../research/20261009_dual_witness/diagnosis.json`.

### F24 - Simple mean TRAIN-review expression profile (`FREEZE`)

TRAIN-only consumer review text was a new observed input in this workspace,
not rating polarity. After support-only identity matching and global removal
of all 13,447 internal targets, frozen CLIP ViT-B/16 encoded summary plus
reviewText with 75 content BPE tokens. The same donor keep mask was applied to
own review, another TRAIN user's same-item review with closest word count, and
the item's frozen text profile. All 13,447 diagnostic users were retained;
105,032 matched review records were kept, with nonzero profile coverage 1.0.

| Arm | Mean AUC | Own minus control | Paired user bootstrap 95% CI |
| --- | ---: | ---: | --- |
| own review | 0.5665994273815721 | 0 | n/a |
| same-item donor review | 0.5620352123150145 | +0.004564215066557596 | [0.0026769446716739796, 0.006446721387670111] |
| item content | 0.5884816315906893 | -0.021882204209117274 | [-0.024661750762251803, -0.019070122518033764] |

The donor comparison has a positive interval but misses the frozen +0.005
investment gate. Item content is strictly better, with negative differences
in both fixed user-hash halves (-0.022177722623345365/-0.021593249007206942).
Coverage and donor-only statistical significance cannot override either
failed condition. The simple mean-review mechanism is scientifically
`discard`, not a performance pass or proof of demand semantics.

All 19 actual-environment tests passed. The diagnostic completed in
126.97315859794617 seconds, all protected source/input SHA256 values were
unchanged, and peak GPU allocated/reserved were 521625088/606076928 bytes.
PID `2907096` has exited. No recommendation model was trained, no official
validation/test content or metrics were used, and no test set was evaluated.
Freeze this family; do not scan encoder, truncation, donor mask, word count,
profile weights or thresholds to rescue it. Preserve the positive donor
result and the stronger item-content failure together.
Protocol: `../research/20261009_review_signal/PROTOCOL.md`.
Remote: `/data/sunzixuan/review_signal_20261009_v1`; commit `9c35573`.
Source result: `../research/20261009_review_signal/diagnosis.json`.

### F25 - Per-review item-direction projection residual (`FREEZE`)

The independent fixed transform uses each matched support review's frozen
CLIP vector z and its corresponding frozen item text unit vector t:
`r = normalize(z - t * dot(z, t))`. Unit-normalized mean residual profiles
are scored against the unchanged frozen global image+text item bank. The
same globally target-removed support, identity mapping, 16 comparisons,
donor keep mask, and 13,447 users are reused; no target/validation/test text
enters any profile. Alpha is fixed at 1, and the projection is not trained.

Before metrics, gates required residual to beat EACH raw-own, same-item donor,
item-content, and residualized same-item donor by >=0.005 mean AUC, paired-user
bootstrap 95% lower bound >0, and positive deltas in both fixed hash halves;
candidate nonzero-profile coverage must be >=0.80. All five arms have coverage
1.0. The residualized-donor positive result cannot override another control.

| Arm | Mean AUC | Residual minus control | Paired user bootstrap 95% CI | Hash half 0 / 1 deltas |
| --- | ---: | ---: | --- | --- |
| residual | 0.5261164200193351 | 0 | n/a | n/a |
| raw-own | 0.5665994273815721 | -0.04048300736223693 | [-0.04369480646240797, -0.03715466275005577] | -0.04160085740072202 / -0.03938998382115017 |
| same-item donor | 0.5620352123150145 | -0.03591879229567933 | [-0.039748991410723584, -0.0320610545102997] | -0.03681558363417569 / -0.03504191792910722 |
| item-content | 0.5884816315906893 | -0.062365211571354204 | [-0.06746428106640887, -0.05728740611288765] | -0.06377858002406739 / -0.06098323282835711 |
| residualized donor | 0.5161746114374953 | +0.009941808581839816 | [0.006316464638952926, 0.013432364096080911] | +0.011573029482551143 / +0.008346815708192382 |

The first three comparisons have strictly negative CIs and both hash halves
are negative. The fourth comparison passes its individual gate, but the full
frozen conjunction fails. Status is scientifically `discard`, not a crash,
positive net benefit, causal preference semantics, or publication evidence.

All 19 actual-environment tests passed. This Baby-only diagnostic completed in
142.86831259727478 seconds, keeping and newly encoding 105,032 matched reviews.
The previous experiment had no persisted review embedding bank; this run
honestly records `reused_previous_encoded_bank=false`. Fresh review embeddings
and their index were saved with SHA256 for future independently frozen use.
All 17 protected input/dependency SHA256 values match before and after, and
the six runtime bundle files plus deployment manifest passed the final guard.
Peak GPU allocated/reserved were 521625088/606076928 bytes. PID `2935432` has
exited. No formal model training or official validation/test evaluation
occurred. Freeze the fixed projection; do not scan alpha, encoder, mask,
normalization, target selection, or decision thresholds to rescue it.
Protocol: `../research/20261009_review_residual/PROTOCOL.md`.
Remote: `/data/sunzixuan/review_residual_20261009_v1`; commit `8ad9e52`.
Source result: `../research/20261009_review_residual/diagnosis.json`.

### F27 - Residual user profile to leave-owner-out review item bank (`FREEZE`)

The unique change from F25 is the target geometry: keep the same residual
user profile, score against normalized per-item residual sums of other owners'
cached support reviews. Preserve all 13,447 users, 105,032 unique support
edges, globally removed target edges, exact 1+16 pairs and explicit
leave-query-owner-out subtraction. No review text, interaction file,
validation/test or recommendation training was used. No weight or mask scan.

| Arm | Mean user AUC | Candidate minus control | Paired-user bootstrap 95% CI |
| --- | ---: | ---: | --- |
| candidate | 0.5880029003 | n/a | n/a |
| item content | 0.5884816316 | -0.0004787313 | [-0.0088108058, 0.0071835261] |
| old target geometry | 0.5261164200 | +0.0618864803 | [0.0525996156, 0.0703919787] |
| raw review bank | 0.6703121514 | -0.0823092511 | [-0.0879889566, -0.0766804399] |
| raw donor bank | 0.6690711683 | -0.0810682680 | [-0.0872825956, -0.0752993233] |
| residual donor bank | 0.5914841600 | -0.0034812598 | [-0.0068230832, -0.0000880773] |
| same-support popularity | 0.7576318138 | -0.1696289135 | [-0.1757389543, -0.1635889394] |
| same-support ItemKNN | 0.6782181899 | -0.0902152897 | [-0.0966414442, -0.0840493000] |
| same-support UserKNN | 0.5906405704 | -0.0026376701 | [-0.0085525582, 0.0032704204] |

Candidate all17-bank/profile coverage passed at 0.8261322228. Only the old
target geometry comparison passed the complete positive margin/CI/hash-half
gate; the required eight-control conjunction failed. Stable author expression
and improvement over a weak geometry do not establish preference transfer.
Raw review-bank association also has a nearly identical donor counterpart,
and popularity is stronger. It is not an independently confirmed method.

17/17 actual MMRec tests passed, CPU runtime 40.503669 seconds, peak RSS
3,962,508 KiB, PID3112371 exited. Fifteen input and all deployed bundle hashes
match before/after; all five F25 per-user AUC arrays were reproduced exactly.
Source commit367069c; separate report commit125fd95. Preserve all arrays and
references; the F23 full-support KNN (72 additional support edges) is reference
only and was not substituted for gate controls. Freeze this geometry and
avoid residual/raw-bank weights, alpha, mask or threshold searches.
Evidence: `../research/20261010_review_bank_alignment/RUN_REPORT.md`.

## Completed expression diagnostic (not a recommendation gain)

The TRAIN review expression audit completed once at frozen commit `4c158f1`;
the separate evidence report is commit `50196fb`. Both views passed their
independent expression gates over all 19,445 support users. Raw real/null AUC
was `0.5485356751/0.4988611334`, delta `+0.0496745417`, paired-user bootstrap
95% CI `[0.0469730338, 0.0522377944]`. F25 residual real/null AUC was
`0.5618260346/0.5006561573`, delta `+0.0611698773`, CI
`[0.0582706771, 0.0640631279]`. Coverage was `10411/19445` in each view,
both fixed hash halves were positive, 17/17 actual MMRec tests passed,
CPU runtime was 10.091 seconds, and input/bundle hashes stayed identical.
PID `3104343` exited. This establishes stable author-expression association
within the frozen cache only; writing style remains a possible explanation.
Preference, recommendation gain, novelty, publication evidence and formal
training authorization remain false. F24/F25 stay rejected and frozen.
Evidence: `../research/20261009_review_identifiability/RUN_REPORT.md`.

## Active candidate

The review-bank geometry diagnostic F27 has completed and failed. The next
independent signal hypothesis is under read-only study, with no new training
launched. Actual Baby CPU cache equivalence passed, as did all twelve saved
Electronics SVD-input CPU projector audits; actual Electronics GPU/model
audits are being prepared before any numerical recovery training. These are
infrastructure work, not innovation gains. Preserve all frozen failures.

Publication confirmation follows `PUBLICATION_PROTOCOL.md`. Electronics
assets have independently been restored on host213 to
`/data/sunzixuan/bridge_extra_assets_20261009/elec`; all three source-file
SHA256 values match the audited source. An independent original-BRIDGE
Electronics baseline queue PID `2933472`, immutable commit `06bcd21`, completed
its CPU graph cache as child PID `2936426` (`exit 0`) in 53.8258
seconds. Graph SHA256 is
`76ccb18835b2d18b2f56e882cf4670b2d7af733951090656ab4be23ef70a9ed2`.
The queue and seed child PID `2936814` have now exited with technical failure.
Seed 2026 failed at zero-based epoch9/batch225 after 5,742 finite updates and
13,723.918 seconds. SVD call0 produced all-nonfinite input gradients despite
finite upstream gradients; it has an equal singular pair 48/49 within band3,
with positive inter-band boundary gaps. No invalid optimizer update was made.
Nine epochs completed; rounded intermediate validation R20/N20 was
`0.0643/0.0283`. There is no completed result, 0/5 seeds finished, and seeds
2027-2030 never started. This is numerical infrastructure failure, not a
negative innovation metric. Source/input/checkpoint audits match, test was
not evaluated, and the original version remains immutable. A recovery must
be a separately declared version with equivalence evidence before retraining.
Evidence: `../research/20261010_electronics_failure/AUDIT.md`.

## Lessons that constrain future ideas

1. Stop editing graph topology from current model predictions or fixed content
   neighbors; three separate mechanism families have failed this way.
2. A gate is not new information. Routing, temperature calibration, and CL
   weights repeatedly failed when they only redistributed existing views.
3. Preserve the carrier exactly at initialization and bound every new residual.
4. New mechanisms should introduce an independently testable signal: temporal
   history, candidate-conditioned preference, stronger foundation-model
   features, or causal exposure information.
5. Diagnose by user activity and modality reliability before expanding seeds;
   an overall tie can hide a useful sparse-user effect, but subgroup gains alone
   are not enough for the main performance claim.
6. MSCA's six auxiliary alignment terms are much larger than BPR on the shared
   ID backbone, but the 36-probe diagnostic found no material directional
   conflict. Do not use gradient surgery as the next mechanism.
7. Explicit star-rating polarity is not a useful shortcut for this implicit
   benchmark. In contrast, unsigned exposure semantics remain predictive, so
   reproduce SG-URInit before inventing another user-profile transformation.
