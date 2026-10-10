# History Pairing Information Screen

Frozen before launch on 2026-09-08. Working label: HPAIR. This is a mechanism
screen, not an established novel method, a paper reproduction, or a promised gain.

## Literature and novelty boundary

- [SynIB, 2026 preprint](https://arxiv.org/html/2606.09853v1): encourages
  multimodal synergy using task prediction and perturbed-modality confidence
  regularization. We borrow the question of information beyond individual
  modalities, not its loss or a claim to estimate partial information decomposition.
- [Bayesian Neural Networks for Functional ANOVA, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/29cd189d6e68a1b2ea7c4bff619fcb39-Abstract-Conference.html):
  distinguishes main effects and higher-order interactions using tensor-product
  neural bases. Our observable centered cross-moment is not its Bayesian model,
  an identifiable functional ANOVA decomposition, or a new mathematical identity.
- [CovMatch, October 2025 preprint](https://arxiv.org/html/2510.18583v1):
  cross-covariance matching for multimodal dataset distillation motivates checking
  paired cross-modal statistics beyond marginal feature means. Its distilled-data
  training and encoder updates are not reproduced or claimed here.
- [SynGR, May 2026 preprint](https://arxiv.org/html/2605.18920v1),
  [PRISM, January 2026 preprint](https://arxiv.org/abs/2601.10944), and
  [MRdIB, September 2025 preprint](https://arxiv.org/html/2509.20225v1)
  already address synergy in recommendation. Generic synergy is not our novelty.
- [MITRec, authors' implementation](https://github.com/HubuKG/MITRec)
  already models intertwined modality tensors. Generic bilinear/tensor fusion is
  not our novelty and this screen's generic tensor arm is not a reproduction of MITRec.
- [OrthoRec, September 2, 2026](https://arxiv.org/abs/2609.02152)
  addresses collaborative-anchor orthogonal purification and conflict-aware
  routing. No claim is made to invent orthogonal purification or gating.

The candidate distinction is a falsifiable information-source hypothesis:
**does within-user image-text correspondence carry preference information beyond
the two unchanged modality multisets and their marginal means?** Literature
search has not established priority. If useful, the next stage must compare
published recommendation methods and search historical second-order user models.

## Why this is a different test

The preceding RCMQR screen completed all 24 jobs, with no R20/N20 improvement
from conditional retrieval and a reduction in candidate recall. Its candidate
sets changed while the final ranker stayed shared and frozen. This new screen
changes actual full-catalog scores and saves actual top-20 item identities.

The failure ledger's ECLI used additive per-modality candidate/history top-3
similarities, not a test isolating within-history image/text pairing. No new
graph edges, confidence gates, alignment losses, temporal kernels, or feature
extractors are introduced here. The residual-scoring implementation is a probe
for paired history statistics, not itself a novelty claim.

## Representation and arms

Existing frozen global features have two 512-dimensional modalities. Center each
modality using the label-free catalog mean and use fixed, seeded orthogonal
projections to 32 dimensions. For support history H_u, let projected features be
a_i and b_i:

    M_u = mean(a_i) outer mean(b_i)
    J_u = mean(a_i outer b_i)
    C_u = J_u - M_u

Candidate tensor is a_j outer b_j. Learn 1024 -> 128 -> 64 user/item heads and a
bounded normalized dot-product score. Tensor arms have exactly the same parameter
count and initial parameters. Four variants differ only in the user statistic:

| Arm | User representation | Purpose |
| --- | --- | --- |
| marginal_tensor | M_u | Ordinary tensor interaction of marginal preferences |
| paired_moment | J_u | Uncentered paired-history second moment |
| paired_cov | C_u | Prespecified main candidate |
| shuffled_cov | Cov(a_i, b_perm(i)) | Within-user pairing control |
| additive | Separate image/text means and heads | Additional additive capacity |

The additive arm has approximately the same capacity, not exact equality. Report
actual counts. The four tensor arms have exactly matched counts. The baseline
already has collaborative and independently learned content scores; every arm
retains exactly this same frozen baseline and receives the same training targets.

Uniform within-user permutations preserve full modality multisets, means and
history lengths. Fixed points are allowed: forced derangements give a biased
negative covariance for very short histories. One-item histories have exactly
zero raw covariance. This is a randomized falsification control, not a causal
counterfactual estimate. Three seeds include independent permutations. Report
history-length, zero-statistic, and unchanged-permutation diagnostics.

All tensor user RMS scales fit role 0 only; item feature normalization is
label-free and transductive. The statistic never contains the training target:
all labels used to train the new branch are held out from support globally.

## Data and score contract

- Read-only source: `/data/sunzixuan/rcmqr_20260908_screen_v2`.
- Output: `/data/sunzixuan/hpair_20260908_screen_v1`.
- Existing MMRec environment, Baby data/features, and three base checkpoints.
- No downloads, no source experiment mutation, no official validation/test reads.
- 99,106 support edges; 19,445 held targets split into role 0 training (11,667),
  role 1 calibration (3,889), role 2 internal development (3,889).
- Reuse each seed's frozen shared content scorer and its recorded alpha (do not
  assume a common alpha). Baseline is support-masked standardized BRIDGE score
  plus alpha times the existing normalized independent-content score.
- Exact full-catalog scoring; exclude support items, not evaluation targets.
- Train new branch using role-0 full-catalog cross-entropy, frozen baseline plus
  train-scale 1 residual. This is an implicit-positive ranking proxy, not an
  exposure-debiased likelihood; unobserved items can contain false negatives.
- Calibrate checkpoint and residual scale on role 1 only. Scale grid:
  0, 0.1, 0.25, 0.5, 1, 2. Select N20, then R20; deterministic lower-scale tie rule.
  Scale 0 reproduces baseline exactly and must be reported as no branch benefit.
- At most 60 epochs, patience 8 evaluations; evaluate every 5 epochs and epoch 1.
- Lock selection before evaluating role 2 once per formal arm.
- Smoke tests use training and calibration subsets only, never role 2.
- Save per-user metrics and top-20 IDs for both new and frozen scores, as well as
  top-20 membership/order change fractions. Equal metrics do not imply equal lists.

This development split was used in preceding screens. Results remain exploratory
even with confidence intervals. Do not treat it as a fresh confirmation set or
compare these values directly with older official-validation numbers.

## Decision and execution budget

15 formal runs: 5 variants x seeds 2026, 2027, 2028, preceded by unit tests and
five real-data GPU smoke tests. Serial GPU 1 only. Wait if GPU memory used exceeds
6 GB or utilization exceeds 40%; cap each child at 25% of GPU memory. Four CPU
threads, 30-minute limit per formal job, eight-hour pipeline limit including waits.
Terminate only this queue's own children on timeout. Never stop unrelated jobs.

Primary comparisons: paired_cov vs additive, marginal_tensor, shuffled_cov, and
the shared frozen baseline. Both R20 and N20 must improve at least 2% in the
three-seed mean with positive deltas in every seed. Mean R10/N10 must not decrease.
Also require positive lower bounds of exploratory paired user-bootstrap 95%
intervals against the three learned primary controls. Bootstrap averages seed
deltas per user before resampling users, rather than pretending 3N independent users.
Intervals are not multiple-testing-adjusted or corrections for earlier screens.

The paired_moment ablation determines whether centering actually adds value. If
it ties/wins, do not attribute gains to covariance centering. If shuffled_cov
ties/wins, do not attribute gains to true history pairing. A favorable Baby
screen is only permission to design fresh confirmation, not evidence that the
method can publish at a strong venue. No automatic Sports/test expansion occurs.

## Artifacts

- `pipeline_launch.json`: detached launcher PID and start time.
- `pipeline_status.json`, `pipeline.log`: queue health and current child.
- `results/<seed>/<variant>/`: configuration, checkpoint, progress, logs, final
  result, per-user arrays, representation diagnostics, and input/source hashes.
- `summary.json`, `RESULTS.md`: automatically refreshed summary and decision.
- `python research/20260908_pairing/manage.py status --brief`: fetch current
  remote state into local `latest_status.json` and `REMOTE_RESULTS.md`.
