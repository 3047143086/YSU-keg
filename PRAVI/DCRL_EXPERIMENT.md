# DCRL: Disagreement-Conditioned Conjunctive Ranking Loss

## Hypothesis

MSCA's fused score can hide a modality-specific ranking error: a positive item
may win because one view is strong while the other view ranks a sampled
negative higher. DCRL adds a score-level conjunctive constraint requiring the
positive to beat the negative in both propagated modality views. The extra
loss is weighted by the detached absolute difference between the two margins,
so it concentrates on cross-modal conflicts.

This is a loss-space intervention only. It does not modify interaction graphs,
candidate pools, modality gates, feature adapters, or user-profile averages.

## Fixed screen

- Carrier: official MSCA (WWW 2026), `n_layers=2`, `fusion_coeff=0.4`,
  `cl_weight=0.005`, `reg_weight=3e-7`.
- Dataset/seed: Baby, 2026.
- DCRL: weight `0.02`, margin `0.10`, temperature `0.20`, minimum conflict
  weight `0.10`.
- Selection: validation Recall@20 only; test is disabled during screening.
- Go gate: both validation R@20 and N@20 must exceed official MSCA
  `0.1043/0.0462`, with no material R@10 or N@10 regression.

The experiment is exploratory until it clears the gate on Baby and then a
pre-specified Sports confirmation. A one-seed gain is not sufficient paper
evidence; a passing candidate requires five seeds on Baby, Sports, and
Electronics with paired significance and efficiency reporting.

## Implementation

`src/models/msca_dcrl.py` caches the six MSCA views during the existing forward
pass. For each sampled pair it computes visual and textual margins, uses their
minimum as the conjunctive margin, and applies a detached disagreement weight
to a softplus margin loss. The official MSCA BPR and alignment losses remain
unchanged.

## Result (2026-09-03)

The fixed Baby/seed-2026 screen completed at epoch 66 (best checkpoint: epoch
45). Validation metrics were R@10 `0.0667`, R@20 `0.1031`, N@10 `0.0363`, and
N@20 `0.0455`. The official MSCA reference is
`0.0675/0.1043/0.0369/0.0462`, so DCRL missed both primary gates and had small
regressions on all four metrics. Test evaluation was intentionally disabled.

Decision: freeze DCRL as a failed screen; do not tune its weight, margin,
temperature, or conflict weighting on this same carrier.
