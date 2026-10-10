# JCEF Joint Cross-Fitted Evidence Fusion

## Status

Frozen after the Baby/seed-2026 signal diagnostic failed its preregistered
gate. No full model integration or test evaluation is authorized, and this
feature/ridge/Top-q family must not be tuned further.

## Hypothesis and mechanism

C3E showed that candidate-specific history matching is predictive and that
global evidence calibration improves equal image/text fusion, while degree
conditioning destroys useful ordering. JCEF therefore removes degree cells
entirely and tests a different mechanism: collaborative relevance depends on
the joint state of image and text evidence, especially agreement and conflict,
not on an average of separately calibrated scores.

For each held-out user fold, JCEF uses training users from the other fold to:

1. Fit one monotone similarity-to-co-consumption likelihood curve per modality.
2. Map image/text similarity for each item pair into calibrated evidence.
3. Fit a fixed ridge-logistic evidence model with four inputs: image evidence,
   text evidence, their product, and their absolute disagreement.
4. Score a candidate against every item in the target user's training history
   and aggregate its strongest three joint evidence values.

The target user's interactions never train its curves or fusion model.
Validation events are used only for the final signal test; test interactions
are not evaluated. A two-feature ridge-logistic model is run as a learned
additive control. JCEF must beat it, so a gain caused only by reweighting image
and text cannot be claimed as cross-modal synergy.

## Frozen diagnostic

- Dataset/seed: Baby/2026.
- Two deterministic user-level cross-fitting folds.
- Sixteen deterministic unobserved negatives per validation event.
- Eight co-consumption pairs per calibration user, capped at 100,000 per fold.
- Sixteen monotone similarity bins; no degree-conditioned evidence cells.
- Ridge coefficient: `0.01`.
- Candidate aggregation: Top-3 history evidence.
- Controls: inverse-degree exposure, raw candidate Top-3, equal globally
  calibrated fusion, and learned additive fusion.

Go only if all conditions hold:

1. JCEF pairwise AUC is at least `0.6314`.
2. JCEF improves equal global calibration by at least `0.0050`, with paired
   95% CI lower bound above zero.
3. JCEF improves learned additive fusion by at least `0.0020`, with paired 95%
   CI lower bound above zero.
4. The JCEF-minus-equal-global delta is positive in both user folds.

Failure freezes this joint feature set and ridge fusion. Do not search ridge,
Top-q, curve bins, polynomial terms, or thresholds after reading the result.
Passing automatically authorizes a function-preserving, validation-only JCEF
residual in official MSCA, followed by the same screen on PNGRec.

## Result (2026-09-03)

The Baby/seed-2026 diagnostic completed with 20,559 validation events and 16
deterministic negatives per event. JCEF pairwise AUC was `0.62784` (95% CI
`[0.62355, 0.63212]`), below the `0.6314` gate. Its paired improvement over
global equal calibration was `+0.00147` (95% CI `[+0.00028, +0.00265]`), and
over learned additive fusion was `+0.00088` (95% CI `[+0.00034, +0.00143]`),
both below their required margins. Fold deltas were positive but small
(`+0.00060`, `+0.00233`). The complete gate therefore failed. No MSCA
integration or test evaluation is authorized; freeze this joint feature set
and its ridge fusion.

## Commands

```bash
cd /data/sunzixuan/PRAVI_remote
/data/sunzixuan/miniconda3/envs/MMRec/bin/python -m unittest \
  tests/test_c3e.py tests/test_jcef.py

cd src
CUDA_VISIBLE_DEVICES=0 /data/sunzixuan/miniconda3/envs/MMRec/bin/python \
  diagnose_jcef.py --dataset baby --seed 2026 --device cuda \
  --output-json ../experiments213/jcef_baby_s2026.json
```
