# C3E Cross-Fitted Collaborative Evidence Calibration

## Status

Diagnostic complete: **NO-GO / FREEZE**. The degree-conditioned C3E mechanism
failed four of six preregistered checks. Do not integrate it into MSCA/PNGRec
or tune degree bins, shrinkage, Top-q, or thresholds to rescue this result.

## Mechanism

C3E converts multimodal similarity into collaborative evidence rather than
treating cosine similarity itself as preference:

1. Split users into two deterministic folds. For each target fold, fit only on
   training interactions from the other fold.
2. Sample positive item pairs co-consumed by calibration users and random item
   pairs matched on both endpoint popularity bins.
3. For each modality and popularity-bin pair, estimate a smoothed monotone
   log-likelihood-ratio curve from semantic similarity to co-consumption.
   Sparse cells shrink toward a global curve.
4. Score each validation candidate against the target user's training history
   and average its strongest three calibrated evidence values. This preserves
   multiple interests instead of collapsing the history into one mean vector.

The target user's interactions never fit that user's calibrator. Validation
interactions are used only for the final diagnostic, and test interactions are
never loaded as positives or used for selection.

## Frozen controls and gate

- Dataset/seed: Baby/2026.
- Two user-level cross-fitting folds.
- Sixteen deterministic unobserved negatives per validation event.
- Eight co-consumption pairs per calibration user, capped at 100,000 per fold.
- Sixteen semantic bins, four item-degree bins, shrinkage 200.
- Candidate aggregation: top three history matches.
- Controls: SG-URInit-style inverse-degree exposure profile, raw candidate
  top-three cosine, and global calibration without degree conditioning.
- Go only if all checks pass:
  - C3E pairwise AUC is at least `0.6192`.
  - C3E improves exposure AUC by at least `0.0200` and its paired 95% CI lower
    bound is above zero.
  - C3E improves raw candidate Top-q AUC by at least `0.0050` and its paired
    95% CI lower bound is above zero.
  - The C3E-minus-exposure delta is positive in both held-out user folds.

Passing only authorizes a validation-only top-k integration into MSCA and
PNGRec. Failure freezes this exact feature/calibration/aggregation branch; it
does not authorize tuning bins, q, shrinkage, or thresholds after seeing the
result.

## Result

The diagnostic used 118,551 training rows and 20,559 validation events. Each
event was compared with 16 deterministic unobserved items.

| Signal | Pairwise AUC | 95% CI |
|---|---:|---:|
| Inverse-degree exposure profile | 0.5995 | [0.5952, 0.6038] |
| Raw candidate Top-3 cosine | 0.6225 | [0.6182, 0.6268] |
| Global cross-fitted calibration | **0.6264** | [0.6221, 0.6307] |
| Degree-conditioned C3E | 0.6121 | [0.6079, 0.6164] |

Degree-conditioned C3E improved exposure by `0.01262`, with paired 95% CI
`[0.00959, 0.01565]`, but missed the required `0.0200`. It regressed against
raw candidate Top-3 by `-0.01039`, paired 95% CI
`[-0.01231, -0.00847]`. Its fold-specific gains over exposure were positive
(`0.01593` and `0.00933`), so the failure is not sampling noise: degree-cell
conditioning consistently discards useful global semantic ordering.

The global calibrator was the strongest observed signal, but its unregistered
gain over raw Top-3 was only `0.00386`, below the frozen `0.0050` threshold.
Treat this as a follow-up observation, not a passed result. Any confirmatory
experiment must be specified independently before rerunning.

## Commands

```bash
cd /data/sunzixuan/PRAVI_remote
/data/sunzixuan/miniconda3/envs/MMRec/bin/python -m unittest tests/test_c3e.py

cd src
CUDA_VISIBLE_DEVICES=0 /data/sunzixuan/miniconda3/envs/MMRec/bin/python \
  diagnose_c3e.py --dataset baby --seed 2026 --device cuda \
  --output-json ../experiments213/c3e_baby_s2026.json
```
