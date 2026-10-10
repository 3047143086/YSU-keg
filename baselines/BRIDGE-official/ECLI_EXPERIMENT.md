# ECLI Validation-Only Screening

## Status

**NO-GO / FREEZE.** Baby passed the directional screen, but Sports violated the
preregistered Recall@10 guard and all paired confidence intervals crossed zero.
Do not tune or reuse this late-interaction correction family.

## Hypothesis

Existing history adapters collapse all consumed items into one user vector.
That erases sparse users' distinct interests and was a documented failure mode
for LCHR. Earlier target-safe evidence diagnostics instead found that the mean
of the strongest three candidate-to-history similarities reached pairwise AUC
`0.6225` on Baby, substantially above the single exposure profile (`0.5995`).

ECLI turns that observation into a local ranking mechanism:

1. Start from the saved validation-selected BRIDGE checkpoint.
2. Remove consumed items, then retain the official top-200 recommendation set.
3. Project frozen image and text features through BRIDGE's learned modality
   projectors and compute candidate-to-history late interactions.
4. Average each modality's strongest three history matches, then fuse image and
   text evidence equally. This preserves multiple interests without adding
   pseudo-edges or changing the encoder graph.
5. Standardize evidence within each user's candidate set. Bound it to two
   standard deviations and express its correction in units of the user's
   median score gap among the top 50. There is no learned or searched fusion
   coefficient.

The construction combines SG-URInit's training-history semantics with
candidate-wise late interaction, while retaining BRIDGE's local-correction
principle. Unlike C2R, it adds candidate-to-history evidence rather than merely
routing the three existing score blocks.

## Frozen first screen

- Dataset/split/seed: Baby validation, seed 2026.
- Carrier: matched BRIDGE checkpoint `bridge_baby_seed2026_screen`.
- Candidate set: 200; late-interaction Top-q: 3.
- Modalities: projected image and text, equal weight.
- Residual: clipped z-score times median top-50 adjacent score gap.
- Selection: no hyperparameter search and no test evaluation.

Proceed only if validation Recall@20 and NDCG@20 both improve, while Recall@10
and NDCG@10 regress by no more than `0.0002`. A pass authorizes an independently
matched Sports checkpoint screen followed by new-seed training. A failure
freezes ECLI, its Top-q variants, and residual-strength tuning.

## Rounded first result (2026-09-03)

BRIDGE validation R@10/R@20/N@10/N@20 was
`0.0709/0.1056/0.0404/0.0492`; ECLI reached
`0.0710/0.1058/0.0405/0.0493`. This clears the original directional rule, but
the margin is within the resolution of the four-decimal evaluator. The exact
rerun adds unrounded per-user metrics and paired 95% confidence intervals
without changing any ECLI setting.

The exact rerun confirmed positive deltas for R@10/R@20/N@10/N@20 of
`+0.0001029/+0.0002657/+0.0000267/+0.0000616`. Paired 95% confidence intervals
for the primary metrics were `[-0.0004243, +0.0009557]` (R@20) and
`[-0.0001289, +0.0002521]` (N@20). This authorizes the declared Sports screen,
but is not statistically sufficient evidence by itself.

## Sports confirmation (2026-09-03)

Sports exact deltas for R@10/R@20/N@10/N@20 were
`-0.0002729/+0.0000843/-0.0000409/+0.0000496`. R@20 and N@20 retained a tiny
positive direction, but R@10 exceeded the allowed `0.0002` regression and all
paired 95% confidence intervals included zero. The declared gate therefore
failed. Test was never evaluated, and no post-hoc Top-q or residual-strength
search is authorized.

## Command

```bash
cd /data/sunzixuan/bridge_official/src
/data/sunzixuan/miniconda3/envs/MMRec/bin/python \
  diagnose_ecli.py --dataset baby --seed 2026 --gpu-id 1 \
  --checkpoint saved_models/BRIDGE/baby/bridge_baby_seed2026_screen/model.pth \
  --output-json ../experiments_ecli/ecli_baby_s2026.json
```
