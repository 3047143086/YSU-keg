# Rating-Residual Multimodal Anchor Diagnostic

## Status

Diagnostic complete: **NO-GO / FREEZE**. The Baby/seed-2026 result failed four
of five preregistered checks. No full rating-residual mechanism or correction-
scale search should be implemented.

## Motivation and novelty boundary

The processed Amazon data retain 1--5 star ratings, but official MSCA and most
implicit-feedback multimodal baselines collapse every observed interaction to
the same positive edge. On Baby, 83.5% of training users use more than one
rating value, so polarity is not merely a constant user attribute.

The candidate separates two meanings of an interaction:

1. An inverse-item-degree exposure profile, matching SG-URInit's local
   aggregation; and
2. An inverse-item-degree rating residual centered within each user, which
   captures whether an item's multimodal content was preferred relative to
   that user's own rating scale.

The second profile is unchanged if a constant is added to all ratings from one
user. It therefore cannot exploit generous-versus-harsh rater bias. Ratings are
shuffled within each user as a negative control, preserving every user's rating
histogram and interaction graph while destroying item-polarity correspondence.

This direction is distinct from the frozen reliability gates, history-only
score residual, graph augmentation, and gradient surgery: it introduces an
observed supervision signal that those experiments discarded. It also changes
the information setting. A future paper comparison is fair only if rating-aware
controls receive the same training ratings; gains over implicit-only models
must be labeled as using additional supervision.

## Fixed diagnostic

- Carrier data: Baby, seed 2026.
- Inputs: training rows (`x_label=0`) only for profile construction.
- Evaluation: validation rows (`x_label=1`) only; test rows are never scored or
  used to select a mechanism.
- Features: frozen image and text arrays, independently L2-normalized.
- Exposure aggregation: SG-URInit inverse-item-degree weights.
- Polarity aggregation: the same weights multiplied by each rating minus the
  user's inverse-degree-weighted mean rating.
- Control: deterministic within-user rating shuffle, seed 2026.
- Retrieval diagnostic: each validation event versus 16 deterministic items
  absent from that user's training history.
- Fusion: mean image/text cosine score.
- Fixed combined score: exposure score + `0.25 * polarity score`.

The diagnostic passes only when all conditions hold:

1. A nonzero polarity profile covers at least 70% of validation events.
2. For validation events rated 4 or 5, fused polarity pairwise AUC is at least
   0.53.
3. That high-rating AUC exceeds the within-user shuffled-rating control by at
   least 0.02.
4. The fixed combined score improves overall validation pairwise AUC over the
   exposure-only score by at least 0.005 absolute.
5. The paired normal-approximation 95% confidence interval for condition 4 has
   a lower bound above zero.

Thresholds and the `0.25` scale are fixed before reading the result. Failure of
any condition freezes this candidate without scale tuning. Passing the
diagnostic only authorizes a function-preserving MSCA implementation; it does
not establish top-k improvement or publishability.

## Result

The diagnostic evaluated 20,559 validation events against 16 deterministic
unobserved items per event.

| Quantity | Result | Required | Pass |
|---|---:|---:|:---:|
| Polarity-profile coverage | 0.8751 | >= 0.70 | Yes |
| High-rating polarity AUC | 0.5086 | >= 0.53 | No |
| High-rating AUC gain over shuffled ratings | 0.0050 | >= 0.02 | No |
| Combined overall AUC gain over exposure | -0.00284 | >= 0.005 | No |
| Paired delta 95% CI lower bound | -0.00416 | > 0 | No |

The exposure-only fused profile achieved pairwise AUC `0.5992`. Adding the
fixed polarity correction reduced it to `0.5963`; the paired 95% confidence
interval was `[-0.00416, -0.00152]`, so the regression is statistically clear.
The polarity profile was only marginally above chance for high ratings
(`0.5086`) and nearly matched the within-user shuffled control (`0.5036`).

The explicit rating residual therefore does not provide enough transferable
multimodal preference signal for the standard implicit-ranking benchmark.
Lowering the gate or changing the `0.25` scale after seeing this result would
be post-hoc search. The strong exposure-only AUC supports evaluating official
SG-URInit as a recent baseline adapter, but it does not rescue this polarity
mechanism.

## Follow-up decision

Freeze the rating-residual branch. The next experiment is baseline calibration:
an MSCA + official-formula SG-URInit exposure adapter using only implicit
training interactions. That adapter is a baseline, not the paper innovation.

## Command

```bash
cd /data/sunzixuan/PRAVI_remote/src
/data/sunzixuan/miniconda3/envs/MMRec/bin/python \
  diagnose_rating_signal.py --dataset baby --seed 2026 --negatives 16 \
  --correction-scale 0.25 \
  --output-json ../experiments213/rating_signal_baby_s2026.json
```
