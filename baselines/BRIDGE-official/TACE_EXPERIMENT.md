# TACE Validation-Only Screening

## Status

**NO-GO / FREEZE.** Baby failed the frozen material and directional gates.
No Sports run, test-set evaluation, or coefficient search is authorized.

## Mechanism

BRIDGE's behavior branch treats every pair of items consumed by the same user
as equally related, even when the events are far apart in that user's history.
TACE (Temporal-Affinity Calibrated Evidence) replaces this ItemKNN relation
with a training-only chronological affinity. A user's contribution to an item
pair is

```text
1 / max(distance between distinct timestamp ranks, 1).
```

Events with equal timestamps form one episode. The affinity is cosine
normalized and row-TopK pruned exactly like BRIDGE. Candidate scope,
sum-sqrt history aggregation, candidate-wise z-normalization, learned gate,
and behavior correction weight are inherited unchanged from the checkpoint.
This isolates the value of temporal cohesion rather than adding a tunable
reranker.

The benchmark split is not chronological, so TACE is not framed as next-item
prediction or target-time modeling. Timestamps are used only inside the
training split to estimate whether co-consumption reflects a coherent episode.

## Frozen first screen

- Dataset/split/seed: Baby validation, seed 2026.
- Carrier: matched validation-selected BRIDGE checkpoint.
- Data available to TACE: training interactions and their timestamps only.
- Parameters searched: none.
- Evaluation: exact per-user R@10/R@20/N@10/N@20 and paired 95% CIs.
- Material gate: R@20 improves by at least `0.0010`, N@20 by at least
  `0.0004`, and neither R@10 nor N@10 regresses by more than `0.0002`.

A pass authorizes the identical Sports validation confirmation. A failure
freezes this temporal ItemKNN family; kernel or mixture searches on Baby are
not allowed.

## Result (2026-09-03)

Exact BRIDGE validation R@10/R@20/N@10/N@20 was
`0.0708605/0.1055593/0.0404249/0.0492366`; TACE reached
`0.0698889/0.1055919/0.0386836/0.0477486`. The corresponding deltas were
`-0.0009716/+0.0000326/-0.0017413/-0.0014880`. N@20's paired 95% confidence
interval was `[-0.0024030, -0.0005730]`, so the ranking-quality regression is
statistically clear. TACE is frozen and the test split was not read.

## Command

```bash
cd /data/sunzixuan/bridge_official/src
/data/sunzixuan/miniconda3/envs/MMRec/bin/python -u diagnose_tace.py \
  --dataset baby --seed 2026 --gpu-id 1 \
  --checkpoint saved_models/BRIDGE/baby/bridge_baby_seed2026_screen/model.pth \
  --output-json ../experiments_tace/tace_baby_s2026_exact.json
```
