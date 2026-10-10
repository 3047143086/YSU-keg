# RAPA Diagnostic: Ranking-Anchored Projected Alignment

## Status

Diagnostic complete: **NO-GO / FREEZE**. The Baby/seed-2026 run failed the
preregistered gate. RAPA remains a rejected hypothesis, not a model or
performance claim; no projection model or hyperparameter sweep should be run.

## Motivation and novelty boundary

Official MSCA jointly optimizes BPR and six equally scaled semantic alignment
terms. Prior experiments show that changing scalar weights, gates, sampling,
and semantic residuals does not improve ranking. The remaining testable
hypothesis is that useful semantic gradients contain components that oppose the
ranking gradient on the shared ID backbone.

The diagnostic measures that hypothesis without changing training. A future
RAPA model would treat BPR as the ordered primary objective and modify only a
materially conflicting part of each auxiliary gradient. Ordinary PCGrad or the
existing Mirror Gradient training option is not, by itself, a publishable
contribution. Any promoted design must exploit MSCA's user/item and
modality/structure decomposition and retain a ranking-descent guarantee.

## Fixed diagnostic

- Carrier: official MSCA (WWW 2026).
- Data/seed: Baby, 2026.
- Training: 12 epochs, three evenly spaced probes per epoch.
- Frozen parameters: `n_layers=2`, `fusion_coeff=0.4`, `cl_weight=0.005`,
  `reg_weight=3e-7`.
- Probe parameters: user and item ID embeddings, separately and jointly.
- Terms: user-image, user-text, item-image, item-text, user-structure, and
  item-structure alignment.
- A conflict is counted only when gradient cosine is below `-0.05`.

The diagnostic passes when either:

1. At least two alignment terms have conflict fraction at least `0.25`, median
   weighted gradient norm at least `0.01` of BPR, and negative-work fraction at
   least `0.10`; or
2. The summed alignment gradient has conflict fraction at least `0.20`, median
   norm ratio at least `0.03`, and negative-work fraction at least `0.10`.

These thresholds are fixed before reading the result. A no-go freezes this
optimization direction without a threshold or parameter sweep.

## Result

The run completed all 36 probes (12 epochs x 3) and the loss-decomposition
check matched the official objective exactly (absolute error `0.0`).

| Gradient | Median cosine | Minimum cosine | Conflict fraction | Median norm ratio | Negative-work fraction |
|---|---:|---:|---:|---:|---:|
| Summed alignment | 0.0147 | -0.0074 | 0.0000 | 9.5240 | 0.0283 |
| User-image | 0.0226 | 0.0031 | 0.0000 | 1.3500 | 0.0000 |
| User-text | 0.0246 | 0.0009 | 0.0000 | 1.2383 | 0.0000 |
| Item-image | 0.0113 | 0.0019 | 0.0000 | 1.5877 | 0.0000 |
| Item-text | 0.0003 | -0.0092 | 0.0000 | 1.4954 | 0.0704 |
| User-structure | 0.0191 | -0.0035 | 0.0000 | 2.4396 | 0.0100 |
| Item-structure | 0.0015 | -0.0180 | 0.0000 | 6.6792 | 0.3522 |

No individual term and no summed gradient crossed the material-conflict
threshold even once. Auxiliary magnitude is large, but the hypothesized
repeated opposition to BPR is absent. Both gate branches therefore fail. This
rules out RAPA/PCGrad as a defensible next mechanism on this carrier; lowering
the threshold after observing the result would violate the protocol.

## Command

```bash
cd /data/sunzixuan/PRAVI_remote/src
python diagnose_msca_gradients.py --dataset baby --seed 2026 --gpu-id 1 \
  --epochs 12 --probes-per-epoch 3 \
  --output-json ../experiments213/rapa_gradient_baby_s2026.json
```
