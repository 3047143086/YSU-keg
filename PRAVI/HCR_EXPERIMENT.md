# LCHR: Leave-One-Out Calibrated History Residual

## Status

Implementation complete. The first Baby/seed-2026 validation-only screening
run is pending or running; no performance claim is valid until its log is
recorded in `../result/FAILED_EXPERIMENTS.md` or promoted here as supported.

## Motivation

Recent multimodal recommenders show that user history can provide an organic
view that complements the user-item graph, while small ID-anchored residuals
are generally safer than replacing collaborative representations. However,
forming a training history prototype that contains the current positive item
creates a self-copy shortcut: the target is part of the feature used to score
it.

LCHR tests a narrower mechanism:

1. Build visual and textual user prototypes from observed training history.
2. During BPR training, remove the current positive item before scoring it.
3. Center item content by the catalog mean and calibrate each modality with
   detached history coherence.
4. Allocate less residual capacity to high-activity users, whose collaborative
   representation already has stronger evidence.
5. Add the correction only in score space through two bounded scalar gates.

It never adds interactions, rewrites graph topology, or uses validation/test
interactions.

## Scoring mechanism

For modality `m`, the target-safe positive history is

```text
h[u,-i,m] = normalize((sum(j in H[u]) c[j,m] - c[i,m]) / (|H[u]| - 1))
```

and evaluation uses the complete training history `h[u,m]`. Let `q[u,m]` be
the norm of the unnormalized mean (history coherence), `w[u,m]` its
temperature-softmax modality weight, and `a[u]` a bounded activity envelope.
The score correction is

```text
delta(u, i) = sum_m alpha[m] * a[u] * w[u,m] * q[u,m]
                        * cosine(h[u,m], c[i,m])
```

where `alpha[m] = max_scale * tanh(raw_alpha[m])`. `raw_alpha` starts at zero,
so epoch-zero predictions are exactly official MSCA, while the derivative is
non-zero and BPR can accept, reject, or reverse either modality correction.

## Distinction from nearby work

- MuSICRec constructs sequence nodes and a sequence-item graph, then aligns
  graph views. LCHR uses no new graph or temporal sequence encoder; its main
  mechanism is candidate-safe leave-one-out history scoring.
- MSCA fuses semantic graph views into embeddings. LCHR leaves that path
  unchanged and adds only a confidence-bounded score residual.
- Earlier PRAVI reliability variants routed or reweighted existing semantic
  objectives. LCHR instead creates an explicit user-history/candidate matching
  signal and prevents target self-inclusion.

Primary references:

- MuSICRec: https://arxiv.org/abs/2602.07208
- MSCA official implementation: https://github.com/recomall/MSCA

## First-stage protocol

- Carrier: official MSCA hyperparameters.
- Dataset/seed: Baby, 2026.
- Selection: validation Recall@20 only; test remains unread.
- Treatment: `max_scale=0.10`, `activity_prior=10`,
  `coherence_temperature=0.25`, `min_context=2`.
- Frozen control: validation R@10/R@20/N@10/N@20 =
  `0.0675/0.1043/0.0369/0.0462`.
- Go: R@20 and N@20 must both exceed control, with no material R@10/N@10
  regression.
- No-go: failure on either primary metric freezes LCHR and appends it to the
  failure ledger. No parameter grid is allowed after a clear no-go.

## Commands

```bash
cd /data/sunzixuan/PRAVI_remote/src
python ../tests/test_hcr.py
python run_pravi_experiment.py --model MSCA_HCR --dataset baby \
  --seed 2026 --gpu-id 0 --validation-only
```
