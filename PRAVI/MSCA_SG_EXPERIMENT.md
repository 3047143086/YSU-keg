# MSCA + SG-URInit Baseline Calibration

## Status

Implementation ready; Baby/seed-2026 validation-only screen completed. This is
a 2026 baseline composition, not the proposed paper innovation.

## Result (2026-09-03)

The fixed screen reached validation R@10/R@20/N@10/N@20 of
`0.0668/0.1031/0.0365/0.0457` (best epoch 39), versus official MSCA
`0.0675/0.1043/0.0369/0.0462`. It therefore missed the frozen gate and is
recorded as F13 in `result/FAILED_EXPERIMENTS.md`. Test evaluation was
intentionally disabled for this screening run.

## Why this baseline is next

SG-URInit (SIGIR 2026) reports consistent gains from inverse-item-degree local
and cluster-level global semantic user representations. Our preregistered
rating diagnostic independently found that the unsigned fused exposure profile
has validation pairwise AUC `0.5992`, while rating polarity failed. Establishing
this stronger user-semantic carrier is therefore more defensible than inventing
another weighting or gating function on official MSCA.

## Adapter boundary

The official SG-URInit release does not include MSCA. This adapter reproduces
its preprocessing equations and its released graph-model integration pattern:

1. For each modality, aggregate interacted item features with inverse item
   degree (SG-URInit Eq. 1).
2. Run deterministic K-Means and replace each item by its nearest centroid,
   then aggregate those centroids with the same weights (Eq. 2--4).
3. Fuse local and cluster profiles using Eq. 5.
4. Apply MSCA's modality projection to the user profile and positive/negative
   item features, adding the released adapter's modality-wise BPR supervision.

MSCA's graphs, forward representation, main BPR loss, six contrastive terms,
regularization, and inference scorer are unchanged. Profiles use training
interactions only and are cached by graph/feature signature.

## Frozen first screen

- Carrier: official MSCA (WWW 2026).
- Baseline method: SG-URInit (SIGIR 2026), formula-level MSCA adapter.
- Dataset/seed: Baby, 2026.
- MSCA: `n_layers=2`, `fusion_coeff=0.4`, `cl_weight=0.005`,
  `reg_weight=3e-7`.
- SG-URInit: `K=4`, cluster weight `0.1`, K-Means random state `0`, ten
  initializations.
- Semantic BPR weight: `0.001`, fixed from the released graph-adapter scale.
- Selection: validation only; do not evaluate test during screening.
- Frozen MSCA validation R@10/R@20/N@10/N@20:
  `0.0675/0.1043/0.0369/0.0462`.
- Go: both R@20 and N@20 must exceed `0.1043/0.0462`, with no material R@10
  or N@10 regression.

If the screen misses, do not call the adapter an innovation. A later official
baseline reproduction may tune the paper's released grid, but that is separate
from innovation search and cannot retroactively change this fixed screen.

## Commands

```bash
cd /data/sunzixuan/PRAVI_remote
/data/sunzixuan/miniconda3/envs/MMRec/bin/python \
  -m unittest tests/test_sg_urinit.py

cd src
/data/sunzixuan/miniconda3/envs/MMRec/bin/python run_pravi_experiment.py \
  --model MSCA_SG --dataset baby --seed 2026 --gpu-id 0 \
  --smoke --validation-only

/data/sunzixuan/miniconda3/envs/MMRec/bin/python run_pravi_experiment.py \
  --model MSCA_SG --dataset baby --seed 2026 --gpu-id 0 \
  --validation-only
```
