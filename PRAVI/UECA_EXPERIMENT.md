# UECA: User-Equalized Collision-Free Alignment

## Status

Frozen after failing the Baby/seed-2026 validation-only screen. UECA obtained
R@10/R@20/N@10/N@20 = `0.0669/0.1035/0.0366/0.0459`, below the official MSCA
control `0.0675/0.1043/0.0369/0.0462` on all four metrics. Do not tune or repeat
this mechanism.

## Mechanism

MSCA samples interactions and reuses their user and positive-item IDs as rows
of six dense auxiliary InfoNCE terms. Consequently, repeated entities collide
inside a contrastive matrix and high-degree users enter more minibatches, while
Recall and NDCG assign equal weight to each evaluation user.

UECA changes only the auxiliary contrastive sample:

1. Deduplicate user and positive-item IDs before all modality and structural
   alignment terms.
2. Weight each unique user anchor by the inverse probability that the user is
   included in a shuffled interaction minibatch.
3. Leave item anchors unweighted after deduplication, preserving the observed
   item distribution in the first controlled screen.

For `N` training interactions, batch size `B`, and user degree `d_u`, the exact
without-replacement inclusion probability is

```text
p_u = 1 - C(N - d_u, B) / C(N, B)
w_u = 1 / p_u
```

The weighted mean is a Horvitz-Thompson-style correction: every user has the
same expected anchor contribution. There is no new trainable parameter or
validation-selected UECA hyperparameter.

## Frozen screen

- Carrier: official MSCA (WWW 2026).
- Data/seed: Baby, 2026.
- MSCA parameters: `n_layers=2`, `fusion_coeff=0.4`, `cl_weight=0.005`,
  `reg_weight=3e-7`.
- Selection: validation only; test remains unread.
- Control: R@10/R@20/N@10/N@20 = `0.0675/0.1043/0.0369/0.0462`.
- Go: both validation R@20 and N@20 exceed `0.1043/0.0462`, without a
  material regression on R@10 or N@10.
- No-go: freeze UECA and record it in `../result/FAILED_EXPERIMENTS.md`.

## Commands

```bash
cd /data/sunzixuan/PRAVI_remote
python tests/test_ueca.py

cd src
python run_pravi_experiment.py --model MSCA_UECA --dataset baby \
  --seed 2026 --gpu-id 0 --smoke --validation-only

python run_pravi_experiment.py --model MSCA_UECA --dataset baby \
  --seed 2026 --gpu-id 0 --validation-only
```
