# PRAVI Experiment

## Research question

Can virtual user-item interactions improve a current multimodal recommender
without propagating modality-specific noise?

## Innovation

PRAVI (Personalized Reliability-Aware Virtual Interactions) differs from
VI-MMRec in two material ways:

1. Modality reliability is estimated per user from training-only history
   reconstruction and shrunk toward the dataset mean for sparse users.
2. Candidate edges carry continuous lower-confidence-bound weights, with
   explicit abstention by posterior confidence and cross-modal support.

Virtual edges affect graph propagation only. Original interactions remain the
sole positives for BPR training and the sole mask for validation/test ranking.
Filtering virtual candidates from sampled BPR negatives is an optional,
separately ablated training effect.

The additive Beta-LCB pilot did not beat MSCA on Baby. The next mechanism
under test is DVT (Degree-preserving Virtual Transport). Given observed edges
`(u, i)` and `(v, j)`, DVT accepts a reciprocal pair of high-confidence
cross-edges `(u, j)` and `(v, i)` and forms the signed 2-switch

```text
Delta = w * [(u, j) + (v, i) - (u, i) - (v, j)].
```

Every involved user and item receives one positive and one negative term, so
row sums, column sums, node degrees, and total propagation mass are preserved
exactly. `Delta` is used only in the non-cascading first-order response branch;
the observed graph, BPR samples, and MSCA contrastive views remain unchanged.
This tests structural transport rather than another positive-edge expansion.

The novelty claim must stay narrow. Degree-preserving 2-switch rewiring is a
known general graph operation (for example, ResiNet/TMLR 2023); DVT does not
claim to invent it. The proposed contribution is to select reciprocal
bipartite switches from training-only multimodal evidence and inject their
signed, degree-preserving differential only through a non-cascading response
branch. Recent multimodal methods that add content-induced edges or route
between original and augmented graphs remain distinct comparison points.

## Primary comparison

- Baseline: MSCA (WWW 2026), MRLib implementation.
- Proposed: MSCA + PRAVI.
- Dataset: Amazon Baby first, then Sports if the primary comparison is sound.
- Metrics: Recall@10/20 and NDCG@10/20.
- Seeds: 2026 for the pilot; 2026, 42, and 999 for the final comparison.

## Ablations

- `--reliability-mode global`: dataset-level rather than personalized weights.
- `--pravi-kappa 0 --pravi-min-confidence 0`: posterior mean without the
  uncertainty penalty or confidence-threshold abstention.
- `--binary-weights`: retain selected edges but discard calibrated confidence.
- `--pravi-response-scale 0 --keep-virtual-negatives`: identity control; it
  must reproduce MSCA under the same seed and carrier hyperparameters.
- `--pravi-response-scale 0`: negative-filtering-only control.
- Additive first-order response versus signed DVT response.
- Confidence-selected DVT versus degree-matched random 2-switches.
- Full signed transport versus positive-cross-edge-only response.
- DVT with unit transport mass versus confidence-weighted transport mass.

## DVT pilot gate

Do not start full training unless reciprocal 2-switches cover at least 10% of
Baby users or produce at least 2,000 disjoint swaps. After that gate, run an
identity smoke test and one validation-only seed. Stop this branch if Recall@20
does not exceed 0.1043 or if NDCG@20 regresses materially.

## Commands

```bash
cd src
python run_pravi_experiment.py --model MSCA --dataset baby --gpu-id 0
python run_pravi_experiment.py --model MSCA_PRAVI --dataset baby --gpu-id 0
python run_pravi_experiment.py --model MSCA_PRAVI --dataset baby --gpu-id 0 --reliability-mode global
python run_pravi_experiment.py --model MSCA_PRAVI --dataset baby --gpu-id 0 --pravi-kappa 0 --pravi-min-confidence 0
python run_pravi_experiment.py --model MSCA_PRAVI --dataset baby --gpu-id 0 --pravi-min-confidence 0.005
python run_pravi_experiment.py --model MSCA_PRAVI --dataset baby --gpu-id 0 --binary-weights
```

No result should be reported as a gain until the baseline and proposed model
finish under identical data, seed, hyperparameters, stopping, and evaluation
settings.
