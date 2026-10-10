# Multimodal Recommendation Validation Protocol

## Frozen baseline set

1. MSCA (WWW 2026): official carrier and primary controlled baseline.
2. VI-MMRec-O and VI-MMRec-S (KDD 2026): direct virtual-interaction competitors.
3. IIMRec (ACM MM 2026): closest graph-expansion and residual-gating method.
4. MAGNET (2026): closest original/augmented dual-view propagation method.
5. SSR (NeurIPS 2025) and SG-URInit (SIGIR 2026): recent overall references.

Official code and preprocessing are required where available. Any formula-level
reimplementation must be labeled as such and cannot be reported as an official
result.

## Selection protocol

- Hyperparameters are selected using validation Recall@20 only.
- Tuning runs use `--validation-only`; the test split is evaluated once after
  the full configuration is frozen.
- Primary metrics: Recall@10/20 and NDCG@10/20.
- Main evidence: Baby, Sports, and Electronics with at least five fixed seeds.
- Report mean, standard deviation, paired significance, training cost, graph
  mass, edge coverage, and user-activity strata.

## Go/no-go rule

The mechanism is supported only if it improves over both official MSCA and the
strongest VI-MMRec variant consistently across datasets and seeds, with no
material regression on NDCG. A single-seed tie or a test-only gain is failure,
not evidence.

## Current evidence

- MSCA Baby seed 2026 validation Recall@20: 0.1043.
- First-order Beta-LCB response (beta=8): 0.1041.
- Formula-level VI-MMRec-O reconstruction: validation Recall@20 0.0915.
- Formula-level VI-MMRec-S reconstruction: validation Recall@20 0.0967.
- These VI values are reconstruction results, not official implementation
  results, and cannot be labeled as official baseline performance.
- The additive PRAVI, DVT, adaptive reliability, consensus, and SDA families
  have not passed the go/no-go rule and are frozen.
- LCHR failed its frozen Baby/seed-2026 gate (`0.1038/0.0461` validation
  R@20/N@20) and is frozen without a parameter sweep.
- UECA failed its frozen Baby/seed-2026 gate (`0.1035/0.0459` validation
  R@20/N@20 versus `0.1043/0.0462`) and is frozen without a parameter sweep.
- RAPA failed its 36-probe preregistered conflict diagnostic and is frozen.
- The rating-residual multimodal anchor failed four of five preregistered
  checks; its fixed correction significantly reduced pairwise validation AUC
  and is frozen.
- Next baseline calibration: reproduce SG-URInit's implicit exposure adapter on
  official MSCA. This is a stronger 2026 baseline candidate, not an innovation.
- The authoritative failure memory is `FAILED_EXPERIMENTS.md`; consult it before
  implementing or scheduling any new mechanism.
