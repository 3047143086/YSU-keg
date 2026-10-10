# BRIDGE-CBCR Experiment

## Hypothesis

BRIDGE applies a behavior residual only within the detached base top-K set, but
its calibrated BPR loss gives sampled pairs equal importance. CBCR tests whether
the residual is more effective when trained as a local repair operator:

1. emphasize candidate pairs whose detached base margin is wrong or uncertain;
2. preserve confident base rankings with a correction-budget penalty;
3. leave the official retrieval backbone, behavior evidence, inference path, and
   primary BRIDGE losses unchanged.

For base margin `m_b`, candidate correction margin `d_c`, and active-candidate
mask `a`, CBCR uses the detached batch RMS `s` and

```text
w = a * sigmoid(-stopgrad(m_b) / (temperature * s))
L_repair = weighted_mean(w, softplus(-(stopgrad(m_b) + d_c) / s))
L_guard  = weighted_mean(a - w, (d_c / s)^2)
L_CBCR   = L_repair + guard_weight * L_guard
```

The experiment adds `cbcr_weight * L_CBCR` to the official BRIDGE objective.
Only the correction branch receives gradients from CBCR.

## First-stage protocol

- Dataset: Baby, existing fixed train/validation/test split.
- Features: identical existing image/text features for both models.
- Baseline: unmodified official BRIDGE configuration.
- Treatment: BRIDGE-CBCR with one predeclared setting (`0.2`, `1.0`, `0.05`).
- Seed: 2026 for the screening run.
- Selection: validation Recall@20; test is reported only at the selected epoch.
- Decision: continue to multi-seed and official BEIT3 features only if CBCR beats
  BRIDGE on validation without a material NDCG regression.
