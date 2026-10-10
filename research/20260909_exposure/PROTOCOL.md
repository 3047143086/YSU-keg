# Temporal Exposure Feasibility Screen

This is a small internal feasibility screen on the existing Baby interaction
file and frozen `global_bank.npy`. It is not a novelty claim, an official
benchmark result, or independent confirmation.

## Split and leakage contract

For every user, interactions are sorted by `(timestamp, itemID)` using a stable
deterministic rule. All but the last two interactions are training history, the
penultimate interaction is calibration, and the last interaction is test. A
training example only uses the prefix before its positive item. Calibration uses
training history only. Test uses training history plus the already-observed
penultimate item. A target is never in its scoring context; duplicate targets
fail the run rather than being silently masked.

## Arms

`long_mean` summarizes all available context, `recent3` summarizes the last
three items, and `learned_two_scale` learns a soft mixture of both summaries.
All arms use the same frozen two-modality bank, the same sampled-negative
pairwise objective, full-catalog evaluation, and three seeds (2026, 2027,
2028). Negatives are sampled from items outside each example's context and
positive. Supports are masked only at evaluation time.

Calibration selects the checkpoint by NDCG@20, then Recall@20, then NDCG@10,
with earlier epoch as the deterministic tie-break. The selected checkpoint is
evaluated once on the final test target per user. Report Recall/NDCG at 10 and
20 and Recall@200, plus per-user arrays.

## Execution and interpretation

`run_pipeline.py` runs unit tests, real-data smoke jobs, and nine formal jobs
serially on GPU 1. `manage.py start` launches the queue detached on server 213.
The v2 queue waits up to 48 hours for GPU 1 because the prior queue's six-hour
resource wait expired before any formal job could run.
The raw interaction and feature paths are read-only. The remote output root is
separate from both source experiments. A completed result is evidence only that
the temporal screen is runnable and gives the observed metrics; it does not
establish generalization, exposure correction, or a publishable improvement.
