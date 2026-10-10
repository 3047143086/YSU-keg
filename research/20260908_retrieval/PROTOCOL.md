# Baby Retrieval Screen

Date: 2026-09-08. This is a bounded development experiment, not a claim of
improvement, novelty, or publication readiness.

Active revision: `/data/sunzixuan/rcmqr_20260908_screen_v2`.
The v1 attempt stopped during the real-data base smoke test before training,
because legacy RecDataset did not initialize inter_num without __str__.
The runner now sets that loader-required field explicitly. The failed v1
directory and logs are retained; no v1 performance result exists.

## Data and isolation

Reuse the existing Baby frozen CLIP ViT-B/16 global image/text features and
training-internal support/held-out split from `rave_20260906_stage1`.
No data or model download is scheduled. No official validation/test is used.
The old internal development split has been inspected historically; this is
not a fresh independent confirmation set.

Split eligible users deterministically (seed 20260908) into retriever training
(60%), calibration (20%), and internal development (20%). Every target edge
is absent from the base graph, support history, and co-occurrence tables.
Base BRIDGE models are retrained per seed and checkpoint-selected ONLY on
calibration users. Old globally-selected checkpoints are not reused.

Important label-budget limitation: the support-only base does not train on
role0 held targets, whereas every learned content control does. Therefore
improvement over the raw base alone is NOT an equal-label method comparison.
The mechanism claim must compare conditional against independent/hard and
other content controls trained on exactly the same role0 labels. A later
publication experiment needs a full-data, equal-label BRIDGE comparison.

## Jobs and controls

Seeds: 2026, 2027, 2028. Per seed: one new BRIDGE base and seven retrieval
controls: independent, hard-positive weighting, CLEAR-style adaptation,
conditional candidate summary, matched shuffled summary, base-user vector,
and simple query feedback. CLEAR is an adaptation, not its original IR task.

For learned variants, item features and network capacity are controlled;
conditional and hard controls use the same hard-positive emphasis.
Final candidate pools always contain 200 distinct nonhistory items:
retain the first 180 or 150 base candidates and retrieve 20 or 50 items from
outside the original base top-200. Quota is selected on calibration only.
This strict forced-replacement policy is a screen, not an optimal learned
fusion claim. Report recovered and displaced positives separately.

The independent content model publishes a shared scoring artifact. All
candidate pools are reranked by the same base/content score combination,
with mixing calibrated on role1, then frozen. Report both the raw base and
base-candidate/shared-ranker controls, plus direct full-catalog ranking by
the same fused score. This lightweight scoring screen is
narrower than the proposal's eventual learned union-trained reranker.

Checkpoint selection never uses role2. Evaluate role2 after configuration
selection, with Recall/NDCG@10/20, candidate Recall@200, and per-user deltas.
No performance number from this internal split is comparable directly to
old official-validation or different-feature results.

## Runtime and outputs

Use existing MMRec environment on server 213. GPU1 only, one new job at a
time, launch under <=6000 MB existing memory use and <=40% utilization.
PyTorch memory fraction capped at 0.25; four CPU threads. No unrelated
process is terminated. Maximum queue elapsed time 12 hours, individual job
2 hours; resource waiting counts toward the queue deadline.

Tests and base smoke run before training. Each job writes stdout.log,
progress.json, result.json, checkpoints and per-user predictions/metrics.
Queue writes pipeline_status.json, summary.json and RESULTS.md atomically.
An explicit failed job is retained, never silently reported as completed.
Pipeline runs detached from SSH; start receipt prevents duplicate launch.

## Interpretation

First inspect candidate net coverage. Then require conditional summaries to
beat equal-capacity hard-positive and shuffled-summary controls. Finally
check whether gains reach shared-ranker Recall@20 and NDCG@20.
If those comparisons fail, do not claim a new effective mechanism. Three
seeds are a screen; a positive outcome still needs independent confirmation,
additional datasets, stronger fusion controls, and bounded tuning budgets.
Sports is not automatically launched by this screen.
