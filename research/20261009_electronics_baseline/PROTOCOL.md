# Electronics Original BRIDGE Validation Baseline

Frozen 2026-10-09 before graph preparation or training. This prerequisite
adds the third original baseline dataset; it is not an innovation experiment.

Use only the recovered, SHA256-audited original Electronics interactions and
image/text features under /data/sunzixuan/bridge_extra_assets_20261009/elec.
Create an independent carrier/data/elec directory with individual read-only
input symlinks. Generate mm_adj_10.pt only in that independent directory.
Do not link the whole recovered dataset directory or change official assets.

Snapshot exactly the thirty-five carrier source/configuration files frozen
by candidate_gate_20261009_v1. Reuse its train.py with the sole change adding
elec to the accepted dataset choices. Preserve its numerical observation
wrapper. No model algorithm, solver, sparse normalization, feature, training
batch, learning rate, seed, or evaluator is changed.

The frozen encoder already computes exact row-block torch.mm cosine followed
by torch.topk, default block size4096. Use that existing implementation;
do not add approximate neighbors, feature selection, a new tie rule, or a
new graph constructor. CPU small-graph checks must compare dense and chunked
results, including duplicate and zero vectors. Record source/input hashes,
dtype, device, k=10, block size4096, image weight0.1, graph hash and graph
indices/values hashes. Preserve the original unweighted adjacency followed
by its row-degree symmetric normalization and self-neighbor behavior.

Electronics has192403 users and63001 items, so the original threshold1e9
selects its existing sparse behavior backend. Do not force dense behavior.
Sparse pair scores and sparse candidate score normalization are preserved
as written; they are not claimed numerically identical to the small-domain
dense backend. Audit TRAIN degree-square and sparse co-occurrence resource
requirements before starting. Solver or finite failure stops the queue;
no automatic retry or numerical repair is allowed.

Train exactly seeds2026,2027,2028,2029,2030 in this order with fresh models.
Each has at most300 epochs, patience20 (stop after21 unimproved rounds),
training batch2048, evaluation batch64, four CPU threads, original Electronics
hyperparameters, original Adam and SVD, and validation every epoch. Select
strictly improved rounded-four-decimal validation Recall@20, keeping the
earlier checkpoint on ties, then recompute selected-checkpoint validation.
Keep all seed outcomes, crashes and timing. No test loader or test evaluation
is constructed. Validation-user bootstrap is distinct from five-seed
training variability. These baseline validation results alone establish no
paper-level improvement and cannot be used to select a seed.

One own job at a time, only physical GPU1 and fraction0.55; require at least
14GiB GPU free,12GiB host RAM and four available CPU cores. Wait until the
review_residual_20261009_v1 launch receipt exists and its exact experiment.py
process has exited with a final diagnosis. Missing predecessor receipts do
not mean completion. Stop on a predecessor technical failure until reported.
Never kill unrelated processes or use GPU0. Graph preparation precedes seeds
and follows the same predecessor/resource gate. Queue budget72 hours including
all waiting; graph preparation and each training job have a6-hour limit.

Remote isolated root: /data/sunzixuan/electronics_baseline_20261009_v1.
Freeze local source revision, audited inputs, deployment and passing actual
MMRec test receipts before a detached queue launch. Record queue and own
child PIDs, process-command checks, stages, progress, peak sampled GPU memory,
per-seed metrics, original input hashes and completion or technical failure.
