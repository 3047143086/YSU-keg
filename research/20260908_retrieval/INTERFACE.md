# Internal implementation contract

Remote root: `/data/sunzixuan/rcmqr_20260908_screen_v2`.
Python: `/data/sunzixuan/miniconda3/envs/MMRec/bin/python` (PyTorch 1.12.1).
Source stage1: `/data/sunzixuan/rave_20260906_stage1`.

Root preparation supplies a read-only `features/` link, a copied `bridge_src/` snapshot, and
`split/baby.inter` copied from the existing training-internal stage1 split.
It also supplies `split_manifest.json` and `roles.npy` indexed by user ID.
Roles are fixed across seeds: 0 retriever training (60%), 1 calibration (20%),
2 final internal development (20%). All role targets are absent from support.
Roles outside eligible users are -1. Official validation/test are never used.

Base program: `train_base.py --seed S --gpu-id G --output DIR [--smoke]`.
Normal DIR is `prepared/S`; it must emit `result.json` on completion with:

- `base_scores.npy`: float32 [n_users,n_items], full BRIDGE final scores before
  support masking. Finite values; prediction user order equals raw user ID.
- `base_top200.npy`: int64 [n_users,200], support-masked final ranking.
- `base_user.npy`: float32 [n_users,embedding_dim], native base user embedding.
- `history.npy`: float32 [n_users,1024], normalized support-only content mean.
- `support_mask.npy`: bool [n_users,n_items], support interactions only.
- `targets.npy`: int64 [n_users], the single held-out target, or -1.
- `roles.npy`: copied role vector, identical for every seed.

Base checkpoint selected ONLY by role1 NDCG@20. No role2 evaluation during
training. Original model graph and caches derive only from support edges.
Base also records configuration, source hashes, checkpoint, progress and logs.
In smoke mode use two train minibatches and one calibration evaluation batch;
write smoke result, not a performance claim; skip full export.

Retrieval program: `train_retrieval.py --seed S --gpu-id G --output DIR
--variant VAR [--smoke]`. DIR is `runs/S/VAR`; input is `prepared/S`.
The independent variant runs first and produces an atomic shared ranker/scorer
artifact, fixed for all other variants. Every learned variant uses identical
input features and parameter capacity. Context-free control uses its own
history as second input. All variants train only on role0 targets, select
checkpoints and quota only on role1, then evaluate role2 once.

Each program writes `progress.json`, `result.json`, and `failure.json` on error.
Result exposes `status`, `variant`, `seed`, selected budget, calibration/dev
metrics, and raw per-user results. Do not label internal development as test.
The root queue reads generic JSON results; it does not assume a metric schema.

Only GPU1 is eligible initially; it has existing low-memory work. Launch one
new job at a time only when existing use is <=6000 MB and utilization <=40%.
Each job caps PyTorch allocations at 25% of GPU memory, CPU threads at 4.
Never stop unrelated jobs. The entire queue has a 12-hour wall-clock limit.
All code must support Python 3.8/PyTorch 1.12 and avoid new dependencies.
