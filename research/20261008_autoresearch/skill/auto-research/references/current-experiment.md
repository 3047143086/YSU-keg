# BRIDGE experiment

Workspace: `C:/Users/zixua/Desktop/exp/exp/exp`.
Current adapter: `research/20261008_autoresearch/program.md`.
Upstream revision and source directory: `research/20261008_autoresearch/skill-source.json`.
Scientific protocol: `research/20261008_autoresearch/PROTOCOL.md`.

Use the existing server and Conda MMRec environment through manage.py; it reads
the workspace's connection guide without printing credentials. No LLM-demo
dataset or new training dependencies are needed for this adaptation.

From the workspace root:

```powershell
python research/20261008_autoresearch/manage.py status --brief
python research/20261008_autoresearch/verify.py --report
python research/20261008_autoresearch/karpathy_results.py
```

The existing recovery queue is
`/data/sunzixuan/target_isolated_20261008_autoresearch_v1`.
Never launch it twice or edit its frozen bundle. It reuses a verified completed
Baby soft-edge arm and retrains baseline/history once, then proceeds to Sports
only if Baby passes. Numerical failure stops the queue. The original interrupted
run is `/data/sunzixuan/target_isolated_20260911_soft_v1`.

Fixed alpha=0.5, seed=2026, maximum 300 epochs, patience 20. Primary metric:
validation Recall@20 (higher is better), with NDCG@20 and R@10/N@10 gates in
PROTOCOL.md. A valid paired baseline is required before candidate acceptance.
The number of completed jobs is progress, not the scientific objective.

Only a future isolated candidate may edit proposal.py. Evaluator, splits,
prepare.py, evidence definitions and thresholds remain frozen for comparisons.
Do not infer new unbounded optimization from the upstream example's wording.
