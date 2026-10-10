# Karpathy autoresearch adapted to the current BRIDGE experiment

User-selected upstream: https://github.com/karpathy/autoresearch,
commit `228791fb499afffb54b46200aca536f79142f117`.
The original repository has program.md instead of a standard SKILL.md. The
local Codex adapter is `skill/auto-research/SKILL.md`, invoked as `$auto-research`.
This adaptation is not an upstream-published skill. Source hashes and the
downloaded directory are recorded in skill-source.json.

Working directory: `research/20261008_autoresearch`.
Branch: `autoresearch/oct8-bridge-recovery`.

| Upstream example | Current experiment adaptation |
| --- | --- |
| Editable train.py with an LLM | Future candidate edits confined to proposal.py |
| Frozen prepare.py/evaluate_bpb | Frozen splits, prepare.py, train.py evaluator and protocol |
| Minimize val_bpb | Maximize validation Recall@20 with NDCG and shortlist guards |
| Five-minute training budget | Existing 300-epoch/patience-20 budget, eight-hour limit |
| results.tsv and keep/discard/crash | results.tsv with validation_recall20, plus full report JSON |
| Indefinite example loop | One bounded recovery/current-candidate evaluation |

## Agent loop

1. Read this file, PROTOCOL.md, prior results and the isolated Git state.
2. Refresh the server state. Finish the current baseline before evaluating any
   candidate improvement; a failed or partial baseline is unusable.
3. Complete the already-declared soft-isolation hypothesis at alpha=0.5. Do not
   introduce another algorithm while this fixed comparison is incomplete.
4. Refresh status and run `python karpathy_results.py`. It writes results.tsv
   and research_report.json, with keep/discard only after all controls finish.
5. Use validation R@20 as the primary metric, require every guard below, and
   preserve failed evidence. Scientific failure means discard; numerical failure
   means crash, not a negative scientific result. Keep the incumbent baseline.
6. Stop after this bounded comparison (and conditional Sports replication).
   Further hypotheses require a new isolated version and declared budget.

The current queue recovers a prior technical interruption: Baby soft-edge
completed; baseline failed with non-finite loss; history never started. The SVD
warning is a diagnostic lead, not a proven root cause. The number of completed
arms is only a progress indicator, never the optimization metric.

## Frozen scientific protocol

See `PROTOCOL.md`. Fixed alpha=0.5 and seed=2026. Relative R@20 and N@20 gain
must both reach 0.5%; absolute R@10/N@10 decline must not exceed 0.0002;
R@20/N@20 must both exceed history-only. Baby gates progression to Sports.
Validation selects checkpoints; test labels are not evaluated.

Up to 300 epochs, patience 20, batch 2048, four CPU threads. One GPU task at
a time, 14 GiB free required, 55% GPU memory limit. Each formal job has an
8-hour timeout; the full queue is bounded to 48 hours including resource waits.

## Recovery rules

- Source: `/data/sunzixuan/target_isolated_20260911_soft_v1`.
- Destination: `/data/sunzixuan/target_isolated_20261008_autoresearch_v1`.
- Reuse only the completed Baby edge arm after verifying data/code/config
  hashes, previously absent inputs, carrier snapshot, unchanged algorithm,
  full budget, alpha, evidence cache, checkpoint and per-user artifact hashes.
- Baseline/history are each retrained from scratch once. No changes to the
  seed, learning rate, batch size, numerical algorithm, or acceptance threshold.
- Recurrent numerical failure stops the queue as technical_failure. No silent
  retries, NaN sanitization, or historical-control substitution.
- Original files and outputs stay in the original directory. Source evidence
  is captured locally in previous_status.json; live recovery status is separate.

## Execution and monitoring

From this directory:

```powershell
python manage.py launch
python manage.py status --brief
python verify.py --report
python karpathy_results.py
python karpathy_results.py --metric baseline --dataset baby
```

Launch creates an isolated frozen bundle, verifies inputs, runs all remote
tests, and starts the queue. It refuses a duplicate launch. Status fetches
current evidence into latest_status.json. Reporting reads that dated snapshot;
the metric command exits with an error until a complete valid baseline exists.

The detached server pipeline continues this fixed queue after the conversation
ends. It does not invent new hypotheses or call a model in the background;
autoresearch guides the active Codex agent. A new scientific iteration needs a
subsequent agent turn and a new bounded configuration.

Remote records: pipeline_status.json, runs/*/result.json,
recovery_receipt.json, baby_screen.json, optional sports_screen.json,
autoresearch/loop-recovery/loop-results.tsv and handoff.json. These remote files
are the recovery infrastructure ledger created before source selection was
clarified. The Karpathy-style scientific ledger is local results.tsv, generated
from complete fetched evidence; it never reports recovery progress as a gain.
An isolated local Git repository tracks this study only.
