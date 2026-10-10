# Stage-One Experiment

Launched task: frozen-encoder global versus local content, Baby, three seeds.
This is the prerequisite information screen for RAVE, not the full acquisition
policy and not a claim of improved test performance.

## Fixed Comparisons

- `global`: BRIDGE plus a user-conditioned matcher over two pooled modality vectors.
- `local`: the same backbone and matcher with the constituent image/text vectors.
- `shuffled`: the same local matcher with local vectors shuffled within identical
  missingness patterns. Correct global vectors remain available.

All groups receive the same CLIP ViT-B/16 features, model size, optimizer budget,
seeds 2026/2027/2028, and fixed internal split. Global vectors pool exactly the
same five image views and up to eight 75-token text chunks available to local.
No original catalog items are removed for missing images or text. Features,
source code, input split, and checkpoints have recorded SHA256 fingerprints.

Only original training interactions enter the experiment. One interaction per
eligible user is held out for internal development, with at least one history
item retained. This development set selects epochs, so gains here require an
independent confirmation later. Official validation and test labels are never
passed to training or evaluation. The original datasets and model code stay intact.

Training uses the existing BRIDGE score hooks and trainer epoch implementation;
the existing fit loop is not called because it evaluates test data each epoch.
The content profile removes the training target for both pairwise alternatives.

## Bounds And Outputs

- Maximum 200 epochs, development evaluation every five epochs plus epoch one.
- Early stop after eight evaluations without improving development NDCG@20.
- Two idle GPUs at most, one run per GPU; busy GPUs are not commandeered.
- Two-hour limit per run; eighteen-hour limit for the entire pipeline.
- The pipeline records failures and continues other training runs after a run fails.
- `RESULTS.md` and `summary.json` update automatically during training.
- `pipeline_status.json` reports the current stage, active PIDs and queue.
- `runs/<variant>_s<seed>/` contains loss curves, best checkpoints and per-user metrics.

The descriptive screening target is at least 2% relative mean gain on both
Recall@20 and NDCG@20, a positive difference at every seed, better than shuffled
content, and no mean Recall@10/NDCG@10 regression. These are acceptance criteria,
not promised results. No policy training or test-set search is triggered automatically.
