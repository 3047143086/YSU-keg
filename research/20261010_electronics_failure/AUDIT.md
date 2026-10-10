# Electronics original baseline v1 failure audit

This is a read-only audit collected on 2026-10-10 at 09:54 Beijing time from
`/data/sunzixuan/electronics_baseline_20261009_v1`. No source, configuration,
checkpoint or remote run was modified. No training, forward, optimizer update,
solver substitution or GPU operation was executed by these collectors.

## Exact failure

Seed 2026 exited with code 1 after 13,723.918316 seconds (3.812 hours), before
the six-hour job budget. At 00:28:00 Beijing time, zero-based epoch 9, batch 225,
`FiniteTrainingGuard` observed non-finite gradients and blocked
`optimizer.step`. The child traceback and `failure.json` agree on
`FloatingPointError: Nonfinite gradients blocked before optimizer.step`.

At failure, model parameters remained finite and 5,742 finite updates had
completed. Non-finite parameter gradients were `weight_u`, `id_preference`, and
`item_id_embedding.weight`. SVD backward call 0 had finite upstream gradients
for U, S and Vh, but its input gradient of shape `[192403, 64]` contained
12,313,792 non-finite elements, the entire tensor. No invalid update was skipped
or repaired and then resumed.

Nine epochs, indexed 0 through 8, completed. The last saved intermediate
validation scores were rounded Recall@20 0.0643 and NDCG@20 0.0283. These are
early checkpoint values, not a completed or converged baseline result.
`result.json` and `numerics_receipt.json` do not exist. `results.tsv` records
2026 as `crash` and 2027–2030 as `pending`; zero seeds completed. The queue
exited at 00:28:45 Beijing time.

## SVD replay evidence

The saved replay contains twelve SVD captures. Every singular-value vector and
all saved upstream gradients for U, S and Vh are finite. Only call 0 has repeated
singular values: zero-based `S[48] == S[49] == 24.078458786010742`, with exactly
one zero adjacent gap. The other eleven captures have no equal adjacent values.

| Call | Input shape | Minimum adjacent gap | Equal adjacent values |
| --- | --- | ---: | ---: |
| 0 | 192403 × 64 | 0 | 1 |
| 1 | 192403 × 64 | 0.004792213439941406 | 0 |
| 2 | 192403 × 64 | 0.0040149688720703125 | 0 |
| 3 | 63001 × 64 | 0.002979278564453125 | 0 |
| 4 | 63001 × 64 | 0.037131309509277344 | 0 |
| 5 | 63001 × 64 | 0.013025283813476562 | 0 |
| 6 | 192403 × 64 | 0.00018644332885742188 | 0 |
| 7 | 192403 × 64 | 0.000041365623474121094 | 0 |
| 8 | 192403 × 64 | 0.00012129545211791992 | 0 |
| 9 | 63001 × 64 | 0.008684158325195312 | 0 |
| 10 | 63001 × 64 | 0.008950233459472656 | 0 |
| 11 | 63001 × 64 | 0.01236724853515625 | 0 |

The actual configuration is `num_freq_bands=3`, `freq_decomp_method='svd'`.
Frozen `models/dual_frequency_encoder.py` lines 919–922 split 64 dimensions into
sizes `[22, 21, 21]`; lines 924–969 reconstruct bands with
`U_i @ diag(S_i) @ V_i`. Bands cover `[0,22)`, `[22,43)`, `[43,64)`.
The equal singular pair 48/49 lies inside the third band. Boundary gaps are
`S[21]-S[22]=0.11014556884765625` and
`S[42]-S[43]=0.031137466430664062`, both positive.

Inference: rotations within this repeated singular subspace should cancel in
the band reconstruction, while a generic SVD backward can encounter zero
singular-value denominators. The recorded failure is consistent with this
numerical failure inside a band; it does not establish a discontinuity at a
band boundary. No replacement backward has been tested by this audit. Adding
jitter, selecting a different decomposition, detaching a basis, clipping or
skipping gradients must not silently be labeled the unchanged original baseline.

## Checkpoint lineage

All paths below are within `v1/runs/elec_baseline_s2026/`.

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| best.pt | 3954730163 | 3f9283407ff51178549c2f78e0f8cf61f895c269164afa6b3b9a0a56d8faacda |
| best_rng.pt | 22698151 | 240f546252241d0b553d2bcb1a25d20df9c98b4bc34df27b5b3ea251f4132725 |
| failure_replay.pt | 4739607753 | 7b6f12731715c0315018f6b3c903b39a8829cc7727035e18eb92eeaa4bd3deb7 |

CPU-loaded `best.pt` contains model, optimizer, scheduler, configuration, epoch,
selection score, validation scores and input hashes. Epoch is 8. Model and
optimizer states are all finite. Its configuration matches the recorded
effective run configuration exactly after standard JSON normalization, and its
input hashes match `inputs.json` exactly. Seed is 2026, dataset elec, maximum
epochs 300, batch size 2048, learning rate 0.0001. The RNG packet also has epoch
8 and contains Python/NumPy/Torch/CUDA RNG states and the matching loader
dataframe of 1,254,441 rows; `test_evaluated=false`.

Bundle commit is `06bcd212af31cd7c7ccd6ccc63de9778130b9f6b`.
`inputs.json` SHA256 is
`18dd014949737077a3ce63e004e2bddd82d1870f6caf430f43732b35db3d9b2e`;
`config.json` SHA256 is
`a8d27ee50f98cb6b8c89acf0a59c9d42846ffbcfbd85f34b8d864f8b95dcb875`.

## Frozen-file and process checks

Every audited hash matched: 12 deployed bundle artifacts, 35 deployed carrier
files, 35 source carrier files, 3 recovered original input files and 19 run
input files. Both absent optional inputs, user embeddings and user graph,
remain absent. The original graph preparation completed successfully and the
graph remains part of the matching run inputs. The checkpoint and failure
records claim no test evaluation; configuration records no test loader.

At collection, queue PID 2933472 and child PID 2936814 were absent. GPU 1 used
26 MiB and reported 24,232 MiB free, utilization 0%; only unrelated PID 600045
was reported on GPU 0, using 22,704 MiB. These are timestamped observations and
must be refreshed before any future GPU launch. Historical sampled peak GPU
memory for the failed seed was 14,046 MiB.

This is a baseline numerical/technical failure. It is neither a positive nor
negative innovation result and cannot support a paper result table as a
completed baseline.

## Reproduction receipts

- `remote_receipt.json`: original statuses, full small JSON artifacts and log
  tails with remote file hashes, process checks and all frozen hash comparisons.
- `replay_metadata.json`: full singular values and adjacent gaps from all twelve
  replay captures, gradient finite flags, source excerpts and large file hashes.
- `checkpoint_metadata.json`: CPU-only checkpoint and RNG lineage, finite-state
  verification and configuration/input equality results.
- `collect.py`, `replay_meta.py`, `checkpoint_meta.py`: read-only collectors using
  the existing connection helper; no credentials are included in receipts.
