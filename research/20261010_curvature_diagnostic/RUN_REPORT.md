# MSCA block-curvature diagnostic

The frozen read-only diagnostic completed on GPU1 in 41.36 seconds at source
commit `ebc83c9d6a2202e27fdb1bc44990e8f5d27db9cc`. It used three private
2,048-row Baby TRAIN batches, two model states, two private transform/gate
blocks, and four directions per block. The private batch archive remains on
the remote run root; its receipt records only shape, hashes, and TRAIN-only
eligibility assertions.

Execution/source records: audit source SHA256
`43ac53fba3a8402559066ac1d7a1c58316a1d86e315e54e062e0a2578ac4a336`;
protocol SHA256
`9ba89146d47193bf7363df1e5a780bac88c029028584dcce4db179769108b6aa`;
exact remote result SHA256
`e9b182a43f7a3966b3812df9330397c1f9501169770b09364ae4cd17171ba920`.
The local execution receipt confirms the audit source SHA was unchanged after
the run. The 48 probes supply 240 component rows in the TSV. Pre/post manifest
bytes have identical SHA256
`cbfd35f568a3e3b1415eb347790f01f14b1f2bd4c49c3aaff95df60f08cc34cd`.

All 48 probes were finite. Every perturbed block was restored bitwise, graph
tensors and global RNG states were unchanged, and all 170 protected source,
feature, split, and cache hashes matched before and after the run. No optimizer
was instantiated or stepped, and no validation or test loader/evaluator was
constructed during the diagnostic. The second state is the already
validation-selected uniform checkpoint at epoch 44; it was not reselected.

The fixed dominance gate failed. At fresh initialization neither block won any
of the three batches in the required four-direction comparisons. At the
uniform checkpoint the image block won only the full-gradient direction in two
of three batches; it did not win at least three of four directions in either
batch. The text block had no valid wins. Therefore no block is declared
dominant, no curvature precondition is established, and no recommendation
training or performance claim follows.

| State | Image direction wins per batch | Image gradient wins | Text direction wins per batch | Gate |
| --- | --- | ---: | --- | --- |
| init | 0, 0, 0 | 0/3 | 0, 0, 0 | fail |
| uniform_best | 0, 1, 1 | 2/3 | 0, 0, 0 | fail |

The signed per-probe BPR, full-loss, weighted regularizer, raw parameter norm,
radius-normalized curvature, fixed numerical floors, restoration jitter and
gradient alignments are in `result.json` and `probes.tsv`. The regularizer is a
decomposition only; it was not substituted for BPR or the full objective.
`post_run_inventory.json` confirms the diagnostic process exited and GPU1 has
no compute process after completion.
The actual child PID was 3265836; the post-run live inventory recorded it exited,
GPU1 free memory 24,232 MiB, utilization 0, and no GPU1 compute process.

The diagnostic is inconclusive under its predeclared gate rather than evidence
of a positive or negative recommendation gain. No follow-up optimizer run was
started automatically.
Signals below the fixed float32 floor remain inconclusive; they do not prove
that true curvature is absent.
