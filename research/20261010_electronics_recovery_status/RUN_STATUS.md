# Electronics recovery v3: stopped at a spectral boundary

The independent numerical recovery at frozen source
`cb7bc6bfff811040f0499bf5ac5b07020bb5c4ba` started once on 2026-10-10.
It stopped after four finite optimizer updates: zero-based epoch 0, batch 4
raised `Unresolved spectral band boundary at index 43` during forward.
The guard blocked that batch's optimizer update and saved its actual inputs.
Parameters remained finite. No epoch or seed completed, no validation metric
was produced, and the test loader was not constructed or evaluated.
This is a failed numerical feasibility run, not an algorithm improvement.

The actual deployed MMRec environment first passed all 50 CPU tests, with
zero skips, errors or failures and CUDA uninitialized. Its raw test receipt
SHA256 is `8f82a9e2e375aafd8782767ee1b3b90862e82be3ada50c0b72f1b88436fe7a6e`;
the bound deployment SHA256 is
`d3ac7ec9ffb6a98ff205ea3f954e7215bbe9c004cf0f31cd5fa911b30ec36e68`.
The preparation test and deployed test are distinct receipts, both preserved.

The queue PID was `3151746`, with seed-2026 child PID `3152157`. The client
05:08:11 UTC snapshot checked both live command identities; its empty child log
only established startup. The client 05:09:58 UTC snapshot confirms both exited,
queue status `technical_failure`, child exit code 1 and elapsed 107.9842545
seconds. Sampled peak GPU memory was 14,518 MiB and sampled peak host RSS
6,137,392 KiB. These are sampled process measurements, not allocation bounds.
Seeds 2027–2030 were never started. Completed seeds: **0/5**.

Snapshot timestamps are the local client's capture clock; raw launch/failure
Unix timestamps originate on the server and differ from that clock. Their
offset has not been independently calibrated. Do not combine those clocks
to infer an absolute launch time or training duration. The 107.9842545-second
child duration uses its monotonic timer and does not depend on clock agreement.

The original budget was 24 hours per seed and 72 hours for the total queue,
with four CPU threads, physical GPU 1 and PyTorch memory fraction 0.55. Fresh
GPU and host-resource checks passed before launch. The queue stopped on the
first failed seed without retry. The original uncached evaluation path remained
configured; evaluation was not reached. The rejected evaluation cache was excluded.

The post-run guard recorded `all_match=true`, `v1_preserved=true`, the original
graph SHA256, all seven prerequisite roles, and 264 protected path-hash checks.
The frozen v3 source, its original v1 failure and earlier failed audits remain
unchanged. The raw `failure.json` records four completed finite steps, no bad
SVD backward, and a **forward** boundary error; do not describe this as another
observed nonfinite-gradient failure or as reproduction of the old crash.
No successful-run `numerics_receipt.json` or `result.json` exists.

| Exact raw receipt in `startup_v3/` | SHA256 |
| --- | --- |
| `pipeline_status.json` | `3b01c164fe28cb4e2a8f4ad1ff5be3891e6dde965eb076a8d3a10b48f625b65d` |
| `seeds/elec_projector_recovery_s2026/failure.json` | `390b9a80dbeaa291e344f7fe8658ce07a508627d3e379b2e77ea79913a7d989b` |
| `seeds/elec_projector_recovery_s2026/post_run_guard.json` | `b6e36489cc08365c66e2cd828acc82f5c79fe2b0d70ce1f0ff09a13c0c218fb3` |

Both [launch snapshot](launch_v3/capture_receipt.json) and
[failure snapshot](startup_v3/capture_receipt.json) record original remote
paths, byte lengths and SHA256 values. `runs/` receipts are archived locally
under `seeds/` so the public snapshot's raw-training-directory exclusion does
not hide these small audit files. Logs are clearly marked as tails rather
than complete raw files. The later capture also includes reservation receipts,
inputs and the post-run guard. Data, weights and the large tensor failure
packet remain in the experiment environment.

The separate [saved-boundary diagnostic](../20261010_recovery_boundary_diagnostic/AUDIT.md)
is now complete and frozen at `6fd840b62173b3f43fc75e319201df9dfc4768cb`.
It reads the saved packet on CPU, with no SVD recomputation or backward.
The sole rejected boundary is capture 11, index 43: stored squared-singular
gap 0.7442226918 versus fixed threshold 0.7574242599, about 1.743% below the
threshold. The stored singular values are distinct; this proves the declared
guard rejected them, not exact cross-band degeneracy or unexecuted VJP stability.
All 24 decisions and 264 protection hashes were independently recomputed or
checked from the compact raw receipt. Preserve the failed v3 and its fixed
threshold; any new numerical method needs a separate declared audit. Do not
repeat training under this version or use partial steps as baseline performance.
