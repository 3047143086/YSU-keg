# Program contract

Before launch, commit this protocol and verify GPU1 is free. Launch the
existing frozen `train.py` with `--dataset baby --mode uniform` and a fresh
output directory on host 213. Keep `CUDA_VISIBLE_DEVICES=1`, the official
four-thread setting, and the existing validation-only stopping rule. Capture
the parent PID, child PID, command identity, start time, and remote log path.

On completion, fetch only aggregate JSON/text receipts and append the result to
the public snapshot. Do not copy private user/item batches, credentials,
weights, or raw logs. Do not start a second run or alter the failed candidate
protocols based on this baseline.

