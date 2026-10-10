# Failure-model GPU audit: completed_fail

Frozen source `fd42959e0a97f24169196851cd6d291477b31c59`, remote
`/data/sunzixuan/failure_model_gpu_audit_20261010_v1`. Physical GPU1,
Torch1.12.1+cu113, four CPU threads, fixed thirty-minute budget. Actual
gate tests6/6 passed. PID3124882 exited after101.908757s. No optimizer
constructed, no update, no validation/test loop or recovery training.

## Archived inputs: a limited positive numerical result

All12 actual archived float32 inputs were checked on the GPU. All36
original versus projector forward bands are exactly equal. All12 repaired
independent-probe gradients are finite. Capture0 reproduces a nonfinite
generic GPU gradient while the repaired gradient is finite; its generic
comparison is unavailable, not an equivalence pass. The11 remaining finite
generic gradient comparisons pass their fixed thresholds. These use
independent dL/dBand probes, not the original training loss gradient.

## Full model: fails the fixed conjunction

The full failure-packet model was strict-loaded with verified preference
aliases, original configuration, seed/order and RNG restored before both
loss calls. The original and repaired probes each have27 participating
parameters, all finite gradients. All27 gradient comparisons pass. Losses
0.09115199744701385 versus0.09115228056907654 pass the frozen loss tolerance;
state_dict is unchanged. Protected source/data/packet hashes match before
and after.

However, none of the12 original-run internal inputs matches the historical
archive exactly, and inputs of the two current model runs differ slightly.
At call0 their relative input difference is4.49555e-8, maxabs1.19209e-7,
while the three band relative errors are6.61514e-5,9.56081e-4,9.94640e-4.
Six calls (0,1,2,3,5,9) contain bands exceeding the fixed5e-5 relative gate.
Therefore `candidate_integration_passed=false`,
`exact_historical_crash_reproduced=false`, and status is `completed_fail`.
Do not loosen the gate, call this exact historical replay, or start a
recovery training queue from this receipt.

A separate original-versus-original repeated-forward diagnostic is the next
step to measure whether the frozen original model itself exhibits upstream
GPU variability. This is a proposed explanation, not yet an established
cause. The original failure and this audit remain immutable.

Full evidence: [gpu_audit_receipt.json](gpu_audit_receipt.json).
