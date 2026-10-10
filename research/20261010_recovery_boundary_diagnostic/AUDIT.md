# Electronics recovery v3: saved-boundary failure diagnosis

The frozen v3 run failed its declared numerical guard. Its saved float32
singular values at the failing boundary are strictly different; the computed
Gram eigenvalue gap is positive but falls below the fixed global-scale
separation threshold. This is evidence of a guard rejection, not evidence
that an exact cross-band repeated singular value made the mapping undefined.
The run remains a technical failure. No threshold, implementation, training
state, prior receipt or result was changed, and no recovery job was restarted.

## Run and immutable evidence

The deployed source was `cb7bc6bfff811040f0499bf5ac5b07020bb5c4ba`, with a
clean source snapshot. Local startup raw files were verified against every
SHA/size in `research/20261010_electronics_recovery_status/startup_v3/capture_receipt.json`.
The frozen projector, recovery guard, numerical guard, training wrapper and
integration helper match both the deployment manifest and that Git revision.

Queue PID3151746 and seed2026 child PID3152157 exited. The archived child
duration is107.98425449803472 seconds from the queue's monotonic timer.
The failure record reports zero-based epoch0/batch4, four completed finite
optimizer steps, stage `forward`, finite parameters, no nonfinite parameter
gradients, and no bad SVD backward events. The failed fifth batch was blocked
before backward/optimizer update. No complete epoch or validation metric was
produced: `progress.jsonl` has zero bytes; no result is present in the startup
archive, and the remote `numerics_receipt.json` does not exist. The frozen
wrapper writes that success receipt only after `train.main()` returns.
Seed2026 is `crash`;2027–2030 are `not_started`; completed seeds are empty.

The saved packet contains all12 actual decomposition inputs and their original
CUDA-computed float32 singular values from the failed forward. All inputs and
all singular values are finite; all12 have zero exactly equal adjacent stored
singular values. The failure arose in capture11, the text component of the
item graph representation `h`, shape[63001,64]. The first11 calls and the
first boundary of call11 passed; only the final boundary failed.

## Exact frozen guard arithmetic

With64 features and3 bands the sizes are[22,21,21]. Boundary43 separates
zero-based singular indices42 and43, i.e. the second and third bands.
The original saved float32 singular values were squared in float64 exactly as
the frozen guard does. No SVD, model forward or backward was recomputed.

| Quantity | Saved/derived value |
| --- | --- |
| s[42] |5.394864559173584 |
| s[43] |5.325442790985107 |
| s[42]−s[43] |0.06942176818847656 |
| λ[42]=s[42]² |29.10456361182719 |
| λ[43]=s[43]² |28.36034092005525 |
| Δλ |0.7442226917719381 |
| s[0] |445.59423828125 |
| max λ |198554.2251894474 |
| eps(float32) |1.1920928955078125e−7 |
| Threshold32×eps×max λ |0.7574242599084755 |
| Δλ/max λ |3.7482087881124145e−6 |
| Required relative gap32×eps |3.814697265625e−6 |
| Δλ/threshold |0.9825704445509408 |

The computed gap is1.742955544905922% below the frozen threshold. The check
rejects `gap <= threshold`; its exception at index43 is therefore consistent
with the exact saved values. The local gap relative to λ[42] is about2.5571%,
while the guard uses the substantially larger leading λ for its global scale.
Independent local Python float64 arithmetic reproduced all24 boundary decisions
from the saved singular-value lists without importing Torch.

The band-projector derivative divides only between a retained band and its
complement. A strictly separated exact spectrum can have a defined derivative;
the saved nonzero gap does not establish a zero denominator. Here the guard
conservatively requires a global float32 separation margin before allowing the
derivative, and that declared numerical criterion was not met.
Float32 SVD values approximate the input's exact spectrum. This packet lacks
the original U/Vh factors and a certified residual/error bound, so the diagnosis
cannot certify the exact spectrum's separation or the accuracy/stability of
an unexecuted VJP. It does not justify lowering the multiplier, bypassing the
guard or declaring training stable. Any alternative numerical method or
criterion requires a separate predeclared audit; the existing failure stays
immutable.

## Gradient observations and protection

All12 captures are labelled `actual_projector_input_only`; custom
Autograd.Function.forward had grad mode disabled internally. Generic U/S/Vh
gradient capture is unavailable, every saved output-gradient dictionary is
empty, and none of the12 actual input-gradient hooks was reached. This follows
from failure during forward, before the scalar loss returned and before
backward. Empty observations are not finite zero gradients, and the old
generic-SVD backward crash is not reproduced by this event.
The saved model's57 state tensors and optimizer's81 state tensors are finite.
No solver error or reported SVD warning was saved in any of the12 calls.

The archived post-run guard passed with264 distinct protected paths, original
v1 preserved and all prior prerequisite receipts unchanged. This independent
read diagnostic hashed every protected path again before/after CPU packet
inspection; the complete dictionaries match the archived post-run dictionary.
The packet itself also has matching pre/post SHA and byte size:

| Artifact | SHA256 |
| --- | --- |
| failure_replay.pt,4347098081 bytes |`4eee8d6c17ac7f3f7cf511a6950b9cd8e0673b5f40af5c692ae242f07a2572f1` |
| failure.json |`390b9a80dbeaa291e344f7fe8658ce07a508627d3e379b2e77ea79913a7d989b` |
| post_run_guard.json |`b6e36489cc08365c66e2cd828acc82f5c79fe2b0d70ce1f0ff09a13c0c218fb3` |
| Frozen projector.py |`e44eb739482f3fe0448e1c9fdb066e36c4d6ea22d05a0c25a32734f29a856ab6` |
| Exact compact diagnostic JSON,168229 bytes |`afb5b187039ae15e2577ebaeb67e45e2edd6b3defdcb7d154545472804722f90` |

The independent script used CPU4 with CUDA_VISIBLE_DEVICES empty; CUDA was
never initialized. It took79.79201865196228 seconds, with peak host RSS
6427099136 bytes and130138406912 bytes available RAM before inspection.
It performed no SVD, model replay, backward, optimizer update or training.
Only the compact JSON was downloaded; no packet, raw interaction, feature
array, model tensor or credential was published locally.

## Clock origins

Remote diagnostic start/finish UTC from the remote clock were
2026-10-10T05:04:34.776382+00:00 and2026-10-10T05:05:54.568401+00:00.
Client transport start/capture UTC from the client clock were
2026-10-10T05:13:08.481215+00:00 and2026-10-10T05:14:36.521264+00:00.
`clock_offset_measured` is false. These timestamps have separate clock origins;
they are not a calibrated common timeline and must not be combined to infer
event order or elapsed training time. The failed training record's remote
unix value remains1791608412.8053644. Use the recorded child monotonic duration
for its runtime. Exact local transport metadata is in `capture_metadata.json`.

This diagnosis supplies no recommendation metric or innovation result. Four
finite steps and an earlier one-minibatch integration pass do not establish
baseline completion, convergence or publication evidence.
