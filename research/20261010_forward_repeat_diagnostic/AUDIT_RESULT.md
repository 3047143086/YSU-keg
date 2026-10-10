# Original forward repetition diagnostic

Version v2 completed with original-original numerical differences under unchanged
state, interaction and restored RNG. The detailed result is in
[v2/AUDIT_RESULT.md](v2/AUDIT_RESULT.md), with source
`dd55a8d1de4a72226847fbe1f629745184ea2d5f`: 3/12 inputs and 9/36 bands were
exact, 13 bands exceeded unchanged inherited bounds, and the single loss differed
by 1.2665987e-7 while remaining within its inherited bound. All123 pre/post input
hashes and model/RNG/lifecycle guards passed. No training occurred. The original
failed projector integration gate remains failed; this pair exposes an attribution
limitation rather than proving projector equivalence.

## Preserved v1 technical failure

The v1 process, source `8a1d7579588f55c7e9ad09530182bbd53c00c4c8`, exited
after 77.152888 seconds before any measured `calculate_loss` forward. Original
construction and strict checkpoint loading completed, but parameter snapshot
enumeration raised `TypeError: named_parameters() got an unexpected keyword
argument 'remove_duplicate'`. No repeatability finding exists. No backward or
optimizer update occurred and no training queue started.

The actual MMRec CPU-only structural suite passed 12/12; it did not exercise
this model-specific public API. This missed compatibility check is addressed
separately in v2 with actual CPU Torch shared-Parameter/shared-child alias tests
and a read-only registry enumeration. Scientific scope and thresholds are fixed.

All 121 protected file checks passed both before the failure and on the failure
path; all predecessor helper/config/final-receipt anchors remained unchanged.
The archived process PID 3130285 was checked exited. Physical GPU1 then had
24232 MiB free, with no compute process on its UUID. Any later launch still
requires another fresh resource check.

Exact raw remote bytes are retained under `archives/v1/receipts`, with the
original payload under `archives/v1/payload`. SHA256 values:

| Receipt | SHA256 |
| --- | --- |
| diagnostic_config.json | a336f9183b3046f58a21d8db6942b536ec458e0c3537a8b56f3395686e8d8fb4 |
| tests.json | 3458b7682db282a2b717e5bc450c752b23701dbd7158c8b058125691d48e4268 |
| launch.json | 19f4ab29a658e5209a02577e5f5da82221b00381cb3f6536e5ea912f5efc0ff0 |
| repeat_receipt.json | 64d8f166b39fdf9302f0feb7c0daa80faf4d93ffb4a3a42a561dcbd9ae7b967c |

The original remote v1 directory remains unchanged. The separate reviewed v2
uses read-only registration enumeration; its raw receipts and deployed payload
are independently frozen under archives/v2.
