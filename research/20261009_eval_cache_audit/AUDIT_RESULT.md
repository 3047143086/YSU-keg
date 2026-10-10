# Baby CPU cache result

V3 is the first completed actual-model comparison. V1 failed before model
construction because of a carrier path error. V2 passed 18 lifecycle tests,
then failed strict checkpoint load due to the original CPU/GPU registration
peculiarity. All immutable remote versions and their logs are preserved.

V3 commit 8d3965ee21d84c42b3ea7d934ef314f96e2c4a01 passed 22/22 tests in the
actual MMRec environment (Torch 1.12.1+cu113). The original Baby seed-2026
checkpoint, original BRIDGE and encoder code, and frozen first four
validation batches were used, with no test loader or model training.

Across four batches/256 users, each maximum absolute full-sort score error
was exactly 0.0 and each original-mask Top-50 ranking was exactly identical.
Parameter/buffer SHA256 before/direct/cached was identical:
c5869885f44c060ed85182119ebfe3390b860d8f16fd27df19eef2914e341d39.
All 59 protected input/source/checkpoint/audit payload SHA256 values were
equal before/after. Generated forward attributes, Torch/NumPy/Python RNG,
and original predictor code/globals were unchanged. The cache generated
once, hit three times, and cleared on pass exit. CUDA was not initialized.

Strict checkpoint load retained exact state-key coverage, exact tensor
values and preference alias object identity; no algorithm weight was
discarded. Direct four-batch scoring took 1.964261 seconds and cached
scoring 0.488212 seconds on CPU; the full audit took 12.323937 seconds.
This small timing sample is not an Electronics or end-to-end speed claim.

Full machine-readable evidence is stored locally at archives/v3/receipt.json
and remotely at /data/sunzixuan/eval_cache_audit_20261009_v3/receipt.json.
This establishes CPU implementation equivalence for the tested checkpoint
and batches. It is infrastructure evidence, not a model innovation, positive
recommendation result, GPU bitwise claim, or publication-ready baseline.

# Electronics GPU result

The independent GPU audit used commit 7c0a76ce33cf81412dac50838618f7567bde5cdc
on physical GPU 1 (RTX 3090), exposing one logical CUDA device. Actual CPU
lifecycle tests passed 23/23 before launch. The audit PID 3117155 exited
after 73.451221 seconds and wrote `non_equivalent` as its final status.

Six predeclared original validation batches (339 users) were compared from
the epoch-8 partial original baseline checkpoint. Maximum absolute score
errors were [0.00026226043701171875, 0.000133514404296875,
0.00007748603820800781, 0.00009250640869140625,
0.00011730194091796875, 0.00012969970703125], exceeding the fixed 1e-6 gate.
Original masked Top-50 identities matched only batches 0, 1503 and 3006;
they differed in batches 1, 1000 and 2000. Generated forward attributes
also differed, while all parameter/buffer state SHA256 values were equal
d5fc68c0ed568ea50bbe4e4bfbbee8d35605e927bf138a25145590212ed4ff21.
Torch CPU/CUDA, NumPy and Python RNG states remained unchanged. All 56
protected source/data/graph/checkpoint/audit payload hashes were equal
before/after. Strict checkpoint state-key/value coverage and alias identity
passed. The cache generated once, hit five times and cleared on exit.

Peak allocated/reserved CUDA memory was 6006633984/6748635136 bytes. Full
evidence is at archives/gpu_v1/electronics_gpu_receipt.json and the immutable
remote electronics_eval_cache_gpu_audit_20261010_v1 directory. No training,
test loader, test evaluation, tolerance change, solver repair, or integration
into the failed baseline occurred. The repeated GPU forward computations
produced unequal values under equal registered state and RNG; the specific
operator causing this difference has not been isolated. The GPU cache must
not be used for the original Electronics evaluator under this protocol.
