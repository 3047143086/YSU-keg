# Electronics checkpoint audit plan (not launched)

A passing Baby CPU audit permits planning, not a GPU equivalence claim or
automatic integration. The next gate must use an existing Electronics
original-baseline checkpoint in a fresh isolated process, with all original
model/evaluator/data hashes unchanged. Never attach to or change the v1 queue.
No optimizer step or checkpoint mutation is allowed.

Pin one original checkpoint and the six original validation batch indices
0, 1, 1000, 1503, 2000, 3006 before scoring. Compare direct and cached
full-sort scores for each selected batch: absolute maximum error <=1e-6,
zero relative tolerance, exact original-mask Top-50 IDs. Both use the exact
BRIDGE.full_sort_predict code and original DualFrequencyEncoder.forward.
Cache one eval/no-grad pass only; one forward generation and five hits.
Check all parameters/buffers, generated forward attributes, CPU/CUDA/NumPy/
Python RNG states, and source/input/checkpoint hashes before and after.
Cache must clear on exit and refuse re-use across passes or model transitions.

GPU physical index 1 only; CUDA_VISIBLE_DEVICES=1 and four CPU threads.
Start only after confirming the predecessor's process has exited and GPU 1
has sufficient free memory. Never signal unrelated jobs or GPU 0. Observe
allocated/reserved peak memory and sampled GPU process memory. A numerical,
memory, RNG or ranking mismatch stops the audit; record technical failure
or non-equivalent as appropriate. No tolerance changes or solver repair.
Set a separate bounded resource budget before launch after measuring the
cost of the frozen constructor; this document grants no launch itself.

If the Electronics gate passes, create a separately declared fresh baseline
queue applying the cache only inside Trainer.evaluate and final_validation.
Keep score processing, train-positive mask, metric formulas, user order,
validation rounding, early stopping, checkpoint selection, training steps
and seed order unchanged. Do not resume a technically failed queue under a
new solver or call a partial seed a finished publication baseline. This aid
removes repeated deterministic encoder work; it is infrastructure, not a
research contribution.
