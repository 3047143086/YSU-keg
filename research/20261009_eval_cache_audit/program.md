# Audit program

`eval_cache.py` contains the independent wrapper. `EvalForwardCache` is a
single-pass context manager; `bind_cached_full_sort_predict` binds an isolated
function object whose code object is the original carrier predictor. The
wrapper is intentionally not installed on the model class. `masked_topk` is a
test-only helper implementing the existing evaluation mask.

`test_eval_cache.py` runs deterministic CPU-only unit tests when PyTorch is
available and skips cleanly in the local no-Torch environment. The actual
MMRec environment must run the same tests before any integration consideration.
No command in this directory deploys, launches, changes a checkpoint, or
touches the Electronics queue.
