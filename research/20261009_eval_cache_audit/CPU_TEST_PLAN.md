# CPU verification plan

This directory is independent of every running queue. Do not load Electronics,
change its checkpoint, signal its process, or attach the cache to its model.
Do not reserve or initialize a GPU. Use the existing MMRec Python environment
with `CUDA_VISIBLE_DEVICES=''` and four CPU threads.

1. Verify original bridge/encoder source hashes and all 35 frozen carrier
   source hashes before and after testing; do not modify those files.
2. Run the synthetic wrapper lifecycle tests: one generation across two user
   batches; exact scores and masked Top-50 identity; eval/no-grad requirements;
   parameter/load_state_dict version invalidation; model/thread/pass isolation;
   exception cleanup; RNG-consuming encoder rejection; unchanged source method.
3. Load the ACTUAL frozen Baby seed-2026 baseline checkpoint, original
   `DualFrequencyEncoder`, `GCN`, `FrequencyDecompositionModule`, and
   `BRIDGE.full_sort_predict` code and frozen TRAIN/validation dataloaders.
   Compare the first four 64-user validation batches without calculating
   ranking metrics. No test loader is constructed.
4. Direct and cached passes use identical users, state and positive masks.
   Require maximum absolute score error <=1e-6 (no relative tolerance), exact
   masked Top-50 item IDs, and matching parameter/buffer state and CPU RNG.
   Compare the known transient forward attributes by value, not object ID.
5. Record source hashes, device, Torch version, fixture dimensions, cache
   generations/hits, score error, item-ID equality, state equality and RNG
   equality. A mismatch marks `non_equivalent`; do not relax thresholds.

No GPU bitwise equivalence or complete Electronics validation is implied by
any CPU result. Only a later independently approved process could consider
using the wrapper for a single evaluation pass; this preparation does not
change the current original-baseline queue.
