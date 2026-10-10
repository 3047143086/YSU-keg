# Program contract

The runner loads the isolated official MSCA source and Baby `image_feat.npy`,
`text_feat.npy`, and split-zero interactions from the deployed carrier. It
loads the private three-batch archive before inspecting any loss. It emits
`batch_receipt.json`, `precheck.json`, `precheck.tsv`, `manifest_pre.json`, and
`manifest_post.json` under a fresh remote run root.

For each fixed batch:

1. Build the original model with seed 2026 and run the exact official
   `calculate_loss` BPR component on the clean batch.
2. Obtain gradients of that BPR with respect to image feature rows only.
3. Evaluate a functional targeted image-row replacement and a functional
   same-norm private Gaussian replacement. The graph tensors are never rebuilt.
4. Restore the untouched state and verify clean loss, state digest, and graph
   digest. Record finite checks, row counts, norms, clean parity, targeted gap,
   random gap, and `targeted > random`.

The implementation must not instantiate an optimizer. It may use autograd for
the one-step attack, but attack directions are detached before the second
functional forward. It must not backpropagate or update feature-table
parameters. A failed assertion exits without retry. CPU fixtures must first
test row replacement, equal-norm construction, exact restoration, and the
two-out-of-three gate on a tiny deterministic model.

The terminal status is one of `eligible_precheck`, `diagnostic_fail`,
`blocked`, or `technical_failure`. `eligible_precheck` still does not launch a
formal arm; root must register any later clean/targeted/random training study
separately with the existing validation-only gate.
