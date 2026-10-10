# Autoresearch numerical infrastructure audit

Read PROTOCOL.md. Hypothesis: the frozen projector can reproduce every original
band exactly on the same realized full-model input and yield finite first-order
gradients for the frozen original 27-parameter participation set. One candidate
full loss forward and one backward only, no optimizer/update or extra forward.
Preserve the earlier failed independent-forward gate and original-repeat finding.
Commit protocol/code/tests for root review before deployment, tests or GPU use.
Only this new directory is editable; metrics belong to results.tsv.
Preparation does not change readiness or authorize training/publication claims.

Final v1 result: completed_pass, 36/36 same-X bands exactly equal, 27/27 actual
participating parameter gradients finite, and all lifecycle/hash gates passed.
One full loss forward and one backward completed in 107.8345558643341 seconds;
PID 3142340 exited. Actual MMRec CPU tests passed 31/31. See AUDIT_RESULT.md and
the immutable archives/v1 payload and raw receipts. This finite-minibatch
numerical integration result does not authorize another run or training and
does not establish recommendation gains or publication readiness.
