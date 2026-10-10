# Autoresearch infrastructure diagnostic

Read PROTOCOL.md. Hypothesis: the original frozen Electronics BRIDGE loss
forward itself may have numerical differences between two calls with the same
learned state, saved minibatch and RNG. Observe exactly two original forwards,
without backward or updates; report 12 inputs, 36 bands and losses separately.
Only this directory is editable. Commit and obtain root code review before
deployment; run CPU structural tests before the one bounded GPU1 launch.
Record an exact/different diagnosis or a technical failure in results.tsv.
Keep all evidence. This is not an innovation, accepted repair or training result.
