# Image-feature perturbation precheck receipt

This report is a conservative aggregate of the bounded attempts. It does not
claim an attack result and it does not authorize recommendation training.

## Frozen implementation

The independent branch `precheck-frozen` contains the frozen protocol,
official MSCA source snapshot, CPU fixtures, and remote runner. The relevant
commits are `74334fd` (protocol, program, source, helpers and fixtures),
`fda593c` (GPU visibility ordering), `b5f3fc8` (the already-existing private
batch archive's `(3, 3, 2048)` orientation), `3c91c93` (the missing official
text transform in the functional forward), and `5356814` (classification-only
zero/nonfinite row receipt). The source snapshot SHA256 remains
`066aafcf2e5373b20b2fdf9b3660302b8dfcaf580710d3ab864fac4f19e2360b`.

The four CPU fixtures pass (`4/4`); remote source compilation also passed.
These checks cover functional row replacement and restoration, equal per-row
norms for targeted/random directions, the strict two-of-three gate, and zero
row rejection. They do not measure recommendation quality.

## Remote attempt ledger

| run root | observed outcome | scientific result |
| --- | --- | --- |
| `.../feature_perturbation_precheck_20261010_v1` | stopped during archive-shape guard because the existing private archive is `(3, 3, 2048)`, not `(3, 2048, 3)` | no model/loss inspected |
| `.../feature_perturbation_precheck_20261010_v2` | stopped during the first model loss with `mat1 and mat2 shapes cannot be multiplied (7050x384 and 64x64)` | no attack metric |
| `.../feature_perturbation_precheck_20261010_v3` | initialized the model and reached the first fixed batch, then raised the combined message `zero feature or image-BPR gradient row` | no targeted/random metric; the message contains neither row type nor count |
| `.../feature_perturbation_precheck_20261010_v4` | stream disconnected before a remote run directory was created; the process exited | no receipt or metric |

The v3 message cannot be honestly split into zero feature rows versus zero
image-BPR gradient rows, and no count is available. Therefore the aggregate
terminal status is **`technical_failure`**, rather than `blocked` or
`eligible_precheck`. The protocol's rule remains that a confirmed zero or
nonfinite row would be `blocked`; this report has no such confirmation.

There are no BPR attack gaps, random-null gaps, or targeted-greater-random
batch count to report. No optimizer, validation loader, test loader, or
recommendation training was launched by these attempts.
