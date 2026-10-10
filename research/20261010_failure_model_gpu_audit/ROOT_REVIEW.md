# Root review before execution

2026-10-10. Original v1 and frozen CPU projector are protected by their
individual source/input/checkpoint hashes. No optimizer is instantiated.
The original forward formula and solver are retained; both model probes
restore the failure packet RNG before calculate_loss. Full checkpoint load
is strict after preference alias equality checks.

The root review identified two missing conjunctions and added them before
execution: a finite generic archived gradient must pass its comparison,
and full-model participating gradient parameter sets must match and be
nonempty. Six gate tests cover those conditions and reject missing calls,
band differences, state changes and loss differences. Actual environment
tests must pass before launch. No numerical thresholds were changed.

Historical input fidelity is reported separately from the current original
versus repaired integration. A current integration pass without exact archived
inputs must not be described as reproduction of the original crash.
GPU 1 availability is checked before launch and again in the child before
CUDA initialization. One immutable directory, exclusive launch ticket and
thirty-minute child alarm bound execution. The failed evaluation cache is
excluded. The caller checks final receipt and process identity before any
subsequent GPU work or recovery training.
