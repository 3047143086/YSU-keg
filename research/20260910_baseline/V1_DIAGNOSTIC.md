# First CPU confirmation attempt

Remote directory: `/data/sunzixuan/bridge_baseline_confirm_20260910_v1` (preserved).
Worker PID: 4079173. Baby stopped before scoring after 6.63 seconds; Sports did not run.

The historical GPU checkpoint omitted `result_embed`, an unused parameter declared
as `nn.Parameter(...).to(device)` in `dual_frequency_encoder.py`. On CPU it remains
a registered parameter, causing a missing-key exception. `rg` and source review
confirm a single occurrence (declaration) and no use in the forward/score path.
The other missing key was the already permitted, disabled `user_activity_z` buffer.

Version 2 permits exactly this extra unused key, asserts its single occurrence,
and records the compatibility exception. Model code, inputs, checkpoint weights,
ranking protocol and predeclared metric tolerance remain unchanged. This startup
exception is not an experimental performance result. The v1 deployment, source,
logs and completion receipt remain in the remote directory above.
