# Instrumented recovery of the decoupled gate experiment

See PROTOCOL.md for the failure evidence, unchanged algorithm, comparison and
limits. Original train.py, model, carrier, preparation and evidence remain
frozen. guarded_train.py/numerics.py observe updates without changing them,
block invalid gradients, and save exact failed-batch replay inputs.

From the workspace root:

```powershell
python research/20261008_gate_recovery/manage.py launch
python research/20261008_gate_recovery/manage.py status --brief
python research/20261008_gate_recovery/reporting.py
```

The queue verifies that the failed candidate and bounded diagnostic exited,
checks completed original baselines and training-only caches, then freshly
trains at most four formal arms with the existing Baby/Sports gates. Remote
root: `/data/sunzixuan/decoupled_gate_20261008_recovery_v1`. One recovery attempt,
with no skipped batches, solver changes or automatic numerical retries.
The root cause of the earlier failure remains unconfirmed.
