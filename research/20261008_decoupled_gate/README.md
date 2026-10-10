# Decoupled gate experiment

See PROTOCOL.md for the failure evidence, algorithm, comparison and limits.
The original validation-only runner, evidence calculator and preparation
logic are frozen copies. isolation.py preserves the original cache identity
checks; proposal.py contains the new gradient separation.

From the workspace root:

```powershell
python research/20261008_decoupled_gate/manage.py launch
python research/20261008_decoupled_gate/manage.py status --brief
python research/20261008_decoupled_gate/reporting.py
```

The detached queue waits for the previous pipeline to finish, verifies its
completed baselines and training-only caches, then runs at most four new
formal arms. Missing or non-finite evidence is never recorded as improvement.
