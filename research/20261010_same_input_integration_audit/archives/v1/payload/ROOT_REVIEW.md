# Preparation checkpoint

One actual candidate full loss forward; same realized X at each12 calls reaches
an original no_grad shadow and the frozen projector. All36 bands must be exactly
equal before the one actual loss backward. All27 gradient parameter names must
match the immutable original GPU reference and every gradient must be finite.

The old failed integration and the original-repeat v2 finding remain immutable.
No recovery training, projector modification or global report edit occurs.
Source/config/data/graphs/checkpoints/old receipts are guarded before/after;
learned state, full parameter aliases, method identities, RNG and backend flags
must remain unchanged. Partial counts and original Gram fallback are rejected.

Root review is required before deployment, actual CPU tests and GPU launch.
Only local structural tests and syntax checks are allowed at preparation.
Actual Torch alias tests will run in the existing MMRec environment with CUDA
excluded after review; local Python does not have Torch. No GPU was started.

Local checks completed: 23/23 tests passed (17 integration conjunction, three
method lifecycle, three warning lifecycle), all Python files compiled and the
generated remote management source compiled. The post-review actual MMRec
suite has31 tests including five genuine Torch alias/registration tests and
three frozen original/projector grad_fn, detached-shadow and hook compatibility
tests. None of the remote tests or the full GPU model has run at preparation.

The fixed forward gate requires all36 exact bands, all24 unchanged SVD calls
successful with no application fallback and no reported internal SVD fallback,
same input hashes/storage/stride/offset/version, per-phase RNG unchanged and
actual custom grad_fn outputs. Before the one backward it requires restored
original method/SVD identities, finite losses and unchanged learned state.
The final gate additionally requires all12 calls reached by actual loss output
and input gradients, every observed gradient finite, the exact27 canonical
parameter names, all parameter aliases/identities/requires_grad, and source,
configuration, RNG, interaction, old receipts and backend guards. Unused output
bands are recorded explicitly. No missing or approximate observation can pass.

After root review only, commands from the workspace root:

```powershell
python research/20261010_same_input_integration_audit/manage.py deploy
python research/20261010_same_input_integration_audit/manage.py test
python research/20261010_same_input_integration_audit/manage.py start
python research/20261010_same_input_integration_audit/manage.py status
python research/20261010_same_input_integration_audit/archive.py
```

Do not run these at preparation. The start command independently performs all
hash guards and immediately refreshes GPU1 resources, then creates one exclusive
launch ticket. The process repeats hashes followed by a fresh idle resource
check immediately before importing Torch/initializing CUDA. Actual total loss
references and ordinary graph transients are cleared after observations; no
Parameter alias is cleared or learned weight changed.
