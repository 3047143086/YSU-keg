# Numerical interruption of the decoupled gate pilot

Checked 2026-10-08. This note records technical failure separately from a
scientific performance decision.

## Formal queue

Source: `/data/sunzixuan/decoupled_gate_20261008_v1`.
Revision: `6b2a9276879e236667325ffbeb8b24ecd1958454`.
Both Baby one-epoch integration smokes succeeded. The Baby edge arm completed
zero-based epoch 38, then `checked_loss` raised
`FloatingPointError: Non-finite training loss before backward`.
The last rounded validation Recall@20 was 0.0984. Training had not converged,
so this is not a completed candidate metric. The history formal control and
Sports were not started. Queue PID 2540694 exited as `technical_failure`.

The preceding log contains a CUDA SVD convergence warning at the existing
carrier's `frequency_decompose_svd`. This proximity alone does not prove that
SVD is the cause of the non-finite loss. No model/settings changes, baseline
substitution, threshold relaxation or blind training restart was made.

## Bounded diagnostic

Diagnostic directory:
`/data/sunzixuan/decoupled_gate_20261008_diagnostic_v2`.
The initial diagnostic v1 stopped during checkpoint loading because BRIDGE
registers three preference parameter aliases on its first forward. v2 binds
only those aliases after verifying equality with the original nested keys,
then loads the checkpoint strictly. The optimizer's original parameter order
is preserved by constructing it before binding aliases. v1 is retained.

Checkpoint: `runs/baby_edge_s2026/best.pt`, epoch 38.
SHA256: `d78363ad1c4c9b674e54e34d9c7949e226df6bf07a0d53ed93707767d3d8649f`.
All loaded model parameters were finite. The diagnostic loaded the optimizer
state, retained the original model and SVD implementation, and checked 58
training batches. All forward losses, SVD outputs, observed input gradients
and parameter gradients were finite. Status:
`bounded_probe_completed_no_failure`. The checkpoint hash was unchanged.
Neither validation nor test was evaluated; diagnostic optimizer updates were
confined to process memory and were not retained as candidate training.

This result does not establish a fix. The original checkpoint does not contain
Python/NumPy/Torch/CUDA RNG states or a saved failed training batch; the probe
uses a new shuffled sequence and is not an exact replay. Future root-cause
diagnosis should capture the first non-finite gradient BEFORE optimizer.step,
along with its batch, RNG states and SVD inputs, in a separately declared
diagnostic. Do not introduce jitter, sanitize NaNs, skip failed batches or
change the spectral solver and call it an unchanged baseline comparison.

## Documentation lead

[PyTorch 1.12.1's SVD documentation source](https://raw.githubusercontent.com/pytorch/pytorch/v1.12.1/torch/linalg/__init__.py)
warns that repeated or small singular values can make singular-vector gradients
unstable. This is a plausible line of investigation, not a diagnosis of this
run. The documentation now matches the existing remote PyTorch 1.12.1.
No numerical correction was inferred solely
from this warning.

Local evidence: `diagnostic.json`, `diagnostic_receipt.txt`, and
`../20261008_decoupled_gate/latest_status.json`. The old soft-isolation queue
has meanwhile finished all six arms and failed the Sports validation gate.

## Expanded probe and guarded recovery

The isolated v3 probe resumed the same finite epoch-38 model, Adam and
scheduler state. It retained the existing training loop, epoch hooks and
validation, checked every gradient before optimizer.step and every parameter
afterward, and observed SVD output and input gradients. It completed 40
additional epochs (39 through 78), 2320 batches and 765.63 seconds without
reproducing nonfinite arithmetic. The checkpoint hash was unchanged. Test was
not evaluated. The unsaved failed RNG prevents claiming an exact replay.
Evidence: `long_probe.json` and `long_probe_receipt.txt`.

Confirmed diagnostic gaps: the original runner checked only loss/validation,
allowing a finite loss with invalid backward arithmetic to reach Adam; the
selected checkpoint omitted RNG and shuffled dataloader order. These gaps
are not proof that SVD caused the original failure.

`../20261008_gate_recovery` adds observational gradient/optimizer-state guards,
a saved first-failure batch/RNG/model/optimizer/SVD packet, and checkpoint RNG
and loader-order companions. It leaves the model, carrier, solver, original
runner, evidence, preparation and gates unchanged. Tests must show identical
healthy Adam updates and RNG. This is a single bounded fresh training recovery
with verified original baselines, not a numerical solver fix. No invalid steps
are skipped and no diagnostic-trained weights enter the scientific comparison.
If it captures a numerical failure, stop and investigate that packet. Any
subsequent solver correction requires fresh matched controls.

The recovery bundle is committed as `478a541`, passed 35 tests in the existing
server environment, and launched queue PID 2567421. Both Baby integration
smokes completed; formal Baby edge PID 2568005 completed epochs 0-2 and 174
finite-gradient steps at the latest snapshot. It remains live on GPU 1, using
about 3.3 GiB. The frozen four-arm queue continues after the active session,
stopping on numerical failure or the existing scientific gate. Track live
state with the recovery's manage.py; early epochs are not final metrics.
