# Candidate-Conditional Reliability Routing (C2R)

## Status

**NO-GO / FREEZE.** The Baby screen did not reproduce on the independently
declared Sports confirmation. Do not tune or reuse this routing family.

## Motivation

BRIDGE deliberately separates retrieval from behavior calibration, but its
three-way ID/visual/text score fusion is fixed by the encoder gates. C2R keeps
the official base top-K retrieval set and learns a pair-specific routing of the
three score blocks only inside that set.

For modality margin `m_m = s_m(u,i+) - s_m(u,i-)`, C2R forms a detached target
`q = softmax(m / (tau * scale))`. The target is weighted by

```text
c = sigmoid((max(m) - min(m)) / (tau_c * scale) - 1)
```

so nearly indistinguishable modality margins do not dominate the gate. User- and
item-side gates are multiplied and normalized to sum to three, preserving the
initial BRIDGE score scale. The training objective adds a confidence-weighted
cross-entropy (`DWR-KL`) and a small batch balance penalty to the official
BRIDGE BPR objective. At inference and training, the routed score is applied as
an entropy-controlled residual: uncertain (near-uniform) gates leave the BRIDGE
score unchanged, while confident gates receive at most `c2r_route_mix` of the
residual. Both gates are initialized uniformly, so the identity model is
recovered at step zero.

## Screening protocol

- Amazon Baby, fixed existing split and features, seed 2026.
- Official BRIDGE result from the preceding controlled run is the baseline.
- Main: `BRIDGE_C2R`, `c2r_loss_weight=0.05`, `balance=0.005`.
- Ablation: same routing architecture with `c2r_loss_weight=0` and
  `c2r_balance_weight=0`, isolating the loss contribution.
- Select by validation Recall@20; report test only at the selected checkpoint.
- Continue to five seeds and other datasets only if validation Recall@20 and
  NDCG@20 both beat official BRIDGE.

## Screening result (2026-08-30)

The initial direct-routing implementation underperformed: validation
R@20/N@20 were `0.1018/0.0479`. The subsequent conservative residual version
at `route_mix=0.25` with the same carrier reached `0.1066/0.0496` (R@10/N@10
`0.0707/0.0405`) versus the matched BRIDGE control `0.1056/0.0492`
(R@10/N@10 `0.0709/0.0404`). The gain is small (`+0.0010` R@20, `+0.0004`
N@20) and R@10 is essentially unchanged, so it is a provisional candidate
rather than a paper claim. The `route_mix=0.50` ablation reached
`0.1058/0.0493`; no-loss direct routing reached `0.0890/0.0417`.

Confirmation is required on the declared additional seeds and datasets before
any claim of a mechanism improvement.

## Cross-dataset confirmation (2026-09-03)

On Sports/seed-2026, matched BRIDGE validation R@10/R@20/N@10/N@20 was
`0.0834/0.1198/0.0474/0.0567`; C2R reached
`0.0827/0.1191/0.0472/0.0565`. All four validation metrics declined. C2R's
small test increases cannot be used for model selection, so the preregistered
confirmation failed and the mechanism is frozen.
