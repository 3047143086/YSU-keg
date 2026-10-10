# Same-input integration: completed pass, bounded scope

The frozen projector passed the preregistered same-realized-input gate in the
actual frozen Electronics BRIDGE loss graph. All 36 original-shadow and candidate
bands were exactly `torch.equal`; the candidate outputs supplied the actual
loss and its one backward produced finite, nonempty gradients for exactly the
27 original-reference participating parameter names. This establishes numerical
integration and finite first derivatives for this one instrumented minibatch.
It does not establish training stability, recommendation gains or publication
readiness. No training or optimizer update was performed.

The reviewed scientific source revision was
`e99e57b42f52f9e6494dd527e0c2eab6ce6d4fc6`. A new immutable remote deployment
ran 31/31 actual MMRec CPU tests with CUDA excluded, then consumed its one launch
ticket. PID 3142340 completed in 107.8345558643341 seconds and exited. There was
exactly one full candidate `calculate_loss` forward and one actual backward;
no second full original forward, rerun or recovery training occurred.

| Frozen check | Actual evidence |
| --- | --- |
| Same-X frequency calls and bands | 12 complete calls, 36/36 exact bands; all absolute and relative descriptive errors zero |
| SVD observations | 12 shadow + 12 candidate successful calls, 0 observed exceptions; unchanged `full_matrices=False` invocation |
| Actual loss | One finite scalar component and sum: 0.09115216135978699 |
| Candidate gradient attribution | All 36 outputs had the frozen projector custom grad_fn; all 36 output hooks and 12 input hooks reached once, finite and nonempty; no unused bands |
| Participating model gradients | Exactly 27 canonical names matching the frozen original GPU reference; all finite/nonempty; all registered alias gradients finite |
| Registered state | 56 parameter paths / 53 distinct objects covered; values, schemas, identities, aliases, requires_grad and module modes unchanged |
| Input/method/RNG/backend | Same storage, version, stride and value hashes; both method identities restored before backward; interaction, saved RNG and backend settings unchanged |
| File lineage | All 129 protected hashes match before and after; old failed GPU receipt and original-repeat v2 receipt/payload remain unchanged |
| Warning observation | Original warning observer restored; one ordinary SciPy SparseEfficiencyWarning; no reported SVD nonconvergence/fallback warning |

The process used physical GPU1, four CPU threads and the declared .55 memory
fraction. Peak allocated/reserved CUDA bytes were 7,023,031,808 / 9,422,503,936;
peak resident host bytes were 7,543,611,392. The read-only archive check found
the process absent and GPU1 idle with 24,232 MiB free. That observation is a
release receipt, not authorization or a fresh gate for another job.

Exact small remote bytes are preserved under `archives/v1/receipts`, and all 16
deployed scientific payload files are copied byte-for-byte under
`archives/v1/payload`. No data, checkpoint or credential was downloaded. The
raw anchors are:

| Artifact | SHA256 |
| --- | --- |
| Final receipt, 236009 bytes | `87251d388816ec3d395f477a096a7f67e1ea6e25cfaf84116b517c6a36c64f2c` |
| Deployment config | `9d217792556b38d685d79cad426a521d78bedfc4e28529e0040ed1774a5df84e` |
| Actual CPU tests | `691fb0304b200c488186c3c63525ecf210da931a8a713989c3b91fc8ee13b57b` |
| Single launch ticket | `4990def397e0a010de8976368610977a875a41aafe34808c04a6e305bd9c6da7` |
| Before-backward progress | `6113ecd81828db49425cc87627470dccd946ae8d410985c4aff26ee5d7cf831c` |
| Deployed runner | `f8e6b1fd409e9a90afdf0f8eb2fa0ef06d9b49eea466efc8f36361a3009eec4d` |
| Deployed gate | `11bee90b47d623c6cd5a19c135c591ed4d40a6da43c5954f420b7837f7179579` |
| Unchanged projector dependency | `e44eb739482f3fe0448e1c9fdb066e36c4d6ea22d05a0c25a32734f29a856ab6` |

`integration_config.json` lists every deployed payload SHA. Reproduce the
compact evidence and independently recompute the frozen conjunction locally:

```text
python research/20261010_same_input_integration_audit/report_evidence.py
```

That command verified all archived raw-receipt and payload SHAs and returned
`gate_recomputed: true`. `INTEGRATION_EVIDENCE.json` preserves the call rows,
actual gradients, warnings, complete guard flags and original evidence lineage.
The top-level program/results are the post-run record; the archive preserves
the original deployed program/protocol unchanged.

The earlier independent-original-versus-projector GPU audit remains
`completed_fail`, with raw receipt
`a840a3a1d952a8300d44951af77d8068b28770057a0ba3ca1dc833e0c21b7eb0`.
The original-original repeat diagnostic remains `completed_different`, with raw
receipt `651f5d066d03433a21926565d885050a14ecb63faae8cd5b0966aac20f565540`.
Their gates were not relaxed or replaced. Old archived-input probes are separate
evidence: 12 candidate probes passed, while generic capture0 had 12,313,792
nonfinite gradient elements and the projector probe was finite. Those probe
gradients are not this minibatch's actual loss gradients.

Extra shadow SVDs, hashes and CPU synchronization change the instrumented
execution and allocator history. The result is confined to its realized inputs;
it is neither an exact replay of the historical backward crash nor proof of
generic VJP equivalence or an unobserved training trajectory. No explicit or
application fallback and no reported internal SVD fallback occurred; solver
internals were not instrumented. A future recovery experiment requires a
separate reviewed protocol and authorization; this audit starts none.
