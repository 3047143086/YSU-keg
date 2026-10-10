# Two original forwards differ under the fixed diagnostic

The actual Electronics GPU diagnostic completed as `completed_different` in
105.7325158 seconds at deployed source
`dd55a8d1de4a72226847fbe1f629745184ea2d5f`. It executed exactly two original
`BRIDGE.calculate_loss` forwards with the same strict-loaded failure-packet
model, same interaction and restored packet RNG. No backward, optimizer,
projector, cache, solver override or training update occurred. This result
detects numerical differences in this one controlled original-original pair;
it is not an innovation gain or an accepted recovery.

| Observed family | Exact | Nonexact | Exceed inherited bounds | Median relative error | Maximum relative error | Maximum absolute error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SVD inputs | 3/12 | 9/12 | 0/12 | 5.0106079e-8 | 8.8120021e-8 | 3.5762787e-6 |
| Original bands | 9/36 | 27/36 | 13/36 | 2.3015828e-5 | 6.7004427e-4 | 6.7040324e-4 |
| Loss | 0/1 | 1/1 | 0/1 | — | — | 1.2665987e-7 |

The first loss was 0.09115207940340042 and the second was
0.09115220606327057. Their difference remains within the unchanged inherited
loss bound. Input and band comparisons retain relative Frobenius <=5e-5 and
maxabs <=5e-4*max(1,maxabs(first)); the loss retains rtol5e-5/atol5e-6. Exactness
is reported independently of those bounds. Bands exceeded a comparison bound
in captures 0, 1, 2, 3, 5, 9 and 10, comprising thirteen bands. No threshold
was changed and no additional pair was tried.

All 123 protected-file hashes matched before and after the diagnostic, including
the original v1 bundle, carrier sources, original data/graphs, archived packet,
early checkpoint and this payload. Predecessor helper/config/final-receipt
anchors and its eight payload files also matched. Model state values, schemas,
56 registered Parameter paths covering 53 independent Parameter objects, three
preference alias identities and module training modes remained unchanged.
The observer was restored after each forward; no parameter gradient appeared.
Saved interaction hashes, both restored RNG states, both post-forward RNG states
and backend settings agreed. Neither forward consumed RNG in this measured pair.

The actual MMRec CPU suite passed 17/17, including genuine Torch shared-Parameter
and shared-child alias checks. Runtime synthetic coverage and the real strict
model's parameter-state coverage passed, with the `user_activity_z` buffer
excluded from Parameter identities. The model kept autograd enabled and training
mode for loss calculation. After each forward ordinary graph representations
were cleared; Parameter aliases were retained. Detached CPU captures survived
the cleanup. No learned state was reloaded between the two calls.

The observed deterministic-algorithms and cuDNN-deterministic flags were false;
cuDNN benchmark was false, cuDNN TF32 true, CUDA matmul TF32 false. These flags
were recorded and remained unchanged. The existing backend was Torch1.12.1+cu113,
CUDA11.3, cuDNN8302. The diagnostic did not force a deterministic kernel or select
another SVD solver.

Peak CUDA allocated memory was 7,391,102,976 bytes, reserved memory
9,185,525,760 bytes and process resident memory 9,897,979,904 bytes. Process
PID3133577 exited. At raw archive time physical GPU1 had 24232 MiB free and no
compute process on its UUID; later work requires its own fresh resource gate.

Raw remote receipt SHA256 is
`651f5d066d03433a21926565d885050a14ecb63faae8cd5b0966aac20f565540`
(218419 bytes), and deployed config SHA256 is
`f5b433ab7962c890c3707b084ffd9c0f52f88adc930f8f23583f6b114a7b1526`.
Exact raw receipt bytes and the ten deployed payload files are frozen under
`../archives/v2`. `DIAGNOSTIC_EVIDENCE.json` contains all twelve per-input and
thirty-six per-band rows, loss components, inherited-bound results and lineage
summaries. `../report_evidence.py` reproduces it from the raw archives.

Both forwards matched the archive exactly only for inputs6–8; nine input tensors
differed from the archive. This was not an exact historical crash replay. The
two-call procedure omits optimizer/backward history and intentionally cleans
graph/allocator state between calls. It cannot estimate the frequency of this
effect or isolate its kernel cause. The tiny input differences and larger band
differences are consistent with upstream rounding and spectral sensitivity,
but this pair alone does not identify the responsible kernel or derivative.

The previous independent original-versus-projector full-forward comparison has
an attribution limitation: the unchanged original path itself can differ across
two measured forwards. That failed integration receipt remains failed. This
diagnostic does not demonstrate projector equivalence, finite full-model recovery
gradients or successful training, and it authorizes no recovery queue. Further
integration work requires its own protocol and review.
