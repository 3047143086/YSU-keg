# Spectral projector infrastructure audit, v2

This isolated audit addresses the numerical crash of the frozen Electronics
BRIDGE baseline at epoch 9, batch 225. It is not an innovation experiment and
does not authorize training or replacement of v1 files. Its purpose is to test
an equivalent spectral-band mapping and its correct first derivative before a
separate baseline recovery can be considered.

## Protected reference and lineage

Original carrier file `models/dual_frequency_encoder.py` SHA256:
`287c1627a289707685028d1852105efa69b8da5106cb175c2871cacd7904e90c`.
Original bundle commit: `06bcd212af31cd7c7ccd6ccc63de9778130b9f6b`.
Original run: `/data/sunzixuan/electronics_baseline_20261009_v1/runs/elec_baseline_s2026`.
Failure replay SHA256:
`7b6f12731715c0315018f6b3c903b39a8829cc7727035e18eb92eeaa4bd3deb7`.
Best checkpoint SHA256:
`3f9283407ff51178549c2f78e0f8cf61f895c269164afa6b3b9a0a56d8faacda`.
The archived replay has twelve original finite SVD input matrices; no tensor
needs to be downloaded to the Windows machine. Its output-gradient hooks
contain dL/dU, dL/dS and dL/dVh, not the upstream gradients dL/dBand. Actual
replay gradient probes must therefore be labeled controlled independent probes,
not a reconstruction of the original training gradient.

## Mapping and derivative

For a real tall matrix X with F columns, obtain the same original SVD
`X = U diag(s) Vᵀ`, ordered by decreasing singular value. Split F dimensions
with the original remainder rule; F=64 and M=3 give [22,21,21]. A band's
reference output is `U_b diag(s_b) V_bᵀ`, mathematically equal to `Y_b=X P_b`,
where `P_b=V_b V_bᵀ`. The candidate retains the original SVD and forward dtype
and returns the exact original `U_b @ diag(s_b) @ V_bᵀ` arithmetic order inside
the custom function. X P is used to derive the gradient and is retained only as
a separate diagnostic identity. The recovery therefore changes only the
implementation of the first derivative. It neither adds a diagonal jitter nor
chooses another solver/fallback or stops gradients through the projector.

Let A=XᵀX, λ_i=s_i², and d_i indicate membership in a band. With upstream G_b,
define `H_b=sym(Xᵀ G_b)`. In the V eigenbasis, the projector derivative uses
`K_ij=(d_i-d_j)/(λ_i-λ_j)` only when d_i differs from d_j; all remaining entries
are exactly zero and are never divided. Then

`B_b = V [K ⊙ (Vᵀ H_b V)] Vᵀ`

`dL/dX = Σ_b [G_b P_b + X(B_b+B_bᵀ)]`.

The factor two follows from differentiating XᵀX, and the denominator is the
squared singular-value difference, not a singular-value difference. This
formula handles repeated eigenvalues wholly inside one band or wholly inside
its complement. A degeneracy across a band boundary remains undefined and
must stop; it must not be masked, jittered, clipped or assigned a zero gradient.
The audit computes the small spectral derivative and accumulation in float64,
then returns the input-gradient dtype. It supports first derivatives only;
second-order autograd or complex inputs are outside this audit.

Before forward outputs are returned, each band-boundary λ gap must exceed
`32 * epsilon(input dtype) * max(λ)`. This fixed guard rejects boundaries that
cannot be numerically resolved; it is not a repair or denominator floor.

## Pre-registered validation and tolerances

Small CPU tests run in the existing MMRec environment with four threads and
CUDA hidden, without loading the failure replay or evaluating a recommendation
metric. Tests must all pass: double gradcheck for distinct, repeated internal,
rotated repeated internal and rank-deficient internal singular values; direct forward/VJP comparison
against the original formula on nondegenerate input; finite arbitrary VJP at
internal degeneracy; exact and unresolved cross-boundary rejection; summation
identity, one-band gradient, unused-output gradient handling, float32 dtype and
nonfinite upstream rejection.

Double gradcheck uses epsilon 1e-6, atol 1e-7, rtol 1e-5. Nondegenerate double
forward comparison uses atol/rtol 1e-10; VJP uses atol 1e-9 and rtol 1e-8.
Float32 small-matrix forward uses atol 1e-5 and rtol 5e-5; VJP uses atol 1e-4
and rtol 1e-3. These thresholds are frozen before actual replay forward metrics.

Actual replay audits require root review of this committed protocol and source.
They must first verify archived hashes, then inspect all twelve inputs, with
all M=3 band reconstructions and no selection based on favorable errors. Each
band's candidate output must be elementwise identical to the original formula
when using the same SVD backend and input. Separately, each direct X P diagnostic
band's relative Frobenius error to the reference must be <=5e-5, and maximum
absolute error must be <=5e-4 * max(1, max_abs(reference)); summation relative
Frobenius error to X must be <=5e-5. Fixed independent upstream probes use
seed 2026, and candidate input gradients must be finite. Comparison with generic
SVD gradients applies only where that generic gradient is finite, and uses
relative Frobenius tolerance 1e-3 and maximum absolute tolerance
1e-3 * max(1, max_abs(reference gradient)). Record invalid generic gradients
as failures of that generic comparison, never as matched gradients.

Every capture uses its original float32 input without casting the input to a
higher precision before SVD or boundary checks. Each capture has independent
upstream probes generated with a fresh CPU generator seed 2026, in band order.
The CPU-recomputed singular values must be recorded alongside the originally
archived singular values; a CPU decomposition is not a replay of the original
CUDA solver's exact rounding. Generic nonfinite reference gradients are logged
with a failed/unavailable comparison, not as zero difference or equivalence.
Candidate forward/finite-gradient validity and available generic-gradient
equivalence are reported as separate conclusions.

Before and after the audit, all archived bundle, carrier, original input,
run input and optional-absence hashes are verified against the immutable v1
receipts, including the original failure replay and checkpoint. The audit's
own committed payload hashes are verified before and after execution. RAM must
be at least 12 GiB available and at least four CPU cores available before
loading the replay and before each capture. Resource failure stops the audit.

CPU replay is capped at twenty minutes, four threads, and one loaded replay
packet plus one matrix at a time. Any future GPU operation requires a fresh
GPU process/resource snapshot and root coordination; this protocol does not
authorize a GPU launch or training. Timing, all gaps, errors, exceptions and
input hashes must be saved in receipts. No pass criterion may be relaxed after
observing actual replay results. A passing audit alone does not establish
baseline convergence or paper-ready evidence; a new separately reviewed queue
would still be required.
