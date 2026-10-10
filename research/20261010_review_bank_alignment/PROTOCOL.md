# Fixed Review Bank Alignment Diagnostic

Declared 2026-10-10 before candidate metrics, deployment or CPU execution.
F24 and F25 failed unseen-item ranking against the original item bank; F26
passed author-expression identification, which can also reflect writing
style. The new independent hypothesis is that a review-derived target
geometry can transfer that representation to internal held-out positive
items. This tests an association with observed TRAIN positives, not causal
preference, genuine dislikes of unobserved negatives or model performance.

## Exactly one intervention and frozen inputs

Retain F25's residual user profile and change only its target item bank.
For each cached support edge (u,i), let z_ui be its frozen unit CLIP vector,
t_i the frozen global item text direction, and
r_ui = frozen_F25_residualize(z_ui,t_i). No direction, alpha, encoder,
tokens, donor, gate or seed is reselected. Use

    p_u = normalize(sum_{i in cached support(u)} r_ui)
    b_i^(-u) = normalize(sum_{v != u, (v,i) in cached support} r_vi)
    candidate_score(u,i) = dot(p_u,b_i^(-u)).

The cache has exactly 105,032 unique support user-item edges from F25, after
global removal of all 13,447 frozen internal target edges and the earlier
same-item donor mask. Every item-bank review appears once under its true
owner; donor reuse never multiplies item-bank counts. Remove all reviews
owned by the query user from each item's sum and donor count before item
normalization. The fixed target and all 16 comparison items are absent from
that query user's cached support, so the actual subtraction count should
be zero; implement and test subtraction rather than rely on that fact.

Read only SHA-frozen F25 review_embeddings/index/per_user caches, global item
bank, residual_core, prior deployment/input/diagnosis/revision receipts,
F26 diagnosis, and F23 evidence/diagnose/per_user/diagnosis for controls.
Read no raw reviews or interaction file. Preserve the exact F25 user order,
targets, 16 comparison IDs, ties=.5 and target digest. Negative repetitions
from the original frozen sampler are retained. Assert every target globally
absent from cached support and every comparison item absent from its query
user's cached support. Full-support KNN reference pairs must match exactly.

All 13,447 diagnostic users remain in every arm, including zero query
profiles and empty banks. Empty/zero banks and zero profiles score zero;
ties give .5, without fallback. Score in blocks of128 query rows. A user
counts toward candidate coverage only if its query profile AND all17
leave-owner-out residual item banks are nonzero. Require coverage>=.80;
report target-bank, all-comparison-bank, profile and catalog coverage too.
This gate cannot be lowered after inspecting results.

## Eight complete controls and unambiguous paired deltas

1. item_content: F25 cached item-content user profile against original
   unit(global image+text) target. Candidate minus item_content.
2. old_target_geometry: SAME F25 residual user profile against the original
   global image+text target. This removes the only intervention. Candidate
   minus old_target_geometry.
3. raw_review_bank: raw-own review profile against other-owner raw review
   bank, with identical edge mask and leave-owner-out subtraction.
4. raw_donor_bank: frozen same-item other-owner donor profile in raw space
   against the SAME raw true-owner bank. This replaces query expression.
5. residual_donor_bank: frozen donor residual profile against the SAME
   residual true-owner bank. This replaces query expression in candidate
   geometry. The two donor maps are exactly the frozen F25 indices.
6. popularity: each item's degree in the same 105,032 cached support edges.
   Every scored comparison is absent from query support, so leave-owner-out
   counts equal the ordinary counts.
7. itemknn: same cached-support item cosine cooccurrence, self diagonal zero,
   top1000 positive neighbors per row, lower item ID first on equal weights,
   support @ similarity.T. F23 frozen diagnose/evidence implementation.
8. userknn: same cached-support binary user cosine, self neighbor zero,
   top100 positive neighbors, lower user ID first on ties, weights @ support.
   F23 frozen build_user_evidence(mode='userknn') and gather_scores.

All cosine denominators, positive-neighbor eligibility, matrix dtype and
deterministic tie policies follow those SHA-frozen F23 functions; all cached
support users 0..19,444 and items 0..7,049 are retained, without degree
filtering. F23 full-support (105,104 edges) ItemKNN/UserKNN are additional
complete references, explicitly differing by72 masked edges. They never
substitute for same-support gate controls and never enter the gate.

The two old geometry arms are recomputed using frozen F25 functions and
must reproduce their cached per-user AUC arrays exactly before new metrics
are accepted. The other three frozen own/donor profiles must also reproduce
their original-bank F25 per-user arrays, validating the index/donor mapping.
Bank arms share the same leave-owner-out scorer and true-owner
bank. All eight candidate-minus-control deltas must be reported by these
names, whether favorable or not. No best arm/view/control selection.

## Fixed scientific gate and interpretation

The one candidate passes only when coverage>=.80, mean user AUC>=.505,
AND its mean paired user AUC delta against EACH of all eight controls is
>=.005, AND each2000 paired-user bootstrap 95% lower bound>0, AND each
delta is positive in both SHA256('dual-witness-half:v1:<user>') parity halves.
Use one fixed NumPy default_rng(2026) bootstrap index stream shared across
all eight deltas. Missing, malformed or nonfinite arrays, provenance drift,
empty hash halves or old-score mismatch are technical_failure, never pass.
Store all per-user scores/AUCs and diagnostics. Report every failed condition.

Pass means review_bank_alignment_supported_in_train_internal_diagnostic;
failure means discard_this_fixed_bank_geometry. Neither is a publication,
novelty, causal preference or formal-training authorization. It is not a
full-sort validation/test result and not independent confirmation after
many prior hypotheses on these TRAIN targets. F24/F25 failures stay frozen.

Using user and item reviews for recommendation is established prior art:
[DeepCoNN](https://arxiv.org/abs/1701.04783) jointly models user/item review
representations, [MPCN](https://arxiv.org/abs/1801.09251) models multiple
review interactions, and [ERRA](https://arxiv.org/abs/2306.12657) uses personalized
review retrieval and aspects for explainable recommendation. The
[SIGIR20 critical study](https://arxiv.org/abs/2005.12210) also motivates
strong controls and careful evaluation of review models. This diagnostic
does not reproduce either model and claims no first review bank mechanism.
The project's PUBLICATION_PROTOCOL.md still requires three domains, five
seeds, strong baselines and independent attribution before paper-level claims.

## Budget and immutable execution

One Baby CPU diagnostic, at most one hour including resource waiting, four
CPU/NumPy/BLAS threads, available host RAM>=12GiB. GPU disabled; no Torch,
CLIP import, new encoding, optimizer, validation/test loader or model training.
Remote root /data/sunzixuan/review_bank_alignment_20261010_v1 and branch
autoresearch/oct10-review-bank-alignment are isolated. Commit protocol,
core/executor/manager and meaningful leakage/control/numeric tests; require
root code review before actual-environment tests and metrics. Deploy once
immutably, test MMRec, then launch once with an exclusive lock. Preserve all
outcomes and before/after hashes, runtime and peak RSS; no budget extension.
