# TRAIN Review Expression Identifiability Audit

Declared 2026-10-09 before this audit's metrics, deployment or CPU execution.
This follows the rejected F24 mean-profile and F25 item-direction-residual
screens. It is a measurement of stable author expression, not another
recommendation experiment, a preference proof, or training authorization.

## Frozen Inputs and Scope

Read the completed F25 diagnosis and deployment receipts, its SHA256-frozen
`cache/review_embeddings.npy` and `cache/review_index.npz`, its audited frozen
global item bank, and its frozen `residual_core.py`. The cache contains 105,032
uniquely aligned support reviews only, after global removal of all 13,447
internal target edges. Verify all hashes before and after execution. Do not
read raw reviews, interaction files, validation/test labels or text. Do not
encode text again, train a model, alter the ranker, or use a GPU.

The cache already omits reviews without the F24/F25 same-item donor mask.
Consequently this audit's review-edge population is the frozen cache. The
user denominator is ALL 19,445 support users from F25's matching audit,
including users with no retained cached edge. The frozen local user IDs are
0 through 19,444; reject any cached owner outside this range. Report this
selection boundary rather than claiming full original review coverage.

## Fixed Split and Retrieval

Every cached edge `(user,item)` is assigned to A when the first byte of
SHA256(`review-identifiability-split:v1:<user>:<item>`) is even, otherwise B.
Require unique user-item edges. User profiles average unit vectors of all A
reviews and then unit-normalize. A-count at least two is required for a
scorable query; unique edges make these at least two distinct products.
The original B item is absent from that author's A profile.

For each B item, partition its B authors by their EXACT A-review count.
Only groups with at least two distinct authors, each with A-count>=2, are
scorable. For a group with authors ordered by local integer user ID, form
`M = A_profiles @ B_review_vectors.T`. Row u retrieves u's own B review
against every other owner's B review for exactly that item and A-degree.
Query AUC gives ties 0.5. No approximate candidates, hard-negative selection,
minimum cosine, norm-based filtering, candidate cap or threshold search.

## Degree-Matched Null and Views

For every `(item,A-count)` group generate one permutation of A profiles with
NumPy RandomState(2026), iterating item then degree in increasing order.
Fixed points are allowed and their count is reported. The same permutation
is used in both views. This null preserves same-item review contents, profile
degree, candidate count and the profile multiset. It disrupts the author
association. It is not a causal intervention and profiles across separate
items are not one global degree-preserving permutation.

Report exactly two views: raw frozen CLIP and the SAME item-direction
residual operation already frozen in F25, applied once per cached review.
Residuals use the frozen item's text direction. Both A profiles and B review
vectors use the same view. No direction, alpha, split, eligibility, encoder,
token, count or null is selected by results.

For each support user, average query AUC over ALL that user's cached B reviews;
unscorable B queries are 0.5 for both real and null. A user with no B reviews
also receives 0.5 for both. These neutral values are not observed
discrimination. Primary metrics average over ALL 19,445 support users;
no favorable user subset is selected. Zero vectors stay zero and ties are
0.5, without fallback. Also report scorable user/query counts and zero-vector
counts without changing eligibility by view.

## Gates and Interpretation

For each view separately require scorable-user coverage>=0.25 over ALL support
users, real mean user AUC>=0.505, real-minus-null mean user AUC>=0.005,
paired USER bootstrap2000 95%
lower bound>0, and positive real-minus-null deltas in both fixed
SHA256(`dual-witness-half:v1:<user>`) parity halves. Missing/nonfinite metrics,
input drift or empty halves are technical failures, never a scientific pass.
All two view outcomes are reported; neither can be hidden or retuned.

A passing view is `expression_identifiable`, not recommendation gain or
proof of needs/preferences: stable writing style can identify an author.
Failure is `expression_not_established_by_this_audit`, not proof that review
text contains no preference information. A positive expression result plus
failed F24/F25 item ranking makes item-bank alignment a separately testable
hypothesis, but does not prove alignment is the cause. This audit alone
authorizes NO BRIDGE training and supplies no publication conclusion.

## Budget and Execution

One Baby CPU audit, at most one hour including resource waiting, at most
four NumPy/BLAS/CPU threads, available host RAM>=12GiB, GPU disabled. Block
group scoring at128 profile rows so no full large user-user/item-item matrix
is allocated. Cache vectors and group scores are read-only inputs. Use an
isolated branch and remote root
`/data/sunzixuan/review_identifiability_20261009_v1`. Commit the protocol,
implementation and meaningful unit tests, have the root agent review the
frozen commit before metrics, then deploy an immutable bundle, pass tests in
the actual MMRec environment, and launch only this bounded CPU audit.
Preserve per-user arrays, both view results, all exclusions and resource use.
