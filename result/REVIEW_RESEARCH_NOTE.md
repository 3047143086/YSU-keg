# Review signal: expression, target geometry, and preference

Recorded 2026-10-10, before the new review-bank diagnostic has produced metrics.
This is a research rationale, not a method result or novelty claim.

F24 found a small own-review advantage over same-item donor text, but the
unchanged item-content history profile was stronger. F25 item-direction
residualization made ranking against the original item bank worse. A separate
support-only author retrieval audit then found positive expression association
in both raw and residual views under exact item and history-count matching.
These outcomes can coexist: a representation can encode stable author style
without providing useful information about an unseen product choice.

The next isolated question is whether using review-derived item target vectors
changes this transfer result. Hold the F25 residual user profile, support mask,
internal held-out targets and comparisons fixed. Form each item's target vector
from other authors' support reviews, exclude all held-out target edges globally,
and remove the querying author's contribution explicitly. Compare against the
original item geometry, raw review geometry, same-item donor queries, content,
popularity and support-matched collaborative neighbors. Keep zero/missing
vectors and all declared users. A pass must satisfy the new protocol's complete
control conjunction; an author-retrieval pass does not substitute for it.

This diagnostic identifies observed-positive ranking association under a fixed
TRAIN split. It does not identify causal preference, future purchase behavior,
full-sort recommendation gains or an advantage over a trained recommender.
Official validation/test remain outside this diagnostic. No residual weight,
encoder, token limit, mask or gate is selected by its results.

Review representations for both users and items already appear in
[DeepCoNN (WSDM 2017)](https://arxiv.org/abs/1701.04783). Target-dependent
review matching also appears in
[MPCN (KDD 2018)](https://arxiv.org/abs/1801.09251). The empirical study
[How Useful are Reviews for Recommendation? (SIGIR 2020)](https://arxiv.org/abs/2005.12210)
reports that review-based methods can fail to beat existing baselines outside
narrow settings and emphasizes compatible evaluation. These sources rule out
a first-use claim for a review user/item bank or review-to-review matching.

A future contribution would need independently supported mechanism evidence
and a concrete distinction from these methods. The current evidence only
motivates a controlled geometry diagnostic. Any later model intervention needs
its own frozen hypothesis, matching controls, validation screen and the
three-domain/five-seed confirmation specified in PUBLICATION_PROTOCOL.md.

The subsequent F27 geometry diagnostic completed and was discarded. Candidate
AUC was 0.5880029; changing the old target geometry improved AUC by 0.0618865,
but item content was 0.5884816, raw review-bank was 0.6703122, same-item raw
donor was 0.6690712, and same-support popularity was 0.7576318. The eight-control
conjunction failed. This supports a geometry effect relative to the weak old
target, while failing to establish useful transfer of personal expression.
The positive author identification audit remains valid in its narrower scope.
These results do not support tuning the residual or raw-bank weights.

Evidence: research/20261009_review_signal/diagnosis.json,
research/20261009_review_residual/diagnosis.json, and
research/20261009_review_identifiability/RUN_REPORT.md and
research/20261010_review_bank_alignment/RUN_REPORT.md.
