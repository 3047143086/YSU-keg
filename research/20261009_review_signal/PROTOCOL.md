# Consumer review expression conditioned on the reviewed item

Declared 2026-10-09 before any review embeddings or ranking measurements.
Baby's original raw reviews were previously used only for identity recovery;
the rating-residual family failed. Adding review TEXT is a new observation in
this workspace, not a novelty claim. DeepCoNN already represents users/items
from reviews (https://arxiv.org/abs/1701.04783); a critical SIGIR20 study warns
that review methods may not beat simple controls
(https://arxiv.org/abs/2005.12210). Novelty remains unestablished.

## Isolated information question

Does the same user's TRAIN review wording add held-out preference information
beyond which products were reviewed? Compare real wording against another
TRAIN consumer's wording for EXACTLY the same support product, and against
the original item's description embedding. This is observational conditional
evidence, not causal identification or a proof of functional-need semantics.

Use the existing Baby x_label=0 only. Reuse the fixed dual-witness internal
targets: each TRAIN-degree>=4 user holds out the minimum SHA256 of
dual-witness-target:v1:<user>:<item>. Remove ALL target edges globally before
matching support reviews or computing histories. User identity is recovered
only by remaining TRAIN event fingerprints (ASIN,timestamp,rating), never by
validation/test preferences or text. Do not guess ambiguous identities or
events. Missing/ambiguous text produces no profile mass. All 13,447 diagnostic
users remain; no favorable subgroup is selected. These reused internal
targets are exploration, not independent confirmation.

Only the summary + reviewText associated with support edges is encoded.
Never use held-out target/validation/test reviews, ratings, reviewerID,
reviewerName, helpful votes, or timestamp as preference features. No raw review
text or external reviewerID is printed. Preserve inputs with SHA256 checks.

## Frozen representation and controls

Use the existing frozen OpenAI CLIP ViT-B/16 and tokenizer from the stage1
cache. Clean HTML and whitespace, concatenate summary + reviewText, truncate
to first 75 BPE content tokens in the standard 77-token context. Encode once,
unit-normalize 512-dimensional vectors. No fine-tuning or encoder selection.

For each support review select a DIFFERENT local TRAIN user reviewing the
same support item whose non-whitespace word count is closest; break ties by
SHA256(review-owner-v1:<other-user>:<item>). Donors must themselves be uniquely
mapped remaining support events, with nonempty text. Same-item matches may
reuse a donor and are not a degree-preserving permutation; record this limit.
If no valid donor exists or own text is empty, remove that edge from ALL
profile arms. This equal mask preserves history item identity and count.

Three user profiles average and unit-normalize: own review, same-item donor
review, and the existing frozen item's global text vector. Score each against
the same unit-normalized sum of existing global image and text item vectors.
Zero profiles or zero target item vectors score zero; no fallback. Use each
user's same 16 deterministic unobserved comparisons from dual-witness,
excluding ALL original TRAIN positives, ties=0.5.

The mechanism comparison is within equal-mask text arms, not versus BRIDGE.
Review and item text are the same encoder but different text coverage; item
bank text pools up to eight chunks. Record this difference. Item-content
profiling is a strong information control, not a capacity-matched trainable
baseline. The diagnostic fits NO numerical model and evaluates no official
validation or test labels.

## Frozen gates and stopping rule

Require at least80% of ALL diagnostic users to have nonzero matched profiles;
own mean AUC at least0.005 above BOTH same-item donor and item-content controls;
paired user bootstrap2000 95% lower bound>0 for both; and positive deltas
against both controls in each of the two fixed user-hash halves used by
dual-witness. Missing/nonfinite metrics or identity-isolation violations stop.
No search over token length, target, encoder, mask, wordcount, seed or gate.

Baby only, one extraction/diagnostic, six-hour budget including GPU waiting,
four CPU threads, >=12GiB host RAM and >=14GiB free on one GPU (prefer GPU1).
Don't reserve GPU0 used by other jobs. No unrelated process is terminated.
Run unit tests in the actual environment, commit and deploy an immutable
bundle, then launch a detached fixed job. A failure freezes this simple
review-expression mechanism. A pass only permits a separately committed
formal hypothesis with review baselines and future three-domain/five-seed
confirmation under ../../result/PUBLICATION_PROTOCOL.md.

Remote: /data/sunzixuan/review_signal_20261009_v1.
Branch: autoresearch/oct9-review-signal.
