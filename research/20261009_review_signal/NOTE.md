# Review Signal Input Boundary

This note records the input audit for the next, review-based experiment. It is
not a method result and does not change the frozen experiment protocol.

## Observed files

On server 213, the Baby review source is available at
`/data/sunzixuan/rave_20260905_inputs/baby/reviews_Baby_5.json.gz`. Its SHA256 is
`e54968e4102a3f03f4bb6be581dc09188fe2a31e4a664803ff21a3ad2ac578c3` and it
contains 160,792 raw records. The matching Baby interaction file is
`/data/sunzixuan/bridge_official/data/baby/baby.inter`, with 118,551 TRAIN,
20,559 validation, and 21,682 test rows. The item mapping has 7,050 unique
item IDs and ASINs.

The review schema includes `reviewerID`, `asin`, `reviewText`, `summary`,
`overall`, `unixReviewTime`, and related metadata. The audit used ratings and
timestamps only for exact identity matching. It did not export reviewer IDs or
review text in its receipt, and it did not use validation or test content.

On server 213, the available Sports directories
`/data/sunzixuan/expirement/LOBSTER/data/sports` and
`/data/sunzixuan/bridge_official/data/sports` contain the interaction file,
precomputed image/text interest tensors, and popularity masks. No raw review
archive, item-to-ASIN mapping, or review-user mapping is present there.

On server 214, `/data/gongshanghui/sun_exp/LOBSTER` exists and contains the
Sports interaction/features directory, but the bounded file search found no
raw review archive, item metadata archive, item-to-ASIN mapping, or reviewer
mapping. The 214 copy has not been identity-audited against the 213 inputs.

Therefore the current review experiment can be an isolated Baby mechanism
screen only. A Baby result cannot be described as a cross-domain confirmation.
Obtaining and mapping Sports reviews would require a separate frozen data
preparation and feature audit; mixing 213 and 214 interaction IDs or existing
feature caches is invalid.

## Matching and exclusion rule

The implemented audit reads only `x_label=0` interactions. It selects one
deterministic internal target for every TRAIN user with at least four TRAIN
items using the existing `dual-witness-target:v1:` hash rule, removes all such
edges globally, and then matches remaining events by `(asin, timestamp,
rating)`. A reviewer identity is accepted only when the complete support
history identifies one owner; ambiguous, duplicated, missing, or colliding
owners are rejected rather than guessed. Ratings are identity keys only, not
model values.

The resulting Baby audit matched all 105,104 non-target support edges and all
19,445 support users uniquely. It found 3,008 single-event fingerprints with
multiple raw owners, but complete support histories resolved them without
guessing. Review text was nonempty for 105,065 support edges. These are input
coverage facts, not recommendation evidence.

## Prior-art boundary

Review text as a user/item representation is established by
[DeepCoNN](https://arxiv.org/abs/1701.04783). The critical review of review-based
recommendation reports that many claimed gains are setting-sensitive; see
[Sachdeva and McAuley](https://arxiv.org/abs/2005.12210). Review-property
weighting and personalized review retrieval are also established directions,
for example [RPRM](https://arxiv.org/abs/2102.03089) and
[ERRA](https://arxiv.org/abs/2306.12657).

Consequently, adding review text, mean pooling reviews, or using an attention
layer is not by itself a novelty claim. Any later method must first pass the
pre-registered identity and same-item permutation controls and must be framed
as an empirical mechanism result unless a broader literature review establishes
a real method distinction.

Last audit: 2026-10-09. No review encoding, metric evaluation, or training was
started by this audit.
