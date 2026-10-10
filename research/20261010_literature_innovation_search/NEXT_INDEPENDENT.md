# Next independent hypothesis: catalog-composition supervision

Audit date: 2026-10-10. This is a read-only proposal written while the
PerPEFT-inspired negative-sampling queue is running. No training process,
source checkout, evaluator, split, threshold, or queue state was changed.
The queue's true-group arm has now failed the frozen Baby primary and shortlist
gates, so its mixture, K, and seed must remain frozen and must not be used to
motivate a sampler rescue.

## The proposed signal

The strongest next candidate is a **TRAIN-only catalog-composition relation
auxiliary**. It introduces explicit relation labels absent from the present
training inputs: the original Amazon item metadata has structured `brand` and
`categories` fields. These facts may already be partially encoded in title
embeddings; information-theoretic independence is not presumed. The local
schema audit found Baby `items.jsonl` rows with
`item_id`, `asin`, `title`, `description`, `brand`, `categories`, and `imUrl`;
the recovered Baby path is
`/data/sunzixuan/rave_20260905_inputs/baby/items.jsonl`. Sports coverage and
path, nonempty-field coverage on both domains, and hashes remain unverified.

The hypothesis is that an item representation that preserves *which catalog
field is shared* will transfer better to sparse user-item ranking than one
trained only by BPR and the existing MSCA alignment terms. This is a
compositional relation hypothesis rather than a claim that a brand or category
is a user dislike/like label. The main recommendation objective remains the
only objective used for the official ranking decision.

For each item in the TRAIN-observed item universe, canonicalize a nonempty
brand and the exact root-to-leaf category paths (Unicode NFKC, trim/collapse
whitespace, casefold; category signature is the sorted unique full-path tuples;
no root deletion, depth truncation or title parsing). Same category means equal
complete nonempty signatures. Build immutable item pairs with three labels:

1. same brand and different category signatures;
2. same category signature and different nonempty brands; and
3. different brand and different category signatures.

Pairs sharing both brand and category, self-pairs and empty/invalid fields are
excluded. This precise equality rule is fixed before the independent CPU asset
audit; no weaker overlap rule can replace it after coverage results.

Only pairs with both item IDs in TRAIN are used for the auxiliary objective.
Title and description are retained for coverage reporting but are not parsed
into attributes: regex-derived "capacity", "compatibility", or "usage" facts
would be unvalidated pseudo-labels, especially under this random split. Empty
metadata is an explicit abstention, never an imputed category.

The one intervention adds a small relation classifier to the existing item
embeddings. If `z_i` is MSCA's item representation returned by the unchanged
forward path, the auxiliary input is
`[z_i, z_j, |z_i-z_j|, z_i*z_j]`; a fixed two-layer 64-unit head predicts the
three relation labels. The head is discarded at inference. The loss is

```text
L = L_MSCA + 0.005 * CE(relation_head(pair(z_i,z_j)), relation_label)
```

where `L_MSCA` denotes the exact unmodified BPR, existing CL and regularizer
losses. The coefficient `0.005`, one partner per item and relation class within
each item partition followed by unordered-pair deduplication, sorted IDs, and
private pair RNG are fixed before the first metric. Select endpoints by frozen hash order;
do not use learned similarity or interaction frequency to rank pairs. No
lambda, pair cap, field depth, or seed search is
allowed. The relation head receives no metadata input; it must decode the
learned representation, though it may decode an already represented fact rather
than force a useful geometry change. BPR negative sampling,
positive order, batch order, graph caches, model initialization, evaluator,
full-catalog mask, and early stopping stay byte-compatible with the fresh
uniform MSCA control.

The matched null arm instantiates the same head and exactly the same pair
endpoints and class counts, but permutes labels within unordered endpoint
degree-bin pairs (`floor(log2 TRAIN_degree)` for each endpoint). This keeps
the exact pair endpoint/marginal popularity distribution and preserves each
stratum's class counts. It does not equalize difficulty inside a stratum or
prove a causal effect. Strata with only one label stay unchanged; require at
least 50% of pair labels to actually change or declare the control infeasible.
A third arm is fresh uniform MSCA with no relation loss. Keep the head outside
the original model so `cal_reg_loss()` still sees exactly its original parameter
set; use a private RNG for head initialization after model construction.
Explicitly save/restore all global RNG states during head preparation and use
no dropout in the new head. Update head parameters in a separate optimizer group
with zero weight decay; original optimizer and regularizer stay unchanged.

## Why this is independent

This intervention changes a training-only supervision source, not the
comparison distribution. It does not add an item-item graph, pseudo-edge,
KNN, temporal relation, candidate reranker, gate, modality weight, residual,
profile initializer, review representation, or gradient surgery. It therefore
does not reopen F01--F27 families: F01/F06/F12/F14/F15/F17 are routing or
calibration; F02--F05 and F23 alter graph/evidence structure; F07 adapts fixed
features; F08/F13/F18 are history or candidate residuals; F09/F10 reweight
anchors or gradients; F11/F24--F27 use review/rating signals; F16/F20--F22
supervise candidate gates or change hard-negative targets. The active
PerPEFT-inspired experiment changes only original negative-item sampling and
has no relation auxiliary head. This candidate leaves that sampler uniform in
all three arms.

The new supervision is the exact *cross-field relation* in catalog
metadata, with a scientific null that preserves pair endpoints and class
marginals. It
does not claim to observe a new user preference. A gain could still reflect
regularization or popularity unless the null and the mechanism diagnostics
agree; a performance gain without relation preservation is not evidence for
the proposed explanation.

## Primary-literature boundary

Recent primary work makes a novelty claim from "using categories" untenable.
[UniRec](https://arxiv.org/abs/2601.19423) (2026) represents typed
attribute name/type/value tuples and heterogeneous item fields, but studies an
LLM/Q-Former sequential model with a different task and inputs.
[CAMMSR](https://arxiv.org/abs/2603.04320) (2026) uses category-guided mixture-of-
experts in multimodal sequential recommendation; this proposal has no expert
gate or category-conditioned inference.
[HGCRec](https://doi.org/10.3390/electronics15163633) (2026) constructs a heterogeneous graph with category and
brand entities and relation-aware propagation; this proposal constructs no
graph. Earlier [AD-DRL](https://arxiv.org/abs/2312.14433) explicitly assigns
category/brand attributes to disentangled multimodal factors.
[OpticalRec](https://arxiv.org/abs/2610.05432) (submitted 2026-10-04) renders title/brand/category metadata
into a unified visual encoder, which is also outside this frozen-feature
carrier. These papers support feasibility and identify the prior-art risk; a
successful result here would be an empirical, controlled transfer finding, not
a guaranteed new algorithm. Review-aspect directions such as
[CLARER](https://arxiv.org/abs/2610.07761) (submitted 2026-10-06) are excluded because the existing review
style diagnostics failed and the proposed input is catalog metadata only.

## Readiness audit and falsification

Before any GPU launch, read-only preparation must produce a canonical metadata
manifest for Baby and Sports containing source path, SHA256, item-ID join
coverage, nonempty brand/category coverage, duplicate-ID count, category-path
counts, pair counts by relation, and item-degree/popularity stratum counts. A
domain is **infeasible** if fewer than 70% of TRAIN-observed items have both
fields or any relation class has fewer than 1,000 deterministic fitting pairs
or 200 diagnostic pairs. These are go/no-go conditions, not knobs. The current inventory proves the Baby
keys, but does not prove these coverage thresholds or Sports availability.

For each domain run three fresh arms, same official MSCA recipe, seed, split,
epoch/patience budget, and GPU. Hash-partition TRAIN item IDs 80/20 before pair
construction; the head fits only pairs entirely within the 80% set and is
tested only on pairs entirely within the 20% set. The backbone may still use
all original TRAIN interactions, as in every arm. Report balanced relation
accuracy on these unseen auxiliary item IDs, using the true diagnostic labels
in every arm, against both the trained null head and a fresh readout trained on
frozen baseline embeddings. The latter uses the same fitting pairs, steps and
head initialization, with backbone gradients disabled, to test whether the
candidate improved representations rather than just fitting a decoder. This
is a mechanism diagnostic only and never selects an
epoch. The candidate must beat both uniform and null on validation R@20 and
N@20 by at least 0.5% relative, while R@10 and N@10 remain within -0.0002
absolute of each reference. In addition, held-out relation accuracy must exceed
both diagnostic references by at least 0.05; otherwise a ranking gain cannot be attributed to
composition preservation. Test labels remain outside selection. A single Baby
screen is exploratory; failure freezes this family rather than prompting a
field, pair-cap, or lambda sweep. Confirmation requires the same fixed rule on
Sports and eventually the existing multi-domain, five-seed evidence standard.

The principal risks are metadata missingness, category-path granularity,
brand/category popularity confounding, and the auxiliary head learning item-ID
memorization. The null controls pair volume and coarse difficulty but cannot
prove that the metadata caused a user-preference gain. Report item-degree
subgroups, pair coverage, relation confusion, and relation-head parameter
count. If the relation head is accurate but ranking does not improve, the
catalog fact is identifiable yet not useful for this carrier. If ranking
improves but the null also improves, attribute composition is not established.

## Cost and scope

Metadata parsing and pair construction are CPU-only. Actual preparation and
training throughput are unknown until the paths and a one-epoch engineering
smoke are checked. Use a proposed ceiling of three arms x two domains x eight
hours = 48 formal GPU-hours, with at most two additional smoke GPU-hours and
a 72-hour elapsed-time planning ceiling on the single permitted GPU after it
becomes available. This is a budget, not a measured estimate. The smoke is for memory and
schema only and cannot select a configuration. No new model download, paid API,
LLM, raw-image encoding, or split is required.

This proposal should be registered only after the active queue has terminated
and its failure evidence is recorded. It must use a fresh baseline commit and
the same validation-only evaluator. Even a successful result would support the
claim that exact catalog cross-field supervision transfers under controlled
conditions; it would not justify naming a new algorithm or promising a paper
without Baby, Sports, Electronics (if numerically feasible), five fixed seeds,
paired uncertainty, and comparisons to the primary relation-supervision prior
art.
