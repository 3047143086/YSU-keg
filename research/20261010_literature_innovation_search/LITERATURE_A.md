# 2025–2026 primary-literature audit and executable interventions

Audit date: 2026-10-10. Status: proposals only. No scores, encoding, validation/test
access, training, or edits to old experiment bundles were performed by this audit.
Read first: `result/FAILED_EXPERIMENTS.md` and
`research/20261010_next_signal_assessment/ASSESSMENT.md`.

Proposal updated before metrics on 2026-10-10: the actual registered sampler uses
the official TRAIN-observed universe and exact-degree shuffled assignments to
the same fixed semantic pools. The original draft's full-catalog support and
coarse degree-bin pool reconstruction are superseded. The executable definition
and fixed decision gates are `research/20261010_interest_negative/PROTOCOL.md`.

**Run first: a PerPEFT-inspired interest-group negative sampler on unchanged MSCA.**
This needs only already available TRAIN interactions and catalog feature arrays on
Baby and Sports. It does not depend on Electronics SVD recovery, raw Sports images,
an LLM download, or a new feature version. It is an adaptation of a published
mechanism, not an official PerPEFT reproduction or an established novel method.

## Verified primary sources

| Source | Verified dates and status | Relevant published mechanism |
| --- | --- | --- |
| Sunwoo Kim, Hyunjin Hwang, Kijung Shin, [Personalized Parameter-Efficient Fine-Tuning of Foundation Models for Multimodal Recommendation](https://arxiv.org/abs/2602.09445), [paper HTML](https://arxiv.org/html/2602.09445v1), [authors' implementation](https://github.com/kswoo97/PerPEFT) | First and only listed arXiv version: 2026-02-10 06:30:03 UTC. arXiv comments and author repository identify WWW 2026. | Interest groups receive separate foundation-model PEFT modules. Section 3.3 samples negatives from items purchased by that group's TRAIN users. Appendix C.3.2 also tests LightGCN, so the evidence is not restricted to sequence encoders. |
| Haibo Xing, Kanefumi Matsuyama, Hao Deng, Jinxin Hu, Yu Zhang, Xiaoyi Zeng, [ESANS: Effective and Semantic-Aware Negative Sampling for Large-Scale Retrieval Systems](https://arxiv.org/abs/2502.16077), [paper HTML](https://arxiv.org/html/2502.16077v1) | First and only listed arXiv version: 2025-02-22 04:43:20 UTC. WWW 2025; DOI [10.1145/3696410.3714600](https://doi.org/10.1145/3696410.3714600). | Multimodal/behavioral semantic clustering and embedding interpolation alter the negative distribution. This establishes substantial prior art for semantic negative sampling; its complete method is not the proposed sampler below. |
| Yuki Yada, Sho Akiyama, Ryo Watanabe, Yuta Ueno, Yusuke Shido, Andre Rusli, [Improving Visual Recommendation on E-commerce Platforms Using Vision-Language Models](https://arxiv.org/abs/2510.13359), [paper HTML](https://arxiv.org/html/2510.13359v1) | First and only listed arXiv version: 2025-10-15 09:46:27 UTC. RecSys 2025; the paper gives conference dates 2025-09-22–26 and DOI [10.1145/3705328.3748128](https://doi.org/10.1145/3705328.3748128). | In-domain product image/title pairs fine-tune SigLIP, then the image encoder supplies retrieval embeddings. The published task is product-image similarity recommendation, not this random-split implicit benchmark. |
| Zijie Lei et al., [UniRec: Unified Multimodal Encoding for LLM-Based Recommendations](https://arxiv.org/abs/2601.19423) | Submitted 2026-01-27; revised 2026-01-29 and 2026-04-30 (v3). arXiv journal reference: TMLR 2026. | Typed attribute name/type/value representations distinguish categorical and numerical fields and their nested item/history structure. Useful next-input precedent, but not selected for immediate execution because complete Sports raw catalog coverage remains unverified. |

Only paper and author-code claims above are attributed to the sources. The
following experimental definitions are our proposed adaptations and require their
own protocol, implementation audit, commit and frozen stopping rule before metrics.

## Candidate A: TRAIN interest-group negative distribution

### Independent intervention and formula

Keep the original MSCA feature files, graph, architecture, parameters, forward,
contrastive terms, optimizer, BPR objective, batch order, validation evaluator and
stopping rule. Replace only the sampler's distribution for its existing BPR
negative item. No additional edge is written and no gate-only objective is added.

Let `H_u` be the original TRAIN item set of user `u`, and `U = union_u H_u` the
sorted unique **TRAIN-observed item universe**. `U` is the support used by the
official MSCA TrainDataLoader's `all_items`; full catalog IDs outside `U` are
not sampled as training negatives. The full catalog remains the validation
ranking scope. Never use validation/test positives for exclusion.
Construct a normalized grouping vector from the original frozen image/text arrays:

```
x_i = concat(normalize(v_i), normalize(t_i))
h_u = normalize(mean_{i in H_u} x_i)
c(u) = frozen_kmeans8(h_u)
P_c = union_{u: c(u)=c} H_u
q_u(j) = 0.5 * Uniform(U \ H_u)
       + 0.5 * Uniform(P_c(u) \ H_u)
L_BPR = -log sigmoid(s(u,i_positive) - s(u,j))
```

`h_u` only determines an immutable sampling pool; it never initializes model
embeddings or enters scores. Freeze 8 groups, seed 2026, KMeans implementation,
float precision, normalization, initialization count, iteration limit and empty
history assignment before training. No search over groups, mixture weight or
semantic temperature. If a group pool is empty after removing own TRAIN positives,
use uniform and record the realized fallback; a pool with positives only must not
cause a rejection loop or treat a positive as negative. The 50% uniform component
keeps a chance of drawing any eligible TRAIN-observed item; it does not solve false
negatives.

### Mechanism controls

Use three fresh arms with the same seed, batches and model initialization:

1. Original uniform negative sampler.
2. True interest groups, fixed 50/50 mixture.
3. Group labels permuted within frozen **exact TRAIN-degree strata**, while
   retaining the same true semantic pools; use the same 50/50 mixture.

Permutation preserves each exact degree-stratum-by-group count, global group
headcounts and positive-pair mass. It assigns a shuffled user the original pool
label of its permuted source user; **it never rebuilds pools from shuffled
members**. Use a private seeded generator; fixed points are allowed and reported.
Sorting users/items and explicit tie behavior make the assets deterministic. A
single-user degree stratum cannot create a contrast, so report source-user fixed
points and the unchanged-label fraction. Same-size user groups do **not** imply
equal-size item pools or equal negative-item popularity; report both rather than
claiming they are matched. Use common mixture coins and rank variates in arms
2/3, with deterministic mapping into their sorted eligible pools. Sampling must
not advance the global Torch/NumPy generator used by model initialization,
dropout or batch order. The original evaluator receives identical full-catalog
scope and TRAIN-positive masks.

The true arm must beat uniform and shuffled on both validation R@20/N@20 without
the existing material R@10/N@10 regression. The registered pilot requires at least
0.5% relative gains on R@20/N@20 against each reference and absolute R@10/N@10
changes of at least -0.0002 against each reference. It adds no statistical
significance gate; single-seed user intervals, if reported, are exploratory.
Do not reinterpret a shuffled tie as semantic personalization. Report drawn
negative TRAIN degree, pool size, fallback/eligible rates and item coverage as
mechanism diagnostics; do not use validation/test positives to remove alleged
false negatives. A future popularity-matched control can isolate an explanation
left unresolved by this pilot, but cannot be added selectively to rescue its gate.
Exact-degree counts and fixed pools preserve unfiltered expected pool exposure,
but excluding a user's own positives changes eligible item marginals differently
between semantic and shuffled assignments. Record this remaining popularity/
difficulty confound rather than call the control fully popularity-matched.

### Feasibility and interpretation

The only setup costs are one pass over feature rows, TRAIN profile accumulation,
8-center KMeans and sorted pool construction. These fit a four-thread CPU process;
the ranker uses the same single GPU as the original MSCA. No foundation encoder
forward/backward is added per batch. Precompute rejection-free eligible sampling
rows or use a bounded, exactly specified rejection method. Freeze at most 8 hours
per arm; a timeout is a technical failure, not a recommendation gain. Baby first;
Sports confirmation follows only its fixed gate. This experiment can actually
train now once root freezes/tests the sampler and verifies resource ownership.

F13 failed profile initialization/semantic-BPR on fixed features; this does neither.
F18 failed candidate-history Top3 inference correction; this changes no inference
formula. F22 used model Top200 negatives and gate-only versus joint supervision;
this uses TRAIN item unions and the original loss for all original parameters.
F02–F04/F23 changed graph/evidence relations; this writes no graph. Thus it is
different from the frozen implementations, while still carrying the general risk
that reorganizing existing information adds no useful signal.

The new identifiable variable is the **user-conditioned comparison distribution**,
not a newly observed dislike. Unobserved items bought by similar users can be
false negatives. A gain may reflect popularity, difficulty or optimization rather
than a correct aspect-level preference. Group-based and semantic negative
sampling already have direct PerPEFT/ESANS precedents. A successful Baby seed only
supports this adaptation's pilot; it cannot establish novelty or publication
readiness. Full cross-dataset/five-seed evidence and an independently distinctive
mechanism are still required.

## Candidate B: group-conditioned CLIP token-level LoRA

This tests a different variable from A: whether **the encoder itself** needs
interest conditioning, rather than the negative sampler. It is a later feature
condition experiment, not a silent replacement of VGG4096/text384 inside the old
baselines.

For the existing OpenAI CLIP ViT-B/16 catalog text Transformer, freeze base weights
and add rank-4 LoRA to query/value projections. Keep the visual branch fixed:

```
W_Q,c = W_Q + B_Q,c A_Q,c
W_V,c = W_V + B_V,c A_V,c
z_i,c = normalize(concat(fixed_CLIP_image_i, CLIP_text_i(theta_c)))
L = fixed support-only co-purchase pair contrastive loss
```

Only the internal text LoRA differs between a single global adapter, 8 true
TRAIN-interest adapters and 8 degree-stratum-shuffled adapters. Use matched total
LoRA parameter count for the large-global control (rank32 versus 8×rank4), the
same pair/negative lists and a fixed optimizer-step budget. The same genuine
TRAIN-only group labels and exclusions determine examples in every arm; do not
also change their negative distribution while testing encoder personalization.
All internal diagnostic target edges are removed globally before grouping,
forming pairs or fitting. Use the already frozen TRAIN internal target/comparison
list for downstream discrimination, plus ordinary popularity/KNN controls.

This is not F07's external adapter on already compressed fixed features: gradients
alter which catalog tokens the Transformer uses before compression. It does not
reopen the mean-review encoder/mask family from F24–F27. A mean support scorer may
be held identical across arms for a diagnostic, but a positive diagnostic is not
automatically a trained-ranker result.

Feasibility: Baby already has catalog titles/description mapping, images and a
CLIP-B/16 checkpoint. OpenAI CLIP packs Q/K/V into `in_proj_weight`; inserting
Q/V LoRA requires a tested explicit implementation rather than using HF target
module names unchanged. Full image backprop is deliberately outside this minimal
text-only proposal. A fixed 2,048 steps, batch32, four CPU threads and 8-hour
ceiling per arm is a bounded proposal, **not a measured runtime guarantee**. Image
features can be reused only when their item alignment and feature version are
audited. Sports catalog mapping/coverage must first be verified. No new model
download is required by this text-only adaptation.

The author code is not directly compatible: [official LoRA file](https://raw.githubusercontent.com/kswoo97/PerPEFT/main/perpeft_lora_ia3.py)
uses HF CLIP-B/32 and SASRec; it reads different Amazon2023 inputs and evaluates
test each epoch. Its published group treatment and negative sampler are combined.
Our adaptation must prohibit the test path and isolate the variable. Do not label
this B/16/support-pair diagnostic an official PerPEFT reproduction. PerPEFT has
already published the central group-PEFT idea; any successful transfer is a
baseline/mechanism result unless a separately supported contribution emerges.

## Candidate C: catalog image/title correspondence adaptation

This introduces a supervision source absent from fixed global/local/crop
similarity tests: observed **catalog image–title pairing**, without owner reviews
or inferred user-item edges. Freeze the existing CLIP text encoder and update
only last-two-block visual query/value rank-4 LoRA on raw catalog images:

```
a_i = normalize(CLIP_image(image_i; theta))
b_i = fixed normalize(CLIP_text(title_i))
L_pair = -(1/B) sum_i log[ exp(a_i·b_i / 0.07)
                         / sum_j exp(a_i·b_j / 0.07) ]
```

The sole intervention is training with correct catalog image/title correspondence.
Controls are frozen B/16, the same trainable architecture trained with a fixed
within-category/brand-available title permutation, and the correct-pair arm. Match
steps, images, title eligibility and optimizer. Do not synthesize item labels or
use a validation purchase to decide a pair. Define category/brand-null strata and
the permutation/fixed-point rate before metrics. Hold a fixed item-hash subset out
of this domain fitting to distinguish association memorization from transfer;
all-item recommendation uses only public catalog content and the official TRAIN
graph after fitting, with no validation/test content labels.

Baby has 7,050 catalog rows and 7,037 raw image files in the prior asset audit;
actual readable-image/title intersection must be frozen rather than presumed.
For at most 7,050 rows, 10 fixed epochs and batch64 imply about 1,110 optimizer
steps. Four CPU image-loader workers and one GPU are sufficient in principle, but
record a short no-metric memory/throughput smoke before committing a walltime
estimate. The ceiling remains 8 hours per arm. Sports raw catalog/image coverage
is an outstanding dependency, so A is more immediately executable on both domains.

New feature banks require new matched frozen-B/16 carrier controls; gains versus
old VGG/text384 historical baselines conflate feature version and algorithm. Never
compare the proposed contrastive training to F24's reviews or rename crops as
domain adaptation. Yada et al. already establish product-pair domain tuning as
prior art, using SigLIP and much larger proprietary data. This CLIP/LoRA adaptation
is independently testable but has weak standalone method novelty; a successful
small-catalog transfer could support an empirical encoder study only after strong
controls, more datasets and repeated seeds.

## Directions not selected

NegGen ([2025-01-25; revised 2025-08-21](https://arxiv.org/abs/2501.15183), ACM MM)
requires MLLM-created counterfactual item attributes and its causal module.
Calling simple feature swaps an official NegGen result would be inaccurate; no
verified local generator/extraction validation set currently makes it an immediate
eight-hour option. Typed UniRec-style brands/categories are real new catalog
fields, but there is no verified Sports raw schema coverage and old factorization
models already used categorical attributes. Usage/compatibility extraction and
summary/body subtraction remain rejected under `ASSESSMENT.md`.

Latest-date routing papers also do not override frozen gate/routing failures.
No result from this literature search permits reopening F13, F18, F19 or F24–F27,
relaxing a prior threshold, restarting an immutable numerical failure, or claiming
that the next idea will produce a paper. The actionable recommendation is A with
fresh matched training; B/C remain separately frozen future hypotheses.
