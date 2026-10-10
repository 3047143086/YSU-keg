# Independent literature and frozen-family boundary audit

Audit date: 2026-10-10. This document is a proposal and implementation review,
not a training result. Only this report was written. No experiment source,
protocol, threshold, input, validation/test data or running process was changed.
The authoritative frozen record is `result/FAILED_EXPERIMENTS.md`.

**Recommendation: start one pre-registered interest-group negative-sampling
pilot on an unchanged, freshly baselined carrier.** It is implementable with
existing TRAIN interactions and frozen Baby/Sports feature arrays. It is a
published-mechanism adaptation, not an established new method or a demonstrated
independent preference signal. Do not delay it for an additional review-profile
diagnostic or for Electronics numerical recovery.

## What the recent primary literature actually establishes

| Primary source | Date/status verified on 2026-10-10 | Implication for this workspace |
| --- | --- | --- |
| Kim, Hwang and Shin, [PerPEFT](https://arxiv.org/abs/2602.09445), [full paper](https://arxiv.org/html/2602.09445v1), [author code](https://github.com/kswoo97/PerPEFT) | Submitted 2026-02-10; WWW 2026, DOI 10.1145/3774904.3792248. | Section 3.3 restricts negative items to purchases within each interest group, alongside group-specific PEFT modules. Appendix C.3.2 includes LightGCN. Thus group-negative sampling is direct prior art, and transferring only that component to an unchanged ranker tests an adaptation rather than reproducing the full method. The paper itself notes that a single hard group cannot represent multi-interest users. |
| Zhou et al., [Does Multimodality Improve Recommender Systems as Expected?](https://arxiv.org/abs/2508.05377) | Submitted 2025-08-07; preprint. | Its controlled analysis motivates retaining strong collaborative baselines and separating retrieval, ranking and modality contributions. It does not establish that a new gate or a larger encoder necessarily improves this benchmark. |
| Xu et al., [CAMMSR](https://arxiv.org/abs/2603.04320) | Submitted 2026-03-04; paper states accepted ICDE 2026. | Category prediction guides a mixture of modality experts; category-based multi-interest or auxiliary supervision already has recent multimodal-recommendation precedent. Metadata is a possible independent observed input here, but using it is not itself a novelty claim. |
| Tang et al., [Learning from the Future: Privileged Self-Distillation for Sequential Recommendation](https://arxiv.org/abs/2607.27055) | Submitted 2026-07-29; preprint. | A future-aware teacher supplies training-only supervision to a prefix-only student. This supports separating training information from deployment inputs, but its future/prefix construction cannot be silently imported into the current random-split non-sequential task. |

The following sampler and controls are our proposed experiment, not claims made
by those papers. No foundation-model download, paid API, fresh split or future
interaction access is needed for the first candidate.

## Boundaries imposed by all frozen families

| Frozen family | Concrete boundary for the next experiment |
| --- | --- |
| F01, F06, F12, F14, F15, F17: routing, confidence/calibration, alignment or fusion of existing views | Do not call another modality weighting, temperature, reliability loss or fusion sweep new information. Group labels here may select a negative distribution only; they do not enter a gate, loss weight or inference score. |
| F02-F05: prediction/content-derived pseudoedges and graph changes | Keep all interaction and item-neighbor graphs byte-identical. A sampling pool is not added to graph propagation. |
| F07: fixed-feature adapters | Add no external feature adapter. Group sampling does not justify reviving rank/adapter/alignment sweeps. |
| F08, F13, F18: history residuals, semantic user initialization and candidate-history reranking | TRAIN profiles can construct immutable groups; they must not initialize learned user embeddings, supply a residual, or alter inference. This reuses a grouping statistic, not the rejected downstream intervention. |
| F09-F10: anchor reweighting and gradient surgery | Preserve batch/positive order and original auxiliary losses. Add no group-equalized CL anchors or gradient routing. |
| F11: star-rating polarity | Ratings are not used as explicit negatives or polarity labels. The proposed negative remains an unobserved implicit item, not a measured dislike. |
| F16, F20-F22: correction or target/candidate gate supervision | Do not substitute current-model Top200 candidates, isolate gate gradients, change uniform soft targets or sweep the negative strength/scope after results. The new sampler is fixed offline and affects the original negative-item argument of the original training loss. |
| F19: temporal KNN | No timestamp ordering, next-item claim or temporal graph change under the existing random split. |
| F23: dual-witness collaborative evidence | Do not reuse hash-partition witness support or tune evidence K/target eligibility. No new evidence feature or graph relation is added. |
| F24-F25, F27: mean review profiles, item-direction residuals and review-bank geometry | Do not reopen review encoder/truncation, residual strength, donor mask, bank mixture or thresholds. Their failure cannot be erased by adding the same scores to a new training arm. |
| Completed expression diagnostic (F26 in the cumulative sequence) | Author-expression association passed; recommendation transfer did not. It is not evidence that user review style identifies desired items. |

The distinction from F22 is material: F22 assigned model-candidate supervision
only to the gate and used a joint-update hard-negative control. This proposal
uses static TRAIN interest pools and the unmodified carrier loss. It adds no
measured preference fact beyond existing TRAIN behavior/features. A successful
pilot would show a useful optimization/comparison distribution, not prove that
group semantics or visual/text aspects caused the improvement.

## Candidate 1: fixed TRAIN interest pools, with a stronger shuffle control

Let `H_u` be all user TRAIN positives, `I` the catalog, and `x_i` the concatenated
separately normalized original frozen vision/text rows. Construct
`h_u = normalize(mean(x_i for i in H_u))`, freeze KMeans with eight groups, and
freeze `P_c = union(H_u for users assigned to c)`. Use one pre-registered sampler:

```
q_u(j) = 0.5 * Uniform(I \ H_u)
       + 0.5 * Uniform(P_c(u) \ H_u)
```

Keep the original learned parameters, forward, loss formula, optimizer,
initialization seed, feature files, graph, positive/batch order, evaluator and
stopping rule. No group-specific adapter, loss, auxiliary head or score is added.
If the carrier is changed from historical BRIDGE to original MSCA, run a fresh
MSCA uniform baseline; the old BRIDGE metrics are not a valid matched baseline.
For original MSCA, `neg_items` enters BPR while CL terms use users/positive items,
so the source-level intervention can stay confined to the sampler. For BRIDGE,
audit every use of `neg_items` and describe all consequences of replacing that
argument; do not describe an unchanged BPR formula as gate-only learning.

Run baseline uniform, true-group mixture, and shuffled-assignment mixture. The
cleaner shuffled control keeps the **same eight true pools** and permutes only
`user -> pool` assignments within exact TRAIN-degree strata. Do not reconstruct
the pools from shuffled users. Exact degree-stratified permutation preserves
each `degree x group` count, group headcounts and the unfiltered expected pool
exposure weighted by each user's number of positive training examples. Fix a
private RNG, sorted identifiers, permutation behavior, allowed fixed points,
normalization, clustering implementation and failure/fallback rules in advance.
Record what proportion of users actually changes group: singleton or single-
label degree strata cannot destroy all user-pool association.

This is stronger than shuffling labels within broad log2-degree bins and then
rebuilding pools. That alternative changes pool unique sizes, popularity
distributions and effective difficulty together with semantics; equal group
headcounts do not match those quantities. It is a useful generic random-group
control, but cannot alone isolate meaningful interest matching.

Even the stronger control is not perfectly popularity-matched after removing
own TRAIN positives. Every true user's `H_u` is a subset of its true pool,
whereas fewer of its positives may occur in the permuted pool. Conditioning on
`P_c \ H_u` therefore changes eligibility and final negative-item marginals.
Save per-pool size and TRAIN item-popularity deciles, per-user eligible mass,
fallback rate, realized negative popularity and duplicate rates in both arms.
Common mixture coins and variates reduce sampling noise; they do not make
filtered item marginals identical. A win over this control must retain that
limitation. Do not claim a causal removal of popularity/exposure confounding.

Use uniform over unique pool items as written; frequency weighting would be a
separate intervention unless frozen before this run. Empty eligible pools fall
back to the same original uniform rule. Prevent infinite rejection loops and
exclude every user TRAIN positive. Do not use validation/test positives to
exclude alleged false negatives. Related users' unobserved purchases can still
be false negatives; the 50% uniform component does not resolve that issue.

Expected outcome to test: true-group mixture improves both validation R@20 and
N@20 over the matched uniform baseline and shuffled control, with the previously
fixed shortlist regression guard. Root must freeze exact margins, paired-user
interval requirements and terminal decision rules before metrics. Coverage,
negative hardness or a Baby-only subgroup gain cannot replace the primary
cross-dataset performance gate. Do not scan eight groups, 0.5 mixture, seeds or
pool rules to rescue failure. The next step is the actual sampler implementation,
training startup audit and bounded Baby pilot, followed by the already declared
Sports confirmation rule, not another literature-only round.

## Candidate 2: metadata multi-interest sampling, conditional on input audit

A more independent observed signal would be existing catalog category labels,
not a different pooling of the same frozen visual/text vectors. For a TRAIN
positive `(u,i)`, select a category from `i` using one frozen category-depth and
multi-label rule, then draw the group component from other eligible TRAIN-seen
items in that category. The 50/50 original-uniform mixture and full TRAIN-positive
exclusion stay identical. Sampling per positive expresses multiple interests
without forcing a user into one group. It still adds no inference input or head.

The matched control should retain the same fixed category item pools and
positive-pair frequency per pool while permuting the positive-pair/category
assignment within pre-fixed item TRAIN-popularity strata; changing categories
and rebuilding pools confounds taxonomy meaning with negative popularity and
pool size. Exact pair counts, category frequency, degree and own-positive
eligibility must be reported. This is a different hypothesis, not a parameter
variant to append selectively after Candidate 1's metrics.

Readiness is currently conditional: this audit has not established an aligned,
complete Baby **and** Sports raw category/brand schema. Check ASIN-to-item mapping,
row uniqueness, category paths/depth, missing rates and raw file hashes first.
Do not silently fetch a different Amazon release or mix category coverage masks
with the frozen old populations. This option is more independent of the fixed
embedding bottleneck and F13/F18/F22, but taxonomy-guided multimodal learning has
CAMMSR and other prior art. It is not a novel mechanism solely because a category
field is new to this workspace. It should not block the feasible Candidate 1
training launch.

## Interpretation and integrity limits

F24's own-review advantage over same-item donors was only +0.004564 mean AUC,
and raw item content was better by 0.021882. F27's residual candidate scored
0.588003 versus item content 0.588482; raw review and raw donor banks were almost
equal at 0.670312/0.669071, while popularity scored 0.757632. These exact frozen
controls make another review-profile geometry a poor immediate investment.
Training-only review prediction is logically a different intervention, but it
would require a new same-item-donor capacity-matched experiment and Sports input
audit; author-style identifiability alone does not justify it.

Electronics remains a separate failed numerical feasibility version with zero
completed seeds after the fixed spectral boundary guard rejected epoch0/batch4.
It supplies no innovation score or valid negative performance result. Neither
relaxing that guard nor restarting its frozen recovery is part of this proposal.

No pilot has been performed by this audit. Successful single-seed Baby/Sports
adaptation would support further confirmation, not publication novelty. Keep
the experiment's terminal status, all arm metrics, provenance, crash/timeout
outcomes and old failures visible together.
