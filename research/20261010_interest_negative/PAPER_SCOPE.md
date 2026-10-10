# Research and paper scope of the interest-negative pilot

Status: pre-metric exploratory protocol, 2026-10-10. No recommendation benefit,
statistical significance, novelty, successful training or publication readiness
is established by these documents.

The tested question is narrow: **does a static TRAIN-interest-conditioned
negative distribution improve an unchanged official MSCA carrier over fresh
uniform training and a fixed-pool, degree-preserving shuffled assignment?**
The only modification is a 50% sampling mixture. It adds no new observed dislike,
no new visual/text representation, no foundation-model adaptation and no new
inference mechanism.

## Direct prior art and what is not claimed

[PerPEFT, WWW 2026](https://arxiv.org/html/2602.09445v1), by Sunwoo Kim,
Hyunjin Hwang and Kijung Shin, assigns separate PEFT modules to user-interest
groups and uses negatives from items purchased within each group. Section3.3 is
direct prior art for this training technique; its non-sequential LightGCN appendix
does not validate our sampler-only MSCA adaptation. Our means of frozen original
features for grouping differ from the paper's trained representation, and we
retain one unchanged MSCA. We are not reproducing the complete published method
or claiming that this is a new group-negative algorithm.

[CAMMSR, ICDE 2026](https://arxiv.org/abs/2603.04320) already uses category-guided
multimodal experts and auxiliary category prediction. Neither interest grouping,
multimodal supervision nor a newly used metadata field alone establishes
originality. [Privileged Self-Distillation, 2026 preprint](https://arxiv.org/abs/2607.27055)
also separates training information from deployment inputs, but its sequential
future-aware construction is not part of this random-split pilot.

A descriptive experiment directory is not a new algorithm name. Do not invent an
acronym, claim first-of-kind personalization, promise a paper, or treat the
originality of a local implementation as research novelty. Even a successful
single-seed pilot would need a separately justified distinctive mechanism and
substantially stronger confirmation before a positive-method paper claim.

## Relation to the frozen failures

| Prior family | Intervention boundary in this pilot |
| --- | --- |
| F13 mean semantic initialization/adapter | The TRAIN mean only partitions sampling pools; it never initializes or changes learned user representations. |
| F18 candidate-history inference residual | No reranker, candidate cutoff, history residual or inference correction. |
| F22 candidate negatives assigned to gate gradients | No gate or gradient ownership intervention. No current-model Top200 negatives. Only the original MSCA BPR negative argument changes. |
| F02-F05/F23 graph/evidence relations | No added edge, changed topology, dual witness or new propagation evidence. |
| F01/F06/F07/F12/F14/F15/F17 existing-view transformations | No adapter, router, confidence table, modality-loss reweighting or calibration sweep. |
| F19 temporal graph | No sequence or causal-time assertion on the original random split. |
| F24-F27 review profiles/expression/bank geometry | No review input or residual/bank weighting. Stable author expression is not reused as preference evidence. |

These distinctions justify a different pre-registered intervention; they do not
erase the cumulative failures or demonstrate independent new information. Failure
of the fixed K8/mix0.5 point freezes it. Renaming it, changing groups/mixture,
swapping seeds or quietly expanding its budget cannot rescue the same hypothesis.

## What a result could and could not support

A complete Baby pass would support spending the pre-authorized Sports budget on
this one fixed adaptation. A complete two-domain pass would support further
confirmation of the adaptation. An incomplete arm, smoke, favorable intermediate
epoch, baseline/control collapse, smaller subgroup or one metric cannot replace
the protocol's completed two-reference conjunction.

Same fixed pools and exact-degree shuffling conserve group headcounts and
positive-pair mass and improve comparability of unfiltered expected pool exposure.
They do not fully isolate semantic meaning: excluding each user's own positives
changes eligible marginals, related purchases may be false negatives, and
popularity/difficulty can affect training. Report pool/negative statistics and
fixed points with every performance result. Do not call this a causal experiment
on exposure or proof of visual/text aspect reasoning.

The validation pipeline uses full-catalog ranking and identical held-out masks,
but training uniform support is the original TRAIN-observed item union. Test data
is outside this search loop. Past test exposure remains disclosed; fresh training
does not make a previously exposed split unseen. Smoke is engineering evidence
only; all formal arms restart from the same fresh within-domain initialization.
One seed supplies no training-randomness estimate and no statistical-significance
claim. User bootstrap, if shown, only describes users at that seed.

The official metric source references the removed `numpy.float` alias in the
audited NumPy 1.24.4 environment. The wrapper installs `np.float = float` while
preserving the protected evaluator bytes and metric formula. Official validation
and the final ranking check run under `torch.no_grad()` with score/metric
finiteness guards. These identical engineering safeguards are not a method
intervention or performance result.

The unchanged confirmation contract is `result/PUBLICATION_PROTOCOL.md`: Baby,
Sports and Electronics, five fixed seeds2026-2030, original BRIDGE and relevant
matched controls plus strong equal-budget official baselines, fixed selection,
seed-level uncertainty, multi-comparison handling, attribution, measured cost
and frozen final test evaluation. This fresh MSCA pilot does not waive or satisfy
those BRIDGE-specific confirmation requirements. Electronics numerical failure
is infrastructure evidence and no performance score; its frozen SVD recovery is
not restarted or weakened by using a different carrier for this pilot.

Report `discard`, `crash`, incomplete and positive results with the same provenance
and limitations. A transparent negative or adaptation study is a valid research
outcome, but these documents cannot guarantee any journal/conference acceptance.
