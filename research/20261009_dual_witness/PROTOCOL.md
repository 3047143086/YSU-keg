# Two disjoint history views supported by the same collaborative user

Declared 2026-10-09 before the full diagnostic. Exploratory mechanism screen,
not a performance or priority claim. Previous candidate-gate queue completed,
failed Sports and is preserved. No gate/strength/TopK rescue scan is allowed.

## Hypothesis and prior evidence

Global pairwise ItemKNN compresses away the identity of supporting co-users.
The unproven question is whether a co-user who overlaps two disjoint history
subsets provides evidence beyond ordinary UserKNN or a minimum-overlap rule.
This is additional TRAIN structure, not newly observed preference or causal
exposure information. User-neighborhood CF is established prior art, including
[SCCF](https://arxiv.org/abs/2103.00442). Further novelty review is necessary.

An exploratory 500-user feasibility probe used degree>=6 and item parity,
and found sparse positive coverage. It was not a complete AUC evaluation,
did not globally remove every internal target, and mishandled AUC ties. It
is not formal evidence. The following full definition fixes those flaws;
its thresholds are retained despite that discouraging observation.

## Fixed data and formulas

Read supplied Baby TRAIN only (x_label=0). Keep original item/user ID spaces.
For EVERY user with TRAIN degree>=4, select one internal target by minimum
SHA256 of dual-witness-target:v1:<user>:<item>. Remove ALL selected target
edges globally before computing any support statistic. No validation or test
labels enter targets, histories, negative exclusions or any selection rule.
Degree<4 users remain in the support matrix but have no diagnostic target.

Draw 16 independent deterministic comparisons per target, with replacement,
using a private PCG64 seed2026 and excluding ALL that user's original TRAIN
positives. Unobserved comparisons can contain false negatives; AUC is an
implicit-feedback probe, not preference truth. AUC ties count as 0.5.

Split columns once using SHA256(dual-witness:v1:<itemID>): first eight digest
bytes big-endian parity yields A/B. This shared partition is label-free.
For u!=v let c_A/c_B be cosine of binary support histories restricted to
each view. Candidate weight w_uv=sqrt(c_A*c_B), zero if either side has no
overlap. Empty-view rows have zero weights, no fallback and no skipped users.
Remove self, keep fixed Top100 by weight descending/userID ascending, score W R.
All formulas are computed in blocks using sparse matrices, no dense user graph.

Controls:

- Original ItemKNN: support item cosine, diagonal0, Top1000 per item row,
  score R S.T with the same orientation as BRIDGE.
- UserKNN: full-history cosine, self0, Top100.
- min2: same UserKNN but require raw overlap>=2 before Top100.
- shuffled: dual with B rows permuted within identical B-degree strata using
  private seed2026. Preserve per-user view degrees and catalog B counts;
  fixed points are allowed. Compute weights with this mismatched B view,
  then score the ORIGINAL support R, self0 and the same Top100.

Raw per-user AUC is invariant to a finite positive row z-score. Do not treat
this probe as training the multimodal encoder or proving its calibration.

## Frozen gates and budget

Require candidate positive nonzero coverage>=25%; mean AUC at least0.005
above EACH correct simple control (equivalent to beating the strongest);
paired user bootstrap2000 95% lower bound>0 against EACH simple control;
mean AUC at least0.005 above shuffled; and positive mean deltas against each
correct control in both fixed user-hash halves. Missing/nonfinite values fail.
No adjustment of partition, eligibility, K, weight, seed or gate after results.

Dataset progression Baby first; only a Baby mechanism pass permits the same
Sports diagnostic. No new numerical training or hyperparameter scan. Maximum
two diagnostics, four CPU threads, two-hour limit per dataset, eight-hour queue
including predecessor/resource waiting. Require12GiB free host RAM. GPU is not
reserved for this sparse CPU probe. Never terminate unrelated processes.

Passing both diagnostics only permits a separately committed formal protocol
with original BRIDGE/UserKNN/min2/candidate/shuffled, matched training budgets
and existing R20/N20 +0.5%, shortlist -0.0002 gates. It is not auto-acceptance.
The bounded diagnostic queue itself does not launch formal training.

For publication, use result/PUBLICATION_PROTOCOL.md and complete equal-budget
strong baselines, independent confirmation, three domains/five seeds,
multimodal attribution and measured efficiency. No guaranteed positive result
or publication outcome is implied by repeated searching.

Remote: /data/sunzixuan/dual_witness_20261009_v1.
Local branch: autoresearch/oct9-dual-witness.
