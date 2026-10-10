# Frozen single-seed interest-negative pilot

Version/date: 2026-10-10, before smoke or formal metrics. Experiment root:
`research/20261010_interest_negative`; remote root:
`/data/sunzixuan/interest_negative_20261010_v1`. This protocol freezes one
published-mechanism adaptation. It does not replace the cumulative failure or
three-domain/five-seed publication protocols.

## Hypothesis and sole intervention

Original MSCA's uniform negative comparisons may often be unrelated to a user's
TRAIN interests. Mixing a fixed interest-group item pool into its original BPR
negative distribution may teach useful within-interest distinctions. The
falsifiable comparison is semantic assignment versus original uniform and a
same-pool, exact-TRAIN-degree-stratum shuffled assignment. This is a
[PerPEFT-inspired training technique](https://arxiv.org/html/2602.09445v1), not
an official PerPEFT reproduction or a tested originality claim.

The only intervention is replacing the existing BPR negative-item draw. Keep
the official model architecture, learned parameters, initialization law, forward,
loss, graphs, features, optimizer and full-catalog evaluator. No group embedding,
group head, adapter, gate, auxiliary loss, graph edge, current-model TopK,
semantic initialization or inference correction is introduced. Original MSCA's
negative items enter BPR; its CL terms use users and positive items. This differs
from F13 initialization, F18 inference residuals and F22 gate-gradient ownership,
without establishing that the regrouped inputs contain new observed information.

## Protected carrier and fixed configurations

Carrier: actual host213 `/data/sunzixuan/MSCA-main/src`, model
`models/msca.py` SHA256
`066aafcf2e5373b20b2fdf9b3660302b8dfcaf580710d3ab864fac4f19e2360b`.
It is the official MSCA carrier without BRIDGE's SVD path. The previously modified
local PRAVI file or historical BRIDGE metrics cannot substitute for it or for the
fresh matched uniform baseline.

Pre-existing audited configuration sources:

| Source | SHA256 |
| --- | --- |
| `configs/model/MSCA.yaml` | `a95df9ba3adb0f0657dc16e7c9231938c35d6c830b4e24da081040f97435f927` |
| `configs/overall.yaml` | `5ebb6772d30bb666add0bc82f3d6306414f1d8663014d7e14bec6c398db754af` |
| `configs/dataset/baby.yaml` | `68a6142a857384b8c2cbaf09e6e418b910c6c4a40462e813c436cd1ed8a4a0c8` |
| `configs/dataset/sports.yaml` | `8ac3dbd974ef2d5ceabb6da3511a4135d786032dff75debb498bf02bbb02a5c0` |
| `utils/dataloader.py` | `08d9d182e04aa105e4b48fe8aafbaf91ab7435eb5add06c148c001900e6cfe7c` |
| `utils/dataset.py` | `4645028fa7b2d5f9c3b7ea0399eeed8522371732bc523c28d0593a1c521315af` |
| `common/trainer.py` | `9520a7a712bb29ec0cb7c6319e362a0f3a89d33cd9ebfa7acd0afe9b48051e39` |

Baby uses the active official singleton recipe. Sports uses the Sports recipe
already present as comments in the same official model config, fixed now before
metrics. These are prior domain recipes, not a search over configuration results.
Within each domain all three arms use exactly the same recipe:

| Parameter | Baby | Sports |
| --- | ---: | ---: |
| embedding_size |64 |64 |
| n_layers |2 |3 |
| knn_k |10 |10 |
| is_sparse |true |true |
| fusion_coeff |0.4 |0.3 |
| tau |0.2 |0.2 |
| cl_weight |0.005 |0.005 |
| reg_weight |3e-7 |5e-8 |

All arms: seed2026, Adam, learning_rate0.001, weight_decay0, original scheduler
`[1.0,50]`, one negative per positive, maximum300 epochs, official
`stopping_step=20`, train batch2048, eval batch64, eval every epoch, four CPU
threads. The historical sweep lists are not traversed. Freeze the resolved
configuration file in each run and verify equality within its domain.

Use original `baby.inter`/`sports.inter` and their existing `x_label` split:
0 TRAIN,1 validation,2 test. No new split, row filtering, resampling, raw review,
category, timestamp or replacement feature version is allowed. Preserve feature
row/item alignment, ID order, TRAIN-positive masks, graph normalization/cache
sources and validation user/positive population. Record actual source, evaluator,
interaction, feature and graph-cache hashes in a pre-run manifest and check them
after every arm. Test loader must never be constructed; official entry points that
construct/evaluate test during fitting must be bypassed by the validation-only
wrapper, without rewriting evaluator semantics. The inspected official evaluator
uses the removed `numpy.float` alias in NumPy 1.24.4. The wrapper installs the
compatibility alias `np.float = float` before importing/evaluating the unchanged
official metric source and protects its source hash; it must not patch the
evaluator formula. Both official evaluation and the final full-sort check run
under `torch.no_grad()`. Nonfinite full-sort scores are a technical failure and
cannot become a ranked finite metric. These declared engineering safeguards are
identical in all arms and do not change the model or objective.

## Frozen support sets, grouping and negative distributions

Let `H_u` be all unique original TRAIN-positive item IDs for user `u`.
Let `U = union_u H_u` be the sorted unique original TRAIN-observed item universe.
This matches the official dataloader's `all_items` sampling support. Full catalog
items outside `U` are not negative candidates even though the full catalog remains
the validation ranking scope. Never exclude negatives using held-out labels.

For each item, separately L2-normalize its original image and text feature rows;
zero rows stay zero. For every user, take the unweighted TRAIN mean within each
view, concatenate the two means, then joint L2-normalize. Zero-history/profile
users remain zero and remain in the original population. All profile accumulation
uses float64 and sorted unique TRAIN IDs. No profile enters model initialization,
training features, scores, gradients or evaluation.

Freeze KMeans K8, private grouping seed2026, kmeans++ initialization, n_init10,
max_iter300, tol1e-4. The actual available implementation/backend and version must
be resolved once during preparation, recorded before metrics, and fixed by its
source/profile/label/center/pool hashes. The explicit NumPy fallback is not claimed
equivalent to sklearn. Do not run multiple backends and choose one by metrics, or
change a backend/normalization after a failure. Build one asset version per domain
and reuse it for all arms and, for Baby, both smokes.

With semantic labels `c(u)`, freeze `P_c = union_{u:c(u)=c} H_u`. Pools contain
unique sorted items, each equally likely; they are not frequency-weighted.
The control uses a private independent RandomState(seed2026) to permute user IDs
within **exact unique TRAIN-degree strata**, then uses their semantic labels as
the shuffled assignments. It never rebuilds the original semantic pools.
Preserve each `degree x group` count, group headcount and positive-pair mass;
report both source-user permutation fixed points and unchanged-label counts.
Fixed points and same-label permutations are permitted and reported, not retried
to increase the apparent control contrast. Grouping, label permutation, sampler
and model generators are separate objects even when seeded2026.

| Arm | Existing negative draw |
| --- | --- |
| `uniform` | Uniform over `U \ H_u`. Fresh reference carrier, not a historical checkpoint. |
| `semantic` | With probability0.5, uniform over `P_c(u) \ H_u`; otherwise uniform over `U \ H_u`. |
| `shuffled` | Identical0.5 mixture using the shuffled user label and the **same fixed** pools. |

Each draw excludes every own TRAIN positive. An empty eligible group pool falls
back to the same uniform eligible draw; if `U \ H_u` is empty, fail explicitly
rather than drop the user, loop indefinitely or label a known positive negative.
Use rejection-free eligible-rank sampling and record fallback counts. All sampler
arms have private same-seed streams and consume mixture-coin, uniform-rank and
group-rank variates in the same order for each positive, including unused branches.
They never advance model/global Torch or NumPy RNG. Preserve batch/positive order,
model initialization hash and non-sampler generator states across fresh arms.
The baseline retains the original uniform probability law, not a claim of an
identical historical global-RNG negative sequence.

## Queue, engineering smokes and resource budget

Before metrics, initialize an isolated git repository/branch
`autoresearch/oct10-interest-negative`, commit the complete tested payload and
record commit, working-tree cleanliness and deployed hashes. A mutable or
hash-mismatched deployment cannot launch formal training. Preserve all old frozen
experiments and the original source tree. Only this experiment's sampler and
validation-only orchestration/reporting wrapper are editable; a compatibility
repair is a separately explained pre-metric engineering change.

Exactly two Baby engineering smokes precede formal arms: `uniform`, then
`semantic`, each a full one-epoch GPU run at the fixed recipe with fresh seed2026
and fresh optimizer. Check finite full-epoch losses, gradients, parameters,
updates, memory, protected hashes, sampler invariants and validation-only behavior.
Discard their weights and optimizer/RNG state. A smoke may emit engineering or
validation diagnostics but cannot contribute to gain claims, seed selection,
stopping thresholds or formal completed-arm counts. No parameter is changed on
the basis of smoke metrics. A genuine smoke failure blocks formal launch until a
reviewable engineering-only fix/version is committed; never relabel a failed
smoke as a scientific pass.

Formal Baby order is fresh `uniform`, fresh `semantic`, fresh `shuffled`, seed2026.
Only a completed Baby gate pass authorizes fresh Sports arms in the same order,
seed and stopping rule. Every arm starts from identical within-domain
initialization and original positive order, without smoke/baseline warm starts.
Training stops by official early-stopping behavior, epoch300 or an8-hour child
wall-clock limit, whichever occurs first. An8-hour timeout is a technical failure,
not a completed poor/good result.

Use GPU1 only, one active child, four CPU threads, CUDA memory fraction0.55.
Admission requires at least14 GiB free GPU1 memory and12 GiB available host RAM,
with no overlapping owned training. Do not kill someone else's process, switch
GPU or weaken admission. Total queue wall ceiling48 hours includes preparation,
waiting, smokes and both domains. Record admission/waiting time separately from
training monotonic elapsed time. No unbounded automatic retries or continuation.

## Validation selection and mandatory decision conjunction

Select each arm's best checkpoint only by validation Recall@20 rounded to four
decimals. Save only a strict improvement; a rounded tie keeps the earlier
checkpoint. Use `stopping_step=20` with the original strict-greater stopping
comparison from the official trainer. Report unrounded R@10/N@10/R@20/N@20 from
the selected checkpoint, not different best epochs for different metrics.
No test loader, labels, epoch metrics or final test evaluation are permitted.

For each domain, let `s` be semantic and `r` be each reference in
`{uniform, shuffled}`. Require **all** eight comparisons:

```
(R20_s - R20_r) / R20_r >= 0.005
(N20_s - N20_r) / N20_r >= 0.005
 R10_s - R10_r >= -0.0002
 N10_s - N10_r >= -0.0002
```

All three formal arms must have terminal `completed` status, finite metrics,
positive R20/N20 denominators, the same validation users/positive counts and
passing provenance/initialization/order/numerical/integrity guards. Missing,
pending, timed-out, crashed, malformed, nonfinite or mismatched arms prevent a
pass. A live best-so-far value is intermediate, never a completed outcome.

Baby failure closes this one-point queue and freezes the hypothesis at K8,
mix0.5 and seed2026. Do not run Sports or rescue it by changing groups, mixture,
pool weighting, exclusions, thresholds, epochs, seed, backend, checkpoint or
dataset. A full Baby pass permits only the registered Sports three-arm queue;
Sports applies the identical conjunction. A full two-domain pass is
`pilot_passed_needs_confirmation`, not statistical significance or publication
confirmation. Technical failures are `crash`/technical failure, not scientific
negative scores; retain completed arms but do not fabricate a gate from them.

## Required evidence and interpretation

Record command/configuration/source/environment, model initialization hashes,
positive-order fingerprints, private RNG seeds, support sets and pool/label/center
hashes, fixed-point/unchanged-label coverage, user/degree/pair group counts, pool
unique sizes/popularity distributions, own-positive eligible mass, fallback rate,
realized negative popularity/coverage and duplicate rates. Save progress logs,
effective optimizer steps, finite-update guards, best checkpoint and per-user
validation arrays with IDs/positive counts. Retain all failures and terminal PID
identity. Snapshot timestamps do not replace live-process checks.

Record training/grouping/waiting time, epoch durations, throughput, parameter
count, graph/cache sizes, accurate peak Torch allocated/reserved memory, host RSS
and measured full-sort inference cost where feasible without extra test access.
An nvidia-smi sample is sampled memory, not an exact peak. Record unavailable
measurements honestly rather than imply they were measured. `results.tsv` lists
formal `uniform`/`semantic`/`shuffled` arms with `keep`/`discard`/`crash` outcomes
and a control marker; smoke evidence is separate, and completed formal-arm counts
include only new complete runs. Baseline/control completion does not itself imply
semantic success.

Exact-degree shuffle and fixed same pools preserve pre-filter group counts and
unfiltered expected pool exposure, but own-positive filtering changes realized
eligible item marginals. A gain may reflect popularity or difficulty, not only
meaningful interests. Similar users' purchases may be false negatives. Fixed
label assignments cannot represent every user's multiple interests. These are
material limitations to report, not reasons to consult held-out labels or adjust
the registered point after metrics.

No paired-bootstrap or training-seed significance gate is added to this pilot.
Any user-level interval is explicitly exploratory and cannot substitute for
independent seed variation. `result/PUBLICATION_PROTOCOL.md` remains unchanged:
three domains, seeds2026-2030, strong equal-budget baselines, attribution,
training-seed uncertainty and independently frozen final test handling. This
MSCA pilot cannot automatically clear that protocol's BRIDGE confirmation gates.
