# Completed support-only expression audit

The frozen commit was `4c158f14b0436cdb8d97acd2cdf8c88c1847c5e4` on
`autoresearch/oct9-review-identifiability`. Root reviewed the implementation
before deployment. The actual MMRec environment passed 17/17 tests with zero
skips, failures, or errors. The immutable remote root was
`/data/sunzixuan/review_identifiability_20261009_v1`; PID 3104343 ran once and
had exited when the final status was collected. Runtime was 10.091080 seconds
under the declared one-hour bound, with four CPU/BLAS threads and no GPU.
Peak host RSS was 3,271,688 KiB. No text encoding, raw-review, interaction,
validation, test, or model-training operation occurred.

Both frozen views passed their independent expression-identifiability gates.
This outcome establishes author-expression association for this cache audit;
it establishes neither preference information nor recommendation gain,
novelty, publication readiness, or authorization for formal model training.
Stable writing style is one possible explanation. No best-view selection was
made, and the rejected F24/F25 item-ranking experiments remain rejected.

| Frozen view | Real user AUC | Null user AUC | Real minus null | 95% paired user bootstrap CI |
| --- | ---: | ---: | ---: | --- |
| Raw CLIP | 0.5485356751 | 0.4988611334 | +0.0496745417 | [0.0469730338, 0.0522377944] |
| F25 residual | 0.5618260346 | 0.5006561573 | +0.0611698773 | [0.0582706771, 0.0640631279] |

Coverage was 10,411/19,445 support users (0.5354075598) in both views. The
fixed user hash-half deltas were raw +0.0489837386/+0.0503579244 and residual
+0.0627739645/+0.0595830207. All 19,445 users stayed in the mean denominator;
unscorable B queries and 1,429 users without B queries received neutral 0.5
in both real and null. Neutral values do not constitute measured
discrimination.

The frozen cache contained 105,032 support edges: 52,727 A and 52,305 B.
There were 6,523 exact-item/exact-A-degree candidate groups and 21,660
scorable B queries. The remaining B queries were excluded for A count below
two (15,772) or no competitor of the same item and exact A count (14,873).
Eligibility was shared by both views. Both had 1,393 zero A profiles and
zero zero-vector B reviews. The fixed null allowed fixed points as declared:
6,522/21,660 assignments (0.3011080332). Its permutations preserve each
group's profiles and are not one global permutation or a causal intervention.

All eight input hashes and all deployed bundle hashes matched before and
after execution. The cache was derived from frozen F25 commit
`8ad9e5209e318db3c7efdf114dea7f16356e755f`, after global removal of all
13,447 internal target edges; it already excludes the earlier donor-mask
omissions. No claim is made for all original reviews beyond this boundary.

Machine-readable evidence is retained in `diagnosis.json`, `results.tsv`,
`latest_status.json`, `test_receipt.json`, `launch.json`, `deployment.json`,
and `input_audit.json`. The three transferred per-user/split archives were
checked against remote SHA256 hashes; `cache/artifact_manifest.json` records
their sizes, shapes, and hashes. No review text or external reviewer identity
is exported in these artifacts.
