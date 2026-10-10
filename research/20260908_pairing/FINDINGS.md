# HPAIR Screen Findings (2026-09-09)

The isolated run `/data/sunzixuan/hpair_20260908_screen_v1` completed all 15
formal jobs (five arms x three seeds) after all 13 remote tests and five real-data
smoke runs passed. It used the existing read-only Baby artifacts in
`/data/sunzixuan/rcmqr_20260908_screen_v2`; no data was downloaded.

| Arm | R@20 | N@20 | R@200 |
| --- | ---: | ---: | ---: |
| additive | 0.09608297 | 0.04105710 | 0.31910517 |
| marginal tensor | 0.09634010 | 0.04102311 | 0.31833376 |
| paired moment | 0.09591155 | 0.04102110 | 0.31713380 |
| shuffled covariance | 0.09668295 | 0.04126372 | 0.31781949 |
| paired covariance (main) | 0.09565441 | 0.04083700 | 0.31859090 |
| frozen baseline | 0.09591155 | 0.04100086 | 0.31841947 |

The primary `paired_cov` arm is below the frozen baseline by 0.27% R@20 and
0.40% N@20, and below `shuffled_cov` by 1.06% and 1.03%, respectively. It does
not meet the prespecified 2% gain, all-seed-positive, or bootstrap lower-bound
criteria. The queue correctly marked `screen_not_passed`. `paired_moment` equals
the frozen baseline on R@20, so any apparent benefit cannot be attributed to
covariance centering. The shuffled control is slightly higher than true pairing,
which rejects the proposed “within-user correspondence” explanation on this
screen rather than supporting it.

Seed detail: `paired_cov` selected scale 0.0 on seed 2026, and 0.1 on seeds 2027
and 2028. The positive movement on seed 2028 is not stable across seeds. Every
formal result records `official_validation_evaluated=false` and
`official_test_evaluated=false`; the internal development split was reused from
earlier screens.

## Research decision

Close HPAIR. Do not run more covariance dimension, scale, permutation, or tensor
capacity searches: that would be a search over the same rejected hypothesis.
The recent literature already covers generic synergy and tensor fusion, including
[SynGR (2026)](https://arxiv.org/html/2605.18920v1), [PRISM (2026)](https://arxiv.org/abs/2601.10944),
[MRdIB (2025)](https://arxiv.org/html/2509.20225v1), and [MITRec (2026)](https://github.com/HubuKG/MITRec).
The cross-domain motivation from [CovMatch (2025)](https://arxiv.org/html/2510.18583v1)
does not make this recommendation scorer novel.

A credible paper contribution now needs an independently testable signal that is
not present in this frozen-feature implicit-feedback setup: for example, exposure
or impression logs for a causal missing-not-at-random objective, a genuinely new
multimodal foundation feature bank, or a benchmark with modality timestamps and
missingness. Existing experiments show that rerouting, graph edits, temporal
kernels, confidence/alignment losses, late interaction, and this paired-statistic
branch do not provide reliable gains on the current split. A new method should
first be prototyped on an untouched confirmation split and compared with the
published synergy/tensor baselines before any paper claim.
