# History Pairing Screen

Exploratory Baby internal development, not official test.

| Variant | Seeds | R20 | N20 | R200 |
| --- | --- | --- | --- | --- |
| additive | 3/3 | 0.09608297 | 0.04105710 | 0.31910517 |
| marginal_tensor | 3/3 | 0.09634010 | 0.04102311 | 0.31833376 |
| paired_moment | 3/3 | 0.09591155 | 0.04102110 | 0.31713380 |
| shuffled_cov | 3/3 | 0.09668295 | 0.04126372 | 0.31781949 |
| paired_cov | 3/3 | 0.09565441 | 0.04083700 | 0.31859090 |
| frozen_baseline | 3/3 | 0.09591155 | 0.04100086 | 0.31841947 |

15 / 15 jobs completed.

screen_not_passed; no reliably positive pairing benefit over all primary controls

All variants use the same support-only history, held training targets, and frozen full-catalog baseline.
The learned branch changes the actual full-catalog ranking. Calibration selects epochs and scale.
Internal development has already been used for earlier screens; this is not an untouched confirmation set.
