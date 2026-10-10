# Retrieval Screen Status

Training-internal Baby screen. Official validation/test are not evaluated.

| Seed | Job | Status |
| --- | --- | --- |
| 2026 | base | completed |
| 2026 | independent | completed |
| 2026 | hard | completed |
| 2026 | clear | completed |
| 2026 | conditional | completed |
| 2026 | shuffled | completed |
| 2026 | base_user | completed |
| 2026 | feedback | completed |
| 2027 | base | completed |
| 2027 | independent | completed |
| 2027 | hard | completed |
| 2027 | clear | completed |
| 2027 | conditional | completed |
| 2027 | shuffled | completed |
| 2027 | base_user | completed |
| 2027 | feedback | completed |
| 2028 | base | completed |
| 2028 | independent | completed |
| 2028 | hard | completed |
| 2028 | clear | completed |
| 2028 | conditional | completed |
| 2028 | shuffled | completed |
| 2028 | base_user | completed |
| 2028 | feedback | completed |

## Internal Development Metrics

| Variant | Seeds | R20 | N20 | Candidate R200 | Net Candidate Gain |
| --- | --- | --- | --- | --- | --- |
| independent | 3/3 | 0.09591155 | 0.04100086 | 0.30873404 | -0.00771407 |
| hard | 3/3 | 0.09591155 | 0.04100086 | 0.30564841 | -0.01079969 |
| clear | 3/3 | 0.09591155 | 0.04100086 | 0.30401989 | -0.01242822 |
| conditional | 3/3 | 0.09591155 | 0.04100086 | 0.30770549 | -0.00874261 |
| shuffled | 3/3 | 0.09591155 | 0.04100086 | 0.30616268 | -0.01028542 |
| base_user | 3/3 | 0.09591155 | 0.04100086 | 0.30641982 | -0.01002828 |
| feedback | 3/3 | 0.09591155 | 0.04100086 | 0.30753407 | -0.00891403 |
| raw_base_baseline | 3/3 | 0.09359733 | 0.03927255 | 0.31644810 | n/a |
| pipeline_baseline | 3/3 | 0.09591155 | 0.04100086 | 0.31644810 | n/a |
| full_catalog_fusion_baseline | 3/3 | 0.09591155 | 0.04100086 | 0.31841947 | n/a |

screen_not_passed; no reliable new mechanism established

Primary comparisons are conditional vs hard/independent/shuffled controls with identical labels.
Raw base has fewer training labels; gains over it alone are not an equal-label comparison.

Full metrics and selected configurations: summary.json and each result.json.
Do not compare these internal metrics directly to old official-validation results.
