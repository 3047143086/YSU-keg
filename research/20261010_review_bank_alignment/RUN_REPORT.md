# Completed review-bank alignment diagnostic

The frozen commit was `367069cd1f132f9176f45460ba05d300f9506f06` on
`autoresearch/oct10-review-bank-alignment`. Root reviewed the implementation
before deployment. The actual MMRec environment passed 17/17 tests with no
skips, failures or errors. The immutable remote root was
`/data/sunzixuan/review_bank_alignment_20261010_v1`; PID 3112371 ran once and
exited normally. Runtime was 40.503669 seconds under the one-hour CPU bound,
with four threads, no GPU and peak RSS 3,962,508 KiB. No raw reviews,
interaction files, new encoding, validation/test labels or model training
were used.

The candidate changed only the F25 residual target from the original global
image+text bank to a leave-query-owner-out sum of the frozen residual review
vectors. The eight controls were evaluated under the same 105,032-edge cache,
the same 13,447 target users and the same 1+16 target/comparison rows. The
candidate coverage gate passed at 0.8261322228, but the joint mechanism gate
failed and the frozen decision is `discard_this_fixed_bank_geometry`.

| Arm | Mean user AUC | Candidate minus arm | 95% paired-user bootstrap CI |
| --- | ---: | ---: | --- |
| Candidate residual review bank | 0.5880029003 | — | — |
| Original item content | 0.5884816316 | -0.0004787313 | [-0.0088108058, 0.0071835261] |
| Original F25 target geometry | 0.5261164200 | +0.0618864803 | [0.0525996156, 0.0703919787] |
| Raw own review bank | 0.6703121514 | -0.0823092511 | [-0.0879889566, -0.0766804399] |
| Raw donor review bank | 0.6690711683 | -0.0810682680 | [-0.0872825956, -0.0752993233] |
| Residual donor review bank | 0.5914841600 | -0.0034812598 | [-0.0068230832, -0.0000880773] |
| Item popularity | 0.7576318138 | -0.1696289135 | [-0.1757389543, -0.1635889394] |
| ItemKNN | 0.6782181899 | -0.0902152897 | [-0.0966414442, -0.0840493007] |
| UserKNN | 0.5906405704 | -0.0026376701 | [-0.0085525582, 0.0032704204] |

The candidate's target-bank coverage was 0.9941250837 and all-comparison
bank coverage was 0.8261322228; every candidate profile was nonzero. The
candidate was below the item-content control by 0.0004787, so it failed the
required +0.005 gain before considering the other controls. It was also below
both raw review banks, residual donor bank, popularity and ItemKNN. The old
target geometry comparison reuses the exactly reproduced F25 residual
profile. Its positive delta supports a change in geometric alignment relative
to that weaker target, but the failed joint gate and donor comparison do not
support unique personal-expression transfer or a useful new mechanism.
F23 full-support references were
reported separately (ItemKNN 0.6782762884, UserKNN 0.5907033167); they use
105,104 edges and 72 more edges than the gate controls and therefore did not
enter the gate.

The all-user denominator remained 13,447. Query-owner subtraction was
implemented and recorded zero removals because the frozen target and 16
comparisons are globally absent from each query user's cached support. Input,
cache, predecessor and deployment hashes matched before and after execution.
Per-user scores, score matrices and full-support references are retained in
`cache/` with SHA256 entries in `cache/artifact_manifest.json`.

This is an internal TRAIN geometry diagnostic only. It does not establish
preferences, recommendation gain, causality, novelty or publication
evidence, and it authorizes no formal BRIDGE training. Review-bank methods
remain established prior art; F27 records a negative transfer result.
