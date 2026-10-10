# Independent pre-launch review of the MSCA interest-negative runner

Review date: 2026-10-10. Read-only review of the root agent's current
`train.py`, `sampling.py`, `manage.py`, prepared receipt and official MSCA source.
No training was launched and no runner/source file was edited by this review.

## Findings

### Resolved before this review was finalized

* The official MSCA evaluator uses `np.float`, removed in NumPy 1.24.4. The
  wrapper now installs the narrow `np.float = float` compatibility alias before
  importing the unchanged evaluator, and records/hashes the public source and
  GPL license. The alias leaves metric formulas and source bytes unchanged.
* Formal training now reads `preparation.json`, checks `status=completed`, domain,
  `cuda_initialized=false`, and exact `arrays.npz`/`diagnostics.json` digests.
  It adds those assets to the protected paths after loading, while deployment
  and source manifests remain protected. This binds model training to the CPU
  asset receipt rather than merely checking a self-consistent replacement cache.
* Training and evaluation now record separate positive-only and full-interaction
  epoch hashes. The latter includes the sampled negative column, so a control
  cannot be mistaken for the same training stream. The model initialization is
  reseeded after asset loading and before each fresh model construction.
* Official validation and final full-sort validation are under `torch.no_grad()`.
  The final path checks all full-sort scores for finiteness, verifies per-user
  lengths/order and compares unrounded per-user aggregate metrics with the
  official rounded checkpoint metrics within the declared tolerance. Test loader
  construction/evaluation flags remain false.
* The current runner wraps `model.full_sort_predict` before every official
  validation call and rejects nonfinite score tensors before top-k ranking. This
  closes the intermediate-score guard that was initially missing from the review.

No remaining launch-blocking issue was found in the runner after this hook was
added. The initially stale pipeline smoke-row assertion was corrected to require
rejection; the formal gate itself was not weakened. Independent local
`python -B -m unittest test_sampling test_pipeline -v` then passed29/29 tests
in1.327 seconds, without launching a child or initializing CUDA. The reviewed
runner passed an independent AST syntax check at SHA256
`94481e1acf1c7aafdba2650aa9b3798e595b6953022ff7d6de5a81de831d4de3`;
later implementation changes require a corresponding updated review/receipt.

## Checks that passed by inspection

* `train.py` constructs only split 0 (TRAIN) and split 1 (validation), never
  calls `RecDataset.split()`, and creates no test `EvalDataLoader`. Validation
  positives are masked through the official additional TRAIN dataset.
* `sampling.py` builds its universe as the unique original TRAIN item union,
  excludes all own TRAIN positives, uses rejection-free rank sampling and a
  private RNG. Semantic and shuffled arms reuse the same fixed pools; shuffle is
  exact-degree-stratum assignment, not pool rebuilding. The prepared asset hash
  and diagnostics are retained for all arms.
* `init_seed(2026)` is called immediately before each model construction. Asset
  grouping and negative draws use separate private NumPy states. After this seed
  reset, official `loader.pretrain_setup()` runs before model construction: it
  restores the TRAIN dataframe and shuffles the Python item list, preserving the
  original setup order. Each epoch's actual positive dataframe shuffle occurs
  later when the official loader is iterated. All arms perform these same steps;
  positive-only and full interaction hashes expose accidental divergence.
* The carrier source/model, interaction/features/graphs, evaluator and public
  license are protected by the deployment/source manifests. Official recipes are
  Baby `(2,.4,.005,3e-7)` and Sports `(3,.3,.005,5e-8)` with Adam `0.001`, zero
  weight decay, scheduler `[1,50]`, batch 2048, evaluation batch 64 and four CPU
  threads as frozen in `PROTOCOL.md`.
* Numerical checks cover loss, gradients before each optimizer step, parameters,
  Adam state, epoch loss, validation values, final scores and final per-user
  metrics. A crash, missing arm, nonfinite value or protected hash mismatch is
  technical failure, never an improvement.
* The result now records exact peak CUDA allocated and reserved bytes. Sampled
  `nvidia-smi` output is not used as an exact peak. Other unmeasured efficiency
  fields still need honest availability labels rather than invented receipts.

## Scope and source boundary

The [official MSCA repository](https://github.com/recomall/MSCA) identifies itself
as the WWW2026 implementation and specifies
[GPL-3.0](https://github.com/recomall/MSCA/blob/main/LICENSE). The actual host213
`/data/sunzixuan/MSCA-main/LICENSE` is35149 bytes with SHA256
`3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986`.
Its README is4221 bytes, SHA256
`85b9bbcc985827c7c2900222bee571c264ac06eecf91bb14a0629ecad5dafb16`.
The public README confirms the same two frozen domain recipes. This review
verified the actual source layout and model/source hashes, but did not establish
a complete remote git-commit association with every public upstream byte.
Use the remote audited hashes as the run provenance. This experiment uses
the audited source under a protected copy and records its license hash;
it does not present the carrier as newly authored code. The group-negative rule
is a PerPEFT-inspired published mechanism adaptation. No algorithm name,
originality or publication outcome is claimed. The separate paper-scope document
retains all frozen-family boundaries and the requirement for multi-seed,
multi-domain confirmation before any method claim.

## Pre-launch decision

The CPU receipt binding, source/evaluator protection, intermediate score hook,
split/no-test behavior, same-seed initialization/order and official configuration
are suitable for the registered pilot. Record the committed/deployed revision,
actual-environment CPU test receipt and required resource admission before the
two engineering smokes. Inspection and pure local tests do not establish
long-run stability, a completed GPU epoch or any recommendation benefit.
