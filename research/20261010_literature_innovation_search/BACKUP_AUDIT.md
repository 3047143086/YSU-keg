# Backup innovation audit (read-only)

Audit date: 2026-10-10. This file records a prior-art and feasibility review;
it does not authorize a run. No source checkout, split, evaluator, threshold,
queue, or GPU process was changed. The active F28 group-negative experiment is
complete on Baby and failed its frozen conjunction gate; Sports was not opened.

## Catalog relation proposal: blocked twice

The proposed `brand x categories` relation head is not an eligible intervention.
The CPU audit used the explicit Baby catalog and the split-zero TRAIN item
union, with an item-disjoint fitting/diagnostic partition. It found 4,923 of
7,047 TRAIN items with a nonempty brand (0.6985951469), below the preregistered
0.70 gate. Categories were present for all rows, but all rows had one identical
category signature. Consequently fitting pairs were 0 for same-brand/different-
category, 1,974 for same-category/different-brand, and 0 for neither. The
frozen result is `overall_training_eligible=false`; Sports was correctly not
audited after the Baby failure. There is no basis for a metadata training run.

Even with adequate coverage, a pairwise CE head on metadata-derived labels
would be a generic auxiliary classification/representation objective. [CAMMSR
(2026)](https://arxiv.org/abs/2603.04320) uses an explicit category prediction
task to guide modality weights in multimodal sequential recommendation.
[UniRec (2026)](https://arxiv.org/abs/2601.19423) represents typed attributes
across heterogeneous item fields. [HGCRec (2026)](https://doi.org/10.3390/electronics15163633)
uses category and brand entities in a relation-aware heterogeneous graph, and
[AD-DRL (2023)](https://arxiv.org/abs/2312.14433) uses category/brand attributes
in disentangled multimodal representations. [OpticalRec (2026)](https://arxiv.org/abs/2610.05432)
also makes title/brand/category metadata part of the item representation. The
proposed pair relation is architecturally simpler than these systems, so a
ranking gain could support only a controlled transfer observation.

The defensible contribution boundary would require all of the following before
any claim: (1) an independently audited metadata join on both domains, (2)
item-disjoint TRAIN relation prediction above a frozen readout reference, (3)
a label-shuffled null preserving endpoints, degree strata, and class counts,
and (4) a ranking gain that survives popularity and degree controls. The
current audit fails the required coverage and relation-count prerequisites,
and therefore this family is frozen. It must not be
rescued by lowering coverage, using title regexes, or inferring Sports IDs.

## Backup A: MASAM-motivated branch sharpness diagnostic

[MASAM (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/94ffbc2939e18adac36eedf8d38d1ef3-Abstract-Conference.html)
argues that heterogeneous modality encoders converge at different rates and
that sharpness imbalance can cause a fast modality to interfere with weaker
ones. Its Adaptive Perturbation Score combines convergence speed and gradient
alignment to identify a dominant modality; Modality-Decoupled Perturbation
Scaling then changes the perturbation allocated to each modality. The official
code supports classification tasks (MIMIC, CREMAD, KINETICS, FOOD101, URFUNNY,
and ADNI), not MSCA or Amazon ranking. This is therefore a mechanism transfer,
not a reproduction.

The hypothesis that is actually testable in MSCA is narrower:

> At a fixed MSCA parameter state and fixed TRAIN minibatch, one modality
> block has consistently larger *second-order* loss excess under the same
> relative parameter perturbation, and reducing that block's perturbation or
> training against its local worst case improves robustness without reducing
> the other block's signal.

MSCA has trainable `image_trs`/`gate_v` and `text_trs`/`gate_t` blocks, but no
independent unimodal prediction head. The current diagnostic should therefore
perturb exactly those two blocks, keeping shared ID embeddings, `query_common`,
feature tables, graph caches, sampler, and all auxiliary terms unchanged. This
scope is a limitation: dominance of the feature tables themselves would not be
measured. A branch one-sided difference such as
`L(theta + rho*g/||g||) - L(theta)` is not sufficient, because it is dominated
by first-order gradient magnitude. Use symmetric probes:

```text
E_m(B) = [L_B(theta + delta_m) + L_B(theta - delta_m)]/2 - L_B(theta)
```

with equal relative Frobenius radius (for example, the preregistered `rho=0.01`)
for image and text blocks, three private fixed TRAIN batches, one gradient
direction, and three private equal-norm random directions. Report the BPR
component, the full objective, and the regularizer component separately. The
`||theta||^2` regularizer has its own curvature and scale, so the full-loss
result cannot be silently substituted for BPR, nor can a favorable normalization
be selected after inspection. Report the signed excess and the per-block radius.

The proposed state points are both fresh initialization and the existing
validation-selected checkpoint. The latter is useful for a practical transfer
diagnostic but is not a pure TRAIN-only precondition; it must be labeled as such
and cannot be chosen from this probe. A preregistered dominance rule should
require the same block to win a specified majority of probes, with the BPR and
full-loss results agreeing after regularizer reporting. If the winner changes
across batches, directions, or states, the dominant-branch premise is falsified.
No validation or test score may choose `rho`, branch, direction, or checkpoint.
Also report `K_m = 2*E_m/||delta_m||^2`. Raw relative-radius excess and
Euclidean norm-squared curvature answer different questions because block
norms and dimensions differ. If they disagree, the audit cannot select the
preferred ruler. Three fixed batches and a few directions give a descriptive
screening result, not a significant curvature claim or maximal Hessian
eigenvalue estimate. This diagnostic is independent of F10's gradient-conflict audit: F10 measured
gradient cosine/work alignment and found no material conflict; it did not test
local curvature.

The prior-art boundary is strict. [BSAM (TKDE 2025)](https://zheyu-chen.github.io/_pages/BSAM.pdf)
already evaluates sharpness-aware multimodal BPR recommendation on Amazon Baby
and Sports, with visual/text features, random 8:1:1 splitting, batch size 2048,
and comparisons to SAM, ASAM, and related flat-minima methods. It also reports a
radius grid and 3090 runs, so "SAM improves MSCA" cannot be presented as a new
method. [gSAM (2023)](https://arxiv.org/abs/2307.08910) applies bi-level
sharpness optimization to graph collaborative filtering; [Mirror Gradient
(2024)](https://arxiv.org/abs/2402.11262) and [SGBD (ICCV Workshop 2025)](https://openaccess.thecvf.com/content/ICCV2025W/RetailVision/html/Srivastava_SGBD_Sharpness-Aware_Mirror_Gradient_with_BLIP-Based_Denoising_for_Robust_Multimodal_ICCVW_2025_paper.html)
also occupy the multimodal recommender flat-minima/robustness space. The only
honest label is **MASAM-motivated MSCA transfer pilot**.

If and only if the fixed diagnostic passes, a minimal pilot could compare a
fresh exact MSCA baseline, one fixed branch-restricted SAM-style update, and a
same-norm random-direction update. The latter is a necessary null for generic
extra noise. Keep the original BPR, contrastive terms, regularizer, negative
sampler, split, evaluator and frozen configuration. The existing gate remains
validation-only (both R@20/N@20 relative margins and R@10/N@10 absolute guard);
test is untouched. The noise null isolates targeted direction from generic
same-norm noise; it does not show an advantage over global SAM or over the
other branch. Those controls and a direct BSAM/MASAM comparison are needed for
a modality-specific mechanism claim. A failed dominance diagnostic means no
GPU pilot. Three Baby arms with eight-hour per-arm caps would have a 24-hour
formal ceiling; a conditional identical Sports trio adds another 24 hours.
Two-pass updates may cost roughly twice ordinary training per epoch, but actual
throughput and completion within these ceilings are unknown. This is a
planning estimate, not an allocation or guaranteed completion time.

## Backup B: adversarial perturbation in fixed feature space

[AMR (2018)](https://arxiv.org/abs/1809.07062) is a distinct input-space
intervention. It attacks a deep image feature vector inside a BPR objective,
solving a bounded inner maximization and minimizing the clean plus adversarial
loss. It deliberately leaves the expensive image extractor fixed and updates
the recommender transformation. This differs from MASAM's parameter-neighborhood
curvature: the perturbation is applied to an item feature row, not to model
parameters. MSCA already has image and text feature tensors for both domains
(the existing input inventory records Baby image/text shapes 7050x4096/7050x384 and
Sports 18357x4096/18357x384), so no metadata, reviews, LLM, or new encoder is
needed.

A feasible transfer would use one fixed L2 feature-row attack on the selected
image or text branch, plus a clean baseline and same-norm random feature noise
null. Keep the training sampler and every MSCA loss term fixed. Report the
training attack-loss gap and fixed validation ranking gate; require robustness
under the same attack as a mechanism check. The exact attack target must be
predeclared: MSCA's image/text KNN graph caches are built from detached initial
features, so perturbing only the transformed row tests direct feature
robustness, while perturbing a graph cache would be a different intervention.
The current `nn.Embedding.from_pretrained(..., freeze=False)` implementation
also means row replacement must be temporary and restored without changing
optimizer state. A source-level hook audit is required before any run.

This is not a new adversarial-recommendation algorithm. AMR itself establishes
the feature-space minimax mechanism; BSAM explicitly compares against AMR on
multimodal recommendation. [VARMAT (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/40054)
adds vulnerability-aware multimodal adversarial training, and SGBD adds a
sharpness-aware mirror-gradient/denoising combination. These papers block a
publication claim for generic feature attack or modality-vulnerability
regularization. The possible value is a bounded transfer diagnostic on this
MSCA carrier, with no claim beyond that.

## Decision and resource boundary

| Candidate | Asset status | Independent signal | Scientific status | Go/no-go |
| --- | --- | --- | --- | --- |
| Catalog relation CE | Baby audited; ineligible; Sports not audited | explicit metadata relation | blocked by coverage and direct category/brand prior art | freeze |
| MASAM-motivated block curvature | existing MSCA parameters/features | symmetric local curvature, not F10 cosine conflict | transfer pilot only; BSAM blocks novelty | diagnostic first; run only if preregistered pass |
| Feature-space AMR transfer | existing image/text feature arrays | adversarial input-row sensitivity | transfer pilot only; AMR/BSAM/VARMAT prior art | source hook audit first |

The two backups do not add an independently new observable fact; they test
different geometry/robustness properties of existing fixed features. If the
project requires a paper-level algorithmic novelty claim, this review found no
qualified candidate under F01--F28, the random-split constraint, and the
available Baby/Sports assets. Any subsequent run should be registered as a
preexisting-mechanism transfer, with primary-art controls and a negative null,
rather than renamed as a new algorithm. Publication-level evidence would need
the same frozen configuration and exact evaluator across three domains and
five matched seeds, uncertainty across seeds as well as users, measured GPU
time/memory, direct prior-art baselines, and public source/protocol manifests
before metrics. Private user/item batches stay private; publish only their
digests and aggregate diagnostics. Protected original-source files and license
must remain attributable in any public implementation.
