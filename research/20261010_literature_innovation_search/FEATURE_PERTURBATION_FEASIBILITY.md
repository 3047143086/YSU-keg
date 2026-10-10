# AMR/VARMAT feature-perturbation feasibility audit

Audit date: 2026-10-10. This is a source and asset review after the curvature
diagnostic returned `diagnostic_inconclusive`. It does not reopen that
diagnostic, start a training process, or authorize a recommendation experiment.

## Verdict

The feature-space intervention is **conditionally eligible for a bounded
TRAIN-loss engineering precheck**, but it is **not eligible as a new-method
formal experiment**. Existing primary work already establishes the mechanism:
[AMR](https://arxiv.org/abs/1809.07062) performs a bounded adversarial attack on
deep image features inside a BPR objective, and [BSAM (TKDE
2025)](https://zheyu-chen.github.io/_pages/BSAM.pdf) compares adversarial
training against AMR while evaluating Amazon Baby and Sports. [VARMAT (AAAI
2026)](https://ojs.aaai.org/index.php/AAAI/article/view/40054) further studies
modality vulnerability-aware adversarial training. A successful result here
would be a controlled transfer to the official MSCA carrier, with no claim of
algorithmic novelty.

The current workspace has not yet proved the source-level hook, zero-row
coverage, or exact feature-path manifest for a perturbation run. Therefore the
candidate status is `precheck_required`, not `eligible_to_train`. A failed
precheck freezes this family; it must not be repaired by changing epsilon,
lambda, attacked rows, graph construction, or feature version.

## What the official carrier actually consumes

The protected official MSCA source is `/data/sunzixuan/MSCA-main/src`, with
`models/msca.py` SHA256
`066aafcf2e5373b20b2fdf9b3660302b8dfcaf580710d3ab864fac4f19e2360b`. Its
dataset configs name `image_feat.npy` and `text_feat.npy` for both Baby and
Sports. The existing input inventory records:

| Domain | image feature | text feature | dtype evidence |
| --- | ---: | ---: | --- |
| Baby | `(7050, 4096)` | `(7050, 384)` | image float64, text float32 |
| Sports | `(18357, 4096)` | `(18357, 384)` | image float64, text float32 |

These are existing feature arrays, not new metadata or raw-image encodings.
The actual run must hash the arrays at the deployed carrier path and verify row
count and item-ID order against the official interaction files. The local
workspace does not contain the remote carrier data directory, so these counts
are inventory evidence rather than a completed local eligibility check.

At construction, MSCA wraps each feature matrix in a trainable
`nn.Embedding.from_pretrained(..., freeze=False)`. It then computes

```text
image_item = item_id * gate_v(image_trs(image_embedding.weight))
text_item  = item_id * gate_t(text_trs(text_embedding.weight))
```

and propagates each through a precomputed sparse item-item adjacency before
multi-view fusion. The constructor builds `image_adj_10_True.pt` and
`text_adj_10_True.pt` from detached initial features when a cache is absent; the
official carrier already has private copies. The adjacency is therefore fixed
after construction while `image_embedding.weight` and `text_embedding.weight`
remain ordinary model parameters. This distinction is central to a fair
intervention.

## Required precheck

The precheck is one deterministic, no-optimizer TRAIN-loss diagnostic on a fresh
official MSCA state. Use the existing private three Baby TRAIN batches from the
curvature receipt (or create a separately hashed equivalent before inspecting
losses), with the same users, positives, and ordinary TRAIN-union negatives for
all arms. It must not load validation/test data or select a checkpoint. Run
these three views on each batch:

1. **clean:** the exact official MSCA BPR loss;
2. **targeted feature perturbation:** one-step image-feature adversary;
3. **random null:** same item rows and per-row L2 norms as the targeted attack,
   with a private Gaussian direction instead of the loss gradient.

AMR attacks the image feature used by the BPR score, so the first pilot should
attack image rows only. Text attack is a separate hypothesis and must not be
silently combined with it. Select the unique image rows appearing as positive
or negative items in each batch. For each selected row `i`, compute the BPR
gradient with respect to that row, require a finite nonzero gradient and feature
norm, and set one fixed relative radius before metrics are read:

```text
delta_i = epsilon_rel * ||x_i||_2 * grad_i / ||grad_i||_2
epsilon_rel = 0.01
```

The random null uses the same `epsilon_rel * ||x_i||_2` row norm and the same
rows, with a fixed private Gaussian direction. It must not be rescaled after
seeing the loss. A zero or nonfinite feature/gradient row is a precheck failure
for that batch; do not silently drop it or use a different modality. Report the
zero-row and invalid-gradient counts before deciding eligibility.

The diagnostic reports clean BPR, targeted perturbed BPR, random perturbed BPR,
targeted-minus-clean loss, random-minus-clean loss, and the targeted/random
ratio. A meaningful attack precondition requires targeted loss increase to be
finite and greater than the random increase on at least two of three batches;
otherwise the targeted direction is not demonstrated and no training arm is
eligible. This is a mechanism check, not a performance result. Do not use
validation Recall@20 to choose epsilon, attacked modality, rows, or state.

## Hook and gradient rules

The current MSCA source has no feature-override argument. A safe implementation
must use an isolated wrapper or a functional forward path; it must never execute
`param.data += delta`, mutate an embedding parameter in-place, or overwrite a
graph-cache file. The wrapper must prove that:

* clean forward bytes and outputs match the official `forward(test=False)`;
* the targeted and random paths differ only in the temporary image feature rows;
* `image_original_adj`, `text_original_adj`, `norm_R`, and `norm_adj` are the
  same tensors in all three paths;
* after each view, every parameter/buffer state hash is bitwise unchanged and
  the feature arrays and cache files have the same SHA256;
* no optimizer is instantiated or stepped in the precheck.

There are two possible gradient semantics, and the choice must be frozen before
the first loss:

* **AMR-compatible:** compute the attack from a detached copy of selected
  feature rows and use the detached `x_i + delta_i` in the second forward. The
  adversarial diagnostic then measures robustness of the trainable MSCA
  transform/gate while the feature-table parameters receive no attack gradient.
* **End-to-end MSCA:** allow the attack loss to backpropagate through the
  trainable feature table. This is a new MSCA-specific variant, not AMR, and
  must report that feature-table updates are part of the intervention.

For a first transfer precheck, use the AMR-compatible semantics. A later formal
training arm would need matched clean/targeted/random implementations and a
receipt proving exactly which parameters receive the adversarial gradient. A
single feature clone plus `index_copy` is acceptable for Baby/Sports memory
budgets; in-place restoration is not. The feature row replacement must be
temporary for both plus and random views.

Keeping the original graph cache fixed is the only fair narrow interpretation of
"feature-row attack" on this carrier. Rebuilding KNN adjacency from perturbed
features would simultaneously change graph topology and normalized propagation,
reopening the failed graph/evidence families F02--F05 and F23. A report must
state that this pilot tests direct feature robustness under a fixed cached graph;
it cannot claim robustness to graph recomputation.

## Minimum controlled training, only after precheck

If the precheck passes and a fresh protocol is separately approved, use three
same-seed arms with the exact official MSCA configuration and ordinary negative
sampler:

* clean official MSCA;
* targeted image-feature adversarial BPR;
* same-row, same-norm random feature perturbation null.

Use a clean plus adversarial BPR objective with a fixed coefficient chosen in the
protocol before metrics (a 1:1 average is the least tunable choice). Do not
search epsilon, coefficient, attack steps, modality, or graph cache. Keep the
contrastive terms, regularizer, optimizer, scheduler, split, checkpoint rule,
validation-only evaluator, and test prohibition identical. Measure targeted
attack-gap reduction on TRAIN as the mechanism guard and retain the existing
R@20/N@20 and R@10/N@10 validation gate. The random null is necessary because
the extra loss/compute alone can alter optimization; it does not establish
superiority over BSAM or global SAM.

The formal run is blocked if any row-level feature/gradient precheck fails, if
clean output parity fails, if graph tensors differ, if state restoration is not
bitwise exact, or if targeted loss does not exceed the random null on the fixed
screen. No epsilon or coefficient rescue is permitted. GPU cost is expected to
be about two forward/backward passes per batch, but actual throughput is
unknown; record measured time and memory instead of promising a budget.

## Boundaries and interpretation

This intervention is independent of F01--F28's routing, graph edits, gates,
candidate reranking, review features, rating polarity, and negative sampling.
It does, however, alter the optimization landscape through a preexisting
adversarial mechanism. AMR's image-feature minimax objective, BSAM's Baby/Sports
multimodal recommendation experiments, and VARMAT's vulnerability-aware
multimodal attack prevent a new-method claim. A positive result can only say
that a fixed-feature adversarial training transfer was or was not useful for
this MSCA carrier under the declared random split.

The current status is therefore: **precheck required; no training started**.
