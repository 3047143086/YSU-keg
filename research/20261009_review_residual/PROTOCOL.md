# Review residual direction diagnostic

Declared 2026-10-09 before this bundle is deployed or any residual vector is
encoded. This is a fixed diagnostic mechanism, not formal model training.

For every matched support review, let `z` be its frozen CLIP ViT-B/16 review
vector and `t` the corresponding frozen item text vector. Normalize `t`, then
compute `r = normalize(z - t * dot(z, t))`. User profiles are the mean of
their unit residual review vectors, followed by unit normalization. The same
held-out internal targets and 16 negative comparisons from the dual-witness
diagnostic are used for every arm. No target, validation, or test text is
read, and no encoder, token, alpha, mask, or threshold is searched.

The fixed arms are residual, raw own-review, same-item donor-review,
item-content, and residualized same-item donor-review. All use the same donor
keep mask. The candidate must beat each of the four controls by at least .005
mean per-user AUC, have a strictly positive paired-user bootstrap 95% lower
bound for every comparison, have positive deltas in both fixed hash halves for
every comparison, and have residual nonzero-profile coverage at least .80.
Any missing or nonfinite statistic discards the mechanism. A passing screen
does not establish novelty, causal semantics, or publication evidence.

The item-direction projection is a geometric residual diagnostic. Review-based
user/item representations and item-conditioned review models are established
prior art (for example DeepCoNN, WSDM 2017,
https://arxiv.org/abs/1701.04783, and ERRA, ACL 2023,
https://aclanthology.org/2023.acl-long.4/). This bundle makes no first-use
claim; novelty requires a later literature review and independent confirmation.

Budget: one Baby diagnostic, six hours, four CPU threads, at least 12 GiB RAM,
only physical GPU 1 with at least 14 GiB free, and a fixed per-process memory
fraction of 0.55. There is no formal validation or test evaluation and no
training authorization.
