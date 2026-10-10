# Confidence-Calibrated Simplex Routing (CCSR)

## Hypothesis

The original encoder stores per-user modality weights after a one-time
softmax, but optimizes those weights as unconstrained values thereafter.  A
user can therefore acquire negative or arbitrarily large modality scales.  CCSR
stores router logits and applies a simplex projection at every forward pass.
Its temperature is calibrated by two training-only signals:

* standardized `log(1 + train interaction count)` (sparse users are smoother);
* detached disagreement between the current ID, visual, and text user views
  (conflicting views are smoother).

The route is then `pi_u = softmax(a_u / tau_u)`, which keeps each user's
modality mass positive and normalized while allowing active, mutually
consistent users to specialize.

## Controlled run

The official BRIDGE configuration remains unchanged when
`use_activity_simplex=false`.  The treatment changes only the user-side
fusion route; all behavior correction, candidate scope, data split, seed, and
validation selection settings are held fixed.  The test split is read once
after selecting the best validation Recall@20 checkpoint.

The treatment is accepted only if it beats the frozen BRIDGE reference on
validation Recall@20 and NDCG@20 without a material regression, and is then
repeated across the predeclared datasets and seeds.
