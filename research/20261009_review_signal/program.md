# TRAIN review identity-conditioned signal audit

Local Codex adapter for user-selected karpathy/autoresearch. Upstream
https://github.com/karpathy/autoresearch.git provenance is preserved in
../20261008_autoresearch/skill-source.json. Read PROTOCOL.md before running.

The three previous gate/evidence families are frozen. This single new
hypothesis adds original observed TRAIN review text. It first tests whether
the correct consumer's expression is useful beyond same-item other-consumer
text and original item content. It is not a novel model or a paper claim.

Editable scope: mapping.py identity/support isolation, screen_core.py fixed
controls, experiment.py frozen-CLIP extraction and TRAIN-internal diagnosis.
Original inter/splits, ID mapping, foundation weights and item features are
read-only. No rating value is a model input; it is only an identity fingerprint.

Commit the isolated branch before launch. Use the original MMRec environment,
four CPU threads and at most one otherwise idle GPU. Budget six hours for a
single Baby diagnostic including GPU waiting; no full recommendation training.
Record fixed AUC and coverage with paired user intervals. Keep/discard by the
frozen gates. A pass requires separately preregistered training and independent
domain confirmation. Never sweep strengths/encoders/truncation to rescue it.
