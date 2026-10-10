# Read-only next-signal assessment

Declared 2026-10-10 after frozen F27. No new rankings, coverage metric, review
extraction, encoding, validation/test access or training was performed.

The summary/body projection suggestion is **rejected before run**. Subtracting
a CLIP summary direction from a CLIP body direction does not identify usage or
need facts separately from length, vocabulary or author style. It is another
fixed-vector residual geometry after F25/F27 and is not implemented.

The read-only asset inventory on server213 found Baby `items.jsonl` with 7,050
catalog rows and keys item_id, asin, title, description, brand, categories and
imUrl. Text fields can be null. There are no structured minimum/maximum age,
device compatibility or usage-context fields. This schema audit read public
catalog metadata only, not reviews or interaction labels. The raw metadata
archive also exists; its content was not inspected and no additional facts
are presumed present.

The bounded model inventory found the existing CLIP ViT-B-16 checkpoint at
`/data/sunzixuan/rave_20260906_stage1/weights/ViT-B-16.pt`. The checked Hugging
Face cache locations were absent; transformers, sentence-transformers,
spaCy, stanza and NLTK were not installed in MMRec. This is a bounded inventory,
not proof that no other machine has a semantic model. No credentials or model
configuration contents were read.

A different, potentially identifiable signal would require explicit support
review evidence of a user-owned model/physical requirement and independently
documented candidate compatibility, with extracted spans and ambiguity rules
frozen before scores. Age is especially weak under the existing random split:
child age changes, multiple children are possible and review date is not a
known request date. Free-form text plus an unvalidated rule is insufficient to
claim that constraint is observed reliably. Other users' reviews would not be
an independent product specification and would return to review-bank matching.

Therefore there is **no defensible ready-to-run new usage-constraint hypothesis
on the inspected frozen inputs**. A future proposal first needs an independent
fact/extraction validation set without target labels, a frozen extractor and
coverage/abstention gate. Its mechanism controls would include same-item donor
requirements, shuffled requirement entities, popularity and strong item/user
KNN under fixed pairs, retaining all users and neutral abstentions. No such
new experiment or resource allocation is preregistered here.

Originality limits are substantial. [ERRA](https://arxiv.org/abs/2306.12657)
already combines personalized review retrieval with aspect learning;
[Explicit Factor Models](https://sigir.org/sigir2014/finalfullpapers.php) uses
review-derived features and opinions; and
[Human computation for constraint-based recommenders](https://doi.org/10.1007/s10844-016-0433-4)
uses explicit user requirements and item-property constraints. These primary
sources support no first-use claim for review aspects or compatibility rules.
The current priority is independently reviewed Electronics numerical recovery,
which is baseline infrastructure and is named as such.
