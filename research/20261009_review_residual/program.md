# Program contract

`experiment.py` is invoked without arguments by the manager. It first checks
the completed `input_audit.json`, frozen predecessor hashes, the fixed target
digest, and unchanged baseline carrier. It removes internal target edges,
reuses the predecessor's identity mapping and same-item donor selection, then
encodes only matched support review text with the frozen CLIP implementation.

The fresh cache records `review_embeddings.npy` and `review_index.npz`; the
previous review experiment did not persist an encoded review bank, so this run
must record that encoding occurred. The five arms are projected/profiled with
`residual_core.py`, scored against the existing global image+text bank, and
written atomically to `diagnosis.json` and `results.tsv`. `progress.json` and
`pipeline.log` are status artifacts. Protected source and audit hashes are
checked before and after the diagnostic.

The manager accepts `audit`, `deploy`, `test`, `verify`, `start`, `launch`, and
`status`. Tests run with `CUDA_VISIBLE_DEVICES=''`; only the experiment chooses
its GPU. Deployment is immutable by bundle hash, refuses drift or a second
launch, checks four CPUs and 12 GiB available RAM, and never kills unrelated
processes.
