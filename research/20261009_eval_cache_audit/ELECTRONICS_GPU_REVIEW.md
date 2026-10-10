# Deferred Electronics GPU cache review

This is preparation only; no GPU job is authorized by this file. The v1
original Electronics baseline queue is already a technical failure after
seed 2026 epoch 8 (nonfinite gradients). Its selected checkpoint remains a
partial checkpoint and cannot support a baseline result or paper claim.

`electronics_gpu_audit.py` reads that checkpoint in a separate audit process.
It hashes the v1 deployment, run inputs/config/checkpoint/failure receipt and
all 35 carrier files before any model construction, then checks the graph
manifest and source lineage. It never resumes training, writes the baseline,
constructs a test loader, or computes test metrics. It selects validation
batches 0, 1, 1000, 1503, 2000 and 3006, compares direct and one-pass cached
full-sort scores with absolute error <=1e-6 and exact original masked Top-50,
and checks parameter/buffer state, transient forward attributes, CPU/CUDA/
NumPy/Python RNG, method code/globals and all protected hashes after scoring.
The frozen checkpoint configuration is reconstructed exactly; only GPU and
evaluation-output runtime keys are allowed to differ. The
`checkpoint_structure.py` compatibility shim is also used on GPU to assert
strict checkpoint key/value coverage and the original `.to(cuda)` registration
peculiarity.

The process must expose only physical GPU 1 as logical CUDA device 0. Start
only when the exited predecessor state is confirmed, GPU 1 has no active
compute process and at least 14 GiB free VRAM, and at least 24 GiB RAM is
available. Use four CPU threads, the original 0.55 Torch memory fraction
and a separate fixed one-hour audit budget including construction/hashing.
The process never signals any job and refuses resource contention. It
records peak allocated/reserved memory and requires one cache generation plus
five hits. Any mismatch stops with `non_equivalent`; no tolerance or solver
repair is allowed. A pass establishes only implementation equivalence for a
partial checkpoint and six batches. It does not validate the failed training
run, imply GPU training correctness, or authorize integrating the cache into
the Electronics baseline queue. Launch requires a separate review after
checking GPU 1 is idle and the v1 process is exited.
