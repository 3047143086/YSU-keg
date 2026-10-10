# Official MSCA baseline confirmation

This is a bounded reproducibility run requested after the October 10 feature
perturbation precheck ended in technical failure. It is **not** an innovation
arm and cannot be reported as a gain. It runs the unchanged official MSCA
carrier with the already used Baby uniform TRAIN negative sampler, seed 2026,
and the existing validation-only evaluator. The original source, split,
features, graph caches, loss, and evaluator remain protected.

The run uses GPU1 only, four CPU threads, the existing MMRec environment, and a
fresh remote output directory. It may run for at most eight hours. It must not
construct a test loader or evaluate test labels. The output records finite
updates, validation checkpoints, source/input hashes, and resource use. A
completed result is a baseline reproducibility record only; a crash is a
technical failure. No innovation gate or candidate rescue follows this run.

