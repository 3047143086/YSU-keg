# Karpathy autoresearch: candidate-negative gate supervision

Source: https://github.com/karpathy/autoresearch.git, revision
228791fb499afffb54b46200aca536f79142f117. Apply the installed auto-research
adapter and read PROTOCOL.md. Branch: autoresearch/oct9-candidate-gate.

The preceding decoupled counterfactual gate candidate completed Baby and
failed its scientific screen. Preserve its immutable bundle and failure.
This iteration tests a different factor: negative supervision scope and its
parameter ownership. Hard-negative sampling is established prior art.

Edit only proposal.py for the model change. Preserve original train.py,
carrier, evaluation, supplied split, features and numerical observation code.
No target isolation cache, new graph edges, features, model parameters,
global RNG draws, evidence scaling or parameter scans.

Commit the single hypothesis and tests; deploy into an isolated immutable
bundle. Verify original controls and actual-environment tests. Run one-epoch
smokes and a preregistered training-only supervision-coverage gate. Freshly
train Baby candidate and joint control before screening; Baby gates Sports.
At most four formal arms and four smokes, seed 2026, max 300 epochs/patience20.
Record metrics/resource use/keep/discard/crash. A live queue is not success.
Numerical failure or a failed scientific/mechanism gate stops the queue.
No automatic retry, threshold relaxation, seed search or test selection.
