# Karpathy autoresearch: instrumented technical recovery

Source: https://github.com/karpathy/autoresearch.git,
revision 228791fb499afffb54b46200aca536f79142f117. Use the installed local
auto-research adapter and read PROTOCOL.md before work.

Working directory: research/20261008_gate_recovery.
Branch: autoresearch/oct8-gate-recovery.
The user explicitly authorized finding a new hypothesis and experimenting
after the unsuccessful cross-dataset soft-isolation result.

1. Preserve the old experiment, wait for its live training to finish, and
   verify completed baseline/evidence artifacts before reuse.
2. Keep proposal.py, isolation.py, train.py, prepare.py and evidence.py
   byte-identical to the failed bundle. Test observation through
   guarded_train.py/numerics.py without changing healthy updates or RNG.
   Wait for the bounded diagnostic to exit before new training. Capture the
   first failed batch before any invalid optimizer update; do not claim an
   unobserved root cause or substitute diagnostic-trained weights.
3. Commit the bounded hypothesis and checks, deploy an isolated immutable
   bundle, run actual-environment tests, then launch only once.
4. Run Baby edge/history integration smokes and fresh formal training.
   Advance to Sports only when every frozen Baby metric guard passes.
5. Record validation R@20 (higher is better), companion metrics, resource use
   and keep/discard/crash. A live queue or a completed smoke is not success.
6. Stop at the declared budget or candidate rejection. No grid searches,
   test-set selection, threshold relaxation or automatic numerical retries.

The server queue executes this declared comparison after the turn. It is not
an autonomous model that invents hypotheses while Codex is inactive.
