# Interest-negative MSCA pilot: autoresearch program

This is a bounded adaptation of [Karpathy autoresearch](https://github.com/karpathy/autoresearch),
using the installed local Codex adapter at
`C:/Users/zixua/.codex/skills/auto-research/SKILL.md`. Upstream is identified by
`research/20261008_autoresearch/skill-source.json`; upstream supplies `program.md`,
not this local adapter's `SKILL.md`. Read `PROTOCOL.md`, `PAPER_SCOPE.md`,
`result/FAILED_EXPERIMENTS.md`, and `result/PUBLICATION_PROTOCOL.md` first.

The single hypothesis is that original MSCA learns more useful distinctions
when half of its original BPR negatives come from a fixed TRAIN-interest pool.
The training technique is inspired by [PerPEFT](https://arxiv.org/html/2602.09445v1),
not an official reproduction or evidence of algorithmic novelty. The intervention
is only the negative distribution. It introduces no adapter, gate, graph edge,
review signal, new feature file, initialization or inference score.

Use the isolated experiment repository on branch
`autoresearch/oct10-interest-negative`; commit code, this program, protocol,
configuration and checks before metrics. The remote source carrier is
`/data/sunzixuan/MSCA-main`, model SHA256
`066aafcf2e5373b20b2fdf9b3660302b8dfcaf580710d3ab864fac4f19e2360b`.
The isolated remote run root is `/data/sunzixuan/interest_negative_20261010_v1`.
Do not modify the original source, inputs, splits, evaluator or frozen old runs.

The authorized queue is finite:

1. Check current process identity, deployed hashes and available GPU1/RAM. A
   snapshot is evidence only at its recorded `checked_unix`; deployment alone
   does not establish training. Do not launch twice or overlap another owned run.
2. Validate sampler invariants and official carrier compatibility. Prepare and
   hash one TRAIN-only group/pool asset version; fix the actual KMeans backend
   before metrics and reuse its labels/pools. No result-dependent backend choice.
3. Run exactly two fresh Baby one-epoch GPU engineering smokes: uniform, then
   semantic. Check full-epoch finite updates, data/RNG/integrity guards and resource
   receipts. Smokes are not gain evidence or tuning; discard their weights.
4. Restart fresh seed2026 for Baby uniform, semantic and shuffled formal arms,
   with identical initial model and positive order. Only negative-item draws
   differ. Each arm has at most300 epochs, official stopping_step20, and8 hours.
5. Evaluate the complete Baby conjunction in `PROTOCOL.md`. A failure freezes
   this K8/mix0.5 point and closes the queue. No Sports, changed seed or parameter
   scan may rescue it. Only a full Baby pass permits the same three fresh Sports
   arms and the same gate, using Sports' pre-existing official recipe.
6. Preserve every arm, crash, timeout, configuration, best checkpoint and per-user
   validation output. Record formal outcomes in `results.tsv` as `keep`, `discard`
   or `crash`, including the shuffled control. Keep smoke rows separate from
   formal completed-arm counts. A missing/pending/nonfinite arm cannot pass.
7. Report live progress or completed decisions truthfully. Passing both domains
   is only a single-seed pilot; it is not statistical significance, publication
   evidence, a new algorithm or permission to silently launch confirmation.

Metric mapping: autoresearch's language-model `val_bpb` is replaced by validation
R@20/N@20, both larger-is-better; R@10/N@10 are guards. Selection uses validation
Recall@20 rounded to four decimals with strict improvement; ties keep the earlier
checkpoint. Test-loader construction and test evaluation are forbidden throughout
this pilot. Uniform negatives come from unique items observed in original TRAIN,
excluding every own TRAIN positive; they do not come from the full catalog.

Editable scope is this isolated experiment's sampler, orchestration and reporting
implementation. The original MSCA forward/loss bytes, dataset preprocessing,
graphs, feature arrays and evaluator are protected. Use a wrapper for bounded
validation-only scheduling, guarded updates and sampler injection; the original
negative uniform law, losses, positive order and full-catalog evaluation remain
the reference. Changes required for infrastructure must be declared and audited
before metrics, not folded into an apparent sampler gain.

GPU1 only, exactly one active child, four CPU threads, at least14 GiB free GPU
memory and12 GiB free host RAM at admission, CUDA memory fraction0.55. Total queue
ceiling48 hours includes preparation, waiting, smokes and both domains; exceeding
the wall budget is a technical failure. Do not consume another GPU or weaken a
guard to finish. The skill is instructions for an active agent, while this fixed
detached queue executes only the registered arms; it is not an unbounded daemon
that invents new experiments.
