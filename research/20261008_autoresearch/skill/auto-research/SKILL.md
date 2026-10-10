---
name: auto-research
description: Run measurable machine-learning experiment loops using karpathy/autoresearch's program.md workflow, including adapting that workflow to an existing experiment. Use for Karpathy autoresearch, auto-research, or requests to conduct iterative training experiments.
---

# Karpathy autoresearch for Codex

This is a local Codex adapter for https://github.com/karpathy/autoresearch,
not an upstream-published SKILL.md. Upstream's lightweight skill is program.md.

Read the project's program.md and scientific protocol before starting. Use the
user-selected upstream; do not substitute a similarly named skills repository.
For this user's BRIDGE workspace, read [current-experiment.md](references/current-experiment.md).

Preserve the core workflow: establish a valid baseline, state one hypothesis,
make a focused change within the experiment's editable scope, commit it, run
with a fixed evaluation protocol and declared budget, record metrics and
resource use in results.tsv, then keep or discard using the declared criterion.
Record crashes separately; missing or non-finite metrics are not improvements.
Use a dedicated experiment branch and preserve failed experiment evidence.

The upstream example trains a language model for five minutes and minimizes
val_bpb. For another domain, explicitly map the editable model, protected data
preparation/evaluator, metric direction, resource budget and stopping rules.
Do not overwrite the existing project's data or environment with the LLM demo.
Use validation for selection, keep test labels outside the search loop, and
compare only compatible configurations and budgets. Do not relax a scientific
gate to turn a failed candidate into a success.

Complete the authorized bounded run. Inspect any active experiment before
launching, and distinguish a live process from an old status snapshot. Keep
algorithm changes separate from infrastructure recovery. A detached fixed job
queue can keep running after the turn; the skill itself is instructions for an
active agent, not a daemon that creates new hypotheses while no agent runs.
