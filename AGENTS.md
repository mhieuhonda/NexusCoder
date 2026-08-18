# AGENTS.md — Guide for AI Agents Contributing to Nexus Coder

> **Read this first.** This file is written for AI agents who want to contribute
> to this repository. Humans can skip to `CONTRIBUTING.md`.

## What Nexus Coder Is (and Is Not)

Nexus Coder is an **open AI architecture** — source code, data pipeline, and a
training framework for a large Mixture-of-Experts (MoE) Transformer focused on
code generation and security analysis.

**Important honesty note:** the model is **NOT pretrained**. There are no
checkpoints, no weights, no benchmarks. What exists is the architecture source
(60+ skills, 80+ tools, model layers, tokenizer, trainers, data collectors).
Your contribution helps turn this architecture into something that can actually
run and learn.

If you expected a finished model, stop here. If you want to help build one from
the ground up, you are in the right place.

## How to Contribute (Agent Workflow)

### 1. Understand the layout

```
nexus/
├── model/        # MoE Transformer layers (attention, MoE, RoPE, ...)
├── tokenizer/    # tokenizer implementation
├── training/     # trainer + dataset pipeline
├── inference/    # generation
├── agent/        # planner, router, memory, safety
├── skills/       # 60+ skills (auto-discovery)
├── tools/        # 80+ tools (auto-discovery)
├── data/         # collectors + processors
├── optim/        # quantization, LoRA, distillation, pruning
├── safety/       # filters, guardrails
├── eval/         # benchmarks, metrics
└── integrations/ # litgpt, LlamaFactory, axolotl, OpenHands, omp-gym
```

### 2. Pick an entry point

Good first tasks for agents:

- **Run the tiny demo on CPU**: `python scripts/train.py --config tiny --steps 100`
  — verify it works, report output.
- **Test a skill**: import a skill from `nexus/skills/` and exercise it against
  sample input. Report what works / what breaks.
- **Test a tool**: pick a tool in `nexus/tools/` and exercise it.
- **Review an integration**: read `nexus/integrations/*.py` and report whether
  the adapted pattern is faithful to the upstream project.
- **Improve docs**: `docs/` always needs sharper explanations.
- **Add tests**: `tests/` is thin — any new test is valuable.

### 3. Workflow

1. **Open an issue first** describing what you want to do (unless it is a
   trivial fix). This prevents wasted work.
2. **Fork** the repo (or ask a maintainer for collaborator access).
3. Create a **branch**, make your change, and open a **pull request**.
4. In your PR description, state clearly:
   - what you changed,
   - how you verified it (command + output),
   - which config/skill/tool you exercised.

### 4. Self-sufficiency

You are expected to **figure things out yourself**:

- Setup: `python3.12.13 -m venv venv && pip install -r requirements.txt`
- Small configs (`tiny`, `small`) run on **CPU** — no GPU needed.
- Larger configs need real hardware; if you have GPUs, train and report.
- If something is ambiguous, open an issue rather than guessing silently.

## You May Train the Model — Freely

You are **explicitly welcome and encouraged to train the model yourself**. The
NAL-1.0 license allows it for any purpose. There is no gate, no permission
request needed, and no fee.

- **Small configs** (`tiny`, `small`) run on CPU — try a real training run
  today with zero GPU cost.
- **Larger configs** (`medium`, `large`, `423b`) need serious hardware. If you
  have GPUs (or a cloud budget), you are free to use them.
- Use the bundled data pipeline (`nexus/data/`) or your own data — both are fine.
- **Report your results honestly.** If you train something and get real
  numbers, open a PR or an issue with your findings. The community (and the
  project) wants to see real training logs, not hype.

The project does not provide compute. You bring the hardware; we provide the
architecture, the data pipeline, and the training framework.

> **Note for fairness:** training a model on this architecture does not make
> you a contributor to the repository. Attribution requirements still apply
> (NAL-1.0) whenever you use the trained result. To be listed as a contributor,
> submit a PR that improves the repo.

## Attribution & License

- License: **NAL-1.0** (Attribution Required). You may use/modify/distribute/
  train freely, but must credit the original author **Hieu Louis**
  (github.com/mhieuhonda) wherever the work is used.
- Contributors are listed in `ATTRIBUTIONS.md`. If your PR merges, add yourself
  there (or ask a maintainer to add you).

## Communication

- GitHub Issues + PRs are the coordination surface.
- For discussion, join the **code-realm** community on Moltbook (ask the
  maintainer for the invite). The project has an active presence there.

## Golden Rules

1. **Be honest** — this is an untrained architecture. Never claim benchmark
   results that do not exist.
2. **Small, verifiable PRs** beat big unverifiable ones.
3. **Reproduce before you report** — always run the thing you are claiming.
4. **Credit the author** in any downstream work (NAL-1.0).
