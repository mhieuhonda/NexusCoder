# Contributing to Nexus Coder

Thanks for your interest in contributing! This project is an open AI
architecture in active development. Both humans and AI agents are welcome.

> **AI agents:** read `AGENTS.md` first — it is written specifically for you.

## Code of Conduct

Be respectful. This project is built by a small team with limited resources.
Good-faith contributions are valued; trolling, spamming, or fake claims are not.

## What We Need Help With

1. **Running the small configs** — verify `tiny` / `small` train and run on CPU.
2. **Testing skills & tools** — exercise `nexus/skills/` and `nexus/tools/`.
3. **Reviewing integrations** — verify patterns adapted from upstream projects.
4. **Tests** — `tests/` is thin; add coverage for model layers, tokenizer, tools.
5. **Docs** — architecture docs always need improvement.
6. **Training experiments** — if you have GPUs, try a small real training run
   and report honestly what you observed.

## Getting Started

```bash
git clone https://github.com/mhieuhonda/NexusCoder.git
cd NexusCoder
python3.12.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Python version is **3.12.13 (strict)**. Use `pyenv` or similar to match it.

## Contribution Workflow

1. **Open an issue first** describing what you plan to do (check for existing
   ones to avoid duplication).
2. **Fork the repo** and create a branch.
3. Make your changes, keeping them **small and focused**.
4. **Verify** your change locally before opening a PR.
5. Open the **pull request** and describe what you did and how you verified it.

## Style

- Follow the existing code style in the file you are touching.
- Add or update tests for any new code.
- Keep commit messages clear and descriptive.

## Labels

- `good first issue` — beginner-friendly tasks (agents: start here)
- `help wanted` — tasks where maintainers explicitly want outside help
- `bug` — something is broken
- `enhancement` — new feature or improvement

## License & Attribution

Contributions are licensed under **NAL-1.0** (Attribution Required). By
contributing, you agree your changes are covered by this license and that the
original author **Hieu Louis** (github.com/mhieuhonda) retains attribution
requirements. See `LICENSE` and `ATTRIBUTIONS.md`.

## Questions

Open an issue, or reach out through the **code-realm** community on Moltbook.
