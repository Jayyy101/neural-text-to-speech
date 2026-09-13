# Project instructions

## Purpose

Build a high-quality local Mandarin audiobook generation system for long Chinese web novels, with one consistent narrator across chapters. Azure `zh-CN-XiaoxiaoNeural` is the quality reference for naturalness, pacing, pronunciation, and audiobook suitability; reproducing Microsoft's proprietary voice exactly is not the goal.

## Working principles

- Read [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md) before starting milestone work, then inspect the relevant code and evidence.
- Work on `v2-development` unless the user explicitly directs otherwise. Check the current branch and working tree before editing; preserve unrelated user changes.
- Preserve MeloTTS as the original working baseline. Do not assume it is the final production backend.
- Test new TTS models in isolated environments. Do not upgrade, replace, or install their dependencies into the existing `melo` environment.
- Prefer small, milestone-scoped changes. Avoid unrelated refactors and premature GUI integration.
- Prioritize long-form reliability, narrator consistency, pronunciation, and naturalness over GUI polish.
- Explain major architectural changes before implementing them.
- Preserve reproducibility: record exact source text, model revision, narrator/settings, environment, and results for comparisons.
- Record known failures instead of hiding them. Do not sanitize shared diagnostic inputs merely to make a model pass.
- Never overwrite original MeloTTS baseline evidence: captured environment records, original corpus version, run manifests, or audio. Put new experiments and outputs in separate locations; version deliberate future corpus changes.
- Preserve the baseline backend and GUI behavior unless a later milestone explicitly authorizes changes.
- Run relevant tests after changes. Model-free checks: `python -B -m unittest discover -s tests -v`. Treat GPU synthesis and listening as separate validation; report which checks actually ran.
- Do not commit or push unless explicitly instructed.

## Evidence and handoff

- [evaluation/README.md](evaluation/README.md): completed baseline findings, known failures, and evaluation workflow.
- `evaluation/baseline/`: historical environment/package capture; an inventory, not a validated installation lockfile.
- `evaluation/inputs/mandarin_diagnostics.json`: shared audiobook diagnostic corpus.
- `outputs/evaluation/`: local, Git-ignored run evidence; it may be absent in a fresh clone.
- Keep the project-state handoff concise and update milestone outcomes when authorized. Distinguish historical capture, measured results, manual listening judgments, and future plans.
