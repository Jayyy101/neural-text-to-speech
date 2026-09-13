# Project state

## Goal and branch

Build a high-quality local Mandarin audiobook system for long Chinese web novels, keeping one narrator consistent across chapters. Priorities are naturalness, faithful pronunciation, controllable pacing, and reliable long-text generation. Azure `zh-CN-XiaoxiaoNeural` is the listening-quality reference, not an exact voice-reproduction target.

Current development branch: **`v2-development`**. Milestone A was committed and pushed as **`e6f8197` — Complete MeloTTS baseline evaluation milestone**. No later implementation milestone is complete.

## Completed milestone: A — Capture the MeloTTS baseline

- **Complete:** environment/hardware capture, seven diagnostic inputs, evaluation runner, GPU baseline, and manual listening review.
- **19/19 model-free tests passing** at milestone completion.
- **12 successful synthesis trials, 2 expected mixed-text failures, 0 incomplete trials.** The run correctly remains `completed_with_failures`.
- Narrator identity is highly consistent across repeated generations; Mandarin is intelligible and stable.
- The approximately **2.4-minute** long passage completed successfully without obvious narrator collapse (145.168 / 144.750 seconds of audio).
- Throughput is very fast after warm-up: long-passage backend timings were 4.678 / 4.484 seconds, excluding model loading and imports.
- Naturalness is below the desired audiobook target. Dialogue has limited expressive differentiation from ordinary narration.
- These observations establish a technical baseline, not whole-novel readiness. **MeloTTS remains baseline only**, not the selected final audiobook backend.

Final run: `outputs/evaluation/melo_baseline_2026-09-13_02-30-28/manifest.json` (local, Git-ignored). The full findings and reproduction procedure are in [evaluation/README.md](../evaluation/README.md).

## Captured MeloTTS environment and hardware

| Component | Baseline |
|---|---|
| OS / environment | Windows 10 build 19045; Conda `melo`; Python 3.10.20 |
| Interpreter | `C:\miniconda3\envs\melo\python.exe` |
| GPU / driver | RTX 4070 Ti SUPER; 16,376 MiB VRAM; driver 591.86 |
| CPU / RAM | Ryzen 7 7800X3D / 32 GB (historical README values) |
| MeloTTS | Distribution `melotts` 0.1.2; Git revision `209145371cff8fc3bd60d7be902ea69cbdb7965a` |
| PyTorch / TorchAudio / TorchVision | 2.11.0+cu126 / 2.11.0+cu126 / 0.26.0+cu126 |
| Transformers / NumPy | 4.27.4 / 1.26.4 |
| Mandarin settings | `language=ZH`, `speaker_name=ZH`, `speed=1.0`; automatic CUDA/CPU selection |

Full versions, source/checkpoint hashes, and capture limitations: [melo-environment.json](../evaluation/baseline/melo-environment.json) and [melo-pip-freeze.txt](../evaluation/baseline/melo-pip-freeze.txt). These are historical records; their initial pending-validation notes predate the completed GPU review. Fresh-environment recreation remains unverified. The root README/requirements use `melo-tts`, while the installed distribution is `melotts`; do not reinstall the working environment to resolve this documentation discrepancy.

## Known limitations and evaluation status

- `_` is a confirmed minimal preprocessing failure. `chapter_01.txt` triggers `AssertionError` in installed `melo/text/chinese.py` after mixed-language normalization retains the underscore. Preserve the original failing corpus case.
- `100%` normalizes to `一百`, losing percentage meaning; `01` loses its leading zero. Numbers and special formatting need future normalization work.
- Short-passage stability and voice consistency do not establish multi-chapter reliability. Application-level chunking, stitching, resumable jobs, and pronunciation overrides are future work.
- Existing GUI threading issues and backend edge cases remain unchanged; do not fold their repair into model bring-up.
- The runner continues after failed trials, records exact inputs/settings/timings/tracebacks, uses descriptive WAV filenames, and summarizes results. `--open-output` is optional.
- Corpus: [mandarin_diagnostics.json](../evaluation/inputs/mandarin_diagnostics.json). Runner: [run_melo_smoke.py](../evaluation/run_melo_smoke.py). Model-free tests: [test_melo_baseline.py](../tests/test_melo_baseline.py).

## Decisions and model strategy

- Keep the original MeloTTS implementation, environment, and benchmark evidence for reproducible comparisons.
- Evaluate **CosyVoice first**; the previously selected candidate is `Fun-CosyVoice3-0.5B-2512`. Verify the exact upstream revision and requirements before installation.
- Qwen3-TTS is a possible later comparison. Fish Speech and other models are later candidates, not current installation scope.
- Use the same diagnostic source text across models and record model-specific settings. Keep Azure Xiaoxiao as the quality reference for naturalness, pacing, pronunciation, and audiobook suitability.
- Choose the final backend through evidence, including narrator consistency and sustained listening, rather than GUI integration or short demos alone.

## Next milestone: B — CosyVoice Bring-Up

Goals: create a separate environment; install and validate CosyVoice without touching `melo`; confirm RTX 4070 Ti SUPER compatibility; generate Mandarin samples from the same diagnostic corpus. **Do not integrate into the GUI or replace MeloTTS yet.** Milestone B has not started.

Do not change the existing `melo` packages, production backend/GUI behavior, captured environment files, original diagnostic corpus, or original baseline run evidence as part of bring-up. Add isolated experiments and new output directories. Do not commit or push without an explicit instruction.

Recommended first steps for the next session:

1. Read root [AGENTS.md](../AGENTS.md), this handoff, and the evaluation README; verify `v2-development`, Git status, and available baseline artifacts.
2. Check current official CosyVoice setup guidance against Windows, Python/CUDA requirements, and the 16 GB GPU. Decide whether native Windows or WSL2 is appropriate; do not assume compatibility is already validated.
3. Explain the scoped environment/runner plan before installation. Reuse the corpus and keep dependencies and outputs isolated.
4. Bring up one Mandarin sample first, then the shared diagnostic cases; record the exact model/reference voice/settings, resource usage, successes, and failures. Run relevant model-free tests after repository changes.
