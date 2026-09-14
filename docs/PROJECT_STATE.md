# Project state

## Goal and branch

Build a high-quality local Mandarin audiobook system for long Chinese web novels, keeping one narrator consistent across chapters. Priorities are naturalness, faithful pronunciation, controllable pacing, and reliable long-text generation. Azure `zh-CN-XiaoxiaoNeural` is the listening-quality reference, not an exact voice-reproduction target.

Current development branch: **`v2-development`**. Milestone A was committed and pushed as **`e6f8197` — Complete MeloTTS baseline evaluation milestone**. **Milestone B is complete**; its closeout changes are not yet committed. Milestone C is the next active milestone; its pipeline features are not implemented yet.

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
- **CosyVoice3 (`Fun-CosyVoice3-0.5B-2512`) is the selected experimental foundation** for continued development. Melo remains the historical baseline and current legacy application backend; CosyVoice is not integrated into the GUI.
- Narrator identity is satisfactory with the preferred approximately 9.8-second Xiaoxiao-style reference. Milestone B model/voice searching is closed.
- Use the same diagnostic source text across models and record model-specific settings. Keep Azure Xiaoxiao as the quality reference for naturalness, pacing, pronunciation, and audiobook suitability.
- Choose the final backend through evidence, including narrator consistency and sustained listening, rather than GUI integration or short demos alone.

## Completed milestone: B — CosyVoice Bring-Up

- Isolated WSL2 Ubuntu 22.04 installation: `/home/jay/CosyVoice`; Conda `cosyvoice-b`, Python 3.10.21, PyTorch/TorchAudio 2.3.1+cu121. CUDA inference validated on the RTX 4070 Ti SUPER.
- Model: `/home/jay/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B`; `load_trt=False`, `load_vllm=False`, `fp16=False`, WeText frontend. DeepSpeed was intentionally removed after an inference import required `CUDA_HOME`; do not reinstall it unless training becomes necessary.
- **Measured stock-reference benchmark:** 14/14 valid WAVs, warm RTF approximately 0.50–0.54 (first narration 0.711), peak **PyTorch CUDA allocated** memory approximately 3.4–3.5 GiB. Long outputs: 174.32/172.80 seconds, eleven internal chunks each. Both mixed-language trials succeeded. This is diagnostic-passage evidence, not whole-novel readiness.
- **User listening:** stable identity/volume, good Mandarin and numbers, generally smooth audio joins; pacing, pauses, dialogue/emotion, repetitive cadence, occasional repetition/name readings, and normalization remain limitations.
- **Separate manual reference experiments:** 27.048-second reference produced approximately 154.28 seconds of rushed narration; 9.816-second reference produced approximately 160.08 seconds with better pacing and fewer prompt-length warnings. Prefer the short reference. These are user-reported experiments, not the fourteen-trial stock-reference benchmark.
- Preferred local reference: `/home/jay/CosyVoice/reference_audio/xiaoxiao_narrator_short.wav`, with matching `.txt`; keep audio outside Git. Exact transcript and reproduction commands are in the milestone record.

Full findings, setup provenance limits, historical float-WAV compatibility failure, reference comparison, and reproduction: [COSYVOICE_MILESTONE_B.md](../evaluation/COSYVOICE_MILESTONE_B.md). Measured run: [manifest.json](../outputs/evaluation/cosyvoice_baseline_2026-09-13_15-34-53/manifest.json) (local, Git-ignored). The closeout runner adds reference CLI/provenance and clearer failure accounting; historical manifests remain unchanged. No new test suite or closeout GPU rerun.

## Next active milestone: C — Audiobook Narration & Prosody Pipeline

Scope: sentence-ending pause behavior, paragraph/scene pauses, pacing, semantic/narration-aware segmentation, long-form continuity, dialogue handling, repetition detection/prevention, and eventual pronunciation controls. No Milestone C behavior was added during B closeout.

Begin by reading [AGENTS.md](../AGENTS.md), this handoff, and the Milestone B record; verify branch/worktree and available evidence. Explain a scoped pipeline experiment before implementation. Keep the selected narrator, original corpus, and old outputs intact; record new evidence separately. Preserve the `melo` environment, baseline backend/GUI, and external CosyVoice source/model files unless later work explicitly authorizes changes. Do not commit or push without explicit instruction.
