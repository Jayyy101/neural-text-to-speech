# Project state

## Goal and branch

Build a high-quality local Mandarin audiobook system for long Chinese web novels, keeping one narrator consistent across chapters. Priorities are naturalness, faithful pronunciation, controllable pacing, and reliable long-text generation. Azure `zh-CN-XiaoxiaoNeural` is the listening-quality reference, not an exact voice-reproduction target.

Current development branch: **`v2-development`**. The validated CosyVoice3 RL production backend and the native Windows user-facing audiobook UI milestone are accepted. The production-backend checkpoint is **`93d4515` — Promote validated CosyVoice3 RL audiobook backend**. Performance profiling is complete; the accepted generator and UI remain frozen.

## Current validated audiobook backend

- **CosyVoice3 RL (`llm.rl.pt`) is the validated production generator.** The normal `run` CLI uses the pinned local RL checkpoint through a verified model view and the unchanged short Xiaoxiao zero-shot reference. The older scene workflow remains available for historical runs and through `run --legacy-scenes`.
- New runs freeze native frontend units, certify ordered source spans under versioned mapping policy, add explicit terminal punctuation to a detected unpunctuated chapter heading, and pass frozen text to CosyVoice with frontend bypass. The only extra source-certification equivalence is the verified native terminal `、` to `。` case; it is recorded per affected unit. Production does not insert generic silence, waveform repairs, donor breaths, or automatic `[breath]` tokens. Explicit official control tokens remain pass-through capable.
- Deterministic per-unit seeds, the validated contiguous Han deletion QC, at most three physical attempts after valid rejection, same-WAV infrastructure recovery, atomic manifest persistence with bounded OneDrive lock retries, and exact PCM assembly are the production path. Historical accepted runs retain their recorded policies and model configuration.
- **Sustained listening acceptance is complete:** the full Chapter 2 RL run improved known flow and content problems; unseen full Chapter 3 was followed naturally during casual listening, with only minor pause wishes and no recurring serious issue. Chapter 3 produced 63/63 selected units, no QC retries, and a verified sample-exact 16:29.52 chapter WAV. The production generator is now frozen. Do not reopen generator research unless future real usage reveals a serious recurring problem.
- The native Windows UI now provides **filename → paste chapter → Generate → progress and elapsed time → validated finished WAV → Open Folder**. It starts the frozen production CLI through `wsl.exe`; manifests and WAVs remain on the Windows filesystem. No automatic resume/recovery controls are exposed.

## Accepted milestone: Windows user-facing audiobook UI

- Architecture: **native Windows Tkinter UI → `wsl.exe` → frozen production `python -B -m src.audiobook run` CLI** in WSL `cosyvoice-b`. The launcher translates Windows paths, starts one job at a time, and logs worker output. The application layer interprets schema-5 progress and validates final assembly without changing production workflow behavior.
- The normal screen accepts the desired Unicode WAV filename and exact pasted chapter text, then shows generation status, unit progress, and elapsed time. After validating canonical `final/chapter.wav`, it makes a non-overwriting copy with the requested name in the same final folder and enables **Open Folder**. The E1 inspector remains available via an explicit run-directory argument, but is hidden from the normal screen.
- **User-reported manual acceptance:** the native UI launched the frozen WSL backend; a short smoke run and a full Chapter 4 run completed. The full Chapter 4 was comfortable to listen to throughout. A one-off extra-speech artifact in the short smoke run did not recur in that full Chapter 4 run; this does not establish that it can never recur. Chapter 1, the previous known-good quality reference, was regenerated through the current production pipeline and accepted again. Unicode filename export, progress, elapsed time, final WAV validation, and Open Folder were also confirmed.
- This acceptance is manual end-to-end and listening evidence, separate from model-free tests. Neither the UI nor this milestone modifies the frozen backend. Retain unrelated evaluation artifacts and baseline evidence until separately reviewed.

## Completed performance-profiling milestone

- Optional, disabled-by-default profiling measures the Windows UI, WSL worker, planning, model loading, per-unit inference, QC, persistence, and assembly without changing the frozen generation policy. Profiling traces and GPU logs are generated under Git-ignored `outputs/`; `evaluation/summarize_performance.py` reports inclusive phase totals and critical-path accounting separately.
- **Real profiled Chapter 4, accepted by listening:** Generate → Complete took approximately **369.57 s** for a finished audiobook of approximately **597 s**. CosyVoice synthesis took approximately **318.22 s** and content QC approximately **12.34 s**. All **38/38 units passed on their first attempt**, with **0 retries** and **0 QC rejections**. Synthesis is the dominant runtime cost.
- An external `nvidia-smi` measurement during a separate representative Chapter 4 generation found active but bursty GPU utilization during inference: approximately **56.2% mean**, **54% median**, and **99% peak**; VRAM usage was approximately **10.7–14.3 GB**. There were no sustained between-unit idle gaps. Windows GPU utilization had a nonzero idle baseline, and process-level WSL attribution was unavailable, so this measurement is inconclusive for a specific optimization and revealed no obvious quality-neutral speed win.
- **Project decision:** Current speed is acceptable. Defer further performance optimization indefinitely unless future real use makes it necessary. Do not pursue deeper inference optimization, parallel synthesis, precision changes, batching, or other speed work before project completion. Quality, reliability, determinism, and the accepted listening experience take priority; profiling remains an optional diagnostic capability.

### Future user-facing audio format

Keep WAV/PCM inside the validated backend for synthesis, exact assembly, validation, and resumability. A future product output should support desirably named `.mp3` audiobook files, consistent with the prior Luvvoice workflow: **validated final WAV → encode/export MP3 → expose MP3 as the normal user-facing audiobook file**. MP3 conversion is not implemented in this milestone; the canonical backend WAV remains unchanged.

## Repository cleanup

Batch 1 removed 88 verified-empty root temporary directories and committed only the `.gitignore` additions for `.DS_Store` and `Thumbs.db` (`d1f2161`). Further cleanup remains under review; preserve all untracked evaluation scripts, tests, and chapter inputs until separately classified. The accepted backend and UI remain frozen.

## Historical milestone A — MeloTTS baseline capture

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

Full versions, source/checkpoint hashes, and capture limitations: [melo-environment.json](../evaluation/baseline/melo-environment.json) and [melo-pip-freeze.txt](../evaluation/baseline/melo-pip-freeze.txt). These are historical records; their initial pending-validation notes predate the completed GPU review. Fresh-environment recreation remains unverified. The legacy `requirements.txt` uses `melo-tts`, while the installed distribution is `melotts`; do not reinstall the working environment to resolve this documentation discrepancy.

## Historical MeloTTS limitations and evaluation status

- `_` is a confirmed minimal preprocessing failure. `chapter_01.txt` triggers `AssertionError` in installed `melo/text/chinese.py` after mixed-language normalization retains the underscore. Preserve the original failing corpus case.
- `100%` normalizes to `一百`, losing percentage meaning; `01` loses its leading zero. Numbers and special formatting need future normalization work.
- At the Melo baseline milestone, short-passage stability did not establish multi-chapter reliability; application-level chunking, stitching, and resumable jobs were still future work. The accepted CosyVoice backend now has unit planning, exact assembly, and resume. Pronunciation overrides remain future work.
- Legacy Melo GUI threading issues and backend edge cases were outside the baseline and model bring-up milestones.
- The runner continues after failed trials, records exact inputs/settings/timings/tracebacks, uses descriptive WAV filenames, and summarizes results. `--open-output` is optional.
- Corpus: [mandarin_diagnostics.json](../evaluation/inputs/mandarin_diagnostics.json). Runner: [run_melo_smoke.py](../evaluation/run_melo_smoke.py). Model-free tests: [test_melo_baseline.py](../tests/evaluation/test_melo_baseline.py).

## Historical model selection decisions

- Keep the original MeloTTS implementation, environment, and benchmark evidence for reproducible comparisons.
- **CosyVoice3 (`Fun-CosyVoice3-0.5B-2512`) is the selected production foundation.** Melo remains the historical baseline and legacy GUI backend. The separate native Windows audiobook UI invokes CosyVoice through WSL.
- Narrator identity is satisfactory with the preferred approximately 9.8-second Xiaoxiao-style reference. Milestone B model/voice searching is closed.
- Use the same diagnostic source text across models and record model-specific settings. Keep Azure Xiaoxiao as the quality reference for naturalness, pacing, pronunciation, and audiobook suitability.
- The final backend was chosen through evidence, including narrator consistency and sustained listening.

## Historical milestone B — CosyVoice bring-up

- Isolated WSL2 Ubuntu 22.04 installation: `/home/jay/CosyVoice`; Conda `cosyvoice-b`, Python 3.10.21, PyTorch/TorchAudio 2.3.1+cu121. CUDA inference validated on the RTX 4070 Ti SUPER.
- Model: `/home/jay/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B`; `load_trt=False`, `load_vllm=False`, `fp16=False`, WeText frontend. DeepSpeed was intentionally removed after an inference import required `CUDA_HOME`; do not reinstall it unless training becomes necessary.
- **Measured stock-reference benchmark:** 14/14 valid WAVs, warm RTF approximately 0.50–0.54 (first narration 0.711), peak **PyTorch CUDA allocated** memory approximately 3.4–3.5 GiB. Long outputs: 174.32/172.80 seconds, eleven internal chunks each. Both mixed-language trials succeeded. This is diagnostic-passage evidence, not whole-novel readiness.
- **User listening:** stable identity/volume, good Mandarin and numbers, generally smooth audio joins; pacing, pauses, dialogue/emotion, repetitive cadence, occasional repetition/name readings, and normalization remain limitations.
- **Separate manual reference experiments:** 27.048-second reference produced approximately 154.28 seconds of rushed narration; 9.816-second reference produced approximately 160.08 seconds with better pacing and fewer prompt-length warnings. Prefer the short reference. These are user-reported experiments, not the fourteen-trial stock-reference benchmark.
- Preferred local reference: `/home/jay/CosyVoice/reference_audio/xiaoxiao_narrator_short.wav`, with matching `.txt`; keep audio outside Git. Exact transcript and reproduction commands are in the milestone record.

Full findings, setup provenance limits, historical float-WAV compatibility failure, reference comparison, and reproduction: [COSYVOICE_MILESTONE_B.md](../evaluation/COSYVOICE_MILESTONE_B.md). Measured run: [manifest.json](../outputs/evaluation/cosyvoice_baseline_2026-09-13_15-34-53/manifest.json) (local, Git-ignored). The closeout runner adds reference CLI/provenance and clearer failure accounting; historical manifests remain unchanged. No new test suite or closeout GPU rerun.

## Historical milestone C — Audiobook narration and prosody

- Coherent-passage generation remains preferred. Hard sentence-by-sentence generation produced undesirable performance resets, while whitespace/newlines did not reliably control prosody.
- A targeted **140 ms** silence addition improved deficient or borderline period pauses without materially changing already-good pauses. Multiple repairs remained natural.
- [repair_pause.py](../evaluation/repair_pause.py) performs sample-preserving quiet-valley insertion near a supplied timestamp; [apply_pause_plan.py](../evaluation/apply_pause_plan.py) safely applies manual multi-boundary plans against original-audio coordinates.
- Separately generated scenes retained narrator identity and joined without awkwardness or artifacts. In the final user listening comparison, **0 ms of extra inserted silence was preferred**; 700 and 1000 ms both felt too long. Zero extra silence preserves the natural trailing and leading silence already present in the two generated clips.
- Historical scene policy: join separately generated scenes with 0 ms added silence by default and add silence only when listening to that specific join justifies it.
- Artifact/repetition policy: flag and regenerate the affected semantic chunk with the same narrator/settings; do not build or apply a complex waveform-repair model during this milestone.
- The quiet-region heuristic missed a known boundary by approximately 398 ms. The Mandarin CTC experiment missed it by approximately 657 ms despite correct slice timing, resampling, and frame-spacing mechanics. Both remain evaluation-only; automatic alignment is deferred.
- **90/90 model-free tests passed** in the isolated WSL `tts-align` environment after the closeout documentation changes on 2026-09-15. No model or GPU synthesis was invoked by this suite.

Full findings, exact scene-break text, user listening judgments, production/evaluation boundaries, rejected alignment results, and future work: [COSYVOICE_MILESTONE_C.md](../evaluation/COSYVOICE_MILESTONE_C.md).

## Historical milestone D — Scene-workflow audiobook backend

- D1 validates one UTF-8 chapter, preserves an exact `source.txt`, and creates a deterministic explicit-marker scene plan with stable IDs, source spans, and hashes.
- D2 generates and validates scene WAV attempts through one initialized CosyVoice3 adapter while recording backend, model, environment, narrator, and artifact provenance.
- D3 adds resumable incomplete runs and targeted regeneration without overwriting prior attempts.
- D4 integrates the validated manual pause-plan workflow and exact PCM chapter assembly. Assembly preserves natural clip silence and adds 0 ms extra silence.
- D5 adds the end-to-end `run` command and explicit seeded regeneration. Seeded attempts record their process-wide Python, NumPy, CPU Torch, and CUDA Torch random-state policy. Duplicate seeded takes remain historical attempts and do not replace the prior selection.
- D5 real acceptance processed a three-scene Mandarin chapter through planning, CosyVoice3 generation, targeted seeded regeneration, and assembly. Boundary diagnostics confirmed that chapter assembly was sample-exact; a brief boundary-area noise was already present in a generated scene's leading silence and was addressed through targeted regeneration.
- Historical scene artifact policy: listen or detect the affected scene, request one explicit seeded regeneration, validate it, then reassemble. This predates the accepted schema-5 unit retry policy.
- A structurally valid WAV is not proof of perceptual quality. Listening remains necessary for narration artifacts, repetitions, pronunciation, pacing, and boundary perception.
- Source snapshots, generation attempts, repairs, selections, hashes, and assembly frame offsets are preserved in the run manifest. Regeneration invalidates repairs bound to an old selection and marks completed assembly stale.
- The supported CLI is `python -B -m src.audiobook` with `plan`, `run`, `generate`, `resume`, `regenerate`, `repair`, and `assemble`. Practical commands and the run layout are documented in the root [README](../README.md).
- **143/143 model-free tests passed** again during the D6 closeout in the isolated WSL `tts-align` environment. CLI help was also verified without loading CosyVoice.

## Historical scene-workflow policy (milestones C–D)

- CosyVoice3 is the selected Mandarin audiobook backend. The preferred short narrator reference and isolated `cosyvoice-b` environment remain unchanged.
- Prefer continuous coherent scenes. Use standalone `***` source lines for intentional semantic scene boundaries.
- Preserve generated natural silence and assemble with 0 ms extra inter-scene silence.
- Use the manual +140 ms period repair only for a confirmed sparse pause issue. Automatic punctuation inference and alignment remain deferred.
- Treat perceptual artifacts as generation issues unless sample-level evidence identifies assembly behavior. Exact PCM assembly neither introduces nor repairs samples within a selected scene.

## Historical milestone E1 — Read-only existing-run inspector

- Milestone E follows a thin local architecture: Tkinter/ttk presentation -> application/state interpretation -> the existing Milestone D backend and persisted run artifacts. The run manifest and artifacts remain the source of truth; there is no independent UI database.
- E1 was the read-only inspector. The subsequently accepted user-facing UI adds new-run generation; scene regeneration/manual repair and recovery/product hardening remain outside its visible workflow.
- [audiobook_application.py](../src/audiobook_application.py) interprets supported Milestone D schemas and reuses existing plan, attempt, repair, and artifact validation rules without writing run state.
- [audiobook_ui.py](../src/audiobook_ui.py) opens or refreshes a run and presents source identity, generation and assembly status, latest operation, ordered scenes, attempts, seed provenance, errors, repairs, and audio metadata. It can open only a validated selected-scene artifact or valid current final chapter in the system player.
- Missing historical seed metadata is reported as not recorded. A missing or invalid selected repair remains an error and never falls back silently to generated audio.
- Launch the normal generation UI with `python -B -m src.audiobook_ui`. Supplying a run directory additionally opens the historical inspector for maintenance.
- Manual acceptance passed on the real D5 run: loading, scene selection, selected-scene playback, final-chapter playback, refresh, and displayed persisted state all matched expectations.
- **154/154 model-free tests passed** in the isolated WSL `tts-align` environment during E1 closeout. This includes 11 focused application-layer tests and the existing audiobook regression coverage. No model or GPU synthesis was invoked.
- Listening found some very short or abrupt scene transitions in the final chapter. This is consistent with the current 0 ms added inter-scene silence policy and is recorded for later listening/evaluation. It is not an E1 defect, and E1 does not change assembly or pause behavior.

## Historical unit-path development notes

The schema-5 unit path freezes and certifies frontend units, generates each frozen unit with a deterministic seed, and assembles selected unit PCM exactly. Step 2 real acceptance passed on the saved two-unit fixture: one warm CosyVoice3 model, two independent attempts, exact PCM assembly with zero injected silence, and unchanged selections/seeds/hashes after resume. Chapter-title punctuation was subsequently promoted as a recorded first-unit synthesis override.

Step 3 added a required content-QC gate for **new** schema-5 runs: a persistent audio-only ASR worker in `tts-align`, deterministic Han comparison, and the fixed contiguous expected deletion threshold of four. Atomic per-attempt sidecars bind the evidence; assembly requires passed evidence. Real acceptance passed on the two-unit fixture: both units passed, one worker served both, exact PCM assembly was preserved, and resume made no new TTS or ASR requests. CosyVoice and ASR CUDA execution completed without failure.

Step 4 added a persisted `bounded_content_qc_retries_v1` policy to new QC-enabled runs: at most three physical synthesis attempts per unit. Only a validated content rejection advances the logical take and its deterministic seed. An interrupted synthesis attempt reuses its logical take seed within the physical-attempt budget. QC infrastructure errors retry recognition on the same WAV; three valid rejections persist exhaustion, leave the unit unresolved, and block assembly. Rejected WAVs and sidecars remain append-only evidence. Existing Step 3 QC runs without the retry policy and Step 2 schema-5 runs without QC retain their recorded behavior; historical schemas 2–4 remain unchanged. Model-free tests cover these paths; the Chapter 1 full-chapter acceptance run recorded a confirmed omission in unit 66 and a successful second deterministic attempt.

The accepted Windows UI now provides the simple paste → Generate → finished audiobook path over the validated schema-5 RL backend. The historical E1 inspector remains available for maintenance. Recovery controls and scene repair are outside the accepted main UI workflow. Performance investigation is complete; the frozen backend and UI remain the accepted production path.

Automatic alignment, automatic pause placement, perceptual artifact detection, random retry loops, crossfades, mastering, MP3 export, EPUB/PDF/DOCX ingestion, deployment, and cloud infrastructure remain separate future work.

Begin future work by reading [AGENTS.md](../AGENTS.md), this handoff, and the root [README](../README.md); verify the branch and worktree before editing. Keep the selected narrator, original corpus, historical outputs, baseline evidence, `melo` environment, legacy GUI, and external CosyVoice source/model files intact unless a later milestone explicitly authorizes changes. Do not commit or push without explicit instruction.
