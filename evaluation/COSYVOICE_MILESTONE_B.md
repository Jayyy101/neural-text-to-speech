# Milestone B — CosyVoice Bring-Up

## Status and purpose

**Complete.** CosyVoice3 was brought up in an isolated WSL environment, validated on the local GPU, and evaluated with the unchanged Mandarin diagnostic corpus. It is the selected experimental foundation for continued audiobook development. MeloTTS remains the preserved historical baseline and the existing application's backend; CosyVoice is not integrated into the GUI.

Narrator voice identity is considered satisfactory. The preferred reference is the approximately 9.8-second Xiaoxiao-style sample described below. Milestone B model/voice searching is closed. Natural audiobook delivery remains work for **Milestone C — Audiobook Narration & Prosody Pipeline**.

## Environment and configuration

| Component | Milestone B configuration |
|---|---|
| Host | Windows 10; NVIDIA driver 591.86 |
| Linux environment | WSL2, Ubuntu 22.04 |
| Conda environment | `cosyvoice-b` |
| Python / interpreter | 3.10.21; `/home/jay/miniconda3/envs/cosyvoice-b/bin/python` |
| PyTorch / TorchAudio | `2.3.1+cu121` / `2.3.1+cu121` |
| CUDA runtime / GPU | CUDA 12.1; NVIDIA RTX 4070 Ti SUPER, 16 GB class |
| External source | `/home/jay/CosyVoice` |
| Model | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` |
| External model directory | `/home/jay/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B` |
| Loading / inference | `load_trt=False`, `load_vllm=False`, `fp16=False`, `stream=False` |
| Text frontend | WeText; upstream text frontend and speed defaults retained |

CUDA inference succeeded. This is FP32 evidence; the earlier proposed FP16 setup was not the configuration used for the recorded benchmark. The Windows `melo` environment was kept separate. Do not use this project's root `requirements.txt` to recreate CosyVoice or install CosyVoice dependencies into `melo`.

**DeepSpeed was intentionally uninstalled:** importing it required `CUDA_HOME`, while this project uses inference only. Do not reinstall it unless training becomes necessary. The external source, models, and local reference audio are not vendored into this project and are not part of the closeout changes.

The environment table combines the user's completed setup report with saved run metadata. The original manifests record Python, interpreter, Linux platform, CUDA build/device, source HEAD, and selected paths/hashes. They do not capture PyTorch/TorchAudio package versions or prove which frontend package was active; those details come from the user's setup report. Closeout did not reinstall or independently recapture the environment.

## Measured benchmark evidence

Full run: [cosyvoice_baseline_2026-09-13_15-34-53/manifest.json](../outputs/evaluation/cosyvoice_baseline_2026-09-13_15-34-53/manifest.json). Its audio and manifest are local, Git-ignored evidence and may be absent from a fresh clone.

The run used the **stock CosyVoice zero-shot reference**, `/home/jay/CosyVoice/asset/zero_shot_prompt.wav`, with the transcript `希望你以后能够做的比我还好呦。` and the required prefix `You are a helpful assistant.<|endofprompt|>`. It did **not** use either Xiaoxiao-style reference.

The [shared corpus](inputs/mandarin_diagnostics.json) contains narration, dialogue, punctuation, names/uncommon vocabulary, numbers, mixed Chinese/English, and a moderately long passage. All seven cases were run twice. The runner forwarded the original text, including the mixed-text failure case from Melo; it introduced no application-level segmentation or text corrections. CosyVoice performed its own internal processing, and the runner concatenated all yielded audio chunks in order without inserting pauses.

One model instance served the whole run. Imports and model loading were timed separately. Per-trial inference timing included zero-shot processing, generation, concatenation, and CUDA synchronization, ending before WAV saving. RTF is inference seconds divided by output audio seconds. The runner cleared unused PyTorch cache and reset its peak-memory counter before each trial. These are local diagnostic timings, not a controlled cross-backend speed comparison.

| Measurement | Recorded result |
|---|---|
| WAV trials | **14/14 passed**, 0 failed |
| Warm inference | Approximately **0.50–0.54 RTF** |
| First narration trial / warm-up | Approximately **0.711 RTF** |
| Peak PyTorch CUDA allocated memory | Approximately **3.4–3.5 GiB**; observed range 3.378–3.496 GiB |
| Long passage, repetition 1 | **174.32 s**, 11 internal chunks |
| Long passage, repetition 2 | **172.80 s**, 11 internal chunks |
| Mixed Chinese/English | Both trials passed WAV checks |
| Output | 24 kHz mono PCM16 WAV |

Inspection confirmed that all fourteen files have complete PCM payloads and match their recorded SHA-256 hashes. The corpus hash and every trial's source text match the unchanged diagnostic corpus. The manifest remains `passed_wav_checks_listening_pending`; the subsequent user listening judgments are recorded separately below rather than rewriting historical run evidence.

The old manifest key `peak_cuda_memory_gb` actually reports `torch.cuda.max_memory_allocated() / 1024**3`: **PyTorch-allocated GiB**, not total process/device VRAM. It excludes allocations outside PyTorch, such as those made by other runtimes. The closeout runner uses `peak_torch_cuda_allocated_gib`; historical manifests remain unchanged.

Recorded provenance from the full run:

- CosyVoice source HEAD: `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`.
- Corpus SHA-256: `cd4d25654d8ea7db7ca944e04aabc124cbfbc31bf55f000fda33f6a028f3aac5`.
- Model configuration SHA-256: `f5a6b2c6f05139d0f18861a1fe506f751e787026b77c05f7e8fef9f8a4405965`.
- Stock prompt WAV SHA-256: `c7b31d6dbe7cc6a716dded00550db5b50940bf209e424e4ad207b12e657c8ff6`.

Source HEAD is not a checkpoint revision or proof of a clean external checkout. The original run did not capture weight hashes, checkpoint snapshot revision, submodule state, or runner hash. Do not infer those missing values from the model name or configuration hash. Fresh-environment restoration and byte-identical resynthesis are unverified.

### Earlier output-format failure

The [15-29-50 trial](../outputs/evaluation/cosyvoice_baseline_2026-09-13_15-29-50/manifest.json) generated a float WAV with format tag 3, which Python's built-in `wave` reader rejected with `unknown format: 3`. This was an evaluator/output-format compatibility issue, not failed speech synthesis. That failed run remains preserved. The runner now explicitly writes signed PCM16 WAV; the [15-33-23 smoke run](../outputs/evaluation/cosyvoice_baseline_2026-09-13_15-33-23/manifest.json) and full benchmark passed WAV checks.

## Manual / subjective listening results

These are the user's listening judgments, not automated scores or new listening performed during closeout.

Strengths: CosyVoice provides a much stronger technical foundation than the existing Melo baseline, with stable narrator identity, consistent volume, good Mandarin pronunciation overall, and excellent numbers/date/time handling. Mixed Chinese/English succeeds where Melo's diagnostic case failed. The moderately long passage is technically stable, and internal chunk joins generally avoid hard audio seams. This evidence does not establish whole-novel reliability.

Weaknesses: audiobook prosody is not yet natural enough. Delivery can feel rushed; sentence-ending pauses are inconsistent, and some periods receive less breathing room than nearby commas. Dialogue is generally narrated rather than acted, emotional variation is weak, and repetitive TTS cadence becomes noticeable over time. Internal boundaries can feel like performance resets even without hard waveform seams. Some manual generations repeated material. Rare names/vocabulary occasionally use unintended readings. WeText normalization can remove useful punctuation distinctions or awkwardly transform identifiers/filenames.

Successful WAV validation does not establish pronunciation accuracy, semantic completeness, absence of repetition, or audiobook quality. Melo's historical findings remain in [evaluation/README.md](README.md), including its stable narrator and known mixed-text failure.

## Manual narrator-reference experiments

These experiments are **outside the official fourteen-trial benchmark manifest**. Durations, formats, warnings, and preferences below are user-reported; corresponding experiment manifests were not found in this project's saved outputs during inspection.

| Observation | Long reference | Short reference |
|---|---|---|
| Reference duration | Approximately 27.048 s | Approximately 9.816 s |
| Reference format | 24 kHz mono, converted to `pcm_f32le` | 24 kHz mono, converted to `pcm_f32le` |
| Voice identity | Very close to desired Xiaoxiao-style narrator | Same desired narrator identity |
| Long-passage output duration | Approximately 154.28 s | Approximately 160.08 s |
| Delivery | Rushed | Noticeably better pacing, reduced rushing |
| Prompt-length warnings | Many target chunks shorter than prompt reference | Most mismatch warnings disappeared |

**Prefer the approximately 9.8-second reference.** Narrator identity is satisfactory; changing the narrator or searching for another model is not further Milestone B work.

Current local reference files, as supplied by the user:

- `/home/jay/CosyVoice/reference_audio/xiaoxiao_narrator_short.wav`
- `/home/jay/CosyVoice/reference_audio/xiaoxiao_narrator_short.txt`

Exact transcript:

> 雨后的山路很安静，雾气沿着河面缓缓散开。林舟停下脚步，望向远处山谷里那盏微弱的灯。

The reference WAV remains local and must not be committed. Its checksum has not been independently captured during closeout. The runner records the selected WAV/transcript hashes on a future run; that does not retroactively establish provenance for earlier listening experiments. Float-format reference input is separate from the runner's PCM16 output requirement.

## Reproduction notes

Use the existing Ubuntu WSL environment and external installation. These are opt-in commands for future reproduction; the expensive benchmark was not rerun for closeout.

```bash
conda activate cosyvoice-b
cd "/mnt/c/Users/Jay Ma/OneDrive/Documents/TTS_Project"
python -B evaluation/run_cosyvoice_smoke.py --help

# Stock reference: one short smoke trial, then the original benchmark selection.
python -B evaluation/run_cosyvoice_smoke.py --case narration --repeat 1
python -B evaluation/run_cosyvoice_smoke.py --all --repeat 2

# Preferred narrator: a separate experiment, not the stock-reference benchmark.
python -B evaluation/run_cosyvoice_smoke.py --case narration --repeat 1 \
  --prompt-wav /home/jay/CosyVoice/reference_audio/xiaoxiao_narrator_short.wav \
  --prompt-text-file /home/jay/CosyVoice/reference_audio/xiaoxiao_narrator_short.txt
```

Alternatively, pass the matching transcript directly with `--prompt-text "..."`. `--prompt-text` and `--prompt-text-file` are mutually exclusive. A custom WAV requires an explicit transcript source to prevent accidental use of the stock transcript. Transcript files use UTF-8; a BOM and surrounding whitespace are removed. The runner adds `You are a helpful assistant.<|endofprompt|>` itself and accepts an already-prefixed transcript without duplicating that prefix. Its transcript hash covers the actual unprefixed text encoded as UTF-8; a separate file hash covers the original transcript-file bytes.

Defaults still point to `~/CosyVoice`, its existing model directory, and the stock reference. Configuration remains FP32, no TensorRT/vLLM, and non-streaming output. There is no speed override or application-level audio/text manipulation. Output directories are newly created under `outputs/evaluation/cosyvoice_baseline_...`; names do not identify the narrator, so use manifest prompt paths/hashes to distinguish experiments.

The closeout runner writes schema version 2 manifests with selected reference/transcript, hashes, Python/interpreter, model path/configuration hash, actual imported PyTorch/TorchAudio versions, CUDA availability/device, and observed upstream HEAD when readable. It records its own source hash but does not claim a verified checkpoint revision or complete installation lockfile. Failed trials retain tracebacks and processing continues; setup errors or keyboard interruption produce an aborted run with remaining trials marked `not_run`, a summary, and a nonzero exit code. An interrupted active trial is marked failed. Manifest updates use a temporary sibling file and replacement to reduce truncation risk.

Argument/corpus errors before a run is created, inability to create/write the output directory, or a forced process kill can still prevent a finalized manifest. A forced kill may leave `pending`/`running` records; this is not a resume system. A failed trial may leave a partial WAV at its expected filename, but `output_filename` remains null. Treat manifest status as authoritative.

WAV checks cover readable, nonempty mono PCM16 audio at the model's sample rate and a complete declared frame payload. They do not inspect narration quality. No new test suite was added; closeout validation is limited to lightweight model-free checks, separate from historical GPU synthesis and user listening.

## Decisions carried into Milestone C

**Milestone C — Audiobook Narration & Prosody Pipeline** is next. Its scope includes sentence-ending pauses, paragraph/scene pauses, pacing, semantic/narration-aware segmentation, long-form continuity, dialogue handling, repetition detection/prevention, and eventual pronunciation controls.

CosyVoice3 and the preferred short narrator reference are the starting foundation. These pipeline features are not implemented in Milestone B. Keep the original corpus and baseline evidence intact, record future experiments separately, and retain Melo's environment/backend/GUI behavior until deliberate integration work is authorized.
