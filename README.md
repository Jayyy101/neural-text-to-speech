# Neural Multilingual Text-to-Speech System

A local Mandarin audiobook app. CosyVoice3 RL is the validated production generator; MeloTTS remains a preserved historical baseline.

## Windows audiobook app

From PowerShell in the repository root, launch the native Windows Tkinter UI:

```powershell
python -B -m src.audiobook_ui
```

Enter a WAV filename (Unicode names such as `神通者04.wav` are supported), paste one chapter, and click **Generate Audiobook**. The UI shows unit progress and elapsed time, then displays the validated finished WAV path. **Open Folder** opens the final output directory in Windows Explorer. An omitted `.wav` extension is added automatically; an existing exported filename is never silently overwritten.

Generation runs through `wsl.exe` in the isolated WSL `cosyvoice-b` environment, using the frozen `python -B -m src.audiobook run` production CLI. The source, manifest, and audio remain on the Windows filesystem. Each UI request has a UTF-8 source snapshot and worker log under `outputs/ui_requests/<job-id>/`. The canonical resumable output is `outputs/audiobooks/<chapter-id>/<run-id>/final/chapter.wav`; after validation, the requested filename is exported as a separate copy in that same `final` folder. The UI does not edit the chapter text or implement resume, cancellation, or manual repair controls.

The read-only existing-run inspector is retained for maintenance: pass a run directory to `python -B -m src.audiobook_ui <run-directory>`. It is not part of the normal generation screen.

For project status and acceptance evidence, see [the project handoff](docs/PROJECT_STATE.md). Historical model evaluations are in [Milestone B](evaluation/COSYVOICE_MILESTONE_B.md) and [Milestone C](evaluation/COSYVOICE_MILESTONE_C.md).

---

## Production backend (developer reference)

The default `run` command provides a manifest-driven chapter workflow:

```text
UTF-8 chapter text
  -> deterministic scene and certified unit plan
  -> CosyVoice3 RL unit attempts with bounded content-QC retries
  -> exact PCM chapter assembly
```

Scene boundaries are explicit standalone `***` lines. Text without markers is one scene; the pinned native CosyVoice frontend then freezes its synthesis units. The historical scene path remains available through `run --legacy-scenes` and existing run manifests.

### Milestone responsibilities

| Milestone | Responsibility |
|---|---|
| D1 | Validate and snapshot source text; create deterministic scene IDs, spans, hashes, and the initial manifest. |
| D2 | Generate and validate one WAV attempt per scene through a single initialized CosyVoice3 adapter. |
| D3 | Resume incomplete runs and regenerate one requested scene while preserving attempt history. |
| D4 | Apply source-bound manual pause plans and assemble selected PCM WAVs in planned order. |
| D5 | Run planning, generation, validation, and assembly through one end-to-end command; support reproducible seeded regeneration. |

### Environment

Planning, manifest inspection, repair, assembly, CLI help, and tests do not load CosyVoice. Real generation uses the isolated WSL2 `cosyvoice-b` environment described in [Milestone B](evaluation/COSYVOICE_MILESTONE_B.md):

- CosyVoice checkout: `~/CosyVoice`
- model: local `~/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/llm.rl.pt`, exposed through a verified view under `outputs/model_views/`
- preferred prompt WAV and transcript: `~/CosyVoice/reference_audio/xiaoxiao_narrator_short.{wav,txt}`
- CUDA-capable PyTorch in the `cosyvoice-b` environment

The paths above are defaults and can be overridden with `--cosyvoice-root`, `--model-dir`, `--prompt-wav`, and `--prompt-text-file` on commands that generate audio.

### Commands

Run commands from the repository root. The default output root is `outputs/audiobooks`.

```bash
# Plan only. The target run directory must not already exist.
python -B -m src.audiobook plan chapter.txt \
  --chapter-id chapter_0001 --run-id run_001

# Historical scene workflow for an existing D1 plan.
python -B -m src.audiobook generate \
  outputs/audiobooks/chapter_0001/run_001

# Default: plan, certify units, generate with RL and bounded QC, and assemble.
python -B -m src.audiobook run chapter.txt \
  --chapter-id chapter_0001 --run-id run_002

# Resume any incomplete run under its recorded policy.
python -B -m src.audiobook resume \
  outputs/audiobooks/chapter_0001/run_001

# Historical scene-only targeted regeneration.
python -B -m src.audiobook regenerate \
  outputs/audiobooks/chapter_0001/run_001 --scene-id scene_0002

# Request a reproducible alternate take. Seed range: 0 through 2**32 - 1.
python -B -m src.audiobook regenerate \
  outputs/audiobooks/chapter_0001/run_001 \
  --scene-id scene_0002 --seed 1

# Apply a manually authored plan bound to the selected attempt ID and WAV hash.
python -B -m src.audiobook repair \
  outputs/audiobooks/chapter_0001/run_001 \
  --scene-id scene_0002 --plan pause-plan.json

# Rebuild final/chapter.wav from current selected artifacts.
python -B -m src.audiobook assemble \
  outputs/audiobooks/chapter_0001/run_001
```

Default `run` retries only validated contiguous Han omissions within the recorded three-attempt cap. If a unit remains unresolved, the run stays resumable and assembly is blocked. Historical scene generation is available with `run --legacy-scenes`; `regenerate` and `repair` remain scene-only operations.

### Unit-planned chapter path (schema 5)

This is the default `run` workflow and is also available as separate commands. It freezes native normalized text, certifies exact source spans, and records explicit punctuation after a detected unpunctuated chapter heading. The narrow native terminal `、` to `。` mapping equivalence is versioned and auditable; it never rewrites the frozen synthesis unit. No breath, silence, trimming, fade, or crossfade is inserted automatically. Explicit native control tokens such as `[breath]` pass through the unit synthesis adapter when supplied in authorized synthesis text.

```bash
python -B -m src.audiobook plan chapter.txt \
  --chapter-id chapter_units --run-id run_001
python -B -m src.audiobook prepare-units \
  outputs/audiobooks/chapter_units/run_001
python -B -m src.audiobook generate \
  outputs/audiobooks/chapter_units/run_001 --root-seed 12345
python -B -m src.audiobook resume \
  outputs/audiobooks/chapter_units/run_001
python -B -m src.audiobook assemble \
  outputs/audiobooks/chapter_units/run_001
```

`--root-seed` is optional on first `generate`; an omitted value is generated and persisted before synthesis. Schema 5 derives an explicit 32-bit seed from the root seed, immutable unit-plan hash, unit ID, take index, and versioned `sha256_root_plan_unit_take_v1` policy. Interrupted physical attempts of the same logical take reuse that seed, and skipping selected units cannot shift later seeds. This changes RNG semantics from historical scene generation and clean12's advancing global random stream.

One warm model synthesizes each frozen normalized unit with `text_frontend=False`. Each attempt has its own WAV, seed, hash, metadata, timing, and append-only history; scenes report aggregate completion. Resume validates and skips selected units, recovers valid unpublished WAVs, and preserves interrupted artifacts. Missing or corrupt selected WAVs stop recovery. Assembly concatenates selected unit PCM in scene and unit order with 0 ms injected silence, records frame offsets and output hash, and becomes stale after a selection change. Manual scene pause repairs and scene regeneration are unsupported for schema 5. Schemas 2–4 retain their historical scene behavior without migration.

New schema-5 runs require the fixed Mandarin content gate before a generated attempt can be selected. A persistent audio-only ASR worker runs in the isolated `tts-align` interpreter while CosyVoice stays in `cosyvoice-b`. The worker receives only the WAV path/hash and returns independent greedy CTC recognition; the parent then compares frozen intended Han text with the recognized sequence. A contiguous expected Han deletion of **4 or more** rejects the attempt. Atomic `content_qc.json` sidecars bind passed/rejected evidence to each attempt. Step 3 real acceptance passed on the two-unit fixture: both units passed, one worker served both, exact PCM assembly was preserved, and resume made no new TTS or ASR requests.

For **new** QC-enabled runs, the persisted `bounded_content_qc_retries_v1` policy permits at most three total physical synthesis attempts per unit. A validated content rejection advances the logical take index and derives a new seed from the existing root/plan/unit/take policy; synthesis interruption can create another physical attempt with the same logical take and seed while the physical-attempt budget remains. Passed QC selects and stops. QC infrastructure errors, pending/running QC, and interrupted QC retry recognition on the existing WAV; a corrupt artifact or evidence fails closed. Three valid content rejections persist an exhausted state, leave the unit unresolved, and block assembly. Substitutions, insertions, CER, and deletions of 1–3 expected Han characters do not trigger synthesis retries. Earlier QC-enabled Step 3 runs without a retry-policy record retain their recorded no-retry behavior. Existing Step 2 schema-5 runs without QC retain their historical selection behavior. The validated whole-WAV ASR path remains limited to 30 seconds per unit. The Chapter 1 full-chapter acceptance run recorded a confirmed omission in unit 66 and a successful second deterministic attempt.

### Run artifacts and state

```text
outputs/audiobooks/<chapter_id>/<run_id>/
|-- source.txt
|-- manifest.json
|-- scenes/
|   `-- scene_0001/
|       |-- attempt_001/generated.wav
|       |-- attempt_002/generated.wav
|       `-- repairs/repair_001/
|           |-- source-plan.json
|           |-- applied-plan.json
|           `-- repaired.wav
`-- final/chapter.wav
```

The source snapshot is byte-exact. The manifest records source and plan hashes, source spans, backend and narrator provenance, every generation attempt, attempt-level random-state policy, selected attempts and repairs, WAV identities, and assembly frame offsets. Paths inside a run are relative where practical.

Recovery preserves history. A successful regeneration selects its new valid attempt; a failure retains the previous valid selection. A seeded result identical to the prior selection is recorded as a duplicate and is not selected. Changing a selected attempt deselects any repair tied to the old attempt and marks an existing assembly stale. Old manifests without random-state metadata remain supported.

Manual period repairs use the validated quiet-valley insertion workflow. A `period` entry without `add_ms` defaults to **+140 ms**. Plans must identify the selected `scene_id`, `attempt_id`, and source WAV SHA-256, so stale plans are rejected. Original generated WAVs remain unchanged.

Assembly selects a valid current repair when present and otherwise uses the selected generated attempt. It concatenates compatible PCM payloads exactly in planned order, preserves each clip's natural leading and trailing silence, and adds **0 ms extra silence**. It does not trim, fade, crossfade, resample, or normalize.

### Production policy and limitations

- CosyVoice3 is the selected Mandarin backend; Azure Xiaoxiao remains a listening-quality reference.
- Prefer coherent, continuous scene generation.
- For audible artifacts, use: listen/detect -> explicit seeded regeneration -> reassemble.
- A structurally valid WAV can still contain a perceptual generation artifact. Assembly does not create or remove artifacts already inside a selected scene.
- Automatic punctuation alignment, forced alignment, pause inference, perceptual artifact detection, random retry loops, mastering, MP3 export, and document ingestion remain deferred.
- The current backend expects prepared UTF-8 chapter text with deliberate scene markers.

### Local interface boundary

The native Windows Tkinter UI reads persisted run manifests through `src/audiobook_application.py` and starts the existing CLI through `src/audiobook_launcher.py`. The CLI retains all planning, synthesis, QC/retry, resume state, and assembly decisions. The UI validates the canonical final WAV before copying it to the user-facing filename. The historical E1 inspector remains available only when a run directory is explicitly supplied.

### Tests

Run the complete model-free suite in the existing WSL `tts-align` environment:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd '/mnt/c/Users/Jay Ma/OneDrive/Documents/TTS_Project' && /home/jay/miniconda3/envs/tts-align/bin/python -B -m unittest discover -s tests -v"
```

CLI help is model-free:

```bash
python -B -m src.audiobook --help
python -B -m src.audiobook regenerate --help
```

---

## Project Overview

The goal of this project is to build an end-to-end multilingual TTS application that can:

- accept English and Chinese text input
- generate natural-sounding speech using a neural TTS model
- run locally without depending on a commercial TTS API
- save generated speech as timestamped WAV files
- provide a simple GUI for user interaction

Azure Neural TTS was used only as a quality benchmark. XTTS was tested as a multilingual experiment. The earlier English VITS version was used as the baseline prototype.

---

## Historical MeloTTS application

The existing application uses:

- **MeloTTS** as the main speech synthesis backend
- **Tkinter** for the graphical user interface
- **PyTorch with CUDA** for GPU-accelerated inference
- **Timestamped WAV outputs** saved to the `outputs/` folder

The GUI allows users to:

- choose a language: English or Chinese
- choose a speaker
- adjust speech speed
- type or clear input text
- generate speech
- open the generated audio file
- view status updates while speech is being generated

---

## Project Files

```text
src/
├── generate.py          # older English VITS baseline prototype
├── xtts.py              # XTTS multilingual experiment
├── generate_azure.py    # Azure Neural TTS benchmark
├── generate_melo.py     # preserved MeloTTS backend
└── gui.py               # existing Tkinter GUI (MeloTTS)
```

---

## Features

- Local neural TTS generation
- English text-to-speech support
- Mandarin Chinese text-to-speech support
- Mixed Chinese-English text support
- Speaker selection
- Speed control
- GUI-based input and playback
- Timestamped WAV output files
- Model caching for faster repeated generation
- Progress indicator during synthesis

---

## System

Tested on:

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER
- CPU: Ryzen 7 7800X3D
- RAM: 32GB
- Python: 3.10
- OS: Windows 10

---

## Setup

Create and activate the conda environment:

```bash
conda create -n melo python=3.10
conda activate melo
```

Install PyTorch with CUDA support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Install MeloTTS:

```bash
pip install melo-tts
```

Depending on the system, additional packages may be needed for Chinese text processing and audio generation.

Make sure commands are run from the main project folder.

For example, if cloned from GitHub:

```bash
cd neural-text-to-speech
```

---

## Run the historical MeloTTS GUI

From the main project folder, run:

```bash
python src/gui.py
```

The GUI allows users to enter text, choose a language, select a speaker, adjust speed, generate speech, and open the most recent audio output.

Generated audio files are saved in:

```text
outputs/
```

The `outputs/` folder is ignored by Git so generated WAV files are not uploaded to the repository.

---

## Evaluation Summary

The system was tested with English, Chinese, and mixed Chinese-English inputs. Testing focused on pronunciation quality, pacing, multilingual support, and inference time.

| Test | Language | Speaker | Speed | Inference Time | Notes |
|---|---|---|---:|---:|---|
| English short sentence | EN | EN-Default | 1.0 | 2.538s | Good pronunciation, but slightly fast and struggled with “MeloTTS.” |
| English paragraph | EN | EN-Default | 1.0 | 0.266s | Clear pronunciation and better pauses. |
| Chinese sentence | ZH | ZH | 1.0 | 2.640s | Good pronunciation and pauses, but slightly fast. |
| Mixed Chinese-English | ZH | ZH | 1.0 | 0.476s | Chinese sounded strong; English words had a noticeable Chinese accent. |
| English paragraph slower | EN | EN-Default | 0.9 | 0.334s | Pacing sounded better than 1.0. |
| Chinese sentence slower | ZH | ZH | 0.9 | 0.190s | Still slightly fast; 0.8 sounded better. |

Repeated generation became faster because the backend caches loaded models during the same GUI session.

---

## Current Limitations

- Mixed Chinese-English input works best with the Chinese model, but English words may sound accented.
- Some technical terms, such as “MeloTTS,” may need input formatting to improve pronunciation.
- Speech speed may need adjustment depending on the language and input length.
- The system currently runs best on the configured Windows GPU environment.
- The GUI is local only and has not yet been deployed as a web application.

---

## Future Improvements

- Add a web frontend using Flask or FastAPI so the system can be accessed from other devices.
- Deploy the backend on a local or cloud GPU server for remote use.
- Improve pronunciation handling for technical words and mixed-language input.
- Add more detailed evaluation metrics for pronunciation quality, speed, and user feedback.
- Compare MeloTTS more directly against Azure Neural TTS, XTTS, and the earlier VITS baseline.
- Package the app as a desktop application for easier use.

---

## Author

Jay Ma  
GitHub: [Jayyy101](https://github.com/Jayyy101)
