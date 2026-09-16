# Neural Multilingual Text-to-Speech System

A locally controlled neural Text-to-Speech (TTS) project. MeloTTS remains the preserved multilingual baseline and legacy GUI backend. CosyVoice3 is the selected Mandarin audiobook backend and is available through the model-free orchestration CLI in `src/audiobook`.

This project started with an English VITS prototype, then expanded through XTTS and Azure Neural TTS testing. MeloTTS remains the preserved baseline and current legacy application backend. CosyVoice3 completed isolated WSL evaluation, audiobook-prosody experiments, and the production chapter backend; it is not integrated into the GUI. See [Milestone B — CosyVoice Bring-Up](evaluation/COSYVOICE_MILESTONE_B.md), [Milestone C — Audiobook Narration & Prosody Pipeline](evaluation/COSYVOICE_MILESTONE_C.md), and the [project handoff](docs/PROJECT_STATE.md).

---

## Mandarin Audiobook Backend

Milestone D provides a manifest-driven chapter workflow:

```text
UTF-8 chapter text
  -> deterministic scene plan
  -> CosyVoice3 scene attempts
  -> optional targeted regeneration or manual pause repair
  -> exact PCM chapter assembly
```

Scene boundaries are explicit standalone `***` lines. Text without markers is one continuous scene; the planner does not automatically segment or pack paragraphs. Continuous scene generation is preferred because sentence-by-sentence synthesis can reset narration prosody.

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
- model: `~/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B`
- preferred prompt WAV and transcript: `~/CosyVoice/reference_audio/xiaoxiao_narrator_short.{wav,txt}`
- CUDA-capable PyTorch in the `cosyvoice-b` environment

The paths above are defaults and can be overridden with `--cosyvoice-root`, `--model-dir`, `--prompt-wav`, and `--prompt-text-file` on commands that generate audio.

### Commands

Run commands from the repository root. The default output root is `outputs/audiobooks`.

```bash
# Plan only. The target run directory must not already exist.
python -B -m src.audiobook plan chapter.txt \
  --chapter-id chapter_0001 --run-id run_001

# Generate all scenes in an existing planned run.
python -B -m src.audiobook generate \
  outputs/audiobooks/chapter_0001/run_001

# Plan, generate all scenes with one model initialization, and assemble.
python -B -m src.audiobook run chapter.txt \
  --chapter-id chapter_0001 --run-id run_002

# Generate one new attempt for every scene without a valid selection.
python -B -m src.audiobook resume \
  outputs/audiobooks/chapter_0001/run_001

# Generate one new attempt for one scene.
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

`run` does not retry, regenerate, or repair automatically. If a required scene fails, it records the failure and stops before assembly. The independent `resume`, `regenerate`, `repair`, and `assemble` commands provide explicit recovery.

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

The future UI/product milestone may present planning, progress, listening, attempt selection, repair-plan authoring, and assembly controls by calling this backend and reading its manifest. It should preserve the backend's explicit recovery and provenance rules.

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

## Current Legacy Application

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

## Run the GUI

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
