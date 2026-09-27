# Neural Multilingual TTS: Mandarin audiobook app

The accepted product is a local Mandarin audiobook app: a native Windows UI backed by the frozen CosyVoice3 RL chapter generator in WSL. The MeloTTS, VITS, XTTS, and Azure files document earlier baselines and experiments.

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

### Historical scene-workflow milestones

Milestones D1 through D5 established the earlier scene workflow. The current default `run` path uses schema-5 certified units, described below; the scene workflow remains available for historical runs.

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

### Run artifacts and state (historical scene example)

The layout below illustrates a scene-workflow run. Current schema-5 runs also store certified units, per-unit attempts, and content-QC sidecars under `units/`; their canonical final file is still `final/chapter.wav`.

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

The source snapshot is byte-exact. The manifest records source and plan hashes, source spans, backend and narrator provenance, generation attempts, selected artifacts, WAV identities, and assembly frame offsets. Historical scene runs also record scene repairs and their selections. Paths inside a run are relative where practical.

In the historical scene workflow, recovery preserves history. A successful regeneration selects its new valid attempt; a failure retains the previous valid selection. A seeded result identical to the prior selection is recorded as a duplicate and is not selected. Changing a selected attempt deselects any repair tied to the old attempt and marks completed assembly stale. Old manifests without random-state metadata remain supported.

Historical scene-only manual period repairs use the validated quiet-valley insertion workflow. A `period` entry without `add_ms` defaults to **+140 ms**. Plans must identify the selected `scene_id`, `attempt_id`, and source WAV SHA-256, so stale plans are rejected. Original generated WAVs remain unchanged.

Historical scene assembly selects a valid current repair when present and otherwise uses the selected generated attempt. It concatenates compatible PCM payloads exactly in planned order, preserves each clip's natural leading and trailing silence, and adds **0 ms extra silence**. It does not trim, fade, crossfade, resample, or normalize.

### Production policy and limitations

- CosyVoice3 is the selected Mandarin backend; Azure Xiaoxiao remains a listening-quality reference.
- The default path freezes native frontend units from coherent chapter text and assembles selected unit PCM exactly.
- A structurally valid WAV can still contain a perceptual generation artifact. Listening remains part of acceptance; the exact assembly does not alter samples within selected units.
- Explicit seeded scene regeneration is available only for historical scene runs, not the default schema-5 path.
- Automatic punctuation alignment, forced alignment, pause inference, perceptual artifact detection, random retry loops, mastering, MP3 export, and document ingestion remain deferred.
- The backend accepts UTF-8 chapter text; standalone `***` markers are optional intentional scene boundaries.

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

## Historical baselines and experiments

These files remain for reproducibility and are outside the accepted CosyVoice audiobook app:

| Files | Historical role |
|---|---|
| `src/generate_melo.py`, `src/gui.py` | Preserved Windows MeloTTS backend and English/Chinese Tkinter GUI. Launch the legacy GUI with `python -B src/gui.py` in its existing `melo` environment. |
| `src/generate.py` | Earlier English VITS prototype. |
| `src/xtts.py` | XTTS multilingual experiment. |
| `src/generate_azure.py` | Azure Neural TTS quality benchmark; Xiaoxiao remains a listening reference. |

The root `requirements.txt` describes the historical Melo GUI, not the WSL CosyVoice production environment. It is not a validated lockfile for recreating the captured `melo` environment. The Melo baseline, including measured results and known failures, is documented in [evaluation/README.md](evaluation/README.md); its environment capture is under `evaluation/baseline/`. Generated audio and run evidence under `outputs/` are local and Git-ignored.

Future product work includes MP3 export from a validated final WAV. Repository cleanup is a separate ongoing milestone; preserve baseline evidence and unrelated evaluation work until reviewed.

---

## Author

Jay Ma

GitHub: [Jayyy101](https://github.com/Jayyy101)
