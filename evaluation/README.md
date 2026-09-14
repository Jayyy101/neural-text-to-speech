# MeloTTS baseline capture — Stage A

This directory preserves the existing MeloTTS system before audiobook model experiments. It adds diagnostic inputs, metadata, a separate smoke runner, and standard-library regression tests. It does not change `src/generate_melo.py`, `src/gui.py`, their defaults, or the installed environment. No other TTS model is required.

## Milestone A status: complete

The real Windows GPU baseline run and the user's manual listening review are complete. The final run, [melo_baseline_2026-09-13_02-30-28/manifest.json](../outputs/evaluation/melo_baseline_2026-09-13_02-30-28/manifest.json), records **12 successful synthesis trials, 2 expected mixed-text failures, and 0 incomplete trials** across seven cases with two repetitions each. The linked manifest and audio are local, Git-ignored artifacts.

The run status remains `completed_with_failures`: both `mixed_text` repetitions failed as expected, while all six other cases completed both repetitions. Milestone completion means the baseline and its limitations have been captured and reviewed; it does not reclassify the failures as passes or establish final audiobook readiness.

### Final baseline findings

Listening judgments below come from the user's completed manual review; durations, timings, and trial counts come from the saved run metadata.

| Area | Final observation |
|---|---|
| Narrator identity | Highly consistent across repeated generations. |
| Mandarin speech | Intelligible and stable. |
| Long-form stability | The approximately 2.4-minute diagnostic passage completed successfully without obvious narrator collapse. The two outputs lasted 145.168 and 144.750 seconds. This establishes stability on this passage, not across an entire novel. |
| Throughput | Very fast after model warm-up. The long-passage trials reported 4.678 and 4.484 seconds around `tts_to_file`, respectively; these timings exclude model loading and imports. |
| Naturalness | Acceptable as a technical baseline, but below the desired audiobook-quality target. |
| Dialogue | Remains relatively close to ordinary narration, with limited expressive differentiation. |
| Mixed Chinese/English | Confirmed preprocessing failure on underscore characters, including `_` in `chapter_01.txt`. Preserve the failing corpus case and tracebacks as baseline evidence. |
| Percent formatting | `100%` normalizes to `一百`, losing percentage meaning even though preprocessing succeeds. |
| Numbers and special formatting | Require future normalization work; successful synthesis alone does not establish semantically correct readings. |

### Baseline role and subsequent milestones

MeloTTS remains the original reproducible baseline for future comparisons and must not be assumed to be the final audiobook backend. The current environment, inputs, and synthesis behavior remain preserved; fresh-environment restoration is still unverified as noted below.

Azure `zh-CN-XiaoxiaoNeural` remains the listening-quality reference for naturalness, pacing, pronunciation, and audiobook suitability. [Milestone B — CosyVoice Bring-Up](COSYVOICE_MILESTONE_B.md) is complete in a separate WSL environment, with Melo preserved. CosyVoice3 is the selected experimental foundation; Milestone C — Audiobook Narration & Prosody Pipeline is next. The Milestone A findings and historical reproduction procedure below remain unchanged.

## Evidence and limits

The root README contains historical listening notes and timings. During the initial capture, Python/PyTorch import, CUDA availability, GPU identity, installed package metadata, and cached English/Chinese checkpoint hashes were checked. The subsequent full GPU run and manual listening review are now complete, with final results recorded above. The mixed-text preprocessing failure remains diagnosed below. Passing mocked tests alone does not establish speech quality, and a fresh installation has not been validated.

The machine-readable capture timestamp, baseline Git commit, and source hashes are in [melo-environment.json](baseline/melo-environment.json). Windows reported version 10.0.19045. The observed runtime is:

| Component | Recorded configuration |
|---|---|
| OS | Windows 10, build 19045 |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER; 16,376 MiB reported VRAM |
| NVIDIA driver | 591.86 |
| CPU / RAM | Ryzen 7 7800X3D / 32 GB, from the existing README; not independently remeasured because CIM access was denied |
| Environment | Conda `melo`, Python 3.10.20 |
| Interpreter | `C:\miniconda3\envs\melo\python.exe` |
| MeloTTS distribution | `melotts` 0.1.2; Git revision `209145371cff8fc3bd60d7be902ea69cbdb7965a` |
| PyTorch / TorchAudio | 2.11.0+cu126 / 2.11.0+cu126 |
| TorchVision | 0.26.0+cu126 |
| PyTorch CUDA build / cuDNN | 12.6 / 91002 (reported integer) |
| Transformers / NumPy | 4.27.4 / 1.26.4 |
| SoundFile / librosa | 0.13.1 / 0.9.1 |
| pypinyin / jieba | 0.50.0 / 0.42.1 |

CUDA availability was true during initial capture; the completed GPU synthesis run now provides workload evidence as well. The shell's base Python is different from the `melo` interpreter; activate the environment explicitly. PyTorch's bundled CUDA version does not establish an installed standalone CUDA toolkit version.

## Environment and asset preservation

[melo-pip-freeze.txt](baseline/melo-pip-freeze.txt) is the unedited output of `python -B -m pip --disable-pip-version-check freeze --all` from `melo`, including 144 entries. It is an inventory, **not a tested installation lockfile**:

- The `pip @ file:///home/...` entry is a Conda build-machine path, not a portable installation source. The actual pip version is recorded in the JSON.
- CUDA wheels were installed from a CUDA-specific source; the freeze does not encode that index. The historical root README uses `https://download.pytorch.org/whl/cu126`.
- Python, Conda components, drivers, language dictionaries, and downloaded model assets are not recreated by pip freeze.
- The root README/requirements say `melo-tts`; the actual installed distribution is `melotts`, installed from the Git revision above. Stage A leaves those historical files unchanged. Do not use that spelling discrepancy as a reason to reinstall the current environment.

The JSON records cached snapshot revisions and SHA-256 hashes for the Melo English/Chinese configuration and checkpoint files, plus installed Melo API/loader source hashes. This identifies the cache currently present; it does not prove which revision generated old WAV files. Auxiliary BERT/tokenizer/NLTK resources are not fully inventoried. No weight files have been copied into Git, and historical WAVs without known inputs are not treated as reproducible fixtures.

Keep the existing environment and caches in place. Do not run package upgrades or `pip install -r` against this snapshot. A future restoration exercise should use a separate environment and verify these records before claiming reproducibility.

## Evaluation layout

```text
evaluation/
  README.md
  baseline/
    melo-environment.json
    melo-pip-freeze.txt
  inputs/
    mandarin_diagnostics.json
  run_melo_smoke.py
tests/
  test_melo_baseline.py
```

Generated artifacts go under `outputs/evaluation/`, which is already ignored by Git. Future model runners can share the versioned input schema and use their own output directories. No new model adapters, application-level chunking, pronunciation substitutions, audio stitching, or resume engine are part of Stage A.

## Diagnostic corpus

[mandarin_diagnostics.json](inputs/mandarin_diagnostics.json) contains seven original passages: narration, dialogue, punctuation, names/uncommon vocabulary, numbers, mixed Chinese/English, and a moderately long multi-paragraph passage. Each has a stable ID and listening checklist. Names have explicit intended readings for this fictional context; notes are not injected into synthesis.

All cases use `language=ZH`, `speaker_name=ZH`, `speed=1.0`. The long case is passed intact to the existing backend; sentence splitting remains MeloTTS's responsibility. Preserve text, punctuation, and paragraph breaks exactly. This is a diagnostic suite, not an audiobook-scale quality benchmark.

### Known mixed-text failure: underscore

The user-run manifest `outputs/evaluation/melo_20260913T090631Z_6fc759a9/manifest.json` recorded ten successful trials, followed by `mixed_text` repetition 1 failing in `chinese.py:122`. The old runner stopped there, so repetition 2 and the long passage were not attempted. That original manifest and corpus are preserved.

Isolated probes of the installed `melo.text.chinese_mix.text_normalize()` followed by `g2p()` reproduced the failure without waveform generation. Network connections were blocked for the probe process; no packages or model weights were installed.

| Isolated input | Preprocessing result |
|---|---|
| `_` | `AssertionError`; minimal one-character trigger |
| `chapter_01.txt`, `文_件` | Same assertion |
| `.`, `01`, `chapter`, `txt` | Pass |
| `chapter01.txt`, `chapter 01.txt` | Pass |
| `backup`, `Python`, `GPU ready`, `MeloTTS`, `USB` | Pass |
| `100%` | Pass, but normalizes to `一百`, losing percent meaning |
| Original full mixed passage | Same assertion |
| Full passage with only `_` removed in memory | Pass |

The mixed-language normalizer explicitly retains underscores. Its English/Chinese splitter routes `_` into the Chinese phoneme converter. There, the initial and final values are both `_`, but underscore is absent from the supported punctuation set, so `assert c in punctuation` fails. This is a preprocessing limitation, not evidence that English words in general are unsupported. The probes also showed `01` normalizing to `一`, losing the leading zero.

Full probe inputs, normalized strings, tracebacks, and failing-frame values are saved locally in `outputs/evaluation/preprocessing_diagnosis.json`. The pass results above refer only to preprocessing, not pronunciation quality or full synthesis. Removing the underscore was a diagnostic comparison only; the corpus remains byte-for-byte unchanged. The filename is realistic audiobook input, so there is no test mistake to correct.

For a minimal manual reproduction in the existing `melo` environment (cached tokenizer assets required):

```powershell
python -B -c "import os; os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; from melo.text import chinese_mix; text=chinese_mix.text_normalize('_'); print(repr(text)); chinese_mix.g2p(text)"
```

This command is expected to raise the assertion. Do not sanitize the baseline input or patch the installed package to hide it.

## Automated regression checks (no models or downloads)

From the repository root, run with base Python or the existing `melo` Python:

```powershell
python -B -m unittest discover -s tests -v
python -B evaluation/run_melo_smoke.py --help
```

Tests load the actual backend source with fake `melo.api` and `torch` modules. They check language aliases/fallback, valid speed bounds, lazy loading, model reuse, CUDA/CPU selection, exact text forwarding, defaults, result fields, speaker errors, and error propagation. Model-free CLI workflow tests also verify continuation after failures (including empty-message assertions), exit codes, per-trial manifests, descriptive filenames, timestamp collisions, setup failures, and optional folder opening. Temporary test files are removed automatically. No GUI is imported and no third-party test package is needed.

These tests characterize current behavior. Known issues such as NaN speed acceptance, second-resolution filenames, GUI threading, and import side effects remain unchanged. There are no golden waveform comparisons: the backend does not set a random seed, so identical inputs need not produce identical bytes.

## Manual real-model GPU smoke test

This is a separate opt-in procedure on the Windows GPU machine. Close other synthesis jobs first. Keep the current environment unchanged.

```powershell
conda activate melo
python -B -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
python -B evaluation/run_melo_smoke.py --case narration --repeat 2
```

The first command should show the `melo` interpreter, the recorded PyTorch version, and `True`. The runner deliberately requires CUDA; the actual backend's CPU fallback remains untouched. If activation is unavailable, use the recorded interpreter's full path with PowerShell's call operator (`&`).

After the short test passes and is listened to, run:

```powershell
python -B evaluation/run_melo_smoke.py --all --repeat 2
```

To open the finished run folder in Windows Explorer, opt in explicitly:

```powershell
python -B evaluation/run_melo_smoke.py --all --repeat 2 --open-output
```

Opening is off by default. It is attempted after metadata and the summary are saved, including runs with failed trials. An Explorer launch failure is recorded as a warning and does not change the synthesis result or exit code. On other operating systems the flag reports that opening is unsupported.

The runner sets Hugging Face and Transformers offline flags before importing MeloTTS, to use already cached assets. These flags do not provide an operating-system-wide network block for every dependency. If an auxiliary resource is missing or an import/checkpoint load fails, retain the failure manifest and investigate separately; do not upgrade packages or add compatibility overrides as part of this milestone.

Each invocation creates `outputs/evaluation/melo_baseline_YYYY-MM-DD_HH-MM-SS/`, using the local clock. Same-second collisions receive readable suffixes `_02`, `_03`, etc.; existing runs are never reused. UTC start/finish times are recorded in the manifest.

Successful WAVs are moved, without re-encoding, to the top level of the new run folder:

```text
melo_baseline_2026-09-13_15-04-05/
  manifest.json
  narration_01.wav
  narration_02.wav
  dialogue_01.wav
  ...
  long_passage_02.wav
  _trials/                 # Isolated working directories; may hold partial failures
```

The stable corpus ID `mixed_text` maps to filenames `mixed_language_01.wav` and `mixed_language_02.wav` if generation succeeds. Failed trials have no successful WAV at the top level; their expected filenames and errors remain in the manifest. Upload the top-level WAVs with `manifest.json`; keep `_trials/` if investigating partial outputs. Existing run folders are not renamed or migrated.

Each case/repeat still gets its own working directory under `_trials/`, preventing collisions from the backend's second-resolution names. The original synthesis function and result dictionary are unchanged; `backend_result.output_path` documents the original location before the evaluator moves the new audio. Use the record's `output_filename`, relative to the run folder, to locate the final WAV. The model cache remains shared within this standalone runner process. Its temporary working-directory changes are not designed for shared threaded use.

Manifest schema version 2 prelists every requested trial and records exact source text, settings, expected/output filenames, state, UTC timestamps, and elapsed trial time, including failures. Successful trials additionally retain the backend's timing/result and WAV format, duration, and SHA-256 hash. Failed trials retain exception type, message (which can be empty), and full traceback. Corpus/backend/baseline hashes and runtime details remain at run level.

A synthesis or WAV-validation failure marks that trial failed, saves the error, and continues with the next repetition/case. No automatic retry or input correction is performed. A runtime setup failure, such as missing CUDA, aborts the run and marks unattempted trials `not_run` with a top-level traceback. Progress is saved after each trial. A force-killed process may leave `pending`/`running` records; this is not a resume system.

At completion, the console and manifest summarize passed/failed/not-completed trial counts and case lists. A case passes only if all its repetitions pass. Exit code **0** means all requested trials passed WAV checks; **1** means a trial failed or the run aborted; argument errors use **2**. `completed_with_failures` is an expected result while the underscore issue remains: the long passage should still be attempted. A failed trial has `output_filename: null`, with its intended name in `expected_output_filename`.

The first synthesis call may include model loading; later calls reuse the model. The backend's `inference_time` excludes loading and imports. Neither timing should be presented as a controlled benchmark. The stored baseline-record hash references the capture; it does not assert that current dependencies or model caches still match it. No seeds, decoding parameters, gain changes, or playback changes are applied.

### Manual acceptance checklist

- Both narration repetitions finish and produce nonempty, playable WAVs.
- All fourteen trials are attempted by `--all --repeat 2`, even if both mixed-text trials fail. Inspect the summary and tracebacks; do not count expected failures as passes.
- Listen to successful cases for complete endings; check that the last long-passage sentence is present if that case succeeds.
- Listen for dropped/repeated words, incorrect names/tones, punctuation behavior, and abrupt changes in voice or pacing using each case's checklist.
- Compare repeats for narrator consistency; do not require byte-identical audio.
- Run the short test again in a fresh process and compare its narrator with the first run.
- Confirm the existing GUI still works with `python -B src/gui.py`: use ZH, speaker ZH, speed 1.0, generate, and Open Output. Also check its existing English path if it is part of your routine baseline.

For future reruns, add a dated listening note inside the ignored run directory with case IDs, observations, errors, and overall pass/fail. A `passed_wav_checks_listening_pending` manifest means only that generation returned readable, nonempty WAVs; it does not certify correct narration. Milestone A's completed manual review is recorded above, with the two mixed-text failures retained as known baseline limitations.
