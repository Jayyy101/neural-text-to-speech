# Milestone C — Audiobook Narration & Prosody Pipeline

## Status and scope

Milestone C is complete. It established production-oriented narration and pause policies for CosyVoice3, validated deterministic manual pause repair, and recorded two unsuccessful automatic punctuation-alignment approaches for possible future research. It did not integrate CosyVoice into the legacy GUI or build a whole-novel job runner.

CosyVoice3 remains the selected foundation. All synthesis experiments used the existing isolated WSL environment, the `Fun-CosyVoice3-0.5B` model, and the preferred approximately 9.8-second reference at `/home/jay/CosyVoice/reference_audio/xiaoxiao_narrator_short.wav` with its matching transcript. No model, narrator, or generation-architecture change was made.

## Validated behavior to carry into production

### Generation and segmentation

- Keep a coherent narration passage intact instead of generating every sentence separately. Sentence-by-sentence generation created audible performance resets and was rejected.
- Whitespace and additional newlines did not materially control CosyVoice prosody. Do not treat text formatting alone as a reliable pause-control mechanism.
- Separately generated scenes retained consistent narrator identity. A stronger performance reset is acceptable at a genuine semantic, time, or location change.
- Join separately generated scene clips with **0 ms of extra inserted silence by default**. This preserves the natural trailing silence from the first clip and natural leading silence from the second clip. It does not remove or shorten either clip's existing silence.
- Add silence at a scene boundary only when listening to that specific join justifies it. Do not apply a universal nonzero scene-gap duration.

### Targeted pause repair

[repair_pause.py](repair_pause.py) inserts silence near a user-supplied timestamp at a nearby quiet valley without resampling, fading, or altering surrounding PCM16 samples. [apply_pause_plan.py](apply_pause_plan.py) applies multiple repairs against original-audio coordinates in one output pass. Both reject overwrites and ambiguous near-duplicate insertion points.

Listening experiments found that an added **140 ms** at clearly deficient or borderline period boundaries was a useful gentle repair. Already-good periods sounded essentially unchanged, and five repairs across one approximately 160-second passage remained natural. The batch tool reproduced the approved manual-repair output bit-for-bit.

These are validated manual post-processing primitives, not automatic punctuation repair. A supplied timestamp must come from listening or another trustworthy source. Outputs still require listening review.

### Artifact and repetition handling

The production-oriented policy is to detect or flag suspected repetition, omission, truncation, or other synthesis artifacts and regenerate the affected semantic chunk with the same narrator and settings. Keep rejected evidence and cap retries. Do not attempt a new ML repair model or splice within a sentence during this milestone. Automated completeness and repetition detection remain future work.

## User listening judgments

The judgments in this section are subjective user listening results, distinct from automated WAV validation.

### Period-pause experiments

- Hard sentence-by-sentence generation improved sentence reset but sounded like a new performance, so it was rejected.
- Added whitespace and newlines did not materially change pacing.
- An extra 140 ms improved clearly bad and borderline period pauses.
- Already-good periods were essentially unchanged after the same small addition.
- Multiple 140 ms repairs across one section still sounded natural.
- The quiet-valley selector placed silence at the manually preferred breath location.

### Final scene-break experiment

Scene A and Scene B were generated independently with the same narrator reference and settings. Each was one CosyVoice chunk and passed the runner's PCM16 WAV checks.

Scene A:

> 夜深了，旧书店里只剩柜台上的一盏灯。林舟合上账本，把最后一把钥匙放进抽屉。窗外的雨声渐渐远去。

Scene B:

> 第二天清晨，城南车站笼在一层薄雾里。林舟提着行李走上站台，远处的列车刚刚亮起前灯。广播响起时，他回头望了一眼空荡荡的街口。

The comparison preserved both generated clips without trimming, fades, resampling, or sentence-level repair. It tested 0, 700, and 1000 ms of **additional** silence. The user found:

- narrator identity remained consistent across the independently generated scenes;
- there was no awkwardness or audible join artifact;
- 0 ms of added silence was preferred;
- Scene B sounded excellent and Scene A was acceptable;
- both 700 ms and 1000 ms of added silence felt too long.

Local, Git-ignored evidence:

- `outputs/evaluation/cosyvoice_baseline_2026-09-15_02-07-01/manifest.json` — Scene A generation;
- `outputs/evaluation/cosyvoice_baseline_2026-09-15_02-08-21/manifest.json` — Scene B generation;
- `outputs/evaluation/milestone_c_scene_break_comparison.json` — exact texts, settings, hashes, gaps, and comparison filenames.

Automated comparison checks confirmed 24 kHz mono PCM16, exact source-payload preservation, exact inserted gap frame counts, and zero clipped samples. Those checks do not replace the user listening judgments above.

## Experimental and rejected alignment approaches

Automatic punctuation-to-audio alignment is explicitly deferred. Neither experiment is authorized to drive production pause insertion.

### Quiet-region heuristic

[estimate_period_boundaries.py](estimate_period_boundaries.py) combines chunk-relative text-position priors with quiet-region candidates and monotonic matching. It is deterministic and useful for evaluation proposals, but quietness and ordering do not establish speech boundaries. Against five manual references, it missed the known `打开北门。` boundary by approximately **398 ms**: manual reference 89.6331 seconds versus estimate approximately 90.0312 seconds.

The heuristic remains proposal-only evaluation tooling. Do not tune it further for Milestone C and do not automatically pass its output to pause repair.

### Known-transcript Mandarin CTC

[align_mandarin_ctc.py](align_mandarin_ctc.py) tested known-transcript forced alignment using `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` in the isolated `tts-align` environment. The model loaded, vocabulary validation passed, CUDA inference ran, and JSON evidence was produced.

The same target boundary remained inaccurate: manual reference 89.6331 seconds versus a CTC estimate of approximately 90.29 seconds, an error of approximately **657 ms**. A deeper audit found no large timing, resampling, Wav2Vec2 frame-spacing, or global-offset error. Per-character and continuous-transcript target construction gave essentially the same anchors around the failure.

This CTC implementation and [requirements-align.txt](requirements-align.txt) remain experimental evidence only. Do not tune it, switch alignment models, or continue alignment research within Milestone C.

## Production boundary versus evaluation tooling

Production-oriented behavior established by this milestone:

- consistent short narrator reference and unchanged CosyVoice settings;
- coherent semantic generation rather than sentence-by-sentence synthesis;
- scene-aware separate generation with zero extra join silence by default;
- sparse, listening-justified 140 ms period repair using quiet-valley placement;
- flag-and-regenerate handling for synthesis artifacts and repetition.

Evaluation-only components retained for reproducibility:

- `run_cosyvoice_smoke.py` custom-text experiments and local run manifests;
- manual chunk/reference metadata under `evaluation/inputs/`;
- the quiet-region boundary estimator and its proposal reports;
- the Mandarin CTC runner, isolated requirements, and alignment reports;
- all generated WAV comparisons under `outputs/evaluation/`.

The pause-repair algorithms are sufficiently validated to reuse in a production pipeline, but the current command-line files remain under `evaluation/` and are not integrated into `src/` or the GUI.

## Future enhancements

- Build the production audiobook job runner with semantic chunking, deterministic scene concatenation, manifests, bounded retries, resume support, and chapter-level assembly.
- Add conservative artifact/completeness/repetition detection that flags outputs rather than silently rewriting them.
- Add pronunciation overrides and safer normalization for names, numbers, identifiers, and mixed-language text.
- Validate narrator and pacing consistency across full chapters and multiple chapters, not only diagnostic passages.
- Revisit automatic alignment only as a separate future research effort with a clear accuracy target and benchmark; it is not required for the production runner.
- Integrate CosyVoice into the application only after the non-GUI pipeline is reliable.

## Reproduction and validation

Versioned Milestone C inputs are [cosyvoice_long_passage_2026-09-14_chunks.json](inputs/cosyvoice_long_passage_2026-09-14_chunks.json) and [cosyvoice_long_passage_2026-09-14_references.json](inputs/cosyvoice_long_passage_2026-09-14_references.json). Generated WAVs and reports remain under the Git-ignored `outputs/evaluation/` tree. Historical Melo and CosyVoice Milestone B evidence remains unchanged.

Run the full model-free suite in the isolated alignment environment so the CTC tests have their declared dependencies:

```bash
conda run -n tts-align python -B -m unittest discover -s tests -v
```

Final closeout result on 2026-09-15: **90 tests ran and all 90 passed** in WSL `tts-align`. GPU synthesis and user listening are separate validation steps; a passing model-free suite does not establish audio quality.
