"""Read-only measurements for the frozen, unpromoted bounded-QC panel."""

from array import array
import math
import sys
import wave

from src.audiobook.content_qc import compare_recognition, han_tokens, levenshtein_steps


def local_edit_candidates(normalized_text, recognition):
    """Describe nearby mixed edits without treating spelling as pronunciation."""
    expected = han_tokens(normalized_text)
    recognized = [item["comparison_token"] for item in recognition["comparison_tokens"]]
    steps, _ = levenshtein_steps(expected, recognized)
    clusters = []
    current = []
    gap = []
    for step in steps:
        if step["operation"] == "match":
            if current:
                gap.append(step)
                if len(gap) > 2:
                    clusters.append(current)
                    current, gap = [], []
        else:
            if gap:
                current.extend(gap)
                gap = []
            current.append(step)
    if current:
        clusters.append(current)
    result = []
    for cluster in clusters:
        edits = [step for step in cluster if step["operation"] != "match"]
        operations = sorted({step["operation"] for step in edits})
        if len(operations) < 2:
            continue
        expected_indices = [step["expected_index"] for step in cluster
                            if step["expected_index"] is not None]
        recognized_indices = [step["recognized_index"] for step in cluster
                              if step["recognized_index"] is not None]
        result.append({
            "operations": operations,
            "edit_count": len(edits),
            "expected_start": min(expected_indices) if expected_indices else None,
            "expected_text": "".join(expected[i] for i in expected_indices),
            "recognized_text": "".join(recognized[i] for i in recognized_indices),
            "recognized_time_span_seconds": (
                [recognition["comparison_tokens"][min(recognized_indices)]["start_seconds"],
                 recognition["comparison_tokens"][max(recognized_indices)]["end_seconds"]]
                if recognized_indices else None
            ),
        })
    return result


def _dbfs(samples):
    if not samples:
        return -200.0
    power = sum(value * value for value in samples) / len(samples)
    return round(20 * math.log10(max(math.sqrt(power) / 32768, 1e-10)), 3)


def measure_endpoint(wav_path, recognition):
    """Measure the physical edge without identifying speech or suggesting cuts."""
    with wave.open(str(wav_path), "rb") as wav:
        rate, channels, width, frames = (wav.getframerate(), wav.getnchannels(),
                                         wav.getsampwidth(), wav.getnframes())
        if (rate, channels, width) != (24000, 1, 2) or frames < 2:
            raise ValueError("Endpoint measurements require mono 24 kHz PCM16 WAV.")
        payload = wav.readframes(frames)
    if len(payload) != frames * 2:
        raise ValueError("Endpoint PCM is incomplete.")
    samples = array("h")
    samples.frombytes(payload)
    if sys.byteorder != "little":
        samples.byteswap()
    bins = []
    quiet = 0
    rms_limit = 32768 * 10 ** (-55 / 20)
    peak_limit = 32768 * 10 ** (-45 / 20)
    window_frames = min(240, frames)
    for offset in range(min(5, frames // window_frames)):
        end = frames - offset * window_frames
        window = samples[end - window_frames:end]
        rms = math.sqrt(sum(value * value for value in window) / len(window))
        peak = max(abs(value) for value in window)
        bins.append({"rms_dbfs": _dbfs(window),
                     "peak_dbfs": round(20 * math.log10(max(peak / 32768, 1e-10)), 3)})
        if offset == quiet and rms <= rms_limit and peak <= peak_limit:
            quiet += 1
    tokens = recognition["comparison_tokens"]
    last_end = tokens[-1]["end_seconds"] if tokens else None
    return {
        "sample_rate_hz": rate, "frames": frames,
        "duration_seconds": frames / rate,
        "last_five_10ms_bins_from_edge": bins,
        "trailing_quiet_ms_within_50ms": round(quiet * window_frames * 1000 / rate, 6),
        "last_sample": samples[-1],
        "last_sample_step": samples[-1] - samples[-2],
        "last_ctc_token_end_seconds": last_end,
        "ctc_terminal_margin_seconds": (
            round(frames / rate - last_end, 6) if last_end is not None else None
        ),
    }


def evaluate_candidates(normalized_text, recognition, wav_path):
    """Reproduce passive evidence; neither unpromoted reason can reject."""
    comparison = compare_recognition(normalized_text, recognition)
    clusters = local_edit_candidates(normalized_text, recognition)
    endpoint = measure_endpoint(wav_path, recognition)
    expected = han_tokens(normalized_text)
    observed = [item["comparison_token"] for item in recognition["comparison_tokens"]]
    tail = {
        "expected_final_two_han": "".join(expected[-2:]),
        "recognized_final_two_tokens": "".join(observed[-2:]),
        "exact_final_token_match": bool(observed and expected[-1] == observed[-1]),
    }
    abstentions = []
    if clusters:
        abstentions.append({"reason": "corroborated_local_content_corruption",
                            "why": "pronunciation_unavailable_or_ambiguous"})
    if (endpoint["trailing_quiet_ms_within_50ms"] == 0
            or endpoint["ctc_terminal_margin_seconds"] is not None
            and endpoint["ctc_terminal_margin_seconds"] <= 0.35):
        abstentions.append({"reason": "corroborated_endpoint_truncation",
                            "why": "edge_and_ctc_do_not_prove_incomplete_speech"})
    reasons = (["contiguous_expected_han_deletion_v1"]
               if comparison["decision"] == "rejected" else [])
    return {
        "policy_version": 2,
        "measurements": {"local_mixed_edit_clusters": clusters,
                         "endpoint": endpoint, "terminal_text": tail},
        "decision": comparison["decision"],
        "rejection_reasons": reasons,
        "abstentions": abstentions,
        "comparison": comparison,
    }
