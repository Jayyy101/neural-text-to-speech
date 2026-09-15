"""Jointly match Chinese punctuation to quiet regions in read-only PCM16 audio.

Metadata: {"schema_version": 1, "chunks": [{"text": "文字。", "duration_seconds": 1.0}]}.
Chunks must describe consecutive, unmodified audio in playback order. Optional
extra provenance fields are allowed. No CosyVoice imports or text normalization.
Count all non-whitespace characters, including punctuation. The fraction through
each 。？！ provides a soft timing prior, including for chunk-final punctuation.
Scan 10 ms RMS windows every 5 ms. Each connected region at or below 25% of
chunk RMS (about -12 dB) supplies one candidate, located with repair_pause's
3 dB earliest-valley rule. No downmixing or audio modification.
Joint cost = sum((timing error / mean punctuation spacing)**2
                 + (candidate RMS / quiet threshold)**2).
Dynamic programming chooses strictly ordered, distinct candidates. Equal costs
prefer earlier candidates. Scores are costs, not calibrated confidence.
If too few candidates exist, the chunk is reported unmatched, without proposals.
Optional --reference is evaluated AFTER matching, never used in the cost.
This is a heuristic, not speech alignment. Listen before applying any proposal.
"""

import argparse
from array import array
import json
import math
from pathlib import Path
import sys

if __package__:
    from .repair_pause import read_pcm16, select_quiet_point, validate_positive
else:
    from repair_pause import read_pcm16, select_quiet_point, validate_positive


def read_chunk_metadata(path):
    try:
        metadata = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (UnicodeError, ValueError) as error:
        raise ValueError(f"Invalid UTF-8 JSON chunk metadata: {error}") from error
    if (not isinstance(metadata, dict)
            or type(metadata.get("schema_version")) is not int
            or metadata["schema_version"] != 1):
        raise ValueError("Chunk metadata must be an object with schema_version 1.")
    chunks = metadata.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Chunk metadata must contain a nonempty chunks list.")
    durations = []
    spans = []
    for index, chunk in enumerate(chunks, 1):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk {index} must be an object.")
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk {index} text must be a nonempty string.")
        duration = chunk.get("duration_seconds")
        validate_positive(f"Chunk {index} duration_seconds", duration)
        start = math.fsum(durations)
        durations.append(duration)
        try:
            end = math.fsum(durations)
        except OverflowError as error:
            raise ValueError("Total chunk duration is too large.") from error
        if end <= start:
            raise ValueError(f"Chunk {index} duration is too small for its timeline.")
        spans.append({"chunk_index": index, "text": text,
                      "start_seconds": start, "end_seconds": end,
                      "duration_seconds": duration})
    return spans


def punctuation_positions(text):
    """Yield original offsets and non-whitespace fractions after each 。？！."""
    total = sum(not character.isspace() for character in text)
    position = 0
    for offset, character in enumerate(text):
        position += not character.isspace()
        if character in "。？！":
            yield offset, position / total


def quiet_candidates(data, rate, channels):
    """One candidate per connected low-RMS region, in chunk-local frame order."""
    samples = array("h")
    samples.frombytes(data)
    if sys.byteorder != "little":
        samples.byteswap()
    energy = [sum(int(v) ** 2 for v in samples[i:i + channels])
              for i in range(0, len(samples), channels)]
    width = min(len(energy), max(2, round(rate * 0.010)))
    hop = max(1, round(rate * 0.005))
    threshold = 0.25 * math.sqrt(sum(energy) / (len(energy) * channels))
    starts = list(range(0, len(energy) - width + 1, hop))
    if starts[-1] != len(energy) - width:
        starts.append(len(energy) - width)
    quiet = [sum(energy[start:start + width]) <= threshold**2 * width * channels
             for start in starts]
    candidates = []
    index = 0
    while index < len(starts):
        if not quiet[index]:
            index += 1
            continue
        last = index
        while last + 1 < len(starts) and quiet[last + 1]:
            last += 1
        left, right = starts[index], starts[last] + width
        region = data[left * channels * 2:right * channels * 2]
        # Slice exactly the region; an oversized radius avoids float edge clipping.
        point, _, _, rms = select_quiet_point(
            region, rate, channels, (right - left) / (2 * rate),
            (right - left) / rate * 1000,
        )
        candidates.append({"candidate_index": len(candidates) + 1,
                           "frame": left + point,
                           "seconds": (left + point) / rate,
                           "region_start_seconds": left / rate,
                           "region_end_seconds": right / rate,
                           "rms_pcm16": rms,
                           "quiet_cost": (rms / threshold)**2 if threshold else 0.0})
        index = last + 1
    return candidates


def match_monotonic(expected, candidates, duration):
    """O(punctuation * candidates) DP; skip candidates, never reuse or reorder.

    Return (candidate index, individual cost) for each punctuation, or None if
    there are too few distinct candidates. Indices here are zero-based.
    """
    if not expected:
        return []
    if len(candidates) < len(expected):
        return None
    spacing = duration / len(expected)
    costs = [[((candidate["seconds"] - time) / spacing)**2 + candidate["quiet_cost"]
              for candidate in candidates] for time in expected]
    previous = costs[0]
    parents = []
    for row in costs[1:]:
        current = [math.inf] * len(candidates)
        parent = [-1] * len(candidates)
        best = -1
        for j in range(1, len(candidates)):
            if best == -1 or previous[j - 1] < previous[best]:
                best = j - 1
            current[j] = previous[best] + row[j]
            parent[j] = best
        parents.append(parent)
        previous = current
    chosen = min(range(len(candidates)), key=lambda j: previous[j])
    result = []
    for i in range(len(expected) - 1, -1, -1):
        result.append((chosen, costs[i][chosen]))
        if i:
            chosen = parents[i - 1][chosen]
    return list(reversed(result))


def estimate_boundaries(input_path, metadata_path, add_ms=140.0):
    validate_positive("add-ms", add_ms)
    spans = read_chunk_metadata(metadata_path)
    data, rate, channels = read_pcm16(input_path)
    frames = len(data) // (2 * channels)
    # Allow at most one frame of floating-point / timestamp rounding difference.
    if abs(spans[-1]["end_seconds"] - frames / rate) > 1 / rate + 1e-12:
        raise ValueError("Total chunk duration must match the WAV duration within one frame.")
    boundaries = []
    chunk_reports = []
    for chunk in spans:
        start_frame = round(chunk["start_seconds"] * rate)
        end_frame = min(frames, round(chunk["end_seconds"] * rate))
        if end_frame - start_frame < 2:
            raise ValueError(f"Chunk {chunk['chunk_index']} must contain at least two audio frames.")
        chunk_data = data[start_frame * channels * 2:end_frame * channels * 2]
        positions = list(punctuation_positions(chunk["text"]))
        candidates = quiet_candidates(chunk_data, rate, channels) if positions else []
        expected = [chunk["start_seconds"] + fraction * chunk["duration_seconds"]
                    - start_frame / rate for _, fraction in positions]
        matches = match_monotonic(expected, candidates, chunk["duration_seconds"])
        chunk_reports.append({**chunk, "candidate_count": len(candidates),
                              "status": "matched" if matches is not None else "insufficient_candidates",
                              "match_cost": sum(cost for _, cost in matches) if matches is not None else None})
        for position_index, (offset, fraction) in enumerate(positions):
            initial = chunk["start_seconds"] + fraction * chunk["duration_seconds"]
            candidate, cost = (None, None)
            if matches is not None:
                candidate_index, cost = matches[position_index]
                candidate = candidates[candidate_index]
            selected = (start_frame + candidate["frame"]) / rate if candidate else None
            boundaries.append({
                "punctuation_index": len(boundaries) + 1,
                "punctuation": chunk["text"][offset],
                "status": "matched" if candidate else "insufficient_candidates",
                "context": chunk["text"][max(0, offset - 12):offset + 13],
                "chunk_index": chunk["chunk_index"],
                "chunk_start_seconds": chunk["start_seconds"],
                "chunk_end_seconds": chunk["end_seconds"],
                "text_offset": offset,
                "relative_text_position": fraction,
                "initial_seconds": initial,
                "selected_seconds": selected,
                "delta_seconds": selected - initial if candidate else None,
                "candidate_index": candidate["candidate_index"] if candidate else None,
                "quiet_region_rms_pcm16": candidate["rms_pcm16"] if candidate else None,
                "quiet_cost": candidate["quiet_cost"] if candidate else None,
                "match_cost": cost,
            })
    periods = [boundary for boundary in boundaries if boundary["punctuation"] == "。"]
    for index, period in enumerate(periods, 1):
        period["period_index"] = index
    return {
        "proposal_only": True,
        "coordinate_system": "original_input_wav_seconds",
        "input_wav": str(input_path),
        "chunk_metadata": str(metadata_path),
        "input_duration_seconds": frames / rate,
        "method": "chunk-wide monotonic punctuation matching v1",
        "scoring": "(timing error / mean punctuation spacing)^2 + (candidate RMS / quiet threshold)^2; lower is better, not confidence",
        "candidate_settings": {"window_ms": 10, "hop_ms": 5, "chunk_rms_fraction": 0.25,
                               "quiet_valley_tolerance_db": 3.0},
        "limitations": "Heuristic matching, not speech alignment. Quietness and order do not prove punctuation location. Listen before applying; unmatched chunks produce no pauses.",
        "chunks": chunk_reports,
        "boundaries": boundaries,
        "periods": periods,
        "pauses": [{"around": period["selected_seconds"], "add_ms": add_ms,
                    "label": "period", "source": "estimated"} for period in periods
                   if period["status"] == "matched"],
    }


def evaluate_references(report, path):
    """Evaluate uniquely identified context suffixes AFTER matching is complete."""
    try:
        metadata = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (UnicodeError, ValueError) as error:
        raise ValueError(f"Invalid UTF-8 JSON reference metadata: {error}") from error
    if (not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int
            or metadata["schema_version"] != 1
            or not isinstance(metadata.get("references"), list) or not metadata["references"]):
        raise ValueError("Reference metadata requires schema_version 1 and a nonempty references list.")
    chunks = {chunk["chunk_index"]: chunk for chunk in report["chunks"]}
    boundaries = {(b["chunk_index"], b["text_offset"]): b for b in report["boundaries"]}
    results, seen = [], set()
    for index, reference in enumerate(metadata["references"], 1):
        if not isinstance(reference, dict):
            raise ValueError(f"Reference {index} must be an object.")
        chunk_index, context = reference.get("chunk_index"), reference.get("context")
        if type(chunk_index) is not int or chunk_index not in chunks:
            raise ValueError(f"Reference {index} has an invalid chunk_index.")
        if not isinstance(context, str) or not context or context[-1] not in "。？！":
            raise ValueError(f"Reference {index} context must end at the target punctuation.")
        text = chunks[chunk_index]["text"]
        start = text.find(context)
        if start < 0 or text.find(context, start + 1) >= 0:
            raise ValueError(f"Reference {index} context must occur exactly once in its chunk.")
        key = (chunk_index, start + len(context) - 1)
        if key in seen:
            raise ValueError(f"Reference {index} duplicates a punctuation boundary.")
        seen.add(key)
        seconds = reference.get("seconds")
        validate_positive(f"Reference {index} seconds", seconds)
        chunk = chunks[chunk_index]
        if not chunk["start_seconds"] <= seconds <= chunk["end_seconds"]:
            raise ValueError(f"Reference {index} seconds must lie within its chunk.")
        boundary = boundaries[key]
        selected = boundary["selected_seconds"]
        results.append({"chunk_index": chunk_index, "context": context,
                        "punctuation_index": boundary["punctuation_index"],
                        "reference_seconds": seconds, "selected_seconds": selected,
                        "absolute_error_seconds": abs(selected - seconds) if selected is not None else None})
    errors = [r["absolute_error_seconds"] for r in results if r["absolute_error_seconds"] is not None]
    return {"reference_file": str(path), "references": results,
            "reference_count": len(results), "matched_count": len(errors),
            "mae_seconds": math.fsum(errors) / len(errors) if errors else None,
            "max_error_seconds": max(errors) if errors else None}


def main(argv=None):
    # Windows redirected streams may default to CP1252, which cannot print Mandarin.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Original PCM16 WAV (read-only).")
    parser.add_argument("--chunks", required=True, type=Path, help="UTF-8 chunk metadata JSON, schema_version 1.")
    parser.add_argument("--add-ms", type=float, default=140.0, help="Proposed silence metadata only (default: 140 ms).")
    parser.add_argument("--reference", type=Path, help="Optional manual reference JSON, used only for evaluation after matching.")
    parser.add_argument("--output", type=Path, help="Optional new JSON proposal file; never overwrites an existing file.")
    args = parser.parse_args(argv)
    try:
        if args.output is not None and args.output.exists():
            raise ValueError(f"Output already exists; choose a new filename: {args.output}")
        report = estimate_boundaries(args.input, args.chunks, args.add_ms)
        if args.reference is not None:
            report["evaluation"] = evaluate_references(report, args.reference)
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as output:
                json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
                output.write("\n")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print("PROPOSAL ONLY — original input-WAV coordinates; audio unchanged.")
    print(report["limitations"])
    for period in report["boundaries"]:
        context = json.dumps(period["context"], ensure_ascii=False)
        print(f"Punctuation {period['punctuation_index']:02d} {period['punctuation']} | chunk {period['chunk_index']:02d} "
              f"[{period['chunk_start_seconds']:.6f}, {period['chunk_end_seconds']:.6f}] s | "
              f"fraction {period['relative_text_position']:.6f} | {context}")
        if period["status"] == "matched":
            print(f"  expected {period['initial_seconds']:.6f} s -> matched {period['selected_seconds']:.6f} s | "
                  f"delta {period['delta_seconds']:+.6f} s | RMS {period['quiet_region_rms_pcm16']:.3f} PCM16 | "
                  f"quiet cost {period['quiet_cost']:.6f} | match cost {period['match_cost']:.6f}")
        else:
            print(f"  expected {period['initial_seconds']:.6f} s -> UNMATCHED: too few quiet candidates in chunk")
    print(f"Punctuation: {len(report['boundaries'])}; period proposals: {len(report['pauses'])}; "
          f"WAV duration: {report['input_duration_seconds']:.6f} s")
    if "evaluation" in report:
        evaluation = report["evaluation"]
        for reference in evaluation["references"]:
            error = reference["absolute_error_seconds"]
            error_text = f"absolute error {error:.6f} s" if error is not None else "UNMATCHED"
            print(f"Reference {reference['context']}: {reference['reference_seconds']:.6f} s | {error_text}")
        print(f"References matched: {evaluation['matched_count']}/{evaluation['reference_count']} "
              "(error statistics cover matched references only)")
        if evaluation["matched_count"]:
            print(f"MAE: {evaluation['mae_seconds']:.6f} s; max error: {evaluation['max_error_seconds']:.6f} s")
    if args.output is not None:
        print(f"Proposal JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
