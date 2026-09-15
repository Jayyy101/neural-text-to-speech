"""Manually extend one existing pause in a PCM16 WAV near a supplied timestamp.

Uses only local audio energy, not speech or boundary detection. Listen to the
result: the quietest region in a user-selected window may still contain speech.
Original sample bytes are preserved; no fades or resampling are applied.
Prefers the earliest local quiet valley within 3 dB RMS of the window minimum.
"""

import argparse
from array import array
import math
from pathlib import Path
import sys
import wave


QUIET_VALLEY_TOLERANCE_DB = 3.0


def select_quiet_point(data, rate, channels, around, search_ms):
    """Return a frame boundary, clipped search bounds, and local RMS in PCM units.

    Analyze overlapping 10 ms regions using squared samples across all channels.
    Choose the earliest local minimum within QUIET_VALLEY_TOLERANCE_DB of the
    minimum RMS. Collapse flat minima into one candidate at their midpoint.
    If the minimum is zero, only zero-energy valleys qualify (no noise floor).
    Inside the middle half of the winning region, minimize the amplitudes on
    both sides of the insertion, then prefer its center. Never downmix channels:
    opposite-polarity stereo must not masquerade as silence.
    """
    frames = len(data) // (2 * channels)
    radius = search_ms / 1000
    start = max(0, math.ceil(max(0.0, around - radius) * rate))
    end = min(frames, math.floor(min(frames / rate, around + radius) * rate))
    if end - start < 2:
        raise ValueError("Search window must contain at least two audio frames.")

    samples = array("h")
    samples.frombytes(data[start * channels * 2:end * channels * 2])
    if sys.byteorder != "little":
        samples.byteswap()
    energy = [
        sum(int(value) ** 2 for value in samples[i:i + channels])
        for i in range(0, len(samples), channels)
    ]
    width = min(len(energy), max(2, round(rate * 0.010)))
    total = sum(energy[:width])
    region_energy = [total]
    for offset in range(1, len(energy) - width + 1):
        total += energy[offset + width - 1] - energy[offset - 1]
        region_energy.append(total)

    # A 3 dB RMS margin is 10**(3/10) in squared energy, not 10**(3/20).
    limit = min(region_energy) * 10 ** (QUIET_VALLEY_TOLERANCE_DB / 10)
    valleys = []
    left = 0
    while left < len(region_energy):
        right = left
        while right + 1 < len(region_energy) and region_energy[right + 1] == region_energy[left]:
            right += 1
        lower_than_left = left == 0 or region_energy[left] < region_energy[left - 1]
        lower_than_right = right == len(region_energy) - 1 or region_energy[right] < region_energy[right + 1]
        if lower_than_left and lower_than_right and region_energy[left] <= limit:
            valleys.append((left + right) // 2)
        left = right + 1
    # The global minimum always supplies at least one qualifying valley.
    best_start = valleys[0]

    center = best_start + width / 2
    first = best_start + max(1, width // 4)
    last = best_start + min(width - 1, (3 * width) // 4)
    point = min(
        range(first, last + 1),
        key=lambda i: (energy[i - 1] + energy[i], abs(i - center), i),
    )
    rms = math.sqrt(region_energy[best_start] / (width * channels))
    return start + point, start, end, rms


def validate_positive(label, value):
    """Validate CLI or JSON numeric input without accepting booleans or strings."""
    try:
        valid = (isinstance(value, (int, float)) and not isinstance(value, bool)
                 and math.isfinite(value) and value > 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label} must be a finite positive number.")


def check_output_paths(input_path, output_path):
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if input_path == output_path:
        raise ValueError("Input and output must be different files.")
    if output_path.exists():
        raise ValueError(f"Output already exists; choose a new filename: {output_path}")
    return input_path, output_path


def read_pcm16(input_path):
    """Read and validate the original PCM payload once, without conversion."""
    try:
        with wave.open(str(input_path), "rb") as audio:
            if audio.getcomptype() != "NONE" or audio.getsampwidth() != 2:
                raise ValueError("Unsupported WAV encoding: only uncompressed PCM16 is supported.")
            rate, channels, frames = (
                audio.getframerate(), audio.getnchannels(), audio.getnframes()
            )
            data = audio.readframes(frames)
    except (wave.Error, EOFError) as error:
        raise ValueError(f"Unsupported or invalid WAV; expected PCM16: {error}") from error
    if rate <= 0 or channels <= 0:
        raise ValueError("Input WAV must have a positive sample rate and channel count.")
    if frames == 0:
        raise ValueError("Input WAV is empty.")
    if len(data) != frames * channels * 2:
        raise ValueError("Input WAV contains a truncated PCM payload.")
    return data, rate, channels


def plan_pause(data, rate, channels, around, add_ms, search_ms=250.0):
    """Resolve one repair in original-audio coordinates without writing audio."""
    for label, value in (("around", around), ("add-ms", add_ms), ("search-ms", search_ms)):
        validate_positive(label, value)
    frames = len(data) // (channels * 2)
    duration = frames / rate
    if around >= duration:
        raise ValueError(f"around must be inside the audio (0 < seconds < {duration:.6f}).")
    requested_frames = add_ms / 1000 * rate
    if not math.isfinite(requested_frames):
        raise ValueError("add-ms is too large for PCM WAV output.")
    added_frames = round(requested_frames)
    if added_frames < 1:
        raise ValueError("add-ms rounds to zero samples; request at least one frame of silence.")
    if 36 + (frames + added_frames) * channels * 2 > 0xFFFFFFFF:
        raise ValueError("add-ms would exceed the standard RIFF WAV output size limit.")

    point, start, end, rms = select_quiet_point(data, rate, channels, around, search_ms)
    return {
        "input_duration_seconds": duration,
        "around_seconds": around,
        "search_start_seconds": start / rate,
        "search_end_seconds": end / rate,
        "insertion_frame": point,
        "insertion_seconds": point / rate,
        "requested_add_ms": add_ms,
        "added_frames": added_frames,
        "actual_add_ms": added_frames / rate * 1000,
        "output_duration_seconds": (frames + added_frames) / rate,
        "sample_rate_hz": rate,
        "channels": channels,
        "quiet_region_rms_pcm16": rms,
        "quiet_valley_tolerance_db": QUIET_VALLEY_TOLERANCE_DB,
    }


def write_pause_insertions(output_path, data, rate, channels, repairs):
    """Validate resolved repairs, then write original samples and silence once.

    Points at most 10 ms apart are effectively duplicate for this prototype.
    Overlapping search windows are allowed when their resolved points differ.
    """
    ordered = sorted(repairs, key=lambda repair: repair["insertion_frame"])
    frames = len(data) // (channels * 2)
    for previous, current in zip(ordered, ordered[1:]):
        if current["insertion_frame"] - previous["insertion_frame"] <= max(1, round(rate * 0.010)):
            raise ValueError(
                "Duplicate or ambiguous insertion points (within 10 ms): "
                f"plan indices {previous.get('plan_index', '?')} and "
                f"{current.get('plan_index', '?')}, original frames "
                f"{previous['insertion_frame']} and {current['insertion_frame']}."
            )
    total_added = sum(repair["added_frames"] for repair in ordered)
    if 36 + (frames + total_added) * channels * 2 > 0xFFFFFFFF:
        raise ValueError("Total inserted silence exceeds the standard RIFF WAV output size limit.")
    # Exclusive creation protects both input evidence and prior repairs.
    with Path(output_path).open("xb") as output_file:
        with wave.open(output_file, "wb") as output:
            output.setparams((channels, 2, rate, frames + total_added, "NONE", "not compressed"))
            cursor = 0
            for repair in ordered:
                split = repair["insertion_frame"] * channels * 2
                output.writeframesraw(data[cursor:split])
                remaining = repair["added_frames"]
                while remaining:
                    count = min(remaining, 65536)
                    output.writeframesraw(b"\x00" * (count * channels * 2))
                    remaining -= count
                cursor = split
            output.writeframesraw(data[cursor:])


def repair_pause(input_path, output_path, around, add_ms, search_ms=250.0):
    """Insert silence at a locally quiet point and return measured repair details."""
    for label, value in (("around", around), ("add-ms", add_ms), ("search-ms", search_ms)):
        validate_positive(label, value)
    input_path, output_path = check_output_paths(input_path, output_path)
    data, rate, channels = read_pcm16(input_path)
    result = plan_pause(data, rate, channels, around, add_ms, search_ms)
    write_pause_insertions(output_path, data, rate, channels, [result])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Existing PCM16 WAV.")
    parser.add_argument("--output", required=True, type=Path, help="New PCM16 WAV; must not exist.")
    parser.add_argument("--around", required=True, type=float, help="Approximate boundary in seconds.")
    parser.add_argument("--add-ms", required=True, type=float, help="Positive silence duration in ms.")
    parser.add_argument(
        "--search-ms", type=float, default=250.0,
        help="Search radius on each side of --around in ms (default: 250).",
    )
    args = parser.parse_args(argv)
    try:
        result = repair_pause(args.input, args.output, args.around, args.add_ms, args.search_ms)
    except (ValueError, OSError, wave.Error) as error:
        parser.error(str(error))

    print(f"Input: {result['input_duration_seconds']:.6f} s; "
          f"{result['sample_rate_hz']} Hz, {result['channels']} channel(s), PCM16")
    print(f"Requested boundary: {result['around_seconds']:.6f} s")
    print(f"Search window: {result['search_start_seconds']:.6f} "
          f"to {result['search_end_seconds']:.6f} s")
    print(f"Selection: earliest quiet valley within {result['quiet_valley_tolerance_db']:g} dB RMS "
          "of the minimum (zero minimum admits only zero-energy valleys)")
    print(f"Selected insertion: {result['insertion_seconds']:.6f} s "
          f"(frame {result['insertion_frame']}); region RMS "
          f"{result['quiet_region_rms_pcm16']:.3f} PCM16 units")
    print(f"Silence: requested {result['requested_add_ms']:.6f} ms; "
          f"added {result['actual_add_ms']:.6f} ms ({result['added_frames']} frames)")
    print(f"Output: {result['output_duration_seconds']:.6f} s -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
