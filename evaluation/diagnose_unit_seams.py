"""Create read-only listening and visual evidence for selected schema-5 seams.

Run with ``python -B -m evaluation.diagnose_unit_seams RUN OUTPUT --after 5 58 69``.
Outputs are written outside RUN; selected source WAVs are never modified.
"""

import argparse
import hashlib
import json
from pathlib import Path
import wave

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.run_seam_pause_ab import edge_quiet_frames


CONTEXT_SECONDS = 2


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_wav(path, expected_hash=None):
    if expected_hash is not None and sha256(path) != expected_hash:
        raise ValueError(f"Selected WAV hash mismatch: {path}")
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(),
                audio.getcomptype()) != (1, 2, 24000, "NONE"):
            raise ValueError(f"Expected mono 24 kHz PCM16: {path}")
        frames = audio.getnframes()
        payload = audio.readframes(frames)
    if len(payload) != frames * 2:
        raise ValueError(f"Incomplete PCM: {path}")
    return frames, payload


def write_wav(path, payload):
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        output.writeframes(payload)
    if read_wav(path)[1] != payload:
        raise ValueError(f"Output PCM mismatch: {path}")


def rms_dbfs(samples):
    if not len(samples):
        return None
    rms = np.sqrt(np.mean(np.square(samples.astype(np.float64))))
    return round(float(20 * np.log10(max(rms, 1e-12) / 32768)), 1)


def edge_bins(payload, side):
    """Return 10 ms RMS/peak bins, ordered from the edge inward."""
    samples = np.frombuffer(payload, dtype="<i2")
    width = 240
    count = min(200, len(samples) // width)
    bins = []
    for offset in range(count):
        segment = (samples[offset * width:(offset + 1) * width]
                   if side == "head" else
                   samples[len(samples) - (offset + 1) * width:
                           len(samples) - offset * width])
        bins.append({
            "offset_from_edge_ms": offset * 10,
            "rms_dbfs": rms_dbfs(segment),
            "peak_dbfs": round(float(20 * np.log10(
                max(int(np.max(np.abs(segment.astype(np.int32)))), 1e-12) / 32768
            )), 1),
        })
    return bins


def plot_seam(path, left_tail, right_head, left_quiet, right_quiet):
    left = np.frombuffer(left_tail, dtype="<i2").astype(np.float32) / 32768
    right = np.frombuffer(right_head, dtype="<i2").astype(np.float32) / 32768
    joined = np.concatenate((left, right))
    seam = len(left) / 24000
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), constrained_layout=True)
    t = np.arange(len(joined)) / 24000 - seam
    axes[0].plot(t, joined, linewidth=0.35, color="#235d8d")
    axes[0].axvline(0, color="red", linewidth=1)
    axes[0].axvspan(-left_quiet / 24000, right_quiet / 24000,
                    color="orange", alpha=0.17)
    axes[0].set(xlabel="Seconds relative to seam", ylabel="Amplitude",
                title="Selected PCM waveform; red = exact unit join, orange = measured quiet edges")
    axes[0].grid(alpha=0.2)
    axes[1].specgram(joined, Fs=24000, NFFT=512, noverlap=384,
                     cmap="magma", vmin=-115, vmax=-20, xextent=(-seam, len(right) / 24000))
    axes[1].axvline(0, color="cyan", linewidth=1)
    axes[1].set(xlabel="Seconds relative to seam", ylabel="Frequency (Hz)",
                ylim=(0, 8000), title="Spectrogram")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def diagnose(run, output, boundaries):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists() or run == output or run in output.parents:
        raise ValueError("Use a new output directory outside the accepted run.")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 5 or manifest.get("assembly", {}).get("status") != "assembled":
        raise ValueError("Expected an assembled schema-5 unit run.")
    units = [unit for scene in manifest["scenes"] for unit in scene["synthesis_units"]]
    clips = manifest["assembly"]["units"]
    if not boundaries or len(set(boundaries)) != len(boundaries) or any(
            not 1 <= boundary < len(units) for boundary in boundaries):
        raise ValueError("Select distinct valid unit numbers after which to inspect.")
    chapter_path = (run / manifest["assembly"]["output_path"]).resolve()
    chapter_path.relative_to(run)
    chapter_frames, chapter_pcm = read_wav(chapter_path, manifest["assembly"]["wav_sha256"])
    if chapter_frames != manifest["assembly"]["audio"]["frames"]:
        raise ValueError("Chapter frame count differs from assembly evidence.")
    output.mkdir(parents=True)
    records = []
    for boundary in boundaries:
        selected = []
        for index in (boundary - 1, boundary):
            unit, assembled = units[index], clips[index]
            attempt_id = unit["generation"]["selected_attempt_id"]
            attempt = next(item for item in unit["generation"]["attempts"]
                           if item["id"] == attempt_id)
            if (assembled["unit_id"] != unit["id"] or
                    assembled["selected_attempt_id"] != attempt_id or
                    assembled["artifact_wav_sha256"] != attempt["wav_sha256"] or
                    attempt["content_qc"]["status"] != "passed"):
                raise ValueError(f"Selection/assembly mismatch: {unit['id']}")
            path = (run / attempt["output_path"]).resolve()
            path.relative_to(run)
            frames, payload = read_wav(path, attempt["wav_sha256"])
            if frames != assembled["frame_count"] or chapter_pcm[
                    assembled["start_frame"] * 2:assembled["end_frame_exclusive"] * 2] != payload:
                raise ValueError(f"Chapter PCM differs from selected WAV: {unit['id']}")
            selected.append((unit, attempt, frames, payload))
        left, right = selected
        left_quiet = edge_quiet_frames(left[3], 24000, "trailing")
        right_quiet = edge_quiet_frames(right[3], 24000, "leading")
        left_tail = left[3][-CONTEXT_SECONDS * 24000 * 2:]
        right_head = right[3][:CONTEXT_SECONDS * 24000 * 2]
        folder = output / f"after_unit_{boundary:04d}"
        folder.mkdir()
        wavs = {"left_tail.wav": left_tail, "right_head.wav": right_head,
                "original_seam.wav": left_tail + right_head}
        for filename, pcm in wavs.items():
            write_wav(folder / filename, pcm)
        plot_seam(folder / "seam_waveform_spectrogram.png", left_tail,
                  right_head, left_quiet, right_quiet)
        records.append({
            "after_unit_number": boundary,
            "left_unit_id": left[0]["id"], "right_unit_id": right[0]["id"],
            "left_selected_attempt_id": left[1]["id"],
            "right_selected_attempt_id": right[1]["id"],
            "left_wav_sha256": left[1]["wav_sha256"],
            "right_wav_sha256": right[1]["wav_sha256"],
            "left_duration_seconds": left[2] / 24000,
            "right_duration_seconds": right[2] / 24000,
            "left_trailing_quiet_ms": left_quiet / 24,
            "right_leading_quiet_ms": right_quiet / 24,
            "seam_chapter_seconds": clips[boundary - 1]["end_frame_exclusive"] / 24000,
            "left_last_two_seconds_rms_dbfs": rms_dbfs(np.frombuffer(left_tail, dtype="<i2")),
            "right_first_two_seconds_rms_dbfs": rms_dbfs(np.frombuffer(right_head, dtype="<i2")),
            "left_tail_10ms_bins_from_edge": edge_bins(left_tail, "tail"),
            "right_head_10ms_bins_from_edge": edge_bins(right_head, "head"),
            "outputs": {name: {"path": str((folder / name).relative_to(output)),
                               "sha256": sha256(folder / name)} for name in wavs},
            "plot": str((folder / "seam_waveform_spectrogram.png").relative_to(output)),
        })
    result = {
        "schema_version": 1, "source_run": str(run),
        "source_chapter_sha256": manifest["assembly"]["wav_sha256"],
        "analysis": "Read-only PCM and 10 ms edge energy; sound identity requires listening.",
        "quiet_gate": "Consecutive edge 10 ms windows with RMS <= -55 dBFS and peak <= -45 dBFS.",
        "boundaries": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--after", type=int, nargs="+", required=True)
    args = parser.parse_args()
    result = diagnose(args.run, args.output, args.after)
    print(f"Diagnosed {len(result['boundaries'])} selected-WAV seams in {args.output}")


if __name__ == "__main__":
    main()
