"""Export a validated chapter WAV as a named MP3 without changing the run."""

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unicodedata
import wave

from .audiobook_launcher import (
    DEFAULT_DISTRIBUTION, DEFAULT_USER, windows_to_wsl_path,
)


INVALID_WINDOWS_FILENAME_CHARACTERS = set('<>:"/\\|?*')
RESERVED_WINDOWS_NAMES = re.compile(r"^(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])$", re.I)


def normalize_mp3_filename(raw):
    """Return one safe Windows MP3 filename without a directory component."""
    name = unicodedata.normalize("NFC", raw.strip())
    if not name or any(
            char in INVALID_WINDOWS_FILENAME_CHARACTERS
            or unicodedata.category(char) in {"Cc", "Cf"}
            for char in name):
        raise ValueError("Enter an MP3 filename without Windows-forbidden characters or paths.")
    if name.lower().endswith(".mp3"):
        name = name[:-4]
    elif "." in name and not name.startswith("."):
        raise ValueError("The output filename must end in .mp3.")
    if (not name or name.endswith((" ", "."))
            or RESERVED_WINDOWS_NAMES.fullmatch(name.split(".", 1)[0].rstrip(" ."))):
        raise ValueError("Choose a different MP3 filename; this name is reserved by Windows.")
    filename = name + ".mp3"
    if len(filename.encode("utf-16-le")) // 2 > 255:
        raise ValueError("The MP3 filename is too long for Windows.")
    return filename


def ffmpeg_command(source, temporary, sample_rate, *, distribution=DEFAULT_DISTRIBUTION,
                   user=DEFAULT_USER, run=subprocess.run):
    """Build the WSL encoder argument list for a validated Windows WAV."""
    source_wsl = windows_to_wsl_path(source, distribution=distribution, user=user, run=run)
    temporary_wsl = windows_to_wsl_path(
        temporary, distribution=distribution, user=user, run=run,
    )
    return [
        "wsl.exe", "--distribution", distribution, "--user", user,
        "--exec", "/usr/bin/ffmpeg", "-nostdin", "-hide_banner",
        "-loglevel", "error", "-y", "-i", source_wsl,
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "libmp3lame", "-b:a", "96k", "-f", "mp3", temporary_wsl,
    ]


def _run_checked(command, *, run):
    result = run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"MP3 tool failed: {detail or result.returncode}")
    return result.stdout


def export_mp3(source_path, filename, *, distribution=DEFAULT_DISTRIBUTION,
               user=DEFAULT_USER, run=subprocess.run):
    """Encode, verify, then atomically publish one named MP3 beside the WAV."""
    source = Path(source_path)
    destination = source.parent / normalize_mp3_filename(filename)
    if destination.exists():
        raise FileExistsError(destination)
    with wave.open(str(source), "rb") as wav:
        sample_rate = wav.getframerate()
        duration = wav.getnframes() / sample_rate

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".mp3-export-", suffix=".mp3", dir=source.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        command = ffmpeg_command(
            source, temporary, sample_rate, distribution=distribution,
            user=user, run=run,
        )
        _run_checked(command, run=run)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("MP3 encoder produced no audio file.")
        temporary_wsl = windows_to_wsl_path(
            temporary, distribution=distribution, user=user, run=run,
        )
        probe = [
            "wsl.exe", "--distribution", distribution, "--user", user,
            "--exec", "/usr/bin/ffprobe", "-v", "error",
            "-select_streams", "a:0", "-show_entries",
            "stream=codec_name,sample_rate,channels:format=duration",
            "-of", "json", temporary_wsl,
        ]
        info = json.loads(_run_checked(probe, run=run).decode("utf-8"))
        streams = info.get("streams", [])
        stream = streams[0] if len(streams) == 1 else {}
        encoded_duration = float(info.get("format", {}).get("duration", 0))
        if (stream.get("codec_name") != "mp3"
                or int(stream.get("channels", 0)) != 1
                or int(stream.get("sample_rate", 0)) != sample_rate
                or abs(encoded_duration - duration) > 0.15):
            raise RuntimeError("MP3 verification failed: codec, channel, rate, or duration mismatch.")
        # The temporary file is complete. A same-directory hard link publishes
        # its final name atomically and fails if that name already exists.
        os.link(temporary, destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)
