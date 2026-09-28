"""Model-free checks for verified, non-overwriting MP3 export."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave

from src.audiobook_mp3 import export_mp3, ffmpeg_command


class Mp3ExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.final = Path(temporary.name) / "final"
        self.final.mkdir()
        self.wav = self.final / "chapter.wav"
        with wave.open(str(self.wav), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(24000)
            audio.writeframes(b"\x00\x00" * 24000)
        self.original = self.wav.read_bytes()

    def _run(self, command, **kwargs):
        self.assertIsInstance(command, list)
        self.assertFalse(kwargs.get("check"))
        if command[6] == "/usr/bin/ffmpeg":
            Path(command[-1]).write_bytes(b"encoded MP3")
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if command[6] == "/usr/bin/ffprobe":
            info = {"streams": [{"codec_name": "mp3", "channels": 1,
                                 "sample_rate": "24000"}],
                    "format": {"duration": "1.048"}}
            return subprocess.CompletedProcess(command, 0, json.dumps(info).encode(), b"")
        self.fail(f"Unexpected command: {command}")

    def _export(self, name="神通者01"):
        with patch("src.audiobook_mp3.windows_to_wsl_path", side_effect=lambda path, **_: str(path)):
            return export_mp3(self.wav, name, run=self._run)

    def test_ffmpeg_argument_list(self):
        with patch("src.audiobook_mp3.windows_to_wsl_path", side_effect=lambda path, **_: str(path)):
            command = ffmpeg_command(self.wav, self.final / "temp.mp3", 24000)
        self.assertEqual(command[:6], ["wsl.exe", "--distribution", "Ubuntu-22.04",
                                       "--user", "jay", "--exec"])
        self.assertEqual(command[6], "/usr/bin/ffmpeg")
        self.assertEqual(command[command.index("-c:a") + 1], "libmp3lame")
        self.assertEqual(command[command.index("-b:a") + 1], "96k")
        self.assertEqual(command[command.index("-ac") + 1], "1")
        self.assertEqual(command[command.index("-ar") + 1], "24000")
        self.assertEqual(command[-1], str(self.final / "temp.mp3"))

    def test_verified_unicode_export_and_temp_cleanup(self):
        destination = self._export()
        self.assertEqual(destination.name, "神通者01.mp3")
        self.assertEqual(destination.read_bytes(), b"encoded MP3")
        self.assertEqual(self.wav.read_bytes(), self.original)
        self.assertEqual(list(self.final.glob(".mp3-export-*")), [])

    def test_encoder_and_probe_hide_windows_without_losing_errors(self):
        calls = []

        def recorded_run(command, **kwargs):
            calls.append((command, kwargs))
            return self._run(command, **kwargs)

        with patch("src.audiobook_mp3.windows_to_wsl_path", side_effect=lambda path, **_: str(path)):
            with patch.object(subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True):
                export_mp3(self.wav, "神通者01", run=recorded_run)
        self.assertEqual([call[0][6] for call in calls],
                         ["/usr/bin/ffmpeg", "/usr/bin/ffprobe"])
        for command, kwargs in calls:
            self.assertIsInstance(command, list)
            self.assertEqual(kwargs["creationflags"], 0x08000000)
            self.assertEqual(kwargs["stdout"], subprocess.PIPE)
            self.assertEqual(kwargs["stderr"], subprocess.PIPE)

    def test_existing_mp3_is_never_overwritten(self):
        destination = self.final / "神通者01.mp3"
        destination.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            self._export()
        self.assertEqual(destination.read_bytes(), b"keep")
        self.assertEqual(self.wav.read_bytes(), self.original)
        self.assertEqual(list(self.final.glob(".mp3-export-*")), [])

    def test_publish_collision_is_never_overwritten(self):
        destination = self.final / "神通者01.mp3"

        def collide(command, **kwargs):
            result = self._run(command, **kwargs)
            if command[6] == "/usr/bin/ffprobe":
                destination.write_bytes(b"other export")
            return result

        with patch("src.audiobook_mp3.windows_to_wsl_path", side_effect=lambda path, **_: str(path)):
            with self.assertRaises(FileExistsError):
                export_mp3(self.wav, "神通者01.mp3", run=collide)
        self.assertEqual(destination.read_bytes(), b"other export")
        self.assertEqual(list(self.final.glob(".mp3-export-*")), [])

    def test_encoder_failure_preserves_wav_and_cleans_temp(self):
        def fail(command, **kwargs):
            return subprocess.CompletedProcess(command, 1, b"", b"encoder failed")

        with patch("src.audiobook_mp3.windows_to_wsl_path", side_effect=lambda path, **_: str(path)):
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                export_mp3(self.wav, "神通者01", run=fail)
        self.assertEqual(self.wav.read_bytes(), self.original)
        self.assertEqual(list(self.final.glob("*.mp3")), [])

    def test_probe_failure_preserves_wav_and_cleans_temp(self):
        def bad_probe(command, **kwargs):
            if command[6] == "/usr/bin/ffprobe":
                return subprocess.CompletedProcess(command, 0, b'{"streams":[]}', b"")
            return self._run(command, **kwargs)

        with patch("src.audiobook_mp3.windows_to_wsl_path", side_effect=lambda path, **_: str(path)):
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                export_mp3(self.wav, "神通者01", run=bad_probe)
        self.assertEqual(self.wav.read_bytes(), self.original)
        self.assertEqual(list(self.final.glob("*.mp3")), [])


if __name__ == "__main__":
    unittest.main()
