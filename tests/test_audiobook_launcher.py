"""Model-free Windows/WSL launcher boundary checks."""

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.audiobook_launcher import AudiobookLauncher, windows_to_wsl_path


def mapped_path(args, **kwargs):
    assert kwargs["check"] is False
    assert args[:7] == [
        "wsl.exe", "--distribution", "Ubuntu-22.04", "--user", "jay",
        "--exec", "wslpath",
    ]
    assert args[7] == "-u"
    windows = args[8]
    linux = "/mnt/c/" + windows[3:].replace("\\", "/")
    return subprocess.CompletedProcess(args, 0, linux.encode("utf-8") + b"\n", b"")


class LauncherTests(unittest.TestCase):
    def test_windows_mapping_preserves_spaces_and_unicode(self):
        path = r"C:\Users\Jay Ma\章节\source text.txt"
        self.assertEqual(windows_to_wsl_path(path, run=mapped_path),
                         "/mnt/c/Users/Jay Ma/章节/source text.txt")
        for unsafe in ("chapter.txt", r"\\server\share\chapter.txt", "/tmp/chapter.txt"):
            with self.subTest(path=unsafe), self.assertRaises(ValueError):
                windows_to_wsl_path(unsafe, run=mapped_path)

    def test_mapping_failure_and_unexpected_output(self):
        failed = Mock(return_value=subprocess.CompletedProcess([], 1, b"", b"no distro"))
        with self.assertRaisesRegex(RuntimeError, "no distro"):
            windows_to_wsl_path(r"C:\input.txt", run=failed)
        invalid = Mock(return_value=subprocess.CompletedProcess([], 0, b"/home/jay/input.txt\n", b""))
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            windows_to_wsl_path(r"C:\input.txt", run=invalid)

    def test_wsl_path_conversion_hides_console_and_keeps_error_output(self):
        failed = Mock(return_value=subprocess.CompletedProcess([], 1, b"", b"mapping failed"))
        with patch.object(subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True):
            with self.assertRaisesRegex(RuntimeError, "mapping failed"):
                windows_to_wsl_path(r"C:\input.txt", run=failed)
        call = failed.call_args
        self.assertIsInstance(call.args[0], list)
        self.assertEqual(call.kwargs["creationflags"], 0x08000000)
        self.assertEqual(call.kwargs["stdout"], subprocess.PIPE)
        self.assertEqual(call.kwargs["stderr"], subprocess.PIPE)

    def test_command_and_single_active_job(self):
        repo = r"C:\Users\Jay Ma\OneDrive\Documents\TTS_Project"
        source = r"C:\Users\Jay Ma\chapters\神通者 03.txt"
        launcher = AudiobookLauncher(
            repository_root=repo,
            run=mapped_path, popen=Mock(), platform="win32",
        )
        self.assertEqual(launcher.output_root,
                         Path(r"C:\Users\Jay Ma\TTS_Audiobooks") / "audiobooks")
        self.assertEqual(launcher.request_root,
                         Path(r"C:\Users\Jay Ma\TTS_Audiobooks") / "ui_requests")
        expected = launcher.command(source, "chapter_03", "run_001")
        self.assertEqual(expected, [
            "wsl.exe", "--distribution", "Ubuntu-22.04", "--user", "jay",
            "--cd", "/mnt/c/Users/Jay Ma/OneDrive/Documents/TTS_Project",
            "--exec", "/home/jay/miniconda3/envs/cosyvoice-b/bin/python",
            "-B", "-m", "src.audiobook", "run",
            "/mnt/c/Users/Jay Ma/chapters/神通者 03.txt",
            "--chapter-id", "chapter_03", "--run-id", "run_001",
            "--output-root", "/mnt/c/Users/Jay Ma/TTS_Audiobooks/audiobooks",
        ])
        with self.assertRaises(ValueError):
            launcher.command(source, "../bad", "run_001")

        process = Mock()
        process.poll.side_effect = [None, None, 0]
        launcher._popen.return_value = process
        with tempfile.TemporaryDirectory() as folder:
            log_path = Path(folder) / "job.log"
            with patch.object(subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True):
                with patch("src.audiobook_launcher.Path.is_file", return_value=True):
                    run_dir = launcher.start(source, "chapter_03", "run_001", log_path)
                    self.assertEqual(run_dir, launcher.run_directory)
                    self.assertEqual(launcher.log_path, log_path)
                    self.assertTrue(log_path.is_file())
                    self.assertIsNone(launcher.poll())
                    with self.assertRaisesRegex(RuntimeError, "already running"):
                        launcher.start(source, "chapter_04", "run_002", Path(folder) / "other.log")
            self.assertEqual(launcher.poll(), 0)
            call = launcher._popen.call_args
            self.assertEqual(call.args[0], expected)
            self.assertFalse(call.kwargs["shell"])
            self.assertEqual(call.kwargs["creationflags"], 0x08000000)
            self.assertEqual(call.kwargs["stderr"], subprocess.STDOUT)
            self.assertTrue(call.kwargs["stdout"].closed)

    def test_start_rejects_missing_source_and_non_windows_host(self):
        launcher = AudiobookLauncher(platform="linux")
        with self.assertRaisesRegex(OSError, "native Windows"):
            launcher.start("missing", "chapter", "run", "job.log")
        launcher = AudiobookLauncher(platform="win32")
        with self.assertRaises(FileNotFoundError):
            launcher.start("missing", "chapter", "run", "job.log")


if __name__ == "__main__":
    unittest.main()
