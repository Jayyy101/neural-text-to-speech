"""Native Windows boundary for the existing WSL audiobook CLI."""

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess
import sys

from .audiobook.planning import validate_id
from .audiobook import profiling


DEFAULT_DISTRIBUTION = "Ubuntu-22.04"
DEFAULT_USER = "jay"
DEFAULT_PYTHON = "/home/jay/miniconda3/envs/cosyvoice-b/bin/python"


def windows_to_wsl_path(path, *, distribution=DEFAULT_DISTRIBUTION,
                        user=DEFAULT_USER, run=subprocess.run):
    """Ask the target WSL distribution to map an absolute local drive path."""
    windows = PureWindowsPath(path)
    if (not windows.is_absolute() or not windows.drive.endswith(":")
            or windows.root != "\\"):
        raise ValueError("Expected an absolute Windows drive path.")
    with profiling.span("launcher.wslpath", path_role=windows.name):
        result = run(
            ["wsl.exe", "--distribution", distribution, "--user", user,
             "--exec", "wslpath", "-u", str(windows)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False,
        )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"WSL path conversion failed: {detail or result.returncode}")
    mapped = result.stdout.decode("utf-8", errors="replace").rstrip("\r\n")
    linux = PurePosixPath(mapped)
    if (not linux.is_absolute() or "\n" in mapped or "\r" in mapped
            or linux.parts[:2] != ("/", "mnt")
            or len(linux.parts) < 4
            or linux.parts[2].lower() != windows.drive[0].lower()):
        raise RuntimeError("WSL returned an unexpected Windows path mapping.")
    return mapped


class AudiobookLauncher:
    """Launch at most one active production job from native Windows Python."""

    def __init__(self, repository_root=None, output_root=None, *,
                 distribution=DEFAULT_DISTRIBUTION, user=DEFAULT_USER,
                 python=DEFAULT_PYTHON, run=subprocess.run,
                 popen=subprocess.Popen, platform=sys.platform):
        self.repository_root = Path(repository_root or Path(__file__).resolve().parents[1])
        self.output_root = Path(output_root or self.repository_root / "outputs/audiobooks")
        self.distribution = distribution
        self.user = user
        self.python = python
        self._run = run
        self._popen = popen
        self._platform = platform
        self._process = None
        self._log = None
        self.run_directory = None
        self.log_path = None

    def command(self, source_path, chapter_id, run_id):
        """Build the exact production command without launching or writing files."""
        with profiling.span("launcher.command_preparation"):
            return self._command(source_path, chapter_id, run_id)

    def _command(self, source_path, chapter_id, run_id):
        validate_id("chapter_id", chapter_id)
        validate_id("run_id", run_id)
        repo = windows_to_wsl_path(
            self.repository_root, distribution=self.distribution,
            user=self.user, run=self._run,
        )
        source = windows_to_wsl_path(
            source_path, distribution=self.distribution,
            user=self.user, run=self._run,
        )
        output = windows_to_wsl_path(
            self.output_root, distribution=self.distribution,
            user=self.user, run=self._run,
        )
        executable = [self.python]
        if profiling.enabled():
            profile_dir = PurePosixPath(source).parent / "profile"
            executable = ["/usr/bin/env", "TTS_PROFILE=1",
                          f"TTS_PROFILE_DIR={profile_dir}", self.python]
        return [
            "wsl.exe", "--distribution", self.distribution,
            "--user", self.user, "--cd", repo, "--exec", *executable,
            "-B", "-m", "src.audiobook", "run", source,
            "--chapter-id", chapter_id, "--run-id", run_id,
            "--output-root", output,
        ]

    def poll(self):
        """Return None while running, or the CLI exit code once finished."""
        if self._process is None:
            return None
        with profiling.span("launcher.poll"):
            code = self._process.poll()
        if code is not None and self._log is not None:
            profiling.mark("launcher.worker_exit_observed", exit_code=code)
            self._log.close()
            self._log = None
        return code

    def start(self, source_path, chapter_id, run_id, log_path):
        """Start one CLI job; the caller owns its UTF-8 source text file."""
        if self._platform != "win32":
            raise OSError("The audiobook launcher requires native Windows Python.")
        if self._process is not None and self.poll() is None:
            raise RuntimeError("An audiobook job is already running.")
        if not Path(source_path).is_file():
            raise FileNotFoundError(source_path)
        command = self.command(source_path, chapter_id, run_id)
        run_directory = self.output_root / chapter_id / run_id
        if run_directory.exists():
            raise FileExistsError(run_directory)
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # WSL writes UTF-8 bytes; leave them unchanged for later UTF-8 reads.
        log = log_path.open("xb")
        process = None
        try:
            with profiling.span("launcher.process_launch"):
                process = self._popen(
                    command, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, shell=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                # Own the child before tracing can run its exit hook.
                self._process = process
                self._log = log
                self.run_directory = run_directory
                self.log_path = log_path
        except Exception:
            if process is None:
                log.close()
                raise
            # Popen succeeded; only the profiling exit hook remains here.
        return run_directory
