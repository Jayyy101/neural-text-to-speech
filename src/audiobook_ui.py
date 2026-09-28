"""Local audiobook generator; optional existing-run inspector for maintenance."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import unicodedata

from src.audiobook_application import inspect_run, open_audio_file
from src.audiobook_launcher import AudiobookLauncher
from src.audiobook import profiling


INVALID_WINDOWS_FILENAME_CHARACTERS = set('<>:"/\\|?*')
RESERVED_WINDOWS_NAMES = re.compile(r"^(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])$", re.I)


def normalize_wav_filename(raw):
    """Return one safe Windows filename, without a directory component."""
    name = unicodedata.normalize("NFC", raw.strip())
    if not name or any(
            char in INVALID_WINDOWS_FILENAME_CHARACTERS
            or unicodedata.category(char) in {"Cc", "Cf"}
            for char in name):
        raise ValueError("Enter a WAV filename without Windows-forbidden characters or paths.")
    if name.lower().endswith(".wav"):
        name = name[:-4]
    elif "." in name and not name.startswith("."):
        raise ValueError("The output filename must end in .wav.")
    if (not name or name.endswith((" ", "."))
            or RESERVED_WINDOWS_NAMES.fullmatch(name.split(".", 1)[0].rstrip(" ."))):
        raise ValueError("Choose a different WAV filename; this name is reserved by Windows.")
    filename = name + ".wav"
    if filename.lower() == "chapter.wav":
        raise ValueError("chapter.wav is reserved for the original audiobook. Choose another name.")
    if len(filename.encode("utf-16-le")) // 2 > 255:
        raise ValueError("The WAV filename is too long for Windows.")
    return filename


def export_wav(source_path, filename):
    """Copy a validated chapter WAV to an exclusive user-facing name."""
    source = Path(source_path)
    destination = source.parent / normalize_wav_filename(filename)
    created = False
    try:
        with source.open("rb") as original, destination.open("xb") as exported:
            created = True
            shutil.copyfileobj(original, exported, length=1024 * 1024)
        if destination.stat().st_size != source.stat().st_size:
            raise OSError("Exported WAV size does not match the original.")
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise
    return destination


def format_elapsed(seconds):
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def _audio_summary(audio):
    if not audio:
        return "unavailable"
    duration = audio.get("duration_seconds")
    duration_text = f"{duration:.3f} s" if isinstance(duration, (int, float)) else "unknown duration"
    return (
        f"{duration_text}; {audio.get('sample_rate_hz', '?')} Hz; "
        f"{audio.get('channels', '?')} channel(s); "
        f"{audio.get('frames', '?')} frames"
    )


def _attempt_text(attempt):
    seed = (
        "not recorded"
        if not attempt.seed_recorded
        else "model default (no explicit seed)"
        if attempt.seed is None
        else str(attempt.seed)
    )
    lines = [
        f"{attempt.id}{' [SELECTED]' if attempt.selected else ''}",
        f"  status: {attempt.status}",
        f"  seed: {seed}",
        f"  random policy: {attempt.random_policy or 'not recorded'}",
        f"  audio: {_audio_summary(attempt.audio)}",
        f"  artifact valid: {attempt.artifact_valid if attempt.artifact_valid is not None else 'not checked'}",
    ]
    if attempt.output_path:
        lines.append(f"  path: {attempt.output_path}")
    if attempt.duplicate_of_attempt_id:
        lines.append(f"  duplicate of: {attempt.duplicate_of_attempt_id}")
    if attempt.artifact_error:
        lines.append(f"  artifact error: {attempt.artifact_error}")
    if attempt.error:
        lines.append(f"  generation error:\n{_json(attempt.error)}")
    return "\n".join(lines)


class AudiobookInspector:
    def __init__(self, root, initial_directory=None):
        self.root = root
        self.root.title("Audiobook Generator")
        self.root.geometry("940x760" if initial_directory is None else "1100x960")
        self.root.minsize(720, 560)
        self.run = None
        self.run_directory = Path(initial_directory).resolve() if initial_directory else None
        self.launcher = AudiobookLauncher()
        self.job_active = False
        self.completed_run_directory = None
        self.inspection_queue = queue.Queue()
        self.inspection_thread = None
        self.launch_queue = queue.Queue()
        self.launch_pending = False
        self.started_at = None
        self.requested_filename = None
        self.request_job_id = None
        self._build()
        if self.run_directory is not None:
            self.refresh()

    def _build(self):
        style = ttk.Style(self.root)
        style.configure(".", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 19, "bold"))
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(18, 9))

        generation = ttk.Frame(self.root, padding=20)
        generation.pack(fill=tk.BOTH, expand=True)
        ttk.Label(generation, text="Audiobook Generator", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(generation, text="Paste one chapter and choose its WAV filename.").pack(
            anchor=tk.W, pady=(2, 18)
        )

        ttk.Label(generation, text="WAV filename", style="Section.TLabel").pack(anchor=tk.W)
        self.output_filename = tk.StringVar(value="audiobook.wav")
        self.filename_entry = ttk.Entry(generation, textvariable=self.output_filename)
        self.filename_entry.pack(fill=tk.X, pady=(5, 16))

        ttk.Label(generation, text="Chapter text", style="Section.TLabel").pack(anchor=tk.W)
        text_frame = ttk.Frame(generation)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 16))
        self.chapter_text = tk.Text(
            text_frame, height=20, wrap=tk.WORD, font=("Microsoft YaHei UI", 11),
            padx=10, pady=10, undo=True,
        )
        text_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.chapter_text.yview)
        self.chapter_text.configure(yscrollcommand=text_scroll.set)
        self.chapter_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        controls = ttk.Frame(generation)
        controls.pack(fill=tk.X)
        self.generate_button = ttk.Button(
            controls, text="Generate Audiobook", command=self.generate_audiobook,
            style="Primary.TButton",
        )
        self.generate_button.pack(side=tk.LEFT)
        self.open_folder_button = ttk.Button(
            controls, text="Open Folder", command=self.open_folder, state=tk.DISABLED
        )
        self.open_folder_button.pack(side=tk.LEFT, padx=(10, 0))

        progress = ttk.LabelFrame(generation, text="Progress", padding=12)
        progress.pack(fill=tk.X, pady=(20, 0))
        progress_heading = ttk.Frame(progress)
        progress_heading.pack(fill=tk.X)
        self.generation_status = ttk.Label(
            progress_heading, text="Ready", style="Status.TLabel", anchor=tk.W
        )
        self.generation_status.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.elapsed_time = tk.StringVar(value="Elapsed 00:00:00")
        ttk.Label(progress_heading, textvariable=self.elapsed_time).pack(side=tk.RIGHT)
        self.generation_progress = ttk.Progressbar(progress, mode="indeterminate")
        self.generation_progress.pack(fill=tk.X, pady=(10, 0))
        self.finished_wav = tk.StringVar()
        ttk.Label(generation, text="Completed WAV", style="Section.TLabel").pack(
            anchor=tk.W, pady=(18, 5)
        )
        ttk.Entry(generation, textvariable=self.finished_wav, state="readonly").pack(fill=tk.X)
        self.job_log = tk.StringVar()
        ttk.Label(generation, textvariable=self.job_log, anchor=tk.W).pack(
            fill=tk.X, pady=(8, 0)
        )

        if self.run_directory is not None:
            self._build_inspector()

    def _build_inspector(self):
        inspector = ttk.LabelFrame(self.root, text="Inspect Existing Run", padding=8)
        inspector.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        toolbar = ttk.Frame(inspector)
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="Choose Run Folder", command=self.choose_run).pack(side=tk.LEFT)
        self.refresh_button = ttk.Button(toolbar, text="Refresh", command=self.refresh, state=tk.DISABLED)
        self.refresh_button.pack(side=tk.LEFT, padx=(8, 0))
        self.run_path = ttk.Label(toolbar, text="No run open", anchor=tk.W)
        self.run_path.pack(side=tk.LEFT, padx=12, fill=tk.X, expand=True)

        summary = ttk.LabelFrame(inspector, text="Run", padding=8)
        summary.pack(fill=tk.X)
        self.summary_text = tk.Text(summary, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self.summary_text.pack(fill=tk.X)

        content = ttk.Panedwindow(inspector, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, pady=8)
        left = ttk.Frame(content)
        right = ttk.Frame(content)
        content.add(left, weight=1)
        content.add(right, weight=3)

        self.scene_tree = ttk.Treeview(
            left, columns=("status", "selected", "repair"), show="tree headings", selectmode="browse"
        )
        self.scene_tree.heading("#0", text="Scene")
        self.scene_tree.heading("status", text="Status")
        self.scene_tree.heading("selected", text="Selected attempt")
        self.scene_tree.heading("repair", text="Selected repair")
        self.scene_tree.column("#0", width=95)
        self.scene_tree.column("status", width=95)
        self.scene_tree.column("selected", width=115)
        self.scene_tree.column("repair", width=105)
        self.scene_tree.pack(fill=tk.BOTH, expand=True)
        self.scene_tree.bind("<<TreeviewSelect>>", self.show_selected_scene)

        self.scene_details = tk.Text(right, wrap=tk.WORD, state=tk.DISABLED)
        details_scroll = ttk.Scrollbar(right, orient=tk.VERTICAL, command=self.scene_details.yview)
        self.scene_details.configure(yscrollcommand=details_scroll.set)
        self.scene_details.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        actions = ttk.Frame(inspector)
        actions.pack(fill=tk.X)
        self.open_scene_button = ttk.Button(
            actions, text="Open Selected Scene Audio", command=self.open_scene, state=tk.DISABLED
        )
        self.open_scene_button.pack(side=tk.LEFT)
        self.open_chapter_button = ttk.Button(
            actions, text="Open Final Chapter Audio", command=self.open_chapter, state=tk.DISABLED
        )
        self.open_chapter_button.pack(side=tk.LEFT, padx=(8, 0))
        self.status = ttk.Label(actions, text="Ready", anchor=tk.W)
        self.status.pack(side=tk.LEFT, padx=12, fill=tk.X, expand=True)

    def _set_generation_status(self, status, detail=""):
        self.generation_status.configure(text=status)
        self.job_log.set(detail)

    def _update_elapsed(self):
        if self.started_at is not None:
            self.elapsed_time.set(f"Elapsed {format_elapsed(time.monotonic() - self.started_at)}")

    def _finish_generation(self, status, detail=""):
        self._update_elapsed()
        self.job_active = False
        self.generation_progress.stop()
        self.generate_button.configure(state=tk.NORMAL)
        self.filename_entry.configure(state=tk.NORMAL)
        self._set_generation_status(status, detail)
        profile_job = getattr(self, "profile_job", None)
        if profile_job is not None:
            profile_job.finish("ok" if status == "Complete" else "error",
                               status=status)
            self.profile_job = None
            profiling.flush()

    @staticmethod
    def _record_ui_failure(log_path, error):
        try:
            with Path(log_path).open("ab") as log:
                log.write(f"\nUI error: {type(error).__name__}: {error}\n".encode("utf-8"))
        except OSError:
            return False
        return True

    def generate_audiobook(self):
        if self.job_active:
            return
        chapter = self.chapter_text.get("1.0", "end-1c")
        if not chapter.strip():
            messagebox.showwarning("Chapter text required", "Paste chapter text before generating.")
            return
        try:
            filename = normalize_wav_filename(self.output_filename.get())
        except ValueError as error:
            messagebox.showwarning("Invalid WAV filename", str(error))
            return
        self.output_filename.set(filename)
        job_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + "_" + secrets.token_hex(4)
        request_dir = self.launcher.repository_root / "outputs" / "ui_requests" / job_id
        source_path = request_dir / "source.txt"
        log_path = request_dir / "worker.log"
        profiling.activate(request_dir / "profile" if os.environ.get("TTS_PROFILE") == "1"
                           else None)
        self.profile_job = profiling.begin("ui.generate_to_complete", job_id=job_id)
        self.requested_filename = filename
        self.request_job_id = job_id
        self.started_at = time.monotonic()
        self._update_elapsed()
        self.completed_run_directory = None
        self.finished_wav.set("")
        self.open_folder_button.configure(state=tk.DISABLED)
        self.job_active = True
        self.generate_button.configure(state=tk.DISABLED)
        self.filename_entry.configure(state=tk.DISABLED)
        self.generation_progress.configure(mode="indeterminate")
        self.generation_progress.start()
        self._set_generation_status("Preparing...")
        try:
            source_bytes = chapter.encode("utf-8")
            with profiling.span("ui.source_write", source_bytes=len(source_bytes)):
                request_dir.mkdir(parents=True)
                source_path.write_bytes(source_bytes)
        except (OSError, RuntimeError, ValueError) as error:
            logged = self._record_ui_failure(log_path, error)
            detail = (f"Could not start. Worker log: {log_path}" if logged else
                      "Could not start. Check WSL and output folder access.")
            self._finish_generation("Failed", detail)
            return
        self.launch_queue = queue.Queue()
        self.launch_pending = True
        self.inspection_queue = queue.Queue()
        self.inspection_thread = None
        self._start_launcher(source_path, job_id, log_path)
        self._poll_generation()

    def _start_launcher(self, source_path, job_id, log_path):
        destination = self.launch_queue

        def launch():
            try:
                profile_job = getattr(self, "profile_job", None)
                with profiling.span("ui.launch", parent_id=(
                        profile_job.span_id if profile_job is not None else None)):
                    self.launcher.start(source_path, "chapter_" + job_id,
                                        "run_" + job_id, log_path)
            except (OSError, RuntimeError, ValueError) as error:
                destination.put(error)
            else:
                destination.put(None)

        threading.Thread(target=launch, daemon=True).start()

    def _start_inspection(self, kind):
        destination = self.inspection_queue
        run_directory = self.launcher.run_directory
        filename = self.requested_filename

        def inspect():
            result = None
            try:
                name = "ui.final_validation" if kind == "final" else "ui.progress_inspection"
                profile_job = getattr(self, "profile_job", None)
                parent_id = profile_job.span_id if profile_job is not None else None
                with profiling.span(name, parent_id=parent_id):
                    result = inspect_run(run_directory)
                if kind == "final" and result.assembly.playable:
                    with profiling.span("ui.named_file_copy", parent_id=parent_id):
                        exported = export_wav(result.assembly.audio_path, filename)
                    profiling.mark("ui.export_ready", job_id=self.request_job_id)
                else:
                    exported = None
            except Exception as error:
                destination.put((kind, result, None, error))
            else:
                destination.put((kind, result, exported, None))

        self.inspection_thread = threading.Thread(target=inspect, daemon=True)
        self.inspection_thread.start()

    def _show_progress(self, inspected):
        if inspected is None or inspected.total_units is None:
            self._set_generation_status("Preparing...")
            return
        total = inspected.total_units
        selected = inspected.selected_units
        if self.generation_progress.cget("mode") != "determinate":
            self.generation_progress.stop()
            self.generation_progress.configure(mode="determinate", maximum=total)
        self.generation_progress.configure(value=selected)
        if selected == total:
            self._set_generation_status("Assembling...")
        else:
            self._set_generation_status(f"Generating {selected} / {total}")

    def _poll_generation(self):
        if not self.job_active:
            return
        self._update_elapsed()
        if self.launch_pending:
            try:
                launch_error = self.launch_queue.get_nowait()
            except queue.Empty:
                self.root.after(1000, self._poll_generation)
                return
            self.launch_pending = False
            if launch_error is not None:
                log_path = self.launcher.repository_root / "outputs" / "ui_requests"
                log_path = log_path / self.request_job_id / "worker.log"
                logged = self._record_ui_failure(log_path, launch_error)
                detail = (f"Could not start. Worker log: {log_path}" if logged else
                          "Could not start. Check WSL and output folder access.")
                self._finish_generation("Failed", detail)
                return
        exit_code = self.launcher.poll()
        if exit_code is not None and exit_code != 0:
            self._finish_generation("Failed", f"Generation stopped. Worker log: {self.launcher.log_path}")
            return
        try:
            kind, inspected, exported, error = self.inspection_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self.inspection_thread = None
            if exit_code is None and error is None:
                self._show_progress(inspected)
            elif exit_code == 0 and kind == "final":
                if error is None and inspected.assembly.playable and exported is not None:
                    self.completed_run_directory = inspected.run_directory
                    self.finished_wav.set(str(exported))
                    self.open_folder_button.configure(state=tk.NORMAL)
                    self._finish_generation("Complete")
                else:
                    validated = inspected is not None and inspected.assembly.playable
                    if validated:
                        self.completed_run_directory = inspected.run_directory
                        self.finished_wav.set(str(inspected.assembly.audio_path))
                        self.open_folder_button.configure(state=tk.NORMAL)
                    reason = error or RuntimeError(
                        getattr(inspected.assembly, "error", None) or "Final WAV is unavailable."
                    )
                    self._record_ui_failure(self.launcher.log_path, reason)
                    if validated and isinstance(error, FileExistsError):
                        detail = (f"{self.requested_filename} already exists. The original WAV is "
                                  "available through Open Folder.")
                    elif validated:
                        detail = ("Could not save the requested WAV. The original is available "
                                  f"through Open Folder. Worker log: {self.launcher.log_path}")
                    else:
                        detail = f"Finished WAV could not be validated. Worker log: {self.launcher.log_path}"
                    self._finish_generation("Failed", detail)
                return
        if self.inspection_thread is None:
            self._start_inspection("final" if exit_code == 0 else "progress")
        if exit_code == 0:
            self._set_generation_status("Assembling...")
        self.root.after(1000, self._poll_generation)

    def open_folder(self):
        if self.completed_run_directory is None:
            return
        try:
            os.startfile(self.completed_run_directory / "final")  # type: ignore[attr-defined]
        except OSError as error:
            messagebox.showerror("Cannot open folder", str(error))

    @staticmethod
    def _set_text(widget, value):
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)
        widget.configure(state=tk.DISABLED)

    def choose_run(self):
        selected = filedialog.askdirectory(title="Choose audiobook run folder")
        if selected:
            self.run_directory = Path(selected).resolve()
            self.refresh()

    def refresh(self):
        if self.run_directory is None:
            return
        try:
            inspected = inspect_run(self.run_directory)
        except Exception as error:
            self.run = None
            self.refresh_button.configure(state=tk.NORMAL)
            self.open_scene_button.configure(state=tk.DISABLED)
            self.open_chapter_button.configure(state=tk.DISABLED)
            self.run_path.configure(text=str(self.run_directory))
            self._set_text(self.summary_text, f"Unable to inspect run:\n{type(error).__name__}: {error}")
            self._set_text(self.scene_details, "")
            for item in self.scene_tree.get_children():
                self.scene_tree.delete(item)
            self.status.configure(text="Run is missing, malformed, or invalid.")
            return

        self.run = inspected
        self.run_directory = inspected.run_directory
        self.run_path.configure(text=str(inspected.run_directory))
        self.refresh_button.configure(state=tk.NORMAL)
        operation = _json(inspected.latest_operation) if inspected.latest_operation else "none recorded"
        assembly = inspected.assembly
        summary = (
            f"Chapter: {inspected.chapter_id}    Run: {inspected.run_id}    Schema: {inspected.schema_version}\n"
            f"Created: {inspected.created_at_utc}\n"
            f"Source: {inspected.source_path} ({inspected.source.get('byte_length', '?')} bytes)\n"
            f"Run status: {inspected.run_status}    Generation status: {inspected.generation_status}\n"
            f"Assembly status: {assembly.status}    Final audio playable: {'yes' if assembly.playable else 'no'}"
        )
        if assembly.error:
            summary += f"\nAssembly issue: {assembly.error}"
        summary += f"\nLatest operation: {operation}"
        self._set_text(self.summary_text, summary)

        for item in self.scene_tree.get_children():
            self.scene_tree.delete(item)
        for scene in inspected.scenes:
            self.scene_tree.insert(
                "", tk.END, iid=scene.id, text=scene.id,
                values=(scene.generation_status, scene.selected_attempt_id or "—", scene.selected_repair_id or "—"),
            )
        self.open_chapter_button.configure(state=tk.NORMAL if assembly.playable else tk.DISABLED)
        self.open_scene_button.configure(state=tk.DISABLED)
        self._set_text(self.scene_details, "Select a scene to inspect its text and artifacts.")
        self.status.configure(text=f"Loaded {len(inspected.scenes)} scene(s).")

    def selected_scene(self):
        if self.run is None or not self.scene_tree.selection():
            return None
        scene_id = self.scene_tree.selection()[0]
        return next((scene for scene in self.run.scenes if scene.id == scene_id), None)

    def show_selected_scene(self, _event=None):
        scene = self.selected_scene()
        if scene is None:
            return
        attempts = "\n\n".join(_attempt_text(attempt) for attempt in scene.attempts) or "none"
        repair = _json(scene.selected_repair) if scene.selected_repair else "none"
        resolved = (
            f"{scene.resolved_artifact_type} {scene.resolved_artifact_id}: {scene.resolved_audio_path}"
            if scene.resolved_audio_path else f"unavailable: {scene.resolution_error}"
        )
        details = (
            f"{scene.id} (order {scene.order})\n"
            f"Source span:\n{_json(scene.source_span)}\n\n"
            f"Narration text:\n{scene.text}\n\n"
            f"Generation status: {scene.generation_status}\n"
            f"Selected attempt: {scene.selected_attempt_id or 'none'}\n"
            f"Selected repair: {scene.selected_repair_id or 'none'}\n"
            f"Resolved selected audio: {resolved}\n\n"
            f"Attempts:\n{attempts}\n\n"
            f"Selected repair record:\n{repair}"
        )
        self._set_text(self.scene_details, details)
        self.open_scene_button.configure(
            state=tk.NORMAL if scene.resolved_audio_path is not None else tk.DISABLED
        )

    def _open(self, path):
        try:
            open_audio_file(path)
            self.status.configure(text=f"Opened {path}")
        except Exception as error:
            messagebox.showerror("Cannot open audio", str(error))
            self.status.configure(text="Audio could not be opened. Refresh to revalidate the run.")

    def open_scene(self):
        scene = self.selected_scene()
        if scene and scene.resolved_audio_path:
            self._open(scene.resolved_audio_path)

    def open_chapter(self):
        if self.run and self.run.assembly.playable and self.run.assembly.audio_path:
            self._open(self.run.assembly.audio_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", nargs="?", type=Path)
    args = parser.parse_args(argv)
    root = tk.Tk()
    AudiobookInspector(root, args.run_directory)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
