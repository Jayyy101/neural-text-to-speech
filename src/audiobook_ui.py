"""Local audiobook generator; optional existing-run inspector for maintenance."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import secrets
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.audiobook_application import inspect_run, open_audio_file
from src.audiobook_launcher import AudiobookLauncher
from src.audiobook_mp3 import export_mp3, normalize_mp3_filename
from src.audiobook import profiling


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
        self.root.title("Local Mandarin Audiobook Generator")
        scale = float(self.root.tk.call("tk", "scaling"))
        # Keep the fixed controls visible even when Windows uses larger text scaling.
        minimum_height = max(620, round(365 + 175 * scale))
        self.root.minsize(720, minimum_height)
        if initial_directory is None:
            width = max(720, min(960, self.root.winfo_screenwidth() - 80))
            height = max(minimum_height, min(760, self.root.winfo_screenheight() - 100))
            self.root.geometry(f"{width}x{height}")
        else:
            self.root.geometry("1100x960")
        self.run = None
        self.run_directory = Path(initial_directory).resolve() if initial_directory else None
        self.launcher = AudiobookLauncher()
        self.job_active = False
        self.completed_run_directory = None
        self.open_folder_target = None
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
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#5b6871")
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 11, "bold"),
                        foreground="#254d60")
        style.configure("Secondary.TLabel", font=("Segoe UI", 9), foreground="#5b6871")
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"),
                        foreground="#1b5267", padding=(20, 10))

        generation = ttk.Frame(self.root, padding=(24, 14))
        generation.pack(fill=tk.BOTH, expand=True)
        generation.columnconfigure(0, weight=1)
        generation.rowconfigure(5, weight=1, minsize=100)
        ttk.Label(generation, text="Local Mandarin Audiobook Generator",
                  style="Title.TLabel").grid(
            row=0, column=0, sticky=tk.W,
        )
        ttk.Label(generation, text="Local CosyVoice3 audiobook generation",
                  style="Subtitle.TLabel").grid(
            row=1, column=0, sticky=tk.W, pady=(2, 16),
        )

        ttk.Label(generation, text="MP3 filename", style="Section.TLabel").grid(
            row=2, column=0, sticky=tk.W,
        )
        self.output_filename = tk.StringVar(value="audiobook.mp3")
        self.filename_entry = ttk.Entry(generation, textvariable=self.output_filename)
        self.filename_entry.grid(row=3, column=0, sticky=tk.EW, pady=(6, 14))

        ttk.Label(generation, text="Chapter text", style="Section.TLabel").grid(
            row=4, column=0, sticky=tk.W,
        )
        self.text_frame = ttk.Frame(generation)
        self.text_frame.columnconfigure(0, weight=1)
        self.text_frame.rowconfigure(0, weight=1)
        self.text_frame.grid(row=5, column=0, sticky=tk.NSEW, pady=(7, 14))
        self.chapter_text = tk.Text(
            self.text_frame, height=8, wrap=tk.WORD, font=("Microsoft YaHei UI", 11),
            padx=12, pady=10, undo=True, relief=tk.FLAT, borderwidth=0,
            highlightthickness=1, highlightbackground="#c9d2d8",
            highlightcolor="#4e798b",
        )
        text_scroll = ttk.Scrollbar(self.text_frame, orient=tk.VERTICAL,
                                    command=self.chapter_text.yview)
        self.chapter_text.configure(yscrollcommand=text_scroll.set)
        self.chapter_text.grid(row=0, column=0, sticky=tk.NSEW)
        text_scroll.grid(row=0, column=1, sticky=tk.NS)

        controls = ttk.Frame(generation)
        controls.grid(row=6, column=0, sticky=tk.EW)
        self.generate_button = ttk.Button(
            controls, text="Generate Audiobook", command=self.generate_audiobook,
            style="Primary.TButton",
        )
        self.generate_button.pack(side=tk.LEFT)

        progress = ttk.LabelFrame(generation, text="Generation progress", padding=12)
        progress.grid(row=7, column=0, sticky=tk.EW, pady=(16, 0))
        self.generation_progress = ttk.Progressbar(progress, mode="indeterminate")
        self.generation_progress.pack(fill=tk.X)
        progress_heading = ttk.Frame(progress)
        progress_heading.pack(fill=tk.X, pady=(10, 0))
        self.generation_status = ttk.Label(
            progress_heading, text="Ready", style="Status.TLabel", anchor=tk.W
        )
        self.generation_status.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.elapsed_time = tk.StringVar(value="Elapsed 00:00:00")
        self.elapsed_label = ttk.Label(progress_heading, textvariable=self.elapsed_time,
                                       style="Secondary.TLabel")
        self.elapsed_label.pack(side=tk.RIGHT)
        self.finished_mp3 = tk.StringVar()
        ttk.Label(generation, text="Completed MP3", style="Section.TLabel").grid(
            row=8, column=0, sticky=tk.W, pady=(15, 5),
        )
        self.completed_mp3_entry = ttk.Entry(
            generation, textvariable=self.finished_mp3, state="readonly",
            font=("Segoe UI", 9),
        )
        self.completed_mp3_entry.grid(
            row=9, column=0, sticky=tk.EW,
        )
        self.open_folder_button = ttk.Button(
            generation, text="Open Folder", command=self.open_folder, state=tk.DISABLED,
        )
        self.open_folder_button.grid(row=10, column=0, sticky=tk.W, pady=(10, 0))
        self.job_log = tk.StringVar()
        ttk.Label(generation, textvariable=self.job_log, anchor=tk.W,
                  style="Secondary.TLabel").grid(
            row=11, column=0, sticky=tk.EW, pady=(8, 0),
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
        if status == "Failed" and self.completed_run_directory is None:
            run_directory = getattr(self.launcher, "run_directory", None)
            if isinstance(run_directory, (str, Path)) and Path(run_directory).is_dir():
                self.completed_run_directory = Path(run_directory)
                self.open_folder_target = Path(run_directory)
                self.open_folder_button.configure(state=tk.NORMAL)
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
            filename = normalize_mp3_filename(self.output_filename.get())
        except ValueError as error:
            messagebox.showwarning("Invalid MP3 filename", str(error))
            return
        self.output_filename.set(filename)
        job_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + "_" + secrets.token_hex(4)
        request_dir = self.launcher.request_root / job_id
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
        self.open_folder_target = None
        self.finished_mp3.set("")
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
                        exported = export_mp3(result.assembly.audio_path, filename)
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
                log_path = self.launcher.request_root / self.request_job_id / "worker.log"
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
                    self.open_folder_target = inspected.run_directory / "final"
                    self.finished_mp3.set(str(exported))
                    self.open_folder_button.configure(state=tk.NORMAL)
                    self._finish_generation("Complete")
                else:
                    validated = inspected is not None and inspected.assembly.playable
                    if validated:
                        self.completed_run_directory = inspected.run_directory
                        self.open_folder_target = inspected.run_directory / "final"
                        self.open_folder_button.configure(state=tk.NORMAL)
                    assembly = getattr(inspected, "assembly", None)
                    reason = error or RuntimeError(
                        getattr(assembly, "error", None) or "Final WAV is unavailable."
                    )
                    self._record_ui_failure(self.launcher.log_path, reason)
                    if validated and isinstance(error, FileExistsError):
                        detail = (f"MP3 export failed: {self.requested_filename} already exists. "
                                  "The original WAV is available through Open Folder.")
                    elif validated:
                        detail = ("MP3 export failed. The original WAV is available "
                                  f"through Open Folder. Worker log: {self.launcher.log_path}")
                    else:
                        detail = f"Finished WAV could not be validated. Worker log: {self.launcher.log_path}"
                    self._finish_generation("Failed", detail)
                return
        if self.inspection_thread is None:
            self._start_inspection("final" if exit_code == 0 else "progress")
        if exit_code == 0:
            self._set_generation_status("Validating WAV and exporting MP3...")
        self.root.after(1000, self._poll_generation)

    def open_folder(self):
        if self.completed_run_directory is None:
            return
        try:
            target = (getattr(self, "open_folder_target", None)
                      or self.completed_run_directory / "final")
            os.startfile(target)  # type: ignore[attr-defined]
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
