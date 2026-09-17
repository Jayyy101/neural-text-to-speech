"""Tkinter read-only inspector for existing Milestone D audiobook runs."""

import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.audiobook_application import inspect_run, open_audio_file


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
        self.root.title("Audiobook Run Inspector")
        self.root.geometry("1100x760")
        self.run = None
        self.run_directory = Path(initial_directory).resolve() if initial_directory else None
        self._build()
        if self.run_directory is not None:
            self.refresh()

    def _build(self):
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="Open Run Folder", command=self.choose_run).pack(side=tk.LEFT)
        self.refresh_button = ttk.Button(toolbar, text="Refresh", command=self.refresh, state=tk.DISABLED)
        self.refresh_button.pack(side=tk.LEFT, padx=(8, 0))
        self.run_path = ttk.Label(toolbar, text="No run open", anchor=tk.W)
        self.run_path.pack(side=tk.LEFT, padx=12, fill=tk.X, expand=True)

        summary = ttk.LabelFrame(self.root, text="Run", padding=8)
        summary.pack(fill=tk.X, padx=8)
        self.summary_text = tk.Text(summary, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self.summary_text.pack(fill=tk.X)

        content = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
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

        actions = ttk.Frame(self.root, padding=(8, 0, 8, 8))
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