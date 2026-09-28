"""Model-free checks for the native Windows generation controls."""

from pathlib import Path
import queue
import sys
import tempfile
import tkinter as tk
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.audiobook_ui import (
    AudiobookInspector, format_elapsed, normalize_mp3_filename,
)


class GenerationUiTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "requires native Windows Tk")
    def test_default_layout_keeps_workflow_visible_and_text_resizes(self):
        root = tk.Tk()
        try:
            view = AudiobookInspector(root)
            root.update()
            self.assertLessEqual(root.winfo_height(), root.winfo_screenheight() - 100)
            controls = [view.filename_entry, view.chapter_text,
                        view.generate_button, view.open_folder_button,
                        view.generation_progress, view.generation_status,
                        view.elapsed_label, view.completed_mp3_entry]

            def assert_visible():
                top = root.winfo_rooty()
                bottom = top + root.winfo_height()
                for widget in controls:
                    with self.subTest(widget=widget):
                        self.assertGreater(widget.winfo_height(), 1)
                        self.assertGreaterEqual(widget.winfo_rooty(), top)
                        self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(), bottom)

            assert_visible()
            self.assertGreaterEqual(view.open_folder_button.winfo_rooty(),
                                    view.completed_mp3_entry.winfo_rooty()
                                    + view.completed_mp3_entry.winfo_height())
            root.geometry("720x620")
            root.update()
            assert_visible()
            short_text_height = view.chapter_text.winfo_height()
            root.geometry("720x740")
            root.update()
            self.assertGreater(view.chapter_text.winfo_height(), short_text_height)
            assert_visible()
        finally:
            root.destroy()

    @unittest.skipUnless(sys.platform == "win32", "requires native Windows Tk")
    def test_scaled_layout_retains_progress_and_output_at_minimum_size(self):
        root = tk.Tk()
        try:
            root.tk.call("tk", "scaling", 2.0)
            view = AudiobookInspector(root)
            minimum_height = root.minsize()[1]
            root.geometry(f"720x{minimum_height}")
            root.update()
            self.assertGreater(view.chapter_text.winfo_height(), 80)
            bottom = root.winfo_rooty() + root.winfo_height()
            for widget in (view.generate_button, view.generation_progress,
                           view.generation_status, view.elapsed_label,
                           view.completed_mp3_entry, view.open_folder_button):
                self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(), bottom)
        finally:
            root.destroy()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.view = AudiobookInspector.__new__(AudiobookInspector)
        view = self.view
        view.root = Mock()
        view.chapter_text = Mock()
        view.generate_button = Mock()
        view.filename_entry = Mock()
        view.output_filename = Mock()
        view.output_filename.get.return_value = "神通者04.mp3"
        view.open_folder_button = Mock()
        view.generation_status = Mock()
        view.generation_progress = Mock()
        view.generation_progress.cget.return_value = "indeterminate"
        view.elapsed_time = Mock()
        view.finished_mp3 = Mock()
        view.job_log = Mock()
        view.launcher = Mock()
        view.launcher.repository_root = self.root
        view.launcher.request_root = self.root / "ui_requests"
        view.launcher.log_path = self.root / "worker.log"
        view.job_active = False
        view.started_at = None
        view.requested_filename = "神通者04.mp3"
        view.request_job_id = None
        view.launch_pending = False
        view.launch_queue = queue.Queue()
        view.completed_run_directory = None
        view.inspection_queue = queue.Queue()
        view.inspection_thread = None

    def test_empty_input_does_not_create_a_job(self):
        self.view.chapter_text.get.return_value = " \n\t "
        with patch("src.audiobook_ui.messagebox.showwarning") as warning:
            self.view.generate_audiobook()
        warning.assert_called_once()
        self.view.launcher.start.assert_not_called()
        self.assertFalse(self.view.launcher.request_root.exists())

    def test_generate_saves_exact_text_and_starts_one_job(self):
        chapter = "  神通者\r\n第一段。\n\n "
        self.view.chapter_text.get.return_value = chapter
        self.view._poll_generation = Mock()

        def launch_now(source, job_id, log):
            self.view.launcher.start(source, "chapter_" + job_id, "run_" + job_id, log)
            self.view.launch_queue.put(None)

        self.view._start_launcher = launch_now
        self.view.generate_audiobook()
        source, chapter_id, run_id, log = self.view.launcher.start.call_args.args
        self.assertEqual(source.read_bytes(), chapter.encode("utf-8"))
        self.assertEqual(source.name, "source.txt")
        self.assertEqual(source.parent.parent, self.view.launcher.request_root)
        self.assertEqual(log, source.parent / "worker.log")
        self.assertEqual(chapter_id, "chapter_" + source.parent.name)
        self.assertEqual(run_id, "run_" + source.parent.name)
        self.assertTrue(self.view.job_active)
        self.view.generate_button.configure.assert_called_with(state="disabled")
        self.view.filename_entry.configure.assert_called_with(state="disabled")
        self.view._poll_generation.assert_called_once()
        self.view.generate_audiobook()
        self.view.launcher.start.assert_called_once()

    def test_start_failure_is_logged_without_exposing_detail_in_status(self):
        self.view.chapter_text.get.return_value = "正文。"
        self.view.launcher.start.side_effect = RuntimeError("technical WSL failure")

        def launch_now(source, job_id, log):
            try:
                self.view.launcher.start(source, "chapter_" + job_id, "run_" + job_id, log)
            except RuntimeError as error:
                self.view.launch_queue.put(error)

        self.view._start_launcher = launch_now
        self.view.generate_audiobook()
        log = next(self.view.launcher.request_root.glob("*/worker.log"))
        self.assertIn("technical WSL failure", log.read_text(encoding="utf-8"))
        self.assertFalse(self.view.job_active)
        self.view.generation_status.configure.assert_called_with(text="Failed")
        self.assertNotIn("technical WSL failure", self.view.job_log.set.call_args.args[0])

    def test_filename_validation_handles_unicode_extension_and_windows_rules(self):
        self.assertEqual(normalize_mp3_filename(" 神通者04 "), "神通者04.mp3")
        self.assertEqual(normalize_mp3_filename("神通者04.MP3"), "神通者04.mp3")
        self.assertEqual(normalize_mp3_filename("épisode.mp3"), "épisode.mp3")
        for invalid in ("", "   ", "bad:name", "bad/name.mp3", "bad\\name.mp3",
                        "a*b.mp3", "a?b.mp3", "bad\nname.mp3", "CON.mp3",
                        "LPT1.mp3", "COM¹.mp3", "track.wav",
                        "bad\u202ename.wav",
                        "trailing. ", "a" * 252):
            with self.subTest(name=invalid), self.assertRaises(ValueError):
                normalize_mp3_filename(invalid)

    def test_invalid_filename_does_not_start_generation(self):
        self.view.chapter_text.get.return_value = "正文。"
        self.view.output_filename.get.return_value = "bad/name.mp3"
        with patch("src.audiobook_ui.messagebox.showwarning") as warning:
            self.view.generate_audiobook()
        warning.assert_called_once()
        self.view.launcher.start.assert_not_called()
        self.assertFalse(self.view.launcher.request_root.exists())

    def test_final_inspection_exports_after_validation_in_worker_thread(self):
        run_dir = self.root / "run"
        final_dir = run_dir / "final"
        final_dir.mkdir(parents=True)
        canonical = final_dir / "chapter.wav"
        canonical.write_bytes(b"validated WAV bytes")
        inspected = SimpleNamespace(
            assembly=SimpleNamespace(playable=True, audio_path=canonical),
            run_directory=run_dir,
        )
        self.view.launcher.run_directory = run_dir
        with patch("src.audiobook_ui.inspect_run", return_value=inspected) as validator, \
                patch("src.audiobook_ui.export_mp3", return_value=final_dir / "神通者04.mp3") as encoder:
            self.view._start_inspection("final")
            self.view.inspection_thread.join(timeout=2)
        kind, result, exported, error = self.view.inspection_queue.get_nowait()
        validator.assert_called_once_with(run_dir)
        self.assertEqual(kind, "final")
        self.assertIs(result, inspected)
        self.assertIsNone(error)
        encoder.assert_called_once_with(canonical, "神通者04.mp3")
        self.assertEqual(exported, final_dir / "神通者04.mp3")

    def test_elapsed_time_updates_and_freezes_on_finish(self):
        self.assertEqual(format_elapsed(5.9), "00:00:05")
        self.assertEqual(format_elapsed(3661.9), "01:01:01")
        self.view.started_at = 100.0
        with patch("src.audiobook_ui.time.monotonic", return_value=165.8):
            self.view._update_elapsed()
        self.view.elapsed_time.set.assert_called_with("Elapsed 00:01:05")
        with patch("src.audiobook_ui.time.monotonic", return_value=166.9):
            self.view._finish_generation("Complete")
        self.view.elapsed_time.set.assert_called_with("Elapsed 00:01:06")
        self.assertFalse(self.view.job_active)

    def test_progress_and_successful_final_validation(self):
        view = self.view
        view._show_progress(SimpleNamespace(total_units=None, selected_units=None))
        view.generation_status.configure.assert_called_with(text="Preparing...")
        view._show_progress(SimpleNamespace(total_units=4, selected_units=2))
        view.generation_progress.configure.assert_any_call(mode="determinate", maximum=4)
        view.generation_progress.configure.assert_any_call(value=2)
        view.generation_status.configure.assert_called_with(text="Generating 2 / 4")
        view._show_progress(SimpleNamespace(total_units=4, selected_units=4))
        view.generation_status.configure.assert_called_with(text="Assembling...")

        run_dir = self.root / "run"
        wav = run_dir / "final" / "chapter.wav"
        view.job_active = True
        view.launcher.poll.return_value = 0
        view.inspection_queue.put(("final", SimpleNamespace(
            assembly=SimpleNamespace(playable=True, audio_path=wav),
            run_directory=run_dir,
        ), run_dir / "final" / "神通者04.mp3", None))
        view._poll_generation()
        self.assertFalse(view.job_active)
        self.assertEqual(view.completed_run_directory, run_dir)
        view.finished_mp3.set.assert_called_with(str(run_dir / "final" / "神通者04.mp3"))
        view.open_folder_button.configure.assert_called_with(state="normal")
        view.generation_status.configure.assert_called_with(text="Complete")
        view.root.after.assert_not_called()

    def test_failed_process_without_run_stays_closed(self):
        view = self.view
        view.job_active = True
        view.launcher.poll.return_value = 1
        view._poll_generation()
        self.assertFalse(view.job_active)
        view.generation_status.configure.assert_called_with(text="Failed")
        view.open_folder_button.configure.assert_not_called()

        view.job_active = True
        view.launcher.poll.return_value = 0
        view.inspection_queue.put(("final", SimpleNamespace(
            assembly=SimpleNamespace(playable=False), run_directory=self.root,
        ), None, None))
        view._poll_generation()
        self.assertFalse(view.job_active)
        view.generation_status.configure.assert_called_with(text="Failed")
        view.open_folder_button.configure.assert_not_called()

    def test_failed_generation_opens_existing_run_root_and_keeps_mp3_blank(self):
        view = self.view
        run = self.root / "failed_run"
        run.mkdir()
        view.launcher.run_directory = run
        view.job_active = True
        view.launcher.poll.return_value = 1
        view._poll_generation()
        self.assertEqual(view.completed_run_directory, run)
        self.assertEqual(view.open_folder_target, run)
        self.assertEqual(view.generation_status.configure.call_args.kwargs["text"], "Failed")
        self.assertIn(str(view.launcher.log_path), view.job_log.set.call_args.args[0])
        view.finished_mp3.set.assert_not_called()
        view.open_folder_button.configure.assert_called_with(state="normal")
        with patch("src.audiobook_ui.os.startfile", create=True) as opener:
            view.open_folder()
        opener.assert_called_once_with(run)

    def test_export_collision_is_clear_and_preserves_canonical_wav_access(self):
        view = self.view
        view.job_active = True
        view.launcher.poll.return_value = 0
        canonical = self.root / "run" / "final" / "chapter.wav"
        view.inspection_queue.put(("final", SimpleNamespace(
            assembly=SimpleNamespace(playable=True, audio_path=canonical),
            run_directory=canonical.parent.parent,
        ), None, FileExistsError("already exists")))
        view._poll_generation()
        self.assertFalse(view.job_active)
        self.assertEqual(view.completed_run_directory, canonical.parent.parent)
        view.finished_mp3.set.assert_not_called()
        view.open_folder_button.configure.assert_called_with(state="normal")
        self.assertIn("already exists", view.job_log.set.call_args.args[0])

    def test_poll_schedules_manifest_inspection_without_blocking_tk(self):
        self.view.job_active = True
        self.view.launcher.poll.return_value = None
        self.view._start_inspection = Mock()
        self.view._poll_generation()
        self.view._start_inspection.assert_called_once_with("progress")
        self.view.root.after.assert_called_once_with(1000, self.view._poll_generation)

    def test_pending_launcher_shows_elapsed_and_avoids_manifest_read(self):
        self.view.job_active = True
        self.view.launch_pending = True
        self.view.started_at = 100.0
        self.view._start_inspection = Mock()
        with patch("src.audiobook_ui.time.monotonic", return_value=107.0):
            self.view._poll_generation()
        self.view.elapsed_time.set.assert_called_with("Elapsed 00:00:07")
        self.view.launcher.poll.assert_not_called()
        self.view._start_inspection.assert_not_called()
        self.view.root.after.assert_called_once_with(1000, self.view._poll_generation)

    def test_open_folder_uses_the_validated_run_directory(self):
        self.view.completed_run_directory = self.root / "run"
        with patch("src.audiobook_ui.os.startfile", create=True) as opener:
            self.view.open_folder()
        opener.assert_called_once_with(self.root / "run" / "final")


if __name__ == "__main__":
    unittest.main()
