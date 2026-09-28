"""Model-free checks for opt-in tracing and exclusive-time summaries."""

import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from evaluation.summarize_performance import (
    exclusive_seconds, main as summary_main, read_events, summarize,
)
from src.audiobook import profiling
from src.audiobook import asr_worker
from src.audiobook.content_qc import MODEL_ID, MODEL_REVISION
from src.audiobook_launcher import AudiobookLauncher
from src.audiobook.manifest import create_planning_run
from src.audiobook.pipeline import save_manifest
from src.audiobook.unit_execution import assemble_units, generate_units
from src.audiobook.unit_planning import prepare_synthesis_unit_run
from tests.test_audiobook_unit_execution import FakeUnitBackend, FakeASRWorker
from tests.test_audiobook_unit_planning import FixtureFrontend


class ProfilingTests(unittest.TestCase):
    def setUp(self):
        profiling.activate(None)
        profiling._local.stack = []

    def tearDown(self):
        profiling.activate(None)
        profiling._local.stack = []

    def wait_for_events(self, directory, count):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                events = read_events(directory)
            except ValueError:
                events = []
            if len(events) >= count:
                return events
            time.sleep(0.01)
        self.fail(f"Writer did not publish {count} trace events")

    def stop_writer(self):
        writer = profiling._writer
        profiling.activate(None)
        if writer is not None:
            writer.join(timeout=2)

    def fault_on_exit(self, name):
        clock = time.perf_counter_ns
        failed = [False]

        def tick():
            stack = profiling._stack()
            if stack and stack[-1].name == name and not failed[0]:
                failed[0] = True
                raise RuntimeError("injected trace clock failure")
            return clock()

        return patch.object(profiling.time, "perf_counter_ns", side_effect=tick)

    def test_disabled_by_default_and_nested_trace(self):
        disabled = profiling.span("disabled")
        self.assertIs(disabled, profiling.span("another disabled span"))
        with disabled:
            pass
        self.assertIsNone(profiling._writer)
        with tempfile.TemporaryDirectory() as temporary:
            profiling.activate(temporary)
            with profiling.span("outer", unit_id="unit_1") as outer:
                with profiling.span("inner", attempt_id="attempt_001"):
                    pass
            events = self.wait_for_events(temporary, 2)
            self.stop_writer()
        self.assertEqual(len(events), 2)
        child = next(item for item in events if item["name"] == "inner")
        self.assertEqual(child["parent_id"], outer.span_id)
        self.assertEqual(child["metadata"]["unit_id"], "unit_1")
        self.assertEqual(child["metadata"]["attempt_id"], "attempt_001")
        self.assertGreaterEqual(child["duration_ns"], 0)
        self.assertIsInstance(child["thread_id"], int)

    def test_trace_write_failure_does_not_raise(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(profiling, "_write_event", side_effect=OSError("trace unavailable")):
                profiling.activate(temporary)
                with profiling.span("work"):
                    pass
                deadline = time.monotonic() + 2
                while not profiling._failed and time.monotonic() < deadline:
                    time.sleep(0.01)
            self.assertTrue(profiling._failed)

    def test_clock_failure_preserves_return_and_original_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            profiling.activate(temporary)
            with patch.object(profiling.time, "perf_counter_ns",
                              side_effect=[100, RuntimeError("trace clock failed")]):
                with profiling.span("return"):
                    returned = 42
            self.assertEqual(returned, 42)
            self.assertFalse(profiling.enabled())
            profiling.activate(temporary)
            with patch.object(profiling.time, "perf_counter_ns",
                              side_effect=[100, RuntimeError("trace clock failed")]):
                with self.assertRaisesRegex(ValueError, "production failed"):
                    with profiling.span("exception"):
                        raise ValueError("production failed")

    def test_annotation_and_activation_failures_are_nonthrowing(self):
        with tempfile.TemporaryDirectory() as temporary:
            profiling.activate(temporary)
            with patch.object(profiling, "_stack", side_effect=RuntimeError("trace stack failed")):
                profiling.annotate(example=1)
            self.assertFalse(profiling.enabled())
            with patch.object(profiling, "Path", side_effect=RuntimeError("trace path failed")):
                profiling.activate(temporary)
            self.assertFalse(profiling.enabled())

    def test_recorder_initialization_failure_disables_tracing(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(profiling, "_lock", None):
                profiling.activate(temporary)
                self.assertFalse(profiling.enabled())
                self.assertIsNone(profiling._writer)

    def test_disabled_job_has_no_writer_or_observer_and_resets_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            profiling.activate(temporary)
            self.assertTrue(profiling.enabled())
            with patch.object(profiling.threading, "Thread") as thread:
                profiling.activate(None)
                self.assertFalse(profiling.enabled())
                self.assertIsNone(profiling._writer)
                self.assertIs(profiling.span("disabled"), profiling._NOOP)
                self.assertIs(profiling.begin("disabled"), profiling._NOOP)
                profiling.annotate(example=1)
                profiling.flush()
                thread.assert_not_called()
            self.assertFalse(hasattr(profiling, "observe_method"))

    def test_profile_directory_alone_does_not_enable_tracing(self):
        environment = os.environ.copy()
        environment.pop("TTS_PROFILE", None)
        environment["TTS_PROFILE_DIR"] = "unused-profile-directory"
        check = subprocess.run(
            [sys.executable, "-B", "-c",
             "from src.audiobook import profiling; "
             "print(profiling.enabled(), profiling._writer is None)"],
            env=environment, capture_output=True, text=True, check=True,
        )
        self.assertEqual(check.stdout.strip(), "False True")

    def test_windows_launcher_passes_profile_dir_to_wsl_python(self):
        with tempfile.TemporaryDirectory() as temporary:
            profiling.activate(temporary)
            launcher = AudiobookLauncher(repository_root=r"C:\TTS Project",
                                        platform="win32")
            with patch("src.audiobook_launcher.windows_to_wsl_path", side_effect=[
                    "/mnt/c/TTS Project",
                    "/mnt/c/Users/Jay Ma/TTS_Audiobooks/ui_requests/job/source.txt",
                    "/mnt/c/Users/Jay Ma/TTS_Audiobooks/audiobooks"]):
                command = launcher.command(
                    r"C:\Users\Jay Ma\TTS_Audiobooks\ui_requests\job\source.txt",
                    "chapter_job", "run_job")
            self.stop_writer()
        self.assertIn("/usr/bin/env", command)
        self.assertIn("TTS_PROFILE=1", command)
        self.assertIn("TTS_PROFILE_DIR=/mnt/c/Users/Jay Ma/TTS_Audiobooks/ui_requests/job/profile",
                      command)
        self.assertEqual(command[-3:], ["run_job", "--output-root",
                                        "/mnt/c/Users/Jay Ma/TTS_Audiobooks/audiobooks"])

    def test_disabled_launcher_passes_no_profile_configuration(self):
        launcher = AudiobookLauncher(repository_root=r"C:\TTS Project", platform="win32")
        with patch("src.audiobook_launcher.windows_to_wsl_path", side_effect=[
                "/mnt/c/TTS Project", "/mnt/c/TTS Project/source.txt",
                "/mnt/c/Users/Jay Ma/TTS_Audiobooks/audiobooks"]):
            command = launcher.command(r"C:\TTS Project\source.txt", "chapter", "run")
        self.assertNotIn("/usr/bin/env", command)
        self.assertFalse(any("TTS_PROFILE_DIR=" in part for part in command))

    def test_launched_child_remains_owned_when_exit_hook_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            profiling.activate(temporary)
            log = io.BytesIO()
            child = Mock()
            launcher = AudiobookLauncher(repository_root=temporary,
                                        platform="win32", popen=Mock(return_value=child))
            original_exit = profiling.Span.__exit__

            def broken_exit(span, kind, error, traceback):
                if span.name == "launcher.process_launch":
                    raise RuntimeError("injected trace exit failure")
                return original_exit(span, kind, error, traceback)

            with patch.object(launcher, "command", return_value=["fake"]), \
                    patch.object(Path, "is_file", return_value=True), \
                    patch.object(Path, "exists", return_value=False), \
                    patch.object(Path, "mkdir"), \
                    patch.object(Path, "open", return_value=log), \
                    patch.object(profiling.Span, "__exit__", broken_exit):
                result = launcher.start("source.txt", "chapter", "run", "worker.log")
            self.assertEqual(result, launcher.run_directory)
            self.assertIs(launcher._process, child)
            self.assertIs(launcher._log, log)
            self.assertFalse(log.closed)
            log.close()

    def test_asr_sends_one_response_after_trace_exit_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            profiling.activate(temporary)
            metadata = {"model_id": MODEL_ID, "resolved_revision": MODEL_REVISION}
            result = dict(raw_transcript="recognized", comparison_tokens=[],
                          comparison_text="", raw_emitted_tokens=[], ignored_tokens=[],
                          audio={}, inference_seconds=0, emission_frames=0,
                          ctc_frame_seconds=0)
            request = {"type": "recognize", "request_id": "request_probe",
                       "audio_path": "unused.wav", "wav_sha256": "digest"}
            output = io.StringIO()
            original_exit = profiling.Span.__exit__

            def broken_exit(span, kind, error, traceback):
                if span.name == "asr.response_serialize":
                    raise RuntimeError("injected trace exit failure")
                return original_exit(span, kind, error, traceback)

            with patch.object(asr_worker, "_load_model",
                              return_value=(None, None, None, None, metadata)), \
                    patch.object(asr_worker, "_infer", return_value=[result]), \
                    patch.object(asr_worker, "_sha256", return_value="digest"), \
                    patch.object(Path, "resolve", return_value=Path("unused.wav")), \
                    patch.object(Path, "is_file", return_value=True), \
                    patch("sys.stdin", io.StringIO(json.dumps(request) + "\n")), \
                    patch("sys.stdout", output), \
                    patch.object(profiling.Span, "__exit__", broken_exit):
                asr_worker.run()
            responses = [json.loads(line)["type"] for line in output.getvalue().splitlines()]
            self.assertEqual(responses, ["ready", "recognized"])
            self.stop_writer()

    def test_manifest_replace_success_survives_trace_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            profiling.activate(Path(temporary) / "profile")
            path = Path(temporary) / "manifest.json"
            with self.fault_on_exit("manifest.replace"):
                self.assertIsNone(save_manifest(path, {"status": "valid"}))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                             {"status": "valid"})
            self.stop_writer()

    def test_final_publication_success_survives_trace_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "chapter.txt"
            source.write_text("甲。乙。", encoding="utf-8")
            run, _ = create_planning_run(source, "chapter", "run", root / "outputs")
            prepared = prepare_synthesis_unit_run(run, FixtureFrontend())
            backend = FakeUnitBackend(prepared["synthesis_unit_plan"]["frontend_identity_sha256"])
            with patch("src.audiobook.unit_execution.ASRWorkerClient", FakeASRWorker):
                self.assertEqual(generate_units(run, backend, root_seed=17)["status"], "generated")
            profiling.activate(root / "profile")
            with self.fault_on_exit("assembly.publish"):
                result = assemble_units(run)
            self.assertEqual(result["assembly"]["status"], "assembled")
            self.assertTrue((run / result["assembly"]["output_path"]).is_file())
            self.stop_writer()

    def test_slow_writer_never_blocks_production_caller(self):
        with tempfile.TemporaryDirectory() as temporary:
            entered = threading.Event()
            release = threading.Event()

            def slow_write(*_args):
                entered.set()
                release.wait(2)

            try:
                with patch.object(profiling, "MAX_PENDING_EVENTS", 1), \
                        patch.object(profiling, "_write_event", side_effect=slow_write):
                    profiling.activate(temporary)
                    with profiling.span("first"):
                        pass
                    self.assertTrue(entered.wait(1))
                    started = time.perf_counter()
                    with profiling.span("second"):
                        pass
                    with profiling.span("dropped"):
                        pass
                    self.assertLess(time.perf_counter() - started, 0.5)
            finally:
                writer = profiling._writer
                profiling.activate(None)
                release.set()
                writer.join(timeout=1)

    def test_previous_job_writer_failure_cannot_disable_next_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            entered = threading.Event()
            release = threading.Event()
            real_write = profiling._write_event

            def write(destination, event, state):
                if destination == first:
                    entered.set()
                    release.wait(2)
                    raise OSError("old job trace failed")
                return real_write(destination, event, state)

            with patch.object(profiling, "_write_event", side_effect=write):
                profiling.activate(first)
                with profiling.span("old job"):
                    pass
                self.assertTrue(entered.wait(1))
                old_writer = profiling._writer
                profiling.activate(second)
                try:
                    with profiling.span("new job"):
                        pass
                    release.set()
                    old_writer.join(timeout=1)
                    self.assertTrue(profiling.enabled())
                    events = self.wait_for_events(second, 1)
                    self.assertEqual(events[0]["name"], "new job")
                finally:
                    release.set()
                    self.stop_writer()


def event(name, start, end, span_id, parent_id=None, *, pid=1,
          process_uid=None, **metadata):
    return {
        "schema": 1, "name": name, "span_id": span_id,
        "parent_id": parent_id, "pid": pid,
        "process_uid": process_uid or f"test:{pid}", "thread_id": 1,
        "start_ns": start * 1_000_000_000,
        "end_ns": end * 1_000_000_000,
        "duration_ns": (end - start) * 1_000_000_000,
        "outcome": "ok", "metadata": metadata,
    }


class SummaryTests(unittest.TestCase):
    def test_report_prints_on_windows_legacy_console_encoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "events-test.jsonl"
            trace.write_text(json.dumps(event("ui.generate_to_complete", 0, 1, "root")) + "\n",
                             encoding="utf-8")
            with io.TextIOWrapper(io.BytesIO(), encoding="cp1252") as output:
                with patch("sys.stdout", output):
                    self.assertEqual(summary_main([temporary]), 0)

    def test_nested_spans_are_not_added_to_chapter_total(self):
        events = [
            event("ui.generate_to_complete", 0, 100, "root"),
            event("ui.source_write", 0, 5, "source", "root"),
            event("launcher.command_preparation", 5, 15, "launch", "root"),
            event("launcher.process_launch", 15, 16, "popen", "root"),
            event("cli.startup_imports", 20, 25, "imports"),
            event("cli.command", 25, 85, "command"),
            event("launcher.worker_exit_observed", 85, 85, "exit", "root"),
            event("workflow.generation", 30, 75, "generation", "command"),
            event("unit.cycle", 35, 70, "cycle", "generation",
                  unit_id="unit_1", selected_attempt_id="attempt_002"),
            event("unit.attempt", 35, 45, "attempt1", "cycle",
                  unit_id="unit_1", attempt_id="attempt_001", audio_seconds=10),
            event("unit.synthesis", 35, 43, "synth1", "attempt1",
                  unit_id="unit_1", attempt_id="attempt_001"),
            event("tts.inference", 36, 42, "tts1", "synth1",
                  unit_id="unit_1", attempt_id="attempt_001"),
            event("tts.conditioning", 36, 37, "condition", "tts1",
                  unit_id="unit_1", attempt_id="attempt_001"),
            event("unit.qc", 45, 47, "qc1", "cycle", unit_id="unit_1",
                  attempt_id="attempt_001", decision="rejected"),
            event("unit.attempt", 47, 55, "attempt2", "cycle",
                  unit_id="unit_1", attempt_id="attempt_002", audio_seconds=12),
            event("unit.synthesis", 47, 54, "synth2", "attempt2",
                  unit_id="unit_1", attempt_id="attempt_002"),
            event("tts.inference", 48, 52, "tts2", "synth2",
                  unit_id="unit_1", attempt_id="attempt_002"),
            event("ui.final_validation", 85, 90, "validate", "root"),
            event("ui.named_file_copy", 90, 95, "copy", "root"),
            event("ui.export_ready", 95, 95, "ready"),
        ]
        result = summarize(events)
        self.assertEqual(result["end_to_end_generate_to_complete_s"], 100)
        self.assertEqual(result["generate_to_export_ready_s"], 95)
        self.assertEqual(result["critical_path_accounted_s"], 95)
        self.assertEqual(result["unclassified_remainder_s"], 5)
        self.assertEqual(result["worker_launch_to_exit_observed_s"], 70)
        self.assertIsNone(result["launch_bootstrap_exit_poll_gap_s"])
        self.assertEqual(result["phases"]["unit.synthesis"]["inclusive_s"], 15)
        self.assertEqual(result["phases"]["unit.synthesis"]["exclusive_s"], 5)
        self.assertEqual(result["phases"]["tts.conditioning"]["inclusive_s"], 1)
        self.assertEqual(result["rejected_qc_attempts"], 1)
        self.assertEqual(result["extra_physical_attempts"], 1)
        self.assertEqual(result["per_unit"][0]["synthesis_inference_s"], 4)
        self.assertEqual(result["per_unit"][0]["generated_audio_s"], 12)
        self.assertEqual(result["per_unit"][0]["rtf"], round(4 / 12, 6))

    def test_conditioning_is_unavailable_without_live_method_observer(self):
        result = summarize([event("ui.generate_to_complete", 0, 1, "root")])
        self.assertIsNone(result["totals"]["narrator_conditioning_s"])
        self.assertEqual(result["phases"]["tts.conditioning"],
                         {"inclusive_s": None, "exclusive_s": None, "count": 0})

    def test_exclusive_merges_overlapping_children(self):
        parent = event("parent", 0, 10, "p")
        children = {"p": [event("a", 1, 5, "a", "p"),
                          event("b", 3, 7, "b", "p")]}
        self.assertEqual(exclusive_seconds(parent, children), 4)

    def test_multiprocess_clocks_and_missing_cli_completion_do_not_create_false_gap(self):
        win = {"pid": 101, "process_uid": "win32:101:ui"}
        backend = {"pid": 202, "process_uid": "linux:202:backend"}
        asr = {"pid": 303, "process_uid": "linux:303:asr"}
        events = [
            event("ui.generate_to_complete", 0, 100, "ui", **win),
            event("ui.source_write", 0, 2, "source", "ui", **win),
            event("ui.launch", 2, 8, "launch", "ui", **win),
            event("launcher.command_preparation", 3, 7, "prepare", "launch", **win),
            event("launcher.process_launch", 7, 8, "popen", "launch", **win),
            event("ui.progress_inspection", 10, 80, "progress", "ui", **win),
            event("launcher.worker_exit_observed", 90, 90, "exit", "ui", **win),
            event("ui.final_validation", 92, 96, "validate", "ui", **win),
            event("ui.named_file_copy", 96, 99, "copy", "ui", **win),
            # The backend has an unrelated monotonic origin, and its final
            # cli.command span was lost by the best-effort trace writer.
            event("cli.entry", 10000, 10000, "entry", **backend),
            event("cli.startup_imports", 10000, 10001, "imports", **backend),
            event("cli.preflight", 10001, 10006, "preflight", **backend),
            event("workflow.planning", 10006, 10020, "plan", "missing_command", **backend),
            event("workflow.generation", 10020, 10105, "generation", "missing_command",
                  **backend),
            event("tts.inference", 10030, 10070, "inference", "generation", **backend),
            event("assembly.write_pcm", 10105, 10107, "assembly_write",
                  "missing_assembly", **backend),
            event("asr.model_initialization", -5000, -4995, "asr_load", **asr),
        ]
        result = summarize(events)
        self.assertEqual(result["end_to_end_generate_to_complete_s"], 100)
        self.assertEqual(result["critical_path_segments_s"], {
            "ui_before_worker_s": 8,
            "worker_launch_to_exit_observed_s": 82,
            "ui_after_worker_s": 10,
        })
        self.assertEqual(result["critical_path_accounted_s"], 97)
        self.assertEqual(result["unclassified_remainder_s"], 3)
        self.assertEqual(result["unmeasured_windows_s"],
                         {"before_worker_s": 0, "after_worker_s": 3})
        self.assertIsNone(result["launch_bootstrap_exit_poll_gap_s"])
        self.assertFalse(result["backend_command_span_recorded"])
        self.assertEqual(result["backend_local_observed_coverage_s"], 107)
        self.assertEqual(result["totals"]["cosyvoice_inference_s"], 40)
        self.assertIsNone(result["totals"]["assembly_s"])
        self.assertEqual(result["totals"]["assembly_observed_subphases_s"], 2)

    def test_exclusive_never_subtracts_other_process_with_reused_pid(self):
        parent = event("parent", 0, 10, "same-id", pid=5, process_uid="linux:5:one")
        child = event("child", 1, 9, "child", "same-id", pid=5,
                      process_uid="linux:5:two")
        self.assertEqual(exclusive_seconds(parent, {"same-id": [child]}), 10)


if __name__ == "__main__":
    unittest.main()
