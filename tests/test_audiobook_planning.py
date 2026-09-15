"""Model-free tests for the Milestone D1 input and planning contract."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.audiobook.manifest import SCHEMA_VERSION, create_planning_run
from src.audiobook.planning import PLANNER_VERSION, PlanningError, build_plan


ROOT = Path(__file__).resolve().parents[1]


class AudiobookPlanningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "第一章.txt"
        self.outputs = self.root / "outputs"

    def write_source(self, text):
        self.source.write_bytes(text.encode("utf-8"))

    def create_run(self, text, chapter_id="chapter_0001", run_id="run_001"):
        self.write_source(text)
        return create_planning_run(
            self.source,
            chapter_id,
            run_id,
            self.outputs,
            now=datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc),
        )

    def test_valid_utf8_chapter_and_no_markers_make_one_scene(self):
        text = "第一段。\n\n第二段！\n"
        _, manifest = self.create_run(text)
        self.assertEqual(len(manifest["scenes"]), 1)
        self.assertEqual(manifest["scenes"][0]["narration_text"], text)

    def test_empty_and_whitespace_only_input_rejected(self):
        for value in ("", " \t\r\n　"):
            with self.subTest(value=repr(value)), self.assertRaisesRegex(
                PlanningError, "empty or whitespace-only"
            ):
                build_plan(value.encode("utf-8"))

    def test_standalone_markers_create_ordered_scenes_and_stable_ids(self):
        plan = build_plan("场景一。\n  ***  \r\n场景二？\n***\n场景三！".encode("utf-8"))
        self.assertEqual(
            [scene["id"] for scene in plan["scenes"]],
            ["scene_0001", "scene_0002", "scene_0003"],
        )
        self.assertEqual([scene["order"] for scene in plan["scenes"]], [1, 2, 3])
        self.assertEqual(
            [scene["narration_text"] for scene in plan["scenes"]],
            ["场景一。\n", "场景二？\n", "场景三！"],
        )

    def test_embedded_marker_is_prose(self):
        text = "他说***这不是场景标记。\n下一段。"
        plan = build_plan(text.encode("utf-8"))
        self.assertEqual(len(plan["scenes"]), 1)
        self.assertEqual(plan["scenes"][0]["narration_text"], text)

    def test_marker_placement_cannot_create_empty_scenes(self):
        invalid = ("***\n正文", "正文\n***\n", "正文\n***\n \t\n***\n结尾")
        for text in invalid:
            with self.subTest(text=text), self.assertRaisesRegex(
                PlanningError, "empty scene"
            ):
                build_plan(text.encode("utf-8"))

    def test_mandarin_punctuation_paragraphs_and_source_spans_are_preserved(self):
        text = "第一段：“你好吗？”\n\n第二段——很好。\n***\n第三段……结束！"
        plan = build_plan(text.encode("utf-8"))
        reconstructed = []
        decoded = text
        for scene in plan["scenes"]:
            span = scene["source_span"]
            selected = decoded[span["start_character"]:span["end_character"]]
            self.assertEqual(selected, scene["narration_text"])
            source_bytes = text.encode("utf-8")
            self.assertEqual(
                source_bytes[span["start_byte"]:span["end_byte"]].decode("utf-8"),
                scene["narration_text"],
            )
            reconstructed.append(selected)
        self.assertEqual(reconstructed, ["第一段：“你好吗？”\n\n第二段——很好。\n", "第三段……结束！"])
        self.assertEqual(
            [(s["source_span"]["start_line"], s["source_span"]["end_line"]) for s in plan["scenes"]],
            [(1, 3), (5, 5)],
        )

    def test_plan_and_hash_are_deterministic(self):
        source = "甲。\n***\n乙。".encode("utf-8")
        first = build_plan(source)
        second = build_plan(source)
        self.assertEqual(first, second)
        self.assertEqual(first["plan_hash"], second["plan_hash"])

    def test_changed_source_changes_source_and_plan_hashes(self):
        first = build_plan("甲。".encode("utf-8"))
        second = build_plan("乙。".encode("utf-8"))
        self.assertNotEqual(first["source_sha256"], second["source_sha256"])
        self.assertNotEqual(first["plan_hash"], second["plan_hash"])

    def test_source_snapshot_preserves_exact_bytes(self):
        original = b"\xef\xbb\xbf\xe7\x94\xb2\xe3\x80\x82\r\n***\r\n\xe4\xb9\x99\xe3\x80\x82\r\n"
        self.source.write_bytes(original)
        run_dir, manifest = create_planning_run(
            self.source, "chapter_0001", "run_001", self.outputs
        )
        self.assertEqual((run_dir / "source.txt").read_bytes(), original)
        self.assertEqual(manifest["source"]["byte_length"], len(original))
        self.assertEqual([s["narration_text"] for s in manifest["scenes"]], ["甲。\r\n", "乙。\r\n"])
        decoded = original.decode("utf-8")
        for scene in manifest["scenes"]:
            span = scene["source_span"]
            self.assertEqual(
                decoded[span["start_character"]:span["end_character"]],
                scene["narration_text"],
            )

    def test_manifest_contains_only_d1_contract_fields_and_relative_snapshot(self):
        run_dir, expected = self.create_run("甲。\n***\n乙。")
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest, expected)
        self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
        self.assertEqual(manifest["planner"]["version"], PLANNER_VERSION)
        self.assertEqual(manifest["status"], "planned")
        self.assertEqual(manifest["source"]["snapshot_path"], "source.txt")
        self.assertEqual(
            set(manifest),
            {"schema_version", "chapter_id", "run_id", "created_at_utc", "status",
             "source", "planner", "plan_hash", "scenes"},
        )
        forbidden = {"model", "attempts", "retries", "repairs", "listening_status", "assembly"}
        self.assertTrue(forbidden.isdisjoint(manifest))

    def test_run_directory_collision_is_rejected_without_overwrite(self):
        run_dir, _ = self.create_run("原始内容。")
        original_manifest = (run_dir / "manifest.json").read_bytes()
        self.write_source("新内容。")
        with self.assertRaisesRegex(PlanningError, "already exists"):
            create_planning_run(self.source, "chapter_0001", "run_001", self.outputs)
        self.assertEqual((run_dir / "manifest.json").read_bytes(), original_manifest)
        self.assertEqual((run_dir / "source.txt").read_text(encoding="utf-8"), "原始内容。")

    def test_invalid_utf8_is_rejected(self):
        with self.assertRaisesRegex(PlanningError, "not valid UTF-8"):
            build_plan(b"\xff\xfe")

    def test_cli_uses_standard_library_only_and_creates_expected_files(self):
        self.write_source("一个完整场景。")
        command = [
            sys.executable, "-B", "-m", "src.audiobook", "plan", str(self.source),
            "--chapter-id", "chapter_0001", "--run-id", "run_001",
            "--output-root", str(self.outputs),
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        run_dir = self.outputs / "chapter_0001" / "run_001"
        self.assertEqual({path.name for path in run_dir.iterdir()}, {"source.txt", "manifest.json"})
        self.assertNotIn("cosyvoice", result.stdout.lower() + result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
