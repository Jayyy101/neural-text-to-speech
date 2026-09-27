"""Model-free tests for immutable synthesis-unit preparation."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.audiobook.cosyvoice import CosyVoiceFrontendAdapter
from src.audiobook.manifest import create_planning_run
from src.audiobook.pipeline import GenerationError, generate_planned_run
from src.audiobook.unit_planning import (
    MAX_NORMALIZED_UNIT_CHARACTERS,
    SynthesisUnitPlanningError,
    UNIT_PLAN_SCHEMA_VERSION,
    LEGACY_SOURCE_MAPPING_POLICY,
    SOURCE_MAPPING_POLICY,
    HAN_SUFFIX_FRONTEND_IDENTITY_SHA256,
    _slice_certification,
    build_synthesis_unit_plan,
    canonical_sha256,
    _find_unique_partition,
    _han_suffix_input_is_safe,
    _heading_has_terminal_punctuation,
    prepare_synthesis_unit_run,
    validate_synthesis_unit_plan,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/cosyvoice_frontend_unit_contract_v1.json"


class FixtureFrontend:
    """Explicit model-free double for the saved frontend contract cases."""

    def __init__(self, revision="fixture-v1"):
        self.revision = revision
        self.initialize_calls = 0
        self.normalize_calls = []

    def initialize(self):
        self.initialize_calls += 1
        return {
            "identity": {
                "frontend": "saved_fixture_double",
                "revision": self.revision,
                "implementation_sha256": "a" * 64,
                "asset_sha256": "b" * 64,
                "splitting_settings": {
                    "token_max_n": 80,
                    "token_min_n": 60,
                    "merge_len": 20,
                    "comma_split": False,
                },
            },
            "runtime": {"kind": "model_free_test_double"},
            "inference_calls": 0,
        }

    def normalize(self, text):
        self.normalize_calls.append(text)
        normalized = (
            text.strip()
            .replace("\r", "")
            .replace("\n", "")
            .replace(" ", "")
            .replace("（", "")
            .replace("）", "")
            .replace("1", "一")
            .replace("?!", "?")
        )
        if not normalized:
            return []
        punctuation = set("。？！?！")
        if normalized[-1] not in punctuation:
            normalized += "。"
        units = []
        start = 0
        for index, character in enumerate(normalized):
            if character in punctuation:
                units.append(normalized[start:index + 1])
                start = index + 1
        return [unit for unit in units if unit]


class AudiobookUnitPlanningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.outputs = self.root / "outputs"

    def create_run(self, source_text, run_id="run_001", bom=False):
        source = self.root / f"{run_id}.txt"
        payload = source_text.encode("utf-8")
        if bom:
            payload = b"\xef\xbb\xbf" + payload
        source.write_bytes(payload)
        return create_planning_run(
            source, "chapter_0001", run_id, self.outputs,
            now=datetime(2026, 9, 20, tzinfo=timezone.utc),
        )

    def prepare(self, source_text, frontend=None, run_id="run_001", bom=False):
        run_dir, _ = self.create_run(source_text, run_id=run_id, bom=bom)
        return run_dir, prepare_synthesis_unit_run(
            run_dir, frontend or FixtureFrontend()
        )

    def test_deterministic_units_hashes_order_and_plan_identity(self):
        text = "第一句。\n第二句！\n***\n第三句？"
        first_dir, first = self.prepare(text, run_id="run_001")
        second_dir, second = self.prepare(text, run_id="run_002")

        self.assertEqual(first["schema_version"], UNIT_PLAN_SCHEMA_VERSION)
        self.assertEqual(first["status"], "units_planned")
        self.assertEqual(
            first["synthesis_unit_plan"]["ordered_unit_plan_sha256"],
            second["synthesis_unit_plan"]["ordered_unit_plan_sha256"],
        )
        self.assertEqual(
            [[unit["id"] for unit in scene["synthesis_units"]]
             for scene in first["scenes"]],
            [["scene_0001_unit_0001", "scene_0001_unit_0002"],
             ["scene_0002_unit_0001"]],
        )
        validate_synthesis_unit_plan(first_dir, first)
        validate_synthesis_unit_plan(second_dir, second)

    def test_source_spans_reconstruct_exact_text_bytes_and_crlf(self):
        text = "甲。\r\n乙。\r\n"
        run_dir, manifest = self.prepare(text, bom=True)
        scene = manifest["scenes"][0]
        units = scene["synthesis_units"]

        self.assertEqual([unit["source_text"] for unit in units], ["甲。\r\n", "乙。\r\n"])
        self.assertEqual([unit["preprocessed_text"] for unit in units], ["甲。\n", "乙。\n"])
        self.assertEqual([unit["text_preprocessing"]["removed_cr_count"] for unit in units], [1, 1])
        self.assertEqual("".join(unit["source_text"] for unit in units), text)
        snapshot = (run_dir / "source.txt").read_bytes()
        for unit in units:
            span = unit["source_span"]
            self.assertEqual(
                snapshot[span["start_byte"]:span["end_byte"]].decode("utf-8"),
                unit["source_text"],
            )
        self.assertEqual(units[0]["source_span"]["start_character"], 1)
        self.assertEqual(units[0]["source_span"]["start_byte"], 3)

    def test_punctuation_number_normalization_and_provenance_are_frozen(self):
        text = "第1章。\n哦?!继续！"
        _, manifest = self.prepare(text)
        units = manifest["scenes"][0]["synthesis_units"]
        self.assertEqual(
            [unit["normalized_text"] for unit in units],
            ["第一章。", "哦?", "继续！"],
        )
        self.assertEqual("".join(unit["source_text"] for unit in units), text)
        self.assertEqual(units[0]["source_text"], "第1章。\n")
        self.assertEqual(units[1]["source_text"], "哦?!")
        self.assertTrue(all(
            unit["mapping"]["independent_slice_reproduced_exactly"]
            for unit in units
        ))
        self.assertEqual(
            manifest["synthesis_unit_plan"]["policy"]["text_preprocessing_policy"],
            "remove_u000d_v1",
        )

    def test_repeated_text_maps_in_source_order_without_loss(self):
        text = "相同。相同。相同。"
        _, manifest = self.prepare(text)
        units = manifest["scenes"][0]["synthesis_units"]
        self.assertEqual([unit["source_text"] for unit in units], ["相同。"] * 3)
        self.assertEqual(
            [unit["source_span"]["start_character"] for unit in units],
            [0, 3, 6],
        )

    def test_chapter_heading_period_is_explicit_without_rewriting_native_unit(self):
        class HeadingFrontend(FixtureFrontend):
            def normalize_heading(self, heading):
                self.assert_heading = heading
                return "第3章标题"

        text = "第3章 标题\r\n\r\n甲。"
        run_dir, manifest = self.prepare(text, HeadingFrontend())
        override = manifest["title_synthesis_override"]
        first = manifest["scenes"][0]["synthesis_units"][0]
        self.assertEqual(first["normalized_text"], "第3章标题甲。")
        self.assertEqual(override["synthesis_text"], "第3章标题。甲。")
        self.assertEqual("".join(u["source_text"] for u in manifest["scenes"][0]["synthesis_units"]), text)
        validate_synthesis_unit_plan(run_dir, manifest)
        override["insertion_index"] -= 1
        with self.assertRaisesRegex(SynthesisUnitPlanningError, "Title override"):
            validate_synthesis_unit_plan(run_dir, manifest)

    def test_non_heading_does_not_get_title_punctuation(self):
        _, manifest = self.prepare("甲。乙。", FixtureFrontend())
        self.assertNotIn("title_synthesis_override", manifest)

    def test_existing_title_punctuation_inside_closing_quote_is_preserved(self):
        self.assertTrue(_heading_has_terminal_punctuation("第3章 标题。”"))
        self.assertFalse(_heading_has_terminal_punctuation("第3章 标题”"))

    def test_ambiguous_mapping_is_rejected_without_changing_manifest(self):
        run_dir, original = self.create_run("甲。（）甲。")
        manifest_path = run_dir / "manifest.json"
        before = manifest_path.read_bytes()
        with self.assertRaisesRegex(SynthesisUnitPlanningError, "ambiguous"):
            prepare_synthesis_unit_run(run_dir, FixtureFrontend())
        self.assertEqual(manifest_path.read_bytes(), before)
        self.assertEqual(original["schema_version"], 1)

    def test_mapping_checks_later_boundaries_after_intermediate_multisplit(self):
        class NonMonotoneFrontend(FixtureFrontend):
            def normalize(self, text):
                cases = {
                    "甲乙丙丁": ["A", "B"],
                    "甲": ["A"],
                    "乙丙丁": ["B"],
                    "甲乙": ["other", "units"],
                    "甲乙丙": ["A"],
                    "丁": ["B"],
                }
                return cases.get(text, ["different"])

        run_dir, _ = self.create_run("甲乙丙丁")
        with self.assertRaisesRegex(SynthesisUnitPlanningError, "ambiguous"):
            prepare_synthesis_unit_run(run_dir, NonMonotoneFrontend())

    def test_oversized_unit_is_rejected_without_fallback_splitting(self):
        text = "甲" * MAX_NORMALIZED_UNIT_CHARACTERS + "乙。"
        run_dir, _ = self.create_run(text)
        before = (run_dir / "manifest.json").read_bytes()
        with self.assertRaisesRegex(
            SynthesisUnitPlanningError, "unsupported oversized"
        ):
            prepare_synthesis_unit_run(run_dir, FixtureFrontend())
        self.assertEqual((run_dir / "manifest.json").read_bytes(), before)

    def test_frontend_identity_change_changes_unit_plan_identity(self):
        text = "甲。乙。"
        _, first = self.prepare(
            text, FixtureFrontend("frontend-a"), run_id="run_a"
        )
        _, second = self.prepare(
            text, FixtureFrontend("frontend-b"), run_id="run_b"
        )
        self.assertNotEqual(
            first["synthesis_unit_plan"]["frontend_identity_sha256"],
            second["synthesis_unit_plan"]["frontend_identity_sha256"],
        )
        self.assertNotEqual(
            first["synthesis_unit_plan"]["ordered_unit_plan_sha256"],
            second["synthesis_unit_plan"]["ordered_unit_plan_sha256"],
        )

    def test_unit_planner_policy_change_changes_plan_identity(self):
        text = "甲。乙。"
        _, first = self.prepare(text, run_id="run_policy_a")
        with patch(
            "src.audiobook.unit_planning.UNIT_PLANNING_POLICY",
            "pinned_cosyvoice_frontend_units_v2_test",
        ):
            _, second = self.prepare(text, run_id="run_policy_b")
        self.assertNotEqual(
            first["synthesis_unit_plan"]["ordered_unit_plan_sha256"],
            second["synthesis_unit_plan"]["ordered_unit_plan_sha256"],
        )

    def test_tampering_breaks_persisted_plan_validation(self):
        run_dir, manifest = self.prepare("甲。乙。")
        manifest["scenes"][0]["synthesis_units"][0]["source_text"] = "篡改。"
        with self.assertRaisesRegex(SynthesisUnitPlanningError, "source text"):
            validate_synthesis_unit_plan(run_dir, manifest)

    def test_persisted_policy_and_frontend_identity_are_checked(self):
        run_dir, manifest = self.prepare("甲。乙。")
        with self.assertRaisesRegex(SynthesisUnitPlanningError, "differs"):
            validate_synthesis_unit_plan(run_dir, manifest, "0" * 64)
        manifest["synthesis_unit_plan"]["policy"]["name"] = "tampered"
        with self.assertRaisesRegex(SynthesisUnitPlanningError, "incompatible"):
            validate_synthesis_unit_plan(run_dir, manifest)

    def test_preparation_rejects_non_d1_schema_without_migration(self):
        run_dir, manifest = self.create_run("甲。")
        manifest["schema_version"] = 4
        manifest["status"] = "generated"
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        before = manifest_path.read_bytes()
        with self.assertRaisesRegex(SynthesisUnitPlanningError, "untouched"):
            prepare_synthesis_unit_run(run_dir, FixtureFrontend())
        self.assertEqual(manifest_path.read_bytes(), before)

    def test_legacy_scene_generation_rejects_a_unit_planned_run(self):
        run_dir, _ = self.prepare("甲。乙。")
        manifest_path = run_dir / "manifest.json"
        before = manifest_path.read_bytes()
        backend = Mock()
        with self.assertRaisesRegex(GenerationError, "untouched"):
            generate_planned_run(run_dir, backend)
        backend.initialize.assert_not_called()
        self.assertEqual(manifest_path.read_bytes(), before)


class TerminalDunhaoCertificationTests(unittest.TestCase):
    def certify(self, source, target, observed, **kwargs):
        return _slice_certification(
            source, target, [observed],
            kwargs.get("identity", HAN_SUFFIX_FRONTEND_IDENTITY_SHA256),
            kwargs.get("policy", SOURCE_MAPPING_POLICY),
        )

    def test_confirmed_terminal_case_records_exact_evidence(self):
        evidence = self.certify("有一道似虎豹、\r\n", "有一道似虎豹、", "有一道似虎豹。")
        self.assertFalse(evidence["independent_slice_reproduced_exactly"])
        exception = evidence["normalization_equivalence"]
        self.assertEqual(exception["source_character_offset"], 6)
        self.assertEqual(exception["independent_normalized_text"], "有一道似虎豹。")
        self.assertEqual(len(exception["independent_normalized_text_sha256"]), 64)

    def test_normal_period_and_internal_dunhao_are_exact(self):
        for text in ("似虎豹。", "似虎豹、似雷鸣。"):
            evidence = self.certify(text, text, text)
            self.assertTrue(evidence["independent_slice_reproduced_exactly"])
            self.assertNotIn("normalization_equivalence", evidence)

    def test_only_verified_terminal_equivalence_is_eligible(self):
        for source, target, observed in (
            ("似虎豹", "似虎豹、", "似虎豹。"),  # cannot cut before the delimiter
            ("似虎豹。", "似虎豹、", "似虎豹。"),
            ("似虎豹，", "似虎豹、", "似虎豹。"),
            ("似虎豹、、", "似虎豹、", "似虎豹。"),
            ("似虎豹、似雷鸣、", "似虎豹、似雷鸣、", "似虎豹。似雷鸣。"),
            ("似虎豹、", "似虎豹、", "似虎猫。"),
        ):
            with self.subTest(source=source, observed=observed):
                self.assertIsNone(self.certify(source, target, observed))
        self.assertIsNone(self.certify("似虎豹、", "似虎豹、", "似虎豹。", identity="other"))
        self.assertIsNone(self.certify("似虎豹、", "似虎豹、", "似虎豹。",
                                      policy=LEGACY_SOURCE_MAPPING_POLICY))

    def test_ch03_partition_and_persisted_provenance(self):
        source = "有一道似虎豹、似雷鸣般的低沉声响。"
        class NativeCaseFrontend(FixtureFrontend):
            def normalize(self, text):
                return {
                    source: ["有一道似虎豹、", "似雷鸣般的低沉声响。"],
                    "有一道似虎豹": ["有一道似虎豹。"],
                    "有一道似虎豹、": ["有一道似虎豹。"],
                    "似雷鸣般的低沉声响。": ["似雷鸣般的低沉声响。"],
                }.get(text, ["不匹配。"])
        frontend = NativeCaseFrontend()
        identity_sha = canonical_sha256(frontend.initialize()["identity"])
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary, patch(
            "src.audiobook.unit_planning.HAN_SUFFIX_FRONTEND_IDENTITY_SHA256", identity_sha
        ):
            root = Path(temporary)
            path = root / "source.txt"
            path.write_bytes(source.encode("utf-8"))
            run, original = create_planning_run(path, "chapter", "case", root / "runs")
            with self.assertRaisesRegex(SynthesisUnitPlanningError, "cannot be mapped"):
                build_synthesis_unit_plan(original, frontend,
                                          mapping_policy=LEGACY_SOURCE_MAPPING_POLICY)
            plan = build_synthesis_unit_plan(original, frontend)
            validate_synthesis_unit_plan(run, plan)
            units = plan["scenes"][0]["synthesis_units"]
            self.assertEqual([u["source_text"] for u in units],
                             ["有一道似虎豹、", "似雷鸣般的低沉声响。"])
            self.assertEqual(units[0]["normalized_text"], "有一道似虎豹、")
            units[0]["mapping"]["normalization_equivalence"]["source_character_offset"] -= 1
            with self.assertRaisesRegex(SynthesisUnitPlanningError, "mapping provenance"):
                validate_synthesis_unit_plan(run, plan)

    def test_equivalent_duplicate_partitions_still_fail(self):
        class AmbiguousFrontend(FixtureFrontend):
            def normalize(self, text):
                if text in {"甲、", "甲、乙、"}:
                    return ["甲。"]
                if text in {"乙、丙。", "丙。"}:
                    return ["丙。"]
                return ["不匹配。"]
        frontend = AmbiguousFrontend()
        identity = frontend.initialize()["identity"]
        with patch("src.audiobook.unit_planning.HAN_SUFFIX_FRONTEND_IDENTITY_SHA256",
                   canonical_sha256(identity)):
            with self.assertRaisesRegex(SynthesisUnitPlanningError, "ambiguous"):
                _find_unique_partition("scene", "甲、乙、丙。", ["甲、", "丙。"],
                                       frontend, identity)

    def test_legacy_plan_remains_valid_and_mapping_unchanged(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            path = root / "source.txt"
            path.write_bytes("甲。乙。".encode("utf-8"))
            run, original = create_planning_run(path, "chapter", "legacy", root / "runs")
            old = build_synthesis_unit_plan(original, FixtureFrontend(),
                                            mapping_policy=LEGACY_SOURCE_MAPPING_POLICY)
            before = json.dumps(old, sort_keys=True)
            validate_synthesis_unit_plan(run, old)
            self.assertEqual(json.dumps(old, sort_keys=True), before)
            new = build_synthesis_unit_plan(original, FixtureFrontend())
            for a, b in zip(old["scenes"][0]["synthesis_units"],
                            new["scenes"][0]["synthesis_units"]):
                b["mapping"]["policy"] = LEGACY_SOURCE_MAPPING_POLICY
                self.assertEqual(a, b)


class GuardedSourceMappingTests(unittest.TestCase):
    def compare_partition(self, source, units=None):
        frontend = FixtureFrontend()
        units = units or frontend.normalize(source.replace("\r", ""))
        exhaustive = {}
        guarded = {}
        old_error = new_error = None
        try:
            old = _find_unique_partition(
                "scene_0001", source, units, frontend, diagnostics=exhaustive
            )[0]
        except SynthesisUnitPlanningError as error:
            old_error = error
        with patch(
            "src.audiobook.unit_planning._supports_han_suffix_mapping",
            return_value=True,
        ):
            try:
                new = _find_unique_partition(
                    "scene_0001", source, units, frontend,
                    frontend_identity={}, diagnostics=guarded,
                )[0]
            except SynthesisUnitPlanningError as error:
                new_error = error
        self.assertEqual(type(old_error), type(new_error))
        if old_error is not None:
            self.assertEqual(
                "ambiguous" in str(old_error), "ambiguous" in str(new_error)
            )
        else:
            self.assertEqual(old, new)
        return guarded

    def test_unique_partition_with_single_and_multiple_boundary_candidates(self):
        for source in ("甲。乙。", "甲。‘乙。", "相同。相同。相同。"):
            with self.subTest(source=source):
                counters = self.compare_partition(source)
                self.assertTrue(counters["guarded_pruning_used"])
                self.assertEqual(counters["fallback_exhaustive_calls"], 0)

    def test_ambiguity_and_dead_end_match_exhaustive_search(self):
        ambiguous = self.compare_partition("甲。（）甲。")
        self.assertTrue(ambiguous["guarded_pruning_used"])
        self.assertEqual(ambiguous["fallback_exhaustive_calls"], 0)
        missing = self.compare_partition("甲。乙。", ["甲。", "丙。"])
        self.assertTrue(missing["fallback_exhaustive_used"])

    def test_possible_early_numeric_cut_uses_exhaustive_search(self):
        counters = self.compare_partition("甲。1乙。")
        self.assertFalse(counters["guarded_pruning_used"])
        self.assertGreater(counters["fallback_exhaustive_calls"], 0)

    def test_guard_accepts_only_supported_input_characters_and_short_numeric_prefix(self):
        self.assertTrue(_han_suffix_input_is_safe("第1章。\r\n“甲”?!乙。"))
        for source in (
            "甲。乙1。", "第１章。乙。", "第²章。乙。", "甲ABC乙。",
            "甲<|control|>乙。", "甲㍿乙。", "甲。" * 40 + "1乙。",
        ):
            with self.subTest(source=source):
                self.assertFalse(_han_suffix_input_is_safe(source))

    def test_unpinned_frontend_keeps_exhaustive_behavior(self):
        frontend = FixtureFrontend()
        counters = {}
        _find_unique_partition(
            "scene_0001", "甲。乙。", frontend.normalize("甲。乙。"),
            frontend, frontend.initialize()["identity"], counters,
        )
        self.assertFalse(counters["guarded_pruning_used"])
        self.assertEqual(counters["candidate_span_frontend_calls"], 0)
        self.assertGreater(counters["fallback_exhaustive_calls"], 0)

    def test_plan_identity_and_source_fidelity_match_exhaustive(self):
        source = "甲。\r\n‘乙。\r\n相同。相同。\r\n"
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            plans = []
            for name, guarded in (("old", False), ("new", True)):
                source_path = root / f"{name}.txt"
                source_path.write_bytes(b"\xef\xbb\xbf" + source.encode("utf-8"))
                run_dir, _ = create_planning_run(
                    source_path, "chapter_0001", name, root / "outputs"
                )
                counters = {}
                with patch(
                    "src.audiobook.unit_planning._supports_han_suffix_mapping",
                    return_value=guarded,
                ):
                    plan = prepare_synthesis_unit_run(
                        run_dir, FixtureFrontend(), diagnostics=counters
                    )
                units = plan["scenes"][0]["synthesis_units"]
                self.assertEqual("".join(unit["source_text"] for unit in units), source)
                snapshot = (run_dir / "source.txt").read_bytes()
                for unit in units:
                    span = unit["source_span"]
                    self.assertEqual(
                        snapshot[span["start_byte"]:span["end_byte"]],
                        unit["source_text"].encode("utf-8"),
                    )
                self.assertEqual(counters["authoritative_frontend_calls"], 1)
                self.assertEqual(
                    counters["final_independent_verification_calls"], len(units)
                )
                plans.append(plan)
            self.assertEqual(
                plans[0]["synthesis_unit_plan"]["ordered_unit_plan_sha256"],
                plans[1]["synthesis_unit_plan"]["ordered_unit_plan_sha256"],
            )
            self.assertEqual(
                plans[0]["scenes"][0]["synthesis_units"],
                plans[1]["scenes"][0]["synthesis_units"],
            )

    def test_certified_long_text_candidate_calls_grow_with_units(self):
        for count in (100, 200):
            with self.subTest(count=count):
                source = "甲。" * count
                frontend = FixtureFrontend()
                counters = {}
                with patch(
                    "src.audiobook.unit_planning._supports_han_suffix_mapping",
                    return_value=True,
                ):
                    spans, calls = _find_unique_partition(
                        "scene_0001", source, frontend.normalize(source),
                        frontend, {}, counters,
                    )
                self.assertEqual(len(spans), count)
                self.assertLessEqual(calls, count * 2)
                self.assertEqual(counters["fallback_exhaustive_calls"], 0)


class SavedFrontendContractTests(unittest.TestCase):
    def test_saved_contract_fixture_is_well_formed_and_not_the_mapping_double(self):
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertTrue(fixture["description"])
        for case in fixture["cases"]:
            with self.subTest(case=case["id"]):
                self.assertTrue(case["source"])
                self.assertTrue(case["normalized_units"])
                self.assertTrue(all(case["normalized_units"]))
        historical = next(
            case for case in fixture["cases"]
            if case["id"] == "historical_clean12_first_two"
        )
        boundaries = (0, 92, 191)
        self.assertEqual(len(historical["source"]), boundaries[-1])
        for index in range(2):
            piece = historical["source"][boundaries[index]:boundaries[index + 1]]
            self.assertEqual(
                hashlib.sha256(piece.encode("utf-8")).hexdigest(),
                historical["historical_source_text_sha256"][index],
            )
            self.assertEqual(
                hashlib.sha256(
                    historical["normalized_units"][index].encode("utf-8")
                ).hexdigest(),
                historical["historical_normalized_text_sha256"][index],
            )

    @unittest.skipUnless(
        os.environ.get("RUN_COSYVOICE_FRONTEND_CONTRACT") == "1",
        "requires the pinned WSL CosyVoice frontend; no synthesis is performed",
    )
    def test_installed_frontend_matches_saved_contract(self):
        cosyvoice_root = Path(os.environ["COSYVOICE_ROOT"])
        model_dir = Path(os.environ["COSYVOICE_MODEL_DIR"])
        frontend = CosyVoiceFrontendAdapter(cosyvoice_root, model_dir)
        frontend.initialize()
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        for case in fixture["cases"]:
            with self.subTest(case=case["id"]):
                clean = case["source"].replace("\r", "")
                self.assertEqual(frontend.normalize(clean), case["normalized_units"])
                if case["id"] == "historical_clean12_first_two":
                    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
                        temporary = Path(temporary)
                        source = temporary / "historical_excerpt.txt"
                        source.write_bytes(case["source"].encode("utf-8"))
                        run_dir, _ = create_planning_run(
                            source, "historical_excerpt", "unit_contract",
                            temporary / "outputs",
                        )
                        prepared = prepare_synthesis_unit_run(run_dir, frontend)
                        self.assertEqual(
                            [unit["source_text_sha256"] for unit in
                             prepared["scenes"][0]["synthesis_units"]],
                            case["historical_source_text_sha256"],
                        )


class CosyVoiceFrontendAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cosyvoice_root = self.root / "CosyVoice"
        self.model_dir = self.cosyvoice_root / "pretrained_models/model"
        (self.cosyvoice_root / "cosyvoice/cli").mkdir(parents=True)
        (self.cosyvoice_root / "cosyvoice/utils").mkdir(parents=True)
        (self.cosyvoice_root / "cosyvoice/tokenizer").mkdir(parents=True)
        self.model_dir.mkdir(parents=True)
        (self.cosyvoice_root / "cosyvoice/cli/frontend.py").write_text(
            "# pinned frontend\n", encoding="utf-8"
        )
        (self.cosyvoice_root / "cosyvoice/utils/frontend_utils.py").write_text(
            "# pinned utilities\n", encoding="utf-8"
        )
        (self.cosyvoice_root / "cosyvoice/tokenizer/tokenizer.py").write_text(
            "# pinned tokenizer\n", encoding="utf-8"
        )
        (self.model_dir / "cosyvoice3.yaml").write_text(
            "model: fixture\n", encoding="utf-8"
        )

    def test_adapter_records_frontend_identity_and_never_calls_inference(self):
        tokenizer = Mock()
        tokenizer.get_vocab.return_value = {"甲": 1, "。": 2}
        frontend = Mock(tokenizer=Mock(tokenizer=tokenizer))
        frontend.text_normalize.return_value = ["甲。"]
        model = Mock(frontend=frontend)
        torch = Mock(__version__="test-torch")
        torch.version.cuda = "test-cuda"
        torch.cuda.is_available.return_value = True
        torchaudio = Mock(__version__="test-audio")
        assets = {
            "zh_tagger": self.root / "zh_tagger.fst",
            "zh_verbalizer": self.root / "zh_verbalizer.fst",
            "en_tagger": self.root / "en_tagger.fst",
            "en_verbalizer": self.root / "en_verbalizer.fst",
        }
        for path in assets.values():
            path.write_bytes(path.name.encode("ascii"))
        adapter = CosyVoiceFrontendAdapter(
            self.cosyvoice_root, self.model_dir
        )
        with (
            patch(
                "src.audiobook.cosyvoice.load_cosyvoice_runtime",
                return_value=(torch, torchaudio, Mock(return_value=model)),
            ),
            patch(
                "src.audiobook.cosyvoice.configure_pinned_wetext_frontend",
                return_value=assets,
            ),
            patch("src.audiobook.cosyvoice.git_head", return_value="abc123"),
            patch("src.audiobook.cosyvoice.version", return_value="fixture-wetext"),
        ):
            metadata = adapter.initialize()
            result = adapter.normalize("甲。")

        self.assertEqual(result, ["甲。"])
        self.assertEqual(metadata["inference_calls"], 0)
        self.assertEqual(
            metadata["identity"]["text_preprocessing_policy"],
            "remove_u000d_v1",
        )
        self.assertEqual(
            set(metadata["identity"]["wetext_asset_sha256"]), set(assets)
        )
        model.inference_zero_shot.assert_not_called()
        frontend.text_normalize.assert_called_once_with(
            "甲。", split=True, text_frontend=True
        )


if __name__ == "__main__":
    unittest.main()
