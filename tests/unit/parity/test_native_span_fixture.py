"""Schema, digest, and drift gates for the shared Swift-Kotlin span fixture."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "parity" / "update_native_span_fixture.py"
FIXTURE_DIR = ROOT / "fixtures" / "parity"
SWIFT_WORKFLOW = ROOT / ".github" / "workflows" / "swift-test.yml"
ANDROID_WORKFLOW = ROOT / ".github" / "workflows" / "android-ci.yml"
ANDROID_BUILD = ROOT / "android" / "openmedkit" / "build.gradle.kts"
PARITY_DOC = ROOT / "docs" / "swift-kotlin-parity.md"

SPEC = importlib.util.spec_from_file_location("update_native_span_fixture", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
fixture_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fixture_tool
SPEC.loader.exec_module(fixture_tool)

FIXTURE_FILES = (
    fixture_tool.SPANS_FILE,
    fixture_tool.MANIFEST_FILE,
    "synthetic_clinical_note.txt",
)


@pytest.fixture
def fixture_copy(tmp_path: Path) -> Path:
    for name in FIXTURE_FILES:
        shutil.copyfile(FIXTURE_DIR / name, tmp_path / name)
    return tmp_path


def _load(directory: Path) -> dict:
    return json.loads((directory / fixture_tool.SPANS_FILE).read_text("utf-8"))


def _dump(directory: Path, payload: dict) -> None:
    (directory / fixture_tool.SPANS_FILE).write_text(
        fixture_tool.canonical_json(payload), encoding="utf-8"
    )


def _case(payload: dict, case_id: str) -> dict:
    return next(case for case in payload["cases"] if case["id"] == case_id)


def test_committed_fixture_schema_derived_fields_and_digest_are_current() -> None:
    cases = fixture_tool.check_fixture(FIXTURE_DIR)

    assert [case.case_id for case in cases] == [
        "clinical_note_ascii",
        "emoji_and_combining_marks",
        "overlapping_identifiers",
        "unicode_without_identifiers",
    ]
    assert fixture_tool.main(["--check"]) == 0


def test_fixture_covers_the_shared_unicode_coordinate_contract() -> None:
    cases = {case.case_id: case for case in fixture_tool.check_fixture(FIXTURE_DIR)}
    unicode_text = cases["emoji_and_combining_marks"].text
    spans = cases["emoji_and_combining_marks"].payload["spans"]

    assert "‍" in unicode_text  # emoji ZWJ sequence
    assert "\U0001f3fd" in unicode_text  # emoji skin-tone modifier
    assert "\U0001f1e8\U0001f1e6" in unicode_text  # regional-indicator flag
    assert "é" in unicode_text  # decomposed combining acute
    assert "ñ̃" in unicode_text  # stacked combining marks
    assert "\U0001fa7aMRN-" in unicode_text  # supplementary emoji touching a span
    assert any(span["utf16_start"] > span["start"] for span in spans)
    assert any(span["utf8_start"] > span["utf16_start"] for span in spans)
    assert cases["unicode_without_identifiers"].payload["spans"] == []
    overlap = cases["overlapping_identifiers"].payload
    assert len(overlap["spans"]) == len(overlap["actions"]) + 1
    assert {action["action"] for action in overlap["actions"]} == {"mask", "keep"}


def test_python_conversions_match_independent_encodings() -> None:
    text = "A\U0001f469\U0001f3fdéB"

    assert fixture_tool.scalar_to_utf16(text, 3) == 5
    assert fixture_tool.scalar_to_utf8(text, 3) == 9
    assert fixture_tool.scalar_to_utf8(text, 5) == 12
    assert fixture_tool.scalar_to_utf16(text, 5) == 7


def test_changed_expected_span_without_digest_refresh_fails(fixture_copy: Path) -> None:
    payload = _load(fixture_copy)
    _case(payload, "emoji_and_combining_marks")["spans"][2]["end"] += 1
    _dump(fixture_copy, payload)

    with pytest.raises(fixture_tool.FixtureError) as error:
        fixture_tool.check_fixture(fixture_copy)

    assert error.value.code == "span_offset_mismatch"
    assert error.value.case_id == "emoji_and_combining_marks"


def test_changed_expected_span_fails_digest_even_when_consistent(
    fixture_copy: Path,
) -> None:
    payload = _load(fixture_copy)
    span = _case(payload, "clinical_note_ascii")["spans"][0]
    span["label"] = "first_name"
    _dump(fixture_copy, payload)

    with pytest.raises(fixture_tool.FixtureError) as error:
        fixture_tool.check_fixture(fixture_copy)

    # The action no longer pairs with a span, and the digest is stale either way.
    assert error.value.code in {"action_without_span", "digest_mismatch"}

    _case(payload, "clinical_note_ascii")["actions"][0]["label"] = "first_name"
    _dump(fixture_copy, payload)
    with pytest.raises(fixture_tool.FixtureError) as error:
        fixture_tool.check_fixture(fixture_copy)
    assert error.value.code == "digest_mismatch"
    assert error.value.detail == fixture_tool.SPANS_FILE


def test_changed_digest_fails(fixture_copy: Path) -> None:
    manifest_path = fixture_copy / fixture_tool.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(fixture_tool.canonical_json(manifest), encoding="utf-8")

    with pytest.raises(fixture_tool.FixtureError) as error:
        fixture_tool.check_fixture(fixture_copy)

    assert error.value.code == "digest_mismatch"


def test_changed_note_bytes_fail_digest(fixture_copy: Path) -> None:
    # Same length and inside a masked span, so only the digest can catch it.
    note = fixture_copy / "synthetic_clinical_note.txt"
    note.write_bytes(note.read_bytes().replace(b"Rivera\n", b"Rivero\n", 1))

    with pytest.raises(fixture_tool.FixtureError) as error:
        fixture_tool.check_fixture(fixture_copy)

    assert error.value.code == "digest_mismatch"
    assert error.value.detail == "synthetic_clinical_note.txt"


def test_changed_note_length_fails_derived_output(fixture_copy: Path) -> None:
    note = fixture_copy / "synthetic_clinical_note.txt"
    note.write_bytes(note.read_bytes().replace(b"Harbor", b"Harbour", 1))

    with pytest.raises(fixture_tool.FixtureError) as error:
        fixture_tool.check_fixture(fixture_copy)

    assert error.value.code == "deidentified_text_mismatch"
    assert error.value.case_id == "clinical_note_ascii"


def test_normalized_or_reformatted_fixture_is_rejected(fixture_copy: Path) -> None:
    path = fixture_copy / fixture_tool.SPANS_FILE
    payload = _load(fixture_copy)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", "utf-8")

    with pytest.raises(fixture_tool.FixtureError) as error:
        fixture_tool.check_fixture(fixture_copy)

    assert error.value.code == "fixture_not_canonical"


def test_wrong_deidentified_text_reports_offsets_not_text(fixture_copy: Path) -> None:
    payload = _load(fixture_copy)
    case = _case(payload, "clinical_note_ascii")
    case["expected_deidentified_text"] = case["expected_deidentified_text"].replace(
        "[EMAIL]", "alex.rivera@example.test"
    )
    _dump(fixture_copy, payload)

    with pytest.raises(fixture_tool.FixtureError) as error:
        fixture_tool.check_fixture(fixture_copy)

    message = str(error.value)
    assert error.value.code == "deidentified_text_mismatch"
    assert "first_differing_utf8_byte=" in message
    assert "rivera" not in message.lower()


def test_write_refreshes_derived_fields_and_digests(fixture_copy: Path) -> None:
    payload = _load(fixture_copy)
    case = _case(payload, "emoji_and_combining_marks")
    for span in case["spans"]:
        span["utf8_start"] = span["utf16_start"] = 0
    case["expected_deidentified_text"] = ""
    _dump(fixture_copy, payload)
    vectors = fixture_copy / "kit_parity_vectors.json"
    shutil.copyfile(fixture_tool.KIT_VECTORS_FILE, vectors)

    assert (
        fixture_tool.main(
            [
                "--write",
                "--fixture-dir",
                str(fixture_copy),
                "--vectors-file",
                str(vectors),
            ]
        )
        == 0
    )

    for name in FIXTURE_FILES:
        assert (fixture_copy / name).read_bytes() == (FIXTURE_DIR / name).read_bytes()
    assert vectors.read_bytes() == fixture_tool.KIT_VECTORS_FILE.read_bytes()


def test_manifest_names_every_consumer_and_its_ci_gate() -> None:
    manifest = json.loads((FIXTURE_DIR / fixture_tool.MANIFEST_FILE).read_text("utf-8"))

    assert manifest["update_command"] == fixture_tool.UPDATE_COMMAND
    assert {consumer["surface"] for consumer in manifest["consumers"]} == {
        "python",
        "swift",
        "kotlin",
    }
    for consumer in manifest["consumers"]:
        source = (ROOT / consumer["test"]).read_text(encoding="utf-8")
        workflow = (ROOT / consumer["workflow"]).read_text(encoding="utf-8")
        assert fixture_tool.SPANS_FILE in source or "SPANS_FILE" in source
        assert fixture_tool.MANIFEST_FILE in source or "MANIFEST_FILE" in source
        assert consumer["gate"] in workflow

    doc = PARITY_DOC.read_text(encoding="utf-8")
    assert fixture_tool.UPDATE_COMMAND in doc
    for consumer in manifest["consumers"]:
        assert consumer["test"] in doc


@pytest.mark.parametrize("workflow_path", [SWIFT_WORKFLOW, ANDROID_WORKFLOW])
def test_native_workflows_trigger_on_root_parity_fixture(workflow_path: Path) -> None:
    workflow = yaml.load(workflow_path.read_text("utf-8"), Loader=yaml.BaseLoader)

    for event in ("push", "pull_request"):
        assert "fixtures/parity/**" in workflow["on"][event]["paths"]
        assert "scripts/parity/**" in workflow["on"][event]["paths"]


def test_native_ci_runs_the_span_equivalence_gates() -> None:
    swift = yaml.load(SWIFT_WORKFLOW.read_text("utf-8"), Loader=yaml.BaseLoader)
    android = ANDROID_WORKFLOW.read_text("utf-8")
    swift_steps = [
        step.get("run", "") for step in swift["jobs"]["build-and-test"]["steps"]
    ]

    assert "swift test --filter SpanEquivalenceParityTests" in swift_steps
    assert "--tests '*SpanEquivalenceTest'" in android
    assert 'resources.srcDir(repoRoot.resolve("fixtures/parity"))' in (
        ANDROID_BUILD.read_text("utf-8")
    )


def test_no_divergent_embedded_fixture_copies() -> None:
    names = set(FIXTURE_FILES)
    skipped = {".build", "build", ".gradle", "node_modules", ".git"}
    copies = []
    for base in (ROOT / "swift", ROOT / "android", ROOT / "tests"):
        for path in base.rglob("*"):
            if path.name in names and not skipped.intersection(path.parts):
                copies.append(path.relative_to(ROOT).as_posix())

    assert copies == []
