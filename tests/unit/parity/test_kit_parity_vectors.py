"""Python gate for the OpenMedKit kit-parity result profile vectors."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from openmed.core import kit_parity
from openmed.core.labels import CANONICAL_LABELS
from openmed.core.pii import DeidentificationResult, PIIEntity

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "parity" / "update_native_span_fixture.py"
VECTORS_FILE = ROOT / "tests" / "fixtures" / "parity" / "kit_parity_vectors.json"
MANIFEST_FILE = ROOT / "fixtures" / "parity" / "span_parity_manifest.json"
PARITY_DOC = ROOT / "docs" / "swift-kotlin-parity.md"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
SWIFT_WORKFLOW = ROOT / ".github" / "workflows" / "swift-test.yml"
ANDROID_WORKFLOW = ROOT / ".github" / "workflows" / "android-ci.yml"

SPEC = importlib.util.spec_from_file_location("update_native_span_fixture", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
fixture_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fixture_tool
SPEC.loader.exec_module(fixture_tool)

VECTORS = json.loads(VECTORS_FILE.read_text(encoding="utf-8"))
SENTINEL = VECTORS["profile"]["privacy_sentinel"]


def _vector(vector_id: str) -> dict:
    return next(item for item in VECTORS["vectors"] if item["id"] == vector_id)


def _spans(vector: dict) -> tuple[kit_parity.KitParitySpan, ...]:
    return tuple(
        kit_parity.KitParitySpan(
            label=span["label"],
            canonical_label=span["canonical_label"],
            action=span["action"],
            start=span["start"],
            end=span["end"],
            confidence_ppm=span.get("confidence_ppm"),
            replacement=span.get("replacement"),
        )
        for span in vector["spans"]
    )


def _differences(vector_id: str, expected: str, actual: str) -> list[str]:
    """Value-free drift report: case id, field path, offsets, labels, actions."""
    if expected.encode("utf-8") == actual.encode("utf-8"):
        return []
    want = json.loads(expected)
    got = json.loads(actual)
    out: list[str] = []
    if want["deidentified_text"] != got["deidentified_text"]:
        left = want["deidentified_text"].encode("utf-8")
        right = got["deidentified_text"].encode("utf-8")
        offset = next(
            (i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
            min(len(left), len(right)),
        )
        out.append(
            f"case={vector_id} path=$.deidentified_text first_differing_utf8_byte="
            f"{offset}"
        )
    if len(want["spans"]) != len(got["spans"]):
        out.append(
            f"case={vector_id} path=$.spans count expected={len(want['spans'])} "
            f"actual={len(got['spans'])}"
        )
        return out
    for index, (left, right) in enumerate(zip(want["spans"], got["spans"])):
        for field in sorted(set(left) | set(right)):
            if left.get(field) == right.get(field):
                continue
            path = f"case={vector_id} path=$.spans[{index}].{field}"
            if field == "replacement":
                out.append(path)
            else:
                out.append(
                    f"{path} expected={left.get(field)} actual={right.get(field)}"
                )
    return out or [f"case={vector_id} path=$ bytes differ"]


def test_vectors_are_current_and_registered_in_the_span_manifest() -> None:
    ids = fixture_tool.check_kit_vectors(VECTORS_FILE)
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    entry = manifest["kit_parity"]

    assert ids == [vector["id"] for vector in VECTORS["vectors"]]
    assert len(ids) >= 13
    assert entry["path"] == "tests/fixtures/parity/kit_parity_vectors.json"
    assert entry["schema"] == VECTORS["schema"] == "openmed.kit_parity_vectors"
    assert entry["version"] == VECTORS["version"] == 1
    assert entry["profile"] == kit_parity.PROFILE_NAME
    assert entry["profile_version"] == kit_parity.PROFILE_VERSION
    assert entry["sha256"] == hashlib.sha256(VECTORS_FILE.read_bytes()).hexdigest()
    assert fixture_tool.main(["--check"]) == 0


def test_profile_block_freezes_the_python_reference() -> None:
    profile = VECTORS["profile"]

    assert profile == fixture_tool.expected_kit_profile()
    assert tuple(profile["actions"]) == kit_parity.KIT_ACTIONS
    assert list(kit_parity.KIT_ACTIONS) == sorted(kit_parity.KIT_ACTIONS)
    assert list(kit_parity.KIT_CANONICAL_LABELS) == sorted(
        kit_parity.KIT_CANONICAL_LABELS
    )
    assert set(kit_parity.KIT_CANONICAL_LABELS) <= set(CANONICAL_LABELS)
    assert "OTHER" in kit_parity.KIT_CANONICAL_LABELS


def test_every_kit_vocabulary_stays_inside_the_profile() -> None:
    labels = set(kit_parity.KIT_CANONICAL_LABELS)
    actions = set(kit_parity.KIT_ACTIONS)

    kotlin = (
        ROOT / "android/openmedkit/src/main/kotlin/com/openmed/openmedkit/policy/"
        "PolicyProfiles.kt"
    ).read_text(encoding="utf-8")
    kotlin_labels = re.search(
        r"private val CANONICAL_LABELS = setOf\((.*?)\)", kotlin, re.S
    )
    kotlin_actions = re.findall(r'^\s+[A-Z]+\("([a-z_]+)"\),', kotlin, re.M)
    swift = (ROOT / "swift/OpenMedKit/Sources/OpenMedKit/Policy.swift").read_text(
        encoding="utf-8"
    )
    swift_labels = re.search(
        r"private static let canonicalLabels: Set<String> = \[(.*?)\]", swift, re.S
    )
    swift_actions = re.search(
        r"public enum PolicyAction.*?\{(.*?)\n\s*var", swift, re.S
    )
    web = (ROOT / "js/openmedkit-web/src/types.ts").read_text(encoding="utf-8")
    web_labels = re.search(r"CANONICAL_LABELS = \[(.*?)\] as const", web, re.S)
    native = (ROOT / "js/openmedkit-react-native/src/index.ts").read_text("utf-8")
    rn_actions = re.search(r"export type SpanAction =(.*?);", native, re.S)
    assert kotlin_labels and swift_labels and web_labels
    assert swift_actions and rn_actions

    assert set(re.findall(r'"([A-Z_]+)"', kotlin_labels.group(1))) == labels
    assert set(re.findall(r'"([A-Z_]+)"', swift_labels.group(1))) <= labels
    assert set(re.findall(r'"([A-Z_]+)"', web_labels.group(1))) <= labels
    assert set(kotlin_actions) == actions
    assert set(re.findall(r"case (\w+)", swift_actions.group(1))) == actions
    assert set(re.findall(r'"(\w+)"', rn_actions.group(1))) == actions


@pytest.mark.parametrize("vector", VECTORS["vectors"], ids=lambda item: item["id"])
def test_canonical_bytes_round_trip(vector: dict) -> None:
    parsed = kit_parity.parse_result(vector["result_json"])
    parsed_from_bytes = kit_parity.parse_result(vector["result_json"].encode("utf-8"))

    assert parsed == parsed_from_bytes
    assert parsed.spans == _spans(vector)
    assert parsed.deidentified_text == vector["deidentified_text"]
    assert kit_parity.serialize_result(parsed) == vector["result_json"]
    assert (
        hashlib.sha256(vector["result_json"].encode("utf-8")).hexdigest()
        == vector["result_sha256"]
    )
    assert (
        hashlib.sha256(vector["deidentified_text"].encode("utf-8")).hexdigest()
        == vector["deidentified_sha256"]
    )
    kit_parity.verify_result(parsed, vector["text"])


@pytest.mark.parametrize("vector", VECTORS["vectors"], ids=lambda item: item["id"])
def test_offset_conversions_match_derived_offsets(vector: dict) -> None:
    text = vector["text"]
    for span in vector["spans"]:
        for scalar, utf8, utf16 in (
            (span["start"], span["utf8_start"], span["utf16_start"]),
            (span["end"], span["utf8_end"], span["utf16_end"]),
        ):
            assert kit_parity.scalar_to_utf8_offset(text, scalar) == utf8
            assert kit_parity.scalar_to_utf16_offset(text, scalar) == utf16
            assert kit_parity.utf8_to_scalar_offset(text, utf8) == scalar
            assert kit_parity.utf16_to_scalar_offset(text, utf16) == scalar
            assert len(text.encode("utf-8")[:utf8].decode("utf-8")) == scalar


def test_unicode_coverage_includes_every_required_shift() -> None:
    categories = {
        item for vector in VECTORS["vectors"] for item in vector["categories"]
    }
    spans = [span for vector in VECTORS["vectors"] for span in vector["spans"]]

    assert fixture_tool.KIT_REQUIRED_CATEGORIES <= categories
    assert any(span["utf16_start"] > span["start"] for span in spans)
    assert any(span["utf8_start"] > span["utf16_start"] for span in spans)
    assert _vector("empty_input")["text"] == ""
    assert "\r\n" in _vector("newlines_and_escapes")["text"]
    assert "́" in _vector("combining_marks")["text"]
    assert "‍" in _vector("emoji_sequences")["text"]
    assert "\U00020000" in _vector("cjk_supplementary")["text"]
    assert "्" in _vector("indic_conjuncts")["text"]
    adjacent = _vector("adjacent_spans")
    assert adjacent["spans"][0]["end"] == adjacent["spans"][1]["start"]
    assert adjacent["spans"][-1]["end"] == len(adjacent["text"])
    assert {span["action"] for span in spans} == set(kit_parity.KIT_ACTIONS)


@pytest.mark.parametrize("vector", VECTORS["vectors"], ids=lambda item: item["id"])
def test_python_result_projection_matches_vectors(vector: dict) -> None:
    if any("confidence_ppm" not in span for span in vector["spans"]):
        # Python entities always carry a score; absent scores are covered by
        # the round-trip test and the web, React Native, and Electron
        # projections.
        assert vector["id"] == "confidence_encoding"
        return
    entities = [
        PIIEntity(
            text=vector["text"][span["start"] : span["end"]],
            label=span["label"],
            start=span["start"],
            end=span["end"],
            confidence=span["confidence_ppm"] / kit_parity.CONFIDENCE_SCALE,
            canonical_label=span["canonical_label"],
            action=span["action"],
            redacted_text=span.get("replacement"),
        )
        for span in reversed(vector["spans"])
    ]
    result = DeidentificationResult(
        original_text=vector["text"],
        deidentified_text=vector["deidentified_text"],
        pii_entities=entities,
        method="mask",
        timestamp=datetime(2026, 1, 1),
    )

    actual = kit_parity.serialize_result(
        kit_parity.from_deidentification_result(result)
    )

    assert _differences(vector["id"], vector["result_json"], actual) == []


def test_python_projection_rejects_unrepresentable_entities() -> None:
    def project(**changes: object) -> kit_parity.KitParityError:
        entity = PIIEntity(
            text="x",
            label="date",
            start=0,
            end=1,
            confidence=0.5,
            canonical_label="DATE",
            action="mask",
            redacted_text="[DATE]",
        )
        result = DeidentificationResult(
            original_text="x",
            deidentified_text="[DATE]",
            pii_entities=[dataclasses.replace(entity, **changes)],
            method="mask",
            timestamp=datetime(2026, 1, 1),
        )
        with pytest.raises(kit_parity.KitParityError) as error:
            kit_parity.from_deidentification_result(result)
        return error.value

    assert project(action="shift_dates").code == "action_unknown"
    assert project(action="format_preserve").field_path == "$.spans[0].action"
    assert project(action=None).code == "field_missing"
    assert project(canonical_label="DISEASE").code == "label_unknown"
    assert project(redacted_text=None).field_path == "$.spans[0].replacement"
    assert project(confidence=1.5).field_path == "$.spans[0].confidence_ppm"
    assert project(start=None).field_path == "$.spans[0].start"
    assert project(label="x y").code == "label_invalid"


@pytest.mark.parametrize("record", VECTORS["rejections"], ids=lambda item: item["id"])
def test_rejections_fail_with_documented_value_free_reasons(record: dict) -> None:
    with pytest.raises(kit_parity.KitParityError) as error:
        kit_parity.parse_result(record["input"])

    assert (error.value.code, error.value.field_path) == (
        record["reason"],
        record["field_path"],
    )
    assert str(error.value) == f"{record['reason']} at {record['field_path']}"
    assert SENTINEL not in str(error.value)


@pytest.mark.parametrize(
    "record", VECTORS["source_rejections"], ids=lambda item: item["id"]
)
def test_source_rejections_fail_verification(record: dict) -> None:
    parsed = kit_parity.parse_result(record["input"])

    with pytest.raises(kit_parity.KitParityError) as error:
        kit_parity.verify_result(parsed, record["text"])

    assert (error.value.code, error.value.field_path) == (
        record["reason"],
        record["field_path"],
    )


@pytest.mark.parametrize(
    "record", VECTORS["offset_rejections"], ids=lambda item: item["id"]
)
def test_offset_conversion_rejections(record: dict) -> None:
    convert = {
        "scalar": kit_parity.scalar_to_utf16_offset,
        "utf8": kit_parity.utf8_to_scalar_offset,
        "utf16": kit_parity.utf16_to_scalar_offset,
    }[record["unit"]]

    with pytest.raises(kit_parity.KitParityError) as error:
        convert(record["text"], record["offset"])

    assert error.value.code == record["reason"]


@pytest.mark.parametrize(
    "record", VECTORS["compatibility"], ids=lambda item: item["id"]
)
def test_bundle_contracts_are_gated_before_results_are_accepted(record: dict) -> None:
    payload = _vector("emoji_sequences")["result_json"]
    versions = {
        "offset_contract_version": record["offset_contract_version"],
        "decoder_contract_version": record["decoder_contract_version"],
    }
    if record["outcome"] == "accepted":
        assert kit_parity.accept_result(payload, **versions) == (
            kit_parity.parse_result(payload)
        )
        return

    for candidate in (payload, "not json", VECTORS["rejections"][0]["input"]):
        with pytest.raises(kit_parity.KitParityError) as error:
            kit_parity.accept_result(candidate, **versions)
        assert error.value.code == record["outcome"]


def test_profile_compatibility_names_the_committed_offset_contract() -> None:
    offset_contract = json.loads(
        (ROOT / "tests/fixtures/parity/offset_contract.json").read_text("utf-8")
    )

    assert offset_contract["version"] in kit_parity.OFFSET_CONTRACT_VERSIONS
    assert offset_contract["offset_unit"] == kit_parity.OFFSET_UNIT


def test_altered_offset_label_action_order_or_bytes_fail_the_gate() -> None:
    vector = _vector("emoji_sequences")
    expected = vector["result_json"]
    spans = list(_spans(vector))
    text = vector["text"]

    def serialized(changed: list[kit_parity.KitParitySpan], output: str) -> str:
        return kit_parity.serialize_result(
            kit_parity.KitParityResult(deidentified_text=output, spans=tuple(changed))
        )

    shifted = [dataclasses.replace(spans[0], end=spans[0].end + 1), *spans[1:]]
    relabeled = [dataclasses.replace(spans[0], canonical_label="PERSON"), *spans[1:]]
    reaction = [spans[0], dataclasses.replace(spans[1], action="redact"), spans[2]]
    output = vector["deidentified_text"]

    assert _differences(vector["id"], expected, serialized(shifted, output)) == [
        f"case=emoji_sequences path=$.spans[0].end expected={spans[0].end} "
        f"actual={spans[0].end + 1}"
    ]
    assert _differences(vector["id"], expected, serialized(relabeled, output)) == [
        "case=emoji_sequences path=$.spans[0].canonical_label expected=OTHER "
        "actual=PERSON"
    ]
    assert _differences(vector["id"], expected, serialized(reaction, output)) == [
        "case=emoji_sequences path=$.spans[1].action expected=mask actual=redact"
    ]
    changed_bytes = serialized(spans, output.replace("[PHONE]", "[PHONE] "))
    assert _differences(vector["id"], expected, changed_bytes)[0].startswith(
        "case=emoji_sequences path=$.deidentified_text first_differing_utf8_byte="
    )
    with pytest.raises(kit_parity.KitParityError) as error:
        kit_parity.verify_result(kit_parity.parse_result(changed_bytes), text)
    assert error.value.code == "deidentified_text_mismatch"

    swapped = json.loads(expected)
    swapped["spans"][0], swapped["spans"][1] = swapped["spans"][1], swapped["spans"][0]
    with pytest.raises(kit_parity.KitParityError) as error:
        kit_parity.parse_result(
            json.dumps(
                swapped, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            )
        )
    assert (error.value.code, error.value.field_path) == (
        "span_order_invalid",
        "$.spans[1]",
    )
    for report in _differences(vector["id"], expected, changed_bytes):
        assert "Bo Reyes" not in report and "555-019-4410" not in report


def test_serializer_rejects_invalid_results_before_emitting_bytes() -> None:
    span = kit_parity.KitParitySpan("date", "DATE", "mask", 0, 4, 500000, "[DATE]")

    for changed, code in (
        (dataclasses.replace(span, canonical_label="date"), "label_unknown"),
        (dataclasses.replace(span, action="tokenize"), "action_unknown"),
        (dataclasses.replace(span, replacement=None), "field_missing"),
        (dataclasses.replace(span, confidence_ppm=-1), "confidence_invalid"),
        (dataclasses.replace(span, end=0), "offset_invalid"),
    ):
        with pytest.raises(kit_parity.KitParityError) as error:
            kit_parity.serialize_result(kit_parity.KitParityResult("x", (changed,)))
        assert error.value.code == code


def test_confidence_encoding_rounds_half_up_in_parts_per_million() -> None:
    assert kit_parity.confidence_to_ppm(0.0) == 0
    assert kit_parity.confidence_to_ppm(1.0) == 1_000_000
    assert kit_parity.confidence_to_ppm(0.97) == 970_000
    assert kit_parity.confidence_to_ppm(0.0000005) == 1
    for invalid in (-0.1, 1.0000001, float("nan"), float("inf")):
        with pytest.raises(kit_parity.KitParityError):
            kit_parity.confidence_to_ppm(invalid)


def test_manifest_names_every_kit_consumer_and_its_ci_gate() -> None:
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    consumers = manifest["kit_parity"]["consumers"]
    doc = PARITY_DOC.read_text(encoding="utf-8")

    assert {consumer["surface"] for consumer in consumers} == {
        "python",
        "swift",
        "kotlin",
        "web",
        "react-native",
        "electron",
    }
    for consumer in consumers:
        source = (ROOT / consumer["test"]).read_text(encoding="utf-8")
        workflow = (ROOT / consumer["workflow"]).read_text(encoding="utf-8")
        assert "kit_parity_vectors.json" in source
        assert "span_parity_manifest.json" in source or consumer["surface"] in {
            "python"
        }
        assert consumer["gate"] in workflow
        assert consumer["test"] in doc


def test_workflows_trigger_on_and_run_the_kit_parity_gates() -> None:
    vectors_path = "tests/fixtures/parity/kit_parity_vectors.json"
    for path in (SWIFT_WORKFLOW, ANDROID_WORKFLOW):
        workflow = yaml.load(path.read_text("utf-8"), Loader=yaml.BaseLoader)
        for event in ("push", "pull_request"):
            assert vectors_path in workflow["on"][event]["paths"]
    swift = yaml.load(SWIFT_WORKFLOW.read_text("utf-8"), Loader=yaml.BaseLoader)
    swift_steps = [
        step.get("run", "") for step in swift["jobs"]["build-and-test"]["steps"]
    ]
    assert "swift test --filter KitParityVectorTests" in swift_steps
    assert "--tests '*KitParityVectorTest'" in ANDROID_WORKFLOW.read_text("utf-8")

    ci = yaml.load(CI_WORKFLOW.read_text("utf-8"), Loader=yaml.BaseLoader)
    job = ci["jobs"]["openmedkit-kit-parity"]
    runs = "\n".join(step.get("run", "") for step in job["steps"])
    for package in ("openmedkit-web", "openmedkit-react-native", "openmedkit-electron"):
        assert f"npm ci --prefix js/{package}" in runs
        assert f"npm --prefix js/{package} test" in runs
