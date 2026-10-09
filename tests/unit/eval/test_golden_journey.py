"""Focused contract tests for golden Journey drift reporting."""

from __future__ import annotations

import json
import runpy
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from openmed.eval.golden_journey import (
    GoldenJourneyError,
    JourneyPrivacySource,
    JourneyPrivacySpan,
    JourneyPrivacyTransformation,
    load_golden_journey_scenario,
    render_semantic_diff,
    semantic_diff,
    verify_golden_journey_privacy,
    verify_journey_privacy_consistency,
)
from tests.fixtures.journey_release import (
    privacy_control_processor,
    privacy_control_sources,
)

ROOT = Path(__file__).resolve().parents[3]
SHIFT_SECRET = b"synthetic-date-shift-secret-key-32-bytes"
PROOF_SECRET = b"synthetic-evidence-proof-secret-32-bytes"
PATIENT_KEY = "synthetic-patient-key-never-in-evidence"


def _privacy(sources=None, processors=None, **kwargs):
    sources = privacy_control_sources() if sources is None else sources
    if processors is None:
        processors = {item.format: privacy_control_processor for item in sources}
    return verify_journey_privacy_consistency(
        sources,
        patient_key=kwargs.pop("patient_key", PATIENT_KEY),
        date_shift_secret=kwargs.pop("date_shift_secret", SHIFT_SECRET),
        proof_secret=kwargs.pop("proof_secret", PROOF_SECRET),
        processors=processors,
        **kwargs,
    )


def test_five_format_controls_use_one_shift_and_subject_surrogate() -> None:
    result = _privacy()
    assert result.state == "success"
    assert result.lane_metrics() == {
        "date_shift_inconsistency_count": 0,
        "surrogate_inconsistency_count": 0,
        "consistency_checked_format_count": 5,
        "consistency_unsupported_format_count": 0,
    }
    assert {item.date_witness_count for item in result.formats} == {2}
    assert len({item.date_shift_digest for item in result.formats}) == 1
    assert len({item.shifted_interval_digest for item in result.formats}) == 1
    assert len({item.surrogate_set_digest for item in result.formats}) == 1
    assert _privacy().to_dict() == result.to_dict()
    rendered = json.dumps(result.to_dict()) + repr(result)
    for value in (
        PATIENT_KEY,
        "SYNTH-001",
        "2026-01-02",
        "20260102",
        SHIFT_SECRET.decode(),
        PROOF_SECRET.decode(),
        "PATIENT_",
    ):
        assert value not in rendered


def test_frozen_golden_sources_are_unchanged_and_unwitnessed_formats_unsupported() -> (
    None
):
    path = ROOT / "tests/fixtures/journey/v3/scenario.json"
    before = path.read_bytes()
    scenario = load_golden_journey_scenario(path)
    result = verify_golden_journey_privacy(
        scenario,
        patient_key=PATIENT_KEY,
        date_shift_secret=SHIFT_SECRET,
        proof_secret=PROOF_SECRET,
        processors={},
    )
    assert result.state == "unsupported"
    assert result.lane_metrics()["date_shift_inconsistency_count"] == 0
    assert result.lane_metrics()["surrogate_inconsistency_count"] == 0
    assert result.lane_metrics()["consistency_checked_format_count"] == 0
    assert result.lane_metrics()["consistency_unsupported_format_count"] == 5
    assert all(not item.date_shift_digest for item in result.formats)
    assert path.read_bytes() == before
    assert scenario == json.loads(before)


@pytest.mark.parametrize("format_name", ["text", "fhir_r4", "hl7v2", "csv", "dicom_sr"])
def test_missing_processor_never_receives_consistency_credit(format_name) -> None:
    sources = privacy_control_sources()
    processors = {
        item.format: privacy_control_processor
        for item in sources
        if item.format != format_name
    }
    result = _privacy(processors=processors)
    assert result.state == "partial"
    assert result.lane_metrics()["consistency_unsupported_format_count"] == 1
    assert (
        next(item for item in result.formats if item.format == format_name).state
        == "unsupported"
    )


@pytest.mark.parametrize("kind", ["offset", "interval", "surrogate", "unchanged"])
def test_actual_replacement_mismatch_is_detected(kind) -> None:
    def bad(source, context):
        transformed = privacy_control_processor(source, context)
        if source.format != "fhir_r4":
            return transformed
        if kind in {"offset", "interval"}:
            text = transformed.text
            dates = transformed.dates if kind == "offset" else transformed.dates[-1:]
            for span in reversed(dates):
                shifted = date.fromisoformat(
                    text[span.replacement_start : span.replacement_end]
                )
                replacement = (shifted + timedelta(days=1)).isoformat()
                text = (
                    text[: span.replacement_start]
                    + replacement
                    + text[span.replacement_end :]
                )
            return replace(transformed, text=text)
        span = transformed.identifiers[0]
        replacement = "SYNTH-001" if kind == "unchanged" else "PRIVATE-SURROGATE-MARKER"
        old_length = span.replacement_end - span.replacement_start
        replacement = replacement.ljust(old_length, "X")[:old_length]
        return replace(
            transformed,
            text=transformed.text[: span.replacement_start]
            + replacement
            + transformed.text[span.replacement_end :],
        )

    processors = {item.format: bad for item in privacy_control_sources()}
    result = _privacy(processors=processors)
    assert result.state == "failure"
    metric = (
        "date_shift_inconsistency_count"
        if kind in {"offset", "interval"}
        else "surrogate_inconsistency_count"
    )
    assert result.lane_metrics()[metric] == 1
    assert "PRIVATE-SURROGATE-MARKER" not in json.dumps(result.to_dict())


@pytest.mark.parametrize("kind", ["no_key", "no_dates", "no_identifiers", "month_only"])
def test_incomplete_witnesses_are_unsupported(kind) -> None:
    def incomplete(source, context):
        transformed = privacy_control_processor(source, context)
        if kind == "no_key":
            return replace(transformed, patient_keyed=False)
        if kind == "no_dates":
            return replace(transformed, dates=())
        if kind == "no_identifiers":
            return replace(transformed, identifiers=())
        span = transformed.dates[0]
        return replace(transformed, dates=(replace(span, end=span.end - 3),))

    result = _privacy(
        processors={item.format: incomplete for item in privacy_control_sources()}
    )
    assert result.state == "unsupported"
    assert result.lane_metrics()["consistency_checked_format_count"] == 0
    assert all(not item.shifted_interval_digest for item in result.formats)


@pytest.mark.parametrize(
    "kind",
    ["exception", "wrong_type", "bad_offsets", "overlap", "bad_label", "large_output"],
)
def test_bad_processors_fail_without_echoing_protected_errors(kind) -> None:
    marker = "PRIVATE-PROCESSOR-ERROR-MARKER"

    def bad(source, context):
        if kind == "exception":
            raise RuntimeError(marker)
        if kind == "wrong_type":
            return {"text": marker}
        transformed = privacy_control_processor(source, context)
        span = transformed.identifiers[0]
        if kind == "bad_offsets":
            return replace(transformed, identifiers=(replace(span, start=True),))
        if kind == "overlap":
            return replace(transformed, identifiers=(span, span))
        if kind == "bad_label":
            return replace(transformed, identifiers=(replace(span, label=marker),))
        return replace(transformed, text=marker * 100_000)

    result = _privacy(
        processors={item.format: bad for item in privacy_control_sources()}
    )
    assert result.state == "failure"
    assert result.lane_metrics()["consistency_unsupported_format_count"] == 5
    assert marker not in json.dumps(result.to_dict()) + repr(result)


@pytest.mark.parametrize(
    "field,value",
    [
        ("patient_key", ""),
        ("patient_key", "\ud800"),
        ("date_shift_secret", b"short"),
        ("date_shift_secret", "private-key"),
        ("proof_secret", SHIFT_SECRET),
        ("proof_secret", b"short"),
        ("date_shift_max_days", True),
        ("date_shift_max_days", 0),
        ("date_shift_max_days", 3651),
    ],
)
def test_invalid_private_inputs_have_fixed_value_free_errors(field, value) -> None:
    with pytest.raises(
        GoldenJourneyError, match="^invalid Journey privacy consistency inputs$"
    ) as captured:
        _privacy(**{field: value})
    assert captured.value.__cause__ is None


@pytest.mark.parametrize(
    "kind", ["missing", "duplicate", "reordered", "unknown", "bad_text"]
)
def test_exact_five_source_contract_rejects_invalid_coverage(kind) -> None:
    sources = list(privacy_control_sources())
    if kind == "missing":
        sources.pop()
    elif kind == "duplicate":
        sources[-1] = sources[0]
    elif kind == "reordered":
        sources.reverse()
    elif kind == "unknown":
        sources[-1] = JourneyPrivacySource("PRIVATE-FORMAT", "PRIVATE-TEXT")
    else:
        sources[-1] = JourneyPrivacySource("dicom_sr", "\ud800")
    with pytest.raises(
        GoldenJourneyError, match="^invalid Journey privacy consistency inputs$"
    ):
        _privacy(sources=sources, processors={})


def test_separate_proof_key_rekeys_evidence_without_changing_consistency() -> None:
    first = _privacy()
    second = _privacy(proof_secret=b"another-synthetic-proof-key-at-least-32-bytes")
    assert first.patient_digest != second.patient_digest
    assert first.formats[0].source_digest != second.formats[0].source_digest
    assert first.lane_metrics() == second.lane_metrics()


def test_semantic_diff_is_path_sorted_and_machine_readable() -> None:
    expected = {"a": [1, {"x": "old"}], "removed": True}
    actual = {"a": [1, {"x": "new"}, 3], "added": False}

    differences = semantic_diff(expected, actual)

    assert [item["path"] for item in differences] == [
        "/a/1/x",
        "/a/2",
        "/added",
        "/removed",
    ]
    assert render_semantic_diff(expected, actual).splitlines()[0] == (
        '{"actual":"new","expected":"old","path":"/a/1/x"}'
    )


def test_semantic_diff_escapes_json_pointer_tokens() -> None:
    assert semantic_diff({"a/b~c": 1}, {"a/b~c": 2}) == [
        {"actual": 2, "expected": 1, "path": "/a~1b~0c"}
    ]


def test_regeneration_defaults_to_read_only_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = ROOT / "scripts" / "regenerate_v3_golden_journey.py"
    golden = ROOT / "tests" / "fixtures" / "journey" / "v3" / "golden.json"
    before = golden.read_bytes()
    monkeypatch.setattr(sys, "argv", [str(script)])

    with pytest.raises(SystemExit) as captured:
        runpy.run_path(str(script), run_name="__main__")

    assert captured.value.code == 0
    assert golden.read_bytes() == before
