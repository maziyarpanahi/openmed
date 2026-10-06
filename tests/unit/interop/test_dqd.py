"""Synthetic DQD normalization, fail-closed semantics and leakage controls."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from openmed.clinical.journey_contracts import canonical_json
from openmed.eval.suites.omop_quality import load_frozen_omop_quality_fixture
from openmed.interop.omop import (
    DQD_SUPPORTED_VERSIONS,
    OmopQualityProtocolError,
    OmopQualityUnsupportedError,
    dqd,
    normalize_dqd_results_file,
    normalize_omop_quality_output,
    verify_omop_quality_report,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "interop" / "omop"
DQD_FIXTURE = FIXTURES / "dqd_results_2_9_0.json"


def _normalize(payload: object, tmp_path: Path) -> dict[str, object]:
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    fixture = load_frozen_omop_quality_fixture(FIXTURES / "quality_reconciliation.json")
    return normalize_dqd_results_file(path, quality_input=fixture.quality_input)


def _payload() -> dict[str, object]:
    return json.loads(DQD_FIXTURE.read_text(encoding="utf-8"))


def test_fixture_is_digest_stable_and_accepted_by_signed_bridge() -> None:
    fixture = load_frozen_omop_quality_fixture(FIXTURES / "quality_reconciliation.json")
    output = normalize_dqd_results_file(
        DQD_FIXTURE, quality_input=fixture.quality_input
    )
    assert output == normalize_dqd_results_file(
        DQD_FIXTURE, quality_input=fixture.quality_input
    )
    assert output["output_digest"] == (
        "sha256:2de84775fa6b0d743b617169f55889cf964bdfe287684a54a4ee0c73b0459723"
    )
    assert DQD_SUPPORTED_VERSIONS == {"2.9.0"}
    assert [row["status"] for row in output["checks"]] == [
        "pass",
        "fail",
        "unknown",
        "unknown",
        "unknown",
    ]
    assert [row["affected_rows"] for row in output["checks"]] == [0, 3, 0, 0, 2]
    assert [row["table"] for row in output["checks"]] == [
        "person",
        "visit_occurrence",
        "measurement",
        "observation",
        "note",
    ]
    key = b"synthetic-dqd-signing-key"
    report = normalize_omop_quality_output(
        output,
        quality_input=fixture.quality_input,
        reconciliation=fixture.reconcile(),
        signing_key=key,
        expected_execution_mode="local",
    )
    assert report.quality_verdict == "fail"
    assert report.requires_review
    assert verify_omop_quality_report(report, key)
    assert "SENTINEL" not in canonical_json(output) + report.to_json()
    assert {
        item.category: (item.failed_count, item.unknown_count)
        for item in report.categories
    } == {"conformance": (0, 1), "completeness": (1, 1), "plausibility": (0, 1)}


@pytest.mark.parametrize(
    "sentinel", ["SYNTHETIC_PRIVATE_TEXT", "نام ساختگی", "患者架空"]
)
def test_free_text_and_source_identifiers_never_affect_output(
    sentinel: str, tmp_path: Path
) -> None:
    payload = _payload()
    original = _normalize(payload, tmp_path)
    payload["Metadata"][0]["cdmSourceName"] = sentinel
    payload["Overview"] = {"notes": sentinel}
    payload["privatePath"] = sentinel
    for row in payload["CheckResults"]:
        for field in (
            "checkId",
            "checkName",
            "queryText",
            "notes",
            "notesValue",
            "checkDescription",
            "error",
            "notApplicableReason",
            "cdmFieldName",
            "conceptId",
            "unknownField",
        ):
            row[field] = sentinel
    output = _normalize(payload, tmp_path)
    assert output == original
    assert sentinel not in canonical_json(output)


@pytest.mark.parametrize(
    "version", [None, "", "1.0.0", "2.8.9", "2.9.1", "3.0.0", [], {}]
)
def test_unknown_or_missing_versions_are_typed_unsupported(
    version: object, tmp_path: Path
) -> None:
    payload = _payload()
    payload["Metadata"][0]["dqdVersion"] = version
    with pytest.raises(OmopQualityUnsupportedError, match="^dqd_version_unsupported$"):
        _normalize(payload, tmp_path)


@pytest.mark.parametrize("metadata", [None, {}, [], [{}, {}], [None]])
def test_unknown_metadata_shapes_are_unsupported(
    metadata: object, tmp_path: Path
) -> None:
    payload = _payload()
    payload["Metadata"] = metadata
    with pytest.raises(OmopQualityUnsupportedError):
        _normalize(payload, tmp_path)


@pytest.mark.parametrize("field", ["category", "cdmTableName"])
@pytest.mark.parametrize("value", ["SYNTHETIC_PRIVATE_TEXT", "", None, [], {}])
def test_unknown_categories_and_tables_never_leak(
    field: str, value: object, tmp_path: Path
) -> None:
    payload = _payload()
    payload["CheckResults"][0][field] = value
    with pytest.raises(OmopQualityUnsupportedError) as caught:
        _normalize(payload, tmp_path)
    assert "SYNTHETIC_PRIVATE_TEXT" not in str(caught.value)


@pytest.mark.parametrize("flags", list(itertools.product((0, 1), repeat=4)))
def test_only_unambiguous_executed_zero_violation_checks_pass(
    flags: tuple[int, ...], tmp_path: Path
) -> None:
    payload = _payload()
    row = payload["CheckResults"][0]
    row.update(zip(("failed", "passed", "isError", "notApplicable"), flags))
    payload["CheckResults"] = [row]
    check = _normalize(payload, tmp_path)["checks"][0]
    assert (check["status"] == "pass") is (flags == (0, 1, 0, 0))
    if flags[2] or flags[3]:
        assert check["status"] == "unknown"


@pytest.mark.parametrize("field", ["failed", "passed", "isError", "notApplicable"])
@pytest.mark.parametrize("value", [None, "0", -1, 2, 0.5, {}, []])
def test_unavailable_flags_abstain_and_invalid_flags_fail_closed(
    field: str, value: object, tmp_path: Path
) -> None:
    payload = _payload()
    payload["CheckResults"][0][field] = value
    if value is None:
        assert _normalize(payload, tmp_path)["checks"][0]["status"] == "unknown"
    else:
        with pytest.raises(OmopQualityProtocolError, match="dqd_flags_invalid"):
            _normalize(payload, tmp_path)


@pytest.mark.parametrize("count", [-1, 1.5, True, "2", [], {}, 2**53, 10**400])
def test_invalid_counts_are_typed_errors(count: object, tmp_path: Path) -> None:
    payload = _payload()
    payload["CheckResults"][0]["numViolatedRows"] = count
    with pytest.raises(OmopQualityProtocolError, match="dqd_count_invalid"):
        _normalize(payload, tmp_path)


@pytest.mark.parametrize("count", [None, 0, 0.0, 2, 2.0])
def test_missing_counts_abstain_and_nonzero_threshold_passes_preserve_counts(
    count: object, tmp_path: Path
) -> None:
    payload = _payload()
    payload["CheckResults"][0]["numViolatedRows"] = count
    check = _normalize(payload, tmp_path)["checks"][0]
    assert check["affected_rows"] == (0 if count is None else int(count))
    assert check["status"] == ("pass" if count == 0 else "unknown")


@pytest.mark.parametrize(
    "field", ["failed", "passed", "isError", "notApplicable", "numViolatedRows"]
)
def test_missing_required_fields_fail_closed(field: str, tmp_path: Path) -> None:
    payload = _payload()
    del payload["CheckResults"][0][field]
    with pytest.raises(OmopQualityProtocolError, match="dqd_check_fields_missing"):
        _normalize(payload, tmp_path)


@pytest.mark.parametrize("checks", [None, {}, [None], ["SYNTHETIC_PRIVATE_TEXT"]])
def test_invalid_check_shapes_fail_closed(checks: object, tmp_path: Path) -> None:
    payload = _payload()
    payload["CheckResults"] = checks
    with pytest.raises(OmopQualityProtocolError):
        _normalize(payload, tmp_path)


def test_empty_results_do_not_create_a_pass(tmp_path: Path) -> None:
    payload = _payload()
    payload["CheckResults"] = []
    output = _normalize(payload, tmp_path)
    fixture = load_frozen_omop_quality_fixture(FIXTURES / "quality_reconciliation.json")
    report = normalize_omop_quality_output(
        output,
        quality_input=fixture.quality_input,
        reconciliation=fixture.reconcile(),
        signing_key=b"synthetic-dqd-signing-key",
    )
    assert report.quality_verdict == "unknown"
    assert report.requires_review


@pytest.mark.parametrize(
    "raw",
    [
        b'{"SYNTHETIC_PRIVATE_TEXT":',
        b"\xff",
        b"[]",
        b"null",
        b'{"Metadata":[],"Metadata":[]}',
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b"[" * 2000 + b"]" * 2000,
    ],
)
def test_malformed_json_is_value_free(raw: bytes, tmp_path: Path) -> None:
    path = tmp_path / "SYNTHETIC_PRIVATE_PATH.json"
    path.write_bytes(raw)
    fixture = load_frozen_omop_quality_fixture(FIXTURES / "quality_reconciliation.json")
    with pytest.raises(OmopQualityProtocolError) as caught:
        normalize_dqd_results_file(path, quality_input=fixture.quality_input)
    assert "SYNTHETIC_PRIVATE" not in str(caught.value)


def test_missing_file_and_bounds_are_value_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = load_frozen_omop_quality_fixture(FIXTURES / "quality_reconciliation.json")
    with pytest.raises(OmopQualityProtocolError, match="^dqd_results_unreadable$"):
        normalize_dqd_results_file(
            tmp_path / "SYNTHETIC_PRIVATE_PATH.json",
            quality_input=fixture.quality_input,
        )
    monkeypatch.setattr(dqd, "_MAX_RESULTS_BYTES", 10)
    with pytest.raises(OmopQualityProtocolError, match="dqd_json_too_large"):
        _normalize(_payload(), tmp_path)
    monkeypatch.setattr(dqd, "_MAX_RESULTS_BYTES", 64 * 1024 * 1024)
    monkeypatch.setattr(dqd, "_MAX_OUTPUT_BYTES", 10)
    with pytest.raises(OmopQualityProtocolError, match="dqd_output_too_large"):
        _normalize(_payload(), tmp_path)
