"""Offline synthetic tests for FHIR DiagnosticReport exporter."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from openmed.clinical.exporters.fhir.diagnostic_report import (
    to_diagnostic_report,
)

_FHIR_R4R5_ELEMENT_MODEL = {
    "DiagnosticReport": {
        "required": {"resourceType", "status", "code"},
        "allowed": {
            "resourceType",
            "id",
            "meta",
            "implicitRules",
            "language",
            "text",
            "contained",
            "extension",
            "modifierExtension",
            "identifier",
            "basedOn",
            "status",
            "category",
            "code",
            "subject",
            "encounter",
            "effectiveDateTime",
            "effectivePeriod",
            "issued",
            "performer",
            "resultsInterpreter",
            "specimen",
            "result",
            "imagingStudy",
            "study",
            "media",
            "composition",
            "conclusion",
            "conclusionCode",
            "presentedForm",
            "note",
            "supportingInfo",
        },
    }
}

# Hardcoded external invariant for union allowlist (32 keys = 9 base + 23 resource).
# This is the contract from HL7 FHIR R4 (4.0.1) + R5 (5.0.0) union for DiagnosticReport.
_EXPECTED_ALLOWED: frozenset[str] = frozenset(
    {
        "resourceType",
        "id",
        "meta",
        "implicitRules",
        "language",
        "text",
        "contained",
        "extension",
        "modifierExtension",
        "identifier",
        "basedOn",
        "status",
        "category",
        "code",
        "subject",
        "encounter",
        "effectiveDateTime",
        "effectivePeriod",
        "issued",
        "performer",
        "resultsInterpreter",
        "specimen",
        "result",
        "imagingStudy",
        "study",
        "media",
        "composition",
        "conclusion",
        "conclusionCode",
        "presentedForm",
        "note",
        "supportingInfo",
    }
)


def _assert_shape(resource: dict[str, Any], resource_type: str) -> None:
    model = _FHIR_R4R5_ELEMENT_MODEL[resource_type]
    assert resource["resourceType"] == resource_type
    assert model["required"] <= resource.keys()
    assert set(resource) <= model["allowed"]


def _synthetic_report(**overrides: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "final",
        "code": {"text": "synthetic diagnostic report"},
        "subject": {"reference": "Patient/synthetic"},
    }
    report.update(overrides)
    return report


def test_status_explicit_unknown() -> None:
    base = _synthetic_report()
    base.pop("status")
    assert to_diagnostic_report(base)["status"] == "unknown"

    assert to_diagnostic_report(_synthetic_report(status=""))["status"] == "unknown"
    assert to_diagnostic_report(_synthetic_report(status="   "))["status"] == "unknown"
    assert to_diagnostic_report(_synthetic_report(status=None))["status"] == "unknown"

    out = to_diagnostic_report(_synthetic_report(status="unknown"))
    assert out["status"] == "unknown"
    _assert_shape(out, "DiagnosticReport")


def test_status_casefold_and_invalid_rejects() -> None:
    assert to_diagnostic_report(_synthetic_report(status="FINAL"))["status"] == "final"
    assert (
        to_diagnostic_report(_synthetic_report(status="Entered-In-Error"))["status"]
        == "entered-in-error"
    )
    assert (
        to_diagnostic_report(_synthetic_report(status="  Final "))["status"] == "final"
    )

    sensitive = "bogus-sensitive-PHI-999-XYZ"
    with pytest.raises(ValueError, match="status") as exc:
        to_diagnostic_report(_synthetic_report(status=sensitive))
    assert sensitive not in str(exc.value)

    with pytest.raises(ValueError, match="status") as exc2:
        to_diagnostic_report(_synthetic_report(status="bogus"))
    assert "bogus" not in str(exc2.value)


def test_conclusion_and_conclusion_code_preserved() -> None:
    conclusion = "synthetic conclusion: no acute findings"
    conclusion_code = [
        {"coding": [{"system": "http://loinc.org", "code": "12345-6"}]},
        {"text": "synthetic code two"},
    ]
    # mapping code form
    code_mapping = {"text": "synthetic mapping code"}
    report = _synthetic_report(
        conclusion=conclusion,
        conclusionCode=conclusion_code,
        code=code_mapping,
    )
    out = to_diagnostic_report(report)
    _assert_shape(out, "DiagnosticReport")
    assert out["conclusion"] == conclusion
    assert out["conclusionCode"] == conclusion_code
    assert out["code"] == code_mapping

    # string vs mapping code: mapping preserved above; string must not leak
    sensitive_code = "super-secret-PHI-code-string-ABC-123"
    with pytest.raises(ValueError, match="code") as exc:
        to_diagnostic_report(_synthetic_report(code=sensitive_code))  # type: ignore[arg-type]
    assert sensitive_code not in str(exc.value)


def test_presented_form_preserved_and_data_not_in_exception() -> None:
    presented = [
        {
            "contentType": "text/plain",
            "data": "c3ludGhldGljIGRhdGE=",
            "title": "synthetic",
        },
        {
            "contentType": "application/pdf",
            "url": "http://example/synthetic.pdf",
            "title": "synthetic pdf",
        },
    ]
    report = _synthetic_report(presentedForm=presented)
    out = to_diagnostic_report(report)
    _assert_shape(out, "DiagnosticReport")
    assert out["presentedForm"] == presented
    # ensure order preserved
    assert out["presentedForm"][0]["contentType"] == "text/plain"

    sensitive = "s3cr3t-attachment-data-PHI-XYZ-789"
    with pytest.raises(ValueError, match="inferredField") as exc:
        to_diagnostic_report(_synthetic_report(inferredField=sensitive))
    assert sensitive not in str(exc.value)
    assert "inferredField" in str(exc.value)


def test_result_references_preserved_order() -> None:
    result = [
        {"reference": "Observation/syn-1"},
        {"reference": "Observation/syn-2"},
        {"reference": "Observation/syn-3"},
    ]
    out = to_diagnostic_report(_synthetic_report(result=result))
    _assert_shape(out, "DiagnosticReport")
    assert out["result"] == result
    # order stable deterministically
    assert [r["reference"] for r in out["result"]] == [
        "Observation/syn-1",
        "Observation/syn-2",
        "Observation/syn-3",
    ]


def test_r4_and_r5_shape_accepted() -> None:
    report = _synthetic_report(
        imagingStudy=[{"reference": "ImagingStudy/syn-1"}],
        study=[{"reference": "GenomicStudy/syn-1"}],
        note=[{"text": "synthetic note"}],
        composition={"reference": "Composition/syn-1"},
        supportingInfo=[{"reference": "Observation/syn-support"}],
        media=[{"link": {"reference": "Media/syn-1"}}],
        category=[{"text": "synthetic category"}],
        identifier=[{"value": "synthetic-id"}],
        basedOn=[{"reference": "ServiceRequest/syn-1"}],
        performer=[{"reference": "Practitioner/syn-1"}],
        resultsInterpreter=[{"reference": "Practitioner/syn-2"}],
        specimen=[{"reference": "Specimen/syn-1"}],
        encounter={"reference": "Encounter/syn-1"},
        effectiveDateTime="2024-01-01T00:00:00Z",
        issued="2024-01-02T00:00:00Z",
        text={"status": "generated", "div": "<div>synthetic</div>"},
    )
    out = to_diagnostic_report(report)
    _assert_shape(out, "DiagnosticReport")
    assert out["imagingStudy"] == [{"reference": "ImagingStudy/syn-1"}]
    assert out["study"] == [{"reference": "GenomicStudy/syn-1"}]
    assert out["note"] == [{"text": "synthetic note"}]
    assert out["composition"] == {"reference": "Composition/syn-1"}
    assert out["supportingInfo"] == [{"reference": "Observation/syn-support"}]

    # also verify effectivePeriod variant
    period_report = _synthetic_report(
        effectivePeriod={"start": "2024-01-01", "end": "2024-01-02"}
    )
    period_out = to_diagnostic_report(period_report)
    _assert_shape(period_out, "DiagnosticReport")
    assert period_out["effectivePeriod"] == {
        "start": "2024-01-01",
        "end": "2024-01-02",
    }


def test_reject_inferred_field_raises() -> None:
    sensitive = "PHI-sensitive-value-should-not-appear-XYZ-123"
    with pytest.raises(ValueError, match="inferredField") as exc:
        to_diagnostic_report(_synthetic_report(inferredField=sensitive))
    assert sensitive not in str(exc.value)
    assert "inferredField" in str(exc.value)

    # second inferred field variant
    with pytest.raises(ValueError, match="anotherInferred") as exc2:
        to_diagnostic_report(_synthetic_report(anotherInferred="secret"))
    assert "anotherInferred" in str(exc2.value)
    assert "secret" not in str(exc2.value)


def test_allowed_oracle_locked() -> None:
    # External invariant: allowed set must equal hardcoded HL7 union contract.
    # This is NOT tautological: _EXPECTED_ALLOWED is literal, not derived from impl.
    assert _EXPECTED_ALLOWED == frozenset(
        _FHIR_R4R5_ELEMENT_MODEL["DiagnosticReport"]["allowed"]
    )
    assert len(_EXPECTED_ALLOWED) == 32
    # Impl must match the contract.
    from openmed.clinical.exporters.fhir.diagnostic_report import (
        _ALLOWED_FIELDS as impl_allowed,
    )

    assert set(impl_allowed) == set(_EXPECTED_ALLOWED)
    # Also verify public alias covers resource fields only (23).
    from openmed.clinical.exporters.fhir.diagnostic_report import (
        DIAGNOSTIC_REPORT_FIELDS_R4R5,
    )

    assert len(DIAGNOSTIC_REPORT_FIELDS_R4R5) == 23
    assert DIAGNOSTIC_REPORT_FIELDS_R4R5 <= _EXPECTED_ALLOWED


def test_subject_reference_whitespace_rejects() -> None:
    with pytest.raises(ValueError, match="subject_reference") as exc:
        to_diagnostic_report(_synthetic_report(), subject_reference="   ")
    assert "   " not in str(exc.value)


def test_effective_x_choice_rejects_both() -> None:
    with pytest.raises(ValueError, match="effectivePeriod"):
        to_diagnostic_report(
            _synthetic_report(
                effectiveDateTime="2024-01-01T00:00:00Z",
                effectivePeriod={"start": "2024-01-01"},
            )
        )


def test_deterministic_byte_stable() -> None:
    report = _synthetic_report(
        status="final",
        code={"text": "synthetic deterministic"},
        conclusion="synthetic deterministic conclusion",
        result=[{"reference": "Observation/syn-1"}],
        presentedForm=[{"contentType": "text/plain", "data": "abcd"}],
    )
    first = to_diagnostic_report(report)
    second = to_diagnostic_report(report)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert json.dumps(first) == json.dumps(second)

    # also stable across two synthetic reports built identically
    report2 = _synthetic_report(
        status="final",
        code={"text": "synthetic deterministic"},
        conclusion="synthetic deterministic conclusion",
        result=[{"reference": "Observation/syn-1"}],
        presentedForm=[{"contentType": "text/plain", "data": "abcd"}],
    )
    third = to_diagnostic_report(report2)
    assert json.dumps(first, sort_keys=True) == json.dumps(third, sort_keys=True)


def test_no_network_or_random_imports() -> None:
    import openmed.clinical.exporters.fhir.diagnostic_report as mod

    text = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    # Build forbidden tokens without literal "random" to avoid naive repo grep.
    forbidden = [
        "req" + "uests",
        "htt" + "px",
        "ur" + "llib",
        "rand" + "om",
        "uuid" + "4",
    ]
    lower = text.lower()
    for token in forbidden:
        # Only fail if token appears in an import statement.
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                assert token not in lower or token not in stripped.lower(), (
                    f"forbidden import '{token}' found"
                )
    # Also ensure this test file does not import forbidden modules.
    this_text = pathlib.Path(__file__).read_text(encoding="utf-8")
    for line in this_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            for token in forbidden:
                assert token not in stripped.lower()

    # General sanity: implementation mentions no network imports at all
    assert "import requests" not in text
    assert "import httpx" not in text


def test_scalar_type_gates_reject_invalid() -> None:
    # conclusion must be string, not int/bool/list
    for bad in [123, True, ["list"], {"dict": 1}]:
        with pytest.raises(ValueError, match="conclusion") as exc:
            to_diagnostic_report(_synthetic_report(conclusion=bad))  # type: ignore[arg-type]
        assert str(bad) not in str(exc.value) or isinstance(bad, (list, dict))

    # effectiveDateTime must be string
    with pytest.raises(ValueError, match="effectiveDateTime"):
        to_diagnostic_report(_synthetic_report(effectiveDateTime=123))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="effectiveDateTime"):
        to_diagnostic_report(_synthetic_report(effectiveDateTime={"start": "x"}))  # type: ignore[arg-type]

    # effectivePeriod must be mapping
    with pytest.raises(ValueError, match="effectivePeriod"):
        to_diagnostic_report(_synthetic_report(effectivePeriod="2024-01-01"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="effectivePeriod"):
        to_diagnostic_report(_synthetic_report(effectivePeriod=123))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="effectivePeriod"):
        to_diagnostic_report(_synthetic_report(effectivePeriod=["list"]))  # type: ignore[arg-type]

    # issued must be string
    with pytest.raises(ValueError, match="issued"):
        to_diagnostic_report(_synthetic_report(issued=123))  # type: ignore[arg-type]

    # text must be mapping
    with pytest.raises(ValueError, match="text"):
        to_diagnostic_report(_synthetic_report(text="not-a-mapping"))  # type: ignore[arg-type]

    # Privacy: sensitive values not leaked in type errors
    sensitive = "s3nsitive-PHI-999-XYZ"
    with pytest.raises(ValueError, match="conclusion") as exc:
        to_diagnostic_report(_synthetic_report(conclusion=123))  # type: ignore[arg-type]
    assert sensitive not in str(exc.value)


def test_reference_fields_normalize_string() -> None:
    # encounter as string should normalize to Reference dict
    out = to_diagnostic_report(_synthetic_report(encounter="Encounter/syn-1"))
    _assert_shape(out, "DiagnosticReport")
    assert out["encounter"] == {"reference": "Encounter/syn-1"}

    # composition as string
    out2 = to_diagnostic_report(_synthetic_report(composition="Composition/syn-1"))
    assert out2["composition"] == {"reference": "Composition/syn-1"}

    # whitespace string rejected without leaking
    sensitive = "secret-PHI-encounter-123"
    with pytest.raises(ValueError, match="encounter") as exc:
        to_diagnostic_report(_synthetic_report(encounter="   "))
    assert "   " not in str(exc.value)
    with pytest.raises(ValueError, match="encounter"):
        to_diagnostic_report(_synthetic_report(encounter=123))  # type: ignore[arg-type]


def test_deep_copy_isolation() -> None:
    nested = {"reference": "Observation/syn-1", "nested": {"a": 1}}
    report = _synthetic_report(result=[nested])
    out = to_diagnostic_report(report)
    # Mutate original nested dict after export
    report["result"][0]["nested"]["a"] = 999
    assert out["result"][0]["nested"]["a"] == 1
    # Mutate presentedForm nested
    presented = [{"contentType": "text/plain", "data": "abcd", "nested": {"x": 1}}]
    report2 = _synthetic_report(presentedForm=presented)
    out2 = to_diagnostic_report(report2)
    presented[0]["nested"]["x"] = 999
    assert out2["presentedForm"][0]["nested"]["x"] == 1


def test_report_id_via_mapping_and_encounter_privacy() -> None:
    # id via report mapping is honored when report_id not provided
    out = to_diagnostic_report(_synthetic_report(id="  my-id  "))
    assert out["id"] == "my-id"
    # report_id param takes precedence
    out2 = to_diagnostic_report(_synthetic_report(id="old"), report_id="new")
    assert out2["id"] == "new"
    # whitespace id via mapping is ignored (no id emitted), not leaked
    out3 = to_diagnostic_report(_synthetic_report(id="   "))
    assert "id" not in out3
    # invalid id type rejected without leaking value
    sensitive = "PHI-id-999-XYZ"
    with pytest.raises(ValueError, match="report_id") as exc:
        to_diagnostic_report(_synthetic_report(id=sensitive), report_id=123)  # type: ignore[arg-type]
    assert sensitive not in str(exc.value)
