"""Synthetic privacy, review and local FHIR checks for passive SDOH export."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from openmed.clinical.exporters.fhir import (
    SDOH_FHIR_IG_VERSION,
    SDOHFHIRCode,
    SDOHFHIRExportError,
    SDOHFHIRRecord,
    SDOHObservationBinding,
    check_bundle,
    to_bundle,
    to_sdoh_observations,
    validate_resource,
)
from openmed.clinical.sdoh_evidence import SDOHEvidence
from openmed.clinical.sdoh_experiencer import SDOHExperiencerEvidence
from openmed.clinical.sdoh_sensitive_use import (
    ProhibitedAutomatedUse,
    SDOHPurpose,
    SDOHSensitiveUseLabel,
)
from openmed.clinical.sdoh_temporal import SDOHTemporalEvidence

_REFS = tuple(f"urn:uuid:00000000-0000-4000-8000-{n:012d}" for n in (1, 2, 3))
_NOW = datetime(2026, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc)
_SYSTEM = "https://synthetic.example/CodeSystem/sdoh"
_PRIVATE = "SYNTHETIC PRIVATE PERSON AND DIAGNOSTIC"
_BINDING = SDOHObservationBinding(
    SDOHFHIRCode(_SYSTEM, "assessment"), SDOHFHIRCode(_SYSTEM, "positive")
)


def _record(
    determinant: str = "food_insecurity",
    *,
    assertion: str = "present",
    review: str = "reviewed",
    subject: str = "patient",
    temporal: str = "current",
    temporal_review: bool = False,
    start: str | None = None,
    end: str | None = None,
    label: SDOHSensitiveUseLabel | None = None,
) -> SDOHFHIRRecord:
    span = (10, 20)
    return SDOHFHIRRecord(
        SDOHEvidence(
            "self_report", assertion, "social_history", span, review, determinant
        ),
        SDOHExperiencerEvidence(
            span, subject, source="provided", review_required=subject == "unknown"
        ),
        SDOHTemporalEvidence(span, temporal, review_required=temporal_review),
        start,
        end,
        label or SDOHSensitiveUseLabel("sdoh"),
    )


def _export(records: Any, **overrides: Any):
    config = {
        "terminology": {
            index: replace(_BINDING, domain_category="housing-instability")
            if type(record) is SDOHFHIRRecord
            and record.evidence.determinant == "housing"
            else _BINDING
            for index, record in enumerate(records)
        },
        "subject_reference": _REFS[0],
        "source_reference": _REFS[1],
        "software_reference": _REFS[2],
        "purpose": SDOHPurpose.CLINICAL_REVIEW,
        "clock": lambda: _NOW,
    }
    config.update(overrides)
    return to_sdoh_observations(records, **config)


@pytest.mark.parametrize(
    "determinant,domain",
    [
        ("food_insecurity", "food-insecurity"),
        ("housing", "housing-instability"),
        ("employment", "employment-status"),
    ],
)
def test_reviewed_current_patient_exports_valid_declared_categories(
    determinant, domain
):
    result = _export([_record(determinant, start="2026-01-01")])
    observation = result.observations[0]
    assert validate_resource(observation).is_valid
    assert observation["status"] == "final"
    assert [item["coding"][0]["code"] for item in observation["category"]] == [
        "social-history",
        "sdoh",
        domain,
    ]
    assert (
        observation["category"][2]["coding"][0]["version"]
        == SDOH_FHIR_IG_VERSION
        == "2.3.0"
    )
    assert observation["category"][1]["coding"][0]["version"] == "7.0.0"
    assert observation["code"] == {
        "coding": [{"system": _SYSTEM, "code": "assessment"}]
    }
    assert observation["valueCodeableConcept"]["coding"][0]["code"] == "positive"
    assert observation["effectiveDateTime"] == "2026-01-01"
    assert "profile" not in observation["meta"]
    assert not result.losses


@pytest.mark.parametrize("review", ["unreviewed", "needs_review"])
@pytest.mark.parametrize("temporal", ["current", "historical", "future", "unknown"])
def test_pending_review_and_temporal_states_never_become_final(review, temporal):
    result = _export([_record(review=review, temporal=temporal)])
    assert result.observations[0]["status"] == "preliminary"
    assert not any(key.startswith("effective") for key in result.observations[0])


@pytest.mark.parametrize("temporal", ["historical", "future", "unknown"])
def test_reviewed_non_current_evidence_is_still_preliminary(temporal):
    result = _export([_record(temporal=temporal)])
    assert result.observations[0]["status"] == "preliminary"


def test_unresolved_temporal_review_is_preliminary_even_with_reviewed_assertion():
    result = _export([_record(temporal_review=True)])
    assert result.observations[0]["status"] == "preliminary"


@pytest.mark.parametrize(
    "record,code",
    [
        (_record(assertion="absent"), "negated_need"),
        (_record(assertion="unknown"), "assertion_unconfirmed"),
        (_record(assertion="uncertain"), "assertion_unconfirmed"),
        (_record(subject="family"), "non_patient_or_unresolved"),
        (_record(subject="household"), "non_patient_or_unresolved"),
        (_record(subject="unknown"), "non_patient_or_unresolved"),
        (_record(review="rejected"), "review_refused"),
        (_record(review="refused"), "review_refused"),
        (_record(review="unknown"), "review_refused"),
        (_record("tobacco"), "domain_unmapped"),
        (_record("other"), "domain_unmapped"),
    ],
)
def test_exclusions_are_explicit_and_skip_recorded_clock(record, code):
    def no_clock():
        pytest.fail("no export means no recorded-clock call")

    result = _export([record], clock=no_clock)
    assert result.observations == result.provenance == ()
    assert result.losses == ({"input_index": 0, "code": code, "excluded": True},)
    assert result.to_dict()["excluded_count"] == 1


def test_missing_terminology_and_missing_answer_have_distinct_loss_semantics():
    missing = _export([_record()], terminology={})
    assert missing.losses[0]["code"] == "terminology_unmapped"
    partial = _export(
        [_record()],
        terminology={0: SDOHObservationBinding(_BINDING.code)},
    )
    assert partial.observations[0]["status"] == "preliminary"
    assert partial.observations[0]["dataAbsentReason"]["coding"][0]["code"] == "unknown"
    assert "valueCodeableConcept" not in partial.observations[0]
    assert partial.losses == (
        {"input_index": 0, "code": "answer_unmapped", "excluded": False},
    )
    assert partial.to_dict()["excluded_count"] == 0


def test_sensitive_labels_propagate_to_observation_and_provenance():
    result = _export([_record()])
    codes = {entry["code"] for entry in result.observations[0]["meta"]["security"]}
    assert {"sensitive-sdoh", "human-review-required"} <= codes
    assert {"prohibited-" + item.value for item in ProhibitedAutomatedUse} <= codes
    assert {"allowed-" + item.value for item in SDOHPurpose} <= codes
    assert (
        result.provenance[0]["meta"]["security"]
        == result.observations[0]["meta"]["security"]
    )
    result.observations[0]["meta"]["security"].clear()
    assert result.provenance[0]["meta"]["security"]


@pytest.mark.parametrize(
    "label",
    [
        SDOHSensitiveUseLabel(
            "sdoh", allowed_purposes=(SDOHPurpose.CARE_COORDINATION,)
        ),
        SDOHSensitiveUseLabel("sdoh", human_review_required=False),
        SDOHSensitiveUseLabel(
            "sdoh", prohibited_automated_uses=(ProhibitedAutomatedUse.CARE_DENIAL,)
        ),
    ],
)
def test_incomplete_or_disallowed_use_is_refused(label):
    result = _export([_record(label=label)])
    assert not result.observations
    assert result.losses[0]["code"] == "sensitive_use_refused"


def test_period_is_explicit_and_preserves_historical_preliminary_status():
    result = _export(
        [_record(temporal="historical", start="2025-01-01", end="2025-12-31")]
    )
    obs = result.observations[0]
    assert obs["status"] == "preliminary"
    assert obs["effectivePeriod"] == {"start": "2025-01-01", "end": "2025-12-31"}
    assert "effectiveDateTime" not in obs


def test_aware_effective_time_normalizes_without_wall_clock():
    result = _export([_record(start="2026-01-01T05:30:00+05:30")])
    assert result.observations[0]["effectiveDateTime"] == "2026-01-01T00:00:00Z"


@pytest.mark.parametrize(
    "start,end,temporal",
    [
        (_PRIVATE, None, "current"),
        ("2026-02-30", None, "current"),
        ("2026-01-01T00:00:00", None, "current"),
        ("20260101", None, "current"),
        ("２０２６-01-01", None, "current"),
        ("2026-01-01", None, "unknown"),
        (None, "2026-01-02", "current"),
        ("2026-01-02", "2026-01-01", "current"),
        ("2026-01-01", "2026-01-02T00:00:00Z", "current"),
    ],
)
def test_invalid_or_unanchored_calendar_data_is_controlled(start, end, temporal):
    with pytest.raises(SDOHFHIRExportError) as error:
        _record(start=start, end=end, temporal=temporal)
    assert error.value.code == "invalid_effective_time"
    assert error.value.__context__ is None and _PRIVATE not in str(error.value)


def test_offsets_must_align_and_fit_fhir_unsigned_integer():
    record = _record()
    with pytest.raises(SDOHFHIRExportError, match="offset_mismatch"):
        replace(record, temporal=SDOHTemporalEvidence((11, 20), "current"))
    with pytest.raises(SDOHFHIRExportError, match="offset_mismatch"):
        SDOHFHIRRecord(
            SDOHEvidence(
                "self_report", "present", "social_history", (2**31, 2**31 + 1)
            ),
            SDOHExperiencerEvidence((2**31, 2**31 + 1), "patient"),
            SDOHTemporalEvidence((2**31, 2**31 + 1), "current"),
        )


@pytest.mark.parametrize(
    "system,code",
    [
        ("https://user:secret@synthetic.example/codes", "safe"),
        ("https://synthetic.example/codes?credential=private", "safe"),
        ("file:///private/source", "safe"),
        (_SYSTEM, _PRIVATE),
        (_SYSTEM, "x\u200b"),
        (_SYSTEM, "x" * 65),
        (_SYSTEM, "x/subject"),
    ],
)
def test_invalid_bindings_echo_no_protected_values(system, code):
    with pytest.raises(SDOHFHIRExportError) as error:
        SDOHFHIRCode(system, code)
    assert error.value.code == "invalid_code"
    assert _PRIVATE not in repr(error.value)


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"subject_reference": "Patient/" + _PRIVATE}, "invalid_reference"),
        (
            {"source_reference": "https://private.example/DocumentReference/0042"},
            "invalid_reference",
        ),
        ({"software_reference": _REFS[2].upper()}, "invalid_reference"),
        ({"purpose": "autonomous_diagnosis"}, "invalid_purpose"),
        ({"terminology": {"private_marker": _BINDING}}, "invalid_terminology"),
        (
            {"terminology": {0: {"text": _PRIVATE}}},
            "invalid_terminology",
        ),
        ({"clock": None}, "invalid_clock"),
    ],
)
def test_bad_configuration_is_controlled_before_projection(overrides, code):
    with pytest.raises(SDOHFHIRExportError) as error:
        _export([_record()], **overrides)
    assert error.value.code == code and _PRIVATE not in repr(error.value)


@pytest.mark.parametrize("records", ["private", [{}], [_record()] * 513])
def test_records_are_typed_and_bounded(records):
    with pytest.raises(SDOHFHIRExportError, match="invalid_input"):
        _export(records)


def test_clock_called_once_for_batch_and_failures_are_not_chained():
    calls = []
    result = _export(
        [_record(), _record("employment")], clock=lambda: calls.append(1) or _NOW
    )
    assert calls == [1]
    assert {p["recorded"] for p in result.provenance} == {"2026-01-02T03:04:05.123456Z"}

    def failed_clock():
        raise RuntimeError(_PRIVATE)

    with pytest.raises(SDOHFHIRExportError) as error:
        _export([_record()], clock=failed_clock)
    assert error.value.code == "clock_failed" and error.value.__context__ is None
    with pytest.raises(SDOHFHIRExportError, match="invalid_clock"):
        _export([_record()], clock=lambda: datetime(2026, 1, 2))


def test_projection_is_deterministic_and_duplicate_loss_is_explicit():
    original = _export([_record()])
    assert original.to_dict() == _export([_record()]).to_dict()
    duplicate = _export([_record(), _record()])
    assert len(duplicate.observations) == len(duplicate.provenance) == 1
    assert duplicate.losses == (
        {"input_index": 1, "code": "duplicate_input", "excluded": True},
    )
    later = _export([_record()], clock=lambda: _NOW + timedelta(seconds=1))
    assert later.observations[0]["id"] == original.observations[0]["id"]
    assert later.provenance[0]["id"] != original.provenance[0]["id"]
    detached = original.to_dict()
    detached["observations"][0]["status"] = "changed"
    assert original.observations[0]["status"] == "final"


def test_provenance_links_resource_and_source_offsets_without_surfaces():
    result = _export([_record()])
    prov = result.provenance[0]
    assert prov["target"] == [
        {"reference": "Observation/" + result.observations[0]["id"]}
    ]
    assert prov["entity"][0]["what"]["reference"] == _REFS[1]
    offsets = prov["entity"][0]["what"]["extension"][0]["extension"]
    assert offsets[:2] == [
        {"url": "start", "valueUnsignedInt": 10},
        {"url": "end", "valueUnsignedInt": 20},
    ]
    serialized = json.dumps(result.to_dict())
    for forbidden in (
        "display",
        "text",
        "diagnostics",
        "name",
        "identifier",
        "valueString",
    ):
        assert '"' + forbidden + '"' not in serialized
    assert _PRIVATE not in serialized


@pytest.mark.integration
def test_local_profile_checks_accept_shape_and_reject_missing_categories(
    tmp_path: Path,
):
    result = _export([_record()])
    observation = copy.deepcopy(result.observations[0])
    profile = "https://synthetic.example/StructureDefinition/sdoh-shape"
    observation["meta"]["profile"] = [profile]
    package = tmp_path / "package"
    package.mkdir()
    (package / "StructureDefinition-sdoh.json").write_text(
        json.dumps(
            {
                "resourceType": "StructureDefinition",
                "url": profile,
                "type": "Observation",
                "snapshot": {
                    "element": [
                        {"id": "Observation", "path": "Observation"},
                        {
                            "id": "Observation.category",
                            "path": "Observation.category",
                            "min": 3,
                            "max": "*",
                        },
                        {
                            "id": "Observation.status",
                            "path": "Observation.status",
                            "min": 1,
                            "max": "1",
                            "fixedCode": "final",
                        },
                    ]
                },
            }
        )
    )
    bundle = to_bundle(
        [observation, *result.provenance],
        doc_id="synthetic-sdoh",
        bundle_type="collection",
    )
    outcome = check_bundle(bundle, tmp_path)
    assert not any(i["severity"] in {"error", "fatal"} for i in outcome["issue"])
    broken = copy.deepcopy(bundle)
    next(
        e["resource"]
        for e in broken["entry"]
        if e["resource"]["resourceType"] == "Observation"
    )["category"].pop()
    assert any(
        i["severity"] == "error" for i in check_bundle(broken, tmp_path)["issue"]
    )


def test_projection_stays_offline_without_model_loading(monkeypatch):
    from openmed.core.models import ModelLoader
    from openmed.core.offline import HF_OFFLINE_ENV_VARS, network_blocked_if_offline

    def no_model(*args, **kwargs):
        pytest.fail("passive projection must not load a model")

    monkeypatch.setattr(ModelLoader, "load_model", no_model)
    for key in HF_OFFLINE_ENV_VARS:
        monkeypatch.setenv(key, "1")
    with network_blocked_if_offline(local_only=True):
        result = _export([_record()])
    assert len(result.observations) == 1


def test_per_record_answer_bindings_do_not_guess_from_shared_determinant():
    first = _record("employment")
    second = replace(
        first,
        evidence=replace(first.evidence, source_span=(30, 40)),
        experiencer=replace(first.experiencer, source_offsets=(30, 40)),
        temporal=replace(first.temporal, source_offsets=(30, 40)),
    )
    opposite = SDOHObservationBinding(
        _BINDING.code, SDOHFHIRCode(_SYSTEM, "alternative-reviewed-answer")
    )
    result = _export([first, second], terminology={0: _BINDING, 1: opposite})
    assert [
        r["valueCodeableConcept"]["coding"][0]["code"] for r in result.observations
    ] == ["positive", "alternative-reviewed-answer"]
    assert len({r["id"] for r in result.observations}) == 2


@pytest.mark.parametrize("key", [True, -1, 1, 512, "private-key"])
def test_terminology_indices_are_exact_bounded_integers(key):
    with pytest.raises(SDOHFHIRExportError, match="invalid_terminology"):
        _export([_record()], terminology={key: _BINDING})


def test_generic_housing_requires_explicit_domain_without_guessing():
    missing = _export([_record("housing")], terminology={0: _BINDING})
    assert not missing.observations
    assert missing.losses[0]["code"] == "domain_ambiguous"
    for domain in ("housing-instability", "homelessness", "inadequate-housing"):
        result = _export(
            [_record("housing")],
            terminology={0: replace(_BINDING, domain_category=domain)},
        )
        assert result.observations[0]["category"][2]["coding"][0]["code"] == domain
    mismatched = _export(
        [_record()],
        terminology={0: replace(_BINDING, domain_category="employment-status")},
    )
    assert mismatched.losses[0]["code"] == "domain_binding_mismatch"
    with pytest.raises(SDOHFHIRExportError, match="invalid_terminology"):
        replace(_BINDING, domain_category=_PRIVATE)


def test_reference_roles_cannot_share_one_resource_identity():
    with pytest.raises(SDOHFHIRExportError, match="invalid_reference"):
        _export([_record()], source_reference=_REFS[0])


def test_documented_sdoh_example_runs_offline_without_models(monkeypatch, capsys):
    from openmed.core.models import ModelLoader
    from openmed.core.offline import HF_OFFLINE_ENV_VARS, network_blocked_if_offline

    def no_model(*args, **kwargs):
        pytest.fail("documentation must not load a model")

    monkeypatch.setattr(ModelLoader, "load_model", no_model)
    for key in HF_OFFLINE_ENV_VARS:
        monkeypatch.setenv(key, "1")
    guide = Path(__file__).resolve().parents[4] / "docs/fhir-interop.md"
    blocks = re.findall(r"(?ms)^```python\n(.*?)^```", guide.read_text())
    runnable = [b for b in blocks if b.startswith("# Runnable: synthetic passive SDOH")]
    assert len(runnable) == 1
    with network_blocked_if_offline(local_only=True):
        exec(compile(runnable[0], "sdoh_projection_example.py", "exec"), {})
    captured = capsys.readouterr()
    assert captured.out == "{'exported_count': 1, 'status': 'preliminary'}\n"
    assert captured.err == ""
