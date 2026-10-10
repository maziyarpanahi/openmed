"""Synthetic offline tests for typed laboratory reference-range provenance."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import (
    ReferenceRangeProvenance,
    ReferenceRangeStatus,
    build_reference_range,
    compare_reference_ranges,
    fingerprint_lab_reference_source,
    resolve_reference_range,
)


def _range(*, source: str = "instrument-a", locale: str | None = "en-US"):
    return build_reference_range(
        "sodium",
        135,
        145,
        unit="mmol/L",
        population="adult",
        precision=0,
        source={"instrument": source, "version": 1},
        locale=locale,
    )


def test_source_fingerprint_is_stable_and_order_independent() -> None:
    first = fingerprint_lab_reference_source(
        {"instrument": "instrument-a", "version": 1}
    )
    second = fingerprint_lab_reference_source(
        {"version": 1, "instrument": "instrument-a"}
    )

    assert first == second
    assert first.startswith("sha256:")
    assert len(first) == len("sha256:") + 64


def test_typed_range_keeps_required_provenance_without_raw_source() -> None:
    reference_range = _range()
    payload = reference_range.to_dict()

    assert payload["unit"] == "mmol/L"
    assert payload["population"] == "adult"
    assert payload["precision"] == 0
    assert payload["locale"] == "en-us"
    assert payload["provenance"]["source_fingerprint"].startswith("sha256:")
    assert "instrument-a" not in json.dumps(payload)


def test_exact_provenance_resolution_is_known() -> None:
    reference_range = _range()

    resolved = resolve_reference_range(
        [reference_range],
        analyte="SODIUM",
        provenance=reference_range.provenance,
    )

    assert resolved.status is ReferenceRangeStatus.KNOWN
    assert resolved.is_known
    assert resolved.reference_range == reference_range


def test_missing_provenance_does_not_select_across_locales() -> None:
    reference_range = _range(locale="en-US")
    target = ReferenceRangeProvenance(
        unit="mmol/L",
        population="adult",
        precision=0,
        source_fingerprint=reference_range.provenance.source_fingerprint,
        locale="fr-FR",
    )

    resolved = resolve_reference_range(
        [reference_range],
        analyte="sodium",
        provenance=target,
    )

    assert resolved.status is ReferenceRangeStatus.UNKNOWN
    assert resolved.reference_range is None


def test_competing_same_context_ranges_are_conflicting() -> None:
    first = _range(source="instrument-a")
    second = build_reference_range(
        "sodium",
        136,
        146,
        unit="mmol/L",
        population="adult",
        precision=0,
        source={"instrument": "instrument-a", "version": 1},
        locale="en-US",
    )

    resolved = resolve_reference_range(
        [first, second],
        analyte="sodium",
        provenance=first.provenance,
    )

    assert resolved.status is ReferenceRangeStatus.CONFLICT
    assert resolved.is_conflicting
    assert resolved.reference_range is None


def test_different_instruments_are_not_silently_compared() -> None:
    first = _range(source="instrument-a")
    second = _range(source="instrument-b")

    resolved = compare_reference_ranges(first, second)

    assert resolved.status is ReferenceRangeStatus.CONFLICT
    assert resolved.reference_range is None


def test_different_units_and_invalid_bounds_are_unknown_or_rejected() -> None:
    first = _range()
    different_unit = build_reference_range(
        "sodium",
        135,
        145,
        unit="mEq/L",
        population="adult",
        precision=0,
        source={"instrument": "instrument-a", "version": 1},
        locale="en-US",
    )

    resolved = compare_reference_ranges(first, different_unit)
    assert resolved.status is ReferenceRangeStatus.UNKNOWN

    with pytest.raises(ValueError, match="low bound"):
        build_reference_range(
            "sodium",
            146,
            145,
            unit="mmol/L",
            population="adult",
            precision=0,
            source="instrument-a",
        )


def test_incomplete_provenance_is_rejected_without_echoing_values() -> None:
    with pytest.raises(ValueError, match="source provenance") as exc_info:
        build_reference_range(
            "sodium",
            135,
            145,
            unit="mmol/L",
            population="adult",
            precision=0,
        )

    assert "instrument" not in str(exc_info.value)


@pytest.mark.parametrize(
    "target", [{"unit": "mg/dL"}, {"locale": "fr-FR"}, {"precision": 2}]
)
def test_partial_target_provenance_never_falls_back_to_single_range(target):
    result = resolve_reference_range([_range()], **target)
    assert result.status is ReferenceRangeStatus.UNKNOWN
    assert result.reference_range is None


def test_optional_locale_and_bounds_have_total_deterministic_order():
    from dataclasses import replace

    first = _range(locale=None)
    second = _range(locale="en-US")
    assert (
        compare_reference_ranges(first, second).status is ReferenceRangeStatus.UNKNOWN
    )
    open_low = replace(first, low=None)
    assert (
        resolve_reference_range([first, open_low]).status
        is ReferenceRangeStatus.CONFLICT
    )


def test_mapping_source_can_be_fingerprinted_without_existing_digest():
    from openmed.clinical.lab_reference_ranges import reference_range_from_mapping

    result = reference_range_from_mapping(
        {
            "analyte": "sodium",
            "low": 135,
            "high": 145,
            "unit": "mmol/L",
            "population": "adult",
            "precision": 0,
            "source": "synthetic-device",
        }
    )
    assert result.source_fingerprint == fingerprint_lab_reference_source(
        "synthetic-device"
    )


@pytest.mark.parametrize(
    "update",
    [
        {"low_inclusive": "false"},
        {"schema_version": True},
        {"unit": "mg/dL"},
        {"analyte": 123},
    ],
)
def test_mapping_rejects_invalid_or_conflicting_contract_fields(update):
    from openmed.clinical.lab_reference_ranges import reference_range_from_mapping

    payload = _range().to_dict()
    payload.update(update)
    with pytest.raises(ValueError):
        reference_range_from_mapping(payload)


def test_numeric_errors_do_not_retain_private_cause_or_context():
    with pytest.raises(ValueError) as caught:
        build_reference_range(
            "sodium",
            "SYNTHETIC-PRIVATE-NUMBER",
            145,
            unit="mmol/L",
            population="adult",
            precision=0,
            source="fixture",
        )
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "SYNTHETIC" not in str(caught.value)


def test_candidate_and_source_nesting_limits(monkeypatch):
    from openmed.clinical import lab_reference_ranges as module

    monkeypatch.setattr(module, "_MAX_ITEMS", 2)
    with pytest.raises(ValueError):
        resolve_reference_range((_range() for _ in range(3)))
    cyclic = {}
    cyclic["nested"] = cyclic
    with pytest.raises(ValueError) as caught:
        fingerprint_lab_reference_source(cyclic)
    assert caught.value.__context__ is None


def test_direct_resolution_cannot_claim_known_without_a_range():
    from openmed.clinical import ReferenceRangeResolution

    with pytest.raises(ValueError):
        ReferenceRangeResolution(
            status=ReferenceRangeStatus.KNOWN,
            reference_range=None,
            reason="single explicit range",
            candidate_count=1,
        )


def test_lab_fingerprint_export_preserves_evidence_coverage_contract():
    from openmed.clinical import fingerprint_source
    from openmed.clinical.evidence_coverage import (
        fingerprint_source as evidence_fingerprint,
    )
    from openmed.clinical.lab_reference_ranges import (
        fingerprint_source as range_fingerprint,
    )

    assert fingerprint_source is evidence_fingerprint
    assert fingerprint_lab_reference_source is range_fingerprint
