"""Synthetic offline tests for typed laboratory reference-range provenance."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import (
    ReferenceRangeProvenance,
    ReferenceRangeStatus,
    build_reference_range,
    compare_reference_ranges,
    fingerprint_source,
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
    first = fingerprint_source({"instrument": "instrument-a", "version": 1})
    second = fingerprint_source({"version": 1, "instrument": "instrument-a"})

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
