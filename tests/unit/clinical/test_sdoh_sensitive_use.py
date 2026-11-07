"""Tests for fail-closed SDOH sensitive-use labels."""

from __future__ import annotations

import json
import socket

import pytest

from openmed.clinical.sdoh_sensitive_use import (
    LabeledSDOHExport,
    ProhibitedAutomatedUse,
    SDOHSensitiveUseLabel,
    label_sdoh_output,
    serialize_labeled_sdoh_output,
)

SYNTHETIC_MARKER = "SYNTHETIC-SENSITIVE-OUTPUT"


def test_default_labels_cover_every_field_and_prohibit_automated_uses() -> None:
    export = label_sdoh_output({"category": "synthetic_housing", "status": "unknown"})
    payload = export.to_dict()

    assert [item["field_name"] for item in payload["sensitive_use_labels"]] == [
        "category",
        "status",
    ]
    for label in payload["sensitive_use_labels"]:
        assert label["human_review_required"] is True
        assert (
            ProhibitedAutomatedUse.ELIGIBILITY_DECISION.value
            in label["prohibited_automated_uses"]
        )
        assert (
            ProhibitedAutomatedUse.AUTONOMOUS_DIAGNOSIS.value
            in label["prohibited_automated_uses"]
        )


def test_partial_custom_labels_are_rejected_without_echoing_values() -> None:
    with pytest.raises(ValueError, match="every exported SDOH field") as error:
        label_sdoh_output(
            {"category": SYNTHETIC_MARKER, "status": "unknown"},
            labels=(SDOHSensitiveUseLabel(field_name="category"),),
        )

    assert SYNTHETIC_MARKER not in str(error.value)


def test_serializer_requires_validated_label_preserving_export() -> None:
    with pytest.raises(TypeError, match="LabeledSDOHExport"):
        serialize_labeled_sdoh_output(  # type: ignore[arg-type]
            {"fields": {"status": "unknown"}}
        )


def test_serialization_is_deterministic_and_preserves_labels() -> None:
    first = label_sdoh_output({"status": "unknown", "category": "synthetic"})
    second = label_sdoh_output({"category": "synthetic", "status": "unknown"})

    assert serialize_labeled_sdoh_output(first) == serialize_labeled_sdoh_output(second)
    payload = json.loads(first.to_json())
    assert set(payload) == {"fields", "schema_version", "sensitive_use_labels"}


def test_export_repr_does_not_disclose_field_values() -> None:
    export = label_sdoh_output({"status": SYNTHETIC_MARKER})

    assert SYNTHETIC_MARKER not in repr(export)


def test_non_json_value_fails_without_value_representation() -> None:
    class ProtectedValue:
        def __repr__(self) -> str:
            return SYNTHETIC_MARKER

    with pytest.raises(TypeError, match="JSON-compatible") as error:
        LabeledSDOHExport(
            fields={"status": ProtectedValue()},
            labels=(SDOHSensitiveUseLabel(field_name="status"),),
        )

    assert SYNTHETIC_MARKER not in str(error.value)


def test_labeling_performs_no_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", reject_network)

    assert label_sdoh_output({"status": "unknown"}).labels
