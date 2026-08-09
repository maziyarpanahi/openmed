from __future__ import annotations

import json
from importlib import import_module
from types import SimpleNamespace

import pytest

from openmed.interop.beam import (
    BeamRedactionError,
    BeamRedactionSpec,
    BeamRedactionTransform,
    run_synthetic_harness,
    serialize_records,
)

beam_adapter = import_module("openmed.interop.beam")
_SENSITIVE = "synthetic-person-001@example.test"


def _fake_deidentifier(text: str, **kwargs):
    assert kwargs["policy"] == "hipaa_safe_harbor"
    assert kwargs["method"] == "mask"
    return SimpleNamespace(
        deidentified_text=text.replace(_SENSITIVE, "[EMAIL]"),
        pii_entities=[object()] if _SENSITIVE in text else [],
    )


def test_spec_is_explicit_bounded_and_value_free():
    spec = BeamRedactionSpec(
        text_field="note",
        max_records=2,
        max_input_bytes=512,
        max_record_bytes=256,
        max_attempts=2,
        extra_kwargs={"seed": 7, "secret": _SENSITIVE},
    )

    metadata = json.dumps(spec.to_dict(), sort_keys=True)

    assert spec.input_schema == "string_or_mapping"
    assert spec.output_schema == "same_as_input"
    assert spec.to_dict()["max_records"] == 2
    assert spec.to_dict()["extra_keys"] == ("secret", "seed")
    assert _SENSITIVE not in metadata
    assert _SENSITIVE not in repr(spec)
    assert spec.fingerprint().startswith("sha256:")


def test_direct_harness_serializes_retries_and_reports_only_aggregates():
    attempts = 0

    def flaky_deidentifier(text: str, **kwargs):
        nonlocal attempts
        del kwargs
        attempts += 1
        if attempts == 1:
            raise RuntimeError(f"driver detail: {text}")
        return text.replace(_SENSITIVE, "[EMAIL]")

    result = run_synthetic_harness(
        [
            {"record_id": "synthetic-1", "note": _SENSITIVE},
            {"record_id": "synthetic-2", "note": "synthetic note"},
        ],
        spec=BeamRedactionSpec(text_field="note", max_attempts=2),
        deidentifier=flaky_deidentifier,
    )

    report = json.dumps(result.report(), sort_keys=True)

    assert result.redacted_records == (
        {"record_id": "synthetic-1", "note": "[EMAIL]"},
        {"record_id": "synthetic-2", "note": "synthetic note"},
    )
    assert result.counters.to_dict() == {
        "attempts": 3,
        "input_bytes": 121,
        "output_bytes": 95,
        "records_changed": 1,
        "records_failed": 0,
        "records_processed": 2,
        "retries": 1,
        "spans_redacted": 1,
    }
    assert result.serialized_output == serialize_records(result.redacted_records)
    assert _SENSITIVE not in report
    assert _SENSITIVE not in repr(result)


def test_harness_bounds_state_and_sanitizes_failures():
    with pytest.raises(BeamRedactionError, match="configured limit") as limit_error:
        run_synthetic_harness(
            ["one", "two"],
            spec=BeamRedactionSpec(max_records=1),
            deidentifier=lambda text, **_: text,
        )
    assert _SENSITIVE not in str(limit_error.value)

    def broken_deidentifier(text: str, **kwargs):
        del kwargs
        raise RuntimeError(f"raw failure: {text}")

    with pytest.raises(BeamRedactionError) as failure_error:
        run_synthetic_harness(
            [_SENSITIVE],
            spec=BeamRedactionSpec(max_attempts=2),
            deidentifier=broken_deidentifier,
        )
    assert _SENSITIVE not in str(failure_error.value)


def test_transform_requires_optional_beam_only_when_expanded(monkeypatch):
    transform = BeamRedactionTransform(deidentifier=_fake_deidentifier)
    monkeypatch.setattr(beam_adapter, "_beam", None)

    with pytest.raises(ImportError, match=r"openmed\[beam\]"):
        transform.expand(object())
