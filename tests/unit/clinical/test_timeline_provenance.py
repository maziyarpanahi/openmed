"""Focused tests for the value-free clinical timeline provenance export."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import (
    DEFAULT_TIMELINE_POLICY_FINGERPRINT,
    export_timeline_provenance,
)
from openmed.core.audit import hash_text

SYNTHETIC_VALUE = "SYNTHETIC_PROTECTED_TIMELINE_VALUE"


def _events() -> list[dict[str, object]]:
    return [
        {
            "event_id": "event-b",
            "start": 24,
            "end": 31,
            "text": SYNTHETIC_VALUE,
            "assertion": {"negation": "affirmed", "certainty": "certain"},
            "temporal_confidence": 0.82,
        },
        {
            "event_id": "event-a",
            "source_offsets": {"start": 4, "end": 13},
            "text": "SYNTHETIC_SECOND_VALUE",
            "assertion_status": "historical",
            "temporal_confidence": 0.91,
        },
    ]


def test_export_is_deterministic_and_value_free() -> None:
    first = export_timeline_provenance(
        _events(),
        policy={"profile": "synthetic-local-policy", "revision": 3},
    )
    second = export_timeline_provenance(
        list(reversed(_events())),
        policy={"revision": 3, "profile": "synthetic-local-policy"},
    )

    assert first == second
    assert [event["event_id"] for event in first["events"]] == [
        "event-a",
        "event-b",
    ]
    assert first["events"][0]["source_offsets"] == {"start": 4, "end": 13}
    assert first["events"][0]["assertion_status"] == "historical"
    assert first["events"][1]["assertion_status"] == "affirmed"
    assert first["events"][1]["temporal_confidence"] == 0.82
    assert first["policy_fingerprint"].startswith("sha256:")
    assert all(
        event["policy_fingerprint"] == first["policy_fingerprint"]
        for event in first["events"]
    )

    serialized = json.dumps(first, sort_keys=True)
    assert SYNTHETIC_VALUE not in serialized
    assert "SYNTHETIC_SECOND_VALUE" not in serialized
    assert '"text"' not in serialized


def test_default_policy_fingerprint_is_stable_and_hashes_are_opt_in() -> None:
    default = export_timeline_provenance(
        [
            {
                "id": "event-1",
                "start": 0,
                "end": 8,
                "text": SYNTHETIC_VALUE,
            }
        ]
    )
    hashed = export_timeline_provenance(
        [
            {
                "id": "event-1",
                "start": 0,
                "end": 8,
                "text": SYNTHETIC_VALUE,
            }
        ],
        include_value_hashes=True,
    )

    assert default["policy_fingerprint"] == DEFAULT_TIMELINE_POLICY_FINGERPRINT
    assert "value_hash" not in default["events"][0]
    assert hashed["events"][0]["value_hash"] == hash_text(SYNTHETIC_VALUE)
    assert SYNTHETIC_VALUE not in json.dumps(hashed, sort_keys=True)


def test_timeline_export_rejects_invalid_offsets_without_echoing_values() -> None:
    with pytest.raises(ValueError, match="source offsets") as error:
        export_timeline_provenance(
            [
                {
                    "event_id": "event-invalid",
                    "start": 8,
                    "end": 8,
                    "text": SYNTHETIC_VALUE,
                }
            ]
        )

    assert SYNTHETIC_VALUE not in str(error.value)


def test_explicit_timeline_position_takes_precedence_over_source_offsets() -> None:
    exported = export_timeline_provenance(
        [
            {"event_id": "offset-first", "start": 1, "end": 2, "position": 1},
            {"event_id": "position-first", "start": 8, "end": 9, "position": 0},
        ]
    )

    assert [event["event_id"] for event in exported["events"]] == [
        "position-first",
        "offset-first",
    ]


def test_policy_to_dict_produces_stable_instance_independent_fingerprint() -> None:
    class SyntheticPolicy:
        def to_dict(self) -> dict[str, object]:
            return {"profile": "synthetic", "revision": 2}

    first = export_timeline_provenance(_events(), policy=SyntheticPolicy())
    second = export_timeline_provenance(_events(), policy=SyntheticPolicy())

    assert first["policy_fingerprint"] == second["policy_fingerprint"]
