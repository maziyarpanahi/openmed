"""Focused tests for the PHI-safe 21 CFR Part 11 audit trail."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from openmed.cli import main_module
from openmed.compliance.part11 import (
    PART11_FORMAT,
    PART11_READINESS_CHECKLIST,
    Part11AuditEmitter,
    Part11State,
    build_part11_audit_trail,
    hash_state,
)
from openmed.core.audit import hash_text

_TIMESTAMP = "2026-08-04T10:00:00Z"
_SYNTHETIC_BEFORE = "synthetic-before-state"
_SYNTHETIC_AFTER = "synthetic-after-state"


def _event() -> dict[str, object]:
    return {
        "actor_id": "reviewer-001",
        "action": "record.update",
        "timestamp_utc": _TIMESTAMP,
        "before_state": {
            "label": "pending",
            "hash": hash_text(_SYNTHETIC_BEFORE),
        },
        "after_state": {
            "label": "approved",
            "hash": hash_text(_SYNTHETIC_AFTER),
        },
        "reason_code": "synthetic-review",
    }


def test_emitter_produces_required_part11_fields_and_utc_timestamp() -> None:
    trail = build_part11_audit_trail([_event()])

    assert trail.verify() is True
    assert len(trail.records) == 1
    record = trail.records[0]
    assert record.actor_id == "reviewer-001"
    assert record.action == "record.update"
    assert record.timestamp_utc == _TIMESTAMP
    assert record.before_state == Part11State("pending", hash_text(_SYNTHETIC_BEFORE))
    assert record.after_state == Part11State("approved", hash_text(_SYNTHETIC_AFTER))
    assert record.reason_code == "synthetic-review"
    assert record.chain_sequence == 0
    assert record.chain_previous_hash == trail.chain.GENESIS_HASH
    assert record.chain_entry_hash == trail.chain.records[0].record_hash
    assert record.verify() is True


def test_export_round_trip_verifies_record_and_chain_as_a_unit() -> None:
    trail = build_part11_audit_trail(
        [_event(), {**_event(), "action": "record.review"}]
    )

    restored = Part11AuditEmitter.from_json(trail.to_json())

    assert restored.verify() is True
    assert restored.to_dict() == trail.to_dict()
    assert [record.chain_sequence for record in restored.records] == [0, 1]
    assert (
        restored.records[1].chain_previous_hash == restored.records[0].chain_entry_hash
    )


def test_tampering_record_or_chain_fails_verification() -> None:
    trail = build_part11_audit_trail([_event()])

    object.__setattr__(trail.records[0], "record_hash", hash_text("tampered-record"))
    assert trail.verify() is False

    trail = build_part11_audit_trail([_event()])
    object.__setattr__(trail.chain.records[0], "payload", {"action": "tampered"})
    assert trail.verify() is False

    payload = build_part11_audit_trail([_event()]).to_dict()
    payload = copy.deepcopy(payload)
    payload["records"][0]["action"] = "tampered-export"
    assert Part11AuditEmitter.from_dict(payload).verify() is False


def test_state_helpers_and_export_contain_no_raw_state_values() -> None:
    before = hash_state(_SYNTHETIC_BEFORE, label="pending")
    after = hash_state(_SYNTHETIC_AFTER, label="approved")
    trail = Part11AuditEmitter()
    trail.emit(
        "reviewer-001",
        "record.update",
        before,
        after,
        "synthetic-review",
        timestamp=_TIMESTAMP,
    )

    serialized = trail.to_json()
    assert _SYNTHETIC_BEFORE not in serialized
    assert _SYNTHETIC_AFTER not in serialized
    assert '"before_state"' in serialized
    assert '"after_state"' in serialized
    assert '"label": "pending"' in serialized
    assert '"label": "approved"' in serialized

    with pytest.raises(ValueError, match="unsupported fields") as exc_info:
        trail.emit(
            "reviewer-001",
            "record.update",
            {"label": "pending", "value": _SYNTHETIC_BEFORE},
            after,
            "synthetic-review",
            timestamp=_TIMESTAMP,
        )
    assert _SYNTHETIC_BEFORE not in str(exc_info.value)


def test_readiness_checklist_maps_each_clause_to_fields() -> None:
    clauses = {item.clause for item in PART11_READINESS_CHECKLIST}
    fields = {
        field for item in PART11_READINESS_CHECKLIST for field in item.emitter_fields
    }

    assert "11.10(e)" in clauses
    assert "11.30" in clauses
    assert "11.300" in clauses
    assert {"actor_id", "timestamp_utc", "before_state", "after_state"} <= fields
    assert all(item.emitter_fields for item in PART11_READINESS_CHECKLIST)
    assert all(item.to_dict()["emitter_fields"] for item in PART11_READINESS_CHECKLIST)


def test_cli_exports_a_verified_part11_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "synthetic-events.json"
    output_path = tmp_path / "part11-audit-trail.json"
    input_path.write_text(json.dumps({"events": [_event()]}), encoding="utf-8")

    result = main_module.main(
        [
            "compliance",
            "part11-export",
            str(input_path),
            "--output",
            str(output_path),
            "--json",
        ]
    )

    assert result == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["ok"] is True
    assert envelope["data"]["record_count"] == 1
    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert exported["format"] == PART11_FORMAT
    assert Part11AuditEmitter.from_dict(exported).verify() is True
    assert _SYNTHETIC_BEFORE not in output_path.read_text(encoding="utf-8")


def test_cli_supports_nested_part11_export_alias(tmp_path: Path) -> None:
    input_path = tmp_path / "events.json"
    output_path = tmp_path / "trail.json"
    input_path.write_text(json.dumps([_event()]), encoding="utf-8")

    result = main_module.main(
        [
            "compliance",
            "part11",
            "export",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    assert Part11AuditEmitter.load(output_path).verify() is True
