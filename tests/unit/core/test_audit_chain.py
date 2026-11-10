"""Tests for PHI-safe, tamper-evident de-identification audit chains."""

from __future__ import annotations

import copy
import json
import socket
from pathlib import Path

import pytest

from openmed.core.audit import AuditReport, AuditSpan, DetectorInfo, hash_text
from openmed.core.audit_chain import (
    AuditChain,
    append_to_chain_file,
    verify_chain,
)


def _report(name: str, phone: str) -> AuditReport:
    original = f"Patient {name} called {phone} from North Clinic."
    deidentified = "Patient [PERSON] called [PHONE] from [ORGANIZATION]."
    name_start = original.index(name)
    phone_start = original.index(phone)
    return AuditReport(
        policy="hipaa_safe_harbor",
        resolved_profile={"method": "mask", "language": "en"},
        detectors=[
            DetectorInfo(
                source="ml",
                model_id="local-unit-model",
                model_format="transformers",
                metadata={"operator_note": f"reviewed {name}"},
            )
        ],
        safety_sweep={"spans_added": 0},
        spans=[
            AuditSpan(
                start=name_start,
                end=name_start + len(name),
                label="NAME",
                canonical_label="PERSON",
                sources=["ml"],
                confidence=0.98,
                threshold=0.7,
                action="replace",
                surrogate=f"Replacement for {name}",
                text_hash=hash_text(name),
                evidence={"matched_text": name, "clinic": "North Clinic"},
                context={"before": "Patient ", "after": f" called {phone}"},
            ),
            AuditSpan(
                start=phone_start,
                end=phone_start + len(phone),
                label="PHONE",
                canonical_label="PHONE",
                sources=["regex"],
                confidence=1.0,
                threshold=0.7,
                action="mask",
                surrogate="[PHONE]",
                text_hash=hash_text(phone),
            ),
        ],
        thresholds={"PERSON": 0.7, "PHONE": 0.7},
        residual_risk={"review_note": f"No remaining reference to {name}"},
        openmed_version="1.9.1",
        manifest_hash=hash_text("unit manifest"),
        document_length=len(original),
        input_hash=hash_text(original),
        deidentified_text_hash=hash_text(deidentified),
    )


def _three_entry_chain() -> AuditChain:
    chain = AuditChain()
    chain.append(_report("Alice Rivera", "555-0101"))
    chain.append(_report("Benoit Martin", "555-0102"))
    chain.append(_report("Chandra Patel", "555-0103"))
    return chain


def test_append_produces_a_verifiable_linked_chain() -> None:
    first_report = _report("Alice Rivera", "555-0101")
    second_report = _report("Benoit Martin", "555-0102")
    chain = AuditChain()

    first = chain.append(first_report)
    second = chain.append(second_report)

    assert first.sequence == 0
    assert first.previous_hash == AuditChain.GENESIS_HASH
    assert second.sequence == 1
    assert second.previous_hash == first.entry_hash
    assert second.matches_report(second_report)
    assert chain.contains_report(first_report)
    result = chain.verify()
    assert result.valid
    assert result.checked_entries == 2
    assert result.reason == "chain is valid"
    assert verify_chain(chain).is_valid


def test_chain_round_trips_through_json_and_disk(tmp_path: Path) -> None:
    chain = _three_entry_chain()
    restored = AuditChain.from_json(chain.to_json())

    assert restored.to_dict() == chain.to_dict()
    assert restored.verify().valid

    path = tmp_path / "audit-chain.json"
    chain.write(path)
    assert AuditChain.load(path).verify().valid


def test_append_to_chain_file_preserves_existing_entries(tmp_path: Path) -> None:
    path = tmp_path / "audit-chain.json"

    first = append_to_chain_file(path, _report("Alice Rivera", "555-0101"))
    second = append_to_chain_file(path, _report("Benoit Martin", "555-0102"))

    restored = AuditChain.load(path)
    assert first.sequence == 0
    assert second.sequence == 1
    assert len(restored.entries) == 2
    assert restored.verify().valid


def test_mutated_entry_fails_with_a_clear_reason() -> None:
    payload = json.loads(_three_entry_chain().to_json())
    payload["entries"][1]["spans"][0]["label"] = "PHONE"

    result = AuditChain.from_dict(payload).verify()

    assert not result.valid
    assert result.entry_index == 1
    assert "mutation detected" in result.reason
    assert "hash mismatch" in result.reason


def test_inserted_entry_fails_with_a_clear_reason() -> None:
    payload = json.loads(_three_entry_chain().to_json())
    payload["entries"].insert(1, copy.deepcopy(payload["entries"][0]))

    result = AuditChain.from_dict(payload).verify()

    assert not result.valid
    assert "insertion detected" in result.reason


def test_deleted_entry_fails_with_a_clear_reason() -> None:
    payload = json.loads(_three_entry_chain().to_json())
    payload["entries"].pop(1)

    result = AuditChain.from_dict(payload).verify()

    assert not result.valid
    assert "deletion detected" in result.reason


def test_reordered_entries_fail_with_a_clear_reason() -> None:
    payload = json.loads(_three_entry_chain().to_json())
    payload["entries"][0], payload["entries"][1] = (
        payload["entries"][1],
        payload["entries"][0],
    )

    result = AuditChain.from_dict(payload).verify()

    assert not result.valid
    assert "reordering detected" in result.reason


def test_retained_head_detects_tail_deletion() -> None:
    payload = json.loads(_three_entry_chain().to_json())
    payload["entries"].pop()

    result = AuditChain.from_dict(payload).verify()

    assert not result.valid
    assert "deletion detected" in result.reason


def test_explicit_invalid_head_is_not_replaced_by_the_computed_head() -> None:
    chain = _three_entry_chain()
    restored = AuditChain(chain.entries, declared_head_hash="")

    result = restored.verify()

    assert not result.valid
    assert "retained head hash" in result.reason


def test_chain_serialization_excludes_raw_phi_and_unsafe_report_fields() -> None:
    report = _report("Alice Rivera", "555-0101")
    chain = AuditChain()
    chain.append(report)

    serialized = chain.to_json()

    for plaintext in (
        "Alice Rivera",
        "555-0101",
        "North Clinic",
        "Replacement for",
        "matched_text",
        "operator_note",
        "review_note",
        "Patient ",
    ):
        assert plaintext not in serialized
    assert set(chain.to_dict()["entries"][0]) == {
        "sequence",
        "previous_hash",
        "report_hash",
        "input_hash",
        "deidentified_text_hash",
        "spans",
        "entry_hash",
    }
    assert set(chain.to_dict()["entries"][0]["spans"][0]) == {
        "start",
        "end",
        "label",
        "text_hash",
    }


def test_loading_rejects_unknown_fields_without_echoing_phi() -> None:
    payload = json.loads(_three_entry_chain().to_json())
    phi_field = "Alice Rivera"
    payload["entries"][0][phi_field] = "unsafe"

    with pytest.raises(ValueError, match="unsupported fields") as exc_info:
        AuditChain.from_dict(payload)

    assert phi_field not in str(exc_info.value)


def test_loading_rejects_unknown_version_without_echoing_phi() -> None:
    payload = json.loads(_three_entry_chain().to_json())
    phi_version = "Alice Rivera"
    payload["version"] = phi_version

    with pytest.raises(ValueError, match="unsupported audit chain version") as exc_info:
        AuditChain.from_dict(payload)

    assert phi_version not in str(exc_info.value)


def test_append_rejects_a_report_with_a_tampered_repro_hash() -> None:
    report = _report("Alice Rivera", "555-0101")
    report.policy = "tampered"

    with pytest.raises(ValueError, match="invalid repro_hash"):
        AuditChain().append(report)


def test_verification_is_fully_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    chain = AuditChain.from_json(_three_entry_chain().to_json())

    def deny_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is forbidden during chain verification")

    monkeypatch.setattr(socket, "socket", deny_network)

    assert chain.verify().valid
