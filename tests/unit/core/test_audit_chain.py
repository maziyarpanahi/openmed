"""Tests for PHI-safe, tamper-evident de-identification audit chains."""

from __future__ import annotations

import copy
import json
import socket
from pathlib import Path

import pytest

from openmed.core.audit import AuditReport, AuditSpan, DetectorInfo, hash_text
from openmed.core.audit_chain import (
    CHAIN_VERIFICATION_CODES,
    CHAIN_VERSION,
    GENESIS_HASH,
    PRUNED_CHAIN_VERSION,
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


def test_unpruned_chains_keep_the_version_one_document() -> None:
    payload = _three_entry_chain().to_dict()

    assert payload["version"] == CHAIN_VERSION
    assert "base_sequence" not in payload
    assert "base_hash" not in payload


def test_pruned_chain_links_to_its_retained_base_and_round_trips(
    tmp_path: Path,
) -> None:
    chain = _three_entry_chain()
    first_hash = chain.entries[0].entry_hash

    removed = chain.prune_through(0)

    assert [entry.sequence for entry in removed] == [0]
    assert chain.is_pruned
    assert chain.base_sequence == 1
    assert chain.base_hash == first_hash
    assert chain.next_sequence == 3
    assert chain.entry_at(0) is None
    assert chain.entry_at(1) is not None
    assert chain.verify().valid
    payload = chain.to_dict()
    assert payload["version"] == PRUNED_CHAIN_VERSION
    assert payload["base_sequence"] == 1
    serialized = chain.to_json()
    assert "Alice Rivera" not in serialized
    assert hash_text("Alice Rivera") not in serialized

    path = tmp_path / "audit-chain.json"
    chain.write(path)
    restored = AuditChain.load(path)
    assert restored.to_dict() == payload
    assert restored.verify().valid
    entry = restored.append(_report("Dmitri Okafor", "555-0104"))
    assert entry.sequence == 3
    assert restored.verify().valid


def test_pruning_every_entry_keeps_the_head_and_continues_the_sequence() -> None:
    chain = _three_entry_chain()
    head = chain.head_hash

    chain.prune_through(2)

    assert chain.entries == ()
    assert chain.head_hash == head == chain.base_hash
    assert chain.verify().valid
    assert chain.append(_report("Dmitri Okafor", "555-0104")).previous_hash == head


def test_pruning_rejects_out_of_range_sequences_and_invalid_chains() -> None:
    chain = _three_entry_chain()
    chain.prune_through(0)
    for sequence in (0, 3, -1):
        with pytest.raises(ValueError, match="retained audit chain entry"):
            chain.prune_through(sequence)

    payload = json.loads(_three_entry_chain().to_json())
    payload["entries"].pop(1)
    with pytest.raises(ValueError, match="cannot prune an invalid audit chain"):
        AuditChain.from_dict(payload).prune_through(0)


def test_pruned_document_without_its_base_fails_closed() -> None:
    chain = _three_entry_chain()
    chain.prune_through(0)
    payload = chain.to_dict()

    stripped = dict(payload)
    stripped.pop("base_sequence")
    stripped.pop("base_hash")
    stripped["version"] = CHAIN_VERSION
    assert AuditChain.from_dict(stripped).verify().code == "sequence_gap"

    for field, value in (("base_sequence", 0), ("base_hash", GENESIS_HASH)):
        with pytest.raises(ValueError):
            AuditChain.from_dict({**payload, field: value})


def test_verification_results_carry_stable_value_free_codes() -> None:
    def broken(mutate):
        payload = json.loads(_three_entry_chain().to_json())
        mutate(payload)
        return AuditChain.from_dict(payload).verify().code

    def swap(payload):
        entries = payload["entries"]
        entries[0], entries[1] = entries[1], entries[0]

    def relabel(payload):
        payload["entries"][1]["spans"][0]["label"] = "PHONE"

    def duplicate(payload):
        payload["entries"].insert(1, copy.deepcopy(payload["entries"][0]))

    def drop(payload):
        payload["entries"].pop(1)

    def renumber(payload):
        payload["entries"][2]["sequence"] = 7

    assert _three_entry_chain().verify().code == "valid"
    assert broken(swap) == "sequence_reordered"
    assert broken(relabel) == "entry_hash_mismatch"
    assert broken(duplicate) == "entry_count_excess"
    assert broken(drop) == "entry_count_deficit"
    assert broken(renumber) == "sequence_gap"
    assert (
        AuditChain(_three_entry_chain().entries, declared_head_hash=GENESIS_HASH)
        .verify()
        .code
        == "head_mismatch"
    )
    assert all(code.replace("_", "").isalpha() for code in CHAIN_VERIFICATION_CODES)
