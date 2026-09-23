from __future__ import annotations

import copy

from openmed.clinical.journey_contracts import canonical_digest
from openmed.clinical.review_packet_migrations import migrate_review_packet
from openmed.clinical.review_transitions import ClinicalReviewPacket
from openmed.structured.store import StoreState


def _legacy_packet() -> dict[str, object]:
    packet = ClinicalReviewPacket(
        packet_id="reviewpacket_aaaaaaaaaaaaaaaa",
        conflict_id="conflict_aaaaaaaaaaaaaaaa",
        fact_ids=("fact_aaaaaaaaaaaaaaaa", "fact_bbbbbbbbbbbbbbbb"),
        state="queued",
        priority="normal",
        created_at="2026-09-21T10:00:00Z",
        expires_at=None,
        policy_id="openmed.fact.reconciliation",
        policy_version="1.0.0",
        provenance_fingerprint=canonical_digest({"synthetic": "packet"}),
    ).to_dict()
    packet["schema_version"] = "1.0.0"
    packet.pop("compatibility_policy")
    packet.pop("extensions")
    packet.pop("transition_ids")
    return packet


def test_forward_migration_adds_integrity_fields_without_mutating_input() -> None:
    legacy = _legacy_packet()
    original = copy.deepcopy(legacy)

    result = migrate_review_packet(legacy)

    assert result.ok and result.value is not None
    assert legacy == original
    assert result.value.packet.schema_version == "1.1.0"
    assert result.value.report.added_fields == (
        "compatibility_policy",
        "extensions",
        "transition_ids",
    )
    assert not result.value.report.lossy


def test_unknown_or_backward_target_versions_are_typed_unsupported() -> None:
    unknown = _legacy_packet() | {"schema_version": "9.0.0"}

    source = migrate_review_packet(unknown)
    target = migrate_review_packet(_legacy_packet(), target_version="1.0.0")

    assert source.state is StoreState.UNSUPPORTED
    assert source.code == "review_packet_source_unsupported"
    assert target.state is StoreState.UNSUPPORTED
    assert target.code == "review_packet_target_unsupported"


def test_invalid_packet_produces_value_free_failure() -> None:
    payload = _legacy_packet()
    payload["fact_ids"] = ["SYNTHETIC-PRIVATE-CANARY"]

    result = migrate_review_packet(payload)

    assert result.state is StoreState.FAILURE
    assert result.code == "review_packet_invalid"
    assert "CANARY" not in repr(result)
