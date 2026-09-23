from __future__ import annotations

import copy
import json

import pytest

from openmed.clinical.review_packet_migrations import (
    REVIEW_PACKET_SCHEMA_VERSION,
    SUPPORTED_REVIEW_PACKET_SCHEMA_VERSIONS,
    LossyReviewPacketMigrationError,
    ReviewPacketMigrationError,
    migrate_review_packet,
)


def test_v1_to_v2_is_lossless_deterministic_and_does_not_mutate_input() -> None:
    packet = {
        "schema_version": 1,
        "packet_id_hash": "sha256:synthetic",
        "review": {"items": [{"start": 4, "end": 9, "label": "NAME"}]},
    }
    original = copy.deepcopy(packet)

    first = migrate_review_packet(packet)
    second = migrate_review_packet(packet)

    assert packet == original
    assert first.packet == second.packet
    assert first.report.to_dict() == second.report.to_dict()
    assert first.packet == {
        **original,
        "schema_version": 2,
        "safety": {"privacy_scan_required": True},
    }
    assert first.report.to_dict() == {
        "schema_version": 1,
        "source_version": 1,
        "target_version": 2,
        "changes": [
            {"field_path": "/safety", "operation": "add"},
            {"field_path": "/schema_version", "operation": "replace"},
        ],
    }


def test_existing_safety_fields_are_preserved() -> None:
    packet = {
        "schema_version": 1,
        "safety": {"citation_check_required": True},
    }

    result = migrate_review_packet(packet)

    assert result.packet["safety"] == {
        "citation_check_required": True,
        "privacy_scan_required": True,
    }
    assert [change.field_path for change in result.report.changes] == [
        "/safety/privacy_scan_required",
        "/schema_version",
    ]


def test_current_version_returns_a_deep_copy_and_empty_report() -> None:
    packet = {
        "schema_version": REVIEW_PACKET_SCHEMA_VERSION,
        "safety": {"privacy_scan_required": True},
        "review": {"items": []},
    }

    result = migrate_review_packet(packet)

    assert result.packet == packet
    assert result.packet is not packet
    assert result.packet["review"] is not packet["review"]
    assert result.report.changes == ()


def test_report_and_repr_never_contain_packet_values() -> None:
    sensitive_marker = "SYNTHETIC_LOCAL_REVIEW_VALUE"
    result = migrate_review_packet(
        {"schema_version": 1, "local_review_content": sensitive_marker}
    )

    assert sensitive_marker not in repr(result)
    assert sensitive_marker not in json.dumps(result.report.to_dict(), sort_keys=True)
    assert set(result.report.to_dict()) == {
        "schema_version",
        "source_version",
        "target_version",
        "changes",
    }


@pytest.mark.parametrize(
    "packet",
    [
        {"schema_version": 1, "safety": "SYNTHETIC_LOCAL_REVIEW_VALUE"},
        {
            "schema_version": 1,
            "safety": {"privacy_scan_required": False},
        },
    ],
)
def test_migration_rejects_changes_that_would_overwrite_fields(packet) -> None:
    with pytest.raises(LossyReviewPacketMigrationError) as error:
        migrate_review_packet(packet)

    assert "SYNTHETIC_LOCAL_REVIEW_VALUE" not in str(error.value)
    assert "False" not in str(error.value)


def test_backward_and_unsupported_migrations_are_rejected_value_free() -> None:
    with pytest.raises(LossyReviewPacketMigrationError, match="backward"):
        migrate_review_packet(
            {"schema_version": 2, "local_review_content": "SYNTHETIC_VALUE"},
            target_version=1,
        )

    with pytest.raises(ReviewPacketMigrationError, match="unsupported"):
        migrate_review_packet({"schema_version": 99})

    assert SUPPORTED_REVIEW_PACKET_SCHEMA_VERSIONS == (1, 2)
