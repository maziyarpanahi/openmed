from __future__ import annotations

from openmed.interop.archive_safety import (
    ArchiveDecision,
    ArchiveMember,
    ArchiveSafetyPolicy,
    assess_archive_members,
    inspect_archive_members,
)


def member(
    path: str,
    *,
    compressed: int = 100,
    uncompressed: int = 200,
    kind: str = "file",
    link_target: str | None = None,
) -> ArchiveMember:
    return ArchiveMember(
        path=path,
        compressed_size=compressed,
        uncompressed_size=uncompressed,
        kind=kind,
        link_target=link_target,
    )


def test_clean_metadata_is_allowed_and_deterministic():
    members = [
        member("records/summary.txt"),
        member("records/attachments/item.bin", compressed=50, uncompressed=100),
    ]

    first = inspect_archive_members(members)
    second = assess_archive_members(members)

    assert first == second
    assert first.decision is ArchiveDecision.ALLOW
    assert first.allowed
    assert first.entry_count == 2
    assert first.total_uncompressed_bytes == 300
    assert first.reason_counts == {}


def test_traversal_and_links_are_rejected_without_echoing_member_names():
    sensitive_name = "synthetic-record-001.txt"
    traversal = member(f"reports\\..\\{sensitive_name}")
    linked = member("reports/linked.txt", kind="symlink", link_target=sensitive_name)

    report = inspect_archive_members([traversal, linked])

    assert report.decision is ArchiveDecision.REJECT
    assert report.reason_counts["path_traversal"] == 1
    assert report.reason_counts["link"] == 1
    rendered = repr(report) + repr(report.to_dict())
    assert sensitive_name not in rendered
    assert sensitive_name not in repr(traversal)


def test_duplicates_and_expansion_limits_are_quarantined():
    policy = ArchiveSafetyPolicy(
        max_entries=5,
        max_member_uncompressed_bytes=500,
        max_total_uncompressed_bytes=700,
        max_expansion_ratio=4,
    )
    members = [
        member("records/one.txt", compressed=100, uncompressed=200),
        member("records/./one.txt", compressed=100, uncompressed=200),
        member("records/two.txt", compressed=1, uncompressed=10),
        member("records/three.txt", compressed=100, uncompressed=400),
    ]

    report = inspect_archive_members(members, policy)

    assert report.decision is ArchiveDecision.QUARANTINE
    assert report.reason_counts["duplicate_path"] == 1
    assert report.reason_counts["expansion_ratio"] == 1
    assert report.reason_counts["total_size_limit"] == 1


def test_mapping_metadata_and_entry_limit_are_bounded():
    policy = ArchiveSafetyPolicy(max_entries=2)
    metadata = [
        {
            "name": "records/one.txt",
            "compressed_size": 10,
            "uncompressed_size": 20,
        },
        {
            "path": "records/two.txt",
            "compressed_size": 10,
            "uncompressed_size": 20,
        },
        {
            "path": "records/three.txt",
            "compressed_size": 10,
            "uncompressed_size": 20,
        },
    ]

    report = inspect_archive_members(metadata, policy)

    assert report.decision is ArchiveDecision.QUARANTINE
    assert report.entry_count == 3
    assert report.reason_counts == {"entry_limit": 1}
    assert report.total_uncompressed_bytes == 40


def test_malformed_sizes_are_rejected_without_raw_values_in_errors():
    report = inspect_archive_members(
        [
            {
                "path": "records/invalid.bin",
                "compressed_size": -1,
                "uncompressed_size": 20,
            },
            {
                "path": "records/unknown.bin",
                "compressed_size": 10,
                "uncompressed_size": 20,
                "kind": "unsupported-kind",
            },
        ]
    )

    assert report.decision is ArchiveDecision.REJECT
    assert report.reason_counts["invalid_metadata"] == 2
    assert "records/invalid.bin" not in repr(report)
    assert "unsupported-kind" not in repr(report)
