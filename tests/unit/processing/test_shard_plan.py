"""Tests for deterministic offline file-shard planning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.processing.shard_plan import (
    DuplicateFileDescriptorError,
    FileDescriptor,
    FileTooLargeError,
    InvalidFileDescriptorError,
    ShardLimits,
    plan_file_shards,
    stable_path_fingerprint,
)


def _descriptors() -> list[FileDescriptor]:
    return [
        FileDescriptor("synthetic/alpha.bin", size_bytes=8),
        FileDescriptor("synthetic/bravo.bin", size_bytes=7),
        FileDescriptor("synthetic/charlie.bin", size_bytes=6),
        FileDescriptor("synthetic/delta.bin", size_bytes=5),
        FileDescriptor("synthetic/echo.bin", size_bytes=4),
        FileDescriptor("synthetic/foxtrot.bin", size_bytes=3),
    ]


def test_plan_is_deterministic_and_balanced_independent_of_input_order() -> None:
    limits = ShardLimits(max_bytes=10, max_files=2)

    first = plan_file_shards(_descriptors(), limits=limits)
    second = plan_file_shards(list(reversed(_descriptors())), limits=limits)

    assert first.to_json() == second.to_json()
    assert first.file_count == 6
    assert first.total_bytes == 33
    assert [(shard.file_count, shard.total_bytes) for shard in first.shards] == [
        (1, 8),
        (1, 7),
        (2, 9),
        (2, 9),
    ]
    assert all(
        shard.total_bytes <= limits.max_bytes and shard.file_count <= limits.max_files
        for shard in first.shards
    )


def test_planner_accepts_limit_aliases_and_never_opens_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_open(*args: object, **kwargs: object) -> None:
        raise AssertionError("planning must not open a path")

    monkeypatch.setattr(Path, "open", fail_open)
    monkeypatch.setattr(Path, "stat", fail_open)

    plan = plan_file_shards(
        [
            {"path": "synthetic/one.bin", "byte_size": 3},
            {"path": "synthetic/two.bin", "byte_size": 2},
        ],
        max_shard_bytes=4,
        max_files_per_shard=2,
    )

    assert plan.shard_count == 2
    assert plan.shards[0].total_bytes == 3
    assert plan.shards[1].total_bytes == 2


def test_duplicate_paths_are_rejected_without_echoing_path_values() -> None:
    with pytest.raises(DuplicateFileDescriptorError) as exc_info:
        plan_file_shards(
            [
                FileDescriptor("synthetic/repeated.bin", size_bytes=1),
                FileDescriptor("synthetic/./repeated.bin", size_bytes=2),
            ],
            max_bytes=4,
            max_files=2,
        )

    message = str(exc_info.value)
    assert "Duplicate file descriptor" in message
    assert "synthetic" not in message
    assert "repeated" not in message


def test_oversized_descriptor_is_rejected_without_echoing_path_values() -> None:
    with pytest.raises(FileTooLargeError) as exc_info:
        plan_file_shards(
            [FileDescriptor("synthetic/oversized.bin", size_bytes=11)],
            max_bytes=10,
            max_files=1,
        )

    message = str(exc_info.value)
    assert "exceeds max_bytes" in message
    assert "oversized" not in message


def test_counts_only_serialization_excludes_paths_and_membership() -> None:
    plan = plan_file_shards(_descriptors(), max_bytes=10, max_files=2)

    serialized = plan.to_dict()
    rendered = json.dumps(serialized, sort_keys=True)

    assert serialized["file_count"] == 6
    assert serialized["total_bytes"] == 33
    assert all("entries" not in shard for shard in serialized["shards"])
    assert all("path" not in shard for shard in serialized["shards"])
    assert "synthetic/alpha.bin" not in rendered
    assert "synthetic/bravo.bin" not in rendered
    assert "path_fingerprint" not in rendered


def test_path_fingerprint_is_lexical_and_stable_without_filesystem_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_stat(*args: object, **kwargs: object) -> None:
        raise AssertionError("fingerprinting must not inspect the filesystem")

    monkeypatch.setattr(Path, "stat", fail_stat)

    assert stable_path_fingerprint("synthetic/./alpha.bin") == (
        stable_path_fingerprint("synthetic/alpha.bin")
    )
    assert stable_path_fingerprint("synthetic/alpha.bin") != stable_path_fingerprint(
        "synthetic/bravo.bin"
    )


def test_invalid_descriptor_metadata_is_rejected_without_raw_values() -> None:
    with pytest.raises(InvalidFileDescriptorError) as exc_info:
        plan_file_shards(
            [{"path": "synthetic/bad.bin", "size_bytes": -1}],
            max_bytes=10,
            max_files=1,
        )

    message = str(exc_info.value)
    assert "invalid metadata" in message
    assert "synthetic" not in message
    assert "bad.bin" not in message
