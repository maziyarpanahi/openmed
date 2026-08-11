"""Tests for deterministic, source-path-free resource path audits."""

from __future__ import annotations

import json

import pytest

from openmed.core.path_portability import (
    ABSOLUTE_ROOT,
    CASE_FOLD_COLLISION,
    NORMALIZATION_DRIFT,
    RESERVED_COMPONENT,
    TRAVERSAL,
    PathPortabilityInputError,
    audit_path_portability,
    audit_resource_paths,
)


def _categories(report):
    return [category for record in report.records for category in record.categories]


def test_audit_covers_required_portability_categories_without_source_text():
    synthetic_paths = [
        "../synthetic-records/entry.json",
        "/synthetic-root/config.json",
        r"C:\synthetic-cache\NUL\config.json",
        "Synthetic/./Cafe\u0301//entry.json",
        "Models/Weights.bin",
        "models/weights.BIN",
    ]

    report = audit_resource_paths(synthetic_paths)
    serialized = report.to_json()

    assert {
        TRAVERSAL,
        ABSOLUTE_ROOT,
        RESERVED_COMPONENT,
        NORMALIZATION_DRIFT,
        CASE_FOLD_COLLISION,
    } <= set(_categories(report))
    assert report.checked_count == len(synthetic_paths)
    assert report.affected_path_count == len(report.findings)
    assert report.is_clean is False
    for source_path in synthetic_paths:
        assert source_path not in serialized
    assert "synthetic-records" not in repr(report)


def test_report_is_deterministic_and_order_independent():
    paths = [
        "Records/Alpha.json",
        "records/alpha.JSON",
        "./records/beta.json",
        "records/../records/gamma.json",
    ]

    forward = audit_resource_paths(paths)
    reverse = audit_path_portability(reversed(paths))

    assert forward == reverse
    assert forward.to_json() == reverse.to_json()
    assert json.loads(forward.to_json()) == forward.to_dict()


def test_clean_relative_path_has_only_a_fingerprint():
    report = audit_resource_paths(["models/weights.bin"])

    assert report.is_clean
    assert report.records[0].issue_categories == ()
    assert report.records[0].normalized_path_fingerprint.startswith("sha256:")
    assert len(report.records[0].normalized_path_fingerprint) == len("sha256:") + 64
    assert report.to_dict()["records"] == [
        {
            "normalized_path_fingerprint": report.records[0].fingerprint,
            "issue_categories": [],
            "occurrences": 1,
        }
    ]


@pytest.mark.parametrize(
    "path",
    [
        "models/CON.txt",
        "models/lpt1 ",
        "models/aux./weights.bin",
    ],
)
def test_windows_reserved_components_are_reported(path):
    report = audit_resource_paths([path])

    assert RESERVED_COMPONENT in report.records[0].issue_categories


@pytest.mark.parametrize(
    "path",
    ["/synthetic-root/file", r"\\server\share\file", "file:///synthetic/file"],
)
def test_host_specific_roots_are_reported(path):
    report = audit_resource_paths([path])

    assert ABSOLUTE_ROOT in report.records[0].issue_categories


def test_input_errors_do_not_echo_source_values():
    secret_marker = "synthetic-sensitive-marker"

    class BrokenPath:
        def __fspath__(self):
            raise ValueError(secret_marker)

    with pytest.raises(PathPortabilityInputError) as exc_info:
        audit_resource_paths([BrokenPath()])

    assert secret_marker not in str(exc_info.value)


def test_duplicate_normalized_paths_are_counted_without_duplicate_records():
    report = audit_resource_paths(["models/weights.bin", "models/weights.bin"])

    assert report.checked_count == 2
    assert len(report.records) == 1
    assert report.records[0].occurrences == 2
