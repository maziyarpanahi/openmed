"""Tests for deterministic, source-path-free resource path audits."""

from __future__ import annotations

import itertools
import json

import pytest

from openmed.core.path_portability import (
    ABSOLUTE_ROOT,
    CASE_FOLD_COLLISION,
    ISSUE_CATEGORIES,
    MAX_AUDIT_PATHS,
    MAX_PATH_CHARACTERS,
    MAX_PATH_COMPONENTS,
    NORMALIZATION_DRIFT,
    RESERVED_COMPONENT,
    TRAVERSAL,
    PathPortabilityInputError,
    PathPortabilityRecord,
    PathPortabilityReport,
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


def test_unicode_normalization_drift_is_detected_without_other_path_changes():
    report = audit_resource_paths(["models/Cafe\u0301.json"])

    assert report.records[0].issue_categories == (NORMALIZATION_DRIFT,)


@pytest.mark.parametrize(
    "path",
    [
        "models/name:stream.json",
        "models/question?.json",
        "models/control\x00.json",
        "models/line\u2028separator.json",
        "models/paragraph\u2029separator.json",
        "models/clock$.txt",
        "models/CONIN$.txt",
        "models/CONOUT$.txt",
        "models/COM\u00b9.txt",
        "models/" + ("a" * 256),
        "models/" + ("é" * 128),
    ],
)
def test_cross_platform_invalid_components_are_reserved(path):
    report = audit_resource_paths([path])

    assert RESERVED_COMPONENT in report.records[0].issue_categories


@pytest.mark.parametrize("component", ["a" * 255, ("é" * 127) + "a"])
def test_component_at_portable_byte_limit_remains_clean(component):
    report = audit_resource_paths([f"models/{component}"])

    assert report.is_clean


@pytest.mark.parametrize(
    "paths",
    [
        itertools.repeat("models/weights.bin", MAX_AUDIT_PATHS + 1),
        ["x" * (MAX_PATH_CHARACTERS + 1)],
        ["/".join(["part"] * (MAX_PATH_COMPONENTS + 1))],
    ],
)
def test_input_work_is_bounded(paths):
    with pytest.raises(PathPortabilityInputError):
        audit_resource_paths(paths)


def test_hostile_iterator_failures_do_not_echo_source_values():
    secret_marker = "synthetic-sensitive-iterator-marker"

    class HostilePaths:
        def __iter__(self):
            return self

        def __next__(self):
            raise RuntimeError(secret_marker)

    with pytest.raises(PathPortabilityInputError) as exc_info:
        audit_resource_paths(HostilePaths())

    assert secret_marker not in str(exc_info.value)


def test_text_subclasses_and_surrogates_are_rejected_without_source_text():
    class UnsafeText(str):
        pass

    for path in (UnsafeText("synthetic-sensitive-path"), "bad\ud800path"):
        with pytest.raises(PathPortabilityInputError) as exc_info:
            audit_resource_paths(path)

        assert "synthetic-sensitive-path" not in str(exc_info.value)


def test_public_report_state_must_be_exact_immutable_and_consistent():
    fingerprint = "sha256:" + "a" * 64
    record = PathPortabilityRecord(fingerprint)

    with pytest.raises(ValueError):
        PathPortabilityRecord(
            fingerprint,
            issue_categories=[TRAVERSAL],  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        PathPortabilityRecord(
            fingerprint,
            issue_categories=(TRAVERSAL, TRAVERSAL),
        )
    with pytest.raises(ValueError):
        PathPortabilityRecord(
            fingerprint,
            issue_categories=(TRAVERSAL,) * (len(ISSUE_CATEGORIES) + 1),
        )
    with pytest.raises(ValueError):
        PathPortabilityReport(records=(record,), checked_count=2)
    with pytest.raises(ValueError):
        PathPortabilityReport(
            records=[record],  # type: ignore[arg-type]
            checked_count=1,
        )
