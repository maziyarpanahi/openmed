"""Focused tests for the counts-only local trace privacy inventory."""

from __future__ import annotations

import json

import pytest

from openmed.traces.audit import (
    TraceAudit,
    TraceAuditReport,
    TraceFinding,
    TraceScan,
    build_trace_audit,
)

_SENSITIVE_VALUE = "synthetic-value-must-not-appear"


def test_aggregates_findings_by_store_category_and_file_without_values() -> None:
    report = build_trace_audit(
        findings=[
            {
                "store": "codex",
                "category": "prompt",
                "file": "session-a.jsonl",
                "start": 20,
                "end": 34,
                "value": _SENSITIVE_VALUE,
            },
            {
                "store": "codex",
                "category": "prompt",
                "file": "session-a.jsonl",
                "byte_range": [2, 8],
                "raw": _SENSITIVE_VALUE,
            },
            {
                "store": "codex",
                "category": "tool-output",
                "file": "session-b.jsonl",
                "start": 4,
                "end": 10,
                "surface": _SENSITIVE_VALUE,
            },
        ],
        scans=[
            TraceScan("codex", "session-a.jsonl", "scanned"),
            TraceScan("codex", "session-b.jsonl", "scanned"),
            TraceScan("claude", "session-c.jsonl", "skipped"),
            TraceScan("cursor", "session-d.jsonl", "unreadable"),
            TraceScan("other", "session-e.bin", "unsupported"),
        ],
    )

    payload = report.to_dict()
    assert payload["totals"] == {
        "scanned": 2,
        "skipped": 1,
        "unreadable": 1,
        "unsupported": 1,
    }
    assert payload["finding_count"] == 3
    assert report.by_store["codex"]["count"] == 3
    assert report.by_category["prompt"]["count"] == 2
    file_label = str(payload["findings"][0]["file"])
    assert file_label.startswith("file_sha256_")
    assert report.by_file[("codex", file_label)]["byte_ranges"] == (
        {"start": 2, "end": 8},
        {"start": 20, "end": 34},
    )
    assert "session-a.jsonl" not in report.to_json()
    assert _SENSITIVE_VALUE not in report.to_json()
    assert _SENSITIVE_VALUE not in report.to_terminal()


def test_json_and_terminal_formats_are_deterministic() -> None:
    first = build_trace_audit(
        findings=[
            TraceFinding("z-store", "z-category", "z-file", 9, 12),
            TraceFinding("a-store", "a-category", "a-file", 1, 3),
        ],
        statuses={"scanned": 2, "skipped": 1},
    )
    second = build_trace_audit(
        findings=[
            TraceFinding("a-store", "a-category", "a-file", 1, 3),
            TraceFinding("z-store", "z-category", "z-file", 9, 12),
        ],
        statuses={"skipped": 1, "scanned": 2},
    )

    assert first.to_json() == second.to_json()
    assert first.to_json(indent=None) == second.to_json(indent=None)
    assert first.to_terminal() == second.to_terminal()
    assert json.loads(first.to_json()) == first.to_dict()


def test_collector_is_read_only_and_accepts_opaque_status_items() -> None:
    source = {
        "store": "synthetic",
        "category": "message",
        "file": "trace.jsonl",
        "start": 0,
        "end": 5,
        "value": _SENSITIVE_VALUE,
    }
    original = dict(source)
    collector = TraceAudit()
    collector.add_finding(source)
    collector.record_status("skipped", 2)
    collector.record_status("unreadable")
    report = collector.snapshot()

    assert source == original
    assert report.scanned == 1
    assert report.skipped == 2
    assert report.unreadable == 1
    assert report.unsupported == 0
    assert _SENSITIVE_VALUE not in report.to_json()


@pytest.mark.parametrize(
    "file_label, directory",
    [
        ("/private/example/synthetic-trace.jsonl", "/private/example"),
        (r"C:\private\example\synthetic-trace.jsonl", "C:/private/example"),
    ],
)
def test_file_labels_are_hashed_without_exposing_paths(
    file_label: str, directory: str
) -> None:
    report = build_trace_audit(
        [
            {
                "store": "synthetic",
                "category": "message",
                "file": file_label,
                "start": 3,
                "end": 7,
            }
        ]
    )

    assert str(report.files[0]["file"]).startswith("file_sha256_")
    assert "synthetic-trace.jsonl" not in report.to_json()
    assert directory not in report.to_terminal()


def test_caller_controlled_dimensions_are_hashed() -> None:
    sensitive = "PatientJaneDoe"
    report = build_trace_audit(
        [
            {
                "store": sensitive,
                "category": sensitive,
                "file": f"../../patients/{sensitive}.jsonl",
                "start": 0,
                "end": 4,
            }
        ]
    )

    json_report = report.to_json()
    terminal_report = report.to_terminal()
    assert sensitive not in json_report
    assert sensitive not in terminal_report
    assert "../" not in terminal_report
    assert str(report.stores[0]["store"]).startswith("store_sha256_")
    assert str(report.categories[0]["category"]).startswith("category_sha256_")
    assert str(report.files[0]["file"]).startswith("file_sha256_")


@pytest.mark.parametrize("input_name", ["findings", "statuses", "scanned"])
def test_input_iteration_errors_do_not_expose_values(input_name: str) -> None:
    sensitive = "PatientJaneDoe"

    def failing_input():
        raise RuntimeError(sensitive)
        yield None

    kwargs = {input_name: failing_input()}
    with pytest.raises(ValueError) as caught:
        build_trace_audit(**kwargs)

    assert sensitive not in str(caught.value)


def test_mapping_lookup_errors_do_not_expose_values() -> None:
    class FailingMapping(dict[str, object]):
        def get(self, key: str, default: object = None) -> object:
            raise RuntimeError(_SENSITIVE_VALUE)

    with pytest.raises(ValueError) as caught:
        build_trace_audit([FailingMapping()])

    assert _SENSITIVE_VALUE not in str(caught.value)


def test_hostile_metadata_strings_are_normalized_without_running_overrides() -> None:
    class HostileString(str):
        def encode(self, *args: object, **kwargs: object) -> bytes:
            del args, kwargs
            raise RuntimeError(_SENSITIVE_VALUE)

        def strip(self, *args: object) -> str:
            del args
            raise RuntimeError(_SENSITIVE_VALUE)

    finding = TraceFinding(
        HostileString("codex"),
        HostileString("message"),
        HostileString("trace.jsonl"),
        0,
        4,
    )
    scan = TraceScan(status=HostileString("scanned"))

    assert finding.store == "codex"
    assert finding.category == "message"
    assert finding.file.startswith("file_sha256_")
    assert scan.status == "scanned"


def test_hostile_byte_range_iteration_error_is_sanitized() -> None:
    class HostileRange(list[int]):
        def __iter__(self):
            raise RuntimeError(_SENSITIVE_VALUE)

    with pytest.raises(TypeError) as caught:
        build_trace_audit([{"category": "message", "byte_range": HostileRange([0, 4])}])

    assert _SENSITIVE_VALUE not in str(caught.value)


def test_direct_scan_iteration_errors_do_not_expose_values() -> None:
    def failing_findings():
        raise RuntimeError(_SENSITIVE_VALUE)
        yield None

    with pytest.raises(ValueError) as caught:
        TraceScan(findings=failing_findings())  # type: ignore[arg-type]

    assert _SENSITIVE_VALUE not in str(caught.value)


def test_byte_totals_keep_identical_ranges_from_distinct_files() -> None:
    report = build_trace_audit(
        [
            TraceFinding("codex", "message", "first.jsonl", 0, 4),
            TraceFinding("codex", "message", "second.jsonl", 0, 4),
        ]
    )

    assert report.stores[0]["byte_count"] == 8
    assert report.categories[0]["byte_count"] == 8
    assert [item["byte_count"] for item in report.files] == [4, 4]


def test_byte_totals_count_overlapping_ranges_once_per_file() -> None:
    report = build_trace_audit(
        [
            TraceFinding("codex", "message", "trace.jsonl", 0, 10),
            TraceFinding("codex", "prompt", "trace.jsonl", 5, 15),
        ]
    )

    assert report.stores[0]["byte_count"] == 15
    assert report.files[0]["byte_count"] == 15
    assert [item["byte_count"] for item in report.categories] == [10, 10]


def test_direct_report_construction_drops_unknown_fields_and_freezes_rows() -> None:
    sensitive = "PatientJaneDoe"
    source_row = {
        "store": sensitive,
        "count": 1,
        "byte_ranges": [{"start": 0, "end": 4}],
        "category_count": 1,
        "file_count": 1,
        "value": sensitive,
    }
    report = TraceAuditReport(
        totals={"scanned": 1},
        stores=(source_row,),
        categories=(),
        files=(),
        findings=(),
    )
    source_row["store"] = "changed-after-construction"
    source_row["byte_ranges"][0]["start"] = 3

    serialized = report.to_json()
    assert sensitive not in serialized
    assert "changed-after-construction" not in serialized
    assert "value" not in report.stores[0]
    assert str(report.stores[0]["store"]).startswith("store_sha256_")
    assert report.stores[0]["byte_ranges"] == ({"start": 0, "end": 4},)
    with pytest.raises(TypeError):
        report.stores[0]["store"] = sensitive  # type: ignore[index]
    with pytest.raises(TypeError):
        report.stores[0]["byte_ranges"][0]["start"] = 3  # type: ignore[index]


def test_serialized_report_rows_are_detached_from_the_frozen_snapshot() -> None:
    report = build_trace_audit([TraceFinding("codex", "message", "trace.jsonl", 0, 4)])

    payload = report.to_dict()
    payload["stores"][0]["byte_ranges"][0]["start"] = _SENSITIVE_VALUE

    assert _SENSITIVE_VALUE not in report.to_json()
    assert report.stores[0]["byte_ranges"] == ({"start": 0, "end": 4},)


def test_invalid_input_is_value_free() -> None:
    with pytest.raises(ValueError, match="byte range") as exc_info:
        build_trace_audit(
            [
                {
                    "store": "synthetic",
                    "category": "message",
                    "file": "trace.jsonl",
                    "start": 4,
                    "end": 1,
                    "value": _SENSITIVE_VALUE,
                }
            ]
        )

    assert _SENSITIVE_VALUE not in str(exc_info.value)


def test_zero_status_totals_are_valid_for_an_empty_inventory() -> None:
    report = build_trace_audit(
        scanned=0,
        skipped=0,
        unreadable=0,
        unsupported=0,
    )

    assert report.status_counts == {
        "scanned": 0,
        "skipped": 0,
        "unreadable": 0,
        "unsupported": 0,
    }
    assert report.findings == ()
