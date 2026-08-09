"""Focused tests for the counts-only local trace privacy inventory."""

from __future__ import annotations

import json

import pytest

from openmed.traces.audit import (
    TraceAudit,
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
    assert report.by_file[("codex", "session-a.jsonl")]["byte_ranges"] == [
        {"start": 2, "end": 8},
        {"start": 20, "end": 34},
    ]
    assert payload["findings"][0]["file"] == "session-a.jsonl"
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
def test_absolute_file_labels_are_reduced_to_a_basename(
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

    assert report.files[0]["file"] == "synthetic-trace.jsonl"
    assert directory not in report.to_terminal()


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
