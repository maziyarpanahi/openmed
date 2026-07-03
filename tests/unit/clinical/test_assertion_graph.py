"""Tests for document-level clinical assertion graph reconciliation."""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

from openmed.clinical import (
    ASSERTION_GRAPH_ADVISORY,
    ASSERTION_GRAPH_AXES,
    SECTION_RECONCILIATION_PRECEDENCE,
    AssertionGraphResult,
    reconcile_assertions,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "fixtures" / "clinical" / "assertion_graph_gold.jsonl"
FORBIDDEN_FIXTURE_MARKERS = (
    "cpt",
    "dua",
    "i2b2",
    "mimic",
    "n2c2",
    "snomed",
    "umls",
)


def test_assertion_graph_gold_corpus_is_synthetic_and_complete() -> None:
    meta, rows = _load_jsonl_suite(FIXTURE)

    assert meta["version"] == 1
    assert meta["suite"] == "clinical_assertion_graph_gold"
    assert {row["case_id"] for row in rows} == {
        "assert-graph-001",
        "assert-graph-002",
        "assert-graph-003",
        "assert-graph-004",
        "assert-graph-005",
        "assert-graph-006",
    }
    assert all(row["synthetic"] is True for row in rows)

    fixture_text = FIXTURE.read_text(encoding="utf-8").casefold()
    for marker in FORBIDDEN_FIXTURE_MARKERS:
        assert (
            re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", fixture_text)
            is None
        )


def test_gold_corpus_axis_accuracy_and_status_consistency_gate() -> None:
    _, rows = _load_jsonl_suite(FIXTURE)
    correct_axes = 0
    total_axes = 0

    for row in rows:
        result = reconcile_assertions(row["spans"])
        assertions_by_entity = result.assertions_by_entity
        for entity_key, expected in row["expected"]["assertions"].items():
            assertion = assertions_by_entity[entity_key]
            actual_axes = assertion.assertion.to_dict()
            for axis in ASSERTION_GRAPH_AXES:
                total_axes += 1
                if actual_axes[axis] == expected[axis]:
                    correct_axes += 1
            assert assertion.clinical_status == expected["clinical_status"]
            assert assertion.conflicted_axes == tuple(
                expected.get("conflicted_axes", ())
            )

    assert correct_axes / total_axes >= 0.85
    assert correct_axes == total_axes


def test_gold_corpus_reports_expected_cross_section_conflicts() -> None:
    _, rows = _load_jsonl_suite(FIXTURE)

    for row in rows:
        result = reconcile_assertions(row["spans"])
        expected_conflicts = row["expected"]["conflicts"]
        assert len(result.conflicts) == len(expected_conflicts), row["case_id"]

        for expected in expected_conflicts:
            conflict = _find_conflict(result, expected["entity_key"], expected["axis"])
            assert conflict.values == tuple(expected["values"])
            assert conflict.sections == tuple(expected["sections"])
            assert {item.section for item in conflict.evidence} == set(
                expected["sections"]
            )


def test_reconciliation_is_independent_of_span_input_order() -> None:
    _, rows = _load_jsonl_suite(FIXTURE)

    for row in rows:
        expected = _result_signature(reconcile_assertions(row["spans"]))
        for permutation in itertools.permutations(row["spans"]):
            assert _result_signature(reconcile_assertions(permutation)) == expected


def test_public_api_exposes_disclaimer_and_per_axis_provenance() -> None:
    _, rows = _load_jsonl_suite(FIXTURE)
    result = reconcile_assertions(rows[0]["spans"])
    assertion = result.assertions_by_entity["entity_id:diabetes"]

    assert result.disclaimer == ASSERTION_GRAPH_ADVISORY
    assert "not a clinical decision" in result.disclaimer
    assert set(assertion.provenance_by_axis) == set(ASSERTION_GRAPH_AXES)
    assert assertion.provenance_by_axis["temporality"].section == "Assessment"
    assert assertion.provenance_by_axis["negation"].span_id == "ag1-s2"
    assert "assessment" in SECTION_RECONCILIATION_PRECEDENCE


def _load_jsonl_suite(path: Path) -> tuple[dict, list[dict]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = rows[0]
    assert meta["kind"] == "meta"
    return meta, rows[1:]


def _find_conflict(result: AssertionGraphResult, entity_key: str, axis: str):
    for conflict in result.conflicts:
        if conflict.entity_key == entity_key and conflict.axis == axis:
            return conflict
    raise AssertionError(f"missing conflict for {entity_key} {axis}")


def _result_signature(result: AssertionGraphResult) -> tuple:
    return (
        tuple(
            (
                assertion.entity_key,
                assertion.assertion.to_dict(),
                assertion.clinical_status,
                assertion.conflicted_axes,
                tuple(
                    (item.axis, item.value, item.section, item.span_id)
                    for item in assertion.provenance
                ),
            )
            for assertion in result.assertions
        ),
        tuple(
            (
                conflict.entity_key,
                conflict.axis,
                conflict.values,
                conflict.sections,
            )
            for conflict in result.conflicts
        ),
    )
