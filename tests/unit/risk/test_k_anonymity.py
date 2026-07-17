"""Tests for targeted k-anonymity analysis and suppression (issue #882)."""

from __future__ import annotations

import json

import pytest

from openmed.risk import (
    KAnonymityEngine,
    analyze_k_anonymity,
    apply_suppression,
    propose_suppression,
)


def _synthetic_rows() -> list[dict[str, object]]:
    return [
        {"age": 30, "zip": "10001", "diagnosis": "flu"},
        {"age": 30, "zip": "10001", "diagnosis": "cold"},
        {"age": 40, "zip": "20001", "diagnosis": "flu"},
        {"age": 40, "zip": "20001", "diagnosis": "cold"},
        {"age": 40, "zip": "20001", "diagnosis": "asthma"},
        {"age": 91, "zip": "99999", "diagnosis": "rare"},
    ]


def test_reports_equivalence_classes_achieved_k_and_violating_rows() -> None:
    report = analyze_k_anonymity(
        _synthetic_rows(),
        quasi_identifiers=["age", "zip"],
        target_k=2,
    )

    assert sorted(item.size for item in report.equivalence_classes) == [1, 2, 3]
    assert report.achieved_k == 1
    assert report.smallest_class_size == 1
    assert report.violating_rows == (5,)
    assert report.meets_target is False


def test_minimal_suppression_reaches_target_and_preserves_retained_rows() -> None:
    rows = _synthetic_rows()
    engine = KAnonymityEngine(["age", "zip"], target_k=2)

    proposal = engine.propose_suppression(rows)
    retained = proposal.apply(rows)
    after = engine.analyze(retained)

    assert proposal.row_indices == (5,)
    assert proposal.suppressed_count == 1
    assert proposal.retained_count == 5
    assert proposal.achieved_k_after_suppression == 2
    assert proposal.feasible is True
    assert retained == rows[:5]
    assert after.meets_target is True
    assert after.achieved_k == 2


def test_all_rows_in_every_undersized_class_are_minimally_suppressed() -> None:
    rows = [
        {"age": 30, "zip": "10001"},
        {"age": 30, "zip": "10001"},
        {"age": 30, "zip": "10001"},
        {"age": 40, "zip": "20001"},
        {"age": 40, "zip": "20001"},
        {"age": 50, "zip": "30001"},
    ]

    proposal = propose_suppression(
        rows,
        quasi_identifiers=["age", "zip"],
        target_k=3,
    )
    retained = apply_suppression(rows, proposal)

    assert proposal.row_indices == (3, 4, 5)
    assert analyze_k_anonymity(
        retained,
        quasi_identifiers=["age", "zip"],
        target_k=3,
    ).meets_target


def test_report_is_deterministic_json_safe_and_does_not_emit_raw_qi_values() -> None:
    engine = KAnonymityEngine(["zip", "age"], target_k=2)

    first = engine.analyze(_synthetic_rows()).to_dict()
    second = engine.analyze(_synthetic_rows()).to_dict()
    serialized = json.dumps(first, sort_keys=True)

    assert first == second
    assert json.loads(serialized) == first
    assert "10001" not in serialized
    assert "99999" not in serialized
    assert all(
        item["class_hash"].startswith("sha256:")
        for item in first["equivalence_classes"]
    )


def test_infeasible_suppression_does_not_claim_an_empty_table_meets_target() -> None:
    rows = [{"age": 30}, {"age": 40}]
    proposal = propose_suppression(rows, ["age"], target_k=2)

    assert proposal.row_indices == (0, 1)
    assert proposal.retained_count == 0
    assert proposal.achieved_k_after_suppression == 0
    assert proposal.feasible is False
    with pytest.raises(ValueError, match="without removing every record"):
        proposal.apply(rows)


class _FrameLike:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def to_dict(self, orient: str) -> list[dict[str, object]]:
        assert orient == "records"
        return self.rows


def test_operates_in_process_on_dataframe_like_input() -> None:
    frame = _FrameLike(_synthetic_rows())

    report = KAnonymityEngine(["age", "zip"], target_k=2).analyze(frame)

    assert report.violating_rows == (5,)


@pytest.mark.parametrize("target_k", [0, -1, 1.5, True])
def test_rejects_invalid_target_k(target_k: object) -> None:
    with pytest.raises(ValueError, match="integer >= 1"):
        KAnonymityEngine(["age"], target_k=target_k)  # type: ignore[arg-type]


def test_rejects_missing_or_unknown_quasi_identifier_columns() -> None:
    with pytest.raises(ValueError, match="At least one"):
        KAnonymityEngine([])
    with pytest.raises(ValueError, match="Unknown quasi-identifier"):
        KAnonymityEngine(["missing"]).analyze(_synthetic_rows())


def test_public_api_is_exported() -> None:
    import openmed.risk as risk

    expected = {
        "KAnonymityEngine",
        "analyze_k_anonymity",
        "apply_suppression",
        "propose_suppression",
    }
    assert expected <= set(risk.__all__)
    assert all(hasattr(risk, name) for name in expected)
