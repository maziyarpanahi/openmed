"""Tests for DirectID model and safety-sweep evidence attribution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openmed.core.safety_sweep import (
    SAFETY_SWEEP_PATTERNS_VERSION,
    SAFETY_SWEEP_SOURCE,
)
from openmed.eval.directid import (
    DIRECTID_MODEL_SOURCE,
    DirectIDEvidenceError,
    build_directid_evidence,
)
from openmed.training.directid import (
    DIRECTID_TINY_HEAD_CONTRACT,
    gate_requirements_by_code,
)

_FIXTURE_PATH = Path("tests/fixtures/eval/directid_safety_sweep.json")


def _fixture(case_id: str) -> dict[str, Any]:
    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert payload["synthetic"] is True
    return next(case for case in payload["cases"] if case["id"] == case_id)


def _offset_spans(case: dict[str, Any], key: str) -> list[dict[str, Any]]:
    text = case["text"]
    spans: list[dict[str, Any]] = []
    for item in case[key]:
        surface = item["surface"]
        start = text.index(surface)
        spans.append(
            {
                "label": item["label"],
                "start": start,
                "end": start + len(surface),
                "confidence": item.get("confidence", 1.0),
            }
        )
    return spans


def _report(case_id: str = "model-misses-recovered-by-sweep"):
    case = _fixture(case_id)
    return case, build_directid_evidence(
        case["text"],
        _offset_spans(case, "gold_spans"),
        _offset_spans(case, "model_spans"),
    )


def test_safety_sweep_recall_is_attributed_without_hiding_model_misses() -> None:
    _case, report = _report()

    assert report.model.per_label_recall == {
        "CREDIT_CARD": 0.0,
        "EMAIL": 0.0,
        "SSN": 1.0,
    }
    assert report.combined.per_label_recall == {
        "CREDIT_CARD": 1.0,
        "EMAIL": 1.0,
        "SSN": 1.0,
    }
    assert report.model.structured_id_recall == 0.5
    assert report.combined.structured_id_recall == 1.0
    assert report.structured_id_recall_gain == 0.5
    assert report.safety_sweep_recovered_count == 2
    assert report.safety_sweep_structured_recovered_count == 1
    assert report.recovered_per_label == {"CREDIT_CARD": 1, "EMAIL": 1}
    assert len(report.model_misses) == 2
    assert all(miss.recovered_by_safety_sweep for miss in report.model_misses)
    assert report.critical_leakage_count == 0
    assert report.residual_leakage_rate == 0.0


def test_report_uses_canonical_labels_and_raw_text_free_provenance() -> None:
    case, report = _report()
    payload = report.to_dict()
    serialized = json.dumps(payload, sort_keys=True)
    provenance_fields = set(DIRECTID_TINY_HEAD_CONTRACT.safety_sweep_provenance_fields)

    assert {span.label for span in report.safety_sweep_spans} == {
        "CREDIT_CARD",
        "EMAIL",
    }
    for span in (*report.model_spans, *report.safety_sweep_spans):
        assert set(span.provenance.to_dict()) == provenance_fields
        assert span.provenance.text_hash.startswith("sha256:")
        assert span.risk.risk_level == "high"
        assert span.risk.critical is True

    assert report.model_spans[0].provenance.source == DIRECTID_MODEL_SOURCE
    assert report.model_spans[0].provenance.patterns_version is None
    assert all(
        span.provenance.source == SAFETY_SWEEP_SOURCE
        and span.provenance.patterns_version == SAFETY_SWEEP_PATTERNS_VERSION
        for span in report.safety_sweep_spans
    )
    for span in (*case["gold_spans"], *case["model_spans"]):
        assert span["surface"] not in serialized
    assert "text" not in payload


def test_combined_recall_populates_g1b_and_g3_benchmark_evidence() -> None:
    _case, report = _report()
    requirements = gate_requirements_by_code()
    gate_evidence = report.gate_evidence
    benchmark = report.to_benchmark_report(device="pytorch")

    assert set(requirements["G1b"].required_fields) <= set(gate_evidence["G1b"])
    assert set(requirements["G3"].required_fields) <= set(gate_evidence["G3"])
    assert benchmark.metrics["per_label_recall"]["CREDIT_CARD"] == 1.0
    assert benchmark.metrics["model_per_label_recall"]["CREDIT_CARD"] == 0.0
    assert benchmark.metrics["structured_id_recall"] == 1.0
    assert benchmark.metrics["model_structured_id_recall"] == 0.5
    assert benchmark.metadata["family"] == "DirectID"
    assert benchmark.metadata["eval_set_hash"] == report.eval_set_hash


def test_model_overlap_wins_without_duplicate_sweep_span() -> None:
    _case, report = _report("model-overlap-wins-without-duplicate")

    assert report.safety_sweep_spans == ()
    assert len(report.combined_spans) == 1
    assert report.combined_spans[0].label == "CREDIT_CARD"
    assert report.combined_spans[0].provenance.source == DIRECTID_MODEL_SOURCE
    assert report.span_integrity == {
        "passed": True,
        "input_model_overlaps": 0,
        "model_overlaps_resolved": 0,
        "combined_residual_overlaps": 0,
    }


def test_invalid_or_unsupported_spans_fail_without_echoing_surface() -> None:
    text = "Synthetic account marker SAMPLE-600."

    with pytest.raises(DirectIDEvidenceError, match="invalid offsets") as error:
        build_directid_evidence(
            text,
            [{"label": "ACCOUNT_NUMBER", "start": -1, "end": 10}],
            [],
        )

    assert "SAMPLE-600" not in str(error.value)
