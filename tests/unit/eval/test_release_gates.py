from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.compliance import (
    CompositionEvidence,
    ReleaseAssumptions,
    build_release_expert_review_evidence,
)
from openmed.core.audit import stable_hash
from openmed.eval import release_gates
from openmed.eval.metrics import (
    compute_metrics_bundle,
    compute_span_grounded_faithfulness,
)
from openmed.eval.release_gates import (
    QUARANTINED,
    RELEASABLE,
    GateReport,
    ReleaseGate,
)
from openmed.eval.report import BenchmarkReport
from openmed.eval.surrogate_quality import load_surrogate_quality_records
from openmed.risk import (
    AnonymityPolicy,
    anonymize_release,
    longitudinal_risk_report,
    validate_released_output,
)

SIGNING_KEY = "unit-release-key"


def _calibration_files(tmp_path: Path) -> tuple[Path, Path]:
    thresholds = tmp_path / "thresholds.json"
    calibration = tmp_path / "calibration_report.json"
    thresholds.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "openmed.calibration.thresholds",
                "thresholds": {"unit-model": {"PERSON": {"en": 0.9}}},
            }
        ),
        encoding="utf-8",
    )
    calibration.write_text(
        json.dumps({"schema_version": 1, "groups": []}),
        encoding="utf-8",
    )
    return thresholds, calibration


def _report(
    tmp_path: Path,
    *,
    metadata_updates: dict[str, object] | None = None,
    metric_updates: dict[str, object] | None = None,
) -> BenchmarkReport:
    thresholds, calibration = _calibration_files(tmp_path)
    metadata: dict[str, object] = {
        "repo_id": "OpenMed/unit-model",
        "family": "PII",
        "tier": "Tiny",
        "param_count": 44_000_000,
        "format": "mlx-fp",
        "eval_set_hash": "sha256:eval",
        "leakage_fixture_hash": "sha256:leakage",
        "policy": "hipaa_safe_harbor",
        "thresholds_path": str(thresholds),
        "calibration_report_path": str(calibration),
        "span_fixtures": [
            {
                "text": "Patient John on 2026-01-02 has ID 123.",
                "predicted_spans": [
                    {"start": 8, "end": 12, "label": "PERSON"},
                    {"start": 16, "end": 26, "label": "DATE"},
                    {"start": 34, "end": 37, "label": "ID_NUM"},
                ],
            }
        ],
    }
    if metadata_updates:
        metadata.update(metadata_updates)

    metrics: dict[str, object] = {
        "per_label_recall": {
            "PERSON": 0.990,
            "DATE": 0.990,
            "ID_NUM": 0.990,
            "API_KEY": 0.995,
        },
        "per_label_precision": {
            "PERSON": 0.98,
            "DATE": 0.98,
            "ID_NUM": 0.98,
            "API_KEY": 0.99,
        },
        "critical_leakage_count": 0,
        "leakage": {
            "overall": 0.0,
            "leaked_chars_by_label": {},
            "total_chars_by_label": {
                "PERSON": 4,
                "DATE": 10,
                "ID_NUM": 3,
                "API_KEY": 8,
            },
        },
        "quant_recall_delta": 0.0,
        "latency": {"p50_ms": 50.0, "p95_ms": 120.0},
        "resources": {"peak_rss_mib": 128.0},
    }
    if metric_updates:
        metrics.update(metric_updates)

    return BenchmarkReport(
        suite="golden",
        model_name="unit-model",
        device="cpu",
        fixture_count=1,
        generated_at="2026-06-15T00:00:00+00:00",
        metrics=metrics,
        metadata=metadata,
    )


def _baseline() -> dict[str, object]:
    return {
        "key": "pii::tiny::mlx-fp",
        "metrics": {
            "per_label_recall": {
                "PERSON": 0.990,
                "DATE": 0.990,
                "ID_NUM": 0.990,
                "API_KEY": 0.995,
            },
            "residual_leakage_rate": 0.0,
        },
    }


def _gate() -> ReleaseGate:
    return ReleaseGate(signing_key=SIGNING_KEY)


def _conformal_report(*, coverage: float = 0.95) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "openmed.calibration.under_shift",
        "alpha": 0.05,
        "target_coverage": 0.95,
        "coverage_tolerance": 0.01,
        "groups": [
            {
                "model_id": "unit-model",
                "label": "SSN",
                "language": "en",
                "target_coverage": 0.95,
                "positive_coverage": coverage,
                "realized_coverage": coverage,
                "positive_gate_weight": 100.0,
                "total_gate_weight": 100.0,
            }
        ],
        "language_coverage": {
            "en": {
                "slice_key": "en",
                "target_coverage": 0.95,
                "realized_coverage": coverage,
                "coverage_gap": max(0.95 - coverage, 0.0),
                "covered_weight": coverage * 100.0,
                "total_weight": 100.0,
            }
        },
    }


def _grounding_report(*, accuracy: float = 0.90, coverage: float = 0.72):
    return {
        "schema_version": 1,
        "artifact_type": "openmed.grounding.calibration.report",
        "minimum_accuracy": 0.85,
        "minimum_coverage": 0.70,
        "vocabularies": {
            "RXNORM": {
                "operating_point": {
                    "system": "RXNORM",
                    "threshold": 0.80,
                    "accuracy": accuracy,
                    "coverage": coverage,
                    "accepted_count": int(coverage * 100),
                    "total_count": 100,
                    "passed": accuracy >= 0.85 and coverage >= 0.70,
                },
                "reliability_diagram": [],
                "coverage_accuracy_curve": [],
            }
        },
    }


def _check(report, gate_name: str):
    return next(check for check in report.gate_results if check.gate == gate_name)


def _structured_release_evidence(
    *,
    composition: CompositionEvidence | None = None,
):
    rows = [
        {
            "patient_id": f"patient-{index}",
            "patient_name": f"Canary Name {index}",
            "age": 30 if index < 2 else 40,
            "postal_code": "10001" if index < 2 else "20001",
            "condition": "a" if index % 2 == 0 else "b",
        }
        for index in range(4)
    ]
    result = anonymize_release(
        rows,
        AnonymityPolicy(
            quasi_identifiers=("age", "postal_code"),
            sensitive_attributes=("condition",),
            direct_identifiers=("patient_name",),
            privacy_unit="patient_id",
            target_k=2,
            target_l=2,
        ),
    )
    return build_release_expert_review_evidence(
        result,
        validation=validate_released_output(result.records, result),
        assumptions=ReleaseAssumptions(
            privacy_unit="patient",
            population_scope="release_cohort",
            release_model="restricted",
            recipient_model="named_researchers",
            auxiliary_data_model="reasonably_available",
            notes_digest=stable_hash(
                {
                    "kind": "structured-release-gate-test",
                    "review": "stored outside shareable evidence",
                }
            ),
        ),
        composition=composition,
    )


def _synthetic_longitudinal_records(
    *,
    diversified_surrogates: bool,
) -> list[dict[str, object]]:
    surrogate_values = (
        ("synthetic-surrogate-a", "synthetic-surrogate-b")
        if diversified_surrogates
        else ("synthetic-surrogate-a", "synthetic-surrogate-a")
    )
    return [
        {
            "patient_id": "synthetic-subject-001",
            "record_id": f"synthetic-note-{index}",
            "text": "Synthetic follow-up note.",
            "audit_spans": [
                {
                    "canonical_label": "SYNTHETIC_SURROGATE",
                    "surrogate": surrogate,
                    "start": 0,
                    "end": 9,
                }
            ],
        }
        for index, surrogate in enumerate(surrogate_values, start=1)
    ]


def _relation_metric(*, strict_lower: float, relaxed_lower: float) -> dict[str, object]:
    strict = {
        "confidence_interval": {
            "lower": strict_lower,
            "point": max(strict_lower, release_gates.G9_STRICT_RE_F1_FLOOR),
            "upper": 1.0,
        },
        "f1": max(strict_lower, release_gates.G9_STRICT_RE_F1_FLOOR),
        "false_negatives": 0,
        "false_positives": 0,
        "precision": 1.0,
        "recall": max(strict_lower, release_gates.G9_STRICT_RE_F1_FLOOR),
        "true_positives": 10,
    }
    relaxed = {
        "confidence_interval": {
            "lower": relaxed_lower,
            "point": max(relaxed_lower, release_gates.G9_RELAXED_RE_F1_FLOOR),
            "upper": 1.0,
        },
        "f1": max(relaxed_lower, release_gates.G9_RELAXED_RE_F1_FLOOR),
        "false_negatives": 0,
        "false_positives": 0,
        "precision": 1.0,
        "recall": max(relaxed_lower, release_gates.G9_RELAXED_RE_F1_FLOOR),
        "true_positives": 10,
    }
    return {
        "relation_extraction": {
            "gold_relation_count": 10,
            "per_relation_type": {
                "INHIBITOR": {
                    "relaxed": relaxed,
                    "strict": strict,
                }
            },
            "predicted_relation_count": 10,
            "relaxed": relaxed,
            "strict": strict,
        }
    }


def test_release_gate_passes_and_emits_signed_section_64_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[str] = []
    original_load_policy = release_gates.policy_module.load_policy

    def spy_load_policy(name: str):
        calls.append(name)
        return original_load_policy(name)

    monkeypatch.setattr(release_gates.policy_module, "load_policy", spy_load_policy)

    result = _gate().evaluate(_report(tmp_path), _baseline())

    assert result.decision == RELEASABLE
    assert result.verify(SIGNING_KEY)
    assert calls == ["hipaa_safe_harbor"]
    assert {
        "repo_id",
        "family",
        "tier",
        "param_count",
        "format",
        "per_label_recall",
        "per_label_precision",
        "critical_leakage_count",
        "residual_leakage_rate",
        "quant_recall_delta",
        "p50_ms",
        "p95_ms",
        "ram_mb",
        "eval_set_hash",
        "leakage_fixture_hash",
        "decision",
    }.issubset(result.to_dict())

    restored = GateReport.from_json(result.to_json())
    assert restored.verify(SIGNING_KEY)
    assert restored.to_json() == result.to_json()


def test_cross_script_gate_blocks_telugu_drop_while_latin_stays_green(
    tmp_path: Path,
) -> None:
    fixture_path = (
        Path(__file__).parents[2] / "fixtures" / "eval" / "non_latin_script_phi.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    bundle = compute_metrics_bundle(
        fixture["gold_spans"],
        fixture["predicted_spans_telugu_drop"],
        source_text=fixture["text"],
    )
    candidate = _report(
        tmp_path,
        metric_updates={
            "leakage": bundle["leakage"],
            "recall_slices": bundle["recall_slices"],
        },
    )

    result = _gate().evaluate(candidate, _baseline())
    check = _check(result, release_gates.CROSS_SCRIPT_GATE)

    assert result.decision == QUARANTINED
    assert check.passed is False
    assert check.details["per_script_recall"]["Latin"] == 1.0
    assert check.details["per_script_recall"]["Telugu"] == 0.0
    assert check.details["recall_floors"]["Telugu"] >= 0.99
    assert "Telugu recall" in check.reason
    assert "Telugu leakage" in check.reason


def test_cross_script_gate_passes_when_all_fixture_scripts_are_covered(
    tmp_path: Path,
) -> None:
    fixture_path = (
        Path(__file__).parents[2] / "fixtures" / "eval" / "non_latin_script_phi.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    bundle = compute_metrics_bundle(
        fixture["gold_spans"],
        fixture["predicted_spans_all_covered"],
        source_text=fixture["text"],
    )
    candidate = _report(
        tmp_path,
        metric_updates={
            "leakage": bundle["leakage"],
            "recall_slices": bundle["recall_slices"],
        },
    )

    result = _gate().evaluate(candidate, _baseline())
    check = _check(result, release_gates.CROSS_SCRIPT_GATE)

    assert result.decision == RELEASABLE
    assert check.passed is True
    assert check.details["applicable_scripts"] == (
        "Devanagari",
        "Han",
        "Telugu",
    )


def test_surrogate_quality_gate_requires_evidence_when_applicable(
    tmp_path: Path,
) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={"surrogate_quality_required": True},
        ),
        _baseline(),
    )

    check = _check(result, release_gates.SURROGATE_QUALITY_GATE)
    assert result.decision == QUARANTINED
    assert check.passed is False
    assert check.reason == "surrogate-quality evidence is required"


def test_surrogate_quality_gate_quarantines_bad_release_evidence(
    tmp_path: Path,
) -> None:
    records: list[object] = list(load_surrogate_quality_records())
    records.append(
        {
            "record_id": "sq-zh-release-regression",
            "language": "zh",
            "locale": "zh_CN",
            "surrogates": {
                "name": "John Doe",
                "date_of_birth": "04/12/1990",
                "national_id": "110105199004123416",
            },
            "expected": {
                "birth_date": "1990-04-12",
                "gender": "female",
                "region_code": "110105",
            },
            "metadata": {
                "synthetic": True,
                "contains_real_phi": False,
                "synthetic_source": "release_gate_regression",
            },
        }
    )

    result = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={"surrogate_quality_required": True},
            metric_updates={"surrogate_quality": {"records": records}},
        ),
        _baseline(),
    )

    check = _check(result, release_gates.SURROGATE_QUALITY_GATE)
    assert result.decision == QUARANTINED
    assert result.verify(SIGNING_KEY)
    assert check.passed is False
    assert check.details["failing_locales"] == {"zh": 0.5}


def test_g9_relation_gate_fails_when_strict_lower_ci_below_floor(
    tmp_path: Path,
) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={"task": "relation"},
            metric_updates=_relation_metric(
                strict_lower=release_gates.G9_STRICT_RE_F1_FLOOR - 0.001,
                relaxed_lower=release_gates.G9_RELAXED_RE_F1_FLOOR,
            ),
        ),
        _baseline(),
    )

    check = _check(result, "G9")
    assert result.decision == QUARANTINED
    assert check.passed is False
    assert check.details["strict_floor"] == release_gates.G9_STRICT_RE_F1_FLOOR
    assert "strict_relation_f1" in check.details["violations"]


def test_g9_relation_gate_passes_at_configured_lower_ci_floor(
    tmp_path: Path,
) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={"task": "relation"},
            metric_updates=_relation_metric(
                strict_lower=release_gates.G9_STRICT_RE_F1_FLOOR,
                relaxed_lower=release_gates.G9_RELAXED_RE_F1_FLOOR,
            ),
        ),
        _baseline(),
    )

    check = _check(result, "G9")
    assert result.decision == RELEASABLE
    assert check.passed is True
    assert check.details["per_relation_type"]["INHIBITOR"]["strict_f1"] == (
        release_gates.G9_STRICT_RE_F1_FLOOR
    )


def test_gate_report_from_json_rejects_malformed_payload() -> None:
    with pytest.raises(ValueError, match="Invalid JSON for GateReport"):
        GateReport.from_json("{")


def test_find_open_issue_returns_none_for_malformed_gh_json(monkeypatch) -> None:
    class Result:
        stdout = "{"

    monkeypatch.setattr(
        release_gates.subprocess,
        "run",
        lambda *args, **kwargs: Result(),
    )

    assert (
        release_gates._find_open_issue(repo="owner/repo", title="Gate failure") is None
    )


@pytest.mark.parametrize(
    ("gate_name", "metric_updates", "metadata_updates"),
    [
        (
            "G1a",
            {"per_label_recall": {"PERSON": 0.989, "API_KEY": 0.995}},
            None,
        ),
        (
            "G1b",
            {"per_label_recall": {"PERSON": 0.990, "DATE": 0.990, "API_KEY": 0.994}},
            None,
        ),
        (
            "G2",
            {
                "per_label_recall": {
                    "PERSON": 0.990,
                    "DATE": 0.979,
                    "API_KEY": 0.995,
                }
            },
            None,
        ),
        ("G3", {"critical_leakage_count": 1}, None),
        ("G5", {"latency": {"p50_ms": 50.0, "p95_ms": 151.0}}, None),
        ("G6", {"latency": {"p50_ms": 50.0}}, None),
        (
            "G8",
            None,
            {
                "span_fixtures": [
                    {
                        "text": "Patient John",
                        "predicted_spans": [
                            {"start": 8, "end": 99, "label": "PERSON"},
                        ],
                    }
                ]
            },
        ),
    ],
)
def test_release_gate_blocks_failed_gate_boundaries(
    tmp_path: Path,
    gate_name: str,
    metric_updates: dict[str, object] | None,
    metadata_updates: dict[str, object] | None,
) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates=metric_updates,
            metadata_updates=metadata_updates,
        ),
        _baseline(),
    )

    assert result.decision == QUARANTINED
    assert _check(result, gate_name).passed is False


def test_critical_leakage_forces_non_releasable(tmp_path: Path) -> None:
    result = _gate().evaluate(
        _report(tmp_path, metric_updates={"critical_leakage_count": 2}),
        _baseline(),
    )

    assert result.decision == QUARANTINED
    assert _check(result, "G3").reason == "critical leakage must be exactly zero"


def test_g10_faithfulness_gate_passes_grounded_outputs(tmp_path: Path) -> None:
    text = "Patient has hypertension."
    start = text.index("hypertension")
    faithfulness = compute_span_grounded_faithfulness(
        [
            {
                "fact_type": "diagnosis",
                "value": "hypertension",
                "supporting_span": {
                    "start": start,
                    "end": start + len("hypertension"),
                },
            }
        ],
        source_text=text,
    )
    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={"faithfulness": faithfulness.to_dict()},
        ),
        _baseline(),
    )

    check = _check(result, "G10")
    assert faithfulness.ungrounded_fact_rate == 0.0
    assert result.decision == RELEASABLE
    assert check.passed is True
    assert check.details["ungrounded_fact_rate"] == pytest.approx(0.0)


def test_g10_faithfulness_gate_quarantines_fabricated_facts(
    tmp_path: Path,
) -> None:
    text = "Patient has hypertension."
    start = text.index("hypertension")
    faithfulness = compute_span_grounded_faithfulness(
        [
            {
                "fact_type": "diagnosis",
                "value": "pneumonia",
                "supporting_span": {
                    "start": start,
                    "end": start + len("hypertension"),
                },
            }
        ],
        source_text=text,
    )
    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={"faithfulness": faithfulness.to_dict()},
        ),
        _baseline(),
    )

    check = _check(result, "G10")
    assert faithfulness.ungrounded_fact_rate > 0.0
    assert result.decision == QUARANTINED
    assert check.passed is False
    assert check.reason == "ungrounded-fact rate exceeds hard ceiling"
    assert check.details["violations"]["ungrounded_fact_rate"]["observed"] == 1.0


@pytest.mark.parametrize("rate", [-0.01, 1.01])
def test_g10_faithfulness_gate_rejects_invalid_rates(
    tmp_path: Path,
    rate: float,
) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={"faithfulness": {"ungrounded_fact_rate": rate}},
        ),
        _baseline(),
    )

    check = _check(result, "G10")
    assert result.decision == QUARANTINED
    assert check.passed is False
    assert check.reason == "ungrounded-fact rate must be between zero and one"


def test_extraction_reemission_critical_identifier_forces_quarantine(
    tmp_path: Path,
) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={
                "critical_leakage_count": 0,
                "extraction_reemission_leakage": {
                    "overall": 1.0,
                    "leaked_chars_by_label": {"SSN": 11},
                    "total_chars_by_label": {"SSN": 11},
                },
            },
        ),
        _baseline(),
    )

    assert result.decision == QUARANTINED
    assert result.critical_leakage_count == 11
    assert _check(result, "G3").passed is False


def test_extraction_reemission_blocks_high_f1_at_zero_leakage_target(
    tmp_path: Path,
) -> None:
    gate = ReleaseGate(
        signing_key=SIGNING_KEY,
        model_steward_config={"default_target_leakage": 0.0},
    )
    result = gate.evaluate(
        _report(
            tmp_path,
            metric_updates={
                "extraction_reemission_leakage": {
                    "overall": 0.01,
                    "leaked_chars_by_label": {"PERSON": 4},
                    "total_chars_by_label": {"PERSON": 4},
                },
                "exact_span_f1": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
            },
        ),
        _baseline(),
    )

    g7 = _check(result, "G7")
    assert result.decision == QUARANTINED
    assert _check(result, "G3").passed is True
    assert g7.passed is False
    assert "target_leakage" in g7.details["violations"]


def test_g11_quarantines_single_missed_drug_allergy(tmp_path: Path) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={
                "critical_finding_recall": {
                    "overall": 2 / 3,
                    "by_category": {
                        "critical_diagnosis": 1.0,
                        "drug_allergy": 0.0,
                        "critical_result": 1.0,
                    },
                    "covered": 2,
                    "total": 3,
                    "missed_findings": [
                        {
                            "category": "drug_allergy",
                            "fixture_id": "golden-critical-findings-synthetic-en",
                            "start": 71,
                            "end": 81,
                            "label": "MEDICATION",
                        }
                    ],
                }
            },
        ),
        _baseline(),
    )

    check = _check(result, "G11")
    assert result.decision == QUARANTINED
    assert check.passed is False
    assert check.details["floor"] == release_gates.G11_CRITICAL_RECALL_FLOOR
    assert check.details["missed_findings"] == [
        {
            "category": "drug_allergy",
            "fixture_id": "golden-critical-findings-synthetic-en",
            "start": 71,
            "end": 81,
            "label": "MEDICATION",
        }
    ]
    assert check.details["violations"]["must_not_miss_findings"][0]["fixture_id"] == (
        "golden-critical-findings-synthetic-en"
    )


def test_g14_passes_when_extraction_fairness_metric_absent(tmp_path: Path) -> None:
    result = _gate().preview(_report(tmp_path), _baseline())

    check = _check(result, "G14")
    assert check.passed is True
    assert check.reason == "not provided"
    assert check.details["ceiling"] == release_gates.G14_EXTRACTION_DISPARITY_CEILING


def test_g14_passes_on_balanced_extraction_corpus(tmp_path: Path) -> None:
    result = _gate().preview(
        _report(
            tmp_path,
            metric_updates={
                "extraction_fairness": {
                    "extraction_f1_gap": 0.01,
                    "worst_group": "site=site_beta",
                    "best_group": "site=site_alpha",
                    "per_group": {
                        "site=site_alpha": {"entity_f1": 0.99},
                        "site=site_beta": {"entity_f1": 0.98},
                    },
                }
            },
        ),
        _baseline(),
    )

    check = _check(result, "G14")
    assert check.passed is True
    assert check.details["extraction_f1_gap"] == pytest.approx(0.01)


def test_g14_quarantines_when_injected_group_exceeds_ceiling(tmp_path: Path) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={
                "extraction_fairness": {
                    "extraction_f1_gap": 0.42,
                    "recall_gap": 0.5,
                    "critical_finding_recall_gap": 0.6,
                    "worst_group": "demographic_group=surrogate_b",
                    "best_group": "demographic_group=surrogate_a",
                    "worst_group_by_metric": {
                        "entity_f1": "demographic_group=surrogate_b",
                        "recall": "demographic_group=surrogate_b",
                        "critical_finding_recall": "demographic_group=surrogate_b",
                    },
                    "per_group": {
                        "demographic_group=surrogate_a": {"entity_f1": 0.95},
                        "demographic_group=surrogate_b": {"entity_f1": 0.53},
                    },
                    "assistive": True,
                }
            },
        ),
        _baseline(),
    )

    check = _check(result, "G14")
    assert result.decision == QUARANTINED
    assert check.passed is False
    assert check.details["ceiling"] == release_gates.G14_EXTRACTION_DISPARITY_CEILING
    assert check.details["extraction_f1_gap"] == pytest.approx(0.42)
    assert check.details["worst_group"] == "demographic_group=surrogate_b"
    assert check.details["worst_group_by_metric"]["recall"] == (
        "demographic_group=surrogate_b"
    )


def test_g14_computes_gap_from_per_group_when_gap_missing(tmp_path: Path) -> None:
    result = _gate().preview(
        _report(
            tmp_path,
            metric_updates={
                "extraction_fairness": {
                    "per_group": {
                        "note_type=discharge_summary": {"entity_f1": 0.9},
                        "note_type=progress_note": {"entity_f1": 0.2},
                    }
                }
            },
        ),
        _baseline(),
    )

    check = _check(result, "G14")
    assert check.passed is False
    assert check.details["extraction_f1_gap"] == pytest.approx(0.7)
    assert check.details["worst_group"] == "note_type=progress_note"


@pytest.mark.parametrize("reported_gap", (-0.1, 0.0, 1.1))
def test_g14_rejects_invalid_or_inconsistent_reported_gap(
    tmp_path: Path,
    reported_gap: float,
) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={
                "extraction_fairness": {
                    "extraction_f1_gap": reported_gap,
                    "per_group": {
                        "site=site_alpha": {"entity_f1": 0.95},
                        "site=site_beta": {"entity_f1": 0.45},
                    },
                }
            },
        ),
        _baseline(),
    )

    check = _check(result, "G14")
    assert result.decision == QUARANTINED
    assert check.passed is False
    assert check.reason == "extraction-fairness metric is malformed"
    assert check.details["computed_gap"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "per_group",
    (
        {"site=only": {"entity_f1": 0.9}},
        {
            "site=alpha": {"entity_f1": 0.9},
            "site=beta": {"entity_f1": -0.1},
        },
        {
            "site=alpha": {"entity_f1": 0.9},
            "site=beta": {"entity_f1": "unknown"},
        },
    ),
)
def test_g14_rejects_malformed_group_evidence(
    tmp_path: Path,
    per_group: dict[str, object],
) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={"extraction_fairness": {"per_group": per_group}},
        ),
        _baseline(),
    )

    check = _check(result, "G14")
    assert result.decision == QUARANTINED
    assert check.passed is False
    assert check.reason == "extraction-fairness metric is malformed"


def test_g14_quarantines_from_extraction_fairness_report(tmp_path: Path) -> None:
    from openmed.eval import (
        extraction_fairness_report,
        load_extraction_fairness_fixtures,
    )

    fixtures = load_extraction_fairness_fixtures()

    def runner(fixture, model_name, device):
        if fixture.metadata.get("demographic_group") == "surrogate_b":
            return []
        return [
            {
                "start": span.start,
                "end": span.end,
                "label": span.label,
                "text": span.text,
            }
            for span in fixture.gold_spans
        ]

    audit = extraction_fairness_report("extract-model", fixtures, runner=runner)

    result = _gate().evaluate(
        _report(tmp_path, metric_updates={"extraction_fairness": audit.gate_metric()}),
        _baseline(),
    )

    check = _check(result, "G14")
    assert result.decision == QUARANTINED
    assert check.passed is False
    assert check.details["worst_group"] == "demographic_group=surrogate_b"


def test_conformal_coverage_gate_quarantines_shifted_critical_labels(
    tmp_path: Path,
) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={
                "calibration_under_shift": _conformal_report(coverage=0.80)
            },
        ),
        _baseline(),
    )

    check = _check(result, "conformal_coverage")
    assert result.decision == QUARANTINED
    assert check.passed is False
    assert check.details["violations"]["SSN:en"]["coverage"] == pytest.approx(0.80)


def test_grounding_coverage_gate_accepts_documented_operating_point(
    tmp_path: Path,
) -> None:
    result = _gate().preview(
        _report(
            tmp_path,
            metadata_updates={"grounding_calibration": _grounding_report()},
        ),
        _baseline(),
    )

    check = _check(result, "grounding_coverage")
    assert check.passed is True
    assert check.details["vocabularies"]["RXNORM"]["coverage"] == pytest.approx(0.72)


def test_grounding_coverage_gate_reads_offline_report_artifact(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "grounding-calibration.json"
    report_path.write_text(json.dumps(_grounding_report()), encoding="utf-8")
    result = _gate().preview(
        _report(
            tmp_path,
            metadata_updates={
                "grounding_calibration_report_path": str(report_path),
                "require_grounding_coverage": True,
            },
        ),
        _baseline(),
    )

    check = _check(result, "grounding_coverage")
    assert check.passed is True
    assert check.details["explicit"] is True
    assert check.details["required"] is True


def test_grounding_coverage_gate_blocks_accuracy_below_coverage_target(
    tmp_path: Path,
) -> None:
    result = _gate().preview(
        _report(
            tmp_path,
            metadata_updates={
                "grounding_calibration": _grounding_report(
                    accuracy=0.84,
                    coverage=0.72,
                )
            },
        ),
        _baseline(),
    )

    check = _check(result, "grounding_coverage")
    assert check.passed is False
    assert "RXNORM" in check.details["violations"]


def test_required_grounding_coverage_gate_fails_closed_when_missing(
    tmp_path: Path,
) -> None:
    result = _gate().preview(
        _report(tmp_path, metadata_updates={"require_grounding_coverage": True}),
        _baseline(),
    )

    check = _check(result, "grounding_coverage")
    assert check.passed is False
    assert check.reason == "grounding calibration report is required"


def test_g4_blocks_only_the_offending_quantized_format(tmp_path: Path) -> None:
    int8 = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={"format": "mlx-8bit"},
            metric_updates={"quant_recall_delta": 0.006},
        ),
        _baseline(),
    )
    fp = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={"format": "mlx-fp"},
            metric_updates={"quant_recall_delta": 0.006},
        ),
        _baseline(),
    )

    assert int8.decision == QUARANTINED
    assert int8.blocked_formats == ("mlx-8bit",)
    assert _check(int8, "G4").passed is False
    assert fp.decision == RELEASABLE
    assert fp.blocked_formats == ()
    assert _check(fp, "G4").passed is True


def test_g5_uses_nano_certificate_for_nano_declared_artifacts(
    tmp_path: Path,
) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={"tier": "Nano", "param_count": 44_000_000},
            metric_updates={
                "latency": {"p50_ms": 20.0, "p95_ms": 50.0},
                "resources": {"peak_rss_mib": 128.0},
            },
        ),
        _baseline(),
    )

    check = _check(result, "G5")
    assert result.decision == QUARANTINED
    assert check.passed is False
    assert check.reason == "Nano sub-tier budget not certified"
    assert check.details["failing_dimension"] == "param_count"
    assert check.details["parent_tier"] == "Tiny"


def test_missing_calibration_artifacts_fail_closed(tmp_path: Path) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={
                "thresholds_path": str(tmp_path / "missing-thresholds.json"),
                "calibration_report_path": str(tmp_path / "missing-report.json"),
            },
        ),
        _baseline(),
    )

    assert result.decision == QUARANTINED
    assert _check(result, "calibration_present").passed is False


def test_g7_blocks_recall_regression_and_residual_leakage(tmp_path: Path) -> None:
    baseline = {
        "metrics": {
            "per_label_recall": {"PERSON": 0.995, "API_KEY": 0.995},
            "residual_leakage_rate": 0.0,
        }
    }
    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={
                "per_label_recall": {"PERSON": 0.992, "API_KEY": 0.995},
                "leakage": {"overall": 0.001, "total_chars_by_label": {"PERSON": 4}},
            },
        ),
        baseline,
    )

    assert result.decision == QUARANTINED
    check = _check(result, "G7")
    assert check.passed is False
    assert "recall_drop" in check.details["violations"]
    assert "residual_leakage_regression" in check.details["violations"]


def test_zero_shot_language_gate_quarantines_transfer_floor_breach(
    tmp_path: Path,
) -> None:
    transfer_matrix = {
        "schema_version": 1,
        "artifact_type": "openmed.cross_lingual_transfer_matrix",
        "languages": ["en", "fr"],
        "leakage_floors": {"en": 0.10, "fr": 0.10},
        "matrix": {
            "en": {
                "en": {
                    "source_language": "en",
                    "target_language": "en",
                    "leakage_rate": 0.0,
                    "leaked_chars": 0,
                    "total_chars": 100,
                    "zero_shot": False,
                },
                "fr": {
                    "source_language": "en",
                    "target_language": "fr",
                    "leakage_rate": 0.25,
                    "leaked_chars": 25,
                    "total_chars": 100,
                    "zero_shot": True,
                },
            },
            "fr": {
                "en": {
                    "source_language": "fr",
                    "target_language": "en",
                    "leakage_rate": 0.0,
                    "leaked_chars": 0,
                    "total_chars": 100,
                    "zero_shot": True,
                },
                "fr": {
                    "source_language": "fr",
                    "target_language": "fr",
                    "leakage_rate": 0.0,
                    "leaked_chars": 0,
                    "total_chars": 100,
                    "zero_shot": False,
                },
            },
        },
    }

    result = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={"cross_lingual_transfer": transfer_matrix},
        ),
        _baseline(),
    )

    check = _check(result, "G9_zero_shot_language_leakage")
    assert result.decision == QUARANTINED
    assert check.passed is False
    assert check.details["violations"] == [
        {
            "source_language": "en",
            "target_language": "fr",
            "leakage_rate": 0.25,
            "leakage_floor": 0.10,
            "excess": 0.15,
            "leaked_chars": 25,
            "total_chars": 100,
        }
    ]


def test_membership_leakage_gate_blocks_leaky_configuration(tmp_path: Path) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={
                "membership_leakage": {
                    "attacker_auc": 0.91,
                    "attacker_advantage": 0.35,
                    "advantage_ceiling": 0.05,
                    "feature_hash": "sha256:features",
                    "per_label": {
                        "PERSON": {
                            "attacker_advantage": 0.35,
                            "feature_hash": "sha256:person",
                        }
                    },
                }
            },
        ),
        _baseline(),
    )

    check = _check(result, "membership_leakage")
    assert result.decision == QUARANTINED
    assert result.verify(SIGNING_KEY)
    assert check.passed is False
    assert "overall_advantage" in check.details["violations"]
    assert "PERSON" in check.details["violations"]["per_label_advantage"]


def test_membership_leakage_gate_passes_defended_configuration(
    tmp_path: Path,
) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={
                "membership_leakage": {
                    "attacker_auc": 0.5,
                    "attacker_advantage": 0.0,
                    "advantage_ceiling": 0.05,
                    "feature_hash": "sha256:features",
                    "defense": {"enabled": True, "clip_min": 0.5, "clip_max": 0.5},
                    "per_label": {
                        "PERSON": {
                            "attacker_advantage": 0.0,
                            "feature_hash": "sha256:person",
                        }
                    },
                }
            },
        ),
        _baseline(),
    )

    check = _check(result, "membership_leakage")
    assert result.decision == RELEASABLE
    assert result.verify(SIGNING_KEY)
    assert check.passed is True
    assert check.details["defense"]["enabled"] is True


def test_g8_consumes_strict_quality_gate_output(tmp_path: Path, monkeypatch) -> None:
    calls = {"strict": 0}
    original = release_gates.quality_gates.validate_entity_spans_strict

    def strict(entities, text):
        calls["strict"] += 1
        return original(entities, text)

    monkeypatch.setattr(
        release_gates.quality_gates,
        "validate_entity_spans_strict",
        strict,
    )

    result = _gate().evaluate(_report(tmp_path), _baseline())

    assert result.decision == RELEASABLE
    assert calls == {"strict": 1}
    assert _check(result, "G8").details["spans_checked"] == 3


def test_cross_document_linkage_gate_signs_blocked_and_mitigated_verdicts(
    tmp_path: Path,
) -> None:
    overlinked = longitudinal_risk_report(
        _synthetic_longitudinal_records(diversified_surrogates=False),
        hmac_key="synthetic-release-gate-key",
    )
    mitigated = longitudinal_risk_report(
        _synthetic_longitudinal_records(diversified_surrogates=True),
        hmac_key="synthetic-release-gate-key",
    )

    blocked = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={"longitudinal_linkage_risk": overlinked},
        ),
        _baseline(),
    )
    releasable = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={"longitudinal_linkage_risk": mitigated},
        ),
        _baseline(),
    )

    blocked_check = _check(blocked, release_gates.CROSS_DOCUMENT_LINKAGE_GATE)
    releasable_check = _check(
        releasable,
        release_gates.CROSS_DOCUMENT_LINKAGE_GATE,
    )
    assert overlinked["linkage_success_upper_bound"] == pytest.approx(1.0)
    assert mitigated["linkage_success_upper_bound"] == pytest.approx(0.0)
    assert blocked.decision == QUARANTINED
    assert blocked.verify(SIGNING_KEY)
    assert GateReport.from_json(blocked.to_json()).verify(SIGNING_KEY)
    assert blocked_check.passed is False
    assert releasable.decision == RELEASABLE
    assert releasable.verify(SIGNING_KEY)
    assert releasable_check.passed is True

    serialized = blocked.to_json()
    assert "synthetic-subject-001" not in serialized
    assert "synthetic-note-1" not in serialized
    assert "synthetic-surrogate-a" not in serialized
    assert blocked_check.details["evidence_hash"].startswith("sha256:")
    assert blocked_check.details["high_risk_evidence"]
    assert all(
        {"start", "end"}.issubset(item)
        for item in blocked_check.details["high_risk_evidence"]
    )
    assert all(
        set(item)
        <= {
            "patient_hash",
            "note_index",
            "note_hash",
            "value_hash",
            "start",
            "end",
        }
        for item in blocked_check.details["high_risk_evidence"]
    )


def test_cross_document_linkage_gate_discards_raw_fields_from_signed_evidence() -> None:
    evidence = longitudinal_risk_report(
        _synthetic_longitudinal_records(diversified_surrogates=True),
        hmac_key="synthetic-release-gate-key",
    )
    evidence["patient_risks"][0]["raw_note"] = "synthetic-private-canary"

    check = release_gates.evaluate_cross_document_linkage_gate(evidence)

    assert check.passed is True
    assert check.details["evidence_valid"] is True
    assert check.details["evidence_hash"].startswith("sha256:")
    assert "synthetic-private-canary" not in json.dumps(check.to_dict())


def test_cross_document_linkage_gate_honors_configured_ceiling(
    tmp_path: Path,
) -> None:
    evidence = longitudinal_risk_report(
        _synthetic_longitudinal_records(diversified_surrogates=False),
        hmac_key="synthetic-release-gate-key",
    )
    gate = ReleaseGate(
        signing_key=SIGNING_KEY,
        cross_document_linkage_ceiling=1.0,
    )

    report = gate.evaluate(
        _report(
            tmp_path,
            metric_updates={"longitudinal_linkage_risk": evidence},
        ),
        _baseline(),
    )
    check = _check(report, release_gates.CROSS_DOCUMENT_LINKAGE_GATE)

    assert report.decision == RELEASABLE
    assert report.verify(SIGNING_KEY)
    assert check.passed is True
    assert check.details["linkage_ceiling"] == pytest.approx(1.0)


def test_k_floor_release_gate_passes_signed_enforcement_evidence(
    tmp_path: Path,
) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={
                "kanon_enforcement": {
                    "target_k": 2,
                    "kanon": {"k": 2},
                    "bounds": {
                        "max_reidentification_upper_bound": 0.5,
                        "numeric_self_check": {"passed": True},
                    },
                }
            },
        ),
        _baseline(),
    )

    assert result.decision == RELEASABLE
    assert result.verify(SIGNING_KEY)
    assert _check(result, "k_floor").passed is True


def test_k_floor_release_gate_fails_realized_k_below_target(
    tmp_path: Path,
) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={
                "kanon_enforcement": {
                    "target_k": 3,
                    "kanon": {"k": 2},
                    "bounds": {
                        "max_reidentification_upper_bound": 0.5,
                        "numeric_self_check": {"passed": False},
                    },
                }
            },
        ),
        _baseline(),
    )

    check = _check(result, "k_floor")
    assert result.decision == QUARANTINED
    assert result.verify(SIGNING_KEY)
    assert check.passed is False
    assert check.details["violations"]["measured_k"] == {"observed": 2, "target": 3}


def test_structured_release_risk_gate_verifies_aggregate_evidence(
    tmp_path: Path,
) -> None:
    evidence = _structured_release_evidence()

    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={
                "structured_release_risk_evidence": evidence.to_dict(),
            },
        ),
        _baseline(),
    )

    check = _check(result, "structured_release_risk")
    assert result.decision == RELEASABLE
    assert result.verify(SIGNING_KEY)
    assert check.passed is True
    assert check.details["integrity_verified"] is True
    assert check.details["search_complete"] is False
    assert check.details["search_optimality_proven"] is True
    assert check.details["qualified_expert_review_required"] is True
    serialized = json.dumps(check.to_dict(), sort_keys=True)
    assert "patient-0" not in serialized
    assert "Canary Name" not in serialized
    assert "10001" not in serialized
    assert "condition" not in serialized


def test_structured_release_risk_gate_rejects_tampered_evidence_without_echo(
    tmp_path: Path,
) -> None:
    payload = _structured_release_evidence().to_dict()
    payload["privacy_models"]["k_anonymity"]["achieved_k"] = 1
    payload["unexpected_raw_value"] = "patient-canary"

    standalone = release_gates.evaluate_release_risk_evidence(payload)
    result = _gate().evaluate(
        _report(
            tmp_path,
            metric_updates={"structured_release_risk_evidence": payload},
        ),
        _baseline(),
    )

    check = _check(result, "structured_release_risk")
    assert standalone.passed is False
    assert result.decision == QUARANTINED
    assert check.passed is False
    assert check.details == {"integrity_verified": False}
    assert "patient-canary" not in json.dumps(check.to_dict(), sort_keys=True)


def test_structured_release_risk_gate_rejects_missing_sensitive_models() -> None:
    payload = _structured_release_evidence().to_dict()
    payload["privacy_models"]["l_diversity"] = None
    payload["privacy_models"]["t_closeness"] = None
    payload["integrity_hash"] = stable_hash(
        {key: value for key, value in payload.items() if key != "integrity_hash"}
    )

    check = release_gates.evaluate_release_risk_evidence(payload)

    assert check.passed is False
    assert check.details == {"integrity_verified": False}


def test_structured_release_risk_gate_rejects_nonzero_pruned_proof() -> None:
    payload = _structured_release_evidence().to_dict()
    information_loss = next(
        item for item in payload["utility"] if item["metric"] == "information_loss"
    )
    information_loss["after"] = 0.14
    information_loss["absolute_delta"] = 0.14
    payload["integrity_hash"] = stable_hash(
        {key: value for key, value in payload.items() if key != "integrity_hash"}
    )

    check = release_gates.evaluate_release_risk_evidence(payload)

    assert check.passed is False
    assert check.details == {"integrity_verified": False}


def test_structured_release_risk_gate_rejects_changed_pre_metrics_in_pruned_proof() -> (
    None
):
    payload = _structured_release_evidence().to_dict()
    payload["metrics"]["pre_transform"] = {
        "privacy_unit_count": 4,
        "equivalence_class_count": 4,
        "class_sizes": {
            "smallest": 1,
            "largest": 1,
            "mean": 1.0,
            "histogram": [
                {
                    "lower_bound": 1,
                    "upper_bound": 1,
                    "class_count": 4,
                    "privacy_unit_count": 4,
                }
            ],
        },
        "violations": {
            "k_class_count": 4,
            "l_class_count": 4,
            "t_class_count": 4,
            "any_class_count": 4,
            "privacy_unit_count": 4,
        },
    }
    payload["privacy_models"]["k_anonymity"]["pre_achieved_k"] = 1
    payload["privacy_models"]["l_diversity"]["pre_achieved_l"] = 1
    payload["privacy_models"]["t_closeness"]["pre_achieved_t"] = 1.0
    payload["integrity_hash"] = stable_hash(
        {key: value for key, value in payload.items() if key != "integrity_hash"}
    )

    check = release_gates.evaluate_release_risk_evidence(payload)

    assert check.passed is False
    assert check.details == {"integrity_verified": False}


def test_structured_release_risk_gate_requires_v3_for_pruned_proof() -> None:
    payload = _structured_release_evidence().to_dict()
    assert payload["search"]["complete"] is False
    payload["schema_version"] = 2
    payload["search"].pop("optimality_proven")
    payload["integrity_hash"] = stable_hash(
        {key: value for key, value in payload.items() if key != "integrity_hash"}
    )

    check = release_gates.evaluate_release_risk_evidence(payload)

    assert check.passed is False
    assert check.details["integrity_verified"] is True
    assert check.details["violations"]["search_optimality_proof"] == (
        "schema_version_3_required_for_pruned_proof"
    )


def test_standalone_release_risk_verifier_requires_evidence() -> None:
    check = release_gates.evaluate_release_risk_evidence(None)

    assert check.passed is False
    assert check.reason == "release-risk evidence is required"
    assert check.details == {"integrity_verified": False}


def test_multi_release_risk_gate_requires_complete_no_increase_review() -> None:
    evidence = _structured_release_evidence(
        composition=CompositionEvidence(
            release_count=2,
            longitudinal_linkage_assessed=True,
            prior_release_overlap_assessed=True,
            risk_status="no_material_increase_observed",
            evidence_digest=stable_hash({"kind": "multi-release-review"}),
        )
    )

    check = release_gates.evaluate_release_risk_evidence(evidence)

    assert check.passed is True
    assert check.details["composition_status"] == "no_material_increase_observed"


def test_release_gate_rejects_inconsistent_single_release_composition() -> None:
    payload = _structured_release_evidence().to_dict()
    payload["composition"].update(
        {
            "longitudinal_linkage_assessed": True,
            "prior_release_overlap_assessed": True,
            "risk_status": "increase_observed",
        }
    )
    payload["integrity_hash"] = stable_hash(
        {key: value for key, value in payload.items() if key != "integrity_hash"}
    )

    check = release_gates.evaluate_release_risk_evidence(payload)

    assert check.passed is False
    assert check.details == {"integrity_verified": False}


@pytest.mark.parametrize(
    ("longitudinal_assessed", "overlap_assessed", "risk_status", "violation"),
    [
        (
            False,
            True,
            "no_material_increase_observed",
            "longitudinal_linkage_assessed",
        ),
        (
            True,
            False,
            "no_material_increase_observed",
            "prior_release_overlap_assessed",
        ),
        (True, True, "inconclusive", "risk_status"),
    ],
)
def test_multi_release_risk_gate_rejects_incomplete_composition_review(
    longitudinal_assessed: bool,
    overlap_assessed: bool,
    risk_status: str,
    violation: str,
) -> None:
    evidence = _structured_release_evidence(
        composition=CompositionEvidence(
            release_count=2,
            longitudinal_linkage_assessed=longitudinal_assessed,
            prior_release_overlap_assessed=overlap_assessed,
            risk_status=risk_status,
            evidence_digest=stable_hash(
                {
                    "kind": "multi-release-review",
                    "longitudinal": longitudinal_assessed,
                    "overlap": overlap_assessed,
                    "status": risk_status,
                }
            ),
        )
    )

    check = release_gates.evaluate_release_risk_evidence(evidence)

    assert check.passed is False
    assert violation in check.details["violations"]["composition_review"]


def test_g4_computes_quant_delta_from_fp_parent_recall(tmp_path: Path) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={"format": "mlx-8bit"},
            metric_updates={
                "quant_recall_delta": None,
                "per_label_recall": {
                    "PERSON": 0.991,
                    "DATE": 0.990,
                    "ID_NUM": 0.990,
                    "API_KEY": 0.995,
                },
                "fp_parent_per_label_recall": {"PERSON": 0.996},
            },
        ),
        _baseline(),
    )

    check = _check(result, "G4")
    assert result.decision == QUARANTINED
    assert result.quant_recall_delta == pytest.approx(0.005)
    assert check.passed is False
    assert check.details["offending_labels"]["PERSON"]["limit"] == 0.005
    assert result.blocked_formats == ("mlx-8bit",)


def test_coreml_manifest_residency_and_parity_gate_passes(tmp_path: Path) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={
                "format": "coreml-fp16",
                "coreml_conversion_manifest": _coreml_manifest(),
            },
        ),
        _baseline(),
    )

    assert result.decision == RELEASABLE
    assert _check(result, "CoreML-ANE").passed is True
    assert _check(result, "CoreML-parity").passed is True


def test_coreml_manifest_blocks_cpu_fallback(tmp_path: Path) -> None:
    manifest = _coreml_manifest()
    manifest["variants"][0]["residency"]["cpu_fallback_layers"] = [
        {"name": "classifier", "compute_unit": "CPU"}
    ]

    result = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={
                "format": "coreml-fp16",
                "coreml_conversion_manifest": manifest,
            },
        ),
        _baseline(),
    )

    check = _check(result, "CoreML-ANE")
    assert result.decision == QUARANTINED
    assert check.passed is False
    assert check.blocking_format == "coreml-fp16"


def test_coreml_manifest_requires_int4_rejection_report(tmp_path: Path) -> None:
    manifest = _coreml_manifest()
    manifest["variants"][2]["parity"] = {"passed": False}

    result = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={
                "format": "coreml-fp16",
                "coreml_conversion_manifest": manifest,
            },
        ),
        _baseline(),
    )

    check = _check(result, "CoreML-parity")
    assert result.decision == QUARANTINED
    assert check.passed is False
    assert "coreml-int4" in check.details["failures"]


def test_manifest_coherence_fails_when_readme_count_drifts(tmp_path: Path) -> None:
    manifest = tmp_path / "models.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "repo_id": "OpenMed/unit-model",
                "family": "PII",
                "task": "token-classification",
                "languages": ["en"],
                "tier": "Tiny",
                "param_count": 44_000_000,
                "formats": ["mlx-fp"],
                "license": "apache-2.0",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    readme = tmp_path / "README.md"
    readme.write_text("2 models\n", encoding="utf-8")

    result = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={
                "manifest_path": str(manifest),
                "readme_path": str(readme),
            },
        ),
        _baseline(),
    )

    check = _check(result, "manifest_coherence")
    assert result.decision == QUARANTINED
    assert check.passed is False
    assert check.details["mismatches"]["readme"]["models"]["readme_floor"] == 2


def test_default_manifest_count_includes_published_android_onnx_fleet() -> None:
    rows = release_gates._load_manifest_rows(release_gates._DEFAULT_MANIFEST_PATH)

    derived_count = release_gates._published_android_onnx_derivative_count(rows)

    assert len(rows) == 1_520
    assert derived_count == 752
    assert len(rows) + derived_count >= 2_000


def _export_variant_manifest() -> dict[str, object]:
    return {
        "parent_format": "pytorch",
        "parent_recall": {"PERSON": 0.995, "DATE": 0.995, "ID_NUM": 0.995},
        "required_variants": ["onnx", "onnx-int8", "webgpu"],
        "variants": [
            {
                "format": "onnx",
                "tier": "base",
                "p50_ms": 90.0,
                "p95_ms": 280.0,
                "ram_mb": 700.0,
            },
            {
                "format": "onnx-int8",
                "tier": "tiny",
                "recall": {"PERSON": 0.992, "DATE": 0.993, "ID_NUM": 0.994},
                "p50_ms": 40.0,
                "p95_ms": 120.0,
                "ram_mb": 300.0,
            },
            {
                "format": "webgpu",
                "tier": "base",
                "p50_ms": 90.0,
                "p95_ms": 280.0,
                "ram_mb": 700.0,
            },
        ],
    }


def test_export_variant_gate_releases_passing_onnx_and_webgpu(
    tmp_path: Path,
) -> None:
    result = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={
                "export_variant_manifest": _export_variant_manifest(),
            },
        ),
        _baseline(),
    )

    assert result.decision == RELEASABLE
    assert result.blocked_formats == ()
    assert _check(result, "export_variants").passed is True
    assert _check(result, "export_variants:onnx-int8").passed is True
    assert _check(result, "export_variants:webgpu").passed is True


def test_export_variant_gate_blocks_degraded_variant_and_reports_format(
    tmp_path: Path,
) -> None:
    manifest = _export_variant_manifest()
    manifest["variants"][1]["recall"] = {
        "PERSON": 0.985,
        "DATE": 0.993,
        "ID_NUM": 0.994,
    }

    result = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={"export_variant_manifest": manifest},
        ),
        _baseline(),
    )

    blocked = _check(result, "export_variants:onnx-int8")
    assert result.decision == QUARANTINED
    assert blocked.passed is False
    assert blocked.blocking_format == "onnx-int8"
    assert result.blocked_formats == ("onnx-int8",)
    # Unrelated passing variants stay releasable within the same report.
    assert _check(result, "export_variants:onnx").passed is True
    assert _check(result, "export_variants:webgpu").passed is True


def test_export_variant_gate_fails_closed_when_required_variant_missing(
    tmp_path: Path,
) -> None:
    manifest = _export_variant_manifest()
    manifest["variants"] = [
        variant for variant in manifest["variants"] if variant["format"] != "webgpu"
    ]

    result = _gate().evaluate(
        _report(
            tmp_path,
            metadata_updates={"export_variant_manifest": manifest},
        ),
        _baseline(),
    )

    coverage = _check(result, "export_variants")
    assert result.decision == QUARANTINED
    assert coverage.passed is False
    assert coverage.details["missing_required"] == ["webgpu"]


def _coreml_manifest() -> dict[str, object]:
    parity_pass = {
        "passed": True,
        "max_recall_delta": 0.0,
        "span_mismatches": [],
    }
    return {
        "format": "openmed-coreml",
        "variants": [
            {
                "name": "coreml-fp16",
                "precision": "float16",
                "quantization": "none",
                "ane_residency_percentage": 0.95,
                "cpu_fallback_layers": [],
                "residency": {
                    "ane_residency_percentage": 0.95,
                    "cpu_fallback_layers": [],
                },
                "parity": dict(parity_pass),
            },
            {
                "name": "coreml-int8",
                "precision": "float16",
                "quantization": "int8",
                "parity": dict(parity_pass),
            },
            {
                "name": "coreml-int4",
                "precision": "float16",
                "quantization": "int4",
                "parity": {
                    "passed": False,
                    "max_recall_delta": 0.01,
                    "span_mismatches": [{"fixture_id": "stub"}],
                    "auto_rejected": True,
                    "rejection_reason": "recall delta exceeds limit",
                },
            },
        ],
    }
