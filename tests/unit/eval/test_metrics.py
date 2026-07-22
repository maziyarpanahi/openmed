"""Unit tests for eval benchmark metrics and reports."""

from __future__ import annotations

import json

import pytest

from openmed.eval.golden import non_latin_golden_fixtures
from openmed.eval.harness import BenchmarkFixture, load_fixtures, run_benchmark
from openmed.eval.leakage_heatmap import compute_extraction_reemission_heatmap
from openmed.eval.metrics import (
    ABSTENTION_ROUTE_REDACT,
    CRITICAL_FINDING_CATEGORY_DRUG_ALLERGY,
    EvalSpan,
    apply_abstention_policy,
    compute_abstention_metrics,
    compute_character_recall,
    compute_critical_finding_recall,
    compute_exact_span_f1,
    compute_extraction_reemission_leakage,
    compute_leakage_rate,
    compute_metrics_bundle,
    compute_recall_slices,
    compute_relaxed_span_f1,
)
from openmed.eval.report import BenchmarkReport


def test_eval_modules_import_cleanly():
    import openmed.eval.harness
    import openmed.eval.metrics
    import openmed.eval.report
    import openmed.eval.suites

    assert openmed.eval.harness.run_benchmark
    assert openmed.eval.metrics.compute_leakage_rate
    assert openmed.eval.report.BenchmarkReport
    assert openmed.eval.suites.DEFAULT_SUITES == (
        "golden",
        "i2b2",
        "n2c2",
        "shield",
        "drugprot",
        "policy_compliance",
        "biomedical-ner",
        "multilingual-clinical-ner",
        "masakhaner",
        "naamapadam",
        "chinese-clinical-ner",
        "multimodal_dicom",
        "code_mixed_routing",
    )


def test_leakage_rate_is_char_weighted_and_sliced():
    text = "SSN 123-45-6789 John"
    gold = [
        {"start": 4, "end": 15, "label": "SSN", "language": "en", "device": "cpu"},
        {"start": 16, "end": 20, "label": "PERSON", "language": "en", "device": "cpu"},
    ]
    predicted = [
        {"start": 16, "end": 20, "label": "PERSON", "language": "en", "device": "cpu"},
    ]

    result = compute_leakage_rate(gold, predicted, source_text=text)

    assert result.overall == pytest.approx(11 / 15)
    assert result["overall"] == pytest.approx(11 / 15)
    assert result.by_label["SSN"] == pytest.approx(1.0)
    assert result.by_label["PERSON"] == pytest.approx(0.0)
    assert result.by_language["en"] == pytest.approx(11 / 15)
    assert result.by_device["cpu"] == pytest.approx(11 / 15)
    assert result.leaked_chars_by_label["SSN"] == 11
    assert result.total_chars_by_label["PERSON"] == 4


def test_extraction_reemission_leakage_counts_surfaces_and_offsets():
    text = "Patient Alice has SSN 123-45-6789."
    name_start = text.index("Alice")
    ssn_start = text.index("123-45-6789")
    gold = [
        {"start": name_start, "end": name_start + 5, "label": "PERSON"},
        {"start": ssn_start, "end": ssn_start + 11, "label": "SSN"},
    ]
    extraction = {
        "facts": [
            {"value": "Alice", "concept": {"label": "masked condition"}},
            {
                "value": "redacted",
                "evidence": {"start": ssn_start, "end": ssn_start + 11},
            },
        ],
        "fhir": {"resourceType": "Observation", "valueString": "123-45-6789"},
    }

    result = compute_extraction_reemission_leakage(
        gold,
        extraction,
        source_text=text,
    )
    heatmap = compute_extraction_reemission_heatmap(
        gold,
        extraction,
        source_text=text,
    )

    assert result.overall == pytest.approx(1.0)
    assert result.leaked_chars_by_label["PERSON"] == 5
    assert result.leaked_chars_by_label["SSN"] == 11
    assert heatmap.cells["PERSON"]["en"].leaked_chars == 5
    assert heatmap.cells["SSN"]["en"].leaked_chars == 11


def test_clean_extraction_output_has_zero_reemission_leakage():
    text = "Patient Alice has SSN 123-45-6789."
    name_start = text.index("Alice")
    ssn_start = text.index("123-45-6789")
    gold = [
        {"start": name_start, "end": name_start + 5, "label": "PERSON"},
        {"start": ssn_start, "end": ssn_start + 11, "label": "SSN"},
    ]

    result = compute_extraction_reemission_leakage(
        gold,
        {"facts": [{"value": "stable", "evidence": {"start": 0, "end": 7}}]},
        source_text=text,
    )
    metrics = compute_metrics_bundle(
        gold,
        gold,
        extraction_outputs={"facts": [{"value": "stable"}]},
        source_text=text,
    )

    assert result.overall == 0.0
    assert result.leaked_chars == 0
    assert metrics["leakage"]["overall"] == 0.0
    assert metrics["extraction_reemission_leakage"]["overall"] == 0.0


def test_extraction_offsets_are_not_scanned_as_numeric_phi_surfaces():
    text = "12 was recorded before a stable clinical observation."
    gold = [{"start": 0, "end": 2, "label": "ID_NUM"}]
    extraction = {
        "facts": [
            {
                "value": "stable",
                "evidence": {"start": 12, "end": 20},
                "offsets": [30, 35],
            }
        ]
    }

    result = compute_extraction_reemission_leakage(
        gold,
        extraction,
        source_text=text,
    )

    assert result.overall == 0.0
    assert result.leaked_chars == 0


def test_extraction_reemission_detects_non_latin_golden_fixture():
    fixture = non_latin_golden_fixtures()[0]
    span = next(
        span
        for span in fixture.gold_spans
        if any(ord(char) > 127 and char.isalpha() for char in span.text)
    )

    result = compute_extraction_reemission_leakage(
        fixture.gold_spans,
        {"grounding": {"concept_label": f"echoed {span.text}"}},
        default_language=fixture.language,
    )

    assert result.leaked_chars_by_language[fixture.language] >= span.length
    assert result.by_language[fixture.language] > 0.0


def test_zero_gold_spans_do_not_report_leakage_and_have_full_recall():
    leakage = compute_leakage_rate([], [])
    recall = compute_character_recall([], [])
    exact = compute_exact_span_f1([], [])
    relaxed = compute_relaxed_span_f1([], [])

    assert leakage.overall == 0.0
    assert leakage.total_chars == 0
    assert recall.rate == 1.0
    assert exact.f1 == 1.0
    assert relaxed.f1 == 1.0


def test_exact_and_relaxed_f1_differ_on_boundary_drift():
    gold = [EvalSpan(start=0, end=4, label="PERSON", text="John")]
    predicted = [EvalSpan(start=0, end=3, label="PERSON", text="Joh")]

    exact = compute_exact_span_f1(gold, predicted)
    relaxed = compute_relaxed_span_f1(gold, predicted)

    assert exact.f1 == 0.0
    assert relaxed.f1 == 1.0
    assert exact.false_negatives == 1
    assert relaxed.true_positives == 1


def test_critical_abstentions_route_to_redaction_not_passthrough():
    decisions = apply_abstention_policy(
        [],
        [
            {
                "start": 0,
                "end": 11,
                "label": "SSN",
                "language": "en",
                "confidence": 0.25,
            }
        ],
        confidence_threshold=0.90,
    )

    assert decisions[0].accepted is False
    assert decisions[0].route == ABSTENTION_ROUTE_REDACT
    assert all(decision.route != "passthrough" for decision in decisions)


def test_abstention_metrics_are_sliced_deterministic_and_risk_bounded():
    gold: list[dict[str, object]] = []
    predicted: list[dict[str, object]] = []
    for index in range(120):
        start = index * 5
        span = {
            "start": start,
            "end": start + 3,
            "label": "SSN",
            "language": "en",
        }
        gold.append(span)
        predicted.append({**span, "confidence": 0.95})
    for index in range(40):
        start = 1000 + index * 5
        predicted.append(
            {
                "start": start,
                "end": start + 3,
                "label": "SSN",
                "language": "en",
                "confidence": 0.40,
            }
        )
    for index in range(4):
        start = 2000 + index * 5
        predicted.append(
            {
                "start": start,
                "end": start + 4,
                "label": "PERSON",
                "language": "fr",
                "confidence": 0.30,
            }
        )

    first = compute_abstention_metrics(
        gold,
        predicted,
        confidence_threshold=0.95,
        target_risk=0.10,
        confidence_level=0.80,
        bootstrap_resamples=100,
        seed=7,
    ).to_dict()
    second = compute_abstention_metrics(
        gold,
        predicted,
        confidence_threshold=0.95,
        target_risk=0.10,
        confidence_level=0.80,
        bootstrap_resamples=100,
        seed=7,
    ).to_dict()

    assert first == second
    assert first["abstention_rate"]["by_label"]["SSN"] == pytest.approx(0.25)
    assert first["abstention_rate"]["by_language"]["en"] == pytest.approx(0.25)
    assert first["abstention_rate"]["by_language"]["fr"] == pytest.approx(1.0)
    assert first["residual_risk"]["by_label"]["SSN"] == 0.0
    assert first["residual_risk"]["by_language"]["en"] == 0.0
    assert first["residual_risk"]["critical"] == 0.0
    assert first["residual_risk"]["bootstrap"]["max"] <= 0.10


def test_recall_slices_cover_label_language_and_device_edges():
    gold = [
        EvalSpan(start=0, end=4, label="PERSON", language="en", device="cpu"),
        EvalSpan(start=10, end=21, label="SSN", language="fr", device="coreml"),
    ]
    predicted = [
        EvalSpan(start=0, end=4, label="PERSON", language="en", device="cpu"),
        EvalSpan(start=10, end=15, label="SSN", language="fr", device="coreml"),
    ]

    recall = compute_recall_slices(gold, predicted)

    assert recall.overall == pytest.approx(9 / 15)
    assert recall.by_label["PERSON"] == pytest.approx(1.0)
    assert recall.by_label["SSN"] == pytest.approx(5 / 11)
    assert recall.by_language["fr"] == pytest.approx(5 / 11)
    assert recall.by_device["coreml"] == pytest.approx(5 / 11)
    assert "mlx-8bit" in recall.by_device


def test_critical_finding_recall_reports_phi_free_missed_details():
    text = (
        "Critical safety probe: The synthetic patient has documented allergy to "
        "penicillin with anaphylaxis, acute myocardial infarction, and potassium "
        "6.9 mmol/L requiring urgent review."
    )
    gold = [
        EvalSpan(
            start=71,
            end=81,
            label="MEDICATION",
            text="penicillin",
            metadata={
                "critical_finding": True,
                "critical_finding_category": CRITICAL_FINDING_CATEGORY_DRUG_ALLERGY,
                "fixture_id": "fixture-critical",
            },
        ),
        EvalSpan(
            start=100,
            end=127,
            label="CONDITION",
            text="acute myocardial infarction",
            metadata={
                "critical_finding": True,
                "critical_finding_category": "critical_diagnosis",
                "fixture_id": "fixture-critical",
            },
        ),
    ]
    predicted = [
        EvalSpan(
            start=100,
            end=127,
            label="CONDITION",
            text="acute myocardial infarction",
        )
    ]

    recall = compute_critical_finding_recall(gold, predicted, source_text=text)
    payload = recall.to_dict()

    assert recall.overall == pytest.approx(0.5)
    assert recall.by_category[CRITICAL_FINDING_CATEGORY_DRUG_ALLERGY] == 0.0
    assert payload["missed_findings"] == [
        {
            "category": CRITICAL_FINDING_CATEGORY_DRUG_ALLERGY,
            "fixture_id": "fixture-critical",
            "start": 71,
            "end": 81,
            "label": "MEDICATION",
        }
    ]
    assert "penicillin" not in json.dumps(payload)


def test_benchmark_report_serializes_deterministically_to_json_and_markdown():
    report = BenchmarkReport(
        suite="golden",
        model_name="privacy-filter",
        device="cpu",
        fixture_count=2,
        generated_at="2026-06-11T00:00:00Z",
        metrics={
            "leakage": {"overall": 0.25, "by_label": {"SSN": 1.0, "PERSON": 0.0}},
            "exact_span_f1": {"f1": 0.8},
        },
        metadata={"z": 1, "a": {"b": True}},
    )

    assert report.to_json() == report.to_json()
    assert json.loads(report.to_json())["metrics"]["leakage"]["overall"] == 0.25
    markdown = report.to_markdown()

    assert markdown == report.to_markdown()
    assert "| `exact_span_f1.f1` | 0.8 |" in markdown
    assert "| `a.b` | true |" in markdown
    assert "| `z` | 1 |" in markdown


def test_harness_runs_with_injected_runner_without_loading_models():
    fixture = BenchmarkFixture.from_mapping(
        {
            "id": "note-1",
            "text": "Patient John",
            "language": "en",
            "gold_spans": [{"start": 8, "end": 12, "label": "PERSON"}],
        }
    )

    def runner(fixture, model_name, device):
        assert model_name == "test-model"
        assert device == "cpu"
        return [{"start": 8, "end": 12, "label": "PERSON"}]

    report = run_benchmark(
        [fixture],
        suite="golden",
        model_name="test-model",
        runner=runner,
        generated_at="2026-06-11T00:00:00Z",
    )

    assert report.fixture_count == 1
    assert report.metrics["leakage"]["overall"] == 0.0
    assert report.metrics["exact_span_f1"]["f1"] == 1.0


def test_harness_surfaces_abstention_metrics_with_thresholds():
    fixture = BenchmarkFixture.from_mapping(
        {
            "id": "note-1",
            "text": "Patient John and Jane",
            "language": "en",
            "gold_spans": [
                {"start": 8, "end": 12, "label": "PERSON"},
                {"start": 17, "end": 21, "label": "PERSON"},
            ],
        }
    )

    def runner(fixture, model_name, device):
        return [
            {
                "start": 8,
                "end": 12,
                "label": "PERSON",
                "confidence": 0.95,
            },
            {
                "start": 17,
                "end": 21,
                "label": "PERSON",
                "confidence": 0.50,
            },
        ]

    report = run_benchmark(
        [fixture],
        suite="golden",
        model_name="test-model",
        runner=runner,
        abstention_thresholds={"PERSON": {"en": 0.90}},
        abstention_target_risk=0.10,
        abstention_confidence_level=0.80,
        abstention_bootstrap_resamples=10,
        abstention_seed=3,
    )

    assert report.metrics["abstention"]["abstention_rate"]["overall"] == 0.5
    assert "abstention.residual_risk.critical" in report.to_markdown()


def test_harness_rejects_duplicate_fixture_ids(tmp_path):
    rows = [
        {
            "id": "duplicate-note",
            "text": "Patient John",
            "language": "en",
            "gold_spans": [{"start": 8, "end": 12, "label": "PERSON"}],
        },
        {
            "id": "duplicate-note",
            "text": "Patient Jane",
            "language": "en",
            "gold_spans": [{"start": 8, "end": 12, "label": "PERSON"}],
        },
    ]
    fixture_path = tmp_path / "fixtures.json"
    fixture_path.write_text(json.dumps({"fixtures": rows}), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate benchmark fixture id"):
        load_fixtures(fixture_path)

    fixtures = [BenchmarkFixture.from_mapping(row) for row in rows]

    def runner(fixture, model_name, device):
        raise AssertionError("runner should not be called for duplicate fixture IDs")

    with pytest.raises(ValueError, match="duplicate benchmark fixture id"):
        run_benchmark(
            fixtures,
            suite="golden",
            model_name="test-model",
            runner=runner,
        )
