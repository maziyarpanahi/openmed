"""Chinese clinical NER foundation and leakage-gate tests."""

from __future__ import annotations

from typing import Any

import pytest

from openmed.core.labels import (
    BODY_SITE,
    CANONICAL_LABELS,
    CONDITION,
    JOB_DEPARTMENT,
    LAB_TEST,
    MEDICATION,
    MICROORGANISM,
    OTHER,
    PROCEDURE,
    normalize_label,
)
from openmed.eval.datasets.cmeee import load_cmeee, map_cmeee_label
from openmed.eval.datasets.multilingual_ner import MultilingualNerCorpusRequired
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.suites.chinese_clinical_ner import (
    CHINESE_CLINICAL_NER,
    ChineseClinicalNerLeakageError,
    load_chinese_clinical_ner_fixtures,
    run_chinese_clinical_ner_suite,
    run_synthetic_chinese_clinical_ner_smoke,
)

CMEEE_EXPECTED = {
    "bod": BODY_SITE,
    "dep": JOB_DEPARTMENT,
    "dis": CONDITION,
    "dru": MEDICATION,
    "equ": OTHER,
    "ite": LAB_TEST,
    "mic": MICROORGANISM,
    "pro": PROCEDURE,
    "sym": CONDITION,
}


def test_cmeee_labels_normalize_through_chinese_core_mapping() -> None:
    for source_label, expected in CMEEE_EXPECTED.items():
        assert normalize_label(source_label, lang="zh") == expected
        mapping = map_cmeee_label(source_label)
        assert mapping.canonical_label == expected
        assert mapping.mapped is True
        assert mapping.canonical_label in CANONICAL_LABELS

    core_clinical = {"bod", "dis", "dru", "ite", "mic", "pro", "sym"}
    assert all(CMEEE_EXPECTED[label] != OTHER for label in core_clinical)


def test_cmeee_real_loader_requires_an_explicit_external_path() -> None:
    with pytest.raises(MultilingualNerCorpusRequired, match="explicit local"):
        load_cmeee()


def test_bundled_chinese_fixture_covers_all_categories_and_synthetic_phi() -> None:
    fixtures = load_chinese_clinical_ner_fixtures()

    assert len(fixtures) == 2
    assert all(fixture.language == "zh" for fixture in fixtures)
    assert all(fixture.metadata["synthetic"] is True for fixture in fixtures)
    assert all(fixture.metadata["contains_real_phi"] is False for fixture in fixtures)
    clinical = fixtures[0]
    assert {span.metadata["source_label"] for span in clinical.gold_spans} == set(
        CMEEE_EXPECTED
    )
    assert all(
        clinical.text[span.start : span.end] == span.text
        for span in clinical.gold_spans
    )
    assert len(fixtures[1].metadata["phi_spans"]) == 3


def test_synthetic_suite_reports_per_label_metrics_and_zero_leakage() -> None:
    report = run_synthetic_chinese_clinical_ner_smoke()

    assert report.suite == CHINESE_CLINICAL_NER
    assert report.metrics["gate"]["passed"] is True
    assert report.metrics["phi_token_leakage"] == {
        "findings": [],
        "leaked_tokens": 0,
        "rate": 0.0,
        "total_tokens": 3,
    }
    assert report.metrics["per_label"][JOB_DEPARTMENT]["recall"] == 1.0
    assert report.metrics["per_label"][MICROORGANISM]["precision"] == 1.0
    assert "user-supplied local inputs" in report.metadata["data_boundary"]
    assert "multilingual privacy fallback" in report.metadata["model_notice"]


def test_suite_raises_without_exposing_surviving_identifier_text() -> None:
    fixtures = load_chinese_clinical_ner_fixtures()

    with pytest.raises(ChineseClinicalNerLeakageError) as caught:
        run_chinese_clinical_ner_suite(
            fixtures,
            model_name="synthetic-leaky",
            runner=_identity_runner,
            redactor=lambda fixture, predicted: fixture.text,
        )

    report = caught.value.report
    leakage = report.metrics["phi_token_leakage"]
    assert leakage["rate"] == 1.0
    assert leakage["leaked_tokens"] == 3
    assert all(
        set(finding) == {"end", "fixture_id", "label", "start", "token_hash"}
        for finding in leakage["findings"]
    )
    serialized = report.to_json()
    assert "王芳" not in serialized
    assert "CN123456" not in serialized
    assert "13800138000" not in serialized


def _identity_runner(
    fixture: BenchmarkFixture,
    model_name: str,
    device: str,
) -> tuple[Any, ...]:
    _ = (model_name, device)
    return tuple(fixture.gold_spans)
