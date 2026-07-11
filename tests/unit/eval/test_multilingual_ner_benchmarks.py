"""Tests for multilingual clinical NER benchmark loaders and scorecards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openmed.core.labels import CONDITION, MEDICATION, OTHER, PROCEDURE
from openmed.eval.datasets import (
    CANTEMIST,
    CMEEE,
    DEFT,
    MULTILINGUAL_NER,
    PHARMACONER,
    MultilingualNerCorpusRequired,
    assert_no_gated_content_committed,
    load_multilingual_ner_fixtures,
)
from openmed.eval.datasets.cantemist import load_cantemist
from openmed.eval.datasets.cmeee import load_cmeee
from openmed.eval.datasets.deft import load_deft
from openmed.eval.datasets.pharmaconer import load_pharmaconer
from openmed.eval.harness import (
    BenchmarkFixture,
    check_training_manifest_overlap,
    run_multilingual_ner_scorecard,
)
from openmed.eval.suites import load_suite_fixtures, suite_metadata

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "multilingual_ner"


def test_loaders_parse_synthetic_smoke_fixtures() -> None:
    assert_no_gated_content_committed(FIXTURE_ROOT)
    expected = {
        PHARMACONER: [MEDICATION],
        CANTEMIST: [CONDITION],
        DEFT: [CONDITION, PROCEDURE],
        CMEEE: [CONDITION, MEDICATION],
    }

    for benchmark, loader in _loaders().items():
        result = loader(_path(benchmark), allow_repo_path=True)
        fixtures = result.to_benchmark_fixtures()

        assert result.unmapped_labels == ()
        assert len(fixtures) == 1
        fixture = fixtures[0]
        assert fixture.metadata["benchmark"] == benchmark
        assert fixture.metadata["suite"] == MULTILINGUAL_NER
        assert fixture.metadata["text_hash"].startswith("sha256:")
        assert [span.label for span in fixture.gold_spans] == expected[benchmark]
        for span in fixture.gold_spans:
            assert fixture.text[span.start : span.end] == span.text
            assert span.metadata["source_label"]
            assert span.metadata["unmapped_label"] is False


def test_cmeee_parses_native_inclusive_offsets() -> None:
    result = load_cmeee(_path(CMEEE), allow_repo_path=True)
    record = result.records[0]

    assert [(span.start, span.end) for span in record.spans] == [(6, 8), (10, 14)]
    assert [span.source_label for span in record.spans] == ["dis", "dru"]
    assert [span.text for span in record.spans] == ["肺炎", "阿莫西林"]


def test_cmeee_rejects_native_entity_offset_mismatch(tmp_path: Path) -> None:
    payload = {
        "id": "cmeee-mismatch",
        "text": "肺炎与胃炎",
        "entities": [{"start_idx": 3, "end_idx": 4, "type": "dis", "entity": "肺炎"}],
    }
    path = tmp_path / "cmeee.json"
    path.write_text(json.dumps([payload]), encoding="utf-8")

    with pytest.raises(ValueError, match="span text mismatch"):
        load_cmeee(path)


def test_cmeee_rejects_incomplete_native_offsets(tmp_path: Path) -> None:
    payload = {
        "id": "cmeee-incomplete",
        "text": "肺炎",
        "entities": [{"start_idx": 0, "type": "dis", "entity": "肺炎"}],
    }
    path = tmp_path / "cmeee.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="integer start_idx and end_idx"):
        load_cmeee(path)


def test_generic_loader_rejects_mismatched_duplicate_mention(tmp_path: Path) -> None:
    payload = {
        "id": "pharmaconer-ambiguous",
        "text": "aspirin then aspirin",
        "spans": [
            {
                "start": 1,
                "end": 8,
                "label": "medication",
                "text": "aspirin",
            }
        ],
    }
    path = tmp_path / "pharmaconer.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="span text mismatch"):
        load_pharmaconer(path)


def test_loaders_require_explicit_external_paths() -> None:
    for loader in _loaders().values():
        with pytest.raises(MultilingualNerCorpusRequired, match="explicit local"):
            loader()

        with pytest.raises(MultilingualNerCorpusRequired, match="repository tree"):
            loader(_path(PHARMACONER))


def test_suite_registry_loads_multilingual_ner_smoke_fixtures() -> None:
    fixtures = load_suite_fixtures(
        MULTILINGUAL_NER,
        paths=_paths(),
        allow_repo_path=True,
    )
    metadata = suite_metadata(MULTILINGUAL_NER)

    assert len(fixtures) == 4
    assert {fixture.metadata["benchmark"] for fixture in fixtures} == set(_paths())
    assert metadata["suite"] == MULTILINGUAL_NER
    assert metadata["redistribution"] == "no licensed benchmark corpus text is bundled"
    assert metadata["languages"][CMEEE] == "zh"


def test_multilingual_scorecard_gates_per_language_benchmark_f1() -> None:
    fixtures = load_multilingual_ner_fixtures(_paths(), allow_repo_path=True)

    report = run_multilingual_ner_scorecard(
        fixtures,
        model_name="synthetic-perfect",
        runner=_identity_runner,
        min_exact_span_f1=0.85,
    )

    assert report.metrics["gate"]["passed"] is True
    assert report.metrics["gate"]["failures"] == []
    assert len(report.metrics["per_benchmark_language"]) == 4
    assert {
        (row["benchmark"], row["language"])
        for row in report.metrics["per_benchmark_language"]
    } == {
        (PHARMACONER, "es"),
        (CANTEMIST, "es"),
        (DEFT, "fr"),
        (CMEEE, "zh"),
    }
    for row in report.metrics["scorecard"]:
        assert row["exact_span_f1"]["f1"] == pytest.approx(1.0)
        assert row["precision"] == pytest.approx(1.0)
        assert row["recall"] == pytest.approx(1.0)
    per_language = {row["language"]: row for row in report.metrics["per_language"]}
    assert set(per_language) == {"es", "fr", "zh"}
    assert per_language["es"]["exact_span_f1"]["f1"] == pytest.approx(1.0)


def test_unmapped_labels_are_reported_not_dropped(tmp_path: Path) -> None:
    text = "Synthetic record mentions SAMPLE."
    start = text.index("SAMPLE")
    payload = {
        "id": "pharmaconer-unmapped",
        "language": "es",
        "spans": [
            {
                "end": start + len("SAMPLE"),
                "label": "UNEXPECTED_TYPE",
                "start": start,
                "text": "SAMPLE",
            }
        ],
        "text": text,
    }
    path = tmp_path / "pharmaconer.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = load_pharmaconer(path)
    fixture = result.to_benchmark_fixtures()[0]
    report = run_multilingual_ner_scorecard(
        [fixture],
        model_name="synthetic-perfect",
        runner=_identity_runner,
    )

    assert result.unmapped_labels == ("UNEXPECTED_TYPE",)
    assert fixture.gold_spans[0].label == OTHER
    assert fixture.gold_spans[0].metadata["unmapped_label"] is True
    assert report.metrics["unmapped_labels"] == {PHARMACONER: ["UNEXPECTED_TYPE"]}


def test_training_manifest_overlap_flags_eval_leakage(tmp_path: Path) -> None:
    fixture = load_pharmaconer(
        _path(PHARMACONER),
        allow_repo_path=True,
    ).to_benchmark_fixtures()[0]
    manifest = tmp_path / "train_manifest.jsonl"
    manifest.write_text(
        json.dumps({"record_id": "train-1", "text_hash": fixture.metadata["text_hash"]})
        + "\n",
        encoding="utf-8",
    )

    findings = check_training_manifest_overlap([fixture], manifest)
    report = run_multilingual_ner_scorecard(
        [fixture],
        model_name="synthetic-perfect",
        runner=_identity_runner,
        training_manifest_path=manifest,
    )

    assert len(findings) == 1
    assert findings[0].fixture_id == fixture.fixture_id
    assert findings[0].overlap_key == fixture.metadata["text_hash"]
    assert report.metrics["train_eval_overlap"]["finding_count"] == 1
    assert report.metrics["train_eval_overlap"]["passed"] is False
    assert report.metrics["gate"]["passed"] is False
    assert report.metrics["gate"]["failures"] == [
        {"finding_count": 1, "reason": "train_eval_overlap"}
    ]


def _identity_runner(
    fixture: BenchmarkFixture,
    model_name: str,
    device: str,
) -> tuple[Any, ...]:
    _ = (model_name, device)
    return tuple(fixture.gold_spans)


def _path(benchmark: str) -> Path:
    return FIXTURE_ROOT / f"{benchmark}.jsonl"


def _paths() -> dict[str, Path]:
    return {benchmark: _path(benchmark) for benchmark in _loaders()}


def _loaders() -> dict[str, Any]:
    return {
        PHARMACONER: load_pharmaconer,
        CANTEMIST: load_cantemist,
        DEFT: load_deft,
        CMEEE: load_cmeee,
    }
