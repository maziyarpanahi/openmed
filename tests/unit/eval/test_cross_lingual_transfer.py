"""Tests for synthetic cross-lingual transfer evaluation."""

from __future__ import annotations

import json

import pytest

from openmed.eval.cross_lingual_transfer import (
    CrossLingualTransferReport,
    cross_lingual_transfer_report,
)
from openmed.eval.harness import BenchmarkFixture


def test_transfer_report_builds_full_matrix_and_per_label_gaps() -> None:
    report = cross_lingual_transfer_report(
        "synthetic-transfer-model",
        _fixtures(),
        runner=_runner,
    )

    assert isinstance(report, CrossLingualTransferReport)
    assert report.languages == ("de", "en", "fr")
    assert report.labels == ("ID_NUM", "PERSON")
    assert set(report.matrix) == set(report.languages)
    assert all(set(row) == set(report.languages) for row in report.matrix.values())

    baseline = report.matrix["en"]["en"]
    transferred = report.matrix["en"]["fr"]
    assert baseline.zero_shot is False
    assert baseline.recall == pytest.approx(1.0)
    assert transferred.zero_shot is True
    assert transferred.recall == pytest.approx(0.5)
    assert transferred.leakage == pytest.approx(0.5)
    assert transferred.by_label["PERSON"].recall == pytest.approx(0.0)
    assert transferred.by_label["ID_NUM"].recall == pytest.approx(1.0)

    gap = report.transfer_gaps["en"]["fr"]
    assert gap.gap == pytest.approx(-0.5)
    assert gap.by_label["PERSON"].gap == pytest.approx(-1.0)
    assert gap.by_label["ID_NUM"].gap == pytest.approx(0.0)
    assert gap.by_label["PERSON"].leakage_gap == pytest.approx(1.0)


def test_transfer_ranking_and_serialization_are_deterministic() -> None:
    first = cross_lingual_transfer_report(
        "synthetic-transfer-model",
        _fixtures(),
        runner=_runner,
    )
    second = cross_lingual_transfer_report(
        "synthetic-transfer-model",
        list(reversed(_fixtures())),
        runner=_runner,
    )

    assert first.ranked_languages == ("fr",)
    assert first.ranked_candidates[0].deficit == pytest.approx(0.5)
    assert first.to_json() == second.to_json()
    assert first.to_markdown() == second.to_markdown()

    payload = json.loads(first.to_json())
    assert payload["artifact_type"] == "openmed.eval.cross_lingual_transfer"
    assert payload["provenance"] == "synthetic-offline"
    assert payload["ranked_candidates"][0]["target_language"] == "fr"
    assert "Synthetic-A" not in first.to_json()
    assert "Synthetic-A" not in first.to_markdown()


def test_runner_receives_source_target_and_held_out_metadata() -> None:
    seen: list[tuple[str, str, str | None, bool]] = []

    def runner(fixture: BenchmarkFixture, model_name: str, device: str):
        assert model_name == "metadata-model"
        assert device == "cpu"
        seen.append(
            (
                fixture.metadata["train_language"],
                fixture.metadata["eval_language"],
                fixture.metadata["held_out_language"],
                fixture.metadata["zero_shot"],
            )
        )
        return _gold_predictions(fixture)

    report = cross_lingual_transfer_report(
        "metadata-model",
        _fixtures(),
        runner=runner,
    )

    assert report.fixture_count == 3
    assert len(seen) == 9
    assert ("en", "en", None, False) in seen
    assert ("en", "fr", "fr", True) in seen


def test_mapping_fixtures_are_supported_without_restricted_corpus_inputs() -> None:
    report = cross_lingual_transfer_report(
        "mapping-model",
        _fixture_mappings(),
        runner=lambda fixture, _model, _device: _gold_predictions(fixture),
        languages=("en", "fr"),
    )

    assert report.languages == ("en", "fr")
    assert report.matrix["fr"]["en"].recall == pytest.approx(1.0)
    assert report.provenance == "synthetic-offline"


def _fixtures() -> list[BenchmarkFixture]:
    return [
        BenchmarkFixture.from_mapping(
            {
                "id": f"{language}-synthetic",
                "language": language,
                "text": "Synthetic-A1",
                "gold_spans": [
                    {"start": 9, "end": 10, "label": "PERSON"},
                    {"start": 10, "end": 11, "label": "ID_NUM"},
                ],
                "metadata": {"synthetic": True},
            }
        )
        for language in ("en", "fr", "de")
    ]


def _fixture_mappings() -> list[dict[str, object]]:
    return [
        {
            "id": f"{language}-synthetic",
            "language": language,
            "text": "Synthetic-A1",
            "gold_spans": [
                {"start": 9, "end": 10, "label": "PERSON"},
                {"start": 10, "end": 11, "label": "ID_NUM"},
            ],
            "metadata": {"synthetic": True},
        }
        for language in ("en", "fr", "de")
    ]


def _runner(fixture: BenchmarkFixture, model_name: str, device: str):
    assert model_name == "synthetic-transfer-model"
    assert device == "cpu"
    if (
        fixture.metadata["source_language"] == "en"
        and fixture.metadata["target_language"] == "fr"
    ):
        return _gold_predictions(fixture, labels={"ID_NUM"})
    return _gold_predictions(fixture)


def _gold_predictions(
    fixture: BenchmarkFixture,
    *,
    labels: set[str] | None = None,
) -> list[dict[str, object]]:
    return [
        {
            "start": span.start,
            "end": span.end,
            "label": span.label,
            "text": span.text,
        }
        for span in fixture.gold_spans
        if labels is None or span.label in labels
    ]
