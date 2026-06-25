"""Tests for robustness perturbations and clean-vs-perturbed reports."""

from __future__ import annotations

import pytest

from openmed.eval.harness import BenchmarkFixture
from openmed.eval.robustness import (
    case_flip_perturbation,
    character_typo_perturbation,
    identity_perturbation,
    ocr_noise_perturbation,
    perturb_fixture,
    robustness_report,
    whitespace_noise_perturbation,
)


def test_perturbers_are_seeded_and_reproject_gold_spans_without_drift():
    fixture = _fixture()
    perturbation = whitespace_noise_perturbation(probability=1.0)

    first = perturb_fixture(fixture, perturbation, seed=17)
    second = perturb_fixture(fixture, perturbation, seed=17)

    assert first.text == second.text
    assert first.gold_spans == second.gold_spans
    assert first.text != fixture.text
    for span in first.gold_spans:
        assert span.text == first.text[span.start : span.end]
        assert span.end > span.start


@pytest.mark.parametrize(
    "perturbation",
    [
        character_typo_perturbation(probability=1.0),
        ocr_noise_perturbation(probability=1.0),
        case_flip_perturbation(probability=1.0),
        whitespace_noise_perturbation(probability=1.0),
    ],
)
def test_builtin_perturbers_modify_text_and_keep_span_offsets_valid(perturbation):
    fixture = BenchmarkFixture.from_mapping(
        {
            "id": "ocr-note",
            "text": "MRN m rn l O 1 belongs to John Doe",
            "language": "en",
            "gold_spans": [
                {"start": 4, "end": 14, "label": "MRN"},
                {"start": 26, "end": 34, "label": "PERSON"},
            ],
        }
    )

    perturbed = perturb_fixture(fixture, perturbation, seed=3)

    assert perturbed.text != fixture.text
    for span in perturbed.gold_spans:
        assert span.text == perturbed.text[span.start : span.end]


def test_identity_perturbation_yields_zero_metric_deltas():
    fixture = _fixture()

    report = robustness_report(
        "exact",
        [fixture],
        [identity_perturbation()],
        seed=11,
        runner=_exact_gold_runner,
    )
    variant = report.variant("identity")

    assert variant.deltas["leakage"] == pytest.approx(0.0)
    assert variant.deltas["recall"] == pytest.approx(0.0)
    assert variant.deltas["f1"] == pytest.approx(0.0)
    assert variant.report.metrics["leakage"]["overall"] == pytest.approx(
        report.clean.metrics["leakage"]["overall"]
    )


def test_robustness_report_scores_perturbation_deltas_against_clean_run():
    fixture = _fixture()
    typo = character_typo_perturbation(probability=1.0, name="typo_all")

    report = robustness_report(
        "exact",
        [fixture],
        [typo],
        seed=5,
        runner=_exact_gold_runner,
        suite_name="synthetic",
        generated_at="2026-06-24T00:00:00Z",
    )
    variant = report.variant("typo_all")

    assert report.clean.metrics["leakage"]["overall"] == pytest.approx(0.0)
    assert (
        variant.report.metrics["leakage"]["overall"]
        >= (report.clean.metrics["leakage"]["overall"])
    )
    assert variant.deltas["leakage"] > 0.0
    assert variant.deltas["recall"] < 0.0
    assert variant.deltas["f1"] < 0.0
    assert variant.to_dict()["deltas"] == dict(variant.deltas)
    assert report.to_dict()["variants"]["typo_all"]["seed"] == 5


def _fixture() -> BenchmarkFixture:
    return BenchmarkFixture.from_mapping(
        {
            "id": "note-1",
            "text": "Patient John Doe has MRN A1001.",
            "language": "en",
            "gold_spans": [
                {"start": 8, "end": 16, "label": "PERSON"},
                {"start": 25, "end": 30, "label": "MRN"},
            ],
        }
    )


def _exact_gold_runner(fixture, model_name, device):
    assert model_name == "exact"
    assert device == "cpu"
    original = {
        "A1001": {"label": "MRN"},
        "John Doe": {"label": "PERSON"},
    }
    predictions = []
    for value, payload in original.items():
        start = fixture.text.find(value)
        if start >= 0:
            predictions.append(
                {
                    "start": start,
                    "end": start + len(value),
                    "label": payload["label"],
                }
            )
    return predictions
