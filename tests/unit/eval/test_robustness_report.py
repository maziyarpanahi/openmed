"""Tests for robustness perturbations and clean-vs-perturbed reports."""

from __future__ import annotations

import json

import pytest

from openmed.eval.harness import BenchmarkFixture
from openmed.eval.robustness import (
    MixedScriptEvasionGateError,
    adversarial_robustness_report,
    case_flip_perturbation,
    character_typo_perturbation,
    generate_confusable_attack_corpus,
    identity_perturbation,
    mixed_script_evasion_report,
    ocr_noise_perturbation,
    perturb_fixture,
    replay_adversarial_attack,
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


def test_adversarial_search_finds_misses_and_unicode_defense_recovers():
    fixture = _fixture()

    report = adversarial_robustness_report(
        "exact",
        [fixture],
        seed=23,
        distance_budget=2,
        beam_width=3,
        runner=_exact_gold_runner,
        suite_name="synthetic",
        generated_at="2026-06-24T00:00:00Z",
    )
    metrics = report.metrics["adversarial_robustness"]

    assert metrics["pre_defense_miss_count"] >= 1
    assert metrics["post_defense_miss_count"] == 0
    assert metrics["post_defense_leaked_chars"] == 0
    assert not metrics["violations"]
    assert set(metrics["post_defense_recall_under_attack_by_label"]) == {
        "ID_NUM",
        "PERSON",
    }
    assert all(
        recall >= metrics["recall_floor"]
        for recall in metrics["post_defense_recall_under_attack_by_label"].values()
    )

    payload = json.dumps(report.to_dict(), sort_keys=True)
    assert "John Doe" not in payload
    assert "A1001" not in payload
    assert report.artifacts
    for artifact in report.artifacts:
        assert artifact.distance <= artifact.distance_budget
        assert artifact.perturbation_classes
        first = replay_adversarial_attack(
            fixture,
            artifact,
            "exact",
            runner=_exact_gold_runner,
        )
        second = replay_adversarial_attack(
            fixture,
            artifact.to_dict(),
            "exact",
            runner=_exact_gold_runner,
        )
        assert first.text == second.text
        assert first.text != fixture.text


def test_confusable_attack_corpus_and_gate_cover_all_required_evasions():
    fixture = _confusable_fixture()

    attacks = generate_confusable_attack_corpus([fixture], seed=31)
    report = mixed_script_evasion_report(
        "exact-confusable",
        [fixture],
        seed=31,
        runner=_exact_confusable_runner,
        suite_name="synthetic-confusables",
    )

    assert {case.attack_class for case in attacks} == {
        "full_width",
        "homoglyph_cjk",
        "homoglyph_cyrillic",
        "homoglyph_greek",
        "zero_width",
    }
    assert report.attack_count == len(attacks) == 5
    assert report.detection_rate >= 0.99
    assert report.pre_defense_leakage.leakage.overall > 0.5
    assert report.post_defense_leakage.leakage.overall <= 0.01
    assert report.post_defense_leakage.leakage.leaked_chars == 0
    assert report.post_defense_leakage.to_dict()["unit"] == "grapheme_cluster"
    assert report.passed
    assert not report.violations
    assert "JOHN DOE" not in json.dumps(report.to_dict(), sort_keys=True)

    for case in attacks:
        span = case.fixture.gold_spans[case.span_index]
        assert span.text == case.fixture.text[span.start : span.end]


def test_mixed_script_gate_raises_for_fixture_that_breaches_leakage_ceiling():
    with pytest.raises(
        MixedScriptEvasionGateError,
        match="mixed_script_leakage_ceiling",
    ):
        mixed_script_evasion_report(
            "leaky-confusable",
            [_confusable_fixture()],
            runner=_exact_confusable_runner,
            defended_runner=lambda fixture, model_name, device: [],
            raise_on_violation=True,
        )


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


def _confusable_fixture() -> BenchmarkFixture:
    return BenchmarkFixture.from_mapping(
        {
            "id": "confusable-name",
            "text": "Patient JOHN DOE arrived.",
            "language": "en",
            "gold_spans": [
                {"start": 8, "end": 16, "label": "PERSON"},
            ],
        }
    )


def _exact_confusable_runner(fixture, model_name, device):
    assert device == "cpu"
    start = fixture.text.find("JOHN DOE")
    if start < 0:
        return []
    return [
        {
            "start": start,
            "end": start + len("JOHN DOE"),
            "label": "PERSON",
        }
    ]


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
