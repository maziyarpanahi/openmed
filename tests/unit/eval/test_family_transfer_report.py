"""Tests for synthetic family-aware cross-lingual transfer reporting."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from openmed.eval import (
    FAMILY_TRANSFER_ARTIFACT_TYPE,
    cross_lingual_family_transfer_report,
    load_family_transfer_fixtures,
)
from openmed.eval.harness import BenchmarkFixture
from openmed.training.adapters import AdapterParameterAccounting


def test_bundled_family_transfer_gold_is_synthetic_and_paired() -> None:
    fixtures = load_family_transfer_fixtures()

    assert len(fixtures) == 4
    assert {fixture.language for fixture in fixtures} == {"hi", "te"}
    assert {fixture.metadata["family_transfer"]["role"] for fixture in fixtures} == {
        "donor",
        "target",
    }
    assert all(fixture.metadata["synthetic"] is True for fixture in fixtures)
    assert all(fixture.gold_spans for fixture in fixtures)


def test_report_exposes_family_modes_deltas_and_donor_non_regression() -> None:
    report = cross_lingual_family_transfer_report(
        "synthetic-family-transfer-model",
        runner=_passing_runner,
    )

    assert len(report.comparisons) == 1
    comparison = report.comparisons[0]
    assert comparison.family == "indic"
    assert comparison.donor_language == "hi"
    assert comparison.target_language == "te"
    assert comparison.donor_baseline.f1 == pytest.approx(1.0)
    assert comparison.donor_adapted.f1 == pytest.approx(1.0)
    assert comparison.donor_non_regression is True
    assert comparison.donor_delta == pytest.approx(0.0)
    assert comparison.target_baseline.f1 == pytest.approx(0.0)
    assert comparison.target_zero_shot.f1 == pytest.approx(2.0 / 3.0)
    assert comparison.target_adapted.f1 == pytest.approx(1.0)
    assert comparison.target_adapted.f1 >= 0.80
    assert comparison.adapted_target_passed is True
    assert report.passed is True

    payload = json.loads(report.to_json())
    transfer = payload["families"]["indic"][0]
    assert payload["artifact_type"] == FAMILY_TRANSFER_ARTIFACT_TYPE
    assert payload["summary"] == {
        "all_adapted_targets_passed": True,
        "all_donors_non_regressed": True,
        "all_zero_shot_improved": True,
        "family_count": 1,
        "passed": True,
        "transfer_count": 1,
    }
    assert transfer["donor"] == "hi"
    assert transfer["target"] == "te"
    assert transfer["baseline"]["donor"]["f1"] == pytest.approx(1.0)
    assert transfer["baseline"]["target"]["f1"] == pytest.approx(0.0)
    assert transfer["zero_shot"]["donor"]["f1"] == pytest.approx(1.0)
    assert transfer["zero_shot"]["target"]["f1"] == pytest.approx(2.0 / 3.0)
    assert transfer["adapted"]["target"]["f1"] == pytest.approx(1.0)
    assert transfer["deltas"]["donor_to_target"]["baseline"] == pytest.approx(-1.0)
    assert transfer["deltas"]["target_adapted_vs_zero_shot"] == pytest.approx(1.0 / 3.0)
    assert transfer["donor_non_regression"] is True

    serialized = report.to_json()
    markdown = report.to_markdown()
    assert serialized == report.to_json()
    assert markdown == report.to_markdown()
    assert "OM-SYN-" not in serialized
    assert "OM-SYN-" not in markdown


def test_report_qualifies_f1_retention_and_parameter_efficiency() -> None:
    accounting = AdapterParameterAccounting(
        shared_backbone_parameter_count=110_000_000,
        adapter_trainable_parameter_count=524_288,
        task_head_trainable_parameter_count=65_536,
        full_language_model_trainable_parameter_count=110_065_536,
    )

    report = cross_lingual_family_transfer_report(
        "synthetic-family-transfer-model",
        runner=_passing_runner,
        parameter_accounting_by_adapter={"family-transfer/indic-hi-to-te": accounting},
        adapted_f1_fraction_floor=0.90,
        trainable_fraction_ceiling=0.01,
    )

    qualification = report.comparisons[0].efficiency
    assert qualification is not None
    assert qualification.full_language_model.f1 == pytest.approx(1.0)
    assert qualification.adapted_f1_fraction_of_full_model == pytest.approx(1.0)
    assert qualification.adapted_f1_fraction_passed is True
    assert qualification.parameter_accounting.trainable_parameter_count == 589_824
    assert qualification.parameter_efficiency_passed is True
    assert qualification.passed is True
    assert report.passed is True

    payload = report.to_dict()
    assert payload["summary"]["all_efficiency_qualifications_passed"] is True
    evidence = payload["families"]["indic"][0]["efficiency"]
    assert evidence["adapted_f1_fraction_floor"] == pytest.approx(0.90)
    assert evidence["parameter_accounting"]["trainable_fraction_of_full_model"] < 0.01
    serialized = report.to_json()
    markdown = report.to_markdown()
    assert "Adapter efficiency qualification" in markdown
    assert "OM-SYN-" not in serialized
    assert "OM-SYN-" not in markdown


def test_report_requires_parameter_accounting_for_each_evaluated_adapter() -> None:
    with pytest.raises(
        ValueError,
        match="missing parameter accounting.*family-transfer/indic-hi-to-te",
    ):
        cross_lingual_family_transfer_report(
            "synthetic-family-transfer-model",
            runner=_passing_runner,
            parameter_accounting_by_adapter={},
        )


def test_report_fails_when_adapted_f1_misses_full_model_fraction() -> None:
    report = cross_lingual_family_transfer_report(
        "synthetic-family-transfer-model",
        runner=_f1_retention_failure_runner,
        parameter_accounting_by_adapter={
            "family-transfer/indic-hi-to-te": AdapterParameterAccounting(
                shared_backbone_parameter_count=110_000_000,
                adapter_trainable_parameter_count=524_288,
                task_head_trainable_parameter_count=65_536,
                full_language_model_trainable_parameter_count=110_065_536,
            )
        },
        adapted_f1_fraction_floor=0.90,
        trainable_fraction_ceiling=0.01,
    )

    comparison = report.comparisons[0]
    qualification = comparison.efficiency
    assert qualification is not None
    assert comparison.target_adapted.f1 == pytest.approx(0.80)
    assert comparison.adapted_target_passed is True
    assert qualification.adapted_f1_fraction_of_full_model == pytest.approx(0.80)
    assert qualification.adapted_f1_fraction_passed is False
    assert qualification.parameter_efficiency_passed is True
    assert report.passed is False


def test_report_fails_when_trainable_fraction_exceeds_ceiling() -> None:
    report = cross_lingual_family_transfer_report(
        "synthetic-family-transfer-model",
        runner=_passing_runner,
        parameter_accounting_by_adapter={
            "family-transfer/indic-hi-to-te": AdapterParameterAccounting(
                shared_backbone_parameter_count=110_000_000,
                adapter_trainable_parameter_count=524_288,
                task_head_trainable_parameter_count=65_536,
                full_language_model_trainable_parameter_count=110_065_536,
            )
        },
        trainable_fraction_ceiling=0.001,
    )

    qualification = report.comparisons[0].efficiency
    assert qualification is not None
    assert qualification.adapted_f1_fraction_passed is True
    assert qualification.parameter_efficiency_passed is False
    assert qualification.passed is False
    assert report.passed is False


def test_report_flags_a_post_adaptation_donor_regression() -> None:
    report = cross_lingual_family_transfer_report(
        "synthetic-family-transfer-model",
        runner=_donor_regression_runner,
    )

    comparison = report.comparisons[0]
    assert comparison.donor_baseline.f1 == pytest.approx(1.0)
    assert comparison.donor_adapted.f1 == pytest.approx(0.0)
    assert comparison.donor_delta == pytest.approx(-1.0)
    assert comparison.donor_non_regression is False
    assert report.to_dict()["summary"]["all_donors_non_regressed"] is False
    assert report.passed is False


def test_custom_family_transfer_fixture_must_be_marked_synthetic() -> None:
    donor, *remaining = load_family_transfer_fixtures()
    tainted = replace(
        donor,
        metadata={**donor.metadata, "synthetic": False},
    )

    with pytest.raises(ValueError, match="must be synthetic"):
        cross_lingual_family_transfer_report(
            "synthetic-family-transfer-model",
            [tainted, *remaining],
            runner=_passing_runner,
        )


def _passing_runner(
    fixture: BenchmarkFixture,
    model_name: str,
    device: str,
):
    assert model_name == "synthetic-family-transfer-model"
    assert device == "cpu"
    role = fixture.metadata["evaluation_role"]
    mode = fixture.metadata["transfer_mode"]
    if mode == "full_model":
        assert role == "target"
        assert fixture.metadata["adapter_language"] is None
        assert fixture.metadata["source_language"] == "te"
        assert fixture.metadata["full_language_model"] is True
    if role == "donor" or mode in {"adapted", "full_model"}:
        return _gold_predictions(fixture)
    if mode == "zero_shot" and fixture.fixture_id.endswith("001"):
        return _gold_predictions(fixture)
    return []


def _donor_regression_runner(
    fixture: BenchmarkFixture,
    model_name: str,
    device: str,
):
    role = fixture.metadata["evaluation_role"]
    mode = fixture.metadata["transfer_mode"]
    if role == "donor" and mode == "adapted":
        return []
    if role == "donor" or mode in {"zero_shot", "adapted"}:
        return _gold_predictions(fixture)
    return []


def _f1_retention_failure_runner(
    fixture: BenchmarkFixture,
    model_name: str,
    device: str,
):
    role = fixture.metadata["evaluation_role"]
    mode = fixture.metadata["transfer_mode"]
    if role == "donor" or mode == "full_model":
        return _gold_predictions(fixture)
    if mode == "adapted":
        predictions = _gold_predictions(fixture)
        if fixture.fixture_id.endswith("001"):
            predictions.append(
                {
                    "start": 0,
                    "end": 1,
                    "label": "PERSON",
                    "text": fixture.text[:1],
                }
            )
        return predictions
    if mode == "zero_shot" and fixture.fixture_id.endswith("001"):
        return _gold_predictions(fixture)
    return []


def _gold_predictions(fixture: BenchmarkFixture) -> list[dict[str, object]]:
    return [
        {
            "start": span.start,
            "end": span.end,
            "label": span.label,
            "text": span.text,
        }
        for span in fixture.gold_spans
    ]
