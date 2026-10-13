"""Declared synthetic constraints and independently failing controls."""

from dataclasses import replace

import pytest

from openmed.training.synthetic.lab_timeseries import (
    LabIntervention,
    LabSeriesSpec,
    generate_lab_series,
    lab_negative_control,
    validate_lab_series,
)


def spec(**changes):
    args = dict(
        analyte="synthetic_marker",
        unit="arb",
        count=6,
        lower=0,
        upper=100,
        baseline=50,
        trend_per_step=1,
        jitter=0.1,
        max_step_change=5,
        missing_indices=(2,),
        interventions=(LabIntervention(3, -2, "synthetic_effect"),),
    )
    args.update(changes)
    return LabSeriesSpec(**args)


def test_reproducible_constrained_series_and_exact_missingness():
    recipe = spec()
    series = generate_lab_series(recipe, seed=42)
    assert series == generate_lab_series(recipe, seed=42)
    assert series != generate_lab_series(recipe, seed=43)
    assert validate_lab_series(recipe, series) == ()
    assert series.samples[2].value is None
    assert all(
        recipe.lower <= r.value <= recipe.upper
        for r in series.samples
        if r.value is not None
    )
    assert [r.day for r in series.samples] == [0, 1, 2, 3, 4, 5]


def test_intervention_effect_is_observable_and_not_silently_clamped():
    recipe = spec(jitter=0)
    assert [r.value for r in generate_lab_series(recipe).samples] == [
        50,
        51,
        None,
        51,
        52,
        53,
    ]
    with pytest.raises(ValueError, match="range"):
        spec(interventions=(LabIntervention(3, 100, "synthetic_effect"),))


@pytest.mark.parametrize(
    "kind,code",
    [
        ("range", "range_violation"),
        ("unit", "unit_mismatch"),
        ("time", "sample_time"),
        ("missingness", "missingness_mismatch"),
    ],
)
def test_negative_controls_fail_independent_validator_and_normal_split(kind, code):
    recipe = spec()
    series = generate_lab_series(recipe)
    bad = lab_negative_control(recipe, series, kind=kind)
    assert code in validate_lab_series(recipe, bad)
    assert "negative_control_split" in validate_lab_series(recipe, bad)
    assert code in validate_lab_series(recipe, replace(bad, split="synthetic"))


def test_wrong_trend_inside_valid_range_is_detected():
    recipe = spec()
    series = generate_lab_series(recipe)
    rows = list(series.samples)
    rows[3] = replace(rows[3], value=70)
    errors = validate_lab_series(recipe, replace(series, samples=tuple(rows)))
    assert "trend_or_intervention_violation" in errors
    assert "range_violation" not in errors


@pytest.mark.parametrize(
    "changes",
    [
        {"count": True},
        {"count": 10001},
        {"jitter": -1},
        {"baseline": float("nan")},
        {"lower": 100},
        {"unit": "private note"},
        {"missing_indices": (2, 2)},
        {"missing_indices": (10,)},
        {"max_step_change": 0.1},
    ],
)
def test_impossible_recipes_fail_with_controlled_errors(changes):
    with pytest.raises(ValueError) as exc:
        spec(**changes)
    assert "private note" not in str(exc.value)


def test_seed_and_recipe_binding_are_checked():
    recipe = spec()
    series = generate_lab_series(recipe)
    with pytest.raises(ValueError):
        generate_lab_series(recipe, seed=True)
    assert "recipe_mismatch" in validate_lab_series(
        recipe, replace(series, recipe_digest="b" * 64)
    )


def test_valid_narrow_range_preserves_declared_baseline_precision():
    recipe = spec(
        lower=0.1234567891,
        upper=0.1234567899,
        baseline=0.1234567892,
        missing_indices=(),
        interventions=(),
        count=1,
        jitter=0,
        trend_per_step=0,
    )
    record = generate_lab_series(recipe)
    assert record.samples[0].value == recipe.baseline
    assert validate_lab_series(recipe, record) == ()


def test_signed_zero_recipe_and_intervention_have_one_digest():
    first = spec(
        trend_per_step=-0.0,
        jitter=-0.0,
        interventions=(LabIntervention(0, -0.0, "synthetic"),),
    )
    second = spec(
        trend_per_step=0.0,
        jitter=0.0,
        interventions=(LabIntervention(0, 0.0, "synthetic"),),
    )
    assert first.digest == second.digest
    assert generate_lab_series(first) == generate_lab_series(second)
