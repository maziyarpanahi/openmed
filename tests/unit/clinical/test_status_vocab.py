"""Tests for SDOH status vocabulary normalization."""

from __future__ import annotations

import pytest

from openmed.clinical import (
    STATUS_NORMALIZATION_ADVISORY,
    load_status_vocab,
    normalize_employment_status,
    normalize_living_status,
    normalize_substance_status,
)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("smoker", "current"),
        ("active use", "current"),
        ("drinks daily", "current"),
        ("former smoker", "former"),
        ("quit 2010", "former"),
        ("substance use disorder in remission", "former"),
        ("status post alcohol use disorder", "former"),
        ("denies alcohol", "never"),
        ("never smoker", "never"),
        ("non-smoker", "never"),
    ],
)
def test_normalize_substance_status_from_surface_cues(phrase, expected):
    assert normalize_substance_status(phrase) == expected


def test_normalize_substance_status_folds_negation_axis_to_never():
    assert normalize_substance_status("alcohol", negated=True) == "never"
    assert normalize_substance_status("alcohol", negated="negated") == "never"


def test_normalize_substance_status_folds_historical_current_cue_to_former():
    assert normalize_substance_status("smoker", temporality="historical") == "former"


@pytest.mark.parametrize("phrase", ["", None, "social history not discussed"])
def test_normalize_substance_status_returns_unknown_without_cue(phrase):
    assert normalize_substance_status(phrase) == "unknown"


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("retired", "retired"),
        ("retired teacher", "retired"),
        ("currently employed", "employed"),
        ("previously employed", "former"),
        ("unemployed", "unemployed"),
        ("on disability", "disabled"),
        ("student", "student"),
        ("never employed", "never"),
    ],
)
def test_normalize_employment_status_from_table(phrase, expected):
    assert normalize_employment_status(phrase) == expected


def test_normalize_employment_status_folds_historical_current_cue():
    assert normalize_employment_status("employed", temporality="historical") == "former"


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("homeless x2 years", "homeless"),
        ("formerly homeless", "former"),
        ("lives alone", "lives_alone"),
        ("lives with family", "lives_with_family"),
        ("assisted living", "assisted_living"),
        ("stable housing", "housed"),
        ("denies homelessness", "never"),
    ],
)
def test_normalize_living_status_from_table(phrase, expected):
    assert normalize_living_status(phrase) == expected


def test_status_vocab_includes_provenance_and_advisory_disclaimer():
    payload = load_status_vocab()

    assert payload["schema_version"] >= 1
    assert payload["provenance"]["task"] == "OM-325"
    assert "clinical decision" in payload["provenance"]["disclaimer"]
    assert "not a clinical decision rule" in STATUS_NORMALIZATION_ADVISORY


def test_status_vocab_contains_required_tables():
    payload = load_status_vocab()

    assert set(payload["vocabularies"]) == {"substance", "employment", "living"}
    assert "former" in payload["vocabularies"]["substance"]["statuses"]
    assert "retired" in payload["vocabularies"]["employment"]["statuses"]
    assert "homeless" in payload["vocabularies"]["living"]["statuses"]


# --- Duplicate cues after normalization (#3104) ---

_VOCAB_TEMPLATE = """schema_version: 1
provenance:
  task: OM-325
  source: synthetic test vocabulary
  disclaimer: Advisory normalization only; not a clinical decision rule.
defaults:
  unknown_status: unknown
vocabularies:
  substance:
    priority: [never, former, current]
    current_statuses: [current]
    axis_overrides:
      negated: never
      historical_current: former
    statuses:
      current:
        cues: [{current}]
      former:
        cues: [{former}]
      never:
        cues: [{never}]
"""


def _write_vocab(tmp_path, *, current, former='"quit"', never='"denies"'):
    path = tmp_path / "status_vocab.yaml"
    path.write_text(
        _VOCAB_TEMPLATE.format(current=current, former=former, never=never),
        encoding="utf-8",
    )
    return path


def test_synthetic_vocab_without_duplicates_loads(tmp_path):
    payload = load_status_vocab(_write_vocab(tmp_path, current='"smoker", "drinks"'))

    assert payload["vocabularies"]["substance"]["statuses"]["current"]["cues"] == [
        "smoker",
        "drinks",
    ]


@pytest.mark.parametrize(
    ("current_cue", "former_cue"),
    [
        pytest.param('"smoker"', '"smoker"', id="exact"),
        pytest.param('"smoker"', '"SMOKER"', id="case-only"),
        pytest.param('"non-smoker"', '"non‐smoker"', id="unicode-hyphen"),
        pytest.param('"smoker"', '"ｓｍｏｋｅｒ"', id="unicode-nfkc"),
        pytest.param('"never smoker"', '"never   smoker"', id="repeated-whitespace"),
        pytest.param('"never smoker"', '"never\tsmoker"', id="tab-whitespace"),
    ],
)
def test_same_normalized_cue_under_two_statuses_is_rejected(
    tmp_path, current_cue, former_cue
):
    path = _write_vocab(tmp_path, current=current_cue, former=former_cue)

    with pytest.raises(ValueError, match="two statuses") as excinfo:
        load_status_vocab(path)

    message = str(excinfo.value)
    assert "substance" in message
    # priority is [never, former, current], so former is seen first.
    assert "former (cue 1)" in message
    assert "current (cue 1)" in message


def test_duplicate_cue_within_one_status_is_rejected(tmp_path):
    path = _write_vocab(tmp_path, current='"smoker", "drinks", "Smoker"')

    with pytest.raises(
        ValueError, match=r"substance\.current repeats a cue"
    ) as excinfo:
        load_status_vocab(path)

    assert "cues 1 and 3" in str(excinfo.value)


def test_duplicate_cue_error_does_not_echo_cue_text(tmp_path):
    path = _write_vocab(
        tmp_path, current='"tobacco dependence"', former='"Tobacco  Dependence"'
    )

    with pytest.raises(ValueError) as excinfo:
        load_status_vocab(path)

    message = str(excinfo.value).casefold()
    assert "tobacco" not in message
    assert "dependence" not in message


def test_hyphen_and_space_variants_stay_distinct(tmp_path):
    # Normalization unifies hyphen code points but does not turn a hyphen into a
    # space, so these are two different cues and must both be accepted.
    payload = load_status_vocab(
        _write_vocab(tmp_path, current='"part-time", "part time"')
    )

    assert len(payload["vocabularies"]["substance"]["statuses"]["current"]["cues"]) == 2


def test_bundled_status_vocab_has_no_duplicate_cues():
    # Loading runs the duplicate check; a regression in the shipped table fails here.
    payload = load_status_vocab()

    assert set(payload["vocabularies"]) == {"substance", "employment", "living"}
