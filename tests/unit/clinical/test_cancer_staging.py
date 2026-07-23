"""Tests for the deterministic TNM/AJCC cancer-staging extractor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.clinical import TNM_STAGING_ADVISORY, parse_tnm

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "cancer_staging.jsonl"
)

STAGE_FIELDS = (
    "basis",
    "t",
    "t_subcategory",
    "n",
    "n_subcategory",
    "m",
    "m_subcategory",
    "confidence",
    "unparsed",
)


def _load_fixture() -> list[dict]:
    with FIXTURE.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------------------
# Fixture-driven gold rows
# ---------------------------------------------------------------------------


def test_fixture_rows_are_synthetic():
    rows = _load_fixture()
    assert rows
    assert all(row["metadata"]["synthetic"] is True for row in rows)


@pytest.mark.parametrize("row", _load_fixture())
def test_parser_matches_gold_row(row):
    parsed = parse_tnm(row["text"])
    for field in STAGE_FIELDS:
        assert parsed[field] == row["gold"][field], field


# ---------------------------------------------------------------------------
# Basis prefixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "basis"),
    [
        ("cT2 N0 M0", "c"),
        ("pT2 N0 M0", "p"),
        ("ycT2 N0 M0", "yc"),
        ("ypT2 N0 M0", "yp"),
        ("rT2 N0 M0", "r"),
        ("aT2 N0 M0", "a"),
    ],
)
def test_each_staging_basis_prefix_is_recognized(text, basis):
    parsed = parse_tnm(text)
    assert parsed["basis"] == basis
    assert parsed["t"] == "T2"
    assert parsed["confidence"] == "high"


def test_no_prefix_yields_no_basis():
    parsed = parse_tnm("T2 N0 M0")
    assert parsed["basis"] is None
    assert parsed["confidence"] == "high"


def test_post_therapy_prefix_attaches_only_where_written():
    # yp is written on T; N and M carry no prefix -> single shared basis, high.
    parsed = parse_tnm("ypT3 N1 M0")
    assert parsed["basis"] == "yp"
    assert (parsed["t"], parsed["n"], parsed["m"]) == ("T3", "N1", "M0")
    assert parsed["confidence"] == "high"


# ---------------------------------------------------------------------------
# Categories, subcategories, and edge tokens
# ---------------------------------------------------------------------------


def test_subcategories_are_split_from_categories():
    parsed = parse_tnm("pT1a N2b M1c")
    assert (parsed["t"], parsed["t_subcategory"]) == ("T1", "a")
    assert (parsed["n"], parsed["n_subcategory"]) == ("N2", "b")
    assert (parsed["m"], parsed["m_subcategory"]) == ("M1", "c")


def test_microinvasion_subcategory():
    parsed = parse_tnm("T1mi N0 M0")
    assert (parsed["t"], parsed["t_subcategory"]) == ("T1", "mi")


def test_in_situ_and_zero_and_x_categories():
    assert parse_tnm("Tis N0 M0")["t"] == "Tis"
    assert parse_tnm("T0 N0 M0")["t"] == "T0"
    x = parse_tnm("TX NX MX")
    assert (x["t"], x["n"], x["m"]) == ("TX", "NX", "MX")


def test_parenthetical_qualifiers_become_subcategories():
    assert parse_tnm("Tis(DCIS)")["t_subcategory"] == "DCIS"
    assert parse_tnm("pN0(i+)")["n_subcategory"] == "i+"


def test_contiguous_notation_without_spaces_is_split():
    parsed = parse_tnm("pT2N1M0")
    assert (parsed["t"], parsed["n"], parsed["m"]) == ("T2", "N1", "M0")
    assert parsed["basis"] == "p"


def test_staging_embedded_in_prose_is_found():
    parsed = parse_tnm("The resection specimen was staged pT3b N2 M0 overall.")
    assert (parsed["basis"], parsed["t"], parsed["t_subcategory"]) == ("p", "T3", "b")
    assert (parsed["n"], parsed["m"]) == ("N2", "M0")


def test_hyphen_separated_categories_all_parse():
    parsed = parse_tnm("pT2-N1-M0")
    assert (parsed["basis"], parsed["t"], parsed["n"], parsed["m"]) == (
        "p",
        "T2",
        "N1",
        "M0",
    )
    assert parsed["confidence"] == "high"


def test_all_caps_in_situ_is_recognized():
    parsed = parse_tnm("TIS N0 M0")
    assert parsed["t"] == "Tis"
    assert parsed["confidence"] == "high"


def test_uppercase_subcategory_is_normalized_not_dropped():
    parsed = parse_tnm("T1A N0 M0")
    assert (parsed["t"], parsed["t_subcategory"]) == ("T1", "a")


def test_letter_and_parenthetical_subcategory_both_preserved():
    parsed = parse_tnm("pN1a(sn)")
    assert parsed["n"] == "N1"
    assert parsed["n_subcategory"] == "a(sn)"


# ---------------------------------------------------------------------------
# Never-coerce: unrecognized tokens surfaced with reasons
# ---------------------------------------------------------------------------


def test_out_of_range_category_is_surfaced_not_coerced():
    parsed = parse_tnm("T5 N1 M0")
    assert parsed["t"] is None  # T5 is not a valid AJCC T category
    assert parsed["n"] == "N1" and parsed["m"] == "M0"
    assert parsed["confidence"] == "low"
    assert parsed["unparsed"] == [
        {"token": "T5", "reason": "unrecognized T category 'T5'"}
    ]


def test_multi_digit_value_is_surfaced_not_truncated():
    # "T10" must not become a confident "T1"; it is surfaced whole.
    parsed = parse_tnm("T10 N0 M0")
    assert parsed["t"] is None
    assert parsed["confidence"] == "low"
    assert parsed["unparsed"] == [
        {"token": "T10", "reason": "unrecognized T category 'T10'"}
    ]


def test_non_ajcc_subcategory_is_surfaced_not_coerced():
    # 'd' is not a valid N subcategory (a/b/c); surface it, do not keep "N2".
    parsed = parse_tnm("pT2 N2d M0")
    assert parsed["t"] == "T2" and parsed["m"] == "M0"
    assert parsed["n"] is None
    assert parsed["confidence"] == "low"
    assert parsed["unparsed"] == [
        {"token": "N2d", "reason": "unrecognized N category 'N2d'"}
    ]


def test_secondary_staging_expression_is_surfaced():
    parsed = parse_tnm("pT2 N1 M0, ypT3a")
    assert (parsed["basis"], parsed["t"], parsed["n"], parsed["m"]) == (
        "p",
        "T2",
        "N1",
        "M0",
    )
    assert parsed["confidence"] == "low"
    assert parsed["unparsed"] == [
        {
            "token": "ypT3a",
            "reason": "additional T category not used for the primary stage",
        }
    ]


def test_invalid_n_and_m_values_are_surfaced():
    n4 = parse_tnm("N4")
    assert n4["n"] is None
    assert n4["unparsed"] == [{"token": "N4", "reason": "unrecognized N category 'N4'"}]
    m2 = parse_tnm("M2")
    assert m2["m"] is None
    assert m2["unparsed"] == [{"token": "M2", "reason": "unrecognized M category 'M2'"}]


# ---------------------------------------------------------------------------
# Confidence flagging
# ---------------------------------------------------------------------------


def test_clean_stage_is_high_confidence():
    assert parse_tnm("pT2 N0 M0")["confidence"] == "high"


def test_mixed_prefixes_lower_confidence():
    parsed = parse_tnm("pT2 cN1 M0")
    assert parsed["basis"] == "p"
    assert parsed["confidence"] == "low"


def test_ambiguous_bare_y_prefix_lowers_confidence():
    parsed = parse_tnm("yT2 N0 M0")
    assert parsed["t"] == "T2"
    assert parsed["basis"] is None  # bare 'y' is not a full basis
    assert parsed["confidence"] == "low"


def test_nothing_recognized_is_low_confidence():
    parsed = parse_tnm("no staging notation here")
    assert (parsed["t"], parsed["n"], parsed["m"]) == (None, None, None)
    assert parsed["confidence"] == "low"
    assert parsed["unparsed"] == []


def test_empty_input_returns_empty_stage_with_advisory():
    parsed = parse_tnm("")
    assert parsed["t"] is None and parsed["n"] is None and parsed["m"] is None
    assert parsed["basis"] is None
    assert parsed["advisory"] == TNM_STAGING_ADVISORY


# ---------------------------------------------------------------------------
# Advisory
# ---------------------------------------------------------------------------


def test_advisory_is_attached_and_disclaims_autonomy():
    parsed = parse_tnm("pT2 N1 M0")
    assert parsed["advisory"] == TNM_STAGING_ADVISORY
    lowered = TNM_STAGING_ADVISORY.lower()
    assert "no stage-group derivation" in lowered
    assert "not a substitute" in lowered
