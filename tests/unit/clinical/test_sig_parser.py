"""Tests for the structured medication sig parser."""

from __future__ import annotations

import pytest

from openmed.clinical import (
    SIG_PARSER_ADVISORY,
    Sig,
    parse_sig,
    parse_sigs,
)

# --------------------------------------------------------------------------
# Full-sig parsing
# --------------------------------------------------------------------------


def test_canonical_sig_parses_all_components():
    sig = parse_sig("1 tab PO BID x7 days")

    assert sig["dose"] == 1.0
    assert sig["form"] == "tablet"
    assert sig["route"] == "oral"
    assert sig["frequency_per_day"] == 2.0
    assert sig["frequency_period"] is None
    assert sig["frequency_period_unit"] is None
    assert sig["duration_days"] == 7
    assert sig["as_needed"] is False
    assert sig["missing"] == []


def test_dose_with_measurement_unit():
    sig = parse_sig("500 mg PO TID x10 days")

    assert sig["dose"] == 500.0
    assert sig["unit"] == "mg"
    assert sig["form"] is None
    assert sig["route"] == "oral"
    assert sig["frequency_per_day"] == 3.0
    assert sig["duration_days"] == 10


def test_by_mouth_phrase_maps_to_oral():
    sig = parse_sig("1 tablet by mouth twice daily")

    assert sig["route"] == "oral"
    assert sig["frequency_per_day"] == 2.0
    assert sig["form"] == "tablet"


@pytest.mark.parametrize(
    ("abbr", "route"),
    [
        ("IV", "intravenous"),
        ("IM", "intramuscular"),
        ("SC", "subcutaneous"),
        ("SL", "sublingual"),
        ("PR", "rectal"),
    ],
)
def test_route_controlled_set(abbr, route):
    sig = parse_sig(f"1 unit {abbr} daily")
    assert sig["route"] == route


def test_prn_with_condition():
    sig = parse_sig("1 tab PO q6h PRN pain")

    assert sig["as_needed"] is True
    assert sig["condition"] == "pain"
    assert sig["frequency_per_day"] == 4.0
    assert sig["route"] == "oral"


def test_puffs_inhaler_sig():
    sig = parse_sig("take 2 puffs q4h PRN")

    assert sig["dose"] == 2.0
    assert sig["form"] == "puff"
    assert sig["route"] == "inhaled"
    assert sig["as_needed"] is True
    assert sig["frequency_per_day"] == 6.0
    assert sig["frequency_period"] == 4
    assert sig["frequency_period_unit"] == "h"


def test_interval_range_frequency_uses_shortest_interval():
    sig = parse_sig("take 2 puffs q4-6h PRN")

    assert sig["dose"] == 2.0
    assert sig["form"] == "puff"
    assert sig["route"] == "inhaled"
    assert sig["as_needed"] is True
    assert sig["frequency_per_day"] == 6.0
    assert sig["frequency_period"] == 4
    assert sig["frequency_period_unit"] == "h"
    assert sig["missing"] == []


# --------------------------------------------------------------------------
# Malformed / partial sigs flag missing components
# --------------------------------------------------------------------------


def test_partial_sig_flags_missing_dose():
    sig = parse_sig("PO daily")

    assert sig["dose"] is None
    assert sig["route"] == "oral"
    assert sig["frequency_per_day"] == 1.0
    assert sig["frequency_period"] == 1
    assert sig["frequency_period_unit"] == "d"
    assert "dose" in sig["missing"]


def test_empty_sig_flags_everything_missing():
    sig = parse_sig("")

    assert sig["dose"] is None
    assert sig["route"] is None
    assert sig["frequency_per_day"] is None
    assert set(sig["missing"]) >= {"dose", "route", "frequency"}


# --------------------------------------------------------------------------
# Frequency composes on the existing normalizer
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cue", "per_day"),
    [("qd", 1.0), ("BID", 2.0), ("TID", 3.0), ("QID", 4.0), ("q8h", 3.0)],
)
def test_frequency_reuses_normalizer(cue, per_day):
    sig = parse_sig(f"1 tab PO {cue}")
    assert sig["frequency_per_day"] == per_day


# --------------------------------------------------------------------------
# Span-attached variant
# --------------------------------------------------------------------------


def test_parse_sigs_over_medication_spans():
    text = "amoxicillin 500 mg PO TID; ibuprofen 1 tab PO q6h PRN pain"
    spans = [
        {
            "start": text.index("500 mg PO TID"),
            "end": text.index(";"),
            "label": "MEDICATION",
        },
        {
            "start": text.index("1 tab PO q6h PRN pain"),
            "end": len(text),
            "label": "MEDICATION",
        },
    ]

    results = parse_sigs(text, spans)

    assert len(results) == 2
    assert results[0]["span"] == (spans[0]["start"], spans[0]["end"])
    assert results[0]["sig"]["frequency_per_day"] == 3.0
    assert results[1]["sig"]["as_needed"] is True
    assert results[1]["sig"]["condition"] == "pain"


def test_parse_sig_is_deterministic():
    text = "2 tab PO BID x5 days PRN nausea"
    assert parse_sig(text) == parse_sig(text)


def test_advisory_exposed():
    assert isinstance(SIG_PARSER_ADVISORY, str) and SIG_PARSER_ADVISORY
    # Sig is a typed mapping usable as a return annotation.
    assert Sig is not None
