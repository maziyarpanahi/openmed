"""Synthetic parenthetical extraction and metadata-only export controls."""

import json

import pytest

from openmed.clinical.languages.abbreviation_mining import (
    AbbreviationCandidate,
    mine_abbreviations,
)

KEY = b"synthetic-test-key-not-for-production" * 2


def mine(*documents, **kwargs):
    return mine_abbreviations(documents, fingerprint_key=KEY, **kwargs)


def test_both_parenthetical_directions_group_same_candidate():
    rows = mine(
        "Blood pressure (BP) is a synthetic observation.",
        "BP (blood pressure) is a synthetic abbreviation.",
    )
    assert len(rows) == 1
    (row,) = rows
    assert (row.abbreviation, row.expansion) == ("bp", "blood pressure")
    assert row.count == 2
    assert row.frequency_band == "few"
    assert len(row.concordance_fingerprints) == 2


def test_multiple_expansions_remain_unadjudicated_alternatives():
    rows = mine("Blood pressure (BP).", "Bipolar pattern (BP).")
    assert len(rows) == 2
    assert {r.expansion for r in rows} == {"blood pressure", "bipolar pattern"}
    assert all(r.to_dict()["review_status"] == "unreviewed" for r in rows)


def test_unicode_clinical_initials_and_nfc_equivalence():
    rows = mine("Évaluation générale (ÉG).", "E\u0301valuation générale (ÉG).")
    # Decomposed accents are recognized only after caller normalization; do not alter offsets implicitly.
    assert rows[0].expansion == "évaluation générale"


def test_candidate_and_concordance_exports_contain_no_snippets():
    text = "Synthetic patient 999 has Blood pressure (BP) today."
    (row,) = mine(text)
    report = json.dumps(row.to_dict())
    assert "999" not in report
    assert "Blood pressure" not in report
    assert "blood pressure" not in report
    assert "bp" not in report
    assert "blood pressure" not in repr(row)


def test_input_order_does_not_change_report_and_key_changes_fingerprints():
    documents = ["Blood pressure (BP).", "BP (blood pressure)."]
    a = mine_abbreviations(documents, fingerprint_key=KEY)
    b = mine_abbreviations(reversed(documents), fingerprint_key=KEY)
    assert a == b
    other = mine_abbreviations(documents, fingerprint_key=b"x" * 32)
    assert a[0].candidate_id != other[0].candidate_id
    assert a[0].concordance_fingerprints != other[0].concordance_fingerprints


@pytest.mark.parametrize(
    "count,band", [(1, "once"), (2, "few"), (5, "repeated"), (20, "frequent")]
)
def test_frequency_bands(count, band):
    (row,) = mine(*["Blood pressure (BP)."] * count)
    assert row.frequency_band == band


@pytest.mark.parametrize(
    "text",
    [
        "Blood pressure (ZZ).",
        "BP (unrelated expansion).",
        "(BP)",
        "Blood pressure. (BP)",
        "Synthetic Name (SN-1234).",
    ],
)
def test_nonmatching_and_identifier_like_controls(text):
    assert mine(text) == ()


def test_limits_are_explicit_and_bounded_iterators_stop():
    consumed = []

    def infinite_documents():
        while True:
            consumed.append(1)
            yield "Blood pressure (BP)."

    with pytest.raises(ValueError, match="document limit"):
        mine_abbreviations(infinite_documents(), fingerprint_key=KEY, max_documents=2)
    assert len(consumed) == 3
    with pytest.raises(ValueError, match="character limit"):
        mine("Blood pressure (BP).", max_characters=2)
    with pytest.raises(ValueError, match="candidate limit"):
        mine("Blood pressure (BP). Magnetic resonance imaging (MRI).", max_candidates=1)


def test_key_and_malformed_documents_have_controlled_errors():
    with pytest.raises(ValueError, match="key"):
        mine_abbreviations([], fingerprint_key=b"short")
    with pytest.raises(ValueError, match="document"):
        mine_abbreviations([None], fingerprint_key=KEY)


def test_public_candidate_constructor_rejects_raw_export_fields():
    valid = dict(
        abbreviation="bp",
        expansion="blood pressure",
        candidate_id="a" * 64,
        count=1,
        concordance_fingerprints=["b" * 64],
    )
    candidate = AbbreviationCandidate(**valid)
    valid["concordance_fingerprints"].clear()
    assert candidate.concordance_fingerprints == ("b" * 64,)
    for overrides in [
        {"candidate_id": "raw source sentinel"},
        {"count": -1},
        {"count": True},
        {"concordance_fingerprints": ["raw snippet"]},
        {"concordance_fingerprints": []},
        {"concordance_fingerprints": ["b" * 64, "b" * 64]},
        {"abbreviation": ""},
    ]:
        with pytest.raises(ValueError):
            AbbreviationCandidate(
                **(valid | {"concordance_fingerprints": ("b" * 64,)} | overrides)
            )


def test_public_candidate_constructor_bounds_infinite_metadata():
    import itertools

    with pytest.raises(ValueError):
        AbbreviationCandidate(
            "bp", "blood pressure", "a" * 64, 100000, itertools.repeat("b" * 64)
        )
