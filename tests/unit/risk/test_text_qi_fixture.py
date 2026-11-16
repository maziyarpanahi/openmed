"""Integrity checks for synthetic free-text quasi-identifier notes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from openmed.core.labels import is_recognized_label, normalize_label

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "risk"
    / "text_quasi_identifiers_v1.jsonl"
)
FIELDS = {
    "AGE": "age",
    "DATE": "date",
    "ZIPCODE": "zipcode",
    "LOCATION": "location",
    "OCCUPATION": "occupation",
}
GENERALIZATION = {
    "AGE": "ten_year_band_or_90_plus",
    "DATE": "year",
    "ZIPCODE": "zip3_star_star",
    "LOCATION": "region_after_comma",
    "OCCUPATION": "broad_category",
}
FORBIDDEN_MARKERS = ("cpt", "dua", "i2b2", "mimic", "n2c2", "snomed", "umls")


def _load() -> tuple[dict, list[dict]]:
    rows = [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows and rows[0]["kind"] == "meta"
    return rows[0], rows[1:]


def _generalized(label: str, surface: str, meta: dict) -> str:
    if label == "AGE":
        assert surface.isdigit(), surface
        age = int(surface)
        assert 0 <= age <= 130
        if age >= 90:
            return "90+"
        lower = (age // 10) * 10
        return f"{lower}-{lower + 9}"
    if label == "DATE":
        year, month, day = surface.split("-")
        assert len(year) == 4 and year.isdigit()
        assert month.isdigit() and 1 <= int(month) <= 12
        assert day.isdigit() and 1 <= int(day) <= 31
        return year
    if label == "ZIPCODE":
        assert len(surface) == 5 and surface.isdigit()
        prefix = surface[:3]
        if prefix in meta["small_zip3"]:
            return "000**"
        return f"{prefix}**"
    if label == "LOCATION":
        place, region = surface.split(",")
        assert place.strip() and region.strip()
        return region.strip()
    categories = meta["occupation_categories"]
    assert surface.casefold() in categories
    return categories[surface.casefold()]


def _check_span(text: str, span: dict, expected: dict, meta: dict) -> None:
    surface = text[span["start"] : span["end"]]
    assert surface == span["text"], "span offsets do not match text"
    label = span["label"]
    assert is_recognized_label(label) and label == normalize_label(label), (
        "label is not canonical"
    )
    generalized = _generalized(label, surface, meta)
    assert span["generalized"] == generalized, (
        "generalized value does not match the fixture rule"
    )
    assert expected[FIELDS[label]] == generalized


def test_text_quasi_identifier_fixture_is_synthetic_and_aligned() -> None:
    meta, notes = _load()
    assert meta["version"] == 1
    assert meta["suite"] == "text_quasi_identifiers_v1"
    assert meta["synthetic"] is True
    assert meta["generalization"] == GENERALIZATION
    assert meta["small_zip3"] == ["000"]
    assert 10 <= len(notes) <= 20
    blob = FIXTURE.read_text(encoding="utf-8").casefold()
    for marker in FORBIDDEN_MARKERS:
        assert re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", blob) is None

    for note in notes:
        assert note["kind"] == "note"
        assert note["synthetic"] is True
        assert "synthetic" in note["text"].casefold()
        assert list(note["expected"]) == list(FIELDS.values())
        assert [span["label"] for span in note["spans"]] == list(FIELDS)
        for span in note["spans"]:
            _check_span(note["text"], span, note["expected"], meta)


def test_seeded_offset_mismatch_fails() -> None:
    meta, notes = _load()
    note = notes[0]
    span = dict(note["spans"][0])
    span["end"] = span["start"]
    with pytest.raises(AssertionError, match="span offsets"):
        _check_span(note["text"], span, note["expected"], meta)


def test_unrecognized_label_fails() -> None:
    meta, notes = _load()
    note = notes[0]
    span = dict(note["spans"][0])
    span["label"] = "NOT_A_REAL_LABEL"
    with pytest.raises(AssertionError, match="canonical"):
        _check_span(note["text"], span, note["expected"], meta)


def test_seeded_generalization_mismatch_fails() -> None:
    meta, notes = _load()
    note = notes[0]
    span = dict(note["spans"][0])
    span["generalized"] = "not-a-band"
    with pytest.raises(AssertionError, match="generalized"):
        _check_span(note["text"], span, note["expected"], meta)
