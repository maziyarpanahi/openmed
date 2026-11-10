"""Tests for the synthetic i2b2 corpus builder used by offline eval tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from openmed.core.labels import (
    AGE,
    DATE,
    EMAIL,
    ID_NUM,
    LOCATION,
    OCCUPATION,
    ORGANIZATION,
    PERSON,
    PHONE,
    SSN,
    STREET_ADDRESS,
    URL,
    USERNAME,
    ZIPCODE,
)
from openmed.eval.datasets.i2b2 import load_i2b2_deid
from tests.fixtures.eval.i2b2_synthetic import (
    SUPPORTED_SYNTHETIC_YEARS,
    SYNTHETIC_MARKER,
    write_i2b2_synthetic_corpus,
)

_EXPECTED_LABELS = {
    2006: [
        PERSON,
        DATE,
        AGE,
        STREET_ADDRESS,
        PHONE,
        ID_NUM,
        USERNAME,
        OCCUPATION,
    ],
    2014: [
        PERSON,
        DATE,
        AGE,
        LOCATION,
        ZIPCODE,
        EMAIL,
        URL,
        SSN,
        ORGANIZATION,
        PERSON,
    ],
}
_G1A_GROUP_LABELS = {
    "names": {PERSON},
    "dates": {DATE},
    "ages_over_89": {AGE},
    "addresses": {LOCATION, STREET_ADDRESS, ZIPCODE},
    "ids": {ID_NUM, SSN},
    "contacts": {PHONE, EMAIL, URL},
}


def test_builder_writes_only_under_the_given_directory(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "corpus"
    sibling = tmp_path / "keep.txt"
    sibling.write_text("untouched", encoding="utf-8")

    written = write_i2b2_synthetic_corpus(root)

    assert set(written) == set(SUPPORTED_SYNTHETIC_YEARS)
    for year, paths in written.items():
        assert paths
        assert {path.parent for path in paths} == {root / str(year)}
        for path in paths:
            assert path.is_file()
            assert path.is_relative_to(root)

    files = sorted(path for path in tmp_path.rglob("*") if path.is_file())
    expected = {sibling, *(path for paths in written.values() for path in paths)}
    assert set(files) == expected
    assert sibling.read_text(encoding="utf-8") == "untouched"


def test_loader_reads_generated_corpus_for_both_years(
    tmp_path: Path,
    subtests,
) -> None:
    corpus = tmp_path / "corpus"
    written = write_i2b2_synthetic_corpus(corpus)

    for year in SUPPORTED_SYNTHETIC_YEARS:
        with subtests.test(year=year):
            fixtures = load_i2b2_deid(corpus / str(year), year=year)

            assert len(fixtures) == len(written[year])
            assert {fixture.metadata["year"] for fixture in fixtures} == {year}
            labels = [span.label for fixture in fixtures for span in fixture.gold_spans]
            assert labels == _EXPECTED_LABELS[year]

            for fixture in fixtures:
                assert SYNTHETIC_MARKER in fixture.text.lower()
                for span in fixture.gold_spans:
                    assert fixture.text[span.start : span.end] == span.text

            for group, expected in _G1A_GROUP_LABELS.items():
                with subtests.test(year=year, group=group):
                    assert expected & set(labels)


def test_builder_rejects_a_non_directory_root(tmp_path: Path) -> None:
    root = tmp_path / "corpus.txt"
    root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a directory"):
        write_i2b2_synthetic_corpus(root)

    assert root.read_text(encoding="utf-8") == "not a directory"
