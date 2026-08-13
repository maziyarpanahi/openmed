"""Unit tests for the eval-only n2c2 2018 de-identification loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from openmed.core.labels import (
    AGE,
    CANONICAL_LABELS,
    DATE,
    ID_NUM,
    LOCATION,
    OCCUPATION,
    ORGANIZATION,
    PERSON,
    PHONE,
)
from openmed.eval.datasets import n2c2_2018
from openmed.eval.datasets.n2c2_2018 import (
    N2C2,
    N2C2_PATH_ENV,
    N2C2_PHI_TAG_TO_CANONICAL,
    N2C2_PHI_TAGS,
    N2C2_SPECIFIC_PHI_TAGS,
    N2C2CredentialRequired,
    load_n2c2_2018_deid,
    map_n2c2_phi_tag,
)
from openmed.eval.suites import DEFAULT_SUITES, load_suite_fixtures, suite_metadata


def test_load_n2c2_2018_deid_parses_synthetic_brat_fixture(
    tmp_path: Path,
) -> None:
    source = tmp_path / "credentialed"
    source.mkdir()
    text = (
        "Synthetic Jordan Example was 47 on 2099-01-02 at Ward-7 for Unit-9; "
        "call 555-0101 about MRN-0001 as a nurse."
    )
    annotations = _brat_annotations(
        text,
        [
            ("NAME", "Jordan Example"),
            ("AGE", "47"),
            ("DATE", "2099-01-02"),
            ("LOCATION", "Ward-7"),
            ("ORGANIZATION", "Unit-9"),
            ("PHONE", "555-0101"),
            ("MEDICALRECORD", "MRN-0001"),
            ("PROFESSION", "nurse"),
        ],
    )
    text_path = source / "synthetic-001.txt"
    text_path.write_text(text, encoding="utf-8")
    text_path.with_suffix(".ann").write_text(annotations, encoding="utf-8")

    fixtures = load_n2c2_2018_deid(source)

    assert len(fixtures) == 1
    fixture = fixtures[0]
    assert fixture.fixture_id.startswith("n2c2-2018-")
    assert fixture.metadata["dua"] == "n2c2/DBMI DUA"
    assert fixture.metadata["annotation_format"] == "brat"
    assert fixture.metadata["track"] == "track_1"
    assert fixture.metadata["year"] == 2018
    assert fixture.metadata["source_path_hash"] != text_path.name
    assert [span.label for span in fixture.gold_spans] == [
        PERSON,
        AGE,
        DATE,
        LOCATION,
        ORGANIZATION,
        PHONE,
        ID_NUM,
        OCCUPATION,
    ]
    for span in fixture.gold_spans:
        assert fixture.text[span.start : span.end] == span.text
        assert span.metadata["n2c2_category"] in N2C2_PHI_TAGS


def test_n2c2_loader_refuses_unconfigured_empty_and_repo_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(N2C2_PATH_ENV, raising=False)

    with pytest.raises(N2C2CredentialRequired, match="DBMI DUA"):
        load_n2c2_2018_deid()

    empty_source = tmp_path / "empty"
    empty_source.mkdir()
    with pytest.raises(N2C2CredentialRequired, match="no paired"):
        load_n2c2_2018_deid(empty_source)

    def fail_if_scanned(root: Path):  # pragma: no cover - should not be called
        raise AssertionError(f"repo path was scanned: {root}")

    monkeypatch.setattr(n2c2_2018, "_iter_document_pairs", fail_if_scanned)
    repo_root = Path(__file__).resolve().parents[3]
    with pytest.raises(N2C2CredentialRequired, match="repository tree"):
        load_n2c2_2018_deid(repo_root)


def test_n2c2_category_map_is_total_canonical_and_strict() -> None:
    assert set(N2C2_PHI_TAG_TO_CANONICAL) == set(N2C2_PHI_TAGS)
    assert set(N2C2_PHI_TAG_TO_CANONICAL.values()) <= CANONICAL_LABELS
    assert N2C2_SPECIFIC_PHI_TAGS

    for tag in N2C2_PHI_TAGS:
        assert map_n2c2_phi_tag(tag) == N2C2_PHI_TAG_TO_CANONICAL[tag]

    assert map_n2c2_phi_tag("medical record number") == ID_NUM
    assert map_n2c2_phi_tag("patient") == PERSON
    assert map_n2c2_phi_tag("IP address") == "IP_ADDRESS"

    with pytest.raises(ValueError, match="unknown n2c2 PHI tag"):
        map_n2c2_phi_tag("MASCOT")


def test_n2c2_suite_registry_loads_configured_path(tmp_path: Path) -> None:
    source = tmp_path / "credentialed"
    source.mkdir()
    text = "Synthetic Jordan Example visited Unit-9."
    text_path = source / "synthetic-002.txt"
    text_path.write_text(text, encoding="utf-8")
    start = text.index("Jordan Example")
    end = start + len("Jordan Example")
    text_path.with_suffix(".ann").write_text(
        f"T1\tNAME {start} {end}\tJordan Example\n",
        encoding="utf-8",
    )

    fixtures = load_suite_fixtures(N2C2, path=source)
    metadata = suite_metadata(N2C2)

    assert N2C2 in DEFAULT_SUITES
    assert len(fixtures) == 1
    assert fixtures[0].gold_spans[0].label == PERSON
    assert metadata["suite"] == N2C2
    assert metadata["path_config"] == N2C2_PATH_ENV
    assert metadata["label_mapping"]["NAME"] == PERSON


def _brat_annotations(text: str, spans: list[tuple[str, str]]) -> str:
    lines = []
    for index, (label, surface) in enumerate(spans, start=1):
        start = text.index(surface)
        end = start + len(surface)
        lines.append(f"T{index}\t{label} {start} {end}\t{surface}")
    return "\n".join(lines) + "\n"
