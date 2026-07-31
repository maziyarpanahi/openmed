"""Unit tests for the DrugProt public clinical eval suite."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest

import openmed.eval.datasets.drugprot as drugprot_module
from openmed.cli import main_module
from openmed.core.labels import OTHER
from openmed.eval.datasets import (
    DRUGPROT,
    DRUGPROT_DOI,
    DRUGPROT_ENTITY_TO_CANONICAL,
    DrugProtRelationFixture,
    corpus_from_rows,
    license_for,
    load_drugprot_corpus,
    load_drugprot_ner_fixtures,
    load_drugprot_relation_fixtures,
    map_drugprot_entity_label,
)
from openmed.eval.suites import load_suite_fixtures, suite_metadata

FIXTURE_DIR = Path(__file__).parents[2] / "fixtures" / "drugprot_synthetic" / "training"


def test_drugprot_label_mapping_is_canonical() -> None:
    assert DRUGPROT_ENTITY_TO_CANONICAL == {
        "CHEMICAL": OTHER,
        "GENE": OTHER,
        "GENE-N": OTHER,
        "GENE-Y": OTHER,
    }
    assert map_drugprot_entity_label("CHEMICAL") == OTHER
    assert map_drugprot_entity_label("GENE-Y") == OTHER
    assert map_drugprot_entity_label("GENE-N") == OTHER


def test_drugprot_loader_builds_ner_fixtures_from_synthetic_tsv() -> None:
    fixtures = load_drugprot_ner_fixtures(FIXTURE_DIR)

    assert len(fixtures) == 1
    fixture = fixtures[0]
    assert fixture.fixture_id == "DP1"
    assert fixture.text == "Aspirin inhibits TP53 Metformin activates EGFR"
    assert fixture.metadata["dataset"] == DRUGPROT
    assert fixture.metadata["doi"] == DRUGPROT_DOI
    assert fixture.metadata["license"] == license_for(DRUGPROT).to_dict()
    assert [span.text for span in fixture.gold_spans] == [
        "Aspirin",
        "TP53",
        "Metformin",
        "EGFR",
    ]
    assert [span.label for span in fixture.gold_spans] == [OTHER, OTHER, OTHER, OTHER]
    assert fixture.gold_spans[0].metadata["drugprot_label"] == "CHEMICAL"
    assert fixture.gold_spans[1].metadata["entity_group"] == "GENE"


def test_drugprot_loader_builds_relation_fixtures_with_entity_refs() -> None:
    fixtures = load_drugprot_relation_fixtures(FIXTURE_DIR)

    assert len(fixtures) == 1
    fixture = fixtures[0]
    assert isinstance(fixture, DrugProtRelationFixture)
    assert fixture.metadata["task"] == "relation"
    assert [relation.to_tuple() for relation in fixture.relations] == [
        ("INHIBITOR", "T1", "T2"),
        ("ACTIVATOR", "T3", "T4"),
    ]
    first = fixture.relations[0]
    assert first.arg1.text == "Aspirin"
    assert first.arg1.entity_group == "CHEMICAL"
    assert first.arg2.text == "TP53"
    assert first.arg2.entity_group == "GENE"


def test_drugprot_loader_rejects_span_text_mismatches() -> None:
    with pytest.raises(ValueError, match="span text mismatch"):
        corpus_from_rows(
            [("DPX", "Aspirin inhibits TP53", "")],
            [("DPX", "T1", "CHEMICAL", "0", "7", "Ibuprofen")],
            [],
        )


def test_drugprot_loader_rejects_unknown_relation_types() -> None:
    with pytest.raises(ValueError, match="unknown DrugProt relation type"):
        corpus_from_rows(
            [("DPX", "Aspirin inhibits TP53", "")],
            [
                ("DPX", "T1", "CHEMICAL", "0", "7", "Aspirin"),
                ("DPX", "T2", "GENE", "17", "21", "TP53"),
            ],
            [("DPX", "NOT-A-RELATION", "Arg1:T1", "Arg2:T2")],
        )


def test_drugprot_loader_accepts_official_abstracts_typo_in_zip(tmp_path) -> None:
    archive_path = tmp_path / "drugprot-gs.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(
            FIXTURE_DIR / "drugprot_training_abstracts.tsv",
            "training/drugprot_training_abstracs.tsv",
        )
        archive.write(
            FIXTURE_DIR / "drugprot_training_entities.tsv",
            "training/drugprot_training_entities.tsv",
        )
        archive.write(
            FIXTURE_DIR / "drugprot_training_relations.tsv",
            "training/drugprot_training_relations.tsv",
        )

    corpus = load_drugprot_corpus(archive_path)

    assert len(corpus.records) == 1
    assert corpus.records[0].pmid == "DP1"
    assert len(corpus.records[0].relations) == 2


def test_drugprot_loader_uses_valid_cached_archive_without_downloading(
    monkeypatch,
    tmp_path,
) -> None:
    archive_path = _write_drugprot_zip(tmp_path / "drugprot-gs.zip")
    monkeypatch.setattr(drugprot_module, "DRUGPROT_ARCHIVE_MD5", _md5(archive_path))

    def fail_downloader(url, target):
        raise AssertionError(f"unexpected downloader call: {url} -> {target}")

    corpus = load_drugprot_corpus(cache_dir=tmp_path, downloader=fail_downloader)

    assert len(corpus.records) == 1
    assert corpus.source_path == str(archive_path)


def test_drugprot_loader_downloads_to_cache_when_archive_is_missing(
    monkeypatch,
    tmp_path,
) -> None:
    source_archive = _write_drugprot_zip(tmp_path / "source-drugprot.zip")
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(drugprot_module, "DRUGPROT_ARCHIVE_MD5", _md5(source_archive))
    calls = []

    def downloader(url, target):
        calls.append((url, target))
        shutil.copyfile(source_archive, target)
        return target

    corpus = load_drugprot_corpus(cache_dir=cache_dir, downloader=downloader)

    assert len(corpus.records) == 1
    assert calls == [
        (drugprot_module.DRUGPROT_DOWNLOAD_URL, cache_dir / "drugprot-gs.zip")
    ]
    assert (cache_dir / "drugprot-gs.zip").exists()


def test_drugprot_suite_registry_resolves_ner_and_relation_tasks() -> None:
    ner_fixtures = load_suite_fixtures(DRUGPROT, task="ner", path=FIXTURE_DIR)
    relation_fixtures = load_suite_fixtures(
        DRUGPROT,
        task="relation",
        path=FIXTURE_DIR,
    )
    relation_metadata = suite_metadata(DRUGPROT, task="relation")

    assert ner_fixtures[0].metadata["task"] == "ner"
    assert relation_fixtures[0].metadata["task"] == "relation"
    assert relation_metadata["doi"] == DRUGPROT_DOI
    assert relation_metadata["task"] == "relation"
    assert relation_metadata["license"] == "CC-BY-4.0"


def test_cli_benchmark_clinical_resolves_drugprot_relation_task(
    capsys,
) -> None:
    result = main_module.main(
        [
            "benchmark",
            "clinical",
            "--suite",
            DRUGPROT,
            "--task",
            "relation",
            "--input",
            str(FIXTURE_DIR),
        ]
    )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["suite"] == DRUGPROT
    assert output["task"] == "relation"
    assert output["fixture_count"] == 1
    assert output["relation_count"] == 2


def _write_drugprot_zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name in (
            "drugprot_training_abstracts.tsv",
            "drugprot_training_entities.tsv",
            "drugprot_training_relations.tsv",
        ):
            archive.write(FIXTURE_DIR / name, f"training/{name}")
    return path


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
