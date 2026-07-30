from __future__ import annotations

import json
import re

import pytest

from openmed.training import (
    DAPT_CORPUS_MANIFEST_PATH,
    PRESET_BY_MODE,
    PUBLIC_DAPT_SOURCES,
    GatedCorpusAccessError,
    MimicIIIDuaSource,
    RecordPassageSource,
    arxiv_qbio_source,
    assemble_dapt_corpus,
    assert_manifest_has_no_raw_text,
    corpus_manifest_hash,
    load_corpus_manifest,
    load_preset,
    pubmed_abstract_source,
)


def test_assemble_manifest_deduplicates_and_reports_rate(tmp_path):
    source = RecordPassageSource(
        name="pubmed",
        license="CC0-1.0",
        records=[
            {
                "id": "pmid-1",
                "text": "Public biomedical passage for privacy model adaptation.",
            },
            {
                "id": "pmid-2",
                "text": "  public biomedical passage for privacy model adaptation.  ",
            },
            {
                "id": "pmid-3",
                "text": "Distinct arXiv q-bio passage for terminology coverage.",
            },
        ],
    )

    result = assemble_dapt_corpus([source], tmp_path / "dapt_corpus.jsonl")

    assert result.input_count == 3
    assert result.passage_count == 2
    assert result.duplicate_count == 1
    assert result.dedup_rate == pytest.approx(1 / 3)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", result.corpus_manifest_hash)
    assert result.manifest_path.read_text(encoding="utf-8").count("\n") == 2

    rows = load_corpus_manifest(result.manifest_path)
    assert {row["source_id"] for row in rows} == {"pmid-1", "pmid-3"}
    for row in rows:
        assert row["source"] == "pubmed"
        assert row["license"] == "CC0-1.0"
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", row["sha256"])
        assert row["token_count"] > 0
        assert "text" not in row


def test_manifest_hash_is_stable_for_identical_inputs(tmp_path):
    records = [
        {
            "pmid": "pmid-100",
            "abstract": "A public abstract passage supports reproducible DAPT.",
        }
    ]
    source = pubmed_abstract_source(records, license="CC0-1.0")

    first = assemble_dapt_corpus([source], tmp_path / "first.jsonl")
    second = assemble_dapt_corpus([source], tmp_path / "second.jsonl")

    assert first.corpus_manifest_hash == second.corpus_manifest_hash
    assert first.manifest_path.read_text() == second.manifest_path.read_text()
    assert corpus_manifest_hash(first.manifest_path) == first.corpus_manifest_hash
    assert corpus_manifest_hash(first.rows) == first.corpus_manifest_hash


def test_source_helpers_map_public_records(tmp_path):
    source = arxiv_qbio_source(
        [
            {
                "arxiv_id": "2401.00001",
                "abstract": "Public q-bio text can be assembled without network access.",
                "category": "q-bio.QM",
                "url": "https://arxiv.org/abs/2401.00001",
            }
        ],
        license="arXiv public metadata",
    )

    result = assemble_dapt_corpus([source], tmp_path / "manifest.jsonl")

    row = result.rows[0]
    assert row["source"] == "arxiv-q-bio"
    assert row["source_id"] == "2401.00001"
    assert row["metadata"] == {"category": "q-bio.QM"}
    assert row["provenance"] == "https://arxiv.org/abs/2401.00001"


def test_mimic_iii_source_refuses_without_credentialed_path():
    with pytest.raises(GatedCorpusAccessError, match="DUA-gated"):
        list(MimicIIIDuaSource().iter_passages())


def test_mimic_iii_source_requires_existing_local_export(tmp_path):
    missing = tmp_path / "missing"

    with pytest.raises(GatedCorpusAccessError, match="credentialed_path"):
        list(MimicIIIDuaSource(credentialed_path=missing).iter_passages())


def test_mimic_iii_source_reads_only_user_supplied_credentialed_export(tmp_path):
    export_path = tmp_path / "passages.jsonl"
    export_path.write_text(
        json.dumps(
            {
                "note_id": "note-hash-1",
                "text": "Credentialed local note text is hashed before manifest output.",
                "subject_id_hash": "subject-hash",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = assemble_dapt_corpus(
        [MimicIIIDuaSource(credentialed_path=export_path)],
        tmp_path / "mimic_manifest.jsonl",
    )

    assert result.passage_count == 1
    row = result.rows[0]
    assert row["source"] == "mimic-iii"
    assert row["source_id"] == "note-hash-1"
    assert row["license"] == "PhysioNet MIMIC-III credentialed DUA"
    assert row["metadata"] == {"subject_id_hash": "subject-hash"}
    assert "text" not in row


def test_committed_manifest_is_hash_only_public_snapshot():
    assert_manifest_has_no_raw_text(DAPT_CORPUS_MANIFEST_PATH)
    rows = load_corpus_manifest(DAPT_CORPUS_MANIFEST_PATH)

    assert rows
    assert {row["source"] for row in rows} <= PUBLIC_DAPT_SOURCES
    for row in rows:
        assert row["schema_version"] == "openmed.training.dapt_corpus.v1"
        assert set(row) >= {"source", "license", "sha256", "token_count"}
        assert "mimic" not in row["source"].casefold()
        assert "text" not in row


def test_committed_presets_pin_the_dapt_corpus_manifest_hash():
    expected_ref = corpus_manifest_hash(DAPT_CORPUS_MANIFEST_PATH)

    for mode in ("A", "B"):
        assert load_preset(mode).dapt.corpus_ref == expected_ref

    assert PRESET_BY_MODE["C"] == "large_teacher"
