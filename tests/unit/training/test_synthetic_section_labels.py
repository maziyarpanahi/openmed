"""Focused tests for the offline synthetic section-label dataset builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.training.corpus import (
    CorpusManifestError,
    jsonl_records_hash,
    write_jsonl_records,
)
from openmed.training.synthetic.section_labels import (
    CANONICAL_DOCUMENT_TYPES,
    CANONICAL_SECTION_LABELS,
    DEFAULT_SECTION_EVAL_FIXTURE,
    SECTION_MANIFEST_SCHEMA_VERSION,
    SECTION_RECORD_SCHEMA_VERSION,
    SectionDatasetLeakageError,
    assert_no_eval_overlap,
    build_section_dataset,
    load_section_dataset,
    load_section_manifest,
    validate_section_labels,
)


def test_fixed_seed_writes_byte_identical_dataset_and_manifest(tmp_path: Path) -> None:
    first = build_section_dataset(23, 4, tmp_path / "first.jsonl")
    second = build_section_dataset(23, 4, tmp_path / "second.jsonl")

    assert first.dataset_path.read_bytes() == second.dataset_path.read_bytes()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.dataset_hash == second.dataset_hash
    assert first.manifest_hash == second.manifest_hash
    assert first.leakage_count == second.leakage_count == 0


def test_records_emit_aligned_bio_tags_canonical_boundaries_and_doc_type(
    tmp_path: Path,
) -> None:
    result = build_section_dataset(31, 6, tmp_path / "section.jsonl")
    rows = load_section_dataset(result.dataset_path)

    assert len(rows) == 6
    assert {row["doc_type"] for row in rows} == set(CANONICAL_DOCUMENT_TYPES)
    for row in rows:
        assert row["schema_version"] == SECTION_RECORD_SCHEMA_VERSION
        assert row["synthetic"] is True
        assert row["contains_real_phi"] is False
        assert row["restricted_data"] is False
        assert len(row["tokens"]) == len(row["section_tags"])
        assert row["labels"] == row["section_tags"]
        assert {
            tag.removeprefix("B-").removeprefix("I-")
            for tag in row["section_tags"]
            if tag != "O"
        } <= CANONICAL_SECTION_LABELS
        assert {
            section["label"] for section in row["sections"]
        } == CANONICAL_SECTION_LABELS


def test_invalid_section_labels_fail_closed() -> None:
    with pytest.raises(ValueError, match="unsupported section label"):
        validate_section_labels(("not_a_canonical_section",))


def test_leakage_guard_rejects_normalized_eval_text_overlap() -> None:
    eval_rows = [
        json.loads(line)
        for line in DEFAULT_SECTION_EVAL_FIXTURE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    held_out_text = next(row["text"] for row in eval_rows if row.get("text"))

    with pytest.raises(SectionDatasetLeakageError, match="record 0/fixture 0"):
        assert_no_eval_overlap(
            ({"id": "overlap", "text": f"  {held_out_text.upper()}  "},),
            eval_fixture_path=DEFAULT_SECTION_EVAL_FIXTURE,
        )


def test_manifest_is_hash_only_provenance_and_records_are_synthetic(
    tmp_path: Path,
) -> None:
    result = build_section_dataset(41, 1, tmp_path / "section.jsonl")
    manifest = load_section_manifest(result.manifest_path)

    assert manifest["schema_version"] == SECTION_MANIFEST_SCHEMA_VERSION
    assert manifest["record_count"] == 1
    assert manifest["synthetic"] is True
    assert manifest["contains_real_phi"] is False
    assert manifest["restricted_data"] is False
    assert manifest["leakage_check"]["overlap_count"] == 0
    assert "text" not in json.dumps(manifest, ensure_ascii=False).lower()


def test_corpus_jsonl_writer_uses_id_text_contract_and_stable_hash(
    tmp_path: Path,
) -> None:
    rows = ({"id": "synthetic-1", "text": "offline example", "labels": ["O"]},)
    path = tmp_path / "records.jsonl"

    written = write_jsonl_records(rows, path)

    assert written == rows
    assert jsonl_records_hash(rows) == jsonl_records_hash(written)
    assert json.loads(path.read_text(encoding="utf-8")) == rows[0]
    with pytest.raises(CorpusManifestError, match="missing required field"):
        write_jsonl_records(({"id": "missing-text"},), tmp_path / "invalid.jsonl")
