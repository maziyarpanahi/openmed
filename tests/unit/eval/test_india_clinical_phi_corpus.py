"""Tests for the synthetic India clinical de-identification corpus."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from openmed.core.labels import CANONICAL_LABELS, ID_NUM, PERSON
from openmed.core.pii_i18n import validate_aadhaar
from openmed.eval.datasets import (
    INDIA_CLINICAL_PHI_CORPUS_ID,
    INDIA_CLINICAL_PHI_SCHEMA_VERSION,
    IndiaClinicalPHICorpus,
    load_clinical_phi_manifest,
    load_india_clinical_phi_corpus,
    resolve_clinical_phi_source,
    validate_india_clinical_phi_manifest,
)
from openmed.eval.golden import list_fixture_paths

FIXTURE_DIR = Path("openmed/eval/golden/fixtures/i18n")
MANIFEST_PATH = FIXTURE_DIR / "india_clinical_manifest.json"
FIXTURE_PATH = FIXTURE_DIR / "india_clinical.jsonl"

REQUIRED_IDENTIFIER_TYPES = {
    "person_name",
    "abha",
    "aadhaar",
    "pan",
    "indian_phone",
    "street_address",
    "pin_code",
}


def test_india_corpus_loads_deterministically_with_required_coverage() -> None:
    corpus = load_india_clinical_phi_corpus()

    assert corpus == load_india_clinical_phi_corpus()
    assert corpus.manifest.corpus_id == INDIA_CLINICAL_PHI_CORPUS_ID
    assert corpus.manifest.schema_version == INDIA_CLINICAL_PHI_SCHEMA_VERSION
    assert corpus.manifest.synthetic_only is True
    assert corpus.manifest.contains_real_phi is False
    assert corpus.manifest.contains_dua_data is False
    assert [record.document_id for record in corpus.records] == [
        "india-doc-001",
        "india-doc-002",
        "india-doc-003",
    ]

    scripts = {script for record in corpus.records for script in record.scripts}
    assert {"Latin", "Devanagari", "Tamil"} <= scripts
    assert all(record.metadata["synthetic"] is True for record in corpus.records)
    assert all(len(record.languages) >= 2 for record in corpus.records)

    identifier_types = {
        span.metadata["identifier_type"]
        for record in corpus.records
        for span in record.gold_spans
    }
    assert identifier_types == REQUIRED_IDENTIFIER_TYPES
    assert {term for record in corpus.records for term in record.clinical_terms} == {
        "Ayurveda",
        "आयुष",
        "योग",
        "சித்தா",
    }


def test_india_corpus_spans_round_trip_and_aadhaar_is_checksum_valid() -> None:
    corpus = load_india_clinical_phi_corpus()
    aadhaar_values: set[str] = set()

    for record in corpus.records:
        assert record.gold_spans
        for span in record.gold_spans:
            assert span.label in CANONICAL_LABELS
            assert record.text[span.start : span.end] == span.text
            assert span.metadata["synthetic"] is True
            assert span.metadata["span_id"]
            if span.metadata["identifier_type"] == "aadhaar":
                assert span.label == ID_NUM
                assert validate_aadhaar(span.text)
                aadhaar_values.add(span.text)

    assert aadhaar_values == {"655227804685"}


def test_india_corpus_declares_one_identity_across_three_scripts() -> None:
    corpus = load_india_clinical_phi_corpus()
    identity = corpus.manifest.cross_document_identities[0]

    assert identity.group_id == "india-person-001"
    assert {alias.document_id for alias in identity.aliases} == {
        "india-doc-001",
        "india-doc-002",
        "india-doc-003",
    }
    assert {alias.script for alias in identity.aliases} == {
        "Latin",
        "Devanagari",
        "Tamil",
    }

    person_spans = [
        span
        for record in corpus.records
        for span in record.gold_spans
        if span.label == PERSON
    ]
    assert {span.text for span in person_spans} == {
        "Aarav Sharma",
        "आरव शर्मा",
        "ஆரவ் சர்மா",
    }
    assert {span.metadata["identity_group"] for span in person_spans} == {
        identity.group_id
    }


def test_india_records_expose_harness_compatible_normalized_metadata() -> None:
    record = load_india_clinical_phi_corpus().records[0]

    fixture = record.to_benchmark_fixture()

    assert fixture.fixture_id == record.fixture_id
    assert fixture.text == record.text
    assert fixture.gold_spans == record.gold_spans
    assert fixture.metadata["document_id"] == record.document_id
    assert fixture.metadata["scripts"] == list(record.scripts)
    assert fixture.metadata["clinical_terms"] == list(record.clinical_terms)


def test_clinical_phi_manifest_registers_dedicated_india_source() -> None:
    source = load_clinical_phi_manifest().source("india_synthetic_clinical_deid")

    assert source.dataset == INDIA_CLINICAL_PHI_CORPUS_ID
    assert source.synthetic is True
    assert source.access == "committed_synthetic"
    resolved = resolve_clinical_phi_source(source.source_id)
    assert isinstance(resolved, IndiaClinicalPHICorpus)
    assert resolved.records


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"synthetic_only": False}, "synthetic-only"),
        ({"contains_real_phi": True}, "contains_real_phi=false"),
        ({"contains_dua_data": True}, "contains_dua_data=false"),
        ({"disclaimer": "Evaluation fixture."}, "assist-only and non-decisional"),
    ),
)
def test_india_manifest_rejects_unsafe_declarations(
    changes: dict[str, object], message: str
) -> None:
    manifest = load_india_clinical_phi_corpus().manifest

    with pytest.raises(ValueError, match=message):
        validate_india_clinical_phi_manifest(replace(manifest, **changes))


def test_india_loader_rejects_missing_safety_declaration(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    del payload["contains_dua_data"]
    manifest_path = tmp_path / MANIFEST_PATH.name
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="contains_dua_data must be boolean"):
        load_india_clinical_phi_corpus(manifest_path, FIXTURE_PATH)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("offset", "span text does not match offsets"),
        ("synthetic", "span must be marked synthetic"),
    ),
)
def test_india_loader_rejects_malformed_or_unsafe_spans(
    tmp_path: Path, mutation: str, message: str
) -> None:
    rows = [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if mutation == "offset":
        rows[0]["gold_spans"][0]["end"] += 1
    else:
        rows[0]["gold_spans"][0]["metadata"]["synthetic"] = False
    fixture_path = tmp_path / FIXTURE_PATH.name
    fixture_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_india_clinical_phi_corpus(MANIFEST_PATH, fixture_path)


def test_india_specialized_fixture_is_not_loaded_as_generic_golden_data() -> None:
    assert FIXTURE_PATH not in list_fixture_paths()
