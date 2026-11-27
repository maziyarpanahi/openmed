"""Unit tests for the public biomedical NER benchmark suite."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from openmed.cli import main_module
from openmed.core.labels import ANATOMY, CONDITION, MEDICATION, MICROORGANISM, OTHER
from openmed.eval.datasets import (
    ANATEM,
    BC2GM,
    BC5CDR,
    BIOMEDICAL_NER,
    BIOMEDICAL_NER_CORPORA,
    JNLPBA,
    NCBI_DISEASE,
    SPECIES_800,
    biomedical_ner_suite_metadata,
    license_for,
    load_biomedical_ner_corpus,
    load_biomedical_ner_fixtures,
    map_biomedical_ner_label,
    run_biomedical_ner_benchmark,
)
from openmed.eval.suites import load_suite_fixtures, suite_metadata


def test_biomedical_ner_label_mapping_is_canonical() -> None:
    assert map_biomedical_ner_label(BC5CDR, "Chemical") == MEDICATION
    assert map_biomedical_ner_label(BC5CDR, "Disease") == CONDITION
    assert map_biomedical_ner_label(NCBI_DISEASE, "SpecificDisease") == CONDITION
    assert map_biomedical_ner_label(JNLPBA, "protein") == OTHER
    assert map_biomedical_ner_label(SPECIES_800, "Species") == MICROORGANISM
    assert map_biomedical_ner_label(BC2GM, "GENE") == OTHER
    assert map_biomedical_ner_label(ANATEM, "Organ") == ANATOMY


def test_biomedical_ner_loader_builds_fixtures_from_synthetic_sources(
    tmp_path: Path,
) -> None:
    _write_synthetic_suite(tmp_path)

    fixtures = load_biomedical_ner_fixtures(tmp_path)

    labels_by_dataset: dict[str, list[str]] = defaultdict(list)
    text_by_dataset: dict[str, str] = {}
    for fixture in fixtures:
        dataset = str(fixture.metadata["dataset"])
        labels_by_dataset[dataset].extend(span.label for span in fixture.gold_spans)
        text_by_dataset[dataset] = fixture.text
        assert fixture.metadata["suite"] == BIOMEDICAL_NER
        assert fixture.metadata["license"] == license_for(dataset).to_dict()

    assert set(labels_by_dataset) == set(BIOMEDICAL_NER_CORPORA)
    assert labels_by_dataset[BC5CDR] == [MEDICATION, CONDITION]
    assert labels_by_dataset[NCBI_DISEASE] == [CONDITION]
    assert labels_by_dataset[JNLPBA] == [OTHER]
    assert labels_by_dataset[SPECIES_800] == [MICROORGANISM]
    assert labels_by_dataset[BC2GM] == [OTHER, OTHER]
    assert labels_by_dataset[ANATEM] == [ANATOMY]
    assert text_by_dataset[SPECIES_800] == "Escherichia coli grows"
    assert text_by_dataset[BC2GM] == "TP53 regulates EGFR"


def test_biomedical_ner_loader_uses_rows_loader_and_cache_dir(tmp_path: Path) -> None:
    calls = []

    def rows_loader(source, split, cache_dir):
        calls.append((source.corpus, split, cache_dir))
        return [_bigbio_row("bc5-rows", "Aspirin treats melanoma.", "Chemical", 0, 7)]

    corpus = load_biomedical_ner_corpus(
        BC5CDR,
        cache_dir=tmp_path,
        rows_loader=rows_loader,
        split="validation",
    )

    assert calls == [(BC5CDR, "validation", tmp_path)]
    assert len(corpus.records) == 1
    assert corpus.records[0].spans[0].canonical_label == MEDICATION


@pytest.mark.parametrize(
    "source_label",
    [
        "Anatomical_system",
        "Cancer",
        "Cell",
        "Cellular_component",
        "Developing_anatomical_structure",
        "Immaterial_anatomical_entity",
        "Multi-tissue_structure",
        "Organ",
        "Organism_subdivision",
        "Organism_substance",
        "Pathological_formation",
        "Tissue",
    ],
)
def test_anatem_injected_source_rows_preserve_offsets(
    tmp_path: Path, source_label: str
) -> None:
    text = "The invented structure is visible."

    def rows_loader(source, split, cache_dir):
        assert source.corpus == ANATEM
        assert source.repository == "bigbio/anat_em"
        assert source.config == "anat_em_source"
        assert (split, cache_dir) == ("validation", tmp_path)
        return [
            {
                "document_id": "synthetic-anatomy",
                "document_type": "PM",
                "text_type": "abstract",
                "text": text,
                "entities": [
                    {
                        "entity_id": "T1",
                        "type": source_label,
                        "offsets": [[4, 22]],
                        "text": ["invented structure"],
                    }
                ],
            },
            {"document_id": "negative", "text": "No mentions.", "entities": []},
        ]

    corpus = load_biomedical_ner_corpus(
        ANATEM, rows_loader=rows_loader, split="validation", cache_dir=tmp_path
    )

    record, negative = corpus.records
    (span,) = record.spans
    assert record.text == text
    assert record.split == "validation"
    assert (span.start, span.end, span.text) == (4, 22, "invented structure")
    assert record.text[span.start : span.end] == span.text
    assert span.source_label == source_label
    assert span.canonical_label == ANATOMY
    assert negative.spans == ()


def test_anatem_rejects_out_of_bounds_offsets() -> None:
    with pytest.raises(ValueError, match="invalid anat_em span offsets"):
        load_biomedical_ner_corpus(
            ANATEM,
            rows_loader=lambda *_: [
                {
                    "text": "Synthetic anatomy.",
                    "entities": [{"type": "Organ", "offsets": [[0, 100]]}],
                }
            ],
        )


def test_biomedical_ner_report_contains_per_corpus_exact_and_relaxed_f1(
    tmp_path: Path,
) -> None:
    _write_synthetic_suite(tmp_path)
    fixtures = load_biomedical_ner_fixtures(tmp_path)

    def runner(fixture, model_name, device):
        assert model_name == "fixture-model"
        assert device == "cpu"
        return list(fixture.gold_spans)

    report = run_biomedical_ner_benchmark(
        fixtures,
        model_name="fixture-model",
        runner=runner,
        metadata=biomedical_ner_suite_metadata(),
    )

    data = report.to_dict()
    assert data["suite"] == BIOMEDICAL_NER
    assert data["fixture_count"] == len(BIOMEDICAL_NER_CORPORA)
    assert data["metrics"]["exact_span_f1"]["f1"] == 1.0
    assert set(data["metrics"]["per_corpus"]) == set(BIOMEDICAL_NER_CORPORA)
    for corpus in BIOMEDICAL_NER_CORPORA:
        corpus_metrics = data["metrics"]["per_corpus"][corpus]
        assert corpus_metrics["fixture_count"] == 1
        assert corpus_metrics["exact_span_f1"]["f1"] == 1.0
        assert corpus_metrics["relaxed_span_f1"]["f1"] == 1.0


def test_biomedical_ner_suite_registry_and_metadata(tmp_path: Path) -> None:
    _write_synthetic_suite(tmp_path)

    fixtures = load_suite_fixtures(BIOMEDICAL_NER, path=tmp_path, task="ner")
    metadata = suite_metadata(BIOMEDICAL_NER)

    assert len(fixtures) == len(BIOMEDICAL_NER_CORPORA)
    assert metadata["suite"] == BIOMEDICAL_NER
    assert metadata["licenses"][BC5CDR] == license_for(BC5CDR).to_dict()
    assert metadata["sources"][BC2GM]["repository"] == "bigbio/blurb"


def test_biomedical_ner_rejects_relation_task(tmp_path: Path) -> None:
    _write_synthetic_suite(tmp_path)

    with pytest.raises(ValueError, match="only supports task='ner'"):
        load_biomedical_ner_fixtures(tmp_path, task="relation")


def test_cli_benchmark_clinical_biomedical_ner_emits_benchmark_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from openmed.eval import harness

    _write_synthetic_suite(tmp_path)

    def runner(fixture, model_name, device):
        return list(fixture.gold_spans)

    monkeypatch.setattr(harness, "default_model_runner", runner)

    result = main_module.main(
        [
            "benchmark",
            "clinical",
            "--suite",
            BIOMEDICAL_NER,
            "--task",
            "ner",
            "--input",
            str(tmp_path),
        ]
    )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["suite"] == BIOMEDICAL_NER
    assert output["model_name"] == "disease_detection_superclinical"
    assert output["metadata"]["licenses"][SPECIES_800]["license_id"]
    assert output["metrics"]["per_corpus"][BC5CDR]["exact_span_f1"]["f1"] == 1.0
    assert output["metrics"]["per_corpus"][SPECIES_800]["relaxed_span_f1"]["f1"] == 1.0


def test_biomedical_ner_licenses_are_registered() -> None:
    for corpus in BIOMEDICAL_NER_CORPORA:
        dataset_license = license_for(corpus)
        assert dataset_license.dataset == corpus
        expected = (
            "not redistributed by OpenMed" if corpus == ANATEM else "download-on-demand"
        )
        assert dataset_license.redistribution == expected
        assert dataset_license.source_url
    assert license_for(ANATEM).license_id == "CC-BY-SA-3.0"


def _write_synthetic_suite(root: Path) -> None:
    _write_json(
        root / f"{ANATEM}.json",
        [
            {
                "document_id": "synthetic-anatomy",
                "text": "The invented structure is visible.",
                "entities": [_bigbio_entity_row("Organ", 4, 22, "invented structure")],
            }
        ],
    )
    _write_json(
        root / f"{BC5CDR}.json",
        [
            _bigbio_row("bc5-1", "Aspirin treats melanoma.", "Chemical", 0, 7),
            _bigbio_entity_row("Disease", 15, 23, "melanoma"),
        ],
    )
    _write_json(
        root / f"{NCBI_DISEASE}.json",
        [_bigbio_row("ncbi-1", "Breast cancer progresses.", "Disease", 0, 13)],
    )
    _write_json(
        root / f"{JNLPBA}.json",
        [_bigbio_row("jnlpba-1", "BRCA1 protein activates T cells.", "protein", 0, 13)],
    )
    (root / f"{SPECIES_800}.conll").write_text(
        "Escherichia B-Species\ncoli I-Species\ngrows O\n",
        encoding="utf-8",
    )
    (root / f"{BC2GM}.conll").write_text(
        "TP53 B-GENE\nregulates O\nEGFR B-GENE\n",
        encoding="utf-8",
    )


def _write_json(path: Path, rows: list[dict[str, object]]) -> None:
    if len(rows) == 2 and "entities" in rows[0] and "type" in rows[1]:
        rows[0]["entities"].append(rows[1])
        rows = [rows[0]]
    path.write_text(json.dumps({"documents": rows}), encoding="utf-8")


def _bigbio_row(
    record_id: str,
    text: str,
    label: str,
    start: int,
    end: int,
) -> dict[str, object]:
    return {
        "entities": [_bigbio_entity_row(label, start, end, text[start:end])],
        "id": record_id,
        "passages": [{"offsets": [[0, len(text)]], "text": text}],
    }


def _bigbio_entity_row(
    label: str,
    start: int,
    end: int,
    text: str,
) -> dict[str, object]:
    return {
        "id": f"T{start}-{end}",
        "offsets": [[start, end]],
        "text": [text],
        "type": label,
    }
