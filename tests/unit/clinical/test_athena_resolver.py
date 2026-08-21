from __future__ import annotations

import re
from pathlib import Path

from openmed.clinical.grounding import AthenaResolver

_CONCEPT_HEADER = "\t".join(
    (
        "concept_id",
        "concept_name",
        "domain_id",
        "vocabulary_id",
        "concept_class_id",
        "standard_concept",
        "concept_code",
        "valid_start_date",
        "valid_end_date",
        "invalid_reason",
    )
)
_RELATIONSHIP_HEADER = "\t".join(
    (
        "concept_id_1",
        "concept_id_2",
        "relationship_id",
        "valid_start_date",
        "valid_end_date",
        "invalid_reason",
    )
)
_VOCABULARY_HEADER = "\t".join(
    (
        "vocabulary_id",
        "vocabulary_name",
        "vocabulary_reference",
        "vocabulary_version",
        "vocabulary_concept_id",
    )
)


def _write_bundle(root: Path) -> Path:
    root.mkdir()
    concepts = [
        (1001, "Synthetic Rx source", "Drug", "RxNorm", "Ingredient", "", "RX-SOURCE"),
        (
            1002,
            "Synthetic Rx standard",
            "Drug",
            "RxNorm",
            "Ingredient",
            "S",
            "RX-STANDARD",
        ),
        (
            2001,
            "Synthetic ICD source",
            "Condition",
            "ICD10CM",
            "Clinical Finding",
            "",
            "ICD-SOURCE",
        ),
        (
            2002,
            "Synthetic ICD standard",
            "Condition",
            "ICD10CM",
            "Clinical Finding",
            "S",
            "ICD-STANDARD",
        ),
        (
            3001,
            "Synthetic CPT source",
            "Procedure",
            "CPT4",
            "Procedure",
            "",
            "CPT-SOURCE",
        ),
        (
            3002,
            "Synthetic CPT standard",
            "Procedure",
            "CPT4",
            "Procedure",
            "S",
            "CPT-STANDARD",
        ),
        (
            4001,
            "Synthetic unmapped source",
            "Observation",
            "RxNorm",
            "Ingredient",
            "",
            "RX-UNKNOWN",
        ),
        (
            5001,
            "Synthetic direct standard",
            "Drug",
            "RxNorm",
            "Ingredient",
            "S",
            "RX-DIRECT",
        ),
    ]
    concept_rows = [
        "\t".join(
            (
                str(concept_id),
                name,
                domain,
                vocabulary,
                concept_class,
                standard,
                code,
                "20260101",
                "20991231",
                "",
            )
        )
        for concept_id, name, domain, vocabulary, concept_class, standard, code in concepts
    ]
    (root / "CONCEPT.csv").write_text(
        _CONCEPT_HEADER + "\n" + "\n".join(concept_rows) + "\n",
        encoding="utf-8",
    )
    relationships = [
        (1001, 1002),
        (2001, 2002),
        (3001, 3002),
    ]
    relationship_rows = [
        "\t".join((str(source), str(target), "Maps to", "20260101", "20991231", ""))
        for source, target in relationships
    ]
    (root / "CONCEPT_RELATIONSHIP.csv").write_text(
        _RELATIONSHIP_HEADER + "\n" + "\n".join(relationship_rows) + "\n",
        encoding="utf-8",
    )
    vocabulary_rows = [
        ("RxNorm", "Synthetic RxNorm", "synthetic-rxnorm", "SYNTHETIC-ATHENA-2026", 0),
        (
            "ICD10CM",
            "Synthetic ICD-10-CM",
            "synthetic-icd10cm",
            "SYNTHETIC-ATHENA-2026",
            0,
        ),
        ("CPT4", "Synthetic CPT4", "synthetic-cpt4", "SYNTHETIC-ATHENA-2026", 0),
    ]
    vocabulary_rows_text = [
        "\t".join(str(value) for value in row) for row in vocabulary_rows
    ]
    (root / "VOCABULARY.csv").write_text(
        _VOCABULARY_HEADER + "\n" + "\n".join(vocabulary_rows_text) + "\n",
        encoding="utf-8",
    )
    return root


def test_resolves_maps_to_sources_and_exposes_standard_metadata(tmp_path: Path) -> None:
    resolver = AthenaResolver(_write_bundle(tmp_path / "athena"))

    assert resolver.resolve("RxNorm", "RX-SOURCE") == 1002
    assert resolver.resolve("ICD-10-CM", " ICD-SOURCE ") == 2002
    assert resolver.source_code("rxnorm", "RX-SOURCE") == 1002
    assert resolver.resolve_source_code("RXNORM", "RX-SOURCE") == 1002
    assert resolver.standard_concept_id("RxNorm", "RX-DIRECT") == 5001
    assert resolver.resolve("RxNorm", "RX-UNKNOWN") == 0

    concept = resolver.lookup("RxNorm", "RX-SOURCE")
    assert concept is not None
    assert concept.concept_name == "Synthetic Rx standard"
    assert concept.domain_id == "Drug"
    assert concept.vocabulary_id == "RxNorm"
    assert concept.standard_concept == "S"
    assert resolver.concept_metadata(1002)["concept_name"] == "Synthetic Rx standard"


def test_reports_version_and_deterministic_reproducibility_hash(tmp_path: Path) -> None:
    first = AthenaResolver(_write_bundle(tmp_path / "first"))
    second = AthenaResolver(_write_bundle(tmp_path / "second"))

    assert first.vocabulary_version == "SYNTHETIC-ATHENA-2026"
    assert first.vocabulary_versions == {
        "ICD10CM": "SYNTHETIC-ATHENA-2026",
        "RxNorm": "SYNTHETIC-ATHENA-2026",
    }
    assert first.reproducibility_hash == second.reproducibility_hash
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", first.reproducibility_hash)
    assert first.provenance()["vocabulary_version"] == first.vocabulary_version


def test_cpt4_is_excluded_by_default_and_requires_explicit_opt_in(
    tmp_path: Path,
) -> None:
    bundle = _write_bundle(tmp_path / "athena")
    resolver = AthenaResolver(bundle)

    assert resolver.resolve("CPT4", "CPT-SOURCE") == 0
    assert resolver.get_concept(3001) is None
    assert "CPT4" not in resolver.vocabulary_ids

    enabled = AthenaResolver(bundle, include_cpt4=True)
    assert enabled.resolve("CPT4", "CPT-SOURCE") == 3002
    assert "CPT4" in enabled.vocabulary_ids


def test_openmed_offline_uses_local_bundle(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENMED_OFFLINE", "1")

    resolver = AthenaResolver(_write_bundle(tmp_path / "athena"))

    assert resolver.offline_requested is True
    assert resolver.local_only is True
    assert resolver.resolve("RxNorm", "RX-SOURCE") == 1002
