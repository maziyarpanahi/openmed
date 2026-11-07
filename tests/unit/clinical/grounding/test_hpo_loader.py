"""Focused tests for the offline, user-supplied HPO vocabulary loader."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.clinical.grounding import ConceptMatch, VocabularyLoader
from openmed.clinical.grounding.loaders import HPOVocabularyLoader
from openmed.clinical.grounding.registry import (
    VocabularyLoaderRegistry,
    validate_vocabulary_loader,
)


def _obo_release(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic-hpo.obo"
    path.write_text(
        """format-version: 1.2

[Term]
id: HP:0000001
name: Synthetic phenotypic abnormality

[Term]
id: HP:0000002
name: Synthetic neurological finding
is_a: HP:0000001 ! Synthetic phenotypic abnormality
synonym: "synthetic neuro sign" EXACT []
synonym: "synthetic broad sign" BROAD []

[Term]
id: HP:0000003
name: Synthetic seizure finding
is_a: HP:0000002 ! Synthetic neurological finding
synonym: "episode of synthetic seizure" RELATED []

[Term]
id: HP:0000004
name: Synthetic obsolete finding
is_obsolete: true
""",
        encoding="utf-8",
    )
    return path


def test_hpo_loader_conforms_and_resolves_synonyms(tmp_path: Path) -> None:
    loader = HPOVocabularyLoader(_obo_release(tmp_path))

    assert isinstance(loader, VocabularyLoader)
    assert validate_vocabulary_loader(loader) is loader
    assert loader.system_uri == "http://human-phenotype-ontology.org"
    assert loader.redistributable is True
    assert "CC BY 4.0" in loader.license_note

    matches = loader.lookup("episode of synthetic seizure")

    assert matches
    assert isinstance(matches[0], ConceptMatch)
    assert matches[0].code == "HP:0000003"
    assert matches[0].display == "Synthetic seizure finding"
    assert matches[0].metadata["ontology_path"] == (
        "HP:0000001",
        "HP:0000002",
        "HP:0000003",
    )
    assert "synthetic neuro sign" in loader.load()
    assert loader.lookup("synthetic broad sign") == ()
    assert loader.concept_count == 3

    registry = VocabularyLoaderRegistry()
    registry.register(loader)
    assert registry.matcher(loader.system_uri).lookup("synthetic neuro sign")[
        0
    ].code == ("HP:0000002")


def test_hpo_is_a_traversal_and_subsumption_are_deterministic(tmp_path: Path) -> None:
    loader = HPOVocabularyLoader(_obo_release(tmp_path))

    assert loader.ancestors("HP:0000003") == frozenset({"HP:0000001", "HP:0000002"})
    assert loader.descendants("HP:0000001") == frozenset({"HP:0000002", "HP:0000003"})
    assert loader.ancestor_ids("HP:0000003") == (
        "HP:0000001",
        "HP:0000002",
    )
    assert loader.is_ancestor("HP:0000001", "HP:0000003")
    assert loader.is_descendant("HP:0000003", "HP:0000001")
    assert not loader.is_ancestor("HP:0000003", "HP:0000003")
    assert loader.subsumes("HP:0000001", "HP:0000001")
    assert loader.is_subsumed_by("HP:0000003", "HP:0000001")
    assert loader.roll_up("HP:0000003") == frozenset({"HP:0000001", "HP:0000002"})


def test_hpo_loader_accepts_obo_json_graphs_and_edges(tmp_path: Path) -> None:
    path = tmp_path / "synthetic-hpo.json"
    path.write_text(
        json.dumps(
            {
                "graphs": [
                    {
                        "nodes": [
                            {
                                "id": "http://purl.obolibrary.org/obo/HP_1000001",
                                "lbl": "Synthetic root",
                            },
                            {
                                "id": "HP:1000002",
                                "lbl": "Synthetic child",
                                "meta": {
                                    "synonyms": [
                                        {
                                            "pred": "hasExactSynonym",
                                            "val": "synthetic alternate",
                                        }
                                    ]
                                },
                            },
                        ],
                        "edges": [
                            {
                                "sub": "HP:1000002",
                                "pred": "is_a",
                                "obj": "http://purl.obolibrary.org/obo/HP_1000001",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    loader = HPOVocabularyLoader(source=path)
    assert loader.lookup("synthetic alternate")[0].code == "HP:1000002"
    assert loader.ancestors("HP:1000002") == frozenset({"HP:1000001"})


def test_hpo_loader_recognizes_obo_without_a_header(tmp_path: Path) -> None:
    path = tmp_path / "synthetic-no-header.obo"
    path.write_text(
        "[Term]\nid: HP:2000001\nname: Synthetic headerless root\n",
        encoding="utf-8",
    )

    assert HPOVocabularyLoader(path).lookup("Synthetic headerless root")[0].code == (
        "HP:2000001"
    )
