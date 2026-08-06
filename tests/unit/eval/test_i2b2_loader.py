"""Unit tests for the eval-only i2b2 de-identification loader."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from openmed.core.labels import (
    AGE,
    CANONICAL_LABELS,
    CONDITION,
    DATE,
    ID_NUM,
    MEDICATION,
    OCCUPATION,
    ORGANIZATION,
    OTHER,
    PERSON,
    PHONE,
    USERNAME,
)
from openmed.eval.datasets import i2b2
from openmed.eval.datasets.dua_stubs import DUACredentialRequired
from openmed.eval.datasets.i2b2 import (
    BIORED,
    BIORED_PATH_ENV,
    I2B2,
    I2B2_PATH_ENV,
    I2B2_PHI_TAG_ALIASES,
    I2B2_PHI_TAG_TO_CANONICAL,
    I2B2_PHI_TAGS,
    I2B2_YEAR_ENV,
    N2C2_2018,
    N2C2_2018_PATH_ENV,
    N2C2_2022,
    N2C2_2022_PATH_ENV,
    I2B2CredentialRequired,
    dua_relation_suite_metadata,
    load_biored_relation_fixtures,
    load_i2b2_deid,
    load_n2c2_2018_relation_fixtures,
    load_n2c2_2022_relation_fixtures,
    map_biored_entity_label,
    map_dua_relation_type,
    map_i2b2_phi_tag,
)
from openmed.eval.datasets.licenses import license_for
from openmed.eval.harness import run_dua_relation_promotion_benchmark
from openmed.eval.suites import (
    DEFAULT_SUITES,
    PROMOTION_ONLY_RELATION_SUITES,
    load_suite_fixtures,
    suite_metadata,
)


def test_load_i2b2_deid_parses_synthetic_xml_fixture(tmp_path: Path) -> None:
    source = tmp_path / "credentialed"
    source.mkdir()
    xml_path, expected_spans = _write_i2b2_xml(
        source,
        "record-001.xml",
        [
            ("NAME", "PATIENT", "Jordan Smith"),
            ("DATE", None, "2024-05-01"),
            ("AGE", None, "47"),
            ("LOCATION", "HOSPITAL", "Mercy General"),
            ("ID", "MEDICALRECORD", "MRN-001"),
            ("CONTACT", "PHONE", "555-0101"),
            ("PHI", "USERNAME", "jsmith"),
            ("PROFESSION", None, "nurse"),
        ],
    )

    fixtures = load_i2b2_deid(source, year=2014)

    assert len(fixtures) == 1
    fixture = fixtures[0]
    assert fixture.fixture_id.startswith("i2b2-2014-")
    assert fixture.metadata["dua"] == "i2b2/DBMI DUA"
    assert fixture.metadata["source_path_hash"] != xml_path.name
    assert fixture.metadata["year"] == 2014
    assert [span.label for span in fixture.gold_spans] == [
        PERSON,
        DATE,
        AGE,
        ORGANIZATION,
        ID_NUM,
        PHONE,
        USERNAME,
        OCCUPATION,
    ]
    for span, expected in zip(fixture.gold_spans, expected_spans, strict=True):
        assert (span.start, span.end, span.text) == expected
        assert fixture.text[span.start : span.end] == span.text
        assert span.metadata["i2b2_tag"] in I2B2_PHI_TAG_TO_CANONICAL


def test_loader_refuses_missing_empty_and_repo_internal_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(I2B2_PATH_ENV, raising=False)

    with pytest.raises(I2B2CredentialRequired, match="i2b2/DBMI DUA"):
        load_i2b2_deid()

    empty_source = tmp_path / "empty"
    empty_source.mkdir()
    with pytest.raises(I2B2CredentialRequired, match="i2b2/DBMI DUA"):
        load_i2b2_deid(empty_source)

    def fail_if_scanned(root: Path):  # pragma: no cover - should not be called
        raise AssertionError(f"repo path was scanned: {root}")

    monkeypatch.setattr(i2b2, "_iter_xml_files", fail_if_scanned)
    repo_root = Path(__file__).resolve().parents[3]
    with pytest.raises(I2B2CredentialRequired, match="repository tree"):
        load_i2b2_deid(repo_root)


def test_i2b2_category_map_is_total_canonical_and_strict() -> None:
    assert set(I2B2_PHI_TAG_TO_CANONICAL) == set(I2B2_PHI_TAGS)
    assert set(I2B2_PHI_TAG_TO_CANONICAL.values()) <= CANONICAL_LABELS
    assert map_i2b2_phi_tag("NAME/PATIENT") == PERSON
    assert map_i2b2_phi_tag("patient") == PERSON
    assert map_i2b2_phi_tag("CONTACT/PHONE") == PHONE
    assert map_i2b2_phi_tag("medical record number") == ID_NUM
    assert map_i2b2_phi_tag("ID/MEDICAL_RECORD_NUMBER") == ID_NUM

    for tag in I2B2_PHI_TAGS:
        assert map_i2b2_phi_tag(tag) == I2B2_PHI_TAG_TO_CANONICAL[tag]

    for alias in I2B2_PHI_TAG_ALIASES:
        assert map_i2b2_phi_tag(alias) in CANONICAL_LABELS

    with pytest.raises(ValueError, match="unknown i2b2 PHI tag"):
        map_i2b2_phi_tag("NAME/MASCOT")


def test_unknown_i2b2_xml_tag_is_surfaced(tmp_path: Path) -> None:
    source = tmp_path / "credentialed"
    source.mkdir()
    _write_i2b2_xml(
        source,
        "unknown.xml",
        [("NAME", "MASCOT", "Example")],
    )

    with pytest.raises(ValueError, match="unknown i2b2 PHI tag"):
        load_i2b2_deid(source)


def test_suite_registry_loads_i2b2_from_configured_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "credentialed"
    source.mkdir()
    _write_i2b2_xml(source, "record-001.xml", [("NAME", "PATIENT", "Jordan")])
    monkeypatch.setenv(I2B2_PATH_ENV, str(source))
    monkeypatch.setenv(I2B2_YEAR_ENV, "2006")

    fixtures = load_suite_fixtures(I2B2)
    metadata = suite_metadata(I2B2)

    assert len(fixtures) == 1
    assert fixtures[0].metadata["year"] == 2006
    assert fixtures[0].gold_spans[0].label == PERSON
    assert metadata["suite"] == I2B2
    assert metadata["label_mapping"]["NAME/PATIENT"] == PERSON


@pytest.mark.parametrize(
    ("loader", "path_env"),
    (
        (load_biored_relation_fixtures, BIORED_PATH_ENV),
        (load_n2c2_2018_relation_fixtures, N2C2_2018_PATH_ENV),
        (load_n2c2_2022_relation_fixtures, N2C2_2022_PATH_ENV),
    ),
)
def test_dua_relation_loaders_refuse_without_credentialed_path(
    loader,
    path_env: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(path_env, raising=False)

    with pytest.raises(DUACredentialRequired, match="No corpus rows were loaded"):
        loader()


def test_biored_loader_maps_bioc_json_to_relation_fixture(tmp_path: Path) -> None:
    fixtures = _write_and_load_biored(tmp_path)

    assert len(fixtures) == 1
    fixture = fixtures[0]
    assert fixture.fixture_id.startswith("biored-")
    assert [entity.canonical_label for entity in fixture.entities] == [
        MEDICATION,
        "GENE_SYMBOL",
    ]
    assert [relation.to_tuple() for relation in fixture.relations] == [
        ("NEGATIVELY_CORRELATED_WITH", "T1", "T2")
    ]
    assert fixture.relations[0].scope == "document"
    assert fixture.metadata["eval_only"] is True
    assert fixture.metadata["cache_corpus_rows"] is False
    assert fixture.metadata["network_fetch"] is False
    assert "source.json" not in json.dumps(fixture.metadata)


def test_biored_loader_accepts_bioc_xml(tmp_path: Path) -> None:
    source = tmp_path / "credentialed"
    source.mkdir()
    path = source / "source.xml"
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<collection><document><id>doc-xml</id><passage><offset>0</offset>
<text>Aspirin inhibits TP53.</text>
<annotation id="T1"><infon key="type">ChemicalEntity</infon>
<location offset="0" length="7"/><text>Aspirin</text></annotation>
<annotation id="T2"><infon key="type">GeneOrGeneProduct</infon>
<location offset="17" length="4"/><text>TP53</text></annotation>
</passage><relation id="R1"><infon key="type">Negative_Correlation</infon>
<node refid="T1" role="entity1"/><node refid="T2" role="entity2"/>
</relation></document></collection>""",
        encoding="utf-8",
    )

    fixtures = load_biored_relation_fixtures(source)

    assert fixtures[0].relations[0].relation_type == "NEGATIVELY_CORRELATED_WITH"
    assert fixtures[0].entities[1].text == "TP53"


def test_n2c2_2018_loader_maps_medication_relations_and_direction(
    tmp_path: Path,
) -> None:
    fixtures = _write_and_load_n2c2_2018(tmp_path)

    fixture = fixtures[0]
    assert [entity.canonical_label for entity in fixture.entities] == [
        MEDICATION,
        "STRENGTH",
        CONDITION,
    ]
    assert [relation.to_tuple() for relation in fixture.relations] == [
        ("DRUG_TO_STRENGTH", "T1", "T2"),
        ("DRUG_TO_ADE", "T1", "T3"),
    ]
    assert all(
        relation.arg1.canonical_label == MEDICATION for relation in fixture.relations
    )


def test_n2c2_2022_loader_maps_sdoh_event_arguments(tmp_path: Path) -> None:
    fixtures = _write_and_load_n2c2_2022(tmp_path)

    fixture = fixtures[0]
    assert [entity.canonical_label for entity in fixture.entities] == [OTHER] * 3
    assert [relation.to_tuple() for relation in fixture.relations] == [
        ("HAS_STATUS", "T1", "T2"),
        ("HAS_TYPE", "T1", "T3"),
    ]
    assert fixture.metadata["cadence"] == "human-run"
    assert fixture.metadata["daily_blocking"] is False


def test_n2c2_2022_loader_maps_employment_type_argument(tmp_path: Path) -> None:
    source = tmp_path / "n2c2-2022"
    source.mkdir()
    (source / "employment.txt").write_text(
        "Patient works as a nurse.\n",
        encoding="utf-8",
    )
    (source / "employment.ann").write_text(
        "T1\tEmployment 8 13\tworks\n"
        "T2\tTypeEmploy 19 24\tnurse\n"
        "E1\tEmployment:T1 TypeEmploy:T2\n",
        encoding="utf-8",
    )

    fixture = load_n2c2_2022_relation_fixtures(source)[0]

    assert [relation.to_tuple() for relation in fixture.relations] == [
        ("HAS_TYPE", "T1", "T2"),
    ]


@pytest.mark.parametrize(
    "corpus",
    (BIORED, N2C2_2018, N2C2_2022),
)
def test_each_dua_relation_loader_scores_through_promotion_g9(
    corpus: str,
    tmp_path: Path,
) -> None:
    fixture_loaders = {
        BIORED: _write_and_load_biored,
        N2C2_2018: _write_and_load_n2c2_2018,
        N2C2_2022: _write_and_load_n2c2_2022,
    }
    fixtures = fixture_loaders[corpus](tmp_path)

    report = run_dua_relation_promotion_benchmark(
        fixtures,
        suite=corpus,
        model_name="synthetic-relation-model",
        runner=lambda fixture, _model, _device: fixture.relations,
        ci_resamples=20,
        ci_seed=17,
    )

    gate = report.metrics["g9_dua_promotion"]
    assert gate["gate"] == "G9"
    assert gate["passed"] is True
    assert gate["details"]["strict"]["lower"] == 1.0
    assert gate["details"]["gate_tier"] == "promotion"
    assert gate["details"]["daily_blocking"] is False
    assert report.metadata["promotion_blocking"] is True


def test_dua_relation_suite_registry_and_license_metadata(tmp_path: Path) -> None:
    source = tmp_path / "credentialed"
    fixtures = _write_and_load_n2c2_2022(tmp_path)

    registered = load_suite_fixtures(N2C2_2022, path=source)
    metadata = suite_metadata(N2C2_2022)

    assert registered == fixtures
    assert set(PROMOTION_ONLY_RELATION_SUITES) == {
        BIORED,
        N2C2_2018,
        N2C2_2022,
    }
    assert set(PROMOTION_ONLY_RELATION_SUITES).isdisjoint(DEFAULT_SUITES)
    assert metadata == dua_relation_suite_metadata(N2C2_2022)
    for corpus in PROMOTION_ONLY_RELATION_SUITES:
        dataset_license = license_for(corpus)
        assert "eval-only" in dataset_license.redistribution
        assert "never redistributed" in dataset_license.redistribution

    assert map_biored_entity_label("DiseaseOrPhenotypicFeature") == CONDITION
    assert map_dua_relation_type(BIORED, "Drug_Interaction") == "DRUG_INTERACTION"


def _write_and_load_biored(tmp_path: Path):
    source = tmp_path / "credentialed"
    source.mkdir(exist_ok=True)
    path = source / "source.json"
    path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "id": "doc-json",
                        "passages": [
                            {
                                "offset": 0,
                                "text": "Aspirin inhibits TP53.",
                                "annotations": [
                                    {
                                        "id": "T1",
                                        "infons": {"type": "ChemicalEntity"},
                                        "locations": [{"offset": 0, "length": 7}],
                                        "text": "Aspirin",
                                    },
                                    {
                                        "id": "T2",
                                        "infons": {"type": "GeneOrGeneProduct"},
                                        "locations": [{"offset": 17, "length": 4}],
                                        "text": "TP53",
                                    },
                                ],
                            }
                        ],
                        "relations": [
                            {
                                "id": "R1",
                                "infons": {"type": "Negative_Correlation"},
                                "nodes": [
                                    {"refid": "T1", "role": "entity1"},
                                    {"refid": "T2", "role": "entity2"},
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return load_biored_relation_fixtures(source)


def _write_and_load_n2c2_2018(tmp_path: Path):
    source = tmp_path / "credentialed"
    source.mkdir(exist_ok=True)
    (source / "record.txt").write_text(
        "Aspirin 10 mg caused rash.",
        encoding="utf-8",
    )
    (source / "record.ann").write_text(
        "\n".join(
            (
                "T1\tDrug 0 7\tAspirin",
                "T2\tStrength 8 13\t10 mg",
                "T3\tADE 21 25\trash",
                "R1\tStrength-Drug Arg1:T2 Arg2:T1",
                "R2\tADE-Drug Arg1:T3 Arg2:T1",
            )
        ),
        encoding="utf-8",
    )
    return load_n2c2_2018_relation_fixtures(source)


def _write_and_load_n2c2_2022(tmp_path: Path):
    source = tmp_path / "credentialed"
    source.mkdir(exist_ok=True)
    (source / "record.txt").write_text(
        "Patient currently smokes cigarettes.",
        encoding="utf-8",
    )
    (source / "record.ann").write_text(
        "\n".join(
            (
                "T1\tTobacco 18 24\tsmokes",
                "T2\tStatusTime 8 17\tcurrently",
                "T3\tType 25 35\tcigarettes",
                "E1\tTobacco:T1 Status:T2 Type:T3",
            )
        ),
        encoding="utf-8",
    )
    return load_n2c2_2022_relation_fixtures(source)


def _write_i2b2_xml(
    source: Path,
    filename: str,
    pieces: list[tuple[str, str | None, str]],
) -> tuple[Path, list[tuple[int, int, str]]]:
    document = ET.Element("deIdi2b2")
    text_node = ET.SubElement(document, "TEXT")
    tags_node = ET.SubElement(document, "TAGS")
    text = ""
    expected: list[tuple[int, int, str]] = []
    for index, (category, source_type, value) in enumerate(pieces, start=1):
        if text:
            text += " "
        start = len(text)
        text += value
        end = len(text)
        attrs = {
            "end": str(end),
            "id": f"P{index}",
            "start": str(start),
            "text": value,
        }
        if source_type is not None:
            attrs["TYPE"] = source_type
        ET.SubElement(tags_node, category, attrs)
        expected.append((start, end, value))

    text_node.text = text
    path = source / filename
    path.write_text(ET.tostring(document, encoding="unicode"), encoding="utf-8")
    return path, expected
