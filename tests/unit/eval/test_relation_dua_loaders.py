"""Focused synthetic tests for the n2c2 Track 2 and MADE DUA loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.core.labels import (
    CANONICAL_LABELS,
    CONDITION,
    DOSAGE,
    DURATION,
    FORM,
    FREQUENCY,
    INDICATION,
    MEDICATION,
    ROUTE,
    SEVERITY,
    STRENGTH,
)
from openmed.eval.datasets.dua_stubs import DUACredentialRequired
from openmed.eval.datasets.made import (
    MADE_DUA_NAME,
    MADE_ENTITY_TO_CANONICAL,
    MADE_ENTITY_TYPES,
    MADE_PATH_ENV,
    MADE_RELATION_TO_CANONICAL,
    MADE_RELATION_TYPES,
    load_made_ner_fixtures,
    load_made_relation_fixtures,
    map_made_entity_label,
    map_made_relation_type,
)
from openmed.eval.datasets.n2c2_2018_track2 import (
    N2C2_2018_TRACK2_DUA_NAME,
    N2C2_2018_TRACK2_ENTITY_TO_CANONICAL,
    N2C2_2018_TRACK2_ENTITY_TYPES,
    N2C2_2018_TRACK2_PATH_ENV,
    N2C2_2018_TRACK2_RELATION_TO_CANONICAL,
    N2C2_2018_TRACK2_RELATION_TYPES,
    load_n2c2_2018_track2_ner_fixtures,
    load_n2c2_2018_track2_relation_fixtures,
    map_n2c2_2018_track2_entity_label,
    map_n2c2_2018_track2_relation_type,
)


def test_relation_dua_mappings_are_total_and_canonical() -> None:
    assert set(N2C2_2018_TRACK2_ENTITY_TYPES) == set(
        N2C2_2018_TRACK2_ENTITY_TO_CANONICAL
    )
    assert set(MADE_ENTITY_TYPES) == set(MADE_ENTITY_TO_CANONICAL)
    assert set(N2C2_2018_TRACK2_RELATION_TYPES) == set(
        N2C2_2018_TRACK2_RELATION_TO_CANONICAL
    )
    assert set(MADE_RELATION_TYPES) == set(MADE_RELATION_TO_CANONICAL)
    assert set(N2C2_2018_TRACK2_ENTITY_TO_CANONICAL.values()) <= CANONICAL_LABELS
    assert set(MADE_ENTITY_TO_CANONICAL.values()) <= CANONICAL_LABELS

    assert map_n2c2_2018_track2_entity_label("Drug") == MEDICATION
    assert map_n2c2_2018_track2_entity_label("Reason") == INDICATION
    assert map_n2c2_2018_track2_relation_type("Strength-Drug") == ("DRUG_TO_STRENGTH")
    assert map_made_entity_label("Drugname") == MEDICATION
    assert map_made_entity_label("Other_SSD") == CONDITION
    assert map_made_entity_label("Severity") == SEVERITY
    assert map_made_relation_type("ADE-Drugname") == "DRUG_TO_ADE"
    assert map_made_relation_type("SSD-Severity") == "SSD_TO_SEVERITY"


def test_n2c2_track2_brat_loads_ner_and_relation_views(tmp_path: Path) -> None:
    source = tmp_path / "n2c2-track2"
    source.mkdir()
    text = "Aspirin 10 mg oral tablet nausea 1 tablet pain daily 7 days."
    entities = [
        ("T1", "Drug", "Aspirin"),
        ("T2", "Strength", "10 mg"),
        ("T3", "Route", "oral"),
        ("T4", "Form", "tablet"),
        ("T5", "ADE", "nausea"),
        ("T6", "Dosage", "1 tablet"),
        ("T7", "Reason", "pain"),
        ("T8", "Frequency", "daily"),
        ("T9", "Duration", "7 days"),
    ]
    lines = [
        f"{entity_id}\t{label} {_span(text, surface)} "
        f"{_span(text, surface) + len(surface)}\t{surface}"
        for entity_id, label, surface in entities
    ]
    lines.extend(
        (
            "R1\tStrength-Drug Arg1:T2 Arg2:T1",
            "R2\tRoute-Drug Arg1:T3 Arg2:T1",
            "R3\tForm-Drug Arg1:T4 Arg2:T1",
            "R4\tADE-Drug Arg1:T5 Arg2:T1",
            "R5\tDosage-Drug Arg1:T6 Arg2:T1",
            "R6\tReason-Drug Arg1:T7 Arg2:T1",
            "R7\tFrequency-Drug Arg1:T8 Arg2:T1",
            "R8\tDuration-Drug Arg1:T9 Arg2:T1",
        )
    )
    (source / "record.txt").write_text(text, encoding="utf-8")
    (source / "record.ann").write_text("\n".join(lines), encoding="utf-8")

    relation_fixtures = load_n2c2_2018_track2_relation_fixtures(source)
    assert len(relation_fixtures) == 1
    relation_fixture = relation_fixtures[0]
    assert relation_fixture.metadata["dua"] == N2C2_2018_TRACK2_DUA_NAME
    assert relation_fixture.metadata["task"] == "relation"
    assert [entity.canonical_label for entity in relation_fixture.entities] == [
        MEDICATION,
        STRENGTH,
        ROUTE,
        FORM,
        CONDITION,
        DOSAGE,
        INDICATION,
        FREQUENCY,
        DURATION,
    ]
    assert [relation.to_tuple() for relation in relation_fixture.relations] == [
        (canonical, "T1", argument_id)
        for canonical, argument_id in (
            ("DRUG_TO_STRENGTH", "T2"),
            ("DRUG_TO_ROUTE", "T3"),
            ("DRUG_TO_FORM", "T4"),
            ("DRUG_TO_ADE", "T5"),
            ("DRUG_TO_DOSE", "T6"),
            ("DRUG_TO_INDICATION", "T7"),
            ("DRUG_TO_FREQUENCY", "T8"),
            ("DRUG_TO_DURATION", "T9"),
        )
    ]

    ner_fixture = load_n2c2_2018_track2_ner_fixtures(source)[0]
    assert ner_fixture.text == text
    assert ner_fixture.metadata["task"] == "ner"
    assert [span.label for span in ner_fixture.gold_spans] == [
        MEDICATION,
        STRENGTH,
        ROUTE,
        FORM,
        CONDITION,
        DOSAGE,
        INDICATION,
        FREQUENCY,
        DURATION,
    ]


def test_made_bioc_loads_entities_and_document_relations(tmp_path: Path) -> None:
    source = tmp_path / "made"
    source.mkdir()
    text = "Aspirin 10 mg oral daily for pain caused nausea over 7 days; mild reaction."
    entity_rows = [
        ("T1", "Drugname", "Aspirin"),
        ("T2", "Dosage", "10 mg"),
        ("T3", "Route", "oral"),
        ("T4", "Frequency", "daily"),
        ("T5", "Indication", "pain"),
        ("T6", "ADE", "nausea"),
        ("T7", "Duration", "7 days"),
        ("T8", "Severity", "mild"),
        ("T9", "Other SSD", "reaction"),
    ]
    annotations = [
        {
            "id": entity_id,
            "infons": {"type": label},
            "locations": [{"offset": _span(text, surface), "length": len(surface)}],
            "text": surface,
        }
        for entity_id, label, surface in entity_rows
    ]
    relations = [
        {
            "id": "R1",
            "infons": {"type": "ADE-Drugname"},
            "nodes": [
                {"refid": "T6", "role": "entity1"},
                {"refid": "T1", "role": "entity2"},
            ],
        },
        {
            "id": "R2",
            "infons": {"type": "Indication-Drugname"},
            "nodes": [
                {"refid": "T5", "role": "entity1"},
                {"refid": "T1", "role": "entity2"},
            ],
        },
        {
            "id": "R3",
            "infons": {"type": "Drugname-Dosage"},
            "nodes": [
                {"refid": "T1", "role": "entity1"},
                {"refid": "T2", "role": "entity2"},
            ],
        },
        {
            "id": "R4",
            "infons": {"type": "Drugname-Route"},
            "nodes": [
                {"refid": "T1", "role": "entity1"},
                {"refid": "T3", "role": "entity2"},
            ],
        },
        {
            "id": "R5",
            "infons": {"type": "Drugname-Frequency"},
            "nodes": [
                {"refid": "T1", "role": "entity1"},
                {"refid": "T4", "role": "entity2"},
            ],
        },
        {
            "id": "R6",
            "infons": {"type": "Drugname-Duration"},
            "nodes": [
                {"refid": "T1", "role": "entity1"},
                {"refid": "T7", "role": "entity2"},
            ],
        },
        {
            "id": "R7",
            "infons": {"type": "SSD-Severity"},
            "nodes": [
                {"refid": "T9", "role": "entity1"},
                {"refid": "T8", "role": "entity2"},
            ],
        },
    ]
    payload = {
        "documents": [
            {
                "id": "made-synthetic-001",
                "passages": [
                    {
                        "offset": 0,
                        "text": text,
                        "annotations": annotations,
                        "relations": relations,
                    }
                ],
            }
        ]
    }
    (source / "made.json").write_text(json.dumps(payload), encoding="utf-8")

    fixtures = load_made_relation_fixtures(source)
    assert len(fixtures) == 1
    fixture = fixtures[0]
    assert fixture.metadata["dua"] == MADE_DUA_NAME
    assert fixture.metadata["task"] == "relation"
    assert fixture.relations[0].scope == "document"
    assert [relation.to_tuple() for relation in fixture.relations] == [
        (canonical, "T1", argument_id)
        for canonical, argument_id in (
            ("DRUG_TO_ADE", "T6"),
            ("DRUG_TO_INDICATION", "T5"),
            ("DRUG_TO_DOSE", "T2"),
            ("DRUG_TO_ROUTE", "T3"),
            ("DRUG_TO_FREQUENCY", "T4"),
            ("DRUG_TO_DURATION", "T7"),
        )
    ] + [("SSD_TO_SEVERITY", "T9", "T8")]
    assert [entity.canonical_label for entity in fixture.entities] == [
        MEDICATION,
        DOSAGE,
        ROUTE,
        FREQUENCY,
        INDICATION,
        CONDITION,
        DURATION,
        SEVERITY,
        CONDITION,
    ]

    ner_fixture = load_made_ner_fixtures(source)[0]
    assert ner_fixture.text == text
    assert ner_fixture.metadata["task"] == "ner"
    assert [span.label for span in ner_fixture.gold_spans] == [
        MEDICATION,
        DOSAGE,
        ROUTE,
        FREQUENCY,
        INDICATION,
        CONDITION,
        DURATION,
        SEVERITY,
        CONDITION,
    ]


@pytest.mark.parametrize(
    ("loader", "path_env", "dua_name"),
    (
        (
            load_n2c2_2018_track2_relation_fixtures,
            N2C2_2018_TRACK2_PATH_ENV,
            N2C2_2018_TRACK2_DUA_NAME,
        ),
        (load_made_relation_fixtures, MADE_PATH_ENV, MADE_DUA_NAME),
    ),
)
def test_relation_dua_loaders_refuse_unconfigured_and_repo_paths(
    loader,
    path_env: str,
    dua_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(path_env, raising=False)

    with pytest.raises(DUACredentialRequired, match=dua_name):
        loader()

    repo_root = Path(__file__).resolve().parents[3]
    with pytest.raises(DUACredentialRequired, match="repository tree"):
        loader(repo_root)


def _span(text: str, surface: str) -> int:
    start = text.index(surface)
    return start
