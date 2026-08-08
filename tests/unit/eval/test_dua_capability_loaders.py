"""Synthetic offline coverage for the five named DUA capability loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.core.labels import DATE, OTHER, PERSON
from openmed.eval.datasets import (
    CEGS_NGRID,
    MEDNLI,
    MIMIC_IV_BHC,
    SHAC,
    THYME,
    DUACredentialRequired,
    assert_no_gated_content_committed,
    license_for,
    load_cegs_ngrid,
    load_mednli,
    load_mimic_iv_bhc,
    load_shac,
    load_thyme,
)


def test_cegs_ngrid_maps_synthetic_ner_rows(tmp_path: Path) -> None:
    text = "Synthetic person has record ZX-001."
    source = tmp_path / "cegs.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "synthetic-cegs-1",
                "text": text,
                "entities": [
                    {
                        "id": "T1",
                        "start": text.index("person"),
                        "end": text.index("person") + len("person"),
                        "label": "NAME",
                    },
                    {
                        "id": "T2",
                        "start": text.index("ZX-001"),
                        "end": text.index("ZX-001") + len("ZX-001"),
                        "label": "MEDICALRECORD",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    [fixture] = load_cegs_ngrid(source)

    assert [span.label for span in fixture.gold_spans] == [PERSON, "ID_NUM"]
    assert fixture.metadata["task_view"] == "deid_ner"
    assert fixture.metadata["network_fetch"] is False


def test_shac_maps_synthetic_sdoh_event_relation(tmp_path: Path) -> None:
    text = "Synthetic tobacco status is historical."
    tobacco_start = text.index("tobacco")
    status_start = text.index("historical")
    source = tmp_path / "shac.json"
    source.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "id": "synthetic-shac-1",
                        "text": text,
                        "entities": [
                            {
                                "id": "T1",
                                "start": tobacco_start,
                                "end": tobacco_start + len("tobacco"),
                                "label": "Tobacco",
                            },
                            {
                                "id": "T2",
                                "start": status_start,
                                "end": status_start + len("historical"),
                                "label": "Status",
                            },
                        ],
                        "relations": [
                            {
                                "id": "R1",
                                "type": "Status",
                                "head": "T1",
                                "tail": "T2",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    [fixture] = load_shac(source)

    assert fixture.task_view == "sdoh_event_relation"
    assert fixture.gold_relations[0].relation_type == "HAS_STATUS"
    assert fixture.gold_relations[0].metadata["source_relation_type"] == "Status"
    assert fixture.entities["T1"].label == OTHER


def test_thyme_maps_synthetic_temporal_relation(tmp_path: Path) -> None:
    text = "Synthetic therapy started Monday."
    event_start = text.index("therapy")
    date_start = text.index("Monday")
    source = tmp_path / "thyme.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "synthetic-thyme-1",
                "text": text,
                "entities": [
                    {
                        "id": "E1",
                        "start": event_start,
                        "end": event_start + len("therapy"),
                        "label": "EVENT",
                        "role": "EVENT",
                    },
                    {
                        "id": "T1",
                        "start": date_start,
                        "end": date_start + len("Monday"),
                        "label": "TIMEX3",
                        "role": "TIMEX3",
                    },
                ],
                "tlinks": [
                    {
                        "id": "TL1",
                        "type": "BEFORE",
                        "source": "E1",
                        "target": "T1",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    [fixture] = load_thyme(source)

    assert fixture.task_view == "temporal_event_relation"
    assert fixture.entities["E1"].label == OTHER
    assert fixture.entities["T1"].label == DATE
    assert fixture.gold_relations[0].relation_type == "BEFORE"


def test_mednli_maps_synthetic_sentence_pair(tmp_path: Path) -> None:
    source = tmp_path / "mednli.jsonl"
    source.write_text(
        json.dumps(
            {
                "pairID": "synthetic-mednli-1",
                "sentence1": "Synthetic note describes a stable condition.",
                "sentence2": "The condition is stable.",
                "gold_label": "entails",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    [fixture] = load_mednli(source)

    assert fixture.task_view == "sentence_pair_nli"
    assert fixture.sentence1.startswith("Synthetic note")
    assert fixture.sentence2 == "The condition is stable."
    assert fixture.gold_label == "entailment"


def test_mimic_iv_bhc_maps_synthetic_document_summary_pair(tmp_path: Path) -> None:
    source = tmp_path / "bhc.json"
    source.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "id": "synthetic-bhc-1",
                        "document": "Synthetic discharge document.",
                        "summary": "Synthetic discharge summary.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    [fixture] = load_mimic_iv_bhc(source)

    assert fixture.task_view == "document_summary_pair"
    assert fixture.source_text == fixture.document
    assert fixture.reference_summary == fixture.summary
    assert fixture.metadata["task"] == "summarization"


@pytest.mark.parametrize(
    ("loader", "authority"),
    (
        (load_cegs_ngrid, "DBMI"),
        (load_shac, "DBMI"),
        (load_thyme, "Mayo-THYME"),
        (load_mednli, "PhysioNet"),
        (load_mimic_iv_bhc, "UW"),
    ),
)
def test_gated_loaders_refuse_without_credentialed_path(
    loader,
    authority: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in (
        "OPENMED_CEGS_NGRID_PATH",
        "OPENMED_SHAC_PATH",
        "OPENMED_THYME_PATH",
        "OPENMED_MEDNLI_PATH",
        "OPENMED_MIMIC_IV_BHC_PATH",
    ):
        monkeypatch.delenv(variable, raising=False)

    with pytest.raises(DUACredentialRequired, match=authority):
        loader()


def test_gated_loaders_refuse_repository_paths_before_reading() -> None:
    repository_root = Path(__file__).resolve().parents[3]

    with pytest.raises(DUACredentialRequired, match="repository"):
        load_mednli(repository_root)


def test_licenses_and_payload_guard_cover_all_five_datasets(tmp_path: Path) -> None:
    for dataset in (CEGS_NGRID, SHAC, THYME, MEDNLI, MIMIC_IV_BHC):
        license_metadata = license_for(dataset)
        assert license_metadata.redistribution == (
            "credentialed eval-only; never redistributed"
        )
        assert license_metadata.notes

    clean_payload = tmp_path / "clean.json"
    clean_payload.write_text(
        json.dumps({"text": "Synthetic fixture", "label": "OTHER"}),
        encoding="utf-8",
    )
    assert_no_gated_content_committed(tmp_path)

    gated_payload = tmp_path / "gated.json"
    gated_payload.write_text(
        json.dumps({"dataset": "cegs-ngrid"}),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="gated dataset content"):
        assert_no_gated_content_committed(tmp_path)


def test_repository_fixture_roots_contain_no_gated_payloads() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    for fixture_root in (
        repository_root / "openmed" / "eval" / "datasets",
        repository_root / "tests" / "unit" / "eval" / "fixtures",
    ):
        assert_no_gated_content_committed(fixture_root)
