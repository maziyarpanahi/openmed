"""Tests for license-gated Chinese clinical terminology grounding."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from openmed.clinical import (
    CHINESE_DRUG_SYSTEM,
    CHINESE_ICD_10_SYSTEM,
    ChineseTerminologyGrounder,
    ChineseTerminologyLicenseError,
    ChineseTerminologyPathError,
)
from openmed.core.labels import (
    CONDITION,
    ID_NUM,
    MEDICATION,
    PERSON,
    system_hints_for,
)
from openmed.core.pii import PIIEntity
from openmed.core.results import AnalyzeResult
from openmed.eval.suites import evaluate_chinese_terminology_leakage
from openmed.processing.outputs import EntityPrediction

REPO_ROOT = Path(__file__).parents[3]
SYNTHETIC_DICTIONARY = (
    REPO_ROOT / "examples" / "data" / "chinese_terminology_synthetic.csv"
)


@pytest.fixture
def grounder() -> ChineseTerminologyGrounder:
    return ChineseTerminologyGrounder.from_path(
        SYNTHETIC_DICTIONARY,
        license_acknowledged=True,
    )


def test_synthetic_condition_and_medication_ground_exactly(
    grounder: ChineseTerminologyGrounder,
) -> None:
    condition = grounder.match("星芒性热病", CONDITION)
    medication = grounder.match("晨露安片", MEDICATION)

    assert condition is not None and condition.exact is True
    assert condition.metadata == {
        "system": CHINESE_ICD_10_SYSTEM,
        "code": "SYN-CN-DX-001",
        "display": "星芒性热病",
    }
    assert medication is not None and medication.exact is True
    assert medication.metadata == {
        "system": CHINESE_DRUG_SYSTEM,
        "code": "SYN-CN-DRUG-001",
        "display": "晨露安片",
    }


def test_normalized_fuzzy_match_is_label_aware(
    grounder: ChineseTerminologyGrounder,
) -> None:
    fuzzy = grounder.match(" 星芒性热症。", CONDITION)

    assert fuzzy is not None
    assert fuzzy.exact is False
    assert fuzzy.entry.code == "SYN-CN-DX-001"
    assert fuzzy.score >= 0.8
    assert grounder.match("星芒性热病", MEDICATION) is None
    assert grounder.match("王小明", PERSON) is None


def test_dictionary_path_and_license_are_explicit_gates() -> None:
    with pytest.raises(ChineseTerminologyPathError, match="user-supplied.*path"):
        ChineseTerminologyGrounder.from_path(None)

    with pytest.raises(
        ChineseTerminologyLicenseError,
        match="license acknowledgement.*license_acknowledged=True",
    ):
        ChineseTerminologyGrounder.from_path(SYNTHETIC_DICTIONARY)


def test_ground_result_adds_only_coding_metadata_and_emits_notices(
    grounder: ChineseTerminologyGrounder,
) -> None:
    note = "诊断星芒性热病，予晨露安片。"
    condition_start = note.index("星芒性热病")
    medication_start = note.index("晨露安片")
    result = AnalyzeResult(
        text=note,
        entities=[
            EntityPrediction(
                text="星芒性热病",
                label=CONDITION,
                confidence=0.97,
                start=condition_start,
                end=condition_start + len("星芒性热病"),
                metadata={"source": "synthetic-model"},
            ),
            EntityPrediction(
                text="晨露安片",
                label=MEDICATION,
                confidence=0.96,
                start=medication_start,
                end=medication_start + len("晨露安片"),
            ),
        ],
        model="synthetic-zh-model",
        timestamp="2026-07-17T00:00:00",
        metadata={"language": "zh"},
    )

    grounded = grounder.ground_result(result)

    assert type(grounded) is AnalyzeResult
    assert grounded.text == result.text
    assert grounded.entities[0].start == result.entities[0].start
    assert grounded.entities[0].end == result.entities[0].end
    assert grounded.entities[0].confidence == result.entities[0].confidence
    added = set(grounded.entities[0].metadata or {}).difference(
        result.entities[0].metadata or {}
    )
    assert added == {"system", "code", "display"}
    assert grounded.entities[1].metadata == {
        "system": CHINESE_DRUG_SYSTEM,
        "code": "SYN-CN-DRUG-001",
        "display": "晨露安片",
    }
    notices = grounded.metadata["chinese_terminology_grounding"]
    assert "not a medical device" in notices["medical_device_disclaimer"]
    assert "not for clinical decisions" in notices["vocabulary_note"]
    assert notices["local_only"] is True


def test_phi_leakage_gate_preserves_offsets_and_actions(
    grounder: ChineseTerminologyGrounder,
) -> None:
    note = "患者王小明，编号甲乙丙，诊断星芒性热病，予晨露安片。"
    person_start = note.index("王小明")
    identifier_start = note.index("甲乙丙")
    condition_start = note.index("星芒性热病")
    medication_start = note.index("晨露安片")
    entities = [
        PIIEntity(
            text="王小明",
            label=PERSON,
            confidence=0.99,
            start=person_start,
            end=person_start + len("王小明"),
            action="mask",
            redacted_text="[PERSON]",
        ),
        PIIEntity(
            text="甲乙丙",
            label=ID_NUM,
            confidence=0.99,
            start=identifier_start,
            end=identifier_start + len("甲乙丙"),
            action="mask",
            redacted_text="[ID_NUM]",
        ),
        EntityPrediction(
            text="星芒性热病",
            label=CONDITION,
            confidence=0.98,
            start=condition_start,
            end=condition_start + len("星芒性热病"),
        ),
        EntityPrediction(
            text="晨露安片",
            label=MEDICATION,
            confidence=0.98,
            start=medication_start,
            end=medication_start + len("晨露安片"),
        ),
    ]
    phi_spans = [
        {
            "start": person_start,
            "end": person_start + len("王小明"),
            "label": PERSON,
            "text": "王小明",
            "language": "zh",
        },
        {
            "start": identifier_start,
            "end": identifier_start + len("甲乙丙"),
            "label": ID_NUM,
            "text": "甲乙丙",
            "language": "zh",
        },
    ]

    report = evaluate_chinese_terminology_leakage(
        source_text=note,
        entities=entities,
        phi_spans=phi_spans,
        grounder=grounder,
    )

    assert report.passed is True
    assert report.groundable_span_count == 2
    assert report.grounded_span_count == 2
    assert report.phi_leakage_rate == 0.0
    assert report.leaked_phi_characters == 0
    assert report.offsets_unchanged is True
    assert report.actions_unchanged is True
    assert report.metadata_keys_valid is True
    serialized = str(report.to_dict())
    assert "王小明" not in serialized
    assert "甲乙丙" not in serialized


def test_synthetic_demo_contains_only_invented_codes() -> None:
    with SYNTHETIC_DICTIONARY.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 12
    assert all(row["code"].startswith("SYN-CN-") for row in rows)
    assert {row["system"] for row in rows} == {
        CHINESE_ICD_10_SYSTEM,
        CHINESE_DRUG_SYSTEM,
    }


def test_chinese_system_hints_are_attached_to_clinical_labels() -> None:
    assert CHINESE_ICD_10_SYSTEM in system_hints_for(CONDITION)
    assert CHINESE_DRUG_SYSTEM in system_hints_for(MEDICATION)
