"""Focused tests for the local LOINC vocabulary loader."""

from __future__ import annotations

from pathlib import Path

from openmed.clinical.grounding import (
    LOINC_SYSTEM_URI,
    LoincLoader,
    VocabularyLoaderRegistry,
    validate_vocabulary_loader,
)


def _release(tmp_path: Path) -> Path:
    release = tmp_path / "loinc-release"
    release.mkdir()
    (release / "Loinc.csv").write_text(
        "LOINC_NUM,LONG_COMMON_NAME,SHORTNAME,COMPONENT,PROPERTY,TIME_ASPCT,"
        "SYSTEM,SCALE_TYP,METHOD_TYP\n"
        "1234-5,Synthetic glucose [Mass/volume] in Serum/Plasma,Synthetic glucose,"
        "LP100-1,LP200-1,LP300-1,LP400-1,LP500-1,LP600-1\n"
        "6789-0,Synthetic glucose [Presence] in Urine,Synthetic urine glucose,"
        "Glucose,Presence,Pt,Urine,Ord,Test strip\n",
        encoding="utf-8",
    )
    (release / "Part.csv").write_text(
        "PartNumber,PartTypeName,PartDisplayName\n"
        "LP100-1,Component,Glucose\n"
        "LP200-1,Property,MCnc\n"
        "LP300-1,Time,Pt\n"
        "LP400-1,System,Ser/Plas\n"
        "LP500-1,Scale,Qn\n"
        "LP600-1,Method,Enzymatic\n",
        encoding="utf-8",
    )
    (release / "AnswerList.csv").write_text(
        "AnswerListId,AnswerListName,AnswerListType,AnswerCode,AnswerDisplay\n"
        "LA100,Synthetic detection status,Nominal,LA100-1,Detected\n"
        "LA100,Synthetic detection status,Nominal,LA100-2,Not detected\n",
        encoding="utf-8",
    )
    (release / "LoincAnswerListLink.csv").write_text(
        "LoincNumber,AnswerListId\n6789-0,LA100\n",
        encoding="utf-8",
    )
    return release


def test_loader_resolves_long_name_with_six_part_axes_and_answers(tmp_path):
    loader = LoincLoader(_release(tmp_path))

    match = loader.resolve_one("Synthetic glucose [Mass/volume] in Serum/Plasma")

    assert match is not None
    assert match.code == "1234-5"
    assert match.display == "Synthetic glucose [Mass/volume] in Serum/Plasma"
    assert match.metadata["parts"] == {
        "component": "Glucose",
        "property": "MCnc",
        "time": "Pt",
        "system": "Ser/Plas",
        "scale": "Qn",
        "method": "Enzymatic",
    }
    assert match.metadata["time_aspect"] == "Pt"

    answer_match = loader.resolve_one("Synthetic urine glucose")
    assert answer_match is not None
    assert answer_match.metadata["answer_list_ids"] == ("LA100",)
    assert answer_match.metadata["answer_list"]["name"] == "Synthetic detection status"
    assert tuple(answer["code"] for answer in answer_match.metadata["answers"]) == (
        "LA100-1",
        "LA100-2",
    )


def test_part_filtered_lookup_returns_expected_candidates(tmp_path):
    loader = LoincLoader(_release(tmp_path))

    candidates = loader.lookup(system="Ser/Plas")
    queried = loader.lookup_by_parts(query="Synthetic glucose", system="Ser/Plas")

    assert [match.code for match in candidates] == ["1234-5"]
    assert [match.code for match in queried] == ["1234-5"]
    assert [match.code for match in loader.lookup(system="Urine")] == ["6789-0"]


def test_loader_conforms_to_registry_and_exposes_answer_lists(tmp_path):
    loader = LoincLoader(_release(tmp_path))

    assert validate_vocabulary_loader(loader) is loader
    registry = VocabularyLoaderRegistry()
    registry.register(loader)
    matcher = registry.matcher(LOINC_SYSTEM_URI)

    matches = matcher.lookup("Synthetic glucose")
    assert [match.code for match in matches] == ["1234-5"]
    assert loader.answer_list_ids == ("LA100",)
    assert loader.answer_lists_for("6789-0")[0].identifier == "LA100"
    assert [answer.code for answer in loader.answers_for("6789-0")] == [
        "LA100-1",
        "LA100-2",
    ]
