"""Multilingual clinical section detection tests for OM-729."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from openmed.clinical import (
    CERTAIN,
    FAMILY_EXPERIENCER,
    HISTORICAL,
    RECENT,
    ClinicalAssertion,
    apply_section_context,
    canonical_section_label,
)
from openmed.clinical.lexicons import SectionLexicon, register_section_lexicon
from openmed.clinical.sections import UNSECTIONED_SECTION, detect_sections
from openmed.eval.harness import (
    DEFAULT_SECTION_MULTILINGUAL_FIXTURE,
    load_section_multilingual_fixtures,
    run_section_multilingual_eval,
)

FORBIDDEN_FIXTURE_MARKERS = ("cpt", "dua", "i2b2", "mimic", "n2c2", "snomed", "umls")
REQUIRED_LANGUAGES = {"en", "es", "fr", "de", "zh"}


def test_section_multilingual_fixture_is_synthetic_and_complete() -> None:
    meta, rows = load_section_multilingual_fixtures()

    assert meta["synthetic"] is True
    assert REQUIRED_LANGUAGES == {row["language"] for row in rows}
    assert all(row.get("synthetic") is True for row in rows)
    for row in rows:
        labels = {section["label"] for section in row["gold_sections"]}
        assert {
            "history_of_present_illness",
            "past_medical_history",
            "family_history",
            "social_history",
            "assessment",
            "plan",
        } <= labels

    fixture_text = Path(DEFAULT_SECTION_MULTILINGUAL_FIXTURE).read_text(
        encoding="utf-8"
    )
    for marker in FORBIDDEN_FIXTURE_MARKERS:
        assert re.search(rf"(?<![a-z0-9]){marker}(?![a-z0-9])", fixture_text) is None


def test_detect_sections_returns_canonical_sections_for_fixture_languages() -> None:
    _, rows = load_section_multilingual_fixtures()

    for row in rows:
        sections = detect_sections(row["text"], language=row["language"])
        labels = [section["label"] for section in sections]

        assert labels == [
            UNSECTIONED_SECTION,
            *[section["label"] for section in row["gold_sections"]],
        ], row["case_id"]
        assert sections[0]["start"] == 0
        assert sections[-1]["end"] == len(row["text"])
        assert all(section["start"] < section["end"] for section in sections)


def test_detected_canonical_sections_drive_context_priors() -> None:
    _, rows = load_section_multilingual_fixtures()
    assertion = ClinicalAssertion(temporality=RECENT, certainty=CERTAIN)

    for row in rows:
        sections = {
            section["label"]: section
            for section in detect_sections(row["text"], language=row["language"])
        }

        historical = apply_section_context(
            {"text": "asthma"},
            sections["past_medical_history"],
            assertion,
        )
        family = apply_section_context(
            {"text": "diabetes"},
            sections["family_history"],
            assertion,
        )

        assert historical.temporality == HISTORICAL
        assert family.experiencer == FAMILY_EXPERIENCER


@pytest.mark.parametrize(
    ("language", "text", "expected"),
    (
        (
            "zh",
            "前言。\n既往史：儿童期哮喘。\n计划：随访。",
            [UNSECTIONED_SECTION, "past_medical_history", "plan"],
        ),
        (
            "ar",
            "تمهيد.\nالتاريخ المرضي: ربو سابق.\nالخطة: متابعة.",
            [UNSECTIONED_SECTION, "past_medical_history", "plan"],
        ),
        (
            "he",
            "רקע.\nהיסטוריה רפואית: אסתמה בעבר.\nתוכנית: מעקב.",
            [UNSECTIONED_SECTION, "past_medical_history", "plan"],
        ),
    ),
)
def test_script_aware_headers_do_not_require_latin_whitespace_boundaries(
    language: str,
    text: str,
    expected: list[str],
) -> None:
    sections = detect_sections(text, language=language)

    assert [section["label"] for section in sections] == expected


def test_section_aliases_are_visible_to_context_canonicalization() -> None:
    assert canonical_section_label("Antécédents") == "past_medical_history"
    assert canonical_section_label("Antecedentes familiares") == "family_history"
    assert canonical_section_label("既往史") == "past_medical_history"


def test_stub_language_section_pack_and_fixture_work_without_detector_changes(
    tmp_path: Path,
) -> None:
    register_section_lexicon(
        SectionLexicon(
            language="xx",
            sections={
                "past_medical_history": ("zz past",),
                "family_history": ("zz family",),
                "plan": ("zz plan",),
            },
        )
    )
    text = "Stub preface.\nzz past: old asthma.\nzz family: parent diabetes.\nzz plan: call."
    fixture = tmp_path / "section_stub.jsonl"
    rows = [
        {"kind": "meta", "suite": "section_multilingual", "synthetic": True},
        {
            "kind": "case",
            "case_id": "section-xx-1",
            "language": "xx",
            "synthetic": True,
            "text": text,
            "gold_sections": [
                {"label": "past_medical_history", "header": "zz past"},
                {"label": "family_history", "header": "zz family"},
                {"label": "plan", "header": "zz plan"},
            ],
            "gold_spans": [
                {"label": "CONDITION", "text": "asthma"},
                {"label": "CONDITION", "text": "diabetes"},
            ],
        },
    ]
    fixture.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    report = run_section_multilingual_eval(fixture)

    assert report.metrics["section_gate_passed"] is True
    assert report.metrics["section_label_f1"]["xx"] == pytest.approx(1.0)


def test_section_multilingual_eval_gate_and_recall_regression() -> None:
    report = run_section_multilingual_eval()

    assert report.metrics["section_gate_passed"] is True
    for language in REQUIRED_LANGUAGES:
        assert report.metrics["section_label_f1"][language] >= 0.85
        assert report.metrics["section_boundary_recall"][language] >= 0.85
        assert (
            report.metrics["section_character_recall"][language]
            >= report.metrics["section_character_recall_baseline"][language]
        )
