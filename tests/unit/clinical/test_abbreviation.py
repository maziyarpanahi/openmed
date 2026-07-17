"""Tests for deterministic clinical abbreviation disambiguation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import openmed.clinical.abbreviation as abbreviation_module
from openmed.clinical import (
    ABBREVIATION_DISAMBIGUATION_ADVISORY,
    AbbreviationDisambiguator,
    SenseInventory,
    disambiguate_abbreviation,
    expand_abbreviations,
    load_sense_inventory,
)

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "abbreviation_senses_gold.json"
)
STARTER = (
    Path(__file__).resolve().parents[3]
    / "openmed"
    / "clinical"
    / "data"
    / "abbreviation_senses_starter.json"
)


@pytest.mark.parametrize("case", json.loads(FIXTURE.read_text(encoding="utf-8")))
def test_golden_contexts_resolve_ambiguous_senses(case: dict) -> None:
    sense = disambiguate_abbreviation(
        case["short_form"],
        case["text"],
        section=case["section"],
        entity_types=case["entity_types"],
    )

    assert sense is not None, case["id"]
    assert sense.long_form == case["expected_long_form"], case["id"]
    assert 0.0 < sense.score <= 1.0
    assert sense.alternatives
    assert sense.matched_features


def test_scores_and_alternatives_are_deterministic_and_normalized() -> None:
    disambiguator = AbbreviationDisambiguator()
    results = [
        disambiguator.disambiguate(
            "PT",
            "PT is prolonged and the INR is elevated.",
            section="coagulation",
            entity_types=("lab_value",),
        )
        for _ in range(20)
    ]

    assert all(result == results[0] for result in results)
    sense = results[0]
    assert sense is not None
    assert sense.long_form == "prothrombin time"
    assert sum(
        [sense.score, *(item.score for item in sense.alternatives)]
    ) == pytest.approx(
        1.0,
        abs=2e-6,
    )


def test_unknown_short_form_returns_no_sense_without_error() -> None:
    assert disambiguate_abbreviation("XYZ", "XYZ noted in the source") is None

    text = "Unknown XYZ was copied from a legacy note."
    start = text.index("XYZ")
    annotations = expand_abbreviations(
        text,
        [{"start": start, "end": start + 3, "label": "ACRONYM"}],
    )

    assert len(annotations) == 1
    assert annotations[0].short_form == "XYZ"
    assert annotations[0].sense is None


def test_user_inventory_overrides_and_extends_starter(tmp_path: Path) -> None:
    custom_path = tmp_path / "custom-senses.json"
    custom_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provenance": {"source": "local synthetic test inventory"},
                "senses": {
                    "MS": [
                        {
                            "long_form": "multiple sclerosis",
                            "semantic_type": "custom_condition",
                            "source": "local-override",
                            "sections": ["neuroimmunology"],
                            "entity_types": ["custom_condition"],
                            "cue_words": ["optic neuritis"],
                            "prior": 0.9,
                        }
                    ],
                    "HCM": [
                        {
                            "long_form": "hypertrophic cardiomyopathy",
                            "semantic_type": "condition",
                            "source": "local-extension",
                            "sections": ["cardiology"],
                            "entity_types": ["condition"],
                            "cue_words": ["septal hypertrophy"],
                            "prior": 0.2,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    inventory = load_sense_inventory(custom_path)

    assert isinstance(inventory, SenseInventory)
    assert set(inventory) >= {"MS", "PT", "RA", "CA", "HCM"}
    assert len(inventory["MS"]) == 3
    assert inventory["MS"][0].source == "local-override"
    hcm = disambiguate_abbreviation(
        "hcm",
        "Septal hypertrophy supports HCM.",
        section="cardiology",
        inventory=inventory,
    )
    assert hcm is not None
    assert hcm.long_form == "hypertrophic cardiomyopathy"


def test_user_inventory_can_be_loaded_without_starter(tmp_path: Path) -> None:
    custom_path = tmp_path / "only-local.json"
    custom_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provenance": {"source": "local synthetic test inventory"},
                "senses": {
                    "ZZ": [
                        {
                            "long_form": "synthetic zeta zone",
                            "semantic_type": "synthetic_type",
                            "source": "local-only",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    inventory = load_sense_inventory(custom_path, include_starter=False)

    assert tuple(inventory) == ("ZZ",)


def test_expand_abbreviations_uses_section_and_nearby_entity_types() -> None:
    text = "Laboratory: PT was prolonged at 18 seconds with INR 1.8."
    pt_start = text.index("PT")
    value_start = text.index("18")
    spans = [
        {
            "start": pt_start,
            "end": pt_start + 2,
            "label": "ACRONYM",
            "section": "laboratory",
        },
        {
            "start": value_start,
            "end": value_start + 2,
            "label": "LAB_VALUE",
        },
    ]

    annotations = expand_abbreviations(text, spans)

    assert len(annotations) == 1
    annotation = annotations[0]
    assert (annotation.start, annotation.end) == (pt_start, pt_start + 2)
    assert annotation.section == "laboratory"
    assert annotation.sense is not None
    assert annotation.sense.long_form == "prothrombin time"
    assert "entity_type:lab_value" in annotation.sense.matched_features
    assert annotation.to_dict()["sense"]["long_form"] == "prothrombin time"


def test_invalid_span_offsets_fail_closed() -> None:
    with pytest.raises(ValueError, match="offsets must fall within text"):
        expand_abbreviations("MS", [{"start": 0, "end": 3}])


def test_starter_inventory_is_explicitly_synthetic_and_permissive() -> None:
    payload = json.loads(STARTER.read_text(encoding="utf-8"))
    provenance = payload["provenance"]

    assert provenance["restricted_data"] is False
    assert provenance["license"] == "CC0-1.0"
    assert "synthetic" in provenance["source"].casefold()
    assert "UMLS LRABR" in provenance["note"]
    assert "restricted vocabulary" in (abbreviation_module.__doc__ or "")


def test_public_advisory_requires_review() -> None:
    assert "review" in ABBREVIATION_DISAMBIGUATION_ADVISORY.casefold()
