"""Tests for openEHR flat-JSON COMPOSITION export."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from openmed.clinical.exporters.openehr import (
    extract_round_trip_coded_values,
    parse_operational_template,
    to_openehr_composition,
    validate_openehr_composition,
)
from openmed.interop.cdm_etl import ClinicalEntity

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "fixtures" / "openehr"
TEMPLATE = FIXTURES / "openmed_grounded_sample_webtemplate.json"
ENTITIES = FIXTURES / "grounded_entities.json"


def _fixture_payload() -> dict:
    return json.loads(ENTITIES.read_text(encoding="utf-8"))


def test_openehr_export_validates_against_sample_webtemplate() -> None:
    payload = _fixture_payload()

    composition = to_openehr_composition(
        payload["entities"],
        operational_template=TEMPLATE,
        doc_id=payload["doc_id"],
        source_text=payload["source_text"],
        time="2026-01-01T00:00:00+00:00",
        vocabulary_key="local-user-vocab",
    )
    result = validate_openehr_composition(
        composition,
        TEMPLATE,
        source_text=payload["source_text"],
    )

    assert result.ok
    assert result.out_of_template_paths == ()
    assert all(
        path.startswith(("ctx/", "openmed_grounded_clinical_data/"))
        for path in composition
    )


def test_every_non_empty_element_has_feeder_audit_source_span() -> None:
    payload = _fixture_payload()

    composition = to_openehr_composition(
        payload["entities"],
        operational_template=TEMPLATE,
        doc_id=payload["doc_id"],
        source_text=payload["source_text"],
        vocabulary_key="local-user-vocab",
    )
    element_bases = {
        path.split("|", 1)[0]
        for path, value in composition.items()
        if value not in ("", None)
        and not path.startswith("ctx/")
        and "/_feeder_audit/" not in path
    }

    assert element_bases
    assert payload["doc_id"] not in json.dumps(composition, sort_keys=True)
    for base in element_bases:
        pointer = composition[f"{base}/_feeder_audit/originating_system_item_ids:0|id"]
        assert pointer.startswith("sha256:")
        assert payload["doc_id"] not in pointer
        _, offsets = pointer.rsplit(":", 1)
        start, end = [int(part) for part in offsets.split("-", 1)]
        assert payload["source_text"][start:end]


def test_grounding_codes_require_user_vocabulary_key() -> None:
    payload = _fixture_payload()

    text_only = to_openehr_composition(
        payload["entities"],
        operational_template=TEMPLATE,
        doc_id=payload["doc_id"],
        source_text=payload["source_text"],
    )
    coded = to_openehr_composition(
        payload["entities"],
        operational_template=TEMPLATE,
        doc_id=payload["doc_id"],
        source_text=payload["source_text"],
        vocabulary_key="local-user-vocab",
    )

    assert not any(path.endswith("|code") for path in text_only)
    assert "type 2 diabetes" in text_only.values()
    assert any(path.endswith("|code") for path in coded)
    assert (
        coded[
            "openmed_grounded_clinical_data/problems/problem_diagnosis:0/"
            "problem_code|code"
        ]
        == "44054006"
    )


def test_canonical_clinical_entity_vocabulary_id_is_exported_when_enabled() -> None:
    entity = ClinicalEntity(
        label="condition",
        text="type 2 diabetes",
        start=0,
        end=15,
        code="44054006",
        vocabulary_id="SNOMED",
    )

    composition = to_openehr_composition(
        [entity],
        operational_template=TEMPLATE,
        doc_id="synthetic-note",
        source_text="type 2 diabetes",
        vocabulary_key="local-user-vocab",
    )

    code_path = (
        "openmed_grounded_clinical_data/problems/problem_diagnosis:0/problem_code"
    )
    assert composition[f"{code_path}|code"] == "44054006"
    assert composition[f"{code_path}|terminology"] == "SNOMED-CT"


def test_round_trip_flat_validator_preserves_codes_and_units() -> None:
    payload = _fixture_payload()

    composition = to_openehr_composition(
        payload["entities"],
        operational_template=TEMPLATE,
        doc_id=payload["doc_id"],
        source_text=payload["source_text"],
        vocabulary_key="local-user-vocab",
    )
    values = extract_round_trip_coded_values(composition)

    assert {
        (item.get("code"), item.get("terminology")) for item in values if "code" in item
    } == {
        ("44054006", "SNOMED-CT"),
        ("6809", "RxNorm"),
        ("2345-7", "LOINC"),
        ("8867-4", "LOINC"),
    }
    assert {
        (item.get("magnitude"), item.get("unit"))
        for item in values
        if "magnitude" in item
    } == {(145, "mg/dL"), (88, "/min")}
    assert values == extract_round_trip_coded_values(
        dict(reversed(composition.items()))
    )


def test_validator_rejects_out_of_template_paths() -> None:
    payload = _fixture_payload()
    composition = to_openehr_composition(
        payload["entities"][:1],
        operational_template=TEMPLATE,
        doc_id=payload["doc_id"],
        source_text=payload["source_text"],
    )
    composition["openmed_grounded_clinical_data/problems/forbidden/path"] = (
        "not allowed"
    )

    result = validate_openehr_composition(
        composition,
        TEMPLATE,
        source_text=payload["source_text"],
    )

    assert not result.ok
    assert result.out_of_template_paths == (
        "openmed_grounded_clinical_data/problems/forbidden/path",
    )


def test_webtemplate_tree_id_drives_flat_paths() -> None:
    payload = _fixture_payload()
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    template["templateId"] = "Human-readable template name"

    parsed = parse_operational_template(template)
    composition = to_openehr_composition(
        payload["entities"][:1],
        operational_template=template,
        doc_id=payload["doc_id"],
        source_text=payload["source_text"],
    )

    assert parsed.template_id == "Human-readable template name"
    assert parsed.root_id == "openmed_grounded_clinical_data"
    assert any(
        path.startswith("openmed_grounded_clinical_data/problems/")
        for path in composition
    )
    assert not any(
        path.startswith("Human-readable template name/") for path in composition
    )


def test_binding_must_match_webtemplate_archetype_constraint() -> None:
    payload = _fixture_payload()
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    incompatible = copy.deepcopy(template)
    incompatible["tree"]["children"][0]["children"][0]["nodeId"] = (
        "openEHR-EHR-EVALUATION.adverse_reaction_risk.v1"
    )

    with pytest.raises(ValueError, match="problem_diagnosis.v1"):
        to_openehr_composition(
            payload["entities"][:1],
            operational_template=incompatible,
            doc_id=payload["doc_id"],
            source_text=payload["source_text"],
        )


def test_zero_length_source_span_is_rejected() -> None:
    payload = _fixture_payload()
    entity = dict(payload["entities"][0], end=payload["entities"][0]["start"])

    with pytest.raises(ValueError, match="non-empty source span"):
        to_openehr_composition(
            [entity],
            operational_template=TEMPLATE,
            doc_id=payload["doc_id"],
            source_text=payload["source_text"],
        )


def test_parse_operational_template_rejects_xml_without_flat_paths() -> None:
    with pytest.raises(ValueError, match="WebTemplate JSON"):
        parse_operational_template("<template></template>")
