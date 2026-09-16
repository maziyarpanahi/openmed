"""Tests for the synthetic RadGraph-style radiology evaluation."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from openmed.eval.datasets import (
    DUACredentialRequired,
    RadiologyEntityRelationFixture,
    license_for,
    load_radgraph_fixtures,
    load_synthetic_radiology_fixtures,
)
from openmed.eval.release_gates import (
    G13_STRICT_ENTITY_F1_FLOOR,
    G13_STRICT_RELATION_F1_FLOOR,
    G13_UNCERTAINTY_ACCURACY_FLOOR,
    evaluate_radiology_entity_relation_gate,
)
from openmed.eval.scorecard import ModelScorecard
from openmed.eval.suites.radiology_relations import (
    build_radiology_entity_relation_report,
    score_radiology_entity_relation_fixtures,
)


def test_radgraph_loader_requires_credentials_and_only_reads_supplied_rows(
    tmp_path,
) -> None:
    with pytest.raises(DUACredentialRequired, match="credentialed local path"):
        load_radgraph_fixtures()

    source = tmp_path / "radgraph.json"
    source.write_text(
        json.dumps(
            {
                "synthetic-dua-row": {
                    "text": "No edema at lung bases.",
                    "entities": {
                        "1": {
                            "tokens": "edema",
                            "label": "OBS-DA",
                            "start_ix": 1,
                            "end_ix": 1,
                            "relations": [["located_at", "2"]],
                        },
                        "2": {
                            "tokens": "lung bases",
                            "label": "ANAT-DP",
                            "start_ix": 3,
                            "end_ix": 4,
                            "relations": [],
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    original = source.read_bytes()

    fixtures = load_radgraph_fixtures(source)

    assert source.read_bytes() == original
    assert list(tmp_path.iterdir()) == [source]
    assert len(fixtures) == 1
    fixture = fixtures[0]
    assert fixture.entities["1"].text == "edema"
    assert fixture.entities["1"].metadata["uncertainty"] == "absent"
    assert fixture.gold_relations[0].relation_type == "LOCATED_AT"
    radgraph_license = license_for("radgraph")
    assert radgraph_license.redistribution == "user-supplied DUA/eval-only"
    assert "never downloads" in radgraph_license.notes


def test_synthetic_radiology_fixtures_score_strict_relaxed_and_breakdowns() -> None:
    fixtures = load_synthetic_radiology_fixtures()
    predictions = _perfect_predictions(fixtures)
    perfect = score_radiology_entity_relation_fixtures(fixtures, predictions)[
        "metrics"
    ]["radiology_entity_relation"]

    assert perfect["entity"]["strict"]["f1"] == 1.0
    assert perfect["entity"]["relaxed"]["f1"] == 1.0
    assert perfect["relation"]["strict"]["f1"] == 1.0
    assert perfect["relation"]["relaxed"]["f1"] == 1.0
    assert perfect["uncertainty"]["accuracy"] == 1.0
    assert evaluate_radiology_entity_relation_gate(
        {"radiology_entity_relation": perfect},
        {"radiology_entity_relation_required": True},
    ).passed
    assert all(fixture.metadata["synthetic"] is True for fixture in fixtures)
    assert all(
        "not a medical device"
        in fixture.metadata["medical_device_disclaimer"].casefold()
        for fixture in fixtures
    )

    nodule_fixture = next(
        fixture
        for fixture in fixtures
        if fixture.fixture_id == "radiology-suggestive-nodule"
    )
    nodule = nodule_fixture.entities["e-nodule"]
    drifted_nodule = replace(
        nodule,
        start=nodule.start - 1,
        text=nodule_fixture.text[nodule.start - 1 : nodule.end],
    )
    predictions[nodule_fixture.fixture_id]["entities"] = {
        **nodule_fixture.entities,
        "e-nodule": drifted_nodule,
    }
    predictions[nodule_fixture.fixture_id]["relations"] = [
        replace(
            relation,
            head=(
                drifted_nodule
                if relation.head.start == nodule.start
                and relation.head.end == nodule.end
                else relation.head
            ),
            tail=(
                drifted_nodule
                if relation.tail.start == nodule.start
                and relation.tail.end == nodule.end
                else relation.tail
            ),
        )
        for relation in nodule_fixture.gold_relations
    ]

    scored = score_radiology_entity_relation_fixtures(fixtures, predictions)
    metrics = scored["metrics"]["radiology_entity_relation"]

    assert metrics["entity"]["strict"]["f1"] == pytest.approx(7 / 8)
    assert metrics["entity"]["relaxed"]["f1"] == 1.0
    assert metrics["relation"]["strict"]["f1"] == pytest.approx(3 / 5)
    assert metrics["relation"]["relaxed"]["f1"] == 1.0
    assert set(metrics["relation"]["per_relation_type"]) == {
        "LOCATED_AT",
        "MODIFY",
        "SUGGESTIVE_OF",
    }
    assert metrics["uncertainty"]["accuracy"] == 1.0
    assert metrics["uncertainty"]["per_class"]["absent"]["total"] == 1
    assert metrics["uncertainty"]["per_class"]["uncertain"]["total"] == 2


def test_negated_finding_scored_present_fails_g13_and_is_reported() -> None:
    fixtures = load_synthetic_radiology_fixtures()
    predictions = _perfect_predictions(fixtures)
    negated_fixture = next(
        fixture
        for fixture in fixtures
        if fixture.fixture_id == "radiology-negated-consolidation"
    )
    consolidation = negated_fixture.entities["e-consolidation"]
    predictions[negated_fixture.fixture_id]["entities"] = {
        **negated_fixture.entities,
        "e-consolidation": replace(
            consolidation,
            metadata={**consolidation.metadata, "uncertainty": "present"},
        ),
    }

    report = build_radiology_entity_relation_report(
        fixtures,
        predictions,
        model_name="synthetic-radiology-model",
    )
    metrics = report.metrics["radiology_entity_relation"]
    check = evaluate_radiology_entity_relation_gate(report.metrics, report.metadata)

    assert metrics["entity"]["strict"]["f1"] == 1.0
    assert metrics["relation"]["strict"]["f1"] == 1.0
    assert metrics["uncertainty"]["accuracy"] == pytest.approx(4 / 5)
    assert metrics["uncertainty"]["accuracy"] < G13_UNCERTAINTY_ACCURACY_FLOOR
    assert check.gate == "G13"
    assert check.passed is False
    assert set(check.details["violations"]) == {"uncertainty_accuracy"}

    scorecard = ModelScorecard.from_reports([report])
    radiology = scorecard.to_dict()["device_tiers"][0]["radiology_entity_relation"]
    assert radiology["per_relation_type"]["LOCATED_AT"]["strict"] == 1.0
    assert radiology["per_uncertainty_class"]["absent"]["accuracy"] == 0.0
    markdown = scorecard.to_markdown()
    assert "Radiology Entity-and-Relation Evaluation" in markdown
    assert "absent: 0.00%" in markdown


@pytest.mark.parametrize(
    ("field", "floor", "violation"),
    [
        ("entity", G13_STRICT_ENTITY_F1_FLOOR, "strict_entity_f1"),
        ("relation", G13_STRICT_RELATION_F1_FLOOR, "strict_relation_f1"),
        ("uncertainty", G13_UNCERTAINTY_ACCURACY_FLOOR, "uncertainty_accuracy"),
    ],
)
def test_g13_fails_when_any_floor_is_breached(field, floor, violation) -> None:
    evidence = {
        "entity": {"strict": {"f1": 1.0}},
        "relation": {"strict": {"f1": 1.0}, "per_relation_type": {}},
        "uncertainty": {"accuracy": 1.0, "per_class": {}},
    }
    if field == "uncertainty":
        evidence[field]["accuracy"] = floor - 0.001
    else:
        evidence[field]["strict"]["f1"] = floor - 0.001

    check = evaluate_radiology_entity_relation_gate(
        {"radiology_entity_relation": evidence},
        {"radiology_entity_relation_required": True},
    )

    assert check.passed is False
    assert set(check.details["violations"]) == {violation}


def test_committed_radiology_fixture_validation_fails_closed() -> None:
    fixture = load_synthetic_radiology_fixtures()[0]
    payload = fixture.to_dict()
    payload["metadata"] = {
        **payload["metadata"],
        "synthetic": False,
    }

    with pytest.raises(ValueError, match="synthetic-only"):
        RadiologyEntityRelationFixture.from_mapping(
            payload,
            require_synthetic=True,
        )

    payload["metadata"] = {
        "synthetic": True,
        "medical_device_disclaimer": "Synthetic evaluation fixture.",
    }
    with pytest.raises(ValueError, match="medical-device disclaimer"):
        RadiologyEntityRelationFixture.from_mapping(
            payload,
            require_synthetic=True,
        )


def _perfect_predictions(fixtures):
    return {
        fixture.fixture_id: {
            "entities": dict(fixture.entities),
            "relations": list(fixture.gold_relations),
        }
        for fixture in fixtures
    }
