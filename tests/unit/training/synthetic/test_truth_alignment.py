"""Cross-artifact truth controls over existing immutable Journey records."""

from dataclasses import replace

import pytest

from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    ClinicalFact,
    EvidenceLocator,
)
from openmed.training.synthetic.truth_alignment import (
    SyntheticArtifactView,
    validate_truth_alignment,
)


def package():
    artifact = ClinicalArtifact(
        "artifact_aaaaaaaaaaaaaaaa",
        "synthetic_note",
        "text/plain",
        "0" * 64,
        10,
        "source_aaaaaaaaaaaaaaaa",
        "2000-01-01T00:00:00Z",
        attributes={"synthetic": True},
    )
    locator = EvidenceLocator(
        "evidence_aaaaaaaaaaaaaaaa",
        "artifact_aaaaaaaaaaaaaaaa",
        "text_span",
        {"start": 0, "end": 4},
    )
    fact = ClinicalFact(
        "fact_aaaaaaaaaaaaaaaa",
        "subject_aaaaaaaaaaaaaaaa",
        "medication",
        {"synthetic_code": "drug_a"},
        "synthetic",
        ("evidence_aaaaaaaaaaaaaaaa",),
        "1" * 64,
        effective_time={"date": "2000-01-01"},
        attributes={"synthetic": True},
    )
    return fact, locator, SyntheticArtifactView("note", artifact, (fact,), (locator,))


def validate(fact, locator, view, **kwargs):
    return validate_truth_alignment(
        [fact], [locator], [view], expected_modalities=["note"], **kwargs
    )


def test_consistent_package_passes_without_new_truth_schema():
    fact, locator, view = package()
    assert validate(fact, locator, view)["valid_for_training"] is True


@pytest.mark.parametrize(
    "changes,category",
    [
        ({"value": {"synthetic_code": "drug_b"}}, "fact_value"),
        ({"effective_time": {"date": "2000-01-02"}}, "effective_time"),
        ({"fact_type": "condition"}, "fact_label"),
    ],
)
def test_medication_time_and_label_each_fail_their_check(changes, category):
    fact, locator, view = package()
    report = validate(fact, locator, replace(view, facts=(replace(fact, **changes),)))
    assert [row["category"] for row in report["findings"]] == [category]
    assert report["valid_for_training"] is False
    assert "drug_b" not in str(report)


def test_altered_rendered_coordinate_fails_separately():
    fact, locator, view = package()
    altered = replace(locator, location={"start": 1, "end": 4})
    result = validate(fact, locator, replace(view, evidence=(altered,)))
    assert result["findings"][0]["category"] == "coordinates"


def test_missing_modalities_are_reported_without_fabrication():
    fact, locator, view = package()
    result = validate_truth_alignment([fact], [locator], [view])
    assert result["artifact_count"] == 1
    assert result["missing_modalities"] == ["fhir", "image", "omop", "waveform"]
    assert result["valid_for_training"] is False


def test_named_negative_control_cannot_enter_normal_training():
    fact, locator, view = package()
    altered = replace(view, facts=(replace(fact, value={"synthetic_code": "drug_b"}),))
    args = dict(
        negative_control_name="altered_medication", expected_corruptions=["fact_value"]
    )
    with pytest.raises(ValueError, match="normal partition"):
        validate(fact, locator, altered, **args)
    result = validate(fact, locator, altered, partition="negative_control", **args)
    assert result["negative_control_passed"] is True
    assert result["valid_for_training"] is False


def test_missing_evidence_and_missing_fact_are_not_ignored():
    fact, locator, view = package()
    assert (
        validate(fact, locator, replace(view, evidence=()))["findings"][0]["category"]
        == "missing_evidence"
    )
    assert (
        validate(fact, locator, replace(view, facts=()))["findings"][0]["category"]
        == "missing_fact"
    )


def test_all_declared_modalities_share_one_existing_truth_graph():
    fact, locator, view = package()
    views = []
    expected_evidence = []
    for index, modality in enumerate(["note", "fhir", "omop", "image", "waveform"]):
        artifact = replace(view.artifact, artifact_id=f"artifact_{index:016x}")
        evidence = replace(
            locator,
            locator_id=f"evidence_{index:016x}",
            artifact_id=artifact.artifact_id,
        )
        expected_evidence.append(evidence)
        views.append(
            SyntheticArtifactView(
                modality,
                artifact,
                (replace(fact, evidence_ids=(evidence.locator_id,)),),
                (evidence,),
            )
        )
    shared_fact = replace(
        fact, evidence_ids=tuple(item.locator_id for item in expected_evidence)
    )
    result = validate_truth_alignment([shared_fact], expected_evidence, views)
    assert result["valid_for_training"] is True
    assert result["artifact_count"] == 5
    assert result["missing_modalities"] == []


def test_valid_locator_from_another_truth_fact_cannot_be_substituted():
    fact, locator, view = package()
    other = replace(locator, locator_id="evidence_" + "b" * 16)
    substituted = replace(
        view,
        facts=(replace(fact, evidence_ids=(other.locator_id,)),),
        evidence=(other,),
    )
    result = validate_truth_alignment(
        [fact], [locator, other], [substituted], expected_modalities=["note"]
    )
    assert "missing_evidence" in {row["category"] for row in result["findings"]}


def test_omitted_fact_evidence_binding_is_a_named_negative_control():
    fact, locator, view = package()
    other = replace(
        locator, locator_id="evidence_" + "b" * 16, artifact_id="artifact_" + "b" * 16
    )
    fact = replace(fact, evidence_ids=(locator.locator_id, other.locator_id))
    missing = replace(
        view,
        facts=(replace(fact, evidence_ids=(other.locator_id,)),),
        evidence=(other,),
    )
    result = validate_truth_alignment(
        [fact], [locator, other], [missing], expected_modalities=["note"]
    )
    assert result["valid_for_training"] is False
    assert "missing_evidence" in {row["category"] for row in result["findings"]}
    result = validate_truth_alignment(
        [fact],
        [locator, other],
        [missing],
        expected_modalities=["note"],
        partition="negative_control",
        negative_control_name="omitted_binding",
        expected_corruptions=["missing_evidence", "artifact_reference"],
    )
    assert result["negative_control_passed"] is True
    assert result["valid_for_training"] is False


def test_partial_fact_evidence_binding_fails():
    fact, locator, view = package()
    second = replace(locator, locator_id="evidence_" + "b" * 16)
    shared = replace(fact, evidence_ids=(locator.locator_id, second.locator_id))
    result = validate_truth_alignment(
        [shared], [locator, second], [view], expected_modalities=["note"]
    )
    assert result["valid_for_training"] is False
    assert "missing_evidence" in {row["category"] for row in result["findings"]}


@pytest.mark.parametrize("expected,views", [([], True), (["note"], False)])
def test_no_comparison_cannot_qualify(expected, views):
    fact, locator, view = package()
    with pytest.raises(ValueError):
        validate_truth_alignment(
            [fact], [locator], [view] if views else [], expected_modalities=expected
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"subject_id": "subject_" + "b" * 16},
        {"encounter_id": "encounter_" + "b" * 16},
    ],
)
def test_shared_fact_cannot_move_to_another_patient_or_encounter(changes):
    fact, locator, view = package()
    altered = replace(view, facts=(replace(fact, **changes),))
    report = validate(fact, locator, altered)
    assert report["valid_for_training"] is False
    assert [row["category"] for row in report["findings"]] == ["fact_identity"]
    assert next(iter(changes.values())) not in str(report)
    control = validate(
        fact,
        locator,
        altered,
        partition="negative_control",
        negative_control_name="altered_fact_identity",
        expected_corruptions=["fact_identity"],
    )
    assert control["negative_control_passed"] is True
    assert control["valid_for_training"] is False
