"""Focused offline tests for validated SNOMED CT post-coordination."""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

from eval.suites.postcoordinated_expressions import (
    SYNTHETIC_EXPRESSION_GOLD,
    evaluate_postcoordinated_expressions,
    synthetic_ecl_validator,
)
from openmed.clinical.exporters.codeable_concept import to_codeable_concept
from openmed.clinical.exporters.fhir import (
    POSTCOORDINATED_CODING_PROVENANCE_EXTENSION_URL,
    postcoordinated_codeable_concept,
    to_fhir,
)
from openmed.clinical.grounding import (
    Candidate,
    ConceptReference,
    ECLConstraint,
    ECLValidationError,
    ECLValidator,
    GroundedSpan,
    PostCoordinationStage,
    Refinement,
    RestrictedVocabularyError,
    RulesPostCoordinationDecomposer,
    VocabLoader,
    VocabSource,
    build_expression,
    ground,
)

_EDITION = "https://synthetic.invalid/sct/edition/20260101"
_FOCUS = ConceptReference("810901")
_ATTRIBUTES = {
    "laterality": ConceptReference("700101"),
    "severity": ConceptReference("700102"),
    "morphology": ConceptReference("700103"),
    "causative_agent": ConceptReference("700104"),
}
_VALUES = {
    "laterality": ConceptReference("820101"),
    "severity": ConceptReference("820102"),
    "morphology": ConceptReference("820103"),
    "causative_agent": ConceptReference("820104"),
}
_FRAGMENTS = {
    "left modifier": "laterality",
    "severe modifier": "severity",
    "shape modifier": "morphology",
    "agent modifier": "causative_agent",
}
_COMPOSITE = (
    "synthetic focus, left modifier, severe modifier, shape modifier, agent modifier"
)


class _Resolver:
    def __init__(self, *, reject_slot: str | None = None) -> None:
        self.reject_slot = reject_slot

    def matches(self, concept_id: str, constraint: str, edition_uri: str) -> bool:
        if edition_uri != _EDITION:
            return False
        if constraint == "<< 710101":
            return concept_id == _FOCUS.concept_id
        for slot, value in _VALUES.items():
            if constraint == f"<< {720101 + list(_VALUES).index(slot)}":
                return slot != self.reject_slot and concept_id == value.concept_id
        return False


def _validator(*, reject_slot: str | None = None) -> ECLValidator:
    constraints = {
        slot: ECLConstraint(
            slot=slot,
            attribute_id=attribute.concept_id,
            value_domain=f"<< {720101 + index}",
            focus_domain="<< 710101",
        )
        for index, (slot, attribute) in enumerate(_ATTRIBUTES.items())
    }
    return ECLValidator(
        edition_uri=_EDITION,
        constraints=constraints,
        resolver=_Resolver(reject_slot=reject_slot),
    )


def _decomposer() -> RulesPostCoordinationDecomposer:
    def focus(fragment):
        return _FOCUS if fragment.text == "synthetic focus" else None

    def refinement(fragment):
        slot = _FRAGMENTS.get(fragment.text)
        if slot is None:
            return None
        return Refinement(slot, _ATTRIBUTES[slot], _VALUES[slot])

    return RulesPostCoordinationDecomposer(focus, refinement)


def _stage(
    *,
    threshold: float = 0.75,
    reject_slot: str | None = None,
) -> PostCoordinationStage:
    return PostCoordinationStage(
        license_key="synthetic-user-license-proof",
        validator=_validator(reject_slot=reject_slot),
        decomposer=_decomposer(),
        precoordination_threshold=threshold,
    )


def _free_loader(tmp_path: Path) -> VocabLoader:
    path = tmp_path / "synthetic-free-vocab.jsonl"
    path.write_text(
        json.dumps(
            {
                "code": "SYN-PRE-1",
                "preferred_term": "precoordinated finding",
                "synonyms": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return VocabLoader(
        cache_dir=tmp_path / "cache",
        registry={
            "icd10cm": VocabSource(
                system="icd10cm",
                path=path,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        },
    )


def test_build_expression_is_deterministic_and_ecl_conformant_for_gold() -> None:
    validator = synthetic_ecl_validator()
    expressions = [
        build_expression(case.focus, case.refinements, validator=validator)
        for case in SYNTHETIC_EXPRESSION_GOLD
    ]

    assert len(expressions) >= 25
    assert [item.to_scg() for item in expressions] == [
        case.expected_expression for case in SYNTHETIC_EXPRESSION_GOLD
    ]
    assert all(validator.validate(item).valid for item in expressions)


def test_decomposer_reuses_composite_offsets_for_all_attribute_types() -> None:
    decomposition = _decomposer().decompose(_COMPOSITE, start=9, byte_start=17)

    assert decomposition is not None
    assert decomposition.normalization.strategy == "coordination"
    assert decomposition.focus_mention.start == 9
    assert {item.refinement.slot for item in decomposition.refinements} == set(
        _ATTRIBUTES
    )
    assert [item.mention.text for item in decomposition.refinements] == list(_FRAGMENTS)
    for item in decomposition.refinements:
        local_start = item.mention.start - 9
        local_end = item.mention.end - 9
        assert _COMPOSITE[local_start:local_end] == item.mention.text


def test_invalid_ecl_composition_is_rejected_and_never_emitted() -> None:
    refinement = Refinement(
        "laterality",
        _ATTRIBUTES["laterality"],
        _VALUES["laterality"],
    )
    with pytest.raises(ECLValidationError, match="outside the allowed ECL domain"):
        build_expression(
            _FOCUS,
            (refinement,),
            validator=_validator(reject_slot="laterality"),
        )

    rejected = _stage(reject_slot="laterality").apply(
        GroundedSpan(_COMPOSITE, 0, len(_COMPOSITE), canonical_label="CONDITION")
    )
    assert rejected.abstained is True
    assert rejected.candidates == ()
    assert (
        "value_outside_domain"
        in rejected.provenance["snomed_postcoordination"]["reasons"]
    )


def test_postcoordination_requires_key_and_does_not_retain_it() -> None:
    with pytest.raises(RestrictedVocabularyError, match="user-supplied"):
        PostCoordinationStage(
            license_key="",
            validator=_validator(),
            decomposer=_decomposer(),
        )

    stage = _stage()
    assert "license_key" not in vars(stage)
    assert "synthetic-user-license-proof" not in repr(vars(stage))


def test_mixed_grounding_prefers_precoordinated_and_composes_only_abstention(
    tmp_path: Path,
) -> None:
    results = ground(
        [
            {"text": "precoordinated finding", "label": "condition"},
            {"text": _COMPOSITE, "label": "condition"},
        ],
        systems=["icd10cm"],
        loader=_free_loader(tmp_path),
        postcoordination=_stage(),
    )

    assert results[0].codes == {"icd10cm": "SYN-PRE-1"}
    assert results[0].candidates[0].source != "post-coordinated"
    assert results[1].codes["snomed"].startswith(f"{_FOCUS.concept_id} :")
    assert results[1].candidates[0].source == "post-coordinated"
    assert results[1].provenance["snomed_postcoordination"]["validated"] is True


def test_low_score_triggers_composition_but_sufficient_score_is_preferred() -> None:
    low = GroundedSpan(
        _COMPOSITE,
        0,
        len(_COMPOSITE),
        candidates=(Candidate("SNOMED", "811101", "lookup", 0.4),),
    )
    high = GroundedSpan(
        _COMPOSITE,
        0,
        len(_COMPOSITE),
        candidates=(Candidate("SNOMED", "811102", "lookup", 0.9),),
    )

    composed = _stage().apply(low)
    preferred = _stage().apply(high)

    assert composed.candidates[0].source == "post-coordinated"
    assert preferred == high

    calibrated_low = GroundedSpan(
        _COMPOSITE,
        0,
        len(_COMPOSITE),
        candidates=(Candidate("SNOMED", "811103", "lookup", 0.99),),
        calibrated_score=0.2,
    )
    assert _stage().apply(calibrated_low).candidates[0].source == "post-coordinated"


def test_fhir_codeable_concept_marks_expression_as_composed() -> None:
    composed = _stage().apply(
        GroundedSpan(
            _COMPOSITE,
            0,
            len(_COMPOSITE),
            canonical_label="CONDITION",
        )
    )
    concept = to_codeable_concept(composed)
    coding = concept["coding"][0]

    assert coding["system"] == "http://snomed.info/sct"
    assert coding["version"] == _EDITION
    provenance = next(
        item
        for item in coding["extension"]
        if item["url"] == POSTCOORDINATED_CODING_PROVENANCE_EXTENSION_URL
    )
    assert {
        item["url"]: item.get("valueCode", item.get("valueBoolean"))
        for item in provenance["extension"]
    } == {
        "origin": "composed",
        "eclValidated": True,
    }

    condition = to_fhir(composed)
    assert condition is not None
    exported_extensions = condition["code"]["coding"][0]["extension"]
    assert any(
        item["url"] == POSTCOORDINATED_CODING_PROVENANCE_EXTENSION_URL
        for item in exported_extensions
    )

    expression = build_expression(
        _FOCUS,
        (
            Refinement(
                "laterality",
                _ATTRIBUTES["laterality"],
                _VALUES["laterality"],
            ),
        ),
        validator=_validator(),
    )
    direct = postcoordinated_codeable_concept(
        expression,
        validator=_validator(),
        text="synthetic finding",
    )
    assert direct["coding"][0]["code"] == expression.to_scg()
    with pytest.raises(ECLValidationError):
        postcoordinated_codeable_concept(
            expression,
            validator=_validator(reject_slot="laterality"),
        )


def test_synthetic_eval_reports_required_metrics_without_network(monkeypatch) -> None:
    def fail_socket(*args, **kwargs):
        raise AssertionError("network egress attempted")

    monkeypatch.setattr(socket, "socket", fail_socket)
    report = evaluate_postcoordinated_expressions()

    assert report["case_count"] >= 25
    assert report["expression_exact_match"] >= 0.70
    assert report["validation_rate"] == 1.0
    assert set(report["attribute_slot_f1"]) == set(_ATTRIBUTES)
    assert all(value == 1.0 for value in report["attribute_slot_f1"].values())
    assert report["metadata"] == {
        "offline": True,
        "synthetic": True,
        "ships_terminology_content": False,
    }


def test_runtime_package_contains_no_snomed_edition_artifacts() -> None:
    package_root = Path(__file__).resolve().parents[4] / "openmed"
    rf2_markers = (
        "sct2_concept_",
        "sct2_description_",
        "sct2_relationship_",
        "der2_refset_",
        "snapshot_terminology",
        "full_terminology",
        "delta_terminology",
    )

    bundled = [
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file()
        and any(marker in path.name.casefold() for marker in rf2_markers)
    ]
    assert bundled == []

    rf2_headers = (
        b"id\teffectiveTime\tactive\tmoduleId\tdefinitionStatusId",
        b"id\teffectiveTime\tactive\tmoduleId\tconceptId\tlanguageCode",
        b"id\teffectiveTime\tactive\tmoduleId\tsourceId\tdestinationId",
    )
    content_matches = [
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file()
        and path.suffix not in {".py", ".pyi", ".pyc"}
        and any(header in path.read_bytes() for header in rf2_headers)
    ]
    assert content_matches == []
