from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from openmed.eval.governance.blinded_adjudication import (
    ADJUDICATION_PACKET_SCHEMA_VERSION,
    AUDIT_MAPPING_MISMATCH,
    AUDIT_VALID,
    CONFLICT_SCREENING_RECUSED,
    BlindedAdjudicationError,
    BlindedCandidate,
    ComparisonCase,
    ConflictOfInterestMetadata,
    RubricCriterion,
    SourceEvidence,
    render_blinded_adjudication_packets,
    validate_packet_balance,
    verify_sealed_identity_mapping,
)


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('ascii')).hexdigest()}"


def _cases() -> tuple[ComparisonCase, ...]:
    return tuple(
        ComparisonCase(
            case_ref=f"case-{case_index:03d}",
            source_evidence=(
                SourceEvidence(
                    evidence_ref=f"source-{case_index:03d}",
                    content=f"Synthetic source evidence {case_index}.",
                ),
            ),
            candidate_outputs={
                "submission-red": f"Synthetic response {case_index}-1.",
                "submission-blue": f"Synthetic response {case_index}-2.",
                "submission-green": f"Synthetic response {case_index}-3.",
            },
        )
        for case_index in range(5)
    )


def _rubric() -> tuple[RubricCriterion, ...]:
    return (
        RubricCriterion(
            criterion_ref="evidence-grounding",
            question="How fully is the response supported by the supplied evidence?",
            minimum_score=1,
            maximum_score=5,
        ),
        RubricCriterion(
            criterion_ref="actionability",
            question="How directly does the response address the benchmark task?",
            minimum_score=1,
            maximum_score=5,
        ),
    )


def _conflict_metadata(**overrides: str) -> ConflictOfInterestMetadata:
    arguments = {
        "reviewer_ref": "reviewer-017",
        "declaration_digest": _digest("synthetic-conflict-declaration"),
    }
    arguments.update(overrides)
    return ConflictOfInterestMetadata(**arguments)


def _manifest_digests() -> dict[str, str]:
    return {
        identity: _digest(f"manifest-{identity}")
        for identity in ("submission-red", "submission-blue", "submission-green")
    }


def _render(**overrides):
    arguments = {
        "packet_set_ref": "adjudication-round-001",
        "cases": _cases(),
        "rubric": _rubric(),
        "conflict_of_interest": _conflict_metadata(),
        "holdout_commitment_digest": _digest("synthetic-holdout"),
        "submission_manifest_digests": _manifest_digests(),
        "randomization_key": b"synthetic-adjudication-key-value-01",
    }
    arguments.update(overrides)
    return render_blinded_adjudication_packets(**arguments)


def test_packets_are_deterministic_canonical_and_identity_blinded() -> None:
    first_packets, first_mapping = _render()
    second_packets, second_mapping = _render(
        cases=tuple(reversed(_cases())),
        submission_manifest_digests=dict(reversed(tuple(_manifest_digests().items()))),
    )

    assert first_packets == second_packets
    assert first_mapping == second_mapping
    assert tuple(packet.to_json() for packet in first_packets) == tuple(
        packet.to_json() for packet in second_packets
    )
    assert all(
        packet.to_json()
        == json.dumps(
            packet.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        for packet in first_packets
    )
    assert all(
        packet.schema_version == ADJUDICATION_PACKET_SCHEMA_VERSION
        for packet in first_packets
    )

    public_json = "\n".join(packet.to_json() for packet in first_packets)
    for hidden_identity in _manifest_digests():
        assert hidden_identity not in public_json
        assert _manifest_digests()[hidden_identity] not in public_json
    assert first_mapping.mapping_commitment in public_json


def test_packets_share_evidence_rubric_and_conflict_metadata_across_candidates() -> (
    None
):
    packets, _ = _render()

    for packet in packets:
        assert tuple(candidate.alias for candidate in packet.candidates) == (
            "candidate-a",
            "candidate-b",
            "candidate-c",
        )
        assert packet.source_evidence == next(
            case.source_evidence
            for case in _cases()
            if case.case_ref == packet.case_ref
        )
        assert packet.rubric == _rubric()
        assert packet.conflict_of_interest == _conflict_metadata()
        assert packet.conflict_of_interest.to_dict()["status"] == "cleared"


def test_candidate_positions_are_balanced_without_identity_disclosure() -> None:
    _, mapping = _render()

    report = validate_packet_balance(mapping)

    assert report.balanced is True
    assert report.packet_count == 5
    assert report.candidate_count == 3
    assert report.maximum_slot_imbalance == 1
    rendered = json.dumps(report.to_dict(), sort_keys=True)
    assert all(identity not in rendered for identity in _manifest_digests())


def test_randomization_key_changes_the_sealed_assignment() -> None:
    first_packets, first_mapping = _render()
    second_packets, second_mapping = _render(
        randomization_key=b"different-synthetic-key-value-0002"
    )

    assert first_mapping.mapping_commitment != second_mapping.mapping_commitment
    assert first_packets != second_packets


def test_private_mapping_is_separate_and_binds_manifests_and_outputs() -> None:
    packets, mapping = _render()

    private_document = mapping.to_private_dict()
    assert private_document["mapping_commitment"] == mapping.mapping_commitment
    assert {
        assignment["candidate_identity"]
        for entry in private_document["entries"]
        for assignment in entry["assignments"]
    } == set(_manifest_digests())
    assert {
        assignment["submission_manifest_digest"]
        for entry in private_document["entries"]
        for assignment in entry["assignments"]
    } == set(_manifest_digests().values())

    audit = verify_sealed_identity_mapping(
        packets,
        mapping,
        b"synthetic-adjudication-key-value-01",
    )
    assert audit.valid is True
    assert audit.reason_codes == (AUDIT_VALID,)


def test_audit_detects_tampered_candidate_content_without_echoing_it() -> None:
    packets, mapping = _render()
    private_value = "Synthetic private replacement content."
    changed_candidate = replace(packets[0].candidates[0], content=private_value)
    changed_packet = replace(
        packets[0],
        candidates=(changed_candidate, *packets[0].candidates[1:]),
    )

    audit = verify_sealed_identity_mapping(
        (changed_packet, *packets[1:]),
        mapping,
        b"synthetic-adjudication-key-value-01",
    )

    assert audit.valid is False
    assert audit.reason_codes == (AUDIT_MAPPING_MISMATCH,)
    assert private_value not in json.dumps(audit.to_dict(), sort_keys=True)


def test_recused_reviewer_fails_closed_before_packet_rendering() -> None:
    with pytest.raises(
        BlindedAdjudicationError, match="conflict_of_interest: reviewer_recused"
    ):
        _render(
            conflict_of_interest=_conflict_metadata(status=CONFLICT_SCREENING_RECUSED)
        )


@pytest.mark.parametrize(
    ("override", "reason"),
    (
        ({"randomization_key": b"short"}, "randomization_key: invalid_key"),
        (
            {"submission_manifest_digests": {"submission-red": _digest("only")}},
            "submission_manifest_digests: invalid_keys",
        ),
        ({"rubric": ()}, "rubric: invalid_sequence"),
    ),
)
def test_invalid_configuration_uses_closed_phi_safe_errors(
    override: dict[str, object], reason: str
) -> None:
    with pytest.raises(BlindedAdjudicationError, match=reason):
        _render(**override)


def test_invalid_values_are_not_reflected_in_exceptions() -> None:
    private_value = "synthetic-private-clinical-value"

    with pytest.raises(BlindedAdjudicationError) as raised:
        _render(
            submission_manifest_digests={
                **_manifest_digests(),
                "submission-red": private_value,
            }
        )
    assert private_value not in str(raised.value)

    with pytest.raises(BlindedAdjudicationError) as candidate_error:
        BlindedCandidate(alias=private_value, content="Synthetic output.")
    assert private_value not in str(candidate_error.value)


def test_cases_require_same_candidates_and_unique_public_references() -> None:
    inconsistent = replace(
        _cases()[0],
        candidate_outputs={
            "submission-red": "Synthetic response one.",
            "submission-blue": "Synthetic response two.",
        },
    )
    with pytest.raises(BlindedAdjudicationError, match="inconsistent_candidates"):
        _render(cases=(inconsistent, *_cases()[1:]))

    with pytest.raises(BlindedAdjudicationError, match="duplicate_reference"):
        _render(cases=(_cases()[0], _cases()[0]))
