from __future__ import annotations

import builtins
import json
import traceback
import urllib.request

import pytest

from openmed.agent.workflows import (
    PRIOR_AUTH_COMPLETENESS_SCHEMA,
    MissingEvidenceCode,
    PacketEvidence,
    PriorAuthCompletenessError,
    PriorAuthorizationPacket,
    PriorAuthRequirement,
    PriorAuthRequirementSchema,
    ReviewerActionCode,
    score_prior_authorization_packet,
)

PACKET_DIGEST = "sha256:" + "a" * 64
EVIDENCE_A = "sha256:" + "b" * 64
EVIDENCE_B = "sha256:" + "c" * 64
EVIDENCE_C = "sha256:" + "d" * 64
CITATION_A = "sha256:" + "e" * 64
CITATION_B = "sha256:" + "f" * 64


def _schema() -> PriorAuthRequirementSchema:
    return PriorAuthRequirementSchema(
        "payer.synthetic_imaging",
        3,
        (
            PriorAuthRequirement("clinical.prior_treatment"),
            PriorAuthRequirement("clinical.requested_service"),
        ),
    )


def _packet(
    evidence: tuple[PacketEvidence, ...],
    *,
    schema_id: str = "payer.synthetic_imaging",
    schema_version: int = 3,
) -> PriorAuthorizationPacket:
    return PriorAuthorizationPacket(
        PACKET_DIGEST,
        schema_id,
        schema_version,
        evidence,
    )


def _evidence(
    requirement_id: str,
    evidence_digest: str,
    *citations: str,
    contradiction_flag: bool = False,
) -> PacketEvidence:
    return PacketEvidence(
        requirement_id,
        evidence_digest,
        citations,
        contradiction_flag,
    )


def test_complete_packet_scores_one_without_authorizing_coverage() -> None:
    report = score_prior_authorization_packet(
        _packet(
            (
                _evidence("clinical.requested_service", EVIDENCE_A, CITATION_A),
                _evidence("clinical.prior_treatment", EVIDENCE_B, CITATION_B),
            )
        ),
        _schema(),
    )

    assert report.completeness_score == 1.0
    assert report.complete_requirement_count == 2
    assert not report.has_missing_evidence
    assert not report.requires_reviewer_action
    assert report.schema == PRIOR_AUTH_COMPLETENESS_SCHEMA
    assert "coverage" not in report.to_dict()
    assert "decision" not in report.to_dict()


def test_missing_evidence_and_citation_return_closed_codes_and_actions() -> None:
    report = score_prior_authorization_packet(
        _packet((_evidence("clinical.requested_service", EVIDENCE_A),)),
        _schema(),
    )

    assert report.completeness_score == 0.0
    assert [finding.to_dict() for finding in report.missing_evidence] == [
        {
            "code": MissingEvidenceCode.REQUIRED_CITATION_MISSING.value,
            "requirement_id": "clinical.requested_service",
        },
        {
            "code": MissingEvidenceCode.REQUIRED_EVIDENCE_MISSING.value,
            "requirement_id": "clinical.prior_treatment",
        },
    ]
    assert [action.to_dict() for action in report.reviewer_actions] == [
        {
            "code": ReviewerActionCode.ADD_REQUIRED_CITATION.value,
            "requirement_id": "clinical.requested_service",
        },
        {
            "code": ReviewerActionCode.PROVIDE_REQUIRED_EVIDENCE.value,
            "requirement_id": "clinical.prior_treatment",
        },
    ]


def test_minimum_evidence_count_contributes_to_completeness_score() -> None:
    schema = PriorAuthRequirementSchema(
        "payer.synthetic_therapy",
        1,
        (
            PriorAuthRequirement(
                "clinical.failed_therapies",
                minimum_evidence_items=2,
            ),
            PriorAuthRequirement("clinical.requested_service"),
        ),
    )
    packet = _packet(
        (
            _evidence("clinical.failed_therapies", EVIDENCE_A, CITATION_A),
            _evidence("clinical.requested_service", EVIDENCE_B, CITATION_B),
        ),
        schema_id="payer.synthetic_therapy",
        schema_version=1,
    )

    report = score_prior_authorization_packet(packet, schema)

    assert report.complete_requirement_count == 1
    assert report.completeness_score == 0.5
    assert (
        report.missing_evidence[0].code is MissingEvidenceCode.REQUIRED_EVIDENCE_MISSING
    )


def test_contradiction_and_unsupported_statement_request_review() -> None:
    packet = _packet(
        (
            _evidence(
                "clinical.requested_service",
                EVIDENCE_A,
                CITATION_A,
                contradiction_flag=True,
            ),
            _evidence("clinical.prior_treatment", EVIDENCE_B, CITATION_B),
            _evidence("packet.unsupported_statement", EVIDENCE_C, CITATION_A),
        )
    )

    report = score_prior_authorization_packet(packet, _schema())

    assert report.completeness_score == 1.0
    assert [action.to_dict() for action in report.reviewer_actions] == [
        {
            "code": ReviewerActionCode.REVIEW_CONTRADICTION.value,
            "requirement_id": "clinical.requested_service",
        },
        {
            "code": ReviewerActionCode.REVIEW_UNSUPPORTED_STATEMENT.value,
            "requirement_id": "packet.unsupported_statement",
        },
    ]


def test_optional_citation_does_not_create_a_false_gap() -> None:
    schema = PriorAuthRequirementSchema(
        "payer.synthetic_exception",
        1,
        (PriorAuthRequirement("administrative.routing", citation_required=False),),
    )
    packet = _packet(
        (_evidence("administrative.routing", EVIDENCE_A),),
        schema_id="payer.synthetic_exception",
        schema_version=1,
    )

    report = score_prior_authorization_packet(packet, schema)

    assert report.completeness_score == 1.0
    assert not report.missing_evidence


def test_scoring_is_order_independent_and_byte_deterministic() -> None:
    first = _evidence("clinical.requested_service", EVIDENCE_A, CITATION_A)
    second = _evidence("clinical.prior_treatment", EVIDENCE_B, CITATION_B)
    schema = _schema()

    report = score_prior_authorization_packet(_packet((first, second)), schema)
    reordered = score_prior_authorization_packet(_packet((second, first)), schema)

    assert report == reordered
    assert report.to_json() == reordered.to_json()
    assert json.loads(report.to_json()) == report.to_dict()
    assert report.report_digest.startswith("sha256:")


@pytest.mark.parametrize(
    ("schema_id", "schema_version", "code"),
    [
        ("payer.other_schema", 3, "schema_id_mismatch"),
        ("payer.synthetic_imaging", 4, "schema_version_mismatch"),
    ],
)
def test_packet_must_bind_the_exact_schema_version(
    schema_id: str,
    schema_version: int,
    code: str,
) -> None:
    packet = _packet((), schema_id=schema_id, schema_version=schema_version)

    with pytest.raises(PriorAuthCompletenessError) as caught:
        score_prior_authorization_packet(packet, _schema())

    assert caught.value.code == code


def test_duplicate_requirements_evidence_and_citations_fail_closed() -> None:
    requirement = PriorAuthRequirement("clinical.requested_service")
    with pytest.raises(PriorAuthCompletenessError, match="duplicate_requirement"):
        PriorAuthRequirementSchema("payer.synthetic", 1, (requirement, requirement))

    evidence = _evidence("clinical.requested_service", EVIDENCE_A, CITATION_A)
    with pytest.raises(PriorAuthCompletenessError, match="duplicate_evidence"):
        _packet((evidence, evidence))

    with pytest.raises(PriorAuthCompletenessError, match="duplicate_citation"):
        _evidence(
            "clinical.requested_service",
            EVIDENCE_A,
            CITATION_A,
            CITATION_A,
        )


@pytest.mark.parametrize(
    "build",
    [
        lambda value: PriorAuthRequirement(value),
        lambda value: PriorAuthRequirementSchema(
            value, 1, (PriorAuthRequirement("a"),)
        ),
        lambda value: PacketEvidence(value, EVIDENCE_A),
        lambda value: PacketEvidence("clinical.service", value),
        lambda value: PriorAuthorizationPacket(value, "payer.synthetic", 1, ()),
    ],
)
def test_rejected_values_never_appear_in_exception_chains(build) -> None:
    sentinel = "Synthetic Person Z99.999 /private/packet.json bearer_secret"

    with pytest.raises(PriorAuthCompletenessError) as caught:
        build(sentinel)

    rendered = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert sentinel not in rendered


def test_report_contains_no_raw_packet_or_citation_values() -> None:
    report = score_prior_authorization_packet(
        _packet((_evidence("clinical.requested_service", EVIDENCE_A),)),
        _schema(),
    )
    rendered = report.to_json()

    assert EVIDENCE_A not in rendered
    assert CITATION_A not in rendered
    assert set(report.to_dict()) == {
        "complete_requirement_count",
        "completeness_score",
        "evidence_metadata_digest",
        "has_missing_evidence",
        "missing_evidence",
        "packet_digest",
        "report_digest",
        "required_requirement_count",
        "requirement_schema_digest",
        "requires_reviewer_action",
        "reviewer_actions",
        "schema",
    }


def test_scoring_performs_no_file_or_network_io(monkeypatch) -> None:
    def unexpected_io(*_args, **_kwargs):
        raise AssertionError("completeness scoring must remain local and in-memory")

    monkeypatch.setattr(builtins, "open", unexpected_io)
    monkeypatch.setattr(urllib.request, "urlopen", unexpected_io)

    report = score_prior_authorization_packet(_packet(()), _schema())

    assert report.completeness_score == 0.0


def test_contract_is_available_from_workflows_public_api() -> None:
    import openmed.agent.workflows as workflows

    assert workflows.PriorAuthRequirement is PriorAuthRequirement
    assert workflows.PriorAuthorizationPacket is PriorAuthorizationPacket
    assert (
        workflows.score_prior_authorization_packet is score_prior_authorization_packet
    )
    assert workflows.PRIOR_AUTH_COMPLETENESS_SCHEMA == PRIOR_AUTH_COMPLETENESS_SCHEMA
