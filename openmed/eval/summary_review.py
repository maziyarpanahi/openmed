"""Local, digest-bound returns from existing blinded adjudication packets.

Private bindings stay in a separately sealed document. Public reports contain
only aggregates and digests; synthetic reviews never qualify a release gate.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openmed.eval.citation_support_metrics import (
    ADJUDICATION_LABELS,
    AtomicClaim,
    CitationSupportReport,
    ClinicianAdjudication,
    EvidenceSpan,
    compute_citation_support_metrics,
)
from openmed.eval.governance.blinded_adjudication import (
    BlindedAdjudicationPacket,
    SealedIdentityMapping,
    verify_sealed_identity_mapping,
)
from openmed.eval.reviewer_disagreement import (
    DisagreementReason,
    ReviewerDecision,
    ReviewerDisagreementReport,
    reviewer_disagreement_report,
)

REVIEW_SCHEMA = "openmed.eval.summary_review.v1"
_MAX_ROWS = 100_000


class SummaryReviewError(ValueError):
    """A controlled, value-free review import failure."""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def artifact_digest(text: str) -> str:
    """Hash an exact local UTF-8 artifact without retaining its content."""
    if type(text) is not str:
        raise SummaryReviewError("invalid_artifact")
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest(value: Any) -> str:
    return artifact_digest(_json(value))


def _is_digest(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 71
        and value.startswith("sha256:")
        and all(c in "0123456789abcdef" for c in value[7:])
    )


def _seal(value: Any, key: bytes) -> str:
    if type(key) is not bytes or len(key) < 32:
        raise SummaryReviewError("invalid_sealing_key")
    return (
        "sha256:"
        + hmac.new(
            key, REVIEW_SCHEMA.encode() + b"\0" + _json(value).encode(), hashlib.sha256
        ).hexdigest()
    )


def _rows(value: Any) -> list:
    if type(value) not in (list, tuple) or len(value) > _MAX_ROWS:
        raise SummaryReviewError("invalid_collection")
    return list(value)


def _fields(value: Any, fields: set[str]) -> None:
    if type(value) is not dict or set(value) != fields:
        raise SummaryReviewError("invalid_record")


@dataclass(frozen=True, slots=True, repr=False)
class ReviewBinding:
    """Private link from an existing packet candidate to a citation edge."""

    case_ref: str
    candidate_identity: str
    claim_id: str
    evidence_id: str

    def __repr__(self) -> str:
        return "ReviewBinding(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class SealedSummaryReview:
    """Access-controlled JSON bindings authenticated with an evaluator key.

    This seal authenticates integrity; it does not encrypt private bindings.
    Never publish ``private_json`` or attach it to reviewer packets.
    """

    private_json: str
    commitment: str

    def __repr__(self) -> str:
        return "SealedSummaryReview(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class ImportedSummaryReview:
    """Accepted metric inputs and aggregate review states, without public IDs."""

    citation_support: CitationSupportReport
    reviewer_agreement: ReviewerDisagreementReport | None
    adjudications: tuple[ClinicianAdjudication, ...]
    states: tuple[tuple[str, int], ...]
    evidence_kind: str
    review_evidence_digest: str
    request_digest: str
    summary_digest: str
    source_digest: str
    claim_digest: str
    evidence_digest: str
    decisions_digest: str

    def __post_init__(self) -> None:
        if (
            self.evidence_kind not in ("synthetic", "reviewer")
            or any(
                not _is_digest(getattr(self, field))
                for field in (
                    "review_evidence_digest",
                    "request_digest",
                    "summary_digest",
                    "source_digest",
                    "claim_digest",
                    "evidence_digest",
                    "decisions_digest",
                )
            )
            or type(self.states) is not tuple
            or len(self.states) != 5
            or {state for state, _ in self.states}
            != {"accepted", "missing", "incomplete", "disputed", "unclear"}
            or any(type(count) is not int or count < 0 for _, count in self.states)
            or type(self.citation_support) is not CitationSupportReport
            or (
                self.reviewer_agreement is not None
                and type(self.reviewer_agreement) is not ReviewerDisagreementReport
            )
            or type(self.adjudications) is not tuple
            or len(self.adjudications) != dict(self.states)["accepted"]
            or any(type(row) is not ClinicianAdjudication for row in self.adjudications)
        ):
            raise SummaryReviewError("invalid_review_result")

    def __repr__(self) -> str:
        return "ImportedSummaryReview(<redacted>)"

    @property
    def complete(self) -> bool:
        """Return whether every requested edge has a usable final decision."""
        return (
            bool(self.states)
            and all(count == 0 for state, count in self.states if state != "accepted")
            and dict(self.states).get("accepted", 0) > 0
            and self.citation_support.adjudication.unadjudicated_claim_count == 0
            and self.citation_support.adjudication.unadjudicated_citation_count == 0
        )

    @property
    def reviewer_evidence_available(self) -> bool:
        """Return completeness of caller-supplied reviewer evidence, not credentials."""
        return self.complete and self.evidence_kind == "reviewer"

    def matches(
        self, claims: Sequence, evidence: Sequence, *, summary: str, source: str
    ) -> bool:
        """Check current evaluated artifacts and citation inputs for drift."""
        machine = compute_citation_support_metrics(claims, evidence)
        reviewed = compute_citation_support_metrics(
            claims, evidence, adjudications=self.adjudications
        )
        return (
            not machine.adjudication.available
            and machine.claim_digest == self.claim_digest
            and machine.evidence_digest == self.evidence_digest
            and artifact_digest(summary) == self.summary_digest
            and artifact_digest(source) == self.source_digest
            and reviewed.to_dict() == self.citation_support.to_dict()
        )

    def to_dict(self) -> dict[str, Any]:
        """Publish aggregate metrics without sealed mappings or reviewer references."""
        return {
            "schema_version": REVIEW_SCHEMA,
            "evidence_kind": self.evidence_kind,
            "review_evidence_digest": self.review_evidence_digest,
            "request_digest": self.request_digest,
            "decisions_digest": self.decisions_digest,
            "complete": self.complete,
            "reviewer_evidence_available": self.reviewer_evidence_available,
            "human_review_required": True,
            "states": dict(self.states),
            "machine_span_checks": self.citation_support.deterministic.to_dict(),
            "citation_support": self.citation_support.to_dict(),
            "reviewer_agreement": (
                self.reviewer_agreement.to_dict() if self.reviewer_agreement else None
            ),
        }


def export_summary_review(
    *,
    packets: Sequence[BlindedAdjudicationPacket],
    mapping: SealedIdentityMapping,
    sealing_key: bytes,
    bindings: Sequence[ReviewBinding],
    claims: Sequence,
    evidence: Sequence,
    summary: str,
    source: str,
    rubric_version: int,
    evidence_kind: str = "synthetic",
) -> tuple[dict[str, Any], SealedSummaryReview]:
    """Export a value-free return request and separate authenticated bindings.

    Uses existing packet rendering and mapping verification. Each citation pair
    must be bound exactly once. Packet output and cited source excerpts must
    equal the evaluated artifacts. The evaluator chooses the evidence kind
    before export; changing it on return is rejected. Store the private result
    under local access control and retain the key separately.

    Args:
        packets: Existing blinded packets in their sealed mapping order.
        mapping: Existing identity mapping for those packets.
        sealing_key: Evaluator-held key of at least 32 bytes.
        bindings: Private packet/candidate-to-citation pairs, one per edge.
        claims: Current atomic claims and their citation references.
        evidence: Current source evidence spans, without embedded review labels.
        summary: Exact evaluated candidate output.
        source: Exact evaluated deidentified source.
        rubric_version: Positive integer version of the review rubric.
        evidence_kind: Evaluator-declared synthetic or reviewer provenance.

    Returns:
        Public request and separate access-controlled sealed bindings.

    Raises:
        SummaryReviewError: If packets, artifacts or bindings fail validation.
    """
    try:
        return _export(
            packets,
            mapping,
            sealing_key,
            bindings,
            claims,
            evidence,
            summary,
            source,
            rubric_version,
            evidence_kind,
        )
    except Exception:
        pass
    raise SummaryReviewError("invalid_review_export")


def _export(
    packets,
    mapping,
    key,
    bindings,
    claims,
    evidence,
    summary,
    source,
    rubric_version,
    evidence_kind,
):
    if type(rubric_version) is not int or rubric_version < 1:
        raise SummaryReviewError("invalid_rubric")
    if evidence_kind not in ("synthetic", "reviewer"):
        raise SummaryReviewError("invalid_evidence_kind")
    _rows(packets)
    bindings = _rows(bindings)
    _rows(claims)
    _rows(evidence)
    if not verify_sealed_identity_mapping(packets, mapping, key).valid:
        raise SummaryReviewError("invalid_mapping")
    machine = compute_citation_support_metrics(claims, evidence)
    if (
        not machine.deterministic.passed
        or machine.adjudication.available
        or machine.orphan_claim_count
    ):
        raise SummaryReviewError("invalid_machine_inputs")
    normalized_claims = [
        row if isinstance(row, AtomicClaim) else AtomicClaim.from_mapping(row)
        for row in claims
    ]
    claims_by_id = {claim.claim_id: claim for claim in normalized_claims}
    expected = {
        (claim.claim_id, ref) for claim in normalized_claims for ref in claim.citations
    }
    evidence_by_id = {
        item.evidence_id: item
        for row in evidence
        for item in [
            row if isinstance(row, EvidenceSpan) else EvidenceSpan.from_mapping(row)
        ]
    }
    packet_by_case = {packet.case_ref: packet for packet in packets}
    packet_index = {packet.case_ref: index for index, packet in enumerate(packets)}
    excerpt_index = {
        (packet.case_ref, item.evidence_ref): index
        for packet in packets
        for index, item in enumerate(packet.source_evidence)
    }
    for span, text in [
        *((claim, summary) for claim in normalized_claims),
        *((item, source) for item in evidence_by_id.values()),
    ]:
        if span.end > len(text) or (
            span.source_length is not None and span.source_length != len(text)
        ):
            raise SummaryReviewError("artifact_span_mismatch")
    assignment_by_case = {
        entry.case_ref: entry.assignments for entry in mapping.entries
    }
    private_rows, public_rows, seen = [], [], set()
    for binding in bindings:
        if type(binding) is not ReviewBinding:
            raise SummaryReviewError("invalid_binding")
        edge = (binding.claim_id, binding.evidence_id)
        if edge not in expected or edge in seen:
            raise SummaryReviewError("invalid_binding")
        seen.add(edge)
        packet = packet_by_case[binding.case_ref]
        if packet.conflict_of_interest.status != "cleared":
            raise SummaryReviewError("reviewer_recused")
        assignment = next(
            row
            for row in assignment_by_case[binding.case_ref]
            if row.candidate_identity == binding.candidate_identity
        )
        if assignment.output_digest != artifact_digest(summary):
            raise SummaryReviewError("stale_output")
        span = evidence_by_id[binding.evidence_id]
        excerpt = next(
            row
            for row in packet.source_evidence
            if row.evidence_ref == binding.evidence_id
        )
        if excerpt.content != source[span.start : span.end]:
            raise SummaryReviewError("stale_evidence")
        ref = _seal([binding.case_ref, assignment.alias, *edge], key)
        public_rows.append(
            {
                "review_ref": ref,
                "case_index": packet_index[packet.case_ref],
                "candidate_alias": assignment.alias,
                "evidence_index": excerpt_index[(packet.case_ref, binding.evidence_id)],
                "claim_start": claims_by_id[binding.claim_id].start,
                "claim_end": claims_by_id[binding.claim_id].end,
            }
        )
        private_rows.append(
            {
                "review_ref": ref,
                "claim_id": binding.claim_id,
                "evidence_id": binding.evidence_id,
            }
        )
    if not seen or seen != expected:
        raise SummaryReviewError("incomplete_binding")
    rubrics = [packet.to_dict()["rubric"] for packet in packets]
    if any(rubric != rubrics[0] for rubric in rubrics):
        raise SummaryReviewError("inconsistent_rubric")
    request = {
        "schema_version": REVIEW_SCHEMA,
        "rubric_version": rubric_version,
        "rubric_digest": _digest(rubrics[0]),
        "evidence_kind": evidence_kind,
        "mapping_commitment": mapping.mapping_commitment,
        "packet_digest": _digest([packet.to_dict() for packet in packets]),
        "summary_digest": artifact_digest(summary),
        "source_digest": artifact_digest(source),
        "claim_digest": machine.claim_digest,
        "evidence_digest": machine.evidence_digest,
        "reviews": sorted(public_rows, key=lambda row: row["review_ref"]),
    }
    request["request_digest"] = _digest(request)
    private = {
        "request": request,
        "bindings": sorted(private_rows, key=lambda row: row["review_ref"]),
    }
    private_json = _json(private)
    if len(private_json) > 32_000_000:
        raise SummaryReviewError("review_export_limit")
    return request, SealedSummaryReview(private_json, _seal(private, key))


def import_summary_review(
    decisions: Mapping[str, Any],
    *,
    sealed: SealedSummaryReview,
    sealing_key: bytes,
    claims: Sequence,
    evidence: Sequence,
    summary: str,
    source: str,
    minimum_cell_size: int = 5,
) -> ImportedSummaryReview:
    """Validate versioned local decisions and join citation and agreement metrics.

    Returns missing, incomplete, disputed and unclear states without inventing
    final labels. Two distinct reviewers are required per edge. Disagreements
    require a bounded reason; a separate adjudicator may resolve them. The
    evidence digest is a caller-supplied local review receipt, not verification
    of reviewer identity, recruitment, or clinical judgment. Unknown fields,
    duplicates, stale artifacts, wrong versions and broken seals fail closed.

    Args:
        decisions: Versioned return document with decisions and resolutions.
        sealed: Access-controlled bindings retained at export.
        sealing_key: Evaluator-held key used at export.
        claims: Current evaluated claims and citations.
        evidence: Current evaluated source evidence spans.
        summary: Current exact evaluated output.
        source: Current exact deidentified source.
        minimum_cell_size: Reviewer metric suppression threshold, at least two.

    Returns:
        Aggregate metrics and private accepted citation inputs for the gate.

    Raises:
        SummaryReviewError: If the return is malformed, stale or inconsistent.
    """
    try:
        return _import(
            decisions,
            sealed,
            sealing_key,
            claims,
            evidence,
            summary,
            source,
            minimum_cell_size,
        )
    except Exception:
        pass
    raise SummaryReviewError("invalid_review_import")


def _import(bundle, sealed, key, claims, evidence, summary, source, minimum):
    if type(sealed) is not SealedSummaryReview or len(sealed.private_json) > 32_000_000:
        raise SummaryReviewError("invalid_sealed_mapping")
    private = json.loads(sealed.private_json)
    if not hmac.compare_digest(_seal(private, key), sealed.commitment):
        raise SummaryReviewError("invalid_seal")
    _fields(private, {"request", "bindings"})
    request = private["request"]
    _fields(
        request,
        {
            "schema_version",
            "rubric_version",
            "rubric_digest",
            "evidence_kind",
            "mapping_commitment",
            "packet_digest",
            "summary_digest",
            "source_digest",
            "claim_digest",
            "evidence_digest",
            "reviews",
            "request_digest",
        },
    )
    if (
        request["schema_version"] != REVIEW_SCHEMA
        or type(request["rubric_version"]) is not int
        or request["rubric_version"] < 1
        or request["evidence_kind"] not in ("synthetic", "reviewer")
        or any(
            not _is_digest(request[field])
            for field in (
                "rubric_digest",
                "mapping_commitment",
                "packet_digest",
                "summary_digest",
                "source_digest",
                "claim_digest",
                "evidence_digest",
                "request_digest",
            )
        )
        or request["request_digest"]
        != _digest(
            {
                field: value
                for field, value in request.items()
                if field != "request_digest"
            }
        )
    ):
        raise SummaryReviewError("invalid_request")
    _rows(request["reviews"])
    _rows(private["bindings"])
    _fields(
        bundle,
        {
            "schema_version",
            "request_digest",
            "rubric_version",
            "rubric_digest",
            "evidence_kind",
            "review_evidence_digest",
            "decisions",
            "resolutions",
        },
    )
    for field in (
        "schema_version",
        "request_digest",
        "rubric_version",
        "rubric_digest",
        "evidence_kind",
    ):
        if (
            type(bundle[field]) is not type(request[field])
            or bundle[field] != request[field]
        ):
            raise SummaryReviewError("request_mismatch")
    if not _is_digest(bundle["review_evidence_digest"]):
        raise SummaryReviewError("missing_review_evidence")
    if type(minimum) is not int or minimum < 2:
        raise SummaryReviewError("invalid_minimum_cell_size")
    _rows(claims)
    _rows(evidence)
    machine = compute_citation_support_metrics(claims, evidence)
    if (
        machine.adjudication.available
        or machine.claim_digest != request["claim_digest"]
        or machine.evidence_digest != request["evidence_digest"]
        or artifact_digest(summary) != request["summary_digest"]
        or artifact_digest(source) != request["source_digest"]
    ):
        raise SummaryReviewError("stale_artifacts")
    edges = {row["review_ref"]: row for row in private["bindings"]}
    grouped, seen = defaultdict(list), set()
    for row in _rows(bundle["decisions"]):
        _fields(row, {"review_ref", "reviewer_ref", "label", "reason"})
        ref, reviewer, label = row["review_ref"], row["reviewer_ref"], row["label"]
        if (
            ref not in edges
            or not _is_digest(reviewer)
            or label not in ADJUDICATION_LABELS
        ):
            raise SummaryReviewError("invalid_decision")
        if (ref, reviewer) in seen:
            raise SummaryReviewError("duplicate_decision")
        seen.add((ref, reviewer))
        if row["reason"] is not None:
            DisagreementReason(row["reason"])
        grouped[ref].append(row)
    resolutions = {}
    for row in _rows(bundle["resolutions"]):
        _fields(row, {"review_ref", "label", "reason", "adjudicator_ref"})
        ref = row["review_ref"]
        if (
            ref not in edges
            or ref in resolutions
            or row["label"] not in ADJUDICATION_LABELS
            or not _is_digest(row["adjudicator_ref"])
        ):
            raise SummaryReviewError("invalid_resolution")
        DisagreementReason(row["reason"])
        resolutions[ref] = row
    states = Counter(
        {
            state: 0
            for state in ("accepted", "missing", "incomplete", "disputed", "unclear")
        }
    )
    accepted, agreement_rows = [], []
    for ref, edge in sorted(edges.items()):
        rows = grouped[ref]
        labels = {row["label"] for row in rows}
        resolution = resolutions.get(ref)
        if resolution and (
            len(rows) < 2
            or len(labels) < 2
            or (ref, resolution["adjudicator_ref"]) in seen
        ):
            raise SummaryReviewError("invalid_adjudicator")
        if not rows:
            states["missing"] += 1
            continue
        if len(rows) < 2:
            states["incomplete"] += 1
            continue
        reasons = {row["reason"] for row in rows}
        if len(labels) > 1:
            if len(reasons) != 1 or None in reasons:
                raise SummaryReviewError("missing_disagreement_reason")
            reason = DisagreementReason(next(iter(reasons)))
            if resolution and resolution["reason"] != reason.value:
                raise SummaryReviewError("inconsistent_disagreement_reason")
        else:
            if reasons != {None}:
                raise SummaryReviewError("unexpected_disagreement_reason")
            reason = None
        agreement_rows.extend(
            ReviewerDecision(
                ref, row["reviewer_ref"], row["label"], reason, resolution is not None
            )
            for row in rows
        )
        label = (
            resolution["label"]
            if resolution
            else next(iter(labels))
            if len(labels) == 1
            else None
        )
        state = (
            "disputed"
            if label is None
            else "unclear"
            if label == "unclear"
            else "accepted"
        )
        states[state] += 1
        if state == "accepted":
            accepted.append(
                ClinicianAdjudication(edge["claim_id"], edge["evidence_id"], label)
            )
    support = compute_citation_support_metrics(claims, evidence, adjudications=accepted)
    agreement = (
        reviewer_disagreement_report(agreement_rows, minimum_cell_size=minimum)
        if agreement_rows
        else None
    )
    return ImportedSummaryReview(
        support,
        agreement,
        tuple(accepted),
        tuple(sorted(states.items())),
        request["evidence_kind"],
        bundle["review_evidence_digest"],
        request["request_digest"],
        request["summary_digest"],
        request["source_digest"],
        request["claim_digest"],
        request["evidence_digest"],
        _digest(
            {
                **bundle,
                "decisions": sorted(
                    bundle["decisions"],
                    key=lambda row: (row["review_ref"], row["reviewer_ref"]),
                ),
                "resolutions": sorted(
                    bundle["resolutions"], key=lambda row: row["review_ref"]
                ),
            }
        ),
    )
