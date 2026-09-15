"""Claim-level citation support metrics for local clinical evaluations.

This module evaluates the relationship between already-atomic claims, cited
evidence spans, and optional clinician adjudications.  It intentionally does
not decide whether a claim is clinically true or retrieve evidence.  A caller
provides those inputs from a local, approved evaluation process.

The deterministic part of the report checks identifier references, source
offsets, source consistency, and unused/orphan relationships.  The optional
adjudication part measures whether reviewed evidence supports a claim.  The
two sections remain separate so a valid span is not mistaken for clinical
support and an adjudication is not mistaken for a span-integrity check.

Reports contain counts, fixed labels, offsets, and one-way provenance
digests.  They never contain claim values, source text, identifiers, or
adjudication comments.  The implementation uses only the standard library,
performs no model loading, and makes no network calls.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

CITATION_SUPPORT_METRICS = "citation_support_metrics"
CITATION_SUPPORT_METRICS_SCHEMA_VERSION = 1
SCHEMA_VERSION = CITATION_SUPPORT_METRICS_SCHEMA_VERSION

ADJUDICATION_SUPPORTS = "supports"
ADJUDICATION_CONTRADICTS = "contradicts"
ADJUDICATION_IRRELEVANT = "irrelevant"
ADJUDICATION_UNCLEAR = "unclear"
ADJUDICATION_LABELS: tuple[str, ...] = (
    ADJUDICATION_SUPPORTS,
    ADJUDICATION_CONTRADICTS,
    ADJUDICATION_IRRELEVANT,
    ADJUDICATION_UNCLEAR,
)

CITATION_SUPPORT_DISCLAIMER = (
    "Claim-level citation metrics are evaluation evidence only; they are not "
    "a clinical decision, compliance certification, or guarantee of clinical "
    "truth. Qualified human review remains required before clinical use."
)
HUMAN_REVIEW_REQUIRED = True

_IDENTIFIER_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,256}$")
_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_MAX_CLAIMS = 100_000
_MAX_EVIDENCE = 200_000
_MAX_CITATIONS = 500_000
_MAX_ADJUDICATIONS = 500_000


class CitationSupportError(ValueError):
    """Raised when a citation-support input cannot be normalized safely."""


@dataclass(frozen=True, slots=True, repr=False)
class AtomicClaim:
    """One atomic claim and its value-free citation references.

    ``start`` and ``end`` are half-open offsets in the generated summary or
    claim-bearing source.  They are optional so callers with structured claim
    records can still compute relationship metrics; missing offsets are
    reported separately by :class:`DeterministicSpanChecks`.
    """

    claim_id: str
    start: int | None = None
    end: int | None = None
    citations: tuple[str, ...] = ()
    source_id: str = ""
    source_length: int | None = None
    claim_type: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _identifier(self.claim_id))
        object.__setattr__(self, "start", _optional_offset(self.start))
        object.__setattr__(self, "end", _optional_offset(self.end))
        object.__setattr__(
            self,
            "citations",
            _identifier_sequence(self.citations, allow_none=True),
        )
        object.__setattr__(self, "source_id", _optional_identifier(self.source_id))
        object.__setattr__(
            self,
            "source_length",
            _optional_length(self.source_length),
        )
        object.__setattr__(self, "claim_type", _optional_label(self.claim_type))

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | Any,
        *,
        default_id: str | None = None,
    ) -> "AtomicClaim":
        """Normalize one mapping or record-like object.

        Common aliases such as ``id``, ``span``, ``citation_ids``, and
        ``document_id`` are accepted to keep fixture adapters small.  Fields
        such as ``text`` are deliberately ignored rather than retained.
        """

        data = _record_mapping(value, "claim")
        claim_id = _first(data, "claim_id", "id", "fact_id", "key")
        if claim_id is None:
            claim_id = default_id
        if claim_id is None:
            raise CitationSupportError("claim identifier is required")

        start, end = _span_values(data)
        source_id = _first(data, "source_id", "document_id", "summary_id")
        source_length = _source_length(data)
        if source_length is None:
            text = _first(data, "source_text", "document_text", "text")
            if isinstance(text, str):
                source_length = len(text)

        citations = _first(
            data,
            "citations",
            "citation_ids",
            "evidence_ids",
            "evidence",
            "sources",
        )
        claim_type = _first(
            data,
            "claim_type",
            "claim_class",
            "category",
            "type",
        )
        return cls(
            claim_id=claim_id,
            start=start,
            end=end,
            citations=_citation_id_values(citations),
            source_id=source_id or "",
            source_length=source_length,
            claim_type=claim_type or "",
        )

    def __repr__(self) -> str:
        """Return a representation that cannot print claim values or IDs."""

        return (
            "AtomicClaim(<redacted>, "
            f"span_present={self.start is not None or self.end is not None}, "
            f"citation_count={len(self.citations)})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class EvidenceSpan:
    """One caller-approved evidence span identified by an opaque reference."""

    evidence_id: str
    start: int | None = None
    end: int | None = None
    source_id: str = ""
    source_length: int | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | Any,
        *,
        default_id: str | None = None,
    ) -> "EvidenceSpan":
        """Normalize one evidence mapping without retaining source text."""

        data = _record_mapping(value, "evidence")
        evidence_id = _first(
            data,
            "evidence_id",
            "id",
            "citation_id",
            "span_id",
        )
        if evidence_id is None:
            evidence_id = default_id
        if evidence_id is None:
            raise CitationSupportError("evidence identifier is required")

        start, end = _span_values(data)
        source_id = _first(data, "source_id", "document_id", "record_id")
        source_length = _source_length(data)
        if source_length is None:
            text = _first(data, "source_text", "document_text", "text")
            if isinstance(text, str):
                source_length = len(text)
        return cls(
            evidence_id=evidence_id,
            start=start,
            end=end,
            source_id=source_id or "",
            source_length=source_length,
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _identifier(self.evidence_id))
        object.__setattr__(self, "start", _optional_offset(self.start))
        object.__setattr__(self, "end", _optional_offset(self.end))
        object.__setattr__(self, "source_id", _optional_identifier(self.source_id))
        object.__setattr__(
            self,
            "source_length",
            _optional_length(self.source_length),
        )

    def __repr__(self) -> str:
        """Return a representation that cannot print evidence values or IDs."""

        return (
            "EvidenceSpan(<redacted>, "
            f"span_present={self.start is not None or self.end is not None})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class Citation:
    """A claim-to-evidence edge, optionally carrying a cited sub-span."""

    claim_id: str
    evidence_id: str
    start: int | None = None
    end: int | None = None
    source_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _identifier(self.claim_id))
        object.__setattr__(self, "evidence_id", _identifier(self.evidence_id))
        object.__setattr__(self, "start", _optional_offset(self.start))
        object.__setattr__(self, "end", _optional_offset(self.end))
        object.__setattr__(self, "source_id", _optional_identifier(self.source_id))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | Any) -> "Citation":
        """Normalize a citation mapping with fixed, value-free failures."""

        data = _record_mapping(value, "citation")
        claim_id = _first(data, "claim_id", "claim", "fact_id")
        evidence_id = _first(
            data,
            "evidence_id",
            "evidence",
            "citation_id",
            "source_id",
        )
        if claim_id is None or evidence_id is None:
            raise CitationSupportError(
                "citation requires claim and evidence identifiers"
            )
        start, end = _span_values(data)
        source_id = _first(
            data,
            "span_source_id",
            "citation_source_id",
            "document_id",
        )
        if source_id is None and "evidence_id" in data:
            source_id = _first(data, "source_id")
        return cls(
            claim_id=claim_id,
            evidence_id=evidence_id,
            start=start,
            end=end,
            source_id=source_id or "",
        )

    def __repr__(self) -> str:
        """Return a representation that cannot print citation identifiers."""

        return "Citation(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class ClinicianAdjudication:
    """Optional human-review label for one claim/evidence edge."""

    claim_id: str
    evidence_id: str
    label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _identifier(self.claim_id))
        object.__setattr__(self, "evidence_id", _identifier(self.evidence_id))
        object.__setattr__(self, "label", _adjudication_label(self.label))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | Any) -> "ClinicianAdjudication":
        """Normalize one adjudication row."""

        data = _record_mapping(value, "adjudication")
        claim_id = _first(data, "claim_id", "claim", "fact_id")
        evidence_id = _first(
            data,
            "evidence_id",
            "evidence",
            "citation_id",
            "source_id",
        )
        label = _first(
            data,
            "label",
            "adjudication",
            "judgment",
            "clinician_label",
            "support_label",
            "verdict",
        )
        if label is None and "supports" in data and type(data["supports"]) is bool:
            label = (
                ADJUDICATION_SUPPORTS if data["supports"] else ADJUDICATION_IRRELEVANT
            )
        if label is None and "supported" in data and type(data["supported"]) is bool:
            label = (
                ADJUDICATION_SUPPORTS if data["supported"] else ADJUDICATION_IRRELEVANT
            )
        if (
            label is None
            and "contradicts" in data
            and type(data["contradicts"]) is bool
        ):
            label = (
                ADJUDICATION_CONTRADICTS
                if data["contradicts"]
                else ADJUDICATION_IRRELEVANT
            )
        if claim_id is None or evidence_id is None or label is None:
            raise CitationSupportError(
                "adjudication requires claim, evidence, and label"
            )
        return cls(claim_id=claim_id, evidence_id=evidence_id, label=label)

    def __repr__(self) -> str:
        """Return a representation that cannot print reviewed identifiers."""

        return f"ClinicianAdjudication(label={self.label!r})"


@dataclass(frozen=True, slots=True)
class DeterministicSpanChecks:
    """Aggregate, value-free results from deterministic reference checks."""

    claim_span_count: int
    valid_claim_span_count: int
    missing_claim_span_count: int
    invalid_claim_span_count: int
    evidence_span_count: int
    valid_evidence_span_count: int
    missing_evidence_span_count: int
    invalid_evidence_span_count: int
    citation_count: int
    valid_citation_count: int
    invalid_citation_count: int
    duplicate_citation_count: int
    invalid_citation_reasons: Mapping[str, int]

    @property
    def citation_span_validity(self) -> float:
        """Return valid cited-edge fraction, or zero for no cited edges."""

        return _rate(self.valid_citation_count, self.citation_count)

    @property
    def claim_span_validity(self) -> float:
        """Return valid claim-span fraction, or zero when no spans are given."""

        return _rate(self.valid_claim_span_count, self.claim_span_count)

    @property
    def evidence_span_validity(self) -> float:
        """Return valid evidence-span fraction, or zero when no spans are given."""

        return _rate(self.valid_evidence_span_count, self.evidence_span_count)

    @property
    def passed(self) -> bool:
        """Return whether all supplied span/reference checks are complete."""

        return (
            self.missing_claim_span_count == 0
            and self.invalid_claim_span_count == 0
            and self.missing_evidence_span_count == 0
            and self.invalid_evidence_span_count == 0
            and self.invalid_citation_count == 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic aggregate mapping."""

        return {
            "claim_span_count": self.claim_span_count,
            "valid_claim_span_count": self.valid_claim_span_count,
            "missing_claim_span_count": self.missing_claim_span_count,
            "invalid_claim_span_count": self.invalid_claim_span_count,
            "claim_span_validity": self.claim_span_validity,
            "evidence_span_count": self.evidence_span_count,
            "valid_evidence_span_count": self.valid_evidence_span_count,
            "missing_evidence_span_count": self.missing_evidence_span_count,
            "invalid_evidence_span_count": self.invalid_evidence_span_count,
            "evidence_span_validity": self.evidence_span_validity,
            "citation_count": self.citation_count,
            "valid_citation_count": self.valid_citation_count,
            "invalid_citation_count": self.invalid_citation_count,
            "citation_span_validity": self.citation_span_validity,
            "duplicate_citation_count": self.duplicate_citation_count,
            "invalid_citation_reasons": {
                reason: self.invalid_citation_reasons[reason]
                for reason in sorted(self.invalid_citation_reasons)
            },
            "passed": self.passed,
        }

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access used by evaluation notebooks."""

        return self.to_dict()[key]


@dataclass(frozen=True, slots=True)
class ClinicianAdjudicationMetrics:
    """Optional aggregate metrics from claim/evidence human review."""

    available: bool
    adjudicated_citation_count: int
    unadjudicated_citation_count: int
    adjudicated_claim_count: int
    unadjudicated_claim_count: int
    supporting_citation_count: int
    contradicting_citation_count: int
    irrelevant_citation_count: int
    unclear_citation_count: int
    supported_claim_count: int
    citation_precision: float | None
    support_recall: float | None
    adjudication_coverage: float
    unmatched_adjudication_count: int
    conflicting_adjudication_count: int
    label_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, value-free adjudication mapping."""

        return {
            "available": self.available,
            "status": "available" if self.available else "not_available",
            "adjudicated_citation_count": self.adjudicated_citation_count,
            "unadjudicated_citation_count": self.unadjudicated_citation_count,
            "adjudicated_claim_count": self.adjudicated_claim_count,
            "unadjudicated_claim_count": self.unadjudicated_claim_count,
            "supporting_citation_count": self.supporting_citation_count,
            "contradicting_citation_count": self.contradicting_citation_count,
            "irrelevant_citation_count": self.irrelevant_citation_count,
            "unclear_citation_count": self.unclear_citation_count,
            "supported_claim_count": self.supported_claim_count,
            "citation_precision": self.citation_precision,
            "support_recall": self.support_recall,
            "adjudication_coverage": self.adjudication_coverage,
            "unmatched_adjudication_count": self.unmatched_adjudication_count,
            "conflicting_adjudication_count": self.conflicting_adjudication_count,
            "label_counts": {
                label: self.label_counts[label] for label in ADJUDICATION_LABELS
            },
        }

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to adjudication metrics."""

        return self.to_dict()[key]


@dataclass(frozen=True, slots=True, repr=False)
class CitationSupportReport:
    """Complete claim-level citation-support report."""

    claim_count: int
    evidence_count: int
    citation_count: int
    orphan_claim_count: int
    unused_evidence_count: int
    deterministic: DeterministicSpanChecks
    adjudication: ClinicianAdjudicationMetrics
    claim_digest: str
    evidence_digest: str
    adjudication_digest: str
    input_digest: str
    schema_version: int = CITATION_SUPPORT_METRICS_SCHEMA_VERSION

    @property
    def orphan_claim_rate(self) -> float:
        """Return orphan claims divided by all atomic claims."""

        return _rate(self.orphan_claim_count, self.claim_count)

    @property
    def unused_evidence_rate(self) -> float:
        """Return unused evidence divided by all supplied evidence."""

        return _rate(self.unused_evidence_count, self.evidence_count)

    @property
    def citation_precision(self) -> float | None:
        """Return optional clinician-adjudicated citation precision."""

        return self.adjudication.citation_precision

    @property
    def support_recall(self) -> float | None:
        """Return optional atomic-claim support recall."""

        return self.adjudication.support_recall

    @property
    def orphan_claims(self) -> int:
        """Compatibility alias for the orphan-claim count."""

        return self.orphan_claim_count

    @property
    def unused_evidence(self) -> int:
        """Compatibility alias for the unused-evidence count."""

        return self.unused_evidence_count

    @property
    def span_checks(self) -> DeterministicSpanChecks:
        """Return the deterministic span section under its short name."""

        return self.deterministic

    @property
    def clinician_adjudication(self) -> ClinicianAdjudicationMetrics:
        """Return the optional clinician-review section."""

        return self.adjudication

    def to_dict(self) -> dict[str, Any]:
        """Return a stable aggregate-only report mapping."""

        return {
            "suite": CITATION_SUPPORT_METRICS,
            "schema_version": self.schema_version,
            "claim_count": self.claim_count,
            "evidence_count": self.evidence_count,
            "citation_count": self.citation_count,
            "orphan_claim_count": self.orphan_claim_count,
            "orphan_claim_rate": self.orphan_claim_rate,
            "unused_evidence_count": self.unused_evidence_count,
            "unused_evidence_rate": self.unused_evidence_rate,
            "citation_precision": self.citation_precision,
            "support_recall": self.support_recall,
            "deterministic_span_checks": self.deterministic.to_dict(),
            "clinician_adjudication": self.adjudication.to_dict(),
            "provenance": {
                "claim_digest": self.claim_digest,
                "evidence_digest": self.evidence_digest,
                "adjudication_digest": self.adjudication_digest,
                "input_digest": self.input_digest,
            },
            "human_review_required": HUMAN_REVIEW_REQUIRED,
            "disclaimer": CITATION_SUPPORT_DISCLAIMER,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON without identifiers or source values."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )

    def to_markdown(self) -> str:
        """Return a deterministic human-review summary."""

        def score(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.4f}"

        lines = [
            "# Claim-level citation support metrics",
            "",
            f"> {CITATION_SUPPORT_DISCLAIMER}",
            "",
            "## Aggregate metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Atomic claims | {self.claim_count} |",
            f"| Evidence spans | {self.evidence_count} |",
            f"| Citation edges | {self.citation_count} |",
            f"| Citation precision | {score(self.citation_precision)} |",
            f"| Support recall | {score(self.support_recall)} |",
            f"| Orphan claims | {self.orphan_claim_count} "
            f"({self.orphan_claim_rate:.4f}) |",
            f"| Unused evidence | {self.unused_evidence_count} "
            f"({self.unused_evidence_rate:.4f}) |",
            "",
            "## Deterministic span checks",
            "",
            f"- Verdict: `{'pass' if self.deterministic.passed else 'review'}`",
            f"- Valid claim spans: `{self.deterministic.valid_claim_span_count}/"
            f"{self.deterministic.claim_span_count}`",
            f"- Valid evidence spans: `"
            f"{self.deterministic.valid_evidence_span_count}/"
            f"{self.deterministic.evidence_span_count}`",
            f"- Valid cited edges: `"
            f"{self.deterministic.valid_citation_count}/"
            f"{self.deterministic.citation_count}`",
            "",
            "## Optional clinician adjudication",
            "",
            f"- Status: `{'available' if self.adjudication.available else 'not_available'}`",
            f"- Reviewed citations: `{self.adjudication.adjudicated_citation_count}`",
            f"- Reviewed claims: `{self.adjudication.adjudicated_claim_count}`",
            f"- Adjudication coverage: `{self.adjudication.adjudication_coverage:.4f}`",
            "",
            "## Provenance",
            "",
            "The report exposes only one-way SHA-256 digests for the input "
            "collections. It does not render claim IDs, evidence IDs, source "
            "text, or reviewer comments.",
            "",
        ]
        return "\n".join(lines)

    def write_json(self, path: str | Path) -> Path:
        """Write the stable JSON artifact to a caller-selected local path."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json(), encoding="utf-8")
        return output

    def write_markdown(self, path: str | Path) -> Path:
        """Write the stable Markdown artifact to a caller-selected path."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_markdown(), encoding="utf-8")
        return output

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to the aggregate report."""

        return self.to_dict()[key]

    def __repr__(self) -> str:
        """Return an aggregate-only representation."""

        return (
            "CitationSupportReport("
            f"claim_count={self.claim_count}, evidence_count={self.evidence_count}, "
            f"citation_count={self.citation_count})"
        )


# Short aliases make the small data contract convenient for callers while the
# descriptive names remain the canonical documentation surface.
Claim = AtomicClaim
Evidence = EvidenceSpan
Adjudication = ClinicianAdjudication
CitationSupportMetrics = CitationSupportReport
ClinicianMetrics = ClinicianAdjudicationMetrics
SpanChecks = DeterministicSpanChecks


def compute_citation_support_metrics(
    claims: Iterable[Mapping[str, Any] | AtomicClaim | Any],
    evidence: Iterable[Mapping[str, Any] | EvidenceSpan | Any] = (),
    citations: Iterable[Mapping[str, Any] | Citation | Any]
    | Mapping[str, Any]
    | None = None,
    adjudications: Iterable[Mapping[str, Any] | ClinicianAdjudication | Any]
    | Mapping[str, Any]
    | None = None,
    *,
    source_lengths: Mapping[str, int] | None = None,
) -> CitationSupportReport:
    """Compute deterministic and optional claim-level citation metrics.

    Args:
        claims: Atomic claim records.  Each record needs an identifier and may
            carry a half-open ``start``/``end`` span plus ``citations`` or
            ``evidence_ids``.  A mapping of claim ID to a two-item offset tuple
            is also accepted for compact adapters.
        evidence: Evidence records with an identifier and half-open span.
            Evidence is never fetched or dereferenced by this function.
        citations: Optional explicit claim-to-evidence records.  Use this when
            citations are stored separately from claims.  It may also be a
            mapping from claim IDs to evidence IDs.  Claim-embedded citations
            and explicit citations are combined and exact duplicate edges are
            counted once.
        adjudications: Optional local clinician-review records with
            ``supports``, ``contradicts``, ``irrelevant``, or ``unclear``
            labels.  Without this input, citation precision and support recall
            are explicitly unavailable rather than inferred from span overlap.
        source_lengths: Optional mapping from opaque source IDs to source
            lengths.  It fills missing per-record lengths for bounded checks.

    Returns:
        A deterministic, aggregate-only :class:`CitationSupportReport`.

    Raises:
        CitationSupportError: If the collection shape or required metadata is
            unsafe or cannot be normalized.  Error messages never echo input
            values.
    """

    lengths = _normalize_source_lengths(source_lengths)
    claim_records, claim_citations = _normalize_claims(claims, lengths)
    evidence_records, embedded_adjudications = _normalize_evidence(evidence, lengths)

    claims_by_id: dict[str, AtomicClaim] = {}
    for claim in claim_records:
        if claim.claim_id in claims_by_id:
            raise CitationSupportError("duplicate claim identifiers")
        claims_by_id[claim.claim_id] = claim

    evidence_by_id: dict[str, EvidenceSpan] = {}
    for item in evidence_records:
        if item.evidence_id in evidence_by_id:
            raise CitationSupportError("duplicate evidence identifiers")
        evidence_by_id[item.evidence_id] = item

    explicit_citations = (
        _normalize_citations(citations) if citations is not None else []
    )
    all_citations = [*claim_citations, *explicit_citations]
    unique_citations, duplicate_citation_count = _deduplicate_citations(all_citations)

    used_evidence: set[str] = set()
    orphan_claim_count = 0
    citations_by_claim: dict[str, list[Citation]] = {
        claim_id: [] for claim_id in claims_by_id
    }
    for citation in unique_citations:
        if citation.claim_id not in claims_by_id:
            continue
        citations_by_claim[citation.claim_id].append(citation)
        if citation.evidence_id in evidence_by_id:
            used_evidence.add(citation.evidence_id)
    for claim_id in claims_by_id:
        if not citations_by_claim[claim_id]:
            orphan_claim_count += 1

    valid_edges: list[tuple[Citation, EvidenceSpan]] = []
    invalid_reasons: dict[str, int] = {}
    for citation in unique_citations:
        reason = _citation_invalid_reason(
            citation,
            claims_by_id=claims_by_id,
            evidence_by_id=evidence_by_id,
        )
        if reason is None:
            valid_edges.append((citation, evidence_by_id[citation.evidence_id]))
        else:
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1

    span_checks = _build_span_checks(
        claim_records,
        evidence_records,
        citation_count=len(unique_citations),
        valid_citation_count=len(valid_edges),
        duplicate_citation_count=duplicate_citation_count,
        invalid_reasons=invalid_reasons,
    )

    explicit_adjudications = (
        _normalize_adjudications(adjudications) if adjudications is not None else []
    )
    all_adjudications = [*embedded_adjudications, *explicit_adjudications]
    adjudication_metrics = _build_adjudication_metrics(
        all_adjudications,
        valid_edges=valid_edges,
        claim_ids=tuple(claims_by_id),
        valid_citation_count=len(valid_edges),
    )

    claim_digest = _digest_claims(claim_records, unique_citations)
    evidence_digest = _digest_evidence(evidence_records)
    adjudication_digest = _digest_adjudications(all_adjudications)
    input_digest = _digest_payload(
        {
            "claims": claim_digest,
            "evidence": evidence_digest,
            "adjudications": adjudication_digest,
        }
    )

    return CitationSupportReport(
        claim_count=len(claim_records),
        evidence_count=len(evidence_records),
        citation_count=len(unique_citations),
        orphan_claim_count=orphan_claim_count,
        unused_evidence_count=len(evidence_by_id) - len(used_evidence),
        deterministic=span_checks,
        adjudication=adjudication_metrics,
        claim_digest=claim_digest,
        evidence_digest=evidence_digest,
        adjudication_digest=adjudication_digest,
        input_digest=input_digest,
    )


def citation_support_metrics(*args: Any, **kwargs: Any) -> CitationSupportReport:
    """Alias for :func:`compute_citation_support_metrics`."""

    return compute_citation_support_metrics(*args, **kwargs)


def evaluate_citation_support(*args: Any, **kwargs: Any) -> CitationSupportReport:
    """Alias emphasizing that this is an evaluation-only operation."""

    return compute_citation_support_metrics(*args, **kwargs)


def score_citation_support(*args: Any, **kwargs: Any) -> CitationSupportReport:
    """Alias for callers that use ``score_*`` evaluation naming."""

    return compute_citation_support_metrics(*args, **kwargs)


def build_citation_support_report(*args: Any, **kwargs: Any) -> CitationSupportReport:
    """Build a serializable report from local claim/evidence records."""

    return compute_citation_support_metrics(*args, **kwargs)


def run_citation_support_metrics(*args: Any, **kwargs: Any) -> CitationSupportReport:
    """Run the offline citation-support evaluator."""

    return compute_citation_support_metrics(*args, **kwargs)


def _normalize_claims(
    values: Iterable[Mapping[str, Any] | AtomicClaim | Any],
    source_lengths: Mapping[str, int],
) -> tuple[list[AtomicClaim], list[Citation]]:
    rows = _claim_rows(values)
    normalized: list[AtomicClaim] = []
    citations: list[Citation] = []
    for row, default_id in rows:
        claim = (
            row
            if isinstance(row, AtomicClaim)
            else AtomicClaim.from_mapping(row, default_id=default_id)
        )
        if claim.source_length is None and claim.source_id in source_lengths:
            claim = replace(claim, source_length=source_lengths[claim.source_id])
        normalized.append(claim)
        if isinstance(row, AtomicClaim):
            citations.extend(
                Citation(claim_id=claim.claim_id, evidence_id=evidence_id)
                for evidence_id in claim.citations
            )
        else:
            citations.extend(_claim_citations(row, claim.claim_id))
    if len(normalized) > _MAX_CLAIMS:
        raise CitationSupportError("claim collection exceeds the record limit")
    return normalized, citations


def _normalize_evidence(
    values: Iterable[Mapping[str, Any] | EvidenceSpan | Any],
    source_lengths: Mapping[str, int],
) -> tuple[list[EvidenceSpan], list[ClinicianAdjudication]]:
    rows = _evidence_rows(values)
    normalized: list[EvidenceSpan] = []
    embedded: list[ClinicianAdjudication] = []
    for row, default_id in rows:
        evidence = (
            row
            if isinstance(row, EvidenceSpan)
            else EvidenceSpan.from_mapping(row, default_id=default_id)
        )
        if evidence.source_length is None and evidence.source_id in source_lengths:
            evidence = replace(
                evidence,
                source_length=source_lengths[evidence.source_id],
            )
        normalized.append(evidence)
        if not isinstance(row, EvidenceSpan):
            embedded.extend(_embedded_adjudications(row, evidence.evidence_id))
    if len(normalized) > _MAX_EVIDENCE:
        raise CitationSupportError("evidence collection exceeds the record limit")
    return normalized, embedded


def _normalize_citations(
    values: Iterable[Mapping[str, Any] | Citation | Any] | Mapping[str, Any],
) -> list[Citation]:
    if isinstance(values, Mapping) and not _is_citation_mapping(values):
        records: list[Citation] = []
        for claim_id, evidence_values in values.items():
            for item in _citation_values(evidence_values):
                if isinstance(item, Citation):
                    records.append(replace(item, claim_id=_identifier(claim_id)))
                elif isinstance(item, Mapping):
                    data = dict(item)
                    data.setdefault("claim_id", claim_id)
                    records.append(Citation.from_mapping(data))
                else:
                    records.append(
                        Citation(
                            claim_id=claim_id,
                            evidence_id=item,
                        )
                    )
        return records

    records = _bounded_sequence(values, "citation collection", _MAX_CITATIONS)
    normalized: list[Citation] = []
    for item in records:
        if isinstance(item, Citation):
            normalized.append(item)
        elif isinstance(item, Mapping):
            normalized.append(Citation.from_mapping(item))
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            if len(item) != 2:
                raise CitationSupportError(
                    "citation tuple must contain two identifiers"
                )
            normalized.append(Citation(claim_id=item[0], evidence_id=item[1]))
        else:
            raise CitationSupportError("citation records must be mappings or pairs")
    return normalized


def _normalize_adjudications(
    values: Iterable[Mapping[str, Any] | ClinicianAdjudication | Any]
    | Mapping[str, Any],
) -> list[ClinicianAdjudication]:
    rows = _adjudication_rows(values)
    normalized: list[ClinicianAdjudication] = []
    for row, default_claim_id in rows:
        if isinstance(row, ClinicianAdjudication):
            normalized.append(row)
            continue
        data = _record_mapping(row, "adjudication")
        if (
            default_claim_id is not None
            and _first(data, "claim_id", "claim", "fact_id") is None
        ):
            data = {**data, "claim_id": default_claim_id}
        normalized.append(ClinicianAdjudication.from_mapping(data))
    if len(normalized) > _MAX_ADJUDICATIONS:
        raise CitationSupportError("adjudication collection exceeds the record limit")
    return normalized


def _build_span_checks(
    claims: Sequence[AtomicClaim],
    evidence: Sequence[EvidenceSpan],
    *,
    citation_count: int,
    valid_citation_count: int,
    duplicate_citation_count: int,
    invalid_reasons: Mapping[str, int],
) -> DeterministicSpanChecks:
    claim_statuses = [
        _span_status(item.start, item.end, item.source_length) for item in claims
    ]
    evidence_statuses = [
        _span_status(item.start, item.end, item.source_length) for item in evidence
    ]
    return DeterministicSpanChecks(
        claim_span_count=sum(status != "missing" for status in claim_statuses),
        valid_claim_span_count=sum(status == "valid" for status in claim_statuses),
        missing_claim_span_count=sum(status == "missing" for status in claim_statuses),
        invalid_claim_span_count=sum(
            status not in {"missing", "valid"} for status in claim_statuses
        ),
        evidence_span_count=sum(status != "missing" for status in evidence_statuses),
        valid_evidence_span_count=sum(
            status == "valid" for status in evidence_statuses
        ),
        missing_evidence_span_count=sum(
            status == "missing" for status in evidence_statuses
        ),
        invalid_evidence_span_count=sum(
            status not in {"missing", "valid"} for status in evidence_statuses
        ),
        citation_count=citation_count,
        valid_citation_count=valid_citation_count,
        invalid_citation_count=sum(invalid_reasons.values()),
        duplicate_citation_count=duplicate_citation_count,
        invalid_citation_reasons=dict(sorted(invalid_reasons.items())),
    )


def _build_adjudication_metrics(
    adjudications: Sequence[ClinicianAdjudication],
    *,
    valid_edges: Sequence[tuple[Citation, EvidenceSpan]],
    claim_ids: Sequence[str],
    valid_citation_count: int,
) -> ClinicianAdjudicationMetrics:
    valid_edge_keys = {
        (citation.claim_id, citation.evidence_id) for citation, _ in valid_edges
    }
    labels_by_edge: dict[tuple[str, str], set[str]] = {}
    unmatched = 0
    for item in adjudications:
        key = (item.claim_id, item.evidence_id)
        if key not in valid_edge_keys:
            unmatched += 1
            continue
        labels_by_edge.setdefault(key, set()).add(item.label)

    conflicting = sum(len(labels) > 1 for labels in labels_by_edge.values())
    effective_labels = {
        key: next(iter(labels)) if len(labels) == 1 else ADJUDICATION_UNCLEAR
        for key, labels in labels_by_edge.items()
    }
    label_counts = {label: 0 for label in ADJUDICATION_LABELS}
    for label in effective_labels.values():
        label_counts[label] += 1

    reviewed_count = len(effective_labels)
    supported_claim_ids = {
        claim_id
        for (claim_id, _), label in effective_labels.items()
        if label == ADJUDICATION_SUPPORTS
    }
    adjudicated_claim_ids = {claim_id for claim_id, _ in effective_labels}
    available = bool(adjudications)
    precision = (
        _rate(label_counts[ADJUDICATION_SUPPORTS], reviewed_count)
        if reviewed_count
        else None
    )
    recall = _rate(len(supported_claim_ids), len(claim_ids)) if reviewed_count else None
    return ClinicianAdjudicationMetrics(
        available=available,
        adjudicated_citation_count=reviewed_count,
        unadjudicated_citation_count=max(valid_citation_count - reviewed_count, 0),
        adjudicated_claim_count=len(adjudicated_claim_ids),
        unadjudicated_claim_count=max(len(claim_ids) - len(adjudicated_claim_ids), 0),
        supporting_citation_count=label_counts[ADJUDICATION_SUPPORTS],
        contradicting_citation_count=label_counts[ADJUDICATION_CONTRADICTS],
        irrelevant_citation_count=label_counts[ADJUDICATION_IRRELEVANT],
        unclear_citation_count=label_counts[ADJUDICATION_UNCLEAR],
        supported_claim_count=len(supported_claim_ids),
        citation_precision=precision,
        support_recall=recall,
        adjudication_coverage=_rate(reviewed_count, valid_citation_count),
        unmatched_adjudication_count=unmatched,
        conflicting_adjudication_count=conflicting,
        label_counts=label_counts,
    )


def _citation_invalid_reason(
    citation: Citation,
    *,
    claims_by_id: Mapping[str, AtomicClaim],
    evidence_by_id: Mapping[str, EvidenceSpan],
) -> str | None:
    claim = claims_by_id.get(citation.claim_id)
    if claim is None:
        return "missing_claim"
    evidence = evidence_by_id.get(citation.evidence_id)
    if evidence is None:
        return "missing_evidence"
    if _span_status(evidence.start, evidence.end, evidence.source_length) != "valid":
        return "invalid_evidence_span"
    if (
        citation.source_id
        and evidence.source_id
        and citation.source_id != evidence.source_id
    ):
        return "source_mismatch"
    if citation.start is None and citation.end is None:
        return None
    citation_status = _span_status(
        citation.start,
        citation.end,
        evidence.source_length,
    )
    if citation_status != "valid":
        return "invalid_citation_span"
    assert citation.start is not None and citation.end is not None
    assert evidence.start is not None and evidence.end is not None
    if citation.start < evidence.start or citation.end > evidence.end:
        return "citation_outside_evidence"
    return None


def _deduplicate_citations(citations: Sequence[Citation]) -> tuple[list[Citation], int]:
    seen: set[tuple[str, str, int | None, int | None, str]] = set()
    unique: list[Citation] = []
    duplicates = 0
    for citation in citations:
        key = (
            citation.claim_id,
            citation.evidence_id,
            citation.start,
            citation.end,
            citation.source_id,
        )
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(citation)
    return sorted(unique, key=_citation_sort_key), duplicates


def _citation_sort_key(citation: Citation) -> tuple[Any, ...]:
    return (
        citation.claim_id,
        citation.evidence_id,
        citation.start if citation.start is not None else -1,
        citation.end if citation.end is not None else -1,
        citation.source_id,
    )


def _claim_rows(value: Any) -> list[tuple[Any, str | None]]:
    if isinstance(value, Mapping):
        if _is_claim_mapping(value):
            return [(value, None)]
        rows: list[tuple[Any, str | None]] = []
        for key, item in value.items():
            default_id = _identifier(key)
            if isinstance(item, Mapping) or isinstance(item, AtomicClaim):
                rows.append((item, default_id))
            elif _offset_pair(item):
                rows.append(
                    ({"claim_id": default_id, "start": item[0], "end": item[1]}, None)
                )
            else:
                rows.append(({"claim_id": default_id, "citations": item}, None))
        return rows
    return [
        (item, None)
        for item in _bounded_sequence(value, "claim collection", _MAX_CLAIMS)
    ]


def _evidence_rows(value: Any) -> list[tuple[Any, str | None]]:
    if isinstance(value, Mapping):
        if _is_evidence_mapping(value):
            return [(value, None)]
        rows: list[tuple[Any, str | None]] = []
        for key, item in value.items():
            default_id = _identifier(key)
            if isinstance(item, Mapping) or isinstance(item, EvidenceSpan):
                rows.append((item, default_id))
            elif _offset_pair(item):
                rows.append(
                    (
                        {"evidence_id": default_id, "start": item[0], "end": item[1]},
                        None,
                    )
                )
            else:
                raise CitationSupportError(
                    "evidence mapping values must be records or spans"
                )
        return rows
    return [
        (item, None)
        for item in _bounded_sequence(value, "evidence collection", _MAX_EVIDENCE)
    ]


def _adjudication_rows(value: Any) -> list[tuple[Any, str | None]]:
    if isinstance(value, Mapping) and not _is_adjudication_mapping(value):
        rows: list[tuple[Any, str | None]] = []
        for key, item in value.items():
            rows.append((item, _identifier(key)))
        return rows
    return [
        (item, None)
        for item in _bounded_sequence(
            value,
            "adjudication collection",
            _MAX_ADJUDICATIONS,
        )
    ]


def _citation_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    raise CitationSupportError("citation values must be identifiers or records")


def _claim_citations(value: Any, claim_id: str) -> list[Citation]:
    data = _record_mapping(value, "claim")
    citation_values = _first(
        data,
        "citations",
        "citation_ids",
        "evidence_ids",
        "evidence",
        "sources",
    )
    normalized: list[Citation] = []
    for item in _citation_values(citation_values):
        if isinstance(item, Citation):
            normalized.append(replace(item, claim_id=claim_id))
        elif isinstance(item, Mapping):
            citation = dict(item)
            citation.setdefault("claim_id", claim_id)
            normalized.append(Citation.from_mapping(citation))
        else:
            normalized.append(Citation(claim_id=claim_id, evidence_id=item))
    return normalized


def _bounded_sequence(value: Any, field_name: str, limit: int) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        raise CitationSupportError(f"{field_name} must be an iterable of records")
    try:
        rows = list(value)
    except (TypeError, ValueError):
        raise CitationSupportError(
            f"{field_name} must be an iterable of records"
        ) from None
    if len(rows) > limit:
        raise CitationSupportError(f"{field_name} exceeds the record limit")
    return rows


def _record_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            converted = to_dict()
        except Exception:
            raise CitationSupportError(
                f"{field_name} record cannot be normalized"
            ) from None
        if isinstance(converted, Mapping):
            return converted
    try:
        converted = vars(value)
    except (TypeError, ValueError):
        raise CitationSupportError(f"{field_name} record must be a mapping") from None
    if not isinstance(converted, Mapping):
        raise CitationSupportError(f"{field_name} record must be a mapping")
    return converted


def _span_values(data: Mapping[str, Any]) -> tuple[int | None, int | None]:
    nested = _first(data, "span", "source_span", "claim_span", "evidence_span")
    nested_start: Any = None
    nested_end: Any = None
    if isinstance(nested, Mapping):
        nested_start = _first(nested, "start", "begin", "offset_start")
        nested_end = _first(nested, "end", "stop", "offset_end")
    elif isinstance(nested, Sequence) and not isinstance(
        nested, (str, bytes, bytearray)
    ):
        if len(nested) != 2:
            raise CitationSupportError("span tuple must contain two offsets")
        nested_start, nested_end = nested
    start = _first(data, "start", "source_start", "span_start")
    end = _first(data, "end", "source_end", "span_end")
    if start is None:
        start = nested_start
    if end is None:
        end = nested_end
    if start is None and end is None:
        return None, None
    return _optional_offset(start), _optional_offset(end)


def _source_length(data: Mapping[str, Any]) -> int | None:
    value = _first(
        data,
        "source_length",
        "document_length",
        "text_length",
        "length",
    )
    return _optional_length(value)


def _normalize_source_lengths(value: Mapping[str, int] | None) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise CitationSupportError("source_lengths must be a mapping")
    result: dict[str, int] = {}
    for key, length in value.items():
        result[_identifier(key)] = _length(length)
    return result


def _embedded_adjudications(
    data_value: Any,
    evidence_id: str,
) -> list[ClinicianAdjudication]:
    data = _record_mapping(data_value, "evidence")
    claim_id = _first(data, "claim_id", "claim", "fact_id")
    if claim_id is None:
        return []
    label: Any = None
    for key in (
        "adjudication",
        "adjudication_label",
        "judgment",
        "clinician_label",
        "support_label",
        "relation",
    ):
        if key in data:
            label = data[key]
            break
    if label is None and "supports_claim" in data:
        label = (
            ADJUDICATION_SUPPORTS
            if data["supports_claim"] is True
            else ADJUDICATION_IRRELEVANT
        )
    if label is None and "supports" in data and type(data["supports"]) is bool:
        label = (
            ADJUDICATION_SUPPORTS
            if data["supports"] is True
            else ADJUDICATION_IRRELEVANT
        )
    if label is None and "supported" in data and type(data["supported"]) is bool:
        label = (
            ADJUDICATION_SUPPORTS
            if data["supported"] is True
            else ADJUDICATION_IRRELEVANT
        )
    if label is None and "contradicts_claim" in data:
        label = (
            ADJUDICATION_CONTRADICTS
            if data["contradicts_claim"] is True
            else ADJUDICATION_IRRELEVANT
        )
    if label is None and "contradicts" in data and type(data["contradicts"]) is bool:
        label = (
            ADJUDICATION_CONTRADICTS
            if data["contradicts"] is True
            else ADJUDICATION_IRRELEVANT
        )
    if label is None:
        return []
    return [
        ClinicianAdjudication(
            claim_id=claim_id,
            evidence_id=evidence_id,
            label=label,
        )
    ]


def _span_status(
    start: int | None,
    end: int | None,
    source_length: int | None,
) -> str:
    if start is None and end is None:
        return "missing"
    if start is None or end is None:
        return "invalid"
    if type(start) is not int or type(end) is not int or start < 0 or end <= start:
        return "invalid"
    if source_length is not None:
        if type(source_length) is not int or source_length < 0:
            return "invalid"
        if end > source_length:
            return "out_of_bounds"
    return "valid"


def _identifier(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise CitationSupportError("identifier must be a string")
    normalized = str(value).strip()
    if _IDENTIFIER_RE.fullmatch(normalized) is None:
        raise CitationSupportError("identifier is empty, bounded, and control-free")
    return normalized


def _optional_identifier(value: Any) -> str:
    if value is None or value == "":
        return ""
    return _identifier(value)


def _optional_offset(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CitationSupportError("span offsets must be integers")
    return value


def _length(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CitationSupportError("source lengths must be integers")
    return value


def _optional_length(value: Any) -> int | None:
    if value is None:
        return None
    return _length(value)


def _optional_label(value: Any) -> str:
    if value is None or value == "":
        return ""
    normalized = re.sub(r"[^a-z0-9_.-]+", "-", str(value).strip().casefold()).strip("-")
    if _LABEL_RE.fullmatch(normalized) is None:
        raise CitationSupportError("claim type must be a bounded label")
    return normalized


def _identifier_sequence(value: Any, *, allow_none: bool = False) -> tuple[str, ...]:
    if value is None and allow_none:
        return ()
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        values = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = list(value)
    else:
        raise CitationSupportError("citation identifiers must be a sequence")
    normalized = [_identifier(item) for item in values]
    return tuple(normalized)


def _citation_id_values(value: Any) -> tuple[str, ...]:
    values = _citation_values(value)
    identifiers: list[Any] = []
    for item in values:
        if isinstance(item, Mapping):
            evidence_id = _first(
                item,
                "evidence_id",
                "evidence",
                "citation_id",
                "id",
            )
            if evidence_id is None:
                raise CitationSupportError(
                    "citation record requires an evidence identifier"
                )
            identifiers.append(evidence_id)
        else:
            identifiers.append(item)
    return _identifier_sequence(identifiers, allow_none=True)


def _adjudication_label(value: Any) -> str:
    if isinstance(value, bool):
        return ADJUDICATION_SUPPORTS if value else ADJUDICATION_IRRELEVANT
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")
    if normalized in {
        "supports",
        "support",
        "supported",
        "entails",
        "entailed",
        "relevant_support",
        "true",
    }:
        return ADJUDICATION_SUPPORTS
    if normalized in {
        "contradicts",
        "contradict",
        "contradicted",
        "refutes",
        "refuted",
        "false",
    }:
        return ADJUDICATION_CONTRADICTS
    if normalized in {
        "irrelevant",
        "unrelated",
        "does_not_support",
        "not_supporting",
        "unsupported",
        "insufficient",
    }:
        return ADJUDICATION_IRRELEVANT
    if normalized in {
        "unclear",
        "uncertain",
        "ambiguous",
        "unknown",
        "unresolved",
        "pending",
    }:
        return ADJUDICATION_UNCLEAR
    raise CitationSupportError("adjudication label is unsupported")


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _first(data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _offset_pair(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def _is_claim_mapping(value: Mapping[str, Any]) -> bool:
    return bool(
        {
            "claim_id",
            "id",
            "fact_id",
            "claim_key",
            "start",
            "end",
            "span",
            "citations",
            "citation_ids",
            "evidence_ids",
            "claim_type",
        }
        & set(value)
    )


def _is_evidence_mapping(value: Mapping[str, Any]) -> bool:
    return bool(
        {
            "evidence_id",
            "id",
            "citation_id",
            "span_id",
            "start",
            "end",
            "span",
            "source_length",
            "document_length",
            "source_id",
            "document_id",
        }
        & set(value)
    )


def _is_citation_mapping(value: Mapping[str, Any]) -> bool:
    return bool(
        {"claim_id", "claim", "fact_id", "evidence_id", "evidence"} & set(value)
    )


def _is_adjudication_mapping(value: Mapping[str, Any]) -> bool:
    return bool(
        {
            "claim_id",
            "claim",
            "fact_id",
            "evidence_id",
            "evidence",
            "label",
            "adjudication",
            "judgment",
            "clinician_label",
        }
        & set(value)
    )


def _digest_payload(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _opaque(value: str) -> str:
    return _digest_payload(value)


def _digest_claims(
    claims: Sequence[AtomicClaim],
    citations: Sequence[Citation],
) -> str:
    citations_by_claim: dict[str, list[dict[str, Any]]] = {}
    for citation in citations:
        citations_by_claim.setdefault(citation.claim_id, []).append(
            {
                "evidence_id": _opaque(citation.evidence_id),
                "start": citation.start,
                "end": citation.end,
                "source_id": _opaque(citation.source_id) if citation.source_id else "",
            }
        )
    payload = []
    for claim in claims:
        payload.append(
            {
                "claim_id": _opaque(claim.claim_id),
                "start": claim.start,
                "end": claim.end,
                "source_id": _opaque(claim.source_id) if claim.source_id else "",
                "source_length": claim.source_length,
                "claim_type": _opaque(claim.claim_type) if claim.claim_type else "",
                "citations": sorted(
                    citations_by_claim.get(claim.claim_id, []),
                    key=lambda item: json.dumps(item, sort_keys=True),
                ),
            }
        )
    return _digest_payload(sorted(payload, key=lambda item: item["claim_id"]))


def _digest_evidence(evidence: Sequence[EvidenceSpan]) -> str:
    payload = [
        {
            "evidence_id": _opaque(item.evidence_id),
            "start": item.start,
            "end": item.end,
            "source_id": _opaque(item.source_id) if item.source_id else "",
            "source_length": item.source_length,
        }
        for item in evidence
    ]
    return _digest_payload(sorted(payload, key=lambda item: item["evidence_id"]))


def _digest_adjudications(adjudications: Sequence[ClinicianAdjudication]) -> str:
    payload = [
        {
            "claim_id": _opaque(item.claim_id),
            "evidence_id": _opaque(item.evidence_id),
            "label": item.label,
        }
        for item in adjudications
    ]
    return _digest_payload(
        sorted(
            payload,
            key=lambda item: (
                item["claim_id"],
                item["evidence_id"],
                item["label"],
            ),
        )
    )


__all__ = [
    "ADJUDICATION_CONTRADICTS",
    "ADJUDICATION_IRRELEVANT",
    "ADJUDICATION_LABELS",
    "ADJUDICATION_SUPPORTS",
    "ADJUDICATION_UNCLEAR",
    "Adjudication",
    "AtomicClaim",
    "CITATION_SUPPORT_DISCLAIMER",
    "CITATION_SUPPORT_METRICS",
    "CITATION_SUPPORT_METRICS_SCHEMA_VERSION",
    "Citation",
    "CitationSupportError",
    "CitationSupportMetrics",
    "CitationSupportReport",
    "ClinicianAdjudication",
    "ClinicianAdjudicationMetrics",
    "ClinicianMetrics",
    "Claim",
    "DeterministicSpanChecks",
    "Evidence",
    "EvidenceSpan",
    "HUMAN_REVIEW_REQUIRED",
    "SCHEMA_VERSION",
    "SpanChecks",
    "build_citation_support_report",
    "citation_support_metrics",
    "compute_citation_support_metrics",
    "evaluate_citation_support",
    "run_citation_support_metrics",
    "score_citation_support",
]
