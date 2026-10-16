"""Privacy-safe unsupported-claim scoring for clinical summaries.

The evaluator scores already-atomic summary claims against caller-supplied,
approved evidence.  It deliberately does not infer clinical truth, run a
model, or fetch evidence.  Callers provide an opaque claim fingerprint,
citation identifiers, and the local adjudication relation for each evidence
record.  The report retains only claim classes, four-state counts, rates, and
deterministic bootstrap intervals; claim values and identifiers never cross
the report boundary.

This is evaluation evidence, not a clinical decision, compliance
certification, or guarantee of summary faithfulness.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import wraps
from itertools import islice
from pathlib import Path
from types import MappingProxyType
from typing import Any

from openmed.eval.metrics import BootstrapCI, bootstrap_ci

SUMMARY_UNSUPPORTED_CLAIMS = "summary_unsupported_claims"
SUMMARY_UNSUPPORTED_CLAIMS_SCHEMA_VERSION = 1
SCHEMA_VERSION = SUMMARY_UNSUPPORTED_CLAIMS_SCHEMA_VERSION

UNSUPPORTED_CLAIMS_DISCLAIMER = (
    "Unsupported-claim rates are evaluation evidence only; they are not a "
    "clinical decision, compliance certification, or faithfulness guarantee."
)
APPROVED_EVIDENCE_DISCLAIMER = UNSUPPORTED_CLAIMS_DISCLAIMER


class ClaimState(str, Enum):
    """Controlled state assigned to one atomic summary claim."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"
    UNCITED = "uncited"


# String constants make the contract convenient for JSON fixture authors.
SUPPORTED = ClaimState.SUPPORTED.value
CONTRADICTED = ClaimState.CONTRADICTED.value
UNRESOLVED = ClaimState.UNRESOLVED.value
UNCITED = ClaimState.UNCITED.value
CLAIM_STATES: tuple[str, ...] = (SUPPORTED, CONTRADICTED, UNRESOLVED, UNCITED)
UNSUPPORTED_STATES: tuple[str, ...] = (CONTRADICTED, UNRESOLVED, UNCITED)

DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_BOOTSTRAP_ALPHA = 0.05
DEFAULT_BOOTSTRAP_SEED = 0

_CLAIM_CLASS_RE = re.compile(r"[a-z0-9][a-z0-9_]{0,63}\Z")
_PUBLIC_CLAIM_CLASSES = frozenset(
    {
        "allergy",
        "diagnosis",
        "finding",
        "history",
        "lab",
        "medication",
        "other",
        "procedure",
        "symptom",
        "treatment",
        "vital",
    }
)
_MAX_IDENTIFIER_LENGTH = 256
_MAX_CLAIMS = 100_000
_MAX_EVIDENCE = 200_000
_MAX_CITATIONS_PER_CLAIM = 512


def _safe_input(function):
    @wraps(function)
    def checked(*args, **kwargs):
        message = "summary evaluation input cannot be normalized"
        kind = ValueError
        try:
            return function(*args, **kwargs)
        except (ValueError, TypeError) as error:
            trace = error.__traceback__
            while trace is not None and trace.tb_next is not None:
                trace = trace.tb_next
            if (
                trace is not None
                and trace.tb_frame.f_globals.get("__name__") == __name__
            ):
                message = str(error)
                kind = TypeError if isinstance(error, TypeError) else ValueError
        except Exception:
            pass
        raise kind(message)

    return checked


@dataclass(frozen=True, slots=True, repr=False)
class SummaryClaim:
    """One atomic summary claim with value-free matching metadata.

    ``claim_key`` may be a caller's structured fact key or opaque fingerprint.
    It is converted to a SHA-256 token immediately, so this record does not
    retain a raw claim value.  The token is used only for matching and is not
    emitted in the aggregate report.

    Args:
        claim_id: Stable identifier used only to connect evidence records.
        claim_class: Controlled class such as ``diagnosis`` or ``medication``.
        claim_key: Structured fact key or opaque fingerprint for matching.
        evidence_ids: Citation identifiers supplied by the summary evaluator.
        summary_id: Optional local summary/document grouping identifier. It is
            retained only for future caller-side grouping and never reported.
    """

    claim_id: str
    claim_class: str
    claim_key: Any = None
    evidence_ids: tuple[str, ...] = ()
    summary_id: str = ""

    @_safe_input
    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _identifier(self.claim_id, "claim_id"))
        object.__setattr__(
            self,
            "claim_class",
            _claim_class(self.claim_class),
        )
        object.__setattr__(
            self,
            "claim_key",
            _fingerprint(self.claim_id if self.claim_key is None else self.claim_key),
        )
        object.__setattr__(
            self,
            "evidence_ids",
            _identifier_sequence(self.evidence_ids, "evidence_ids"),
        )
        if self.summary_id:
            object.__setattr__(
                self,
                "summary_id",
                _identifier(self.summary_id, "summary_id"),
            )

    @classmethod
    @_safe_input
    def from_mapping(
        cls,
        value: Mapping[str, Any] | Any,
        *,
        default_id: str | None = None,
    ) -> "SummaryClaim":
        """Normalize one mapping or object without exposing submitted values."""

        data = _record_mapping(value, "summary claims")
        claim_id = _first(data, "claim_id", "id", "fact_id", "key")
        if claim_id is None:
            claim_id = default_id
        if claim_id is None:
            raise ValueError("summary claims require a claim_id")
        claim_class = _first(
            data,
            "claim_class",
            "class",
            "claim_type",
            "type",
            "category",
            "label",
        )
        if claim_class is None:
            raise ValueError("summary claims require a claim_class")
        claim_key = _first(
            data,
            "claim_key",
            "claim_fingerprint",
            "fact_key",
            "canonical_key",
            "content_hash",
            "value_hash",
            "value",
            "text",
        )
        evidence = _first(
            data,
            "evidence_ids",
            "evidence_id",
            "citation_ids",
            "citation_id",
            "citations",
            "evidence",
            "sources",
        )
        summary_id = _first(
            data,
            "summary_id",
            "document_id",
            "fixture_id",
            "case_id",
        )
        return cls(
            claim_id=claim_id,
            claim_class=claim_class,
            claim_key=claim_key,
            evidence_ids=_citation_ids(evidence),
            summary_id=summary_id if summary_id is not None else "",
        )

    def __repr__(self) -> str:
        """Return a representation that cannot print claim values."""

        return (
            "SummaryClaim(claim_id=<redacted>, "
            f"claim_class={self.claim_class!r}, "
            f"evidence_count={len(self.evidence_ids)})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ApprovedEvidence:
    """One local evidence relation eligible for claim scoring.

    Evidence is linked to a claim by the cited ``evidence_id`` first.  An
    optional ``claim_id`` or ``claim_key`` adds an integrity check; when one is
    present it must match the citing claim.  Evidence without a relation is
    unresolved rather than silently treated as support.
    """

    evidence_id: str
    relation: str = UNRESOLVED
    approved: bool = True
    claim_id: str = ""
    claim_key: Any = None
    claim_class: str = ""

    @_safe_input
    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_id",
            _identifier(self.evidence_id, "evidence_id"),
        )
        object.__setattr__(self, "relation", _relation(self.relation))
        if type(self.approved) is not bool:
            raise TypeError("evidence approved flag must be a boolean")
        if self.claim_id:
            object.__setattr__(
                self,
                "claim_id",
                _identifier(self.claim_id, "evidence claim_id"),
            )
        if self.claim_key is not None:
            object.__setattr__(self, "claim_key", _fingerprint(self.claim_key))
        if self.claim_class:
            object.__setattr__(
                self,
                "claim_class",
                _claim_class(self.claim_class),
            )

    @classmethod
    @_safe_input
    def from_mapping(
        cls,
        value: Mapping[str, Any] | Any,
        *,
        default_id: str | None = None,
    ) -> "ApprovedEvidence":
        """Normalize one evidence mapping or object."""

        data = _record_mapping(value, "approved evidence")
        evidence_id = _first(
            data,
            "evidence_id",
            "id",
            "citation_id",
            "source_id",
        )
        if evidence_id is None:
            evidence_id = default_id
        if evidence_id is None:
            raise ValueError("approved evidence requires an evidence_id")
        approved = _first(data, "approved", "is_approved", "approved_evidence")
        if approved is None:
            approved = True
        if type(approved) is not bool:
            raise TypeError("evidence approved flag must be a boolean")
        relation = _relation_from_mapping(data)
        claim_id = _first(data, "claim_id", "summary_claim_id", "fact_id")
        claim_key = _first(
            data,
            "claim_key",
            "claim_fingerprint",
            "fact_key",
            "canonical_key",
            "content_hash",
            "value_hash",
            "value",
            "text",
        )
        claim_class = _first(
            data,
            "claim_class",
            "class",
            "claim_type",
            "type",
            "category",
            "label",
        )
        return cls(
            evidence_id=evidence_id,
            relation=relation,
            approved=approved,
            claim_id=claim_id if claim_id is not None else "",
            claim_key=claim_key,
            claim_class=claim_class if claim_class is not None else "",
        )

    def __repr__(self) -> str:
        """Return a representation without evidence or claim values."""

        return (
            "ApprovedEvidence(evidence_id=<redacted>, "
            f"relation={self.relation!r}, approved={self.approved})"
        )


@dataclass(frozen=True, slots=True)
class ClaimAssessment:
    """Value-free state and citation counts for one scored claim."""

    claim_class: str
    state: str
    cited_evidence_count: int
    approved_evidence_count: int
    matching_evidence_count: int

    @_safe_input
    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_class", _claim_class(self.claim_class))
        if self.state not in CLAIM_STATES:
            raise ValueError("unsupported claim state")
        for value in (
            self.cited_evidence_count,
            self.approved_evidence_count,
            self.matching_evidence_count,
        ):
            _count(value)
        if (
            not self.matching_evidence_count
            <= self.approved_evidence_count
            <= self.cited_evidence_count
        ):
            raise ValueError("inconsistent claim evidence counts")
        if (self.state == UNCITED) != (self.cited_evidence_count == 0):
            raise ValueError("inconsistent uncited claim state")
        if (
            self.state in (SUPPORTED, CONTRADICTED)
            and self.matching_evidence_count != self.cited_evidence_count
        ):
            raise ValueError("claim state requires matching approved evidence")

    @property
    def unsupported(self) -> bool:
        """Return whether this claim is outside the supported state."""

        return self.state in UNSUPPORTED_STATES

    @property
    def review_required(self) -> bool:
        """Return whether this claim should be routed to human review."""

        return self.state != SUPPORTED

    def to_dict(self) -> dict[str, Any]:
        """Return safe per-claim evidence without identifiers or values."""

        return {
            "approved_evidence_count": self.approved_evidence_count,
            "cited_evidence_count": self.cited_evidence_count,
            "claim_class": self.claim_class,
            "matching_evidence_count": self.matching_evidence_count,
            "review_required": self.review_required,
            "state": self.state,
        }


@dataclass(frozen=True, slots=True)
class ClaimClassMetrics:
    """Four-state counts and an unsupported-rate interval for one class."""

    claim_class: str
    supported: int
    contradicted: int
    unresolved: int
    uncited: int
    bootstrap_ci: BootstrapCI

    @_safe_input
    def __post_init__(self) -> None:
        counts = (
            self.supported,
            self.contradicted,
            self.unresolved,
            self.uncited,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in counts
        ):
            raise TypeError("claim-state counts must be integers")
        if any(value < 0 for value in counts):
            raise ValueError("claim-state counts must be non-negative")
        name = (
            "overall"
            if self.claim_class == "overall"
            else _claim_class(self.claim_class)
        )
        object.__setattr__(self, "claim_class", name)
        if not isinstance(self.bootstrap_ci, BootstrapCI):
            raise TypeError("invalid bootstrap interval")
        interval = replace(self.bootstrap_ci)
        _validate_bootstrap(interval.n_resamples, interval.alpha, 0)
        for value in (interval.point, interval.lower, interval.upper):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("invalid bootstrap interval bounds")
        if (
            interval.point != self.unsupported_rate
            or not interval.lower <= interval.point <= interval.upper
            or type(interval.degenerate) is not bool
            or interval.degenerate != (self.claim_count < 2)
            or (
                interval.degenerate
                and not interval.lower == interval.point == interval.upper
            )
        ):
            raise ValueError("bootstrap interval does not match claim counts")
        object.__setattr__(self, "bootstrap_ci", interval)

    @property
    def claim_count(self) -> int:
        """Return the number of atomic claims in this class."""

        return self.supported + self.contradicted + self.unresolved + self.uncited

    @property
    def unsupported_count(self) -> int:
        """Return contradicted, unresolved, and uncited claims."""

        return self.contradicted + self.unresolved + self.uncited

    @property
    def unsupported_rate(self) -> float:
        """Return the fraction of claims that are not supported."""

        return _rate(self.unsupported_count, self.claim_count)

    @property
    def rate(self) -> float:
        """Compatibility alias for :attr:`unsupported_rate`."""

        return self.unsupported_rate

    @property
    def total(self) -> int:
        """Compatibility alias for :attr:`claim_count`."""

        return self.claim_count

    @property
    def ci(self) -> BootstrapCI:
        """Compatibility alias for :attr:`bootstrap_ci`."""

        return self.bootstrap_ci

    @property
    def counts(self) -> dict[str, int]:
        """Return four-state counts in the stable contract order."""

        return {
            SUPPORTED: self.supported,
            CONTRADICTED: self.contradicted,
            UNRESOLVED: self.unresolved,
            UNCITED: self.uncited,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic aggregate metrics without claim content."""

        return {
            "bootstrap_ci": self.bootstrap_ci.to_dict(),
            "claim_class": self.claim_class,
            "claim_count": self.claim_count,
            "counts": self.counts,
            "supported_count": self.supported,
            "contradicted_count": self.contradicted,
            "unresolved_count": self.unresolved,
            "uncited_count": self.uncited,
            "unsupported_count": self.unsupported_count,
            "unsupported_rate": self.unsupported_rate,
        }

    def __getitem__(self, key: str) -> Any:
        """Provide mapping-style access to the serialized metric."""

        return self.to_dict()[key]


@dataclass(frozen=True, slots=True)
class SummaryUnsupportedClaimsReport:
    """Aggregate unsupported-claim report with safe JSON and Markdown views."""

    overall: ClaimClassMetrics
    by_claim_class: Mapping[str, ClaimClassMetrics]
    assessments: tuple[ClaimAssessment, ...] = ()
    evidence_count: int = 0
    approved_evidence_count: int = 0
    evidence_digest: str = ""
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA
    seed: int = DEFAULT_BOOTSTRAP_SEED
    schema_version: int = SUMMARY_UNSUPPORTED_CLAIMS_SCHEMA_VERSION

    @_safe_input
    def __post_init__(self) -> None:
        _validate_bootstrap(self.n_resamples, self.alpha, self.seed)
        _count(self.evidence_count)
        _count(self.approved_evidence_count)
        if self.approved_evidence_count > self.evidence_count:
            raise ValueError("inconsistent approved evidence count")
        if (
            not isinstance(self.evidence_digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.evidence_digest) is None
        ):
            raise ValueError("invalid evidence digest")
        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError("unsupported report schema")
        if not isinstance(self.overall, ClaimClassMetrics):
            raise TypeError("invalid overall claim metrics")
        overall = replace(self.overall)
        if overall.claim_class != "overall":
            raise ValueError("invalid overall claim class")
        classes = {}
        for key, metric in _bounded_items(self.by_claim_class.items(), _MAX_CLAIMS):
            if not isinstance(metric, ClaimClassMetrics):
                raise TypeError("invalid per-class metrics")
            normalized = replace(metric)
            if key != normalized.claim_class or key == "overall":
                raise ValueError("invalid report class key")
            classes[key] = normalized
        assessments = []
        for item in _bounded_items(self.assessments, _MAX_CLAIMS):
            if not isinstance(item, ClaimAssessment):
                raise TypeError("invalid claim assessment")
            assessments.append(replace(item))
        by_class = {}
        for item in assessments:
            by_class.setdefault(item.claim_class, Counter())[item.state] += 1
        if set(by_class) != set(classes) or len(assessments) != overall.claim_count:
            raise ValueError("report assessments do not match class metrics")
        for name, metric in classes.items():
            if any(
                by_class[name][state] != metric.counts[state] for state in CLAIM_STATES
            ):
                raise ValueError("inconsistent per-class counts")
        if any(
            sum(metric.counts[state] for metric in classes.values())
            != overall.counts[state]
            for state in CLAIM_STATES
        ):
            raise ValueError("inconsistent overall counts")
        for metric in (overall, *classes.values()):
            if (
                metric.bootstrap_ci.n_resamples != self.n_resamples
                or metric.bootstrap_ci.alpha != self.alpha
            ):
                raise ValueError("inconsistent bootstrap metadata")
        object.__setattr__(self, "overall", overall)
        object.__setattr__(self, "by_claim_class", MappingProxyType(classes))
        object.__setattr__(self, "assessments", tuple(assessments))

    @property
    def claim_count(self) -> int:
        """Return the number of scored atomic claims."""

        return self.overall.claim_count

    @property
    def per_class(self) -> Mapping[str, ClaimClassMetrics]:
        """Compatibility alias for :attr:`by_claim_class`."""

        return self.by_claim_class

    @property
    def by_class(self) -> Mapping[str, ClaimClassMetrics]:
        """Compatibility alias for :attr:`by_claim_class`."""

        return self.by_claim_class

    @property
    def unsupported_count(self) -> int:
        """Return the total unsupported claim count."""

        return self.overall.unsupported_count

    @property
    def unsupported_claims(self) -> int:
        """Compatibility alias for :attr:`unsupported_count`."""

        return self.unsupported_count

    @property
    def unsupported_rate(self) -> float:
        """Return the overall unsupported claim rate."""

        return self.overall.unsupported_rate

    @property
    def state_counts(self) -> dict[str, int]:
        """Return overall state counts."""

        return self.overall.counts

    @property
    def review_required_count(self) -> int:
        """Return the number of claims that should be reviewed by a person."""

        return sum(assessment.review_required for assessment in self.assessments)

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate and value-free per-claim scoring evidence."""

        return {
            "alpha": self.alpha,
            "assessments": [assessment.to_dict() for assessment in self.assessments],
            "by_claim_class": {
                claim_class: self.by_claim_class[claim_class].to_dict()
                for claim_class in sorted(self.by_claim_class)
            },
            "claim_count": self.claim_count,
            "disclaimer": UNSUPPORTED_CLAIMS_DISCLAIMER,
            "evidence_count": self.evidence_count,
            "approved_evidence_count": self.approved_evidence_count,
            "evidence_digest": self.evidence_digest,
            "human_review_required": True,
            "n_resamples": self.n_resamples,
            "overall": self.overall.to_dict(),
            "review_required_count": self.review_required_count,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "suite": SUMMARY_UNSUPPORTED_CLAIMS,
            "unsupported_count": self.unsupported_count,
            "unsupported_rate": self.unsupported_rate,
        }

    def __getitem__(self, key: str) -> Any:
        """Provide mapping-style access to the serialized report."""

        return self.to_dict()[key]

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report as deterministic JSON."""

        if isinstance(indent, bool) or not isinstance(indent, int) or indent < 0:
            raise ValueError("JSON indent must be a non-negative integer")
        return (
            json.dumps(
                self.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                indent=indent,
                sort_keys=True,
            )
            + "\n"
        )

    def to_markdown(self) -> str:
        """Render counts, rates, intervals, and review guidance only."""

        lines = [
            "# Unsupported claim rate for clinical summaries",
            "",
            UNSUPPORTED_CLAIMS_DISCLAIMER,
            "",
            f"Claims: {self.claim_count}",
            f"Unsupported: {self.unsupported_count} ({self.unsupported_rate:.6f})",
            (
                "Evidence: "
                f"{self.approved_evidence_count}/{self.evidence_count} approved "
                f"(digest {self.evidence_digest})"
            ),
            (
                "Bootstrap: "
                f"{self.n_resamples} resamples, alpha={self.alpha:.6f}, "
                f"seed={self.seed}"
            ),
            "",
            "| Claim class | Claims | Supported | Contradicted | Unresolved | "
            f"Uncited | Unsupported rate | {100.0 * (1.0 - self.alpha):.1f}% interval |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for claim_class in sorted(self.by_claim_class):
            metric = self.by_claim_class[claim_class]
            interval = (
                f"[{metric.bootstrap_ci.lower:.6f}, {metric.bootstrap_ci.upper:.6f}]"
            )
            lines.append(
                f"| {claim_class} | {metric.claim_count} | {metric.supported} | "
                f"{metric.contradicted} | {metric.unresolved} | {metric.uncited} | "
                f"{metric.unsupported_rate:.6f} | {interval} |"
            )
        lines.extend(
            [
                "",
                "Claims in contradicted, unresolved, or uncited states require "
                "human review before any downstream clinical use.",
                "",
            ]
        )
        return "\n".join(lines)

    @_safe_input
    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write deterministic JSON to *path*."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json(indent=indent), encoding="utf-8")
        return output

    @_safe_input
    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown to *path*."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_markdown(), encoding="utf-8")
        return output

    def to_benchmark_report(
        self,
        *,
        model_name: str = "summary-unsupported-claim-evaluator",
        device: str = "local",
        generated_at: str | None = None,
    ) -> Any:
        """Return a standard benchmark wrapper containing aggregate metrics."""

        from openmed.eval.report import BenchmarkReport

        return BenchmarkReport(
            suite=SUMMARY_UNSUPPORTED_CLAIMS,
            model_name=model_name,
            device=device,
            fixture_count=self.claim_count,
            generated_at=generated_at,
            metrics=self.to_dict(),
            metadata={
                "artifact_type": "openmed.eval.summary_unsupported_claims",
                "aggregate_only": True,
                "human_review_required": True,
            },
        )


# Singular alias keeps imports readable for callers scoring one report.
SummaryUnsupportedClaimReport = SummaryUnsupportedClaimsReport
ClaimStatus = ClaimState
EvidenceRecord = ApprovedEvidence


@_safe_input
def score_summary_claims(
    claims: Iterable[SummaryClaim | Mapping[str, Any] | Any],
    evidence: Iterable[ApprovedEvidence | Mapping[str, Any] | Any] = (),
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int | None = None,
    confidence_level: float | None = None,
    approved_evidence: Iterable[ApprovedEvidence | Mapping[str, Any] | Any]
    | None = None,
) -> SummaryUnsupportedClaimsReport:
    """Score atomic summary claims against approved local evidence.

    A claim with no citation is ``uncited``.  A cited claim is ``supported``
    only when every cited evidence row is approved, linked to the claim, and
    has a support relation.  A valid contradiction is ``contradicted``.
    Missing, unapproved, mismatched, unknown, or conflicting evidence is
    ``unresolved``.  Unsupported rate is the sum of the latter three states
    divided by all claims.

    Bootstrap resampling is claim-level, uses the standard library, and is
    seeded.  No model loading or network access occurs.
    """

    if approved_evidence is not None:
        if evidence:
            raise ValueError("pass either evidence or approved_evidence, not both")
        evidence = approved_evidence
    if bootstrap_resamples is not None:
        n_resamples = bootstrap_resamples
    if confidence_level is not None:
        if isinstance(confidence_level, bool):
            raise TypeError("confidence_level must be numeric")
        alpha = 1.0 - float(confidence_level)
    _validate_bootstrap(n_resamples, alpha, seed)

    normalized_claims = _normalize_claims(claims)
    normalized_evidence = _normalize_evidence(evidence)
    if 2 * len(normalized_claims) * n_resamples > 20_000_000:
        raise ValueError("bootstrap work exceeds the evaluation limit")
    evidence_by_id = {item.evidence_id: item for item in normalized_evidence}
    assessments = tuple(
        _assess_claim(claim, evidence_by_id) for claim in normalized_claims
    )
    groups: dict[str, list[ClaimAssessment]] = {}
    for assessment in assessments:
        groups.setdefault(assessment.claim_class, []).append(assessment)
    by_claim_class = {
        name: _class_metrics(
            name, rows, n_resamples=n_resamples, alpha=alpha, seed=seed
        )
        for name, rows in sorted(groups.items())
    }
    overall = _class_metrics(
        "overall",
        list(assessments),
        n_resamples=n_resamples,
        alpha=alpha,
        seed=seed,
    )
    return SummaryUnsupportedClaimsReport(
        overall=overall,
        by_claim_class=by_claim_class,
        assessments=assessments,
        evidence_count=len(normalized_evidence),
        approved_evidence_count=sum(
            evidence_item.approved for evidence_item in normalized_evidence
        ),
        evidence_digest=_evidence_digest(normalized_evidence),
        n_resamples=n_resamples,
        alpha=alpha,
        seed=seed,
    )


def evaluate_summary_unsupported_claims(
    claims: Iterable[SummaryClaim | Mapping[str, Any] | Any],
    approved_evidence: Iterable[ApprovedEvidence | Mapping[str, Any] | Any] = (),
    **kwargs: Any,
) -> SummaryUnsupportedClaimsReport:
    """Compatibility entry point with an explicit approved-evidence name."""

    return score_summary_claims(claims, approved_evidence, **kwargs)


def compute_summary_unsupported_claims(
    claims: Iterable[SummaryClaim | Mapping[str, Any] | Any],
    evidence: Iterable[ApprovedEvidence | Mapping[str, Any] | Any] = (),
    **kwargs: Any,
) -> SummaryUnsupportedClaimsReport:
    """Return a deterministic unsupported-claim report."""

    return score_summary_claims(claims, evidence, **kwargs)


def run_summary_unsupported_claims(
    claims: Iterable[SummaryClaim | Mapping[str, Any] | Any],
    evidence: Iterable[ApprovedEvidence | Mapping[str, Any] | Any] = (),
    **kwargs: Any,
) -> SummaryUnsupportedClaimsReport:
    """Alias for :func:`score_summary_claims` used by eval suites."""

    return score_summary_claims(claims, evidence, **kwargs)


def build_summary_unsupported_claims_report(
    claims: Iterable[SummaryClaim | Mapping[str, Any] | Any],
    evidence: Iterable[ApprovedEvidence | Mapping[str, Any] | Any] = (),
    **kwargs: Any,
) -> SummaryUnsupportedClaimsReport:
    """Build the typed report used by JSON, Markdown, or benchmark wrappers."""

    return score_summary_claims(claims, evidence, **kwargs)


@_safe_input
def score_summary_claim(
    claim: SummaryClaim | Mapping[str, Any] | Any,
    evidence: Iterable[ApprovedEvidence | Mapping[str, Any] | Any] = (),
) -> ClaimAssessment:
    """Score one claim and return its value-free four-state assessment."""

    normalized_claims = _normalize_claims((claim,))
    normalized_evidence = _normalize_evidence(evidence)
    return _assess_claim(
        normalized_claims[0],
        {item.evidence_id: item for item in normalized_evidence},
    )


def unsupported_claim_rate(
    claims: Iterable[SummaryClaim | Mapping[str, Any] | Any],
    evidence: Iterable[ApprovedEvidence | Mapping[str, Any] | Any] = (),
) -> float:
    """Return the overall unsupported rate without retaining the report."""

    return score_summary_claims(claims, evidence).unsupported_rate


def compute_unsupported_claim_rate(
    claims: Iterable[SummaryClaim | Mapping[str, Any] | Any],
    evidence: Iterable[ApprovedEvidence | Mapping[str, Any] | Any] = (),
) -> float:
    """Compatibility alias for :func:`unsupported_claim_rate`."""

    return unsupported_claim_rate(claims, evidence)


def _normalize_claims(
    claims: Iterable[SummaryClaim | Mapping[str, Any] | Any],
) -> tuple[SummaryClaim, ...]:
    records = _record_sequence(claims, "summary claims", _MAX_CLAIMS)
    normalized: list[SummaryClaim] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(records, start=1):
        claim = (
            replace(value)
            if isinstance(value, SummaryClaim)
            else SummaryClaim.from_mapping(value, default_id=f"claim-{index}")
        )
        if claim.claim_id in seen_ids:
            raise ValueError("summary claims contain duplicate claim_id values")
        seen_ids.add(claim.claim_id)
        normalized.append(claim)
    return tuple(
        sorted(
            normalized,
            key=lambda item: (item.claim_class, item.claim_key, item.claim_id),
        )
    )


def _normalize_evidence(
    evidence: Iterable[ApprovedEvidence | Mapping[str, Any] | Any],
) -> tuple[ApprovedEvidence, ...]:
    records = _record_sequence(evidence, "approved evidence", _MAX_EVIDENCE)
    normalized: list[ApprovedEvidence] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(records, start=1):
        item = (
            replace(value)
            if isinstance(value, ApprovedEvidence)
            else ApprovedEvidence.from_mapping(value, default_id=f"evidence-{index}")
        )
        if item.evidence_id in seen_ids:
            raise ValueError("approved evidence contains duplicate evidence_id values")
        seen_ids.add(item.evidence_id)
        normalized.append(item)
    return tuple(sorted(normalized, key=lambda item: item.evidence_id))


def _assess_claim(
    claim: SummaryClaim,
    evidence_by_id: Mapping[str, ApprovedEvidence],
) -> ClaimAssessment:
    cited_ids = claim.evidence_ids
    cited = [evidence_by_id[item] for item in cited_ids if item in evidence_by_id]
    approved = [item for item in cited if item.approved]
    matching = [item for item in approved if _matches_claim(claim, item)]

    if not cited_ids:
        state = UNCITED
    elif len(cited) != len(cited_ids):
        state = UNRESOLVED
    elif len(approved) != len(cited):
        state = UNRESOLVED
    elif len(matching) != len(approved) or not matching:
        state = UNRESOLVED
    else:
        relations = {item.relation for item in matching}
        if relations == {SUPPORTED}:
            state = SUPPORTED
        elif relations == {CONTRADICTED}:
            state = CONTRADICTED
        else:
            state = UNRESOLVED

    return ClaimAssessment(
        claim_class=claim.claim_class,
        state=state,
        cited_evidence_count=len(cited_ids),
        approved_evidence_count=len(approved),
        matching_evidence_count=len(matching),
    )


def _matches_claim(claim: SummaryClaim, evidence: ApprovedEvidence) -> bool:
    if evidence.claim_class and evidence.claim_class != claim.claim_class:
        return False
    if evidence.claim_id and evidence.claim_id != claim.claim_id:
        return False
    if evidence.claim_key is not None and evidence.claim_key != claim.claim_key:
        return False
    return True


def _class_metrics(
    claim_class: str,
    assessments: Sequence[ClaimAssessment],
    *,
    n_resamples: int,
    alpha: float,
    seed: int,
) -> ClaimClassMetrics:
    counts = Counter(assessment.state for assessment in assessments)
    flags = [1 if assessment.unsupported else 0 for assessment in assessments]
    ci = bootstrap_ci(
        flags,
        lambda sample: _sample_rate(sample),
        n_resamples=n_resamples,
        alpha=alpha,
        seed=seed,
    )
    return ClaimClassMetrics(
        claim_class=claim_class,
        supported=counts.get(SUPPORTED, 0),
        contradicted=counts.get(CONTRADICTED, 0),
        unresolved=counts.get(UNRESOLVED, 0),
        uncited=counts.get(UNCITED, 0),
        bootstrap_ci=ci,
    )


def _sample_rate(values: Sequence[Any]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def _evidence_digest(evidence: Sequence[ApprovedEvidence]) -> str:
    payload = [
        {
            "approved": item.approved,
            "claim_class": item.claim_class,
            "claim_id": item.claim_id,
            "claim_key": item.claim_key,
            "evidence_id": item.evidence_id,
            "relation": item.relation,
        }
        for item in evidence
    ]
    canonical = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _relation_from_mapping(data: Mapping[str, Any]) -> str:
    labels = [
        _relation(data[key])
        for key in (
            "relation",
            "stance",
            "verdict",
            "claim_state",
            "state",
            "status",
            "assessment",
            "support",
        )
        if key in data
    ]
    flags = []
    for key, target in (
        ("supports", SUPPORTED),
        ("supported", SUPPORTED),
        ("contradicts", CONTRADICTED),
        ("contradicted", CONTRADICTED),
    ):
        if key in data:
            if type(data[key]) is not bool:
                raise TypeError("evidence relation flags must be boolean")
            flags.append((target, data[key]))
            if data[key]:
                labels.append(target)
    if not labels:
        return UNRESOLVED
    label = labels[0]
    if any(item != label for item in labels) or any(
        target == label and not flag for target, flag in flags
    ):
        raise ValueError("conflicting evidence relation aliases")
    return label


def _relation(value: Any) -> str:
    if isinstance(value, ClaimState):
        return value.value
    if isinstance(value, bool):
        return SUPPORTED if value else CONTRADICTED
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")
    if normalized in {
        "supported",
        "support",
        "supports",
        "entailed",
        "entails",
        "affirmed",
        "true",
    }:
        return SUPPORTED
    if normalized in {
        "contradicted",
        "contradiction",
        "contradicts",
        "refuted",
        "refutes",
        "false",
    }:
        return CONTRADICTED
    if normalized in {
        "unresolved",
        "unknown",
        "uncertain",
        "ambiguous",
        "pending",
    }:
        return UNRESOLVED
    raise ValueError("evidence relation must be supported, contradicted, or unresolved")


def _claim_class(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("claim_class must be a string")
    if re.fullmatch(r"class_[0-9a-f]{16}", value):
        return value
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")
    if not _CLAIM_CLASS_RE.fullmatch(normalized):
        raise ValueError("claim_class must be a bounded identifier")
    if normalized in _PUBLIC_CLAIM_CLASSES:
        return normalized
    return "class_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _identifier(value: Any, field_name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"{field_name} must be a string identifier")
    normalized = str(value).strip()
    if not normalized or len(normalized) > _MAX_IDENTIFIER_LENGTH:
        raise ValueError(f"{field_name} must be a bounded non-empty identifier")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError(f"{field_name} must not contain control characters")
    return normalized


def _identifier_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        raise TypeError(f"{field_name} must be a sequence of identifiers")
    if len(values) > _MAX_CITATIONS_PER_CLAIM:
        raise ValueError(f"{field_name} exceeds the citation limit")
    normalized = tuple(
        _identifier(item, field_name)
        for item in _bounded_items(values, _MAX_CITATIONS_PER_CLAIM)
    )
    return tuple(sorted(set(normalized)))


def _citation_ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        item = _first(value, "evidence_id", "id", "citation_id", "source_id")
        return _identifier_sequence(item, "evidence_ids")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        extracted: list[Any] = []
        for item in _bounded_items(value, _MAX_CITATIONS_PER_CLAIM):
            if isinstance(item, Mapping):
                extracted.append(
                    _first(item, "evidence_id", "id", "citation_id", "source_id")
                )
            else:
                extracted.append(item)
        return _identifier_sequence(extracted, "evidence_ids")
    return _identifier_sequence(value, "evidence_ids")


def _fingerprint(value: Any) -> str:
    if isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        return value
    try:
        canonical = json.dumps(
            _json_value(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        raise TypeError("claim matching keys must be JSON-compatible") from None
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _json_value(value: Any, depth: int = 0, budget: list[int] | None = None) -> Any:
    if budget is None:
        budget = [10000]
    budget[0] -= 1
    if depth > 32 or budget[0] < 0:
        raise ValueError("claim matching key exceeds the structure limit")
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > 1_000_000:
            raise ValueError("claim matching key exceeds the value limit")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("claim matching keys must contain finite numbers")
        return value
    if isinstance(value, bytes):
        if len(value) > 1_000_000:
            raise ValueError("claim matching key exceeds the value limit")
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Mapping):
        result = {}
        for key, item in _bounded_items(value.items(), budget[0]):
            if not isinstance(key, str) or len(key) > 1024:
                raise TypeError("claim matching object keys must be bounded strings")
            result[key] = _json_value(item, depth + 1, budget)
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, depth + 1, budget)
            for item in _bounded_items(value, budget[0])
        ]
    raise TypeError("claim matching keys must be JSON-compatible")


def _record_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        data = vars(value)
    except TypeError:
        raise TypeError(
            f"{field_name} must contain mappings or record objects"
        ) from None
    if not isinstance(data, Mapping):
        raise TypeError(f"{field_name} must contain mappings or record objects")
    return data


def _bounded_items(value: Any, limit: int) -> tuple[Any, ...]:
    rows = tuple(islice(iter(value), limit + 1))
    if len(rows) > limit:
        raise ValueError("input collection exceeds the record limit")
    return rows


def _record_sequence(value: Any, field_name: str, limit: int) -> tuple[Any, ...]:
    if isinstance(value, Mapping):
        # A single record is accepted; a mapping of ids to records is also
        # useful for local evidence stores and is kept value-free in errors.
        marker_keys = {
            "claim_id",
            "claim_class",
            "claim_key",
            "evidence_id",
            "relation",
            "approved",
        }
        if marker_keys.intersection(value):
            return (value,)
        rows: list[dict[str, Any]] = []
        for key, item in _bounded_items(value.items(), limit):
            if not isinstance(item, Mapping):
                raise TypeError(f"{field_name} mapping values must be records")
            row = dict(item)
            if "claim_id" not in row and "evidence_id" not in row:
                # The neutral ``id`` alias is interpreted as claim_id or
                # evidence_id by the respective normalizer.
                row["id"] = key
            rows.append(row)
        if len(rows) > limit:
            raise ValueError(f"{field_name} exceeds the record limit")
        return tuple(rows)
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be an iterable of records")
    try:
        record_rows = _bounded_items(value, limit)
    except TypeError:
        raise TypeError(f"{field_name} must be an iterable of records") from None
    if len(record_rows) > limit:
        raise ValueError(f"{field_name} exceeds the record limit")
    return record_rows


def _first(data: Mapping[str, Any], *keys: str) -> Any:
    values = [data[key] for key in keys if key in data and data[key] is not None]
    if values and any(value != values[0] for value in values[1:]):
        raise ValueError("conflicting input aliases")
    return values[0] if values else None


def _count(value: Any) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("counts must be nonnegative integers")


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _validate_bootstrap(n_resamples: Any, alpha: Any, seed: Any) -> None:
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int):
        raise TypeError("n_resamples must be an integer")
    if not 1 <= n_resamples <= 100_000:
        raise ValueError("n_resamples must be between one and 100000")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be numeric")
    if not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")


__all__ = [
    "APPROVED_EVIDENCE_DISCLAIMER",
    "ApprovedEvidence",
    "CLAIM_STATES",
    "CONTRADICTED",
    "ClaimAssessment",
    "ClaimClassMetrics",
    "ClaimState",
    "ClaimStatus",
    "DEFAULT_BOOTSTRAP_ALPHA",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "EvidenceRecord",
    "SCHEMA_VERSION",
    "SUPPORTED",
    "SUMMARY_UNSUPPORTED_CLAIMS",
    "SUMMARY_UNSUPPORTED_CLAIMS_SCHEMA_VERSION",
    "SummaryClaim",
    "SummaryUnsupportedClaimReport",
    "SummaryUnsupportedClaimsReport",
    "UNCITED",
    "UNRESOLVED",
    "UNSUPPORTED_CLAIMS_DISCLAIMER",
    "UNSUPPORTED_STATES",
    "build_summary_unsupported_claims_report",
    "compute_summary_unsupported_claims",
    "compute_unsupported_claim_rate",
    "evaluate_summary_unsupported_claims",
    "run_summary_unsupported_claims",
    "score_summary_claim",
    "score_summary_claims",
    "unsupported_claim_rate",
]
