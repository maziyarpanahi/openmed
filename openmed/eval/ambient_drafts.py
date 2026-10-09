"""Offline evaluation of annotated, fixed ambient drafts against encounter truth.

No NLP inference or note assembly is performed. Proposition digests and citations
are caller annotations, not proof of semantic equivalence. Review imports are
ambient-specific and do not extend the summary adjudication/release gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from openmed.eval.reviewer_disagreement import (
    DisagreementReason,
    ReviewerDecision,
    reviewer_disagreement_report,
)
from openmed.eval.summary_coverage import compute_summary_fact_coverage
from openmed.eval.summary_unsupported_claims import (
    CONTRADICTED,
    SUPPORTED,
    UNRESOLVED,
    ApprovedEvidence,
    SummaryClaim,
    score_summary_claims,
)

SCHEMA_VERSION = "openmed.ambient_drafts.v1"
REVIEW_SCHEMA_VERSION = "openmed.ambient_draft_reviews.v1"
NOTICE = "Evaluation only; non-diagnostic. Explicit clinician review is required."
_SECTIONS = ("subjective", "objective", "assessment", "plan", "other")
_ROLES = ("patient", "clinician", "caregiver")
_CLASSES = (
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
)
_METRICS = ("omission", "unsupported", "contradiction", "misattribution", "negation")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require_digest(value: Any) -> None:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError("invalid_opaque_identifier")


def _require_code(value: Any, choices: tuple[str, ...]) -> None:
    if type(value) is not str or value not in choices:
        raise ValueError("invalid_controlled_code")


@dataclass(frozen=True, slots=True, repr=False)
class AmbientFact:
    """One annotated truth fact or atomic draft statement.

    Args:
        fact_id: Random opaque 64-hex identifier, unique within its collection.
        proposition: Digest of a caller-defined proposition excluding polarity,
            speaker and experiencer. Matching is exact, never inferred.
        section: Controlled section code.
        fact_class: Controlled clinical class code.
        speaker: Who supplied the fact: patient, clinician or caregiver.
        experiencer: Whose clinical state the fact describes, independently of
            who spoke. Uses the same controlled role codes.
        negated: Whether the proposition is explicitly negated.
        required: Whether a truth fact must appear in the draft. Draft statements
            must use the default True; optional truth is still valid evidence.
        citations: Truth fact identifiers for a draft statement; empty for truth.
    """

    fact_id: str
    proposition: str
    section: str
    fact_class: str
    speaker: str
    experiencer: str
    negated: bool = False
    required: bool = True
    citations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_digest(self.fact_id)
        _require_digest(self.proposition)
        _require_code(self.section, _SECTIONS)
        _require_code(self.fact_class, _CLASSES)
        _require_code(self.speaker, _ROLES)
        _require_code(self.experiencer, _ROLES)
        if type(self.negated) is not bool or type(self.required) is not bool:
            raise ValueError("invalid_boolean")
        if type(self.citations) is not tuple or len(self.citations) > 64:
            raise ValueError("invalid_citations")
        for citation in self.citations:
            _require_digest(citation)
        if len(set(self.citations)) != len(self.citations):
            raise ValueError("duplicate_citation")


def _facts(value: Any) -> tuple[AmbientFact, ...]:
    if type(value) is not tuple or len(value) > 10_000:
        raise ValueError("invalid_fact_collection")
    if any(not isinstance(row, AmbientFact) for row in value):
        raise ValueError("invalid_fact_record")
    if len({row.fact_id for row in value}) != len(value):
        raise ValueError("duplicate_fact_identifier")
    return value


@dataclass(frozen=True, slots=True, repr=False)
class AmbientDraft:
    """Fixed encounter truth and draft annotations, bound to a text digest.

    Args:
        blinded_case_id: Random review alias with no candidate or patient identity.
        text_digest: SHA-256 of the exact draft text reviewed, computed locally.
        truth: Nonempty caller-annotated encounter truth; no draft citations.
        statements: Atomic draft statements, including unsupported statements.

    No transcript or draft text is retained. The caller must include every atomic
    statement and distribute the exact text whose digest is supplied here.
    """

    blinded_case_id: str
    text_digest: str
    truth: tuple[AmbientFact, ...]
    statements: tuple[AmbientFact, ...]

    def __post_init__(self) -> None:
        _require_digest(self.blinded_case_id)
        _require_digest(self.text_digest)
        _facts(self.truth)
        _facts(self.statements)
        if not self.truth or any(row.citations for row in self.truth):
            raise ValueError("invalid_encounter_truth")
        if any(not row.required for row in self.statements):
            raise ValueError("invalid_draft_statement")

    @property
    def revision(self) -> str:
        """Return a binding to text, truth, annotations and evaluator version."""
        return _digest({"schema": SCHEMA_VERSION, **asdict(self)})


def _rate(numerator: int, denominator: int, minimum: int) -> dict[str, Any]:
    # Complementary suppression: no total, point estimate or interval can reveal
    # a small nonzero numerator or its complement by subtraction.
    suppressed = denominator < minimum or any(
        0 < count < minimum for count in (numerator, denominator - numerator)
    )
    if suppressed:
        return dict(
            numerator=None, denominator=None, rate=None, ci95=None, suppressed=True
        )
    point = numerator / denominator
    z = 1.959963984540054
    scale = 1 + z * z / denominator
    center = (point + z * z / (2 * denominator)) / scale
    radius = (
        z
        * math.sqrt(point * (1 - point) / denominator + z * z / (4 * denominator**2))
        / scale
    )
    return dict(
        numerator=numerator,
        denominator=denominator,
        rate=point,
        ci95=[max(0.0, center - radius), min(1.0, center + radius)],
        suppressed=False,
    )


def _minimum(value: Any) -> int:
    if type(value) is not int or value < 2:
        raise ValueError("invalid_minimum_cell_size")
    return value


def _slice_codes(fact: AmbientFact) -> tuple[str, ...]:
    return (
        "overall",
        f"section:{fact.section}",
        f"class:{fact.fact_class}",
        f"speaker:{fact.speaker}",
        f"experiencer:{fact.experiencer}",
    )


def evaluate_ambient_drafts(
    drafts: Iterable[AmbientDraft],
    *,
    minimum_cell_size: int = 5,
) -> dict[str, Any]:
    """Score independent error rates over fixed, annotated drafts locally.

    Coverage only credits correctly supported statements in the truth section.
    Contradiction includes polarity flips; misattribution separately counts wrong
    speakers or experiencers even with complete proposition/citation coverage.
    Known cited statements form attribution/negation denominators. All statements
    form unsupported/contradiction denominators; required truth forms omissions.

    Reports use fixed section, class and role slices. A small cell suppresses its
    entire metric family (including overall) to prevent subtraction across slices.
    Intervals are descriptive 95% Wilson intervals over atomic observations, not
    independent-encounter uncertainty or clinical validation. No quality score
    combines the rates. Consequential consumers must obtain explicit review.

    Args:
        drafts: Fixed drafts, with one revision per blinded case.
        minimum_cell_size: Minimum publishable nonzero count, at least two.

    Returns:
        Aggregate counts/rates/intervals and controlled safety codes only.
    """
    minimum = _minimum(minimum_cell_size)
    observations: dict[str, dict[str, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen: set[str] = set()
    for draft in drafts:
        if not isinstance(draft, AmbientDraft):
            raise ValueError("invalid_draft_record")
        if draft.blinded_case_id in seen:
            raise ValueError("duplicate_encounter")
        seen.add(draft.blinded_case_id)
        truth = {fact.fact_id: fact for fact in draft.truth}
        claims: list[SummaryClaim] = []
        evidence: list[ApprovedEvidence] = []
        valid_citations: list[str] = []
        for statement in draft.statements:
            cited = [truth[c] for c in statement.citations if c in truth]
            misattributed = any(
                (row.speaker, row.experiencer)
                != (statement.speaker, statement.experiencer)
                for row in cited
            )
            negation = any(row.negated != statement.negated for row in cited)
            relations = []
            for index, citation in enumerate(statement.citations):
                source = truth.get(citation)
                if source is None or (source.proposition, source.fact_class) != (
                    statement.proposition,
                    statement.fact_class,
                ):
                    relation = UNRESOLVED
                elif (source.negated, source.speaker, source.experiencer) != (
                    statement.negated,
                    statement.speaker,
                    statement.experiencer,
                ):
                    relation = CONTRADICTED
                else:
                    relation = SUPPORTED
                relations.append(relation)
                evidence.append(
                    ApprovedEvidence(
                        f"{statement.fact_id}:{index}",
                        relation,
                        claim_id=statement.fact_id,
                    )
                )
            claims.append(
                SummaryClaim(
                    statement.fact_id,
                    statement.fact_class,
                    evidence_ids=tuple(
                        f"{statement.fact_id}:{i}" for i in range(len(relations))
                    ),
                )
            )
            if relations and set(relations) == {SUPPORTED}:
                valid_citations.extend(
                    row.fact_id for row in cited if row.section == statement.section
                )
            if cited:
                for code in _slice_codes(statement):
                    observations[code]["misattribution"].append(misattributed)
                    observations[code]["negation"].append(negation)
        # Existing four-state metric supplies the unsupported/contradiction counts;
        # its unsuppressed reports are never exposed at this publication boundary.
        scored = score_summary_claims(claims, evidence, n_resamples=1)
        for statement, assessment in zip(draft.statements, scored.assessments):
            for code in _slice_codes(statement):
                observations[code]["unsupported"].append(assessment.unsupported)
                observations[code]["contradiction"].append(
                    assessment.state == CONTRADICTED
                )
        required = [row for row in draft.truth if row.required]
        for code in {c for row in required for c in _slice_codes(row)}:
            subset = [row.fact_id for row in required if code in _slice_codes(row)]
            covered = set(valid_citations).intersection(subset)
            coverage = compute_summary_fact_coverage(subset, sorted(covered))
            observations[code]["omission"].extend(
                [True] * coverage.omission_count + [False] * coverage.cited_fact_count
            )
    if not seen:
        raise ValueError("empty_encounter_collection")
    codes = (
        "overall",
        *(f"section:{c}" for c in _SECTIONS),
        *(f"class:{c}" for c in _CLASSES),
        *(f"speaker:{c}" for c in _ROLES),
        *(f"experiencer:{c}" for c in _ROLES),
    )
    slices = {
        code: {
            metric: _rate(
                sum(observations[code][metric]),
                len(observations[code][metric]),
                minimum,
            )
            for metric in _METRICS
        }
        for code in codes
    }
    for metric in _METRICS:
        if any(
            observations[code][metric] and slices[code][metric]["suppressed"]
            for code in codes
        ):
            for code in codes:
                slices[code][metric] = _rate(0, 0, minimum)
    return {
        "schema_version": SCHEMA_VERSION,
        "minimum_cell_size": minimum,
        "notice": NOTICE,
        "reviewer_confirmation_required": True,
        "slices": slices,
    }


def import_ambient_reviews(
    payload: Mapping[str, Any],
    current_drafts: Iterable[AmbientDraft],
    *,
    minimum_cell_size: int = 5,
) -> dict[str, Any]:
    """Import strict blinded review returns bound to current ambient revisions.

    Each row has exactly ``blinded_case_id``, ``draft_revision``, ``reviewer_id``,
    ``decision`` (accept/revise/unclear), ``reason`` and ``adjudicated``. Reasons
    are controlled DisagreementReason values or null. Every current case needs
    two distinct reviewers; disagreement needs a consistent reason and status.
    No identity, candidate name, source payload or free text is accepted.

    The caller separately retains the blinded alias mapping and verifies clinician
    credentials and adjudication provenance. Imports assert decisions only; even
    unanimous accept never grants clinical/export permission. Disagreement remains
    visible after adjudication. Synthetic imports are not clinical validation.

    Args:
        payload: Mapping with exactly schema_version and rows (a list).
        current_drafts: Exact current snapshots presented to blinded reviewers.
        minimum_cell_size: Minimum publishable nonzero aggregate count.

    Returns:
        Suppressed aggregate agreement, adjudication and unclear-review rates
        with confidence intervals; no case, reviewer or decision values.

    Raises:
        ValueError: For unblinded, stale, incomplete or malformed imports.
    """
    minimum = _minimum(minimum_cell_size)
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"schema_version", "rows"}
        or payload["schema_version"] != REVIEW_SCHEMA_VERSION
        or type(payload["rows"]) is not list
        or len(payload["rows"]) > 100_000
    ):
        raise ValueError("invalid_review_schema")
    current: dict[str, AmbientDraft] = {}
    for draft in current_drafts:
        if not isinstance(draft, AmbientDraft) or draft.blinded_case_id in current:
            raise ValueError("invalid_current_drafts")
        current[draft.blinded_case_id] = draft
    decisions = []
    unclear = []
    keys = {
        "blinded_case_id",
        "draft_revision",
        "reviewer_id",
        "decision",
        "reason",
        "adjudicated",
    }
    for row in payload["rows"]:
        if not isinstance(row, Mapping) or set(row) != keys:
            raise ValueError("unblinded_or_invalid_review")
        for key in ("blinded_case_id", "draft_revision", "reviewer_id"):
            _require_digest(row[key])
        snapshot = current.get(row["blinded_case_id"])
        if snapshot is None or row["draft_revision"] != snapshot.revision:
            raise ValueError("stale_or_unknown_review")
        _require_code(row["decision"], ("accept", "revise", "unclear"))
        if type(row["adjudicated"]) is not bool:
            raise ValueError("invalid_review_status")
        reason = row["reason"]
        if reason is not None:
            _require_code(reason, tuple(item.value for item in DisagreementReason))
            reason = DisagreementReason(reason)
        decisions.append(
            ReviewerDecision(
                row["blinded_case_id"],
                row["reviewer_id"],
                row["decision"],
                reason,
                row["adjudicated"],
            )
        )
        unclear.append(row["decision"] == "unclear")
    if not current or {row.case_id for row in decisions} != set(current):
        raise ValueError("incomplete_review_import")
    report = reviewer_disagreement_report(decisions, minimum_cell_size=minimum)
    rates = {"agreement": report.agreement, "adjudication": report.adjudication}
    result = {
        name: (
            _rate(rate.numerator, rate.denominator, minimum)
            if rate.numerator is not None and rate.denominator is not None
            else _rate(0, 0, minimum)
        )
        for name, rate in rates.items()
    }
    result["unclear"] = _rate(sum(unclear), len(unclear), minimum)
    # Do not expose reason partitions alongside totals: their complements could
    # reconstruct suppressed disagreement cells. Existing grouping validates them.
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "minimum_cell_size": minimum,
        "notice": NOTICE,
        "reviewer_confirmation_required": True,
        "rates": result,
    }


__all__ = [
    "AmbientDraft",
    "AmbientFact",
    "evaluate_ambient_drafts",
    "import_ambient_reviews",
]
