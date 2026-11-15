"""Local extraction-to-context adapter; review authority remains application-owned.

Only the existing synthetic EvidencePacket is admitted. No receipt, review
transition, source approval, calibration or clinical axis is manufactured here.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Protocol

from openmed.clinical.brief import (
    BriefContext,
    BriefFact,
    _digest,
    brief_policy_fingerprint,
)
from openmed.clinical.context import (
    CERTAINTY_VALUES,
    NEGATION_VALUES,
    TEMPORALITY_VALUES,
)
from openmed.clinical.evidence_packet import EvidencePacket, validate_evidence_packet
from openmed.clinical.journey_contracts import (
    ClinicalFact,
    ConflictSet,
    EvidenceLocator,
)
from openmed.clinical.nli_gate import NLIThresholds
from openmed.clinical.sections import detect_sections
from openmed.clinical.summary_envelope import verify_deidentified_artifact
from openmed.clinical.summary_profiles import get_summary_profile
from openmed.core.pii import DeidentificationResult


class BriefContextCode(str, Enum):
    """Controlled context construction outcomes; no rejected values are exposed."""

    READY = "ready"
    MISSING_FACTS = "missing_facts"
    MISSING_MAPPING = "missing_mapping"
    CONFLICTED_FACTS = "conflicted_facts"
    UNSUPPORTED_MAPPING = "unsupported_mapping"
    UNKNOWN_FACT = "unknown_fact"
    MISSING_SOURCE = "missing_source"
    AMBIGUOUS_SOURCE = "ambiguous_source"
    INVALID_OFFSETS = "invalid_offsets"
    REVIEW_REQUIRED = "review_required"
    EVIDENCE_CHANGED = "evidence_changed"
    INVALID_EVIDENCE = "invalid_evidence"
    PROVIDER_UNAVAILABLE = "provider_unavailable"


@dataclass(frozen=True, repr=False)
class BriefFactMapping:
    """Caller-declared profile field and temporal axis for one Journey fact.

    Assertion, certainty and experiencer come from normalized fact attributes.
    Temporality cannot be inferred from lifecycle status or effective time.
    These declarations are included in the receipt binding before verification.
    """

    fact_id: str
    profile_field: str
    temporality: str


@dataclass(frozen=True, repr=False)
class BriefExtraction:
    """Protected local extraction snapshot, with offsets in de-identified text.

    The resolver must authorize access and return a complete selected fact set
    and its conflicts. Original-text or transformed offsets require a separate
    upstream remapping adapter and renewed review; they are not guessed here.
    """

    artifact: DeidentificationResult
    artifact_id: str
    source_id: str
    facts: tuple[ClinicalFact, ...]
    locators: tuple[EvidenceLocator, ...]
    mappings: tuple[BriefFactMapping, ...]
    conflicts: tuple[ConflictSet, ...] = ()
    synthetic: bool = False


@dataclass(frozen=True)
class BriefContextBinding:
    """Value-free receipt identities, including full calibration policy."""

    source_digest: str
    content_digest: str
    extraction_digest: str
    profile_digest: str
    calibration_digest: str
    policy_fingerprint: str

    @property
    def digest(self) -> str:
        """Return the domain-separated identity to verify in a review store."""
        return _digest({"kind": "brief_context_binding_v1", **asdict(self)})


@dataclass(frozen=True)
class BriefContextSpan:
    """Planned synthetic reference using digested identities and exact offsets."""

    reference_id: str
    source_id: str
    start: int
    end: int


@dataclass(frozen=True, repr=False)
class BriefContextPlan:
    """Unapproved plan for an independent local review verifier."""

    binding: BriefContextBinding
    facts: tuple[BriefFact, ...]
    spans: tuple[BriefContextSpan, ...]


class BriefExtractionProvider(Protocol):
    """Application-owned access-controlled extraction lookup."""

    def resolve(self, text: str, review_id: str) -> BriefExtraction | None:
        """Resolve a current protected snapshot without logging source values."""
        ...


class BriefReviewVerifier(Protocol):
    """Independent verification of existing explicit review receipts."""

    def verify(
        self, review_id: str, plan: BriefContextPlan
    ) -> tuple[str, EvidencePacket] | None:
        """Return stored binding digest and approved packet, or no approval.

        Verify current authorization and receipt validity/revocation locally.
        Never approve because a plan was submitted. Return the digest recorded
        at review time, not a newly calculated replacement for an old receipt.
        """
        ...


@dataclass(frozen=True, repr=False)
class BriefContextOutcome:
    """Typed outcome with protected context available only on success."""

    code: BriefContextCode
    artifact: DeidentificationResult | None = field(default=None, repr=False)
    context: BriefContext | None = field(default=None, repr=False)
    binding: BriefContextBinding | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize controlled status and digests only."""
        return {
            "code": self.code.value,
            "binding_digest": self.binding.digest if self.binding else None,
        }


class BriefContextError(ValueError):
    """Value-free provider error for existing transport injection seams."""

    def __init__(self, code: BriefContextCode):
        self.code = code
        super().__init__(code.value)


def plan_brief_context(
    extraction: BriefExtraction, *, profile: str, thresholds: NLIThresholds
) -> BriefContextPlan:
    """Map a local snapshot without approving it or loading any model.

    Args:
        extraction: Current selected Journey facts and de-identified offsets.
        profile: Built-in summary profile name.
        thresholds: Explicit calibrated NLI policy to bind to review.

    Returns:
        An unapproved, value-free plan for the independent verifier.

    Raises:
        BriefContextError: A controlled refusal; never contains source values.
    """
    failed = None
    try:
        return _plan(extraction, profile, thresholds)
    except BriefContextError as error:
        failed = error.code
    except Exception:
        failed = BriefContextCode.UNSUPPORTED_MAPPING
    # Raise outside handlers to avoid chaining protected third-party exceptions.
    raise BriefContextError(failed)


def _plan(extraction, profile_name, thresholds):
    if type(extraction) is not BriefExtraction:
        raise BriefContextError(BriefContextCode.MISSING_SOURCE)
    if extraction.synthetic is not True:
        raise BriefContextError(BriefContextCode.UNSUPPORTED_MAPPING)
    if any(
        type(identity) is not str or not identity
        for identity in (extraction.artifact_id, extraction.source_id)
    ):
        raise BriefContextError(BriefContextCode.MISSING_SOURCE)
    artifact = extraction.artifact
    if type(artifact) is not DeidentificationResult:
        raise BriefContextError(BriefContextCode.MISSING_SOURCE)
    verify_deidentified_artifact(artifact)
    text = artifact.deidentified_text
    if not text or len(text.encode("utf-8")) > 16384:
        raise BriefContextError(BriefContextCode.INVALID_OFFSETS)
    profile = get_summary_profile(profile_name)
    if type(thresholds) is not NLIThresholds:
        raise BriefContextError(BriefContextCode.UNSUPPORTED_MAPPING)
    if not extraction.facts:
        raise BriefContextError(BriefContextCode.MISSING_FACTS)
    if (
        any(
            type(items) is not tuple
            for items in (
                extraction.facts,
                extraction.locators,
                extraction.mappings,
                extraction.conflicts,
            )
        )
        or len(extraction.facts) > 64
        or len(extraction.locators) > 64
        or len(extraction.mappings) > 64
        or len(extraction.conflicts) > 64
    ):
        raise BriefContextError(BriefContextCode.UNSUPPORTED_MAPPING)
    for items, expected in (
        (extraction.facts, ClinicalFact),
        (extraction.locators, EvidenceLocator),
        (extraction.mappings, BriefFactMapping),
        (extraction.conflicts, ConflictSet),
    ):
        if any(type(item) is not expected for item in items):
            raise BriefContextError(BriefContextCode.UNSUPPORTED_MAPPING)
        for item in items:
            replace(item)  # Revalidate even frozen records at the trust boundary.
    facts = {f.fact_id: f for f in extraction.facts}
    mappings = {m.fact_id: m for m in extraction.mappings}
    locators = {loc.locator_id: loc for loc in extraction.locators}
    if len(facts) != len(extraction.facts):
        raise BriefContextError(BriefContextCode.CONFLICTED_FACTS)
    if len(mappings) != len(extraction.mappings) or set(mappings) != set(facts):
        raise BriefContextError(BriefContextCode.MISSING_MAPPING)
    if len(locators) != len(extraction.locators):
        raise BriefContextError(BriefContextCode.AMBIGUOUS_SOURCE)
    if len({(f.subject_id, f.encounter_id) for f in facts.values()}) != 1:
        raise BriefContextError(BriefContextCode.AMBIGUOUS_SOURCE)
    if any(
        c.status == "open" and set(c.fact_ids).intersection(facts)
        for c in extraction.conflicts
    ):
        raise BriefContextError(BriefContextCode.CONFLICTED_FACTS)
    sections = detect_sections(text)
    source_id = "synthetic:" + _digest(extraction.source_id)[7:]
    rows = []
    used = set()
    for fact in facts.values():
        mapping = mappings[fact.fact_id]
        attrs = fact.attributes
        states = attrs.get("field_states", {})
        if (
            fact.value is None
            or fact.status == "unknown"
            or any(
                state in {"unknown", "missing", "unsupported", "partial"}
                for state in states.values()
            )
        ):
            raise BriefContextError(BriefContextCode.UNKNOWN_FACT)
        if fact.status in {"conflicted", "conflict"} or "conflict" in states.values():
            raise BriefContextError(BriefContextCode.CONFLICTED_FACTS)
        if any(state != "known" for state in states.values()):
            raise BriefContextError(BriefContextCode.UNKNOWN_FACT)
        axes = (
            attrs.get("assertion"),
            attrs.get("certainty"),
            mapping.temporality,
            attrs.get("experiencer"),
        )
        if None in axes or "unknown" in axes:
            raise BriefContextError(BriefContextCode.UNKNOWN_FACT)
        if (
            mapping.profile_field not in profile.field_names
            or axes[0] not in NEGATION_VALUES
            or axes[1] not in CERTAINTY_VALUES
            or axes[2] not in TEMPORALITY_VALUES
            or axes[3] not in {"patient", "family", "other"}
            or attrs.get("demographic_class") is not None
        ):
            raise BriefContextError(BriefContextCode.UNSUPPORTED_MAPPING)
        if len(fact.evidence_ids) != 1:
            raise BriefContextError(BriefContextCode.AMBIGUOUS_SOURCE)
        locator = locators.get(fact.evidence_ids[0])
        if locator is None or locator.artifact_id != extraction.artifact_id:
            raise BriefContextError(BriefContextCode.MISSING_SOURCE)
        if locator.location_type != "text_span" or locator.transform:
            raise BriefContextError(BriefContextCode.UNSUPPORTED_MAPPING)
        start, end = locator.location["start"], locator.location["end"]
        if not 0 <= start < end <= len(text):
            raise BriefContextError(BriefContextCode.INVALID_OFFSETS)
        if sum(s["start"] <= start < end <= s["end"] for s in sections) != 1:
            raise BriefContextError(BriefContextCode.AMBIGUOUS_SOURCE)
        if locator.locator_id in used:
            raise BriefContextError(BriefContextCode.AMBIGUOUS_SOURCE)
        used.add(locator.locator_id)
        ref = "synthetic:" + _digest(locator.locator_id)[7:]
        rows.append(
            (
                BriefContextSpan(ref, source_id, start, end),
                BriefFact(ref, mapping.profile_field, *axes),
            )
        )
    rows.sort(key=lambda row: (row[0].start, row[0].end, row[0].reference_id))
    if any(left[0].end > right[0].start for left, right in zip(rows, rows[1:])):
        raise BriefContextError(BriefContextCode.AMBIGUOUS_SOURCE)
    if len({text[row[0].start : row[0].end] for row in rows}) != len(rows):
        raise BriefContextError(BriefContextCode.AMBIGUOUS_SOURCE)
    brief_facts = tuple(row[1] for row in rows)
    binding = BriefContextBinding(
        source_digest=_digest(
            {
                "original": artifact.original_text,
                "artifact_id": extraction.artifact_id,
                "source_id": extraction.source_id,
                "method": artifact.method,
                "mapping": artifact.mapping,
                "entities": [asdict(e) for e in artifact.pii_entities],
                "synthetic": extraction.synthetic,
            }
        ),
        content_digest=_digest(text),
        extraction_digest=_digest(
            {
                "facts": [facts[key].to_dict() for key in sorted(facts)],
                "locators": [locators[key].to_dict() for key in sorted(locators)],
                "mappings": [asdict(mappings[key]) for key in sorted(mappings)],
                "conflicts": sorted(
                    (c.to_dict() for c in extraction.conflicts),
                    key=lambda c: c["conflict_id"],
                ),
            }
        ),
        profile_digest=profile.digest,
        calibration_digest=_digest(thresholds.to_dict()),
        policy_fingerprint=brief_policy_fingerprint(text, brief_facts, profile_name),
    )
    return BriefContextPlan(binding, brief_facts, tuple(row[0] for row in rows))


class LocalBriefContextProvider:
    """Inject a local extraction resolver and independent review verifier.

    This Python adapter fits the existing Python, CLI, REST and MCP provider
    seams. A configured profile/calibration is fixed per provider instance.
    The existing composer also checks the requested profile before generation.
    """

    def __init__(
        self,
        *,
        extraction_provider: BriefExtractionProvider,
        review_verifier: BriefReviewVerifier,
        thresholds: NLIThresholds,
        nli_predict: Callable[[str, str], Any],
        privacy_detector: Callable[[str], Any],
        profile: str = "bhc",
    ):
        """Configure trusted dependencies without loading a model.

        Args:
            extraction_provider: Authorized current-snapshot resolver.
            review_verifier: Independent existing-receipt verifier.
            thresholds: Explicit calibrated NLI policy recorded in the receipt.
            nli_predict: Local calibrated callback used by the composer.
            privacy_detector: Local privacy callback used by the composer.
            profile: Built-in summary profile bound to the receipt.
        """
        self._extraction_provider = extraction_provider
        self._review_verifier = review_verifier
        self._thresholds = thresholds
        self._nli_predict = nli_predict
        self._privacy_detector = privacy_detector
        self._profile = profile

    def build(self, text: str, review_id: str) -> BriefContextOutcome:
        """Return a typed result without exposing source/provider exception text."""
        from openmed.core.offline import network_blocked_if_offline

        try:
            with network_blocked_if_offline(local_only=True):
                return self._build(text, review_id)
        except BriefContextError as error:
            return BriefContextOutcome(error.code)
        except Exception:
            return BriefContextOutcome(BriefContextCode.PROVIDER_UNAVAILABLE)

    def _build(self, text, review_id):
        if (
            type(text) is not str
            or not text
            or len(text.encode("utf-8")) > 16384
            or type(review_id) is not str
            or not re.fullmatch(r"[a-f0-9]{64}", review_id)
        ):
            return BriefContextOutcome(BriefContextCode.INVALID_EVIDENCE)
        extraction = self._extraction_provider.resolve(text, review_id)
        if extraction is None:
            return BriefContextOutcome(BriefContextCode.MISSING_SOURCE)
        if extraction.artifact.original_text != text:
            return BriefContextOutcome(BriefContextCode.EVIDENCE_CHANGED)
        if not callable(self._nli_predict) or not callable(self._privacy_detector):
            return BriefContextOutcome(BriefContextCode.PROVIDER_UNAVAILABLE)
        plan = plan_brief_context(
            extraction, profile=self._profile, thresholds=self._thresholds
        )
        verification = self._review_verifier.verify(review_id, plan)
        if verification is None:
            return BriefContextOutcome(BriefContextCode.REVIEW_REQUIRED)
        digest, packet = verification
        if digest != plan.binding.digest:
            return BriefContextOutcome(BriefContextCode.EVIDENCE_CHANGED)
        # The artifact is mutable. An injected verifier must not change reviewed
        # evidence between planning and context construction.
        if (
            plan_brief_context(
                extraction, profile=self._profile, thresholds=self._thresholds
            )
            != plan
        ):
            return BriefContextOutcome(BriefContextCode.EVIDENCE_CHANGED)
        try:
            packet = validate_evidence_packet(packet)
        except Exception:
            return BriefContextOutcome(BriefContextCode.INVALID_EVIDENCE)
        if (
            packet.policy_fingerprint != plan.binding.policy_fingerprint
            or tuple(
                BriefContextSpan(r.reference_id, r.source_id, r.start, r.end)
                for r in packet.references
            )
            != plan.spans
            or packet.rejection_report.rejected_count
        ):
            return BriefContextOutcome(BriefContextCode.INVALID_EVIDENCE)
        context = BriefContext(
            packet,
            plan.binding.content_digest,
            plan.facts,
            self._nli_predict,
            self._thresholds,
            self._privacy_detector,
        )
        return BriefContextOutcome(
            BriefContextCode.READY, extraction.artifact, context, plan.binding
        )

    def __call__(
        self, text: str, review_id: str
    ) -> tuple[DeidentificationResult, BriefContext]:
        """Fit existing transport seams; typed diagnostics use ``build`` instead."""
        outcome = self.build(text, review_id)
        if outcome.code is not BriefContextCode.READY:
            raise BriefContextError(outcome.code)
        assert outcome.artifact is not None and outcome.context is not None
        return outcome.artifact, outcome.context
