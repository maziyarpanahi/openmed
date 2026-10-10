"""Fixed-order clinical briefs over explicitly reviewed local evidence.

No review is minted by this module. Raw input without an approved evidence
context is de-identified, then refused. The initial composer deliberately accepts
only exact, atomic source claims; paraphrases need a future alignment adapter.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable

from openmed.clinical.evidence_packet import EvidencePacket, validate_evidence_packet
from openmed.clinical.nli_gate import NLIThresholds
from openmed.core.pii import DeidentificationResult

STAGES = (
    "deidentification",
    "sections",
    "evidence",
    "summary_input",
    "section_plan",
    "length_budget",
    "profile",
    "generation",
    "envelope",
    "demographics",
    "claims",
    "assertion",
    "temporality",
    "experiencer",
    "nli",
    "citations",
    "boundaries",
    "minimality",
    "temporal_order",
    "unsupported_claims",
    "coverage",
    "citation_support",
    "empty_evidence",
    "privacy",
    "review",
    "provenance",
)


class BriefRefusal(str, Enum):
    """Stable, value-free failure categories."""

    INVALID_INPUT = "invalid_input"
    REVIEW_REQUIRED = "review_required"
    EMPTY_EVIDENCE = "empty_evidence"
    INVALID_EVIDENCE = "invalid_evidence"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    NLI_UNAVAILABLE = "nli_unavailable"
    NLI_REJECTED = "nli_rejected"
    PRIVACY = "privacy"
    STAGE_FAILED = "stage_failed"


def _digest(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode()
        ).hexdigest()
    )


@dataclass(frozen=True)
class BriefFact:
    """Reviewed clinical axes and profile field for one packet reference.

    Clinical values remain in the de-identified artifact, not in this record.
    Axis metadata must come from upstream extraction and review, not generation.
    """

    reference_id: str
    profile_field: str
    negation: str
    certainty: str
    temporality: str
    experiencer: str
    demographic_class: str | None = None


def brief_policy_fingerprint(
    text: str, facts: tuple[BriefFact, ...], profile: str = "bhc"
) -> str:
    """Bind evidence review to content, reviewed axes, and versioned profile."""
    from openmed.clinical.summary_profiles import get_summary_profile

    return _digest(
        {
            "content": _digest(text),
            "facts": [asdict(f) for f in facts],
            "profile": get_summary_profile(profile).digest,
        }
    )


@dataclass(frozen=True, repr=False)
class BriefContext:
    """Explicit reviewed evidence and calibrated local NLI integration.

    The existing packet currently admits synthetic evidence only. This composer
    preserves that restriction. A callback must return calibrated three-class
    probabilities bound to ``thresholds.calibration_id``; there is no heuristic
    fallback and no automatic calibration or review approval.
    """

    packet: EvidencePacket
    content_digest: str
    facts: tuple[BriefFact, ...]
    nli_predict: Callable[[str, str], Any] = field(repr=False)
    thresholds: NLIThresholds
    privacy_detector: Callable[[str], Any] = field(repr=False)


@dataclass(frozen=True)
class ClinicalBrief:
    """Immutable result; safe serialization never includes generated text."""

    summary: str = field(repr=False)
    refusal_reason: BriefRefusal | None
    _audit_json: str = field(repr=False)

    @property
    def digest(self) -> str:
        """Return the canonical value-free packet digest."""
        return _digest(json.loads(self._audit_json))

    @property
    def citations(self) -> tuple[dict[str, Any], ...]:
        """Return defensive copies of ordered source/claim offsets."""
        return tuple(self.to_dict()["citations"])

    @property
    def verdicts(self) -> tuple[dict[str, Any], ...]:
        """Return defensive copies of per-claim verdicts."""
        return tuple(self.to_dict()["verdicts"])

    @property
    def metrics(self) -> dict[str, Any]:
        """Return aggregate-only metrics."""
        return self.to_dict()["metrics"]

    @property
    def envelope(self) -> dict[str, Any]:
        """Return the value-free safety envelope."""
        return self.to_dict()["envelope"]

    @property
    def review_packet(self) -> dict[str, Any]:
        """Return the value-free human review handoff."""
        return self.to_dict()["review_packet"]

    def to_dict(self) -> dict[str, Any]:
        """Serialize counts, offsets, hashes and fixed reasons only."""
        return {**json.loads(self._audit_json), "digest": self.digest}

    def to_response(self) -> dict[str, Any]:
        """Explicit protected response for an authorized caller, never logs."""
        return {**self.to_dict(), "summary": self.summary}


class _Stop(Exception):
    def __init__(self, reason: BriefRefusal):
        self.reason = reason


def build_clinical_brief(
    note_or_deid_result: str | DeidentificationResult,
    *,
    model: object = None,
    profile: str = "bhc",
    context: BriefContext | None = None,
) -> ClinicalBrief:
    """Compose the fixed guarded pipeline; never approve unreviewed evidence.

    Args:
        note_or_deid_result: Raw note or completed local de-identification result.
        model: Registered local summarizer alias or trusted local callback.
        profile: Built-in versioned summary profile.
        context: Explicit reviewed evidence and calibrated local NLI callback.

    Returns:
        A review-required brief, or a typed refusal with no source values.
        Missing stages, models, evidence or failed checks always fail closed.
    """
    from openmed.clinical.review_packet_privacy import ReviewPacketPrivacyBlocked
    from openmed.clinical.summarize import SummarizationLeakageError
    from openmed.core.offline import network_blocked_if_offline

    completed: list[str] = []
    reason = BriefRefusal.STAGE_FAILED
    try:
        with network_blocked_if_offline(local_only=True):
            return _compose(note_or_deid_result, model, profile, context, completed)
    except _Stop as error:
        reason = error.reason
    except (SummarizationLeakageError, ReviewPacketPrivacyBlocked):
        reason = BriefRefusal.PRIVACY
    except Exception:
        pass
    # Outside the handler: third-party exception contexts can contain source text.
    return _result("", reason, completed)


def _result(summary, reason, completed, **values):
    from openmed.clinical.review_packet import build_review_packet
    from openmed.clinical.summary_envelope import SUMMARY_SAFETY_DISCLAIMER

    audit = {
        "schema_version": 1,
        "status": "refused" if reason else "needs_review",
        "refusal_reason": reason.value if reason else None,
        "summary_digest": _digest(summary),
        "summary_characters": len(summary),
        "stages": list(completed),
        "citations": [],
        "verdicts": [],
        "metrics": {},
        "backend_id": None,
        "profile_digest": None,
        "provenance": {},
        "envelope": {
            "requires_human_review": True,
            "is_diagnostic": False,
            "disclaimer": SUMMARY_SAFETY_DISCLAIMER,
        },
        "review_packet": build_review_packet(
            gates=[
                {
                    "gate_id": "clinical_brief",
                    "passed": reason is None,
                    "blocking": reason is not None,
                    "reason": reason.value if reason else "human_review_required",
                }
            ]
        ).to_dict(),
        **values,
    }
    return ClinicalBrief(
        summary, reason, json.dumps(audit, sort_keys=True, separators=(",", ":"))
    )


def _compose(value, model, profile_name, context, completed):
    from openmed.clinical import summarize_deidentified
    from openmed.clinical.citation_boundaries import (
        build_deidentification_offset_map,
        validate_citation_boundaries,
    )
    from openmed.clinical.citation_minimality import check_citation_minimality
    from openmed.clinical.guarded_provenance import build_guarded_provenance_record
    from openmed.clinical.nli_assertion_pairs import build_nli_pair
    from openmed.clinical.nli_experiencer_pairs import build_experiencer_nli_pair
    from openmed.clinical.nli_gate import EvidenceLink, evaluate_nli
    from openmed.clinical.nli_temporal_pairs import build_temporal_nli_pair
    from openmed.clinical.review_packet import build_review_packet
    from openmed.clinical.review_packet_privacy import enforce_review_packet_privacy
    from openmed.clinical.sections import detect_sections
    from openmed.clinical.summarize import _build_leakage_check
    from openmed.clinical.summarize_backends import resolve_summarizer_backend
    from openmed.clinical.summary_citations import compute_summary_citation_metrics
    from openmed.clinical.summary_claim_segments import segment_summary_claims
    from openmed.clinical.summary_demographic_minimizer import (
        DemographicPurposePolicy,
        minimize_demographic_evidence,
    )
    from openmed.clinical.summary_empty_evidence import require_summary_evidence
    from openmed.clinical.summary_envelope import (
        build_summary_envelope,
        verify_deidentified_artifact,
    )
    from openmed.clinical.summary_input import guard_summary_input
    from openmed.clinical.summary_length_budget import build_summary_length_budget
    from openmed.clinical.summary_profiles import (
        get_summary_profile,
        validate_summary_output,
    )
    from openmed.clinical.summary_section_plan import require_summary_section_plan
    from openmed.clinical.summary_temporal_order import validate_summary_temporal_order
    from openmed.clinical.timeline import order_events
    from openmed.core.config import OpenMedConfig
    from openmed.core.pii import deidentify
    from openmed.eval.citation_support_metrics import compute_citation_support_metrics
    from openmed.eval.summary_coverage import compute_summary_fact_coverage
    from openmed.eval.summary_unsupported_claims import score_summary_claims

    def stage(name):
        if len(completed) >= len(STAGES) or STAGES[len(completed)] != name:
            raise _Stop(BriefRefusal.STAGE_FAILED)
        completed.append(name)

    stage("deidentification")
    backend = resolve_summarizer_backend(model)
    if isinstance(value, str):
        if len(value.encode()) > 16384:
            raise _Stop(BriefRefusal.INVALID_INPUT)
        value = deidentify(value, method="mask", config=OpenMedConfig(local_only=True))
    if (
        type(value) is not DeidentificationResult
        or len(value.deidentified_text.encode()) > 16384
    ):
        raise _Stop(BriefRefusal.INVALID_INPUT)
    artifact = verify_deidentified_artifact(value)
    text = value.deidentified_text
    stage("sections")
    sections = detect_sections(text)
    stage("evidence")
    if context is None:
        raise _Stop(BriefRefusal.REVIEW_REQUIRED)
    if type(context) is not BriefContext or context.content_digest != _digest(text):
        raise _Stop(BriefRefusal.INVALID_EVIDENCE)
    if (
        type(context.facts) is not tuple
        or len(context.facts) > 64
        or type(context.packet) is not EvidencePacket
        or len(context.packet.references) > 64
    ):
        raise _Stop(BriefRefusal.INVALID_EVIDENCE)
    packet = validate_evidence_packet(context.packet)
    if packet.policy_fingerprint != brief_policy_fingerprint(
        text, context.facts, profile_name
    ):
        raise _Stop(BriefRefusal.INVALID_EVIDENCE)
    refs = tuple(sorted(packet.references, key=lambda r: (r.start, r.end)))
    if not refs:
        raise _Stop(BriefRefusal.EMPTY_EVIDENCE)
    if len(refs) > 64 or len(context.facts) != len(refs):
        raise _Stop(BriefRefusal.INVALID_EVIDENCE)
    facts = {f.reference_id: f for f in context.facts}
    if set(facts) != {r.reference_id for r in refs} or any(
        r.end > len(text) for r in refs
    ):
        raise _Stop(BriefRefusal.INVALID_EVIDENCE)
    stage("summary_input")
    guard_summary_input(
        [
            {
                "evidence_type": "structured_fact",
                "source_ref": {
                    "source_id": r.source_id,
                    "start": r.start,
                    "end": r.end,
                },
                "policy_fingerprint": packet.policy_fingerprint,
                "review_status": "approved",
                "fields": {"category": "finding", "count": 1},
            }
            for r in refs
        ],
        policy_fingerprint=packet.policy_fingerprint,
    )
    rows = [
        {
            "evidence_id": r.reference_id,
            "fact_id": r.reference_id,
            "start": r.start,
            "end": r.end,
            "section_id": _digest(
                next((s for s in sections if s["start"] <= r.start < s["end"]), {})
            ),
            "approved": True,
        }
        for r in refs
    ]
    stage("section_plan")
    require_summary_section_plan(rows)
    stage("length_budget")
    budget = build_summary_length_budget(
        2048, {"key_findings": sum(r.end - r.start for r in refs)}
    )
    if budget.deferred_evidence_classes:
        raise _Stop(BriefRefusal.INVALID_INPUT)
    stage("profile")
    profile = get_summary_profile(profile_name)
    if any(f.profile_field not in profile.field_names for f in facts.values()):
        raise _Stop(BriefRefusal.INVALID_EVIDENCE)
    stage("generation")
    # Refuse before generation if reviewed evidence includes demographic classes.
    # The minimization stage below also records the actual class counts.
    if any(f.demographic_class is not None for f in context.facts):
        raise _Stop(BriefRefusal.INVALID_EVIDENCE)
    # Only reviewed spans enter generation; retain the original result for privacy checks.
    from dataclasses import replace

    admitted = " ".join(text[r.start : r.end] for r in refs)
    generated = summarize_deidentified(
        replace(value, deidentified_text=admitted, audit_report=None), model=backend
    )
    summary = generated.summary
    stage("envelope")
    envelope = build_summary_envelope(
        {"summary_digest": _digest(summary)}, artifact=artifact, human_review_mode=True
    )
    if envelope.status != "ready":
        raise _Stop(BriefRefusal.STAGE_FAILED)
    stage("demographics")
    # This profile admits clinical finding evidence only, never demographic attributes.
    demographics = minimize_demographic_evidence(
        [], DemographicPurposePolicy(_digest(profile.name), frozenset())
    )
    stage("claims")
    segments = segment_summary_claims(summary).segments
    if not segments or any(s.review_required for s in segments):
        raise _Stop(BriefRefusal.UNSUPPORTED_CLAIM)
    aligned = []
    for index, segment in enumerate(segments):
        matches = [r for r in refs if text[r.start : r.end] == segment.text]
        if len(matches) != 1:
            raise _Stop(BriefRefusal.UNSUPPORTED_CLAIM)
        aligned.append(
            ("claim-" + str(index), segment, matches[0], facts[matches[0].reference_id])
        )
    output = {}
    for _, segment, _, fact in aligned:
        spec = next(f for f in profile.fields if f.name == fact.profile_field)
        if spec.repeated:
            output.setdefault(spec.name, []).append(segment.text)
        else:
            output[spec.name] = (output.get(spec.name, "") + " " + segment.text).strip()
    if not validate_summary_output(profile, output).valid:
        raise _Stop(BriefRefusal.UNSUPPORTED_CLAIM)
    stage("assertion")
    for _, s, r, f in aligned:
        axes = {
            "negation": f.negation,
            "certainty": f.certainty,
            "temporality": f.temporality,
        }
        build_nli_pair(
            text[r.start : r.end],
            s.text,
            premise_assertion=axes,
            hypothesis_assertion=axes,
        )
    stage("temporality")
    for _, s, r, f in aligned:
        build_temporal_nli_pair(
            text[r.start : r.end],
            s.text,
            premise_status=f.temporality,
            hypothesis_status=f.temporality,
        )
    stage("experiencer")
    for _, s, r, f in aligned:
        build_experiencer_nli_pair(
            text[r.start : r.end],
            s.text,
            premise_experiencer=f.experiencer,
            hypothesis_experiencer=f.experiencer,
        )
    stage("nli")
    if (
        not callable(context.nli_predict)
        or type(context.thresholds) is not NLIThresholds
    ):
        raise _Stop(BriefRefusal.NLI_UNAVAILABLE)
    verdicts = []
    for claim_id, segment, ref, _ in aligned:
        scores = context.nli_predict(text[ref.start : ref.end], segment.text)
        if (
            not isinstance(scores, dict)
            or scores.get("calibration_id") != context.thresholds.calibration_id
        ):
            raise _Stop(BriefRefusal.NLI_UNAVAILABLE)
        verdict = evaluate_nli(
            scores,
            EvidenceLink.from_text(
                source_id=ref.reference_id,
                claim_id=claim_id,
                source_text=text[ref.start : ref.end],
                claim_text=segment.text,
                start=ref.start,
                end=ref.end,
            ),
            thresholds=context.thresholds,
        )
        if verdict.outcome != "entailment":
            raise _Stop(BriefRefusal.NLI_REJECTED)
        verdicts.append({"claim_index": len(verdicts), "label": verdict.outcome})
    claims = [
        {
            "claim_id": c,
            "claim_class": "finding",
            "evidence_ids": [r.reference_id],
            "citations": [{"evidence_id": r.reference_id}],
        }
        for c, _, r, _ in aligned
    ]
    stage("citations")
    if not compute_summary_citation_metrics(claims, rows).passed:
        raise _Stop(BriefRefusal.UNSUPPORTED_CLAIM)
    stage("boundaries")
    offset_map = build_deidentification_offset_map(value)
    validate_citation_boundaries(
        [
            {
                "start": r.start,
                "end": r.end,
                "document_digest": offset_map.document_digest,
            }
            for _, _, r, _ in aligned
        ],
        offset_map,
    )
    stage("minimality")
    minimality = check_citation_minimality(
        text,
        [
            {"claim_id": _digest(c), "required_span": [r.start, r.end]}
            for c, _, r, _ in aligned
        ],
        [
            {"claim_id": _digest(c), "source_span": [r.start, r.end]}
            for c, _, r, _ in aligned
        ],
    )
    if any(r.review_required for r in minimality.records):
        raise _Stop(BriefRefusal.UNSUPPORTED_CLAIM)
    stage("temporal_order")
    timeline = order_events(
        text,
        [
            {
                "id": r.reference_id,
                "label": "EVENT",
                "role": "EVENT",
                "start": r.start,
                "end": r.end,
            }
            for r in refs
        ],
    )
    temporal = validate_summary_temporal_order(
        [r.reference_id for _, _, r, _ in aligned], timeline
    )
    if any(f.code == "order_inversion" for f in temporal.findings):
        raise _Stop(BriefRefusal.UNSUPPORTED_CLAIM)
    # No event order is inferred from document order; the unresolved result is preserved.
    stage("unsupported_claims")
    unsupported = score_summary_claims(
        [{k: v for k, v in c.items() if k != "citations"} for c in claims],
        [
            {
                "evidence_id": r.reference_id,
                "relation": "supported",
                "approved": True,
                "claim_id": c,
            }
            for c, _, r, _ in aligned
        ],
    )
    stage("coverage")
    coverage = compute_summary_fact_coverage(rows, claims)
    stage("citation_support")
    support = compute_citation_support_metrics(
        [
            {
                **{k: v for k, v in claim.items() if k != "evidence_ids"},
                "start": segment.start,
                "end": segment.end,
                "source_length": len(summary),
            }
            for claim, (_, segment, _, _) in zip(claims, aligned)
        ],
        rows,
    )
    if not support.deterministic.passed:
        raise _Stop(BriefRefusal.UNSUPPORTED_CLAIM)
    stage("empty_evidence")
    require_summary_evidence(rows)
    stage("privacy")
    leakage = _build_leakage_check(value, summary)
    if not leakage.passed:
        raise _Stop(BriefRefusal.PRIVACY)
    enforce_review_packet_privacy(summary, context.privacy_detector)
    stage("review")
    review = build_review_packet(
        findings=[
            {
                "finding_id": c,
                "label": "clinical_claim",
                "citation_ids": [_digest(r.reference_id)],
            }
            for c, _, r, _ in aligned
        ],
        citations=[
            {"citation_id": _digest(r.reference_id), "source": "reviewed_evidence"}
            for _, _, r, _ in aligned
        ],
        gates=[
            {
                "gate_id": "temporal_order",
                "passed": not temporal.review_required,
                "reason": "human_review_required",
                "blocking": False,
            }
        ],
    )
    stage("provenance")
    provenance = build_guarded_provenance_record(
        output=summary,
        input_value=text,
        evidence=rows,
        policy_fingerprint=packet.policy_fingerprint,
        model_id=generated.metadata["backend_id"],
        review_status="queued",
    )
    result = _result(
        summary,
        None,
        completed,
        citations=[
            {
                "claim_index": i,
                "source_start": r.start,
                "source_end": r.end,
                "output_start": s.start,
                "output_end": s.end,
            }
            for i, (_, s, r, _) in enumerate(aligned)
        ],
        verdicts=verdicts,
        metrics={
            "coverage": coverage.to_dict(),
            "unsupported_claims": unsupported.to_dict(),
            "citation_support": support.to_dict(),
            "temporal_order": temporal.to_dict(),
            "demographics": demographics.to_dict(),
            "leakage": leakage.to_dict(),
        },
        provenance=provenance.to_dict(),
        profile_digest=profile.digest,
        envelope=envelope.to_dict(),
        backend_id=generated.metadata["backend_id"],
        review_packet=review.to_dict(),
    )
    # Scan the assembled protected response, including labels and metadata, not
    # only generation. There is no export or persistence before this check.
    enforce_review_packet_privacy(
        json.dumps(result.to_response(), sort_keys=True), context.privacy_detector
    )
    return result
