"""Trusted offline lane and metric requirements for capability evidence packs."""

from __future__ import annotations

from openmed.core.intended_use import RegulatoryTier, load_intended_use_registry
from openmed.eval.evidence_pack import EvidencePackError

# Gate IDs use the lower-case wire vocabulary of ValidationEvidencePack.
_PRIVACY = {
    "g1a": ("direct_identifier_recall",),
    "g2": ("recall",),
    "g3": ("critical_leakage_count",),
    "g4": ("quantized_recall_delta",),
    "span_integrity": ("invalid_span_count",),
}
_EXTRACTION = {"extraction": ("precision", "recall", "f1")}
_GENERATION = {
    "unsupported_claims": ("unsupported_claim_rate",),
    "citation_support": ("citation_precision", "support_recall"),
}
_MEASUREMENTS = {"measurement_candidates": ("abstention_rate", "reviewed_count")}
_CAPABILITIES = {
    "analyze_text": _EXTRACTION,
    "brief": _GENERATION,
    "clinical_measurements": _MEASUREMENTS,
    "decisions": {"decision_qualification": ("selective_accuracy", "coverage")},
    "deidentify": _PRIVACY,
    "ecg": _MEASUREMENTS,
    "extract_pii": _PRIVACY,
    "generate_text": _GENERATION,
    "ground": {"grounding": ("precision", "recall")},
    "journey": {"evidence_coverage": ("evidence_coverage",)},
    "mental_health": _EXTRACTION,
    "nli": {"claim_verification": ("macro_f1", "abstention_rate")},
    "review_evidence": {"evidence_coverage": ("evidence_coverage",)},
    "structured_extraction": _EXTRACTION,
    "summarize": _GENERATION,
    "transcribe": {"transcription": ("word_error_rate",)},
    "visual_qa": {"visual_description": ("abstention_rate", "reviewed_count")},
    "waveform": _MEASUREMENTS,
}
_OBLIGATION_LANES = {
    "notice_required": {"notice": ("notice_coverage",)},
    "acknowledgement_required": {"acknowledgement": ("acknowledgement_coverage",)},
    "clinician_mode_required": {
        "human_factors": ("use_error_count", "review_disagreement_rate"),
        "shadow": ("compared_count", "disagreement_count"),
    },
}


def required_lanes(
    capability: str, *, tier: RegulatoryTier | str | None = None
) -> dict[str, tuple[str, ...]]:
    """Return required lanes and numeric metric names in stable order.

    Args:
        capability: Stable capability code from the reviewed registry.
        tier: Optional additional deployment tier. Its safeguards supplement
            the capability's declared obligations.
    """
    declarations = load_intended_use_registry().declarations
    if (
        not isinstance(capability, str)
        or capability not in declarations
        or capability not in _CAPABILITIES
    ):
        raise EvidencePackError("unknown_capability")
    obligations = declarations[capability].regulatory_tier.obligations
    if tier is not None:
        try:
            obligations |= RegulatoryTier(tier).obligations
        except (TypeError, ValueError):
            raise EvidencePackError("unknown_regulatory_tier") from None
    lanes = dict(_CAPABILITIES[capability])
    for obligation, extra in _OBLIGATION_LANES.items():
        if obligation in obligations:
            lanes.update(extra)
    return dict(sorted(lanes.items()))
