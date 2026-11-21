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


def site_validation_lane(data: dict, key: bytes | None) -> tuple[dict, list[dict]]:
    """Verify a signed local validation study and project aggregate evidence.

    Args:
        data: Serialized ValidationReport from the site's validation runner.
        key: Independently supplied site-specific HMAC key.
    """
    from openmed.clinical.validation.study import (
        _DEFAULT_SIGNING_KEY,
        VALIDATION_REPORT_SCHEMA_VERSION,
        AcceptanceResult,
        ValidationReport,
    )
    from openmed.core.audit import AuditSignature, stable_hash

    lane = {
        "id": "site_validation",
        "mandatory": False,
        "state": "failure",
        "report_digest": None,
        "required_metrics": [
            "fixture_count",
            "precision",
            "recall",
            "f1",
            "leakage_rate",
        ],
        "metrics": {},
        "claims": [],
    }

    def failure(code):
        return {**lane, "reason_code": code}, []

    if not key or key == _DEFAULT_SIGNING_KEY.encode():
        return failure("site_validation_key_required")
    try:
        signed = dict(data)
        accepted = signed.pop("accepted")
        signature = signed.pop("signature")
        if not signature or not signed.get("repro_hash"):
            return failure("site_validation_unverified")
        signed["acceptance"] = [AcceptanceResult(**row) for row in signed["acceptance"]]
        study = ValidationReport(**signed, signature=AuditSignature(**signature))
        if (
            study.schema_version != VALIDATION_REPORT_SCHEMA_VERSION
            or study.to_dict() != data
            or not study.verify(key)
        ):
            return failure("site_validation_unverified")
        metrics = {
            m: study.metrics["overall"][m]
            for m in ("precision", "recall", "f1", "leakage_rate")
        }
        if (
            type(study.fixture_count) is not int
            or study.fixture_count < 1
            or any(
                type(v) not in (int, float) or not 0 <= v <= 1 for v in metrics.values()
            )
        ):
            return failure("site_validation_invalid")
        refs = []
        for name in ("dataset_id", "data_revision"):
            value = study.provenance[name]
            if not isinstance(value, str) or not value.strip():
                return failure("site_validation_invalid")
            refs.append(
                {
                    "id": "site_validation_" + name,
                    "origin": "aggregate",
                    "digest": stable_hash(value),
                }
            )
        lane.update(
            report_digest=study.repro_hash,
            metrics={**metrics, "fixture_count": study.fixture_count},
        )
        if accepted is not True or not any(c.primary is True for c in study.acceptance):
            lane["reason_code"] = "site_validation_not_accepted"
        else:
            lane["state"] = "success"
        return lane, refs
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return failure("site_validation_invalid")
