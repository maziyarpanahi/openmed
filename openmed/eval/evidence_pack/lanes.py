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


def human_factors_lane(capability: str, data: dict) -> tuple[str, dict, list[dict]]:
    """Project a scoped use-error snapshot and a suppressed disagreement report.

    Args:
        capability: Declared capability for the measurement session.
        data: Local aggregate inputs and canonical snapshot digest.
    """
    import json
    from importlib.resources import files

    from jsonschema import Draft202012Validator

    from openmed.clinical.use_errors import UseErrorCode
    from openmed.core.no_phi_telemetry import MAX_COUNTER_VALUE, MAX_SNAPSHOT_SAMPLES
    from openmed.core.schemas import schema_fingerprint
    from openmed.eval.reviewer_disagreement import _publish_rate

    if not isinstance(data, dict) or data.get("capability") != capability:
        raise EvidencePackError("human_factors_scope_mismatch")
    if not data.keys() <= {
        "capability",
        "use_error_snapshot",
        "use_error_snapshot_digest",
        "reviewer_disagreement",
    }:
        raise EvidencePackError("invalid_human_factors_input")
    snap, review = data.get("use_error_snapshot"), data.get("reviewer_disagreement")
    if snap is None or review is None or data.get("use_error_snapshot_digest") is None:
        return "unknown", {}, []
    if (
        not isinstance(snap, dict)
        or set(snap) != {"schema_version", "counts"}
        or type(snap["schema_version"]) is not int
        or snap["schema_version"] != 1
    ):
        raise EvidencePackError("invalid_use_error_snapshot")
    digest = schema_fingerprint(snap)
    if digest != data["use_error_snapshot_digest"]:
        raise EvidencePackError("use_error_digest_mismatch")
    total = 0
    seen = set()
    if (
        not isinstance(snap["counts"], list)
        or len(snap["counts"]) > MAX_SNAPSHOT_SAMPLES
    ):
        raise EvidencePackError("invalid_use_error_snapshot")
    for row in snap["counts"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"code", "capability", "tier", "count"}
            or row["capability"] not in (capability, "clinician_mode", "notice")
            or row["code"] not in tuple(UseErrorCode)
            or row["tier"] not in tuple(RegulatoryTier)
        ):
            raise EvidencePackError("invalid_use_error_snapshot")
        dimension = (row["code"], row["capability"], row["tier"])
        if (
            dimension in seen
            or type(row["count"]) is not int
            or not 1 <= row["count"] <= MAX_COUNTER_VALUE
        ):
            raise EvidencePackError("invalid_use_error_snapshot")
        total += row["count"]
        seen.add(dimension)
    schema = files("openmed.core.schemas.json").joinpath(
        "reviewer_disagreement_report.schema.json"
    )
    if not Draft202012Validator(json.loads(schema.read_bytes())).is_valid(review):
        raise EvidencePackError("invalid_reviewer_disagreement")
    refs = [
        {"id": "use_error_snapshot", "origin": "aggregate", "digest": digest},
        {
            "id": "reviewer_disagreement",
            "origin": "aggregate",
            "digest": schema_fingerprint(review),
        },
    ]
    agreement = review["agreement"]
    if agreement["suppressed"] or not agreement["denominator"]:
        return "unknown", {}, refs
    n, d = agreement["numerator"], agreement["denominator"]
    if (
        not 0 <= n <= d
        or _publish_rate(n, d, minimum_cell_size=review["minimum_cell_size"]).to_dict()
        != agreement
    ):
        raise EvidencePackError("invalid_reviewer_disagreement")
    return (
        "success",
        {"use_error_count": total, "review_disagreement_rate": (d - n) / d},
        refs,
    )
