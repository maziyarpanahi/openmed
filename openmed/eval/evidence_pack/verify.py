"""Offline authentication and release-readiness checks for capability evidence."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from datetime import datetime, timedelta

from openmed.compliance.evidence_freshness import (
    EvidenceFreshnessPolicy,
    EvidenceRecord,
    evaluate_evidence_freshness,
)
from openmed.core.intended_use import RegulatoryTier, load_intended_use_registry
from openmed.eval.evidence_pack import EvidencePackError, ValidationEvidencePack
from openmed.eval.evidence_pack.lanes import CLINICAL_CREDENTIALED, required_lanes
from openmed.eval.journey_release import _release_hmac


@dataclass(frozen=True)
class EvidenceVerification:
    """Offline verdict containing only controlled reason codes."""

    exit_code: int
    reasons: tuple[str, ...]


def _signature(pack: ValidationEvidencePack, key_id: str, key: bytes) -> dict:
    envelope = {
        "algorithm": "HMAC-SHA256",
        "key_id": key_id,
        "payload_digest": pack.digest(),
    }
    return {**envelope, "value": _release_hmac(key, envelope)}


def sign_evidence_pack(
    pack: ValidationEvidencePack, *, key_id: str, key: bytes
) -> ValidationEvidencePack:
    """Authenticate a pack's digest and key identifier with a local HMAC key."""
    return ValidationEvidencePack(
        {**pack.signing_payload(), "signature": _signature(pack, key_id, key)}
    )


def verify_evidence_pack(
    raw: str | bytes,
    *,
    key_id: str,
    key: bytes,
    as_of: datetime,
    max_age: timedelta,
    profile: str = "capability",
) -> EvidenceVerification:
    """Verify JSON against the caller's trusted key, clock and freshness limit.

    Returns exit code 0 for ready, 1 for failed checks or 2 for invalid input.
    """
    try:
        pack = ValidationEvidencePack.from_json(raw)
    except EvidencePackError as exc:
        return EvidenceVerification(2, (str(exc),))

    if not isinstance(key, bytes) or len(key) < 32 or not key_id:
        return EvidenceVerification(2, ("invalid_key_configuration",))
    data = pack.to_dict()
    try:
        policy = EvidenceFreshnessPolicy("v1", {"pack": max_age})
        record = EvidenceRecord("pack", "pack", data["generated_at"], "v1")
        freshness = evaluate_evidence_freshness([record], policy, as_of=as_of)
    except (ValueError, TypeError, OverflowError):
        return EvidenceVerification(2, ("invalid_freshness_policy",))

    reasons = set(freshness.reason_counts)
    signature = data["signature"]
    if signature is None:
        reasons.add("unsigned_pack")
    else:
        expected = _signature(pack, key_id, key)
        if signature["payload_digest"] != expected["payload_digest"]:
            reasons.add("digest_mismatch")
        if signature["key_id"] != key_id or not hmac.compare_digest(
            signature["value"], expected["value"]
        ):
            reasons.add("signature_mismatch")

    code = data["capability"]
    declaration = load_intended_use_registry().declarations.get(code)
    if declaration is None:
        reasons.add("unknown_capability")
    else:
        if declaration.digest() != data["intended_use_digest"]:
            reasons.add("intended_use_mismatch")
        tier = RegulatoryTier(data["regulatory_tier"])
        if not declaration.regulatory_tier.obligations <= tier.obligations:
            reasons.add("tier_mismatch")
        actual = {lane["id"]: lane for lane in data["lanes"]}
        if profile == "journey" and code == "journey" and "journey_release" in data:
            from openmed.eval.journey_release import (
                journey_metric_blockers,
                journey_release_profile,
            )

            expected_lanes = journey_release_profile()
            if set(actual) != set(expected_lanes):
                reasons.add("journey_lane_set_mismatch")
            detail = data["journey_release"]
            reasons.update(detail["blocking_codes"])
            inputs = {i["id"]: i for i in data["frozen_inputs"]}
            if (
                inputs.get("release_manifest", {}).get("digest")
                != detail["manifest_digest"]
            ):
                reasons.add("journey_manifest_mismatch")
            for name, metrics in expected_lanes.items():
                lane = actual.get(name)
                if lane is None:
                    continue
                if inputs.get(name, {}).get("digest") != lane["report_digest"]:
                    reasons.add("journey_report_mismatch")
                if set(metrics) <= lane["metrics"].keys():
                    reasons.update(
                        f"{name}:{reason}"
                        for reason in journey_metric_blockers(name, lane["metrics"])
                    )
        else:
            expected_lanes = required_lanes(code, tier=tier)
            if profile != "capability" or "journey_release" in data:
                reasons.add("evidence_profile_mismatch")
        for name, metrics in expected_lanes.items():
            lane = actual.get(name)
            if lane is None or not lane["mandatory"]:
                reasons.add("missing_required_lane")
            elif not set(metrics) <= lane["metrics"].keys():
                reasons.add("missing_required_metric")
        if "credentialed_corpus_measured" in data["claims"]:
            expected_lane, expected_metrics = CLINICAL_CREDENTIALED.get(
                code, (None, ())
            )
            measured = actual.get(expected_lane)
            if not (
                measured
                and measured["state"] == "success"
                and measured["report_digest"]
                and "credentialed_corpus_measured" in measured["claims"]
                and all(
                    0 <= measured["metrics"].get(m, -1) <= 1 for m in expected_metrics
                )
                and any(
                    ref["id"] == expected_lane
                    and ref["origin"] == "credentialed"
                    and ref["digest"] == measured["report_digest"]
                    for ref in data["frozen_inputs"]
                )
            ):
                reasons.add("credentialed_lane_required")
        if code == "clinical.temporal_relations" and (
            temporal := actual.get("temporal_tlinks")
        ):
            from openmed.eval.metrics import TEMPORAL_AWARENESS_F1_FLOOR

            if not (
                TEMPORAL_AWARENESS_F1_FLOOR
                <= temporal["metrics"].get("awareness_f1", -1)
                <= 1
                and temporal["metrics"].get("closure_violations") == 0
            ):
                reasons.add("temporal_readiness_floor")
        if code == "structured_release" and (
            lane := actual.get("membership_inference")
        ):
            metrics = lane["metrics"]
            if not (
                0
                <= metrics.get("membership_inference_rate", -1)
                <= metrics.get("max_inference_rate", -1)
                <= 1
                and all(
                    type(metrics.get(k)) is int and metrics[k] > 0
                    for k in ("release_count", "candidate_count")
                )
                and lane["report_digest"]
                and any(
                    item["id"] == "membership_inference"
                    and item["origin"] == "aggregate"
                    and item["digest"] == lane["report_digest"]
                    for item in data["frozen_inputs"]
                )
            ):
                reasons.add("invalid_membership_evidence")
        if "human_factors" in expected_lanes:
            sources = {
                item["id"]
                for item in data["frozen_inputs"]
                if item["origin"] == "aggregate"
            }
            if not {"use_error_snapshot", "reviewer_disagreement"} <= sources:
                reasons.add("missing_human_factors_sources")
    if pack.lane_blockers():
        reasons.add("incomplete_lane")
    return EvidenceVerification(int(bool(reasons)), tuple(sorted(reasons)))
