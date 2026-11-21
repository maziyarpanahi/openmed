"""Offline authentication and release-readiness checks for capability evidence."""

from __future__ import annotations

import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timedelta

from openmed.compliance.corrective_action import CorrectiveActionLedger
from openmed.compliance.evidence_freshness import (
    EvidenceFreshnessPolicy,
    EvidenceRecord,
    evaluate_evidence_freshness,
)
from openmed.core.intended_use import RegulatoryTier, load_intended_use_registry
from openmed.eval.evidence_pack import EvidencePackError, ValidationEvidencePack
from openmed.eval.evidence_pack.lanes import required_lanes
from openmed.eval.journey_release import _reject_duplicate_keys, _release_hmac


@dataclass(frozen=True)
class EvidenceVerification:
    """Offline verdict containing only controlled reason codes."""

    exit_code: int
    reasons: tuple[str, ...]
    corrective_action_ids: tuple[str, ...] = ()


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
    corrective_action_ledger: str | bytes | None = None,
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
        for name, metrics in required_lanes(code, tier=tier).items():
            lane = actual.get(name)
            if lane is None or not lane["mandatory"]:
                reasons.add("missing_required_lane")
            elif not set(metrics) <= lane["metrics"].keys():
                reasons.add("missing_required_metric")
    if pack.lane_blockers():
        reasons.add("incomplete_lane")
    blocking_ids = ()
    if corrective_action_ledger is not None:
        try:
            ledger = CorrectiveActionLedger.from_dict(
                json.loads(
                    corrective_action_ledger, object_pairs_hook=_reject_duplicate_keys
                )
            )
        except (ValueError, TypeError, RecursionError, OverflowError):
            return EvidenceVerification(2, ("invalid_corrective_action_ledger",))
        blocking_ids = tuple(
            record.record_id
            for record in ledger.records()
            if record.state != "closed"
            and record.severity in {"high", "critical"}
            and code in record.capabilities
            and data["version"] in record.versions
        )
        if blocking_ids:
            reasons.add("open_corrective_action")
    return EvidenceVerification(
        int(bool(reasons)), tuple(sorted(reasons)), blocking_ids
    )
