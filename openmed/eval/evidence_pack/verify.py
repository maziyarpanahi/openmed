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
from openmed.eval.journey_release import (
    _canonical_json,
    _reject_duplicate_keys,
    _release_hmac,
)


@dataclass(frozen=True)
class EvidenceVerification:
    """Offline verdict containing only controlled reason codes."""

    exit_code: int
    reasons: tuple[str, ...]
    corrective_action_ids: tuple[str, ...] = ()


def _envelope(pack: ValidationEvidencePack, key_id: str, algorithm: str) -> dict:
    return {"algorithm": algorithm, "key_id": key_id, "payload_digest": pack.digest()}


def sign_evidence_pack(
    pack: ValidationEvidencePack,
    *,
    key_id: str,
    key: bytes,
    algorithm: str = "HMAC-SHA256",
) -> ValidationEvidencePack:
    """Sign the canonical digest envelope with a caller-provided HMAC key or Ed25519 seed."""
    envelope = _envelope(pack, key_id, algorithm)
    if algorithm == "HMAC-SHA256":
        value = _release_hmac(key, envelope)
    elif algorithm == "Ed25519":
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
        except ImportError:
            raise EvidencePackError("ed25519_unavailable") from None
        if not isinstance(key, bytes) or len(key) != 32:
            raise EvidencePackError("invalid_key_configuration")
        value = (
            Ed25519PrivateKey.from_private_bytes(key)
            .sign(_canonical_json(envelope).encode("utf-8"))
            .hex()
        )
    else:
        raise EvidencePackError("unsupported_signature_algorithm")
    return ValidationEvidencePack(
        {**pack.signing_payload(), "signature": {**envelope, "value": value}}
    )


def verify_evidence_pack(
    raw: str | bytes,
    *,
    key_id: str,
    key: bytes | None = None,
    public_key: bytes | None = None,
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

    if not key_id or not (
        (public_key is None and isinstance(key, bytes) and len(key) >= 32)
        or (key is None and isinstance(public_key, bytes) and len(public_key) == 32)
    ):
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
        algorithm = "Ed25519" if public_key is not None else "HMAC-SHA256"
        expected = _envelope(pack, key_id, algorithm)
        if signature["payload_digest"] != expected["payload_digest"]:
            reasons.add("digest_mismatch")
        if signature["algorithm"] != algorithm or signature["key_id"] != key_id:
            reasons.add("signature_mismatch")
        elif public_key is not None:
            from openmed.core.portable_bundle import (
                PortableBundleError,
                _ed25519_verify,
            )

            try:
                valid = _ed25519_verify(
                    public_key,
                    bytes.fromhex(signature["value"]),
                    _canonical_json(expected).encode("utf-8"),
                )
            except PortableBundleError:
                return EvidenceVerification(2, ("ed25519_unavailable",))
            if not valid:
                reasons.add("signature_mismatch")
            wire = raw if isinstance(raw, bytes) else raw.encode("utf-8")
            if wire != pack.to_json().encode("utf-8"):
                reasons.add("noncanonical_signed_pack")
        elif not hmac.compare_digest(signature["value"], _release_hmac(key, expected)):
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
