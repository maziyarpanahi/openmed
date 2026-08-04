"""Expert-authored signatures over verified de-identification evidence.

This module does not generate, infer, or approve an Expert Determination.
It creates a provider-neutral Ed25519 envelope only after an expert supplies
their own identity, qualifications, methodology statement, conclusion, and
reassessment time for an existing verified technical evidence report.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Final

from .expert_review import ExpertReviewEvidenceReport

EXPERT_ATTESTATION_SCHEMA_VERSION: Final = 1
EXPERT_ATTESTATION_REPORT_TYPE: Final = "expert_authored_deidentification_attestation"
EXPERT_ATTESTATION_CANONICALIZATION: Final = "RFC8785"
EXPERT_ATTESTATION_DISCLAIMER: Final = (
    "This is an expert-authored cryptographic signature envelope over "
    "technical evidence and the expert's stated conclusion. OpenMed does not "
    "generate, infer, validate, or approve the expert's conclusion, "
    "qualifications, or methodology. Signature verification is not an "
    "automated Expert Determination or release authorization."
)
EXPERT_ATTESTATION_CONCLUSIONS: Final = frozenset(
    {"very_small_risk", "not_approved", "requires_changes"}
)

_SIGNATURE_ALGORITHM: Final = "Ed25519"
_FRESHNESS_STATUSES: Final = frozenset({"current", "not_yet_valid", "reassessment_due"})
_SHA256_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_MAX_EXPERT_IDENTITY_LENGTH: Final = 512
_MAX_QUALIFICATIONS_LENGTH: Final = 4_096
_MAX_SCOPE_METHODOLOGY_LENGTH: Final = 16_384
_UTC: Final = timezone.utc

__all__ = [
    "EXPERT_ATTESTATION_CANONICALIZATION",
    "EXPERT_ATTESTATION_CONCLUSIONS",
    "EXPERT_ATTESTATION_DISCLAIMER",
    "EXPERT_ATTESTATION_REPORT_TYPE",
    "EXPERT_ATTESTATION_SCHEMA_VERSION",
    "ExpertAttestationBindings",
    "ExpertAttestationEnvelope",
    "ExpertAttestationVerification",
    "create_expert_attestation",
]


class _InvalidAttestationJson(ValueError):
    """Raised for duplicate keys or non-finite JSON constants."""


@dataclass(frozen=True)
class ExpertAttestationBindings:
    """Technical evidence digests covered by an expert's signature."""

    evidence_integrity_hash: str
    source_dataset_digest: str
    released_dataset_digest: str
    released_schema_digest: str
    policy_digest: str
    hierarchy_digest: str
    config_digest: str
    software_digest: str

    def __post_init__(self) -> None:
        for value in (
            self.evidence_integrity_hash,
            self.source_dataset_digest,
            self.released_dataset_digest,
            self.released_schema_digest,
            self.policy_digest,
            self.hierarchy_digest,
            self.config_digest,
            self.software_digest,
        ):
            _require_digest(value)

    def to_dict(self) -> dict[str, str]:
        """Return the exact allow-listed binding schema."""

        return {
            "evidence_integrity_hash": self.evidence_integrity_hash,
            "source_dataset_digest": self.source_dataset_digest,
            "released_dataset_digest": self.released_dataset_digest,
            "released_schema_digest": self.released_schema_digest,
            "policy_digest": self.policy_digest,
            "hierarchy_digest": self.hierarchy_digest,
            "config_digest": self.config_digest,
            "software_digest": self.software_digest,
        }


@dataclass(frozen=True)
class ExpertAttestationVerification:
    """Independent signature, binding, conclusion, and freshness results.

    No combined approval flag is provided. A valid signature proves only that
    the holder of the supplied key signed the envelope; the expert's conclusion
    and the envelope's freshness remain separate facts.
    """

    cryptographically_valid: bool
    key_id_matches: bool
    evidence_integrity_valid: bool
    bindings_match: bool
    conclusion: str
    freshness_status: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.cryptographically_valid, "cryptographically_valid"),
            (self.key_id_matches, "key_id_matches"),
            (self.evidence_integrity_valid, "evidence_integrity_valid"),
            (self.bindings_match, "bindings_match"),
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} must be a boolean")
        _require_choice(
            self.conclusion,
            EXPERT_ATTESTATION_CONCLUSIONS,
            "expert conclusion",
        )
        _require_choice(
            self.freshness_status,
            _FRESHNESS_STATUSES,
            "freshness status",
        )

    @property
    def fresh(self) -> bool:
        """Return whether reassessment is not yet due at verification time."""

        return self.freshness_status == "current"

    def __bool__(self) -> bool:
        """Reject accidental use as a combined release-approval decision."""

        raise TypeError(
            "expert-attestation verification has no combined truth value; "
            "inspect its independent signature, key, evidence, binding, "
            "conclusion, and freshness fields"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the non-conflating verification result."""

        return {
            "cryptographically_valid": self.cryptographically_valid,
            "key_id_matches": self.key_id_matches,
            "evidence_integrity_valid": self.evidence_integrity_valid,
            "bindings_match": self.bindings_match,
            "conclusion": self.conclusion,
            "freshness_status": self.freshness_status,
            "fresh": self.fresh,
        }


@dataclass(frozen=True)
class ExpertAttestationEnvelope:
    """An expert-authored Ed25519 signature envelope.

    Construct new envelopes with :func:`create_expert_attestation`, which
    requires an already verified :class:`ExpertReviewEvidenceReport`. Parsing
    an envelope does not endorse its expert, conclusion, or methodology.
    """

    expert_identity: str
    qualifications: str
    scope_and_methodology: str
    conclusion: str
    issued_at: datetime
    reassessment_at: datetime
    bindings: ExpertAttestationBindings
    supporting_evidence_digests: tuple[tuple[str, str], ...]
    key_id: str
    signature: str
    schema_version: int = EXPERT_ATTESTATION_SCHEMA_VERSION
    report_type: str = EXPERT_ATTESTATION_REPORT_TYPE
    disclaimer: str = EXPERT_ATTESTATION_DISCLAIMER
    signature_algorithm: str = _SIGNATURE_ALGORITHM
    canonicalization: str = EXPERT_ATTESTATION_CANONICALIZATION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != EXPERT_ATTESTATION_SCHEMA_VERSION
        ):
            raise ValueError("unsupported expert-attestation schema version")
        if self.report_type != EXPERT_ATTESTATION_REPORT_TYPE:
            raise ValueError("unsupported expert-attestation report type")
        if self.disclaimer != EXPERT_ATTESTATION_DISCLAIMER:
            raise ValueError("expert-attestation disclaimer must not be changed")
        if self.signature_algorithm != _SIGNATURE_ALGORITHM:
            raise ValueError("expert attestations require Ed25519 signatures")
        if self.canonicalization != EXPERT_ATTESTATION_CANONICALIZATION:
            raise ValueError("unsupported expert-attestation canonicalization")
        _require_text(
            self.expert_identity,
            name="expert_identity",
            maximum_length=_MAX_EXPERT_IDENTITY_LENGTH,
            allow_newlines=False,
        )
        _require_text(
            self.qualifications,
            name="qualifications",
            maximum_length=_MAX_QUALIFICATIONS_LENGTH,
            allow_newlines=True,
        )
        _require_text(
            self.scope_and_methodology,
            name="scope_and_methodology",
            maximum_length=_MAX_SCOPE_METHODOLOGY_LENGTH,
            allow_newlines=True,
        )
        _require_choice(
            self.conclusion,
            EXPERT_ATTESTATION_CONCLUSIONS,
            "expert conclusion",
        )
        issued_at = _require_utc_datetime(self.issued_at, name="issued_at")
        reassessment_at = _require_utc_datetime(
            self.reassessment_at,
            name="reassessment_at",
        )
        if reassessment_at <= issued_at:
            raise ValueError("reassessment_at must be later than issued_at")
        if not isinstance(self.bindings, ExpertAttestationBindings):
            raise TypeError("bindings must be ExpertAttestationBindings")
        supporting = _supporting_evidence_tuple(self.supporting_evidence_digests)
        _require_safe_identifier(self.key_id, name="key_id")
        _decode_signature(self.signature)

        object.__setattr__(self, "issued_at", issued_at)
        object.__setattr__(self, "reassessment_at", reassessment_at)
        object.__setattr__(self, "supporting_evidence_digests", supporting)

    def _signed_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_type": self.report_type,
            "disclaimer": self.disclaimer,
            "expert": {
                "identity": self.expert_identity,
                "qualifications": self.qualifications,
                "scope_and_methodology": self.scope_and_methodology,
            },
            "conclusion": self.conclusion,
            "issued_at": _format_utc_datetime(self.issued_at),
            "reassessment_at": _format_utc_datetime(self.reassessment_at),
            "bindings": self.bindings.to_dict(),
            "supporting_evidence_digests": dict(self.supporting_evidence_digests),
            "signature": {
                "algorithm": self.signature_algorithm,
                "canonicalization": self.canonicalization,
                "key_id": self.key_id,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the strict, provider-neutral JSON schema."""

        payload = self._signed_payload()
        payload["signature"] = {
            **payload["signature"],
            "value": self.signature,
        }
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize deterministically without provider or environment data."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
        )

    def verify(
        self,
        *,
        evidence: ExpertReviewEvidenceReport,
        public_key: Any,
        expected_key_id: str,
        expected_supporting_evidence_digests: (
            Mapping[str, str] | tuple[tuple[str, str], ...]
        ) = (),
        as_of: datetime | None = None,
    ) -> ExpertAttestationVerification:
        """Verify signature and bindings without interpreting the conclusion.

        The caller supplies the trusted public key and its expected ``key_id``.
        Supporting evidence must be supplied by name and digest when the
        envelope binds any; omission then produces ``bindings_match=False``.
        """

        if not isinstance(evidence, ExpertReviewEvidenceReport):
            raise TypeError("evidence must be an ExpertReviewEvidenceReport")
        _require_safe_identifier(expected_key_id, name="expected_key_id")
        expected_supporting = _supporting_evidence_tuple(
            expected_supporting_evidence_digests
        )
        verification_time = (
            datetime.now(_UTC)
            if as_of is None
            else _require_aware_datetime(as_of, name="as_of").astimezone(_UTC)
        )
        evidence_integrity_valid = evidence.verify()
        bindings_match = (
            self.bindings == _bindings_from_evidence(evidence)
            and self.supporting_evidence_digests == expected_supporting
        )
        key_id_matches = hmac.compare_digest(self.key_id, expected_key_id)
        signature_valid = _verify_signature(
            public_key,
            _canonical_bytes(self._signed_payload()),
            _decode_signature(self.signature),
        )

        if verification_time < self.issued_at:
            freshness_status = "not_yet_valid"
        elif verification_time >= self.reassessment_at:
            freshness_status = "reassessment_due"
        else:
            freshness_status = "current"

        return ExpertAttestationVerification(
            cryptographically_valid=signature_valid,
            key_id_matches=key_id_matches,
            evidence_integrity_valid=evidence_integrity_valid,
            bindings_match=bindings_match,
            conclusion=self.conclusion,
            freshness_status=freshness_status,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExpertAttestationEnvelope:
        """Parse the exact schema and reject missing or unknown keys."""

        item = _object(
            data,
            {
                "schema_version",
                "report_type",
                "disclaimer",
                "expert",
                "conclusion",
                "issued_at",
                "reassessment_at",
                "bindings",
                "supporting_evidence_digests",
                "signature",
            },
            "expert attestation",
        )
        expert = _object(
            item["expert"],
            {"identity", "qualifications", "scope_and_methodology"},
            "expert",
        )
        binding_values = _object(
            item["bindings"],
            {
                "evidence_integrity_hash",
                "source_dataset_digest",
                "released_dataset_digest",
                "released_schema_digest",
                "policy_digest",
                "hierarchy_digest",
                "config_digest",
                "software_digest",
            },
            "bindings",
        )
        signature = _object(
            item["signature"],
            {"algorithm", "canonicalization", "key_id", "value"},
            "signature",
        )
        supporting = item["supporting_evidence_digests"]
        if not isinstance(supporting, Mapping):
            raise TypeError("supporting_evidence_digests must be an object")
        return cls(
            expert_identity=expert["identity"],
            qualifications=expert["qualifications"],
            scope_and_methodology=expert["scope_and_methodology"],
            conclusion=item["conclusion"],
            issued_at=_parse_utc_datetime(item["issued_at"], name="issued_at"),
            reassessment_at=_parse_utc_datetime(
                item["reassessment_at"],
                name="reassessment_at",
            ),
            bindings=ExpertAttestationBindings(**binding_values),
            supporting_evidence_digests=_supporting_evidence_tuple(supporting),
            key_id=signature["key_id"],
            signature=signature["value"],
            schema_version=item["schema_version"],
            report_type=item["report_type"],
            disclaimer=item["disclaimer"],
            signature_algorithm=signature["algorithm"],
            canonicalization=signature["canonicalization"],
        )

    @classmethod
    def from_json(cls, data: str | bytes) -> ExpertAttestationEnvelope:
        """Parse strict JSON, including duplicate-key rejection."""

        if not isinstance(data, (str, bytes)):
            raise TypeError("expert-attestation JSON must be text or bytes")
        try:
            value = json.loads(
                data,
                object_pairs_hook=_json_object_without_duplicates,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, UnicodeError, _InvalidAttestationJson):
            raise ValueError("expert-attestation JSON is malformed") from None
        if not isinstance(value, Mapping):
            raise TypeError("expert-attestation JSON must contain an object")
        return cls.from_dict(value)


def create_expert_attestation(
    evidence: ExpertReviewEvidenceReport,
    *,
    expert_identity: str,
    qualifications: str,
    scope_and_methodology: str,
    conclusion: str,
    issued_at: datetime,
    reassessment_at: datetime,
    private_key: Any,
    key_id: str,
    supporting_evidence_digests: (Mapping[str, str] | tuple[tuple[str, str], ...]) = (),
) -> ExpertAttestationEnvelope:
    """Create an expert-authored signature envelope over verified evidence.

    OpenMed does not choose the conclusion or treat the resulting signature as
    an automated Expert Determination. The named expert must supply and own
    every substantive statement and the signing key.
    """

    if not isinstance(evidence, ExpertReviewEvidenceReport):
        raise TypeError("evidence must be an ExpertReviewEvidenceReport")
    if not evidence.verify():
        raise ValueError("expert-review evidence must verify before attestation")
    issued = _require_utc_datetime(issued_at, name="issued_at")
    reassessment = _require_utc_datetime(
        reassessment_at,
        name="reassessment_at",
    )
    bindings = _bindings_from_evidence(evidence)
    supporting = _supporting_evidence_tuple(supporting_evidence_digests)
    _require_safe_identifier(key_id, name="key_id")

    unsigned = ExpertAttestationEnvelope(
        expert_identity=expert_identity,
        qualifications=qualifications,
        scope_and_methodology=scope_and_methodology,
        conclusion=conclusion,
        issued_at=issued,
        reassessment_at=reassessment,
        bindings=bindings,
        supporting_evidence_digests=supporting,
        key_id=key_id,
        signature=base64.b64encode(b"\x00" * 64).decode("ascii"),
    )
    signature = _sign(
        private_key,
        _canonical_bytes(unsigned._signed_payload()),
    )
    return ExpertAttestationEnvelope(
        expert_identity=unsigned.expert_identity,
        qualifications=unsigned.qualifications,
        scope_and_methodology=unsigned.scope_and_methodology,
        conclusion=unsigned.conclusion,
        issued_at=unsigned.issued_at,
        reassessment_at=unsigned.reassessment_at,
        bindings=unsigned.bindings,
        supporting_evidence_digests=unsigned.supporting_evidence_digests,
        key_id=unsigned.key_id,
        signature=base64.b64encode(signature).decode("ascii"),
    )


def _bindings_from_evidence(
    evidence: ExpertReviewEvidenceReport,
) -> ExpertAttestationBindings:
    return ExpertAttestationBindings(
        evidence_integrity_hash=evidence.integrity_hash,
        source_dataset_digest=evidence.digests.source_dataset,
        released_dataset_digest=evidence.digests.dataset,
        released_schema_digest=evidence.digests.schema,
        policy_digest=evidence.digests.policy,
        hierarchy_digest=evidence.digests.hierarchy,
        config_digest=evidence.digests.config,
        software_digest=evidence.digests.software,
    )


def _supporting_evidence_tuple(
    value: Mapping[str, str] | tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    if isinstance(value, Mapping):
        items = tuple(value.items())
    elif isinstance(value, tuple):
        items = value
    else:
        raise TypeError(
            "supporting_evidence_digests must be a mapping or tuple of pairs"
        )
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("supporting evidence entries must be name/digest pairs")
        name, digest = item
        _require_safe_identifier(name, name="supporting evidence name")
        _require_digest(digest)
        if name in seen:
            raise ValueError("supporting evidence names must be unique")
        seen.add(name)
        normalized.append((name, digest))
    return tuple(sorted(normalized))


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return RFC 8785 bytes for the version-one allow-listed schema.

    Version one has only ASCII object keys and integer JSON numbers, while
    free-form expert text is validated Unicode. Those constraints let the
    standard library produce the RFC 8785 representation exactly.
    """

    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _format_utc_datetime(value: datetime) -> str:
    canonical = _require_utc_datetime(value, name="timestamp")
    timespec = "microseconds" if canonical.microsecond else "seconds"
    return canonical.isoformat(timespec=timespec).replace("+00:00", "Z")


def _parse_utc_datetime(value: Any, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise ValueError(f"{name} must be a canonical UTC timestamp") from None
    canonical = _require_utc_datetime(parsed, name=name)
    if _format_utc_datetime(canonical) != value:
        raise ValueError(f"{name} must be a canonical UTC timestamp")
    return canonical


def _require_aware_datetime(value: Any, *, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _require_utc_datetime(value: Any, *, name: str) -> datetime:
    aware = _require_aware_datetime(value, name=name)
    if aware.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use UTC")
    return aware.astimezone(_UTC)


def _require_text(
    value: Any,
    *,
    name: str,
    maximum_length: int,
    allow_newlines: bool,
) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty without surrounding whitespace")
    if len(value) > maximum_length:
        raise ValueError(f"{name} exceeds its maximum length")
    permitted_controls = {"\n", "\t"} if allow_newlines else set()
    if any(
        (
            unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
            and character not in permitted_controls
        )
        for character in value
    ):
        raise ValueError(
            f"{name} contains an unsupported Unicode surrogate, control, or "
            "invisible character"
        )
    if not allow_newlines and any(character in {"\n", "\r"} for character in value):
        raise ValueError(f"{name} must be a single line")


def _require_digest(value: Any) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError("attestation bindings must use canonical sha256 digests")


def _require_safe_identifier(value: Any, *, name: str) -> None:
    if not isinstance(value, str) or not _SAFE_IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{name} must be a safe metadata identifier")


def _require_choice(value: Any, choices: frozenset[str], name: str) -> None:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must use an allowed coded value")


def _object(
    value: Any,
    expected_keys: set[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError(f"{name} must contain exactly its documented fields")
    return dict(value)


def _json_object_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidAttestationJson("duplicate object key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise _InvalidAttestationJson("non-finite JSON number")


def _decode_signature(value: Any) -> bytes:
    if not isinstance(value, str):
        raise TypeError("signature value must be base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("signature value must use canonical base64") from None
    if len(decoded) != 64 or base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError("signature value must be a canonical Ed25519 signature")
    return decoded


def _load_crypto_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except ImportError as exc:
        raise ImportError(
            "Expert attestation signing and verification requires the "
            "'integrity' extra. Install with `pip install openmed[integrity]`."
        ) from exc
    return (
        Ed25519PrivateKey,
        Ed25519PublicKey,
        serialization,
        InvalidSignature,
        UnsupportedAlgorithm,
    )


def _private_key(value: Any) -> Any:
    (
        private_type,
        _public_type,
        serialization,
        _invalid_signature,
        unsupported_algorithm,
    ) = _load_crypto_dependencies()
    if isinstance(value, private_type):
        return value
    raw = value.encode("utf-8") if isinstance(value, str) else value
    if not isinstance(raw, bytes):
        raise TypeError("private_key must be an Ed25519 key or key bytes")
    if len(raw) == 32 and not raw.startswith(b"-----BEGIN"):
        try:
            return private_type.from_private_bytes(raw)
        except ValueError:
            raise ValueError("Ed25519 private key bytes are invalid") from None
    try:
        loaded = serialization.load_pem_private_key(raw, password=None)
    except (TypeError, ValueError, unsupported_algorithm):
        raise ValueError("Ed25519 private key could not be loaded") from None
    if not isinstance(loaded, private_type):
        raise TypeError("private_key must contain an Ed25519 private key")
    return loaded


def _public_key(value: Any) -> Any:
    (
        _private_type,
        public_type,
        serialization,
        _invalid_signature,
        unsupported_algorithm,
    ) = _load_crypto_dependencies()
    if isinstance(value, public_type):
        return value
    raw = value.encode("utf-8") if isinstance(value, str) else value
    if not isinstance(raw, bytes):
        raise TypeError("public_key must be an Ed25519 key or key bytes")
    if len(raw) == 32 and not raw.startswith(b"-----BEGIN"):
        try:
            return public_type.from_public_bytes(raw)
        except ValueError:
            raise ValueError("Ed25519 public key bytes are invalid") from None
    try:
        loaded = serialization.load_pem_public_key(raw)
    except (TypeError, ValueError, unsupported_algorithm):
        raise ValueError("Ed25519 public key could not be loaded") from None
    if not isinstance(loaded, public_type):
        raise TypeError("public_key must contain an Ed25519 public key")
    return loaded


def _sign(private_key: Any, payload: bytes) -> bytes:
    signature = _private_key(private_key).sign(payload)
    if len(signature) != 64:
        raise ValueError("Ed25519 signing returned an invalid signature")
    return signature


def _verify_signature(public_key: Any, payload: bytes, signature: bytes) -> bool:
    (
        _private_type,
        _public_type,
        _serialization,
        invalid_signature,
        _unsupported_algorithm,
    ) = _load_crypto_dependencies()
    verifier = _public_key(public_key)
    try:
        verifier.verify(signature, payload)
    except invalid_signature:
        return False
    return True
