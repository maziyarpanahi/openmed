"""Safety envelope for local summaries after de-identification.

The summary stage is deliberately separate from de-identification.  Callers
must first turn a structured de-identification result into a
:class:`VerifiedDeidentifiedArtifact`.  :func:`run_summary_stage` then checks
that artifact and an explicit human-review mode before it invokes a summary
producer.

This module stores only the de-identified payload in memory and emits hashes,
stable identifiers, and fixed safety metadata in its envelope.  It has no
network or model dependencies.  Summary producers are caller-supplied and are
expected to be local and deterministic; the envelope never copies the source
artifact's original text, mapping, or PII entities into provenance.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

SUMMARY_ENVELOPE_SCHEMA_VERSION = 1

SUMMARY_SAFETY_DISCLAIMER = (
    "This summary is for human review only. It is not a diagnosis, medical "
    "advice, or a substitute for qualified clinical judgment."
)

# Descriptive aliases make the policy easy to discover without allowing a
# caller to provide a weaker disclaimer.
SUMMARY_ENVELOPE_DISCLAIMER = SUMMARY_SAFETY_DISCLAIMER
NON_DIAGNOSTIC_SUMMARY_DISCLAIMER = SUMMARY_SAFETY_DISCLAIMER

SUMMARY_REFUSAL_MISSING_ARTIFACT = "verified_deidentified_artifact_required"
SUMMARY_REFUSAL_INVALID_ARTIFACT = "verified_deidentified_artifact_invalid"
SUMMARY_REFUSAL_HUMAN_REVIEW = "explicit_human_review_mode_required"
SUMMARY_REFUSAL_SUMMARIZER = "local_summary_producer_required"
SUMMARY_REFUSAL_GENERATION = "summary_generation_failed"

_ARTIFACT_TOKEN = object()
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_REASON = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_RESERVED_PROVENANCE = frozenset(
    {
        "artifact_id",
        "content_hash",
        "deidentification_method",
        "source_hash",
        "verification_method",
        "verified",
    }
)
_RAW_PROVENANCE_KEYS = frozenset(
    {
        "address",
        "date",
        "deidentified_text",
        "email",
        "entities",
        "mapping",
        "name",
        "original",
        "original_text",
        "path",
        "phone",
        "pii_entities",
        "raw",
        "source_path",
        "source_text",
        "text",
        "value",
    }
)


class SummaryEnvelopeError(ValueError):
    """Raised when a summary safety invariant cannot be satisfied."""


def _hash_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _field(value: object, name: str) -> object | None:
    try:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)
    except Exception:
        raise SummaryEnvelopeError("summary input metadata is not accessible") from None


def _safe_identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not _SAFE_IDENTIFIER.fullmatch(value):
        raise SummaryEnvelopeError(f"{field_name} must be safe metadata")
    return value


def _safe_hash(value: object) -> str | None:
    if isinstance(value, str) and _SHA256.fullmatch(value):
        return value
    return None


def _safe_provenance(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SummaryEnvelopeError("provenance must be safe metadata")

    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not _SAFE_IDENTIFIER.fullmatch(key):
            raise SummaryEnvelopeError("provenance must be safe metadata")
        if item is None or isinstance(item, bool):
            normalized[key] = item
        elif isinstance(item, int):
            normalized[key] = item
        elif isinstance(item, float) and math.isfinite(item):
            normalized[key] = item
        elif isinstance(item, str) and _SAFE_IDENTIFIER.fullmatch(item):
            normalized[key] = item
        else:
            raise SummaryEnvelopeError("provenance must be safe metadata")
    return {key: normalized[key] for key in sorted(normalized)}


def _safe_additional_provenance(value: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = _safe_provenance(value)
    if set(normalized).intersection(_RESERVED_PROVENANCE):
        raise SummaryEnvelopeError("provenance contains reserved metadata")
    if set(normalized).intersection(_RAW_PROVENANCE_KEYS):
        raise SummaryEnvelopeError("provenance contains source content")
    return normalized


def _safe_envelope_provenance(value: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = _safe_provenance(value)
    if set(normalized).intersection(_RAW_PROVENANCE_KEYS):
        raise SummaryEnvelopeError("provenance contains source content")
    return normalized


def _metadata_from_result(result: object) -> dict[str, Any]:
    metadata = _field(result, "metadata")
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise SummaryEnvelopeError("de-identification metadata is not safe")
    return _safe_additional_provenance(metadata)


def _audit_value(report: object, name: str) -> object | None:
    if report is None:
        return None
    try:
        if isinstance(report, Mapping):
            return report.get(name)
        return getattr(report, name, None)
    except Exception:
        raise SummaryEnvelopeError(
            "de-identification audit verification failed"
        ) from None


def _has_audit_value(report: object, name: str) -> bool:
    try:
        if isinstance(report, Mapping):
            return name in report
        return hasattr(report, name)
    except Exception:
        raise SummaryEnvelopeError(
            "de-identification audit verification failed"
        ) from None


def _verify_audit(report: object, deidentified_text: str) -> str:
    """Return a safe source hash after checking optional audit evidence."""

    if report is None:
        return _hash_text(deidentified_text)

    declared_deidentified_hash = _audit_value(report, "deidentified_text_hash")
    if declared_deidentified_hash is not None:
        deidentified_hash = _safe_hash(declared_deidentified_hash)
        if deidentified_hash != _hash_text(deidentified_text):
            raise SummaryEnvelopeError("de-identification audit verification failed")
    elif _has_audit_value(report, "deidentified_text_hash"):
        raise SummaryEnvelopeError("de-identification audit verification failed")

    matches = _audit_value(report, "repro_hash_matches")
    if callable(matches):
        try:
            verified = matches()
        except Exception:
            raise SummaryEnvelopeError(
                "de-identification audit verification failed"
            ) from None
        if verified is not True:
            raise SummaryEnvelopeError("de-identification audit verification failed")

    declared_source_hash = _audit_value(report, "input_hash")
    if declared_source_hash is None:
        if _has_audit_value(report, "input_hash"):
            raise SummaryEnvelopeError("de-identification audit verification failed")
        return _hash_text(deidentified_text)
    source_hash = _safe_hash(declared_source_hash)
    if source_hash is None:
        raise SummaryEnvelopeError("de-identification audit verification failed")
    return source_hash


def _extract_deidentified_text(result: object) -> str:
    value = _field(result, "deidentified_text")
    if not isinstance(value, str):
        raise SummaryEnvelopeError("de-identification result is not verifiable")
    return value


@dataclass(frozen=True, init=False)
class VerifiedDeidentifiedArtifact:
    """A verified, privacy-safe input boundary for the summary stage.

    Instances can only be created by :func:`verify_deidentified_artifact` or
    :meth:`from_deidentification_result`.  The original text, PII entities, and
    re-identification mapping are intentionally not retained.  ``payload`` is
    the de-identified text passed to a local summary producer, while the
    serialized artifact contains only safe provenance metadata.
    """

    _payload: str = field(repr=False, compare=False)
    artifact_id: str
    source_hash: str
    content_hash: str
    deidentification_method: str
    verification_method: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    verified: Literal[True] = True

    def __init__(
        self,
        *,
        payload: str,
        artifact_id: str,
        source_hash: str,
        content_hash: str,
        deidentification_method: str,
        verification_method: str,
        provenance: Mapping[str, Any] | None = None,
        verified: Literal[True] = True,
        _token: object | None = None,
    ) -> None:
        if _token is not _ARTIFACT_TOKEN:
            raise SummaryEnvelopeError(
                "use verify_deidentified_artifact to create the input artifact"
            )
        if not isinstance(payload, str):
            raise SummaryEnvelopeError("de-identification result is not verifiable")
        if verified is not True:
            raise SummaryEnvelopeError("de-identification result is not verified")
        artifact_id = _safe_identifier(artifact_id, field_name="artifact_id")
        source_hash = _safe_hash(source_hash) or _raise_safe_hash("source_hash")
        content_hash = _safe_hash(content_hash) or _raise_safe_hash("content_hash")
        deidentification_method = _safe_identifier(
            deidentification_method,
            field_name="deidentification_method",
        )
        verification_method = _safe_identifier(
            verification_method,
            field_name="verification_method",
        )
        object.__setattr__(self, "_payload", payload)
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(self, "source_hash", source_hash)
        object.__setattr__(self, "content_hash", content_hash)
        object.__setattr__(self, "deidentification_method", deidentification_method)
        object.__setattr__(self, "verification_method", verification_method)
        object.__setattr__(
            self,
            "provenance",
            _safe_additional_provenance(provenance),
        )
        object.__setattr__(self, "verified", True)

    @property
    def payload(self) -> str:
        """Return the de-identified payload for the local summary producer."""

        return self._payload

    @property
    def is_verified(self) -> Literal[True]:
        """Return the immutable verification marker."""

        return True

    @classmethod
    def from_deidentification_result(
        cls,
        result: object,
        *,
        artifact_id: str | None = None,
        verification_method: str = "deidentification_result",
        provenance: Mapping[str, Any] | None = None,
    ) -> "VerifiedDeidentifiedArtifact":
        """Create an artifact from a structured de-identification result.

        The accepted result must expose a string ``deidentified_text`` and a
        non-empty safe ``method``.  If an audit report is present, its
        de-identified hash and reproducibility check are verified locally.
        Only the de-identified text is retained; original content is never
        copied into this artifact.
        """

        deidentified_text = _extract_deidentified_text(result)
        method = _field(result, "method")
        if not isinstance(method, str) or not _SAFE_IDENTIFIER.fullmatch(method):
            raise SummaryEnvelopeError("de-identification result is not verifiable")

        content_hash = _hash_text(deidentified_text)
        source_hash = _verify_audit(_field(result, "audit_report"), deidentified_text)
        combined_provenance = _metadata_from_result(result)
        if provenance is not None:
            combined_provenance.update(_safe_additional_provenance(provenance))

        resolved_artifact_id = artifact_id or (
            "deidentified-" + content_hash.removeprefix("sha256:")[:24]
        )
        return cls(
            payload=deidentified_text,
            artifact_id=resolved_artifact_id,
            source_hash=source_hash,
            content_hash=content_hash,
            deidentification_method=method,
            verification_method=verification_method,
            provenance=combined_provenance,
            _token=_ARTIFACT_TOKEN,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return provenance without the de-identified payload."""

        return {
            "artifact_id": self.artifact_id,
            "source_hash": self.source_hash,
            "content_hash": self.content_hash,
            "deidentification_method": self.deidentification_method,
            "verification_method": self.verification_method,
            "verified": True,
            "provenance": copy.deepcopy(dict(self.provenance)),
        }


def _raise_safe_hash(field_name: str) -> str:
    raise SummaryEnvelopeError(f"{field_name} must be a sha256 hash")


def verify_deidentified_artifact(
    result: object,
    *,
    artifact_id: str | None = None,
    verification_method: str = "deidentification_result",
    provenance: Mapping[str, Any] | None = None,
) -> VerifiedDeidentifiedArtifact:
    """Verify a de-identification result for use by the summary stage.

    This function is the only supported public artifact factory.  Passing raw
    text, a plain summary mapping, or an unverified object raises a fixed,
    privacy-safe error without echoing the supplied value.
    """

    if isinstance(result, VerifiedDeidentifiedArtifact):
        return result
    return VerifiedDeidentifiedArtifact.from_deidentification_result(
        result,
        artifact_id=artifact_id,
        verification_method=verification_method,
        provenance=provenance,
    )


def _artifact_provenance(
    artifact: VerifiedDeidentifiedArtifact,
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "artifact_id": artifact.artifact_id,
        "source_hash": artifact.source_hash,
        "content_hash": artifact.content_hash,
        "deidentification_method": artifact.deidentification_method,
        "verification_method": artifact.verification_method,
        "verified": True,
    }
    result.update(artifact.provenance)
    result.update(_safe_additional_provenance(extra))
    return {key: result[key] for key in sorted(result)}


def _coerce_reasons(reasons: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for reason in reasons:
        if not isinstance(reason, str) or not _REASON.fullmatch(reason):
            raise SummaryEnvelopeError("refusal reasons must be safe reason codes")
        if reason not in normalized:
            normalized.append(reason)
    return tuple(normalized)


@dataclass(frozen=True)
class SummaryEnvelope:
    """Deterministic, review-gated output from a local summary stage."""

    status: Literal["ready", "refused"]
    summary: Any | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    refusal_reasons: tuple[str, ...] = ()
    disclaimer: str = SUMMARY_SAFETY_DISCLAIMER
    human_review_mode: bool = False
    requires_human_review: Literal[True] = True
    is_diagnostic: Literal[False] = False
    schema_version: int = SUMMARY_ENVELOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.status, str) or self.status not in {"ready", "refused"}:
            raise SummaryEnvelopeError("summary envelope status is invalid")
        if self.disclaimer != SUMMARY_SAFETY_DISCLAIMER:
            raise SummaryEnvelopeError("summary envelope requires the disclaimer")
        if self.human_review_mode is not True and self.human_review_mode is not False:
            raise SummaryEnvelopeError("human_review_mode must be boolean")
        if self.requires_human_review is not True:
            raise SummaryEnvelopeError("summary output always requires human review")
        if self.is_diagnostic is not False:
            raise SummaryEnvelopeError("summary output is never diagnostic")
        if type(self.schema_version) is not int or self.schema_version < 1:
            raise SummaryEnvelopeError("summary envelope schema version is invalid")

        reasons = _coerce_reasons(tuple(self.refusal_reasons))
        if self.status == "ready":
            if reasons or self.human_review_mode is not True:
                raise SummaryEnvelopeError(
                    "ready summary envelope failed its safety gate"
                )
            if not self.provenance:
                raise SummaryEnvelopeError("ready summary envelope lacks provenance")
        elif self.summary is not None:
            raise SummaryEnvelopeError(
                "refused summary envelope cannot contain a summary"
            )

        object.__setattr__(self, "refusal_reasons", reasons)
        object.__setattr__(
            self,
            "provenance",
            _safe_envelope_provenance(self.provenance),
        )

    @property
    def summary_provenance(self) -> Mapping[str, Any]:
        """Alias for the source traceability metadata."""

        return self.provenance

    @property
    def human_review_required(self) -> Literal[True]:
        """Return the immutable human-review requirement."""

        return True

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible envelope."""

        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "summary": (
                _json_safe_value(self.summary) if self.status == "ready" else None
            ),
            "provenance": copy.deepcopy(dict(self.provenance)),
            "refusal_reasons": list(self.refusal_reasons),
            "disclaimer": self.disclaimer,
            "human_review_mode": self.human_review_mode,
            "requires_human_review": True,
            "is_diagnostic": False,
        }

    def to_json(self) -> str:
        """Serialize the envelope with stable separators and key ordering."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SummaryEnvelope":
        """Rebuild and revalidate a serialized envelope."""

        if not isinstance(data, Mapping):
            raise SummaryEnvelopeError("summary envelope must be a mapping")
        try:
            status = data["status"]
            reasons = tuple(data.get("refusal_reasons", ()))
            human_review_mode = data.get("human_review_mode", False)
            requires_human_review = data.get("requires_human_review", True)
            is_diagnostic = data.get("is_diagnostic", False)
            schema_version = data.get("schema_version", SUMMARY_ENVELOPE_SCHEMA_VERSION)
        except (KeyError, TypeError):
            raise SummaryEnvelopeError(
                "summary envelope mapping is malformed"
            ) from None
        return cls(
            status=status,
            summary=data.get("summary"),
            provenance=data.get("provenance", {}),
            refusal_reasons=reasons,
            disclaimer=data.get("disclaimer", ""),
            human_review_mode=human_review_mode,
            requires_human_review=requires_human_review,
            is_diagnostic=is_diagnostic,
            schema_version=schema_version,
        )


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return copy.deepcopy(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SummaryEnvelopeError("summary is not JSON-safe")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SummaryEnvelopeError("summary is not JSON-safe")
            normalized[key] = _json_safe_value(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    try:
        to_dict = getattr(value, "to_dict", None)
    except Exception:
        raise SummaryEnvelopeError("summary is not JSON-safe") from None
    if callable(to_dict):
        try:
            serialized = to_dict()
        except Exception:
            raise SummaryEnvelopeError("summary is not JSON-safe") from None
        return _json_safe_value(serialized)
    raise SummaryEnvelopeError("summary is not JSON-safe")


def _refused_envelope(
    reasons: tuple[str, ...],
    *,
    artifact: VerifiedDeidentifiedArtifact | None = None,
    human_review_mode: bool = False,
    provenance: Mapping[str, Any] | None = None,
) -> SummaryEnvelope:
    try:
        safe_provenance = (
            _artifact_provenance(artifact, provenance)
            if artifact is not None
            else _safe_additional_provenance(provenance)
        )
    except SummaryEnvelopeError:
        safe_provenance = {}
    return SummaryEnvelope(
        status="refused",
        provenance=safe_provenance,
        refusal_reasons=reasons,
        human_review_mode=human_review_mode if human_review_mode is True else False,
    )


def _resolve_artifact(
    artifact: object | None,
    input_artifact: object | None,
) -> VerifiedDeidentifiedArtifact | None:
    if artifact is not None and input_artifact is not None:
        return None
    candidate = input_artifact if input_artifact is not None else artifact
    if isinstance(candidate, VerifiedDeidentifiedArtifact):
        return candidate
    return None


def build_summary_envelope(
    summary: Any | None = None,
    *,
    artifact: object | None = None,
    input_artifact: object | None = None,
    human_review_mode: bool = False,
    provenance: Mapping[str, Any] | None = None,
) -> SummaryEnvelope:
    """Wrap an already-produced summary behind the safety envelope.

    Invalid or missing gates produce a refusal envelope and discard the supplied
    summary.  Use :func:`run_summary_stage` when the summary producer itself
    must be prevented from running before these checks.
    """

    if artifact is not None and input_artifact is not None:
        reasons = [SUMMARY_REFUSAL_INVALID_ARTIFACT]
        if human_review_mode is not True:
            reasons.append(SUMMARY_REFUSAL_HUMAN_REVIEW)
        return _refused_envelope(
            tuple(reasons),
            human_review_mode=human_review_mode,
            provenance=provenance,
        )

    resolved = _resolve_artifact(artifact, input_artifact)
    reasons: list[str] = []
    if artifact is None and input_artifact is None:
        reasons.append(SUMMARY_REFUSAL_MISSING_ARTIFACT)
    elif resolved is None:
        reasons.append(SUMMARY_REFUSAL_INVALID_ARTIFACT)
    if human_review_mode is not True:
        reasons.append(SUMMARY_REFUSAL_HUMAN_REVIEW)
    if reasons:
        return _refused_envelope(
            tuple(reasons),
            artifact=resolved,
            human_review_mode=human_review_mode,
            provenance=provenance,
        )

    assert resolved is not None
    return SummaryEnvelope(
        status="ready",
        summary=summary,
        provenance=_artifact_provenance(resolved, provenance),
        human_review_mode=True,
    )


def run_summary_stage(
    input_artifact: object | None,
    summary_producer: Callable[[str], Any] | None,
    *,
    human_review_mode: bool = False,
    provenance: Mapping[str, Any] | None = None,
) -> SummaryEnvelope:
    """Run a local summary producer only after both safety gates pass.

    The producer receives only the verified de-identified payload.  It is never
    called for a missing/unverified artifact or without ``human_review_mode=True``.
    Producer failures become a fixed refusal code; the original exception is not
    chained so a sensitive producer error cannot escape through this boundary.
    """

    resolved = _resolve_artifact(input_artifact, None)
    if input_artifact is None:
        reasons = [SUMMARY_REFUSAL_MISSING_ARTIFACT]
        if human_review_mode is not True:
            reasons.append(SUMMARY_REFUSAL_HUMAN_REVIEW)
        return _refused_envelope(
            tuple(reasons),
            human_review_mode=human_review_mode,
            provenance=provenance,
        )
    if resolved is None:
        reasons = [SUMMARY_REFUSAL_INVALID_ARTIFACT]
        if human_review_mode is not True:
            reasons.append(SUMMARY_REFUSAL_HUMAN_REVIEW)
        return _refused_envelope(
            tuple(reasons),
            human_review_mode=human_review_mode,
            provenance=provenance,
        )
    if human_review_mode is not True:
        return _refused_envelope(
            (SUMMARY_REFUSAL_HUMAN_REVIEW,),
            artifact=resolved,
            human_review_mode=human_review_mode,
            provenance=provenance,
        )
    if not callable(summary_producer):
        return _refused_envelope(
            (SUMMARY_REFUSAL_SUMMARIZER,),
            artifact=resolved,
            human_review_mode=True,
            provenance=provenance,
        )

    try:
        summary = summary_producer(resolved.payload)
    except Exception:
        return _refused_envelope(
            (SUMMARY_REFUSAL_GENERATION,),
            artifact=resolved,
            human_review_mode=True,
            provenance=provenance,
        )
    return build_summary_envelope(
        summary,
        artifact=resolved,
        human_review_mode=True,
        provenance=provenance,
    )


def validate_summary_envelope(candidate: object) -> SummaryEnvelope:
    """Validate a :class:`SummaryEnvelope` or serialized envelope mapping."""

    if isinstance(candidate, SummaryEnvelope):
        # Reconstruct to defend against object.__setattr__ tampering.
        return SummaryEnvelope.from_dict(candidate.to_dict())
    if isinstance(candidate, Mapping):
        return SummaryEnvelope.from_dict(candidate)
    raise SummaryEnvelopeError("summary envelope must be a SummaryEnvelope or mapping")


__all__ = [
    "NON_DIAGNOSTIC_SUMMARY_DISCLAIMER",
    "SUMMARY_ENVELOPE_DISCLAIMER",
    "SUMMARY_ENVELOPE_SCHEMA_VERSION",
    "SUMMARY_REFUSAL_GENERATION",
    "SUMMARY_REFUSAL_HUMAN_REVIEW",
    "SUMMARY_REFUSAL_INVALID_ARTIFACT",
    "SUMMARY_REFUSAL_MISSING_ARTIFACT",
    "SUMMARY_REFUSAL_SUMMARIZER",
    "SUMMARY_SAFETY_DISCLAIMER",
    "SummaryEnvelope",
    "SummaryEnvelopeError",
    "VerifiedDeidentifiedArtifact",
    "build_summary_envelope",
    "run_summary_stage",
    "validate_summary_envelope",
    "verify_deidentified_artifact",
]
