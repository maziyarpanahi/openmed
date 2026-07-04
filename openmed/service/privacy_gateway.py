"""Privacy gateway for redacting before external LLM calls.

The gateway keeps the reversible placeholder map in process memory for one
request, forwards only the redacted prompt to the configured transport, and
restores placeholders in the returned model text after validating that the
model did not invent or mangle gateway tokens.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

SERVICE_PRIVACY_GATEWAY_ENDPOINT_ENV_VAR = "OPENMED_SERVICE_PRIVACY_GATEWAY_ENDPOINT"
DEFAULT_PRIVACY_GATEWAY_POLICY = "strict"
DEFAULT_PRIVACY_GATEWAY_MIN_CONFIDENCE = 0.85
_ZERO_HASH = "0" * 64
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_PLACEHOLDER_PATTERN = re.compile(r"<<OPENMED_PHI_[A-Z0-9_]+_[0-9A-F]{8}_[0-9]{6,}>>")
_PLACEHOLDER_CANDIDATE_PATTERN = re.compile(r"<<OPENMED_PHI_[^<>\s]+>>")
_PLACEHOLDER_FRAGMENT_PATTERN = re.compile(r"OPENMED[_-]PHI", re.IGNORECASE)

EntityExtractor = Callable[..., Any]
ExternalLLMTransport = Callable[..., Any]


class PrivacyGatewayError(ValueError):
    """Base class for privacy-gateway fail-closed errors."""

    error_code = "privacy_gateway_error"
    reason_code = "privacy_gateway_error"
    audit_status = "blocked"

    def __init__(self, message: str, *, reason_code: Optional[str] = None) -> None:
        super().__init__(message)
        if reason_code is not None:
            self.reason_code = reason_code


class PrivacyPolicyViolation(PrivacyGatewayError):
    """Raised when policy blocks forwarding before external egress."""

    error_code = "privacy_gateway_blocked"
    reason_code = "policy_violation"


class PrivacyTripwireViolation(PrivacyPolicyViolation):
    """Raised when the independent outbound scan finds residual PHI."""

    reason_code = "outbound_tripwire_detected"


class PrivacyReidentificationError(PrivacyGatewayError):
    """Raised when the external model response has invalid placeholders."""

    error_code = "privacy_gateway_reidentification_error"
    reason_code = "reidentification_failed"
    audit_status = "rejected"


class PrivacyTransportError(RuntimeError):
    """Raised when the configured external transport fails safely."""

    error_code = "privacy_gateway_transport_error"
    reason_code = "transport_failed"

    def __init__(self, message: str, *, reason_code: Optional[str] = None) -> None:
        super().__init__(message)
        if reason_code is not None:
            self.reason_code = reason_code


class PrivacyGatewayConfigurationError(RuntimeError):
    """Raised when service egress is not operator-configured."""

    error_code = "privacy_gateway_not_configured"
    reason_code = "missing_external_endpoint"


@dataclass(frozen=True)
class PrivacyGatewayEntity:
    """Normalized PII span consumed by the gateway redactor."""

    label: str
    start: int
    end: int
    confidence: float

    @property
    def category(self) -> str:
        """Return a stable audit/placeholder category."""
        return normalize_entity_category(self.label)


@dataclass(frozen=True)
class PrivacyGatewayPolicy:
    """Fail-closed policy for one privacy-gateway request."""

    name: str = DEFAULT_PRIVACY_GATEWAY_POLICY
    min_confidence: float = DEFAULT_PRIVACY_GATEWAY_MIN_CONFIDENCE
    detector_confidence_floor: float = 0.0
    disallowed_entity_categories: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.min_confidence) <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if not 0.0 <= float(self.detector_confidence_floor) <= 1.0:
            raise ValueError("detector_confidence_floor must be between 0.0 and 1.0")
        categories = frozenset(
            normalize_entity_category(category)
            for category in self.disallowed_entity_categories
        )
        object.__setattr__(self, "disallowed_entity_categories", categories)

    def enforce(self, entities: Sequence[PrivacyGatewayEntity]) -> None:
        """Block forwarding when any span violates confidence or category policy."""
        for entity in entities:
            if entity.confidence < self.min_confidence:
                raise PrivacyPolicyViolation(
                    "Redaction confidence fell below the gateway threshold",
                    reason_code="redaction_confidence_below_threshold",
                )
            if entity.category in self.disallowed_entity_categories:
                raise PrivacyPolicyViolation(
                    "A disallowed entity category was detected",
                    reason_code="disallowed_entity_category",
                )

    def to_audit_dict(self) -> dict[str, Any]:
        """Return policy metadata that is safe for the PHI-free audit trail."""
        return {
            "name": self.name,
            "min_confidence": float(self.min_confidence),
            "detector_confidence_floor": float(self.detector_confidence_floor),
            "disallowed_entity_categories": sorted(self.disallowed_entity_categories),
        }


@dataclass(frozen=True)
class RedactionSession:
    """Request-scoped reversible redaction state."""

    request_id: str
    redacted_text: str
    placeholder_map: Mapping[str, str]
    entity_counts: Mapping[str, int]
    placeholder_hashes: tuple[str, ...]


@dataclass(frozen=True)
class PrivacyGatewayResult:
    """Successful privacy-gateway round-trip result."""

    request_id: str
    redacted_prompt: str
    external_response: str
    reidentified_text: str
    entity_counts: Mapping[str, int]
    placeholder_hashes: tuple[str, ...]
    audit_record_hash: str
    audit_verified: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable response without the re-identification map."""
        return {
            "request_id": self.request_id,
            "redacted_prompt": self.redacted_prompt,
            "external_response": self.external_response,
            "reidentified_text": self.reidentified_text,
            "entity_counts": dict(self.entity_counts),
            "placeholder_hashes": list(self.placeholder_hashes),
            "audit": {
                "record_hash": self.audit_record_hash,
                "verified": self.audit_verified,
            },
        }


@dataclass(frozen=True)
class PrivacyGatewayTransportResponse:
    """Typed response object for custom transports."""

    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class InMemoryReidentificationStore:
    """Thread-safe request map store that never persists to disk."""

    def __init__(self) -> None:
        self._maps: dict[str, dict[str, str]] = {}
        self._lock = threading.RLock()

    def save(self, request_id: str, placeholder_map: Mapping[str, str]) -> None:
        """Save a request-scoped placeholder map in process memory."""
        with self._lock:
            self._maps[request_id] = dict(placeholder_map)

    def get(self, request_id: str) -> dict[str, str]:
        """Return a copy of the request map."""
        with self._lock:
            try:
                return dict(self._maps[request_id])
            except KeyError as exc:
                raise PrivacyReidentificationError(
                    "No re-identification map exists for this request",
                    reason_code="missing_reidentification_map",
                ) from exc

    def remove(self, request_id: str) -> None:
        """Delete a request map after completion or failure."""
        with self._lock:
            self._maps.pop(request_id, None)

    def contains(self, request_id: str) -> bool:
        """Return whether a request map is still present."""
        with self._lock:
            return request_id in self._maps


class PrivacyGatewayAuditTrail:
    """Append-only, hash-chained audit trail with PHI-free records."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    def append(
        self,
        *,
        request_id: str,
        status: str,
        policy: PrivacyGatewayPolicy,
        entity_counts: Mapping[str, int],
        placeholder_hashes: Sequence[str],
        reason_code: Optional[str] = None,
        transport: Optional[Mapping[str, Any]] = None,
        residual_counts: Optional[Mapping[str, int]] = None,
        response_placeholder_count: int = 0,
    ) -> dict[str, Any]:
        """Append one safe audit record and return a copy."""
        record = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "reason_code": reason_code,
            "policy": policy.to_audit_dict(),
            "entity_counts": dict(sorted(entity_counts.items())),
            "placeholder_hashes": list(placeholder_hashes),
            "residual_counts": dict(sorted((residual_counts or {}).items())),
            "response_placeholder_count": int(response_placeholder_count),
            "transport": dict(transport or {}),
        }
        with self._lock:
            previous_hash = (
                self._records[-1]["record_hash"] if self._records else _ZERO_HASH
            )
            record["previous_hash"] = previous_hash
            record["record_hash"] = _record_hash(record)
            self._records.append(record)
            return dict(record)

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        """Return copies of the audit records."""
        with self._lock:
            return tuple(dict(record) for record in self._records)

    def to_json(self) -> str:
        """Serialize audit records deterministically."""
        return json.dumps(
            list(self.records),
            separators=(",", ":"),
            sort_keys=True,
        )

    def verify(self) -> bool:
        """Return whether the in-memory chain is intact."""
        return self.verify_records(self.records)

    @staticmethod
    def verify_records(records: Sequence[Mapping[str, Any]]) -> bool:
        """Verify a supplied audit-record sequence."""
        previous_hash = _ZERO_HASH
        for record in records:
            current = dict(record)
            if current.get("previous_hash") != previous_hash:
                return False
            if current.get("record_hash") != _record_hash(current):
                return False
            previous_hash = str(current["record_hash"])
        return True

    def contains_plaintext(self, values: Sequence[str]) -> bool:
        """Return whether any plaintext value appears in the audit JSON."""
        audit_json = self.to_json()
        return any(
            value and len(value) >= 3 and value in audit_json for value in values
        )


@dataclass(frozen=True)
class HttpExternalLLMTransport:
    """HTTP transport for the operator-configured external LLM endpoint."""

    endpoint: str
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "HttpExternalLLMTransport":
        """Create a transport from service environment configuration."""
        endpoint = os.getenv(SERVICE_PRIVACY_GATEWAY_ENDPOINT_ENV_VAR, "").strip()
        if not endpoint:
            raise PrivacyGatewayConfigurationError(
                f"{SERVICE_PRIVACY_GATEWAY_ENDPOINT_ENV_VAR} is not configured"
            )
        return cls(endpoint=endpoint)

    def __call__(
        self,
        redacted_text: str,
        *,
        request_id: str,
        metadata: Mapping[str, Any],
    ) -> str:
        """POST a redacted prompt and return the model text."""
        import httpx

        try:
            response = httpx.post(
                self.endpoint,
                json={
                    "prompt": redacted_text,
                    "request_id": request_id,
                    "metadata": dict(metadata),
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except Exception as exc:
            raise PrivacyTransportError("External LLM transport failed") from exc

        try:
            payload = response.json()
        except ValueError:
            return response.text
        return coerce_transport_response_text(payload)

    def audit_metadata(self) -> dict[str, str]:
        """Return safe transport metadata for audit records."""
        return {
            "type": "http",
            "endpoint_hash": sha256_text(self.endpoint),
        }


class PrivacyGateway:
    """Redact, forward to a transport, and re-identify one LLM response."""

    def __init__(
        self,
        *,
        transport: ExternalLLMTransport,
        extractor: Optional[EntityExtractor] = None,
        tripwire_extractor: Optional[EntityExtractor] = None,
        audit_trail: Optional[PrivacyGatewayAuditTrail] = None,
        store: Optional[InMemoryReidentificationStore] = None,
    ) -> None:
        self.transport = transport
        self.extractor = extractor or default_entity_extractor
        self.tripwire_extractor = tripwire_extractor or safety_sweep_tripwire
        self.audit_trail = audit_trail or PrivacyGatewayAuditTrail()
        self.store = store or InMemoryReidentificationStore()

    def complete(
        self,
        text: str,
        *,
        policy: Optional[PrivacyGatewayPolicy] = None,
        request_id: Optional[str] = None,
        model_name: str = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
        use_smart_merging: bool = True,
        lang: str = "en",
        normalize_accents: Optional[bool] = None,
        detector_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> PrivacyGatewayResult:
        """Run a complete privacy-gateway request."""
        active_policy = policy or PrivacyGatewayPolicy()
        active_request_id = coerce_request_id(request_id)
        entities: list[PrivacyGatewayEntity] = []
        session: Optional[RedactionSession] = None
        response_placeholder_count = 0
        residual_counts: Mapping[str, int] = {}

        try:
            entities = self.detect_entities(
                text,
                policy=active_policy,
                model_name=model_name,
                use_smart_merging=use_smart_merging,
                lang=lang,
                normalize_accents=normalize_accents,
                detector_kwargs=detector_kwargs,
            )
            active_policy.enforce(entities)
            session = redact_text(text, entities, request_id=active_request_id)
            self.store.save(active_request_id, session.placeholder_map)

            residual_entities = self.scan_outbound(
                session.redacted_text,
                model_name=model_name,
                lang=lang,
                normalize_accents=normalize_accents,
                detector_kwargs=detector_kwargs,
            )
            if residual_entities:
                residual_counts = entity_counts(residual_entities)
                raise PrivacyTripwireViolation(
                    "Outbound tripwire detected residual PHI",
                    reason_code="outbound_tripwire_detected",
                )

            external_response = self.call_transport(
                session.redacted_text,
                request_id=active_request_id,
                policy=active_policy,
                entity_counts=session.entity_counts,
            )
            response_placeholder_count = len(
                _PLACEHOLDER_CANDIDATE_PATTERN.findall(external_response)
            )
            reidentified = reidentify_placeholders(
                external_response,
                self.store.get(active_request_id),
            )
            record = self.audit_trail.append(
                request_id=active_request_id,
                status="forwarded",
                policy=active_policy,
                entity_counts=session.entity_counts,
                placeholder_hashes=session.placeholder_hashes,
                transport=transport_audit_metadata(self.transport),
                response_placeholder_count=response_placeholder_count,
            )
            return PrivacyGatewayResult(
                request_id=active_request_id,
                redacted_prompt=session.redacted_text,
                external_response=external_response,
                reidentified_text=reidentified,
                entity_counts=session.entity_counts,
                placeholder_hashes=session.placeholder_hashes,
                audit_record_hash=str(record["record_hash"]),
                audit_verified=self.audit_trail.verify(),
            )
        except PrivacyGatewayError as exc:
            self.audit_trail.append(
                request_id=active_request_id,
                status=exc.audit_status,
                policy=active_policy,
                entity_counts=entity_counts(entities),
                placeholder_hashes=(
                    session.placeholder_hashes if session is not None else ()
                ),
                reason_code=exc.reason_code,
                transport=transport_audit_metadata(self.transport),
                residual_counts=residual_counts,
                response_placeholder_count=response_placeholder_count,
            )
            raise
        except PrivacyTransportError as exc:
            self.audit_trail.append(
                request_id=active_request_id,
                status="transport_error",
                policy=active_policy,
                entity_counts=(
                    session.entity_counts
                    if session is not None
                    else entity_counts(entities)
                ),
                placeholder_hashes=(
                    session.placeholder_hashes if session is not None else ()
                ),
                reason_code=exc.reason_code,
                transport=transport_audit_metadata(self.transport),
                residual_counts=residual_counts,
                response_placeholder_count=response_placeholder_count,
            )
            raise
        finally:
            self.store.remove(active_request_id)

    def detect_entities(
        self,
        text: str,
        *,
        policy: PrivacyGatewayPolicy,
        model_name: str,
        use_smart_merging: bool,
        lang: str,
        normalize_accents: Optional[bool],
        detector_kwargs: Optional[Mapping[str, Any]],
    ) -> list[PrivacyGatewayEntity]:
        """Run the primary local detector and normalize spans."""
        result = self.extractor(
            text,
            model_name=model_name,
            confidence_threshold=policy.detector_confidence_floor,
            use_smart_merging=use_smart_merging,
            lang=lang,
            normalize_accents=normalize_accents,
            **dict(detector_kwargs or {}),
        )
        return coerce_gateway_entities(result, text)

    def scan_outbound(
        self,
        redacted_text: str,
        *,
        model_name: str,
        lang: str,
        normalize_accents: Optional[bool],
        detector_kwargs: Optional[Mapping[str, Any]],
    ) -> list[PrivacyGatewayEntity]:
        """Run the independent outbound tripwire detector pass."""
        result = self.tripwire_extractor(
            redacted_text,
            model_name=model_name,
            confidence_threshold=0.0,
            use_smart_merging=True,
            lang=lang,
            normalize_accents=normalize_accents,
            **dict(detector_kwargs or {}),
        )
        return coerce_gateway_entities(result, redacted_text)

    def call_transport(
        self,
        redacted_text: str,
        *,
        request_id: str,
        policy: PrivacyGatewayPolicy,
        entity_counts: Mapping[str, int],
    ) -> str:
        """Call the configured transport and normalize its response text."""
        try:
            response = self.transport(
                redacted_text,
                request_id=request_id,
                metadata={
                    "policy": policy.to_audit_dict(),
                    "entity_counts": dict(entity_counts),
                },
            )
        except PrivacyTransportError:
            raise
        except Exception as exc:
            raise PrivacyTransportError("External LLM transport failed") from exc
        return coerce_transport_response_text(response)


def default_entity_extractor(text: str, **kwargs: Any) -> Any:
    """Run the default OpenMed PII detector."""
    import openmed

    return openmed.extract_pii(text, **kwargs)


def safety_sweep_tripwire(text: str, *, lang: str = "en", **_: Any) -> list[Any]:
    """Independent deterministic tripwire used for outbound residual PHI."""
    from openmed.core.safety_sweep import safety_sweep

    return safety_sweep(text, [], lang=lang)


def redact_text(
    text: str,
    entities: Sequence[PrivacyGatewayEntity],
    *,
    request_id: str,
) -> RedactionSession:
    """Replace detected PHI spans with collision-free placeholders."""
    ordered = validate_gateway_entities(text, entities)
    pieces: list[str] = []
    cursor = 0
    placeholder_map: dict[str, str] = {}
    placeholder_hashes: list[str] = []
    counts: Counter[str] = Counter()

    for index, entity in enumerate(ordered, start=1):
        placeholder = build_placeholder_token(
            entity.category,
            request_id=request_id,
            index=index,
        )
        if placeholder in placeholder_map:
            raise PrivacyPolicyViolation(
                "Placeholder collision detected",
                reason_code="placeholder_collision",
            )
        pieces.append(text[cursor : entity.start])
        pieces.append(placeholder)
        placeholder_map[placeholder] = text[entity.start : entity.end]
        placeholder_hashes.append(sha256_text(placeholder))
        counts[entity.category] += 1
        cursor = entity.end

    pieces.append(text[cursor:])
    return RedactionSession(
        request_id=request_id,
        redacted_text="".join(pieces),
        placeholder_map=placeholder_map,
        entity_counts=dict(sorted(counts.items())),
        placeholder_hashes=tuple(placeholder_hashes),
    )


def reidentify_placeholders(text: str, placeholder_map: Mapping[str, str]) -> str:
    """Replace known placeholders and reject hallucinated or mangled tokens."""
    candidates = _PLACEHOLDER_CANDIDATE_PATTERN.findall(text)
    unknown = [
        placeholder for placeholder in candidates if placeholder not in placeholder_map
    ]
    if unknown:
        raise PrivacyReidentificationError(
            "External response contained an unknown privacy placeholder",
            reason_code="unknown_placeholder",
        )

    scrubbed = _PLACEHOLDER_CANDIDATE_PATTERN.sub("", text)
    if _PLACEHOLDER_FRAGMENT_PATTERN.search(scrubbed):
        raise PrivacyReidentificationError(
            "External response contained a mangled privacy placeholder",
            reason_code="mangled_placeholder",
        )

    return _PLACEHOLDER_CANDIDATE_PATTERN.sub(
        lambda match: placeholder_map[match.group(0)],
        text,
    )


def build_placeholder_token(category: str, *, request_id: str, index: int) -> str:
    """Return a deterministic, PHI-free placeholder token."""
    if index < 1:
        raise ValueError("index must be positive")
    request_digest = sha256_text(request_id)[:8].upper()
    safe_category = normalize_entity_category(category)
    return f"<<OPENMED_PHI_{safe_category}_{request_digest}_{index:06d}>>"


def coerce_gateway_entities(
    result: Any, source_text: str
) -> list[PrivacyGatewayEntity]:
    """Normalize detector output into gateway entities."""
    raw_entities = result
    if hasattr(result, "entities"):
        raw_entities = getattr(result, "entities")
    if raw_entities is None:
        return []

    entities: list[PrivacyGatewayEntity] = []
    for raw in raw_entities:
        label = _entity_value(raw, "label", "entity_type", "entity_group", "entity")
        start = _entity_value(raw, "start")
        end = _entity_value(raw, "end")
        confidence = _entity_value(raw, "confidence", "score")
        if start is None or end is None:
            raise PrivacyPolicyViolation(
                "Detected PHI span did not include character offsets",
                reason_code="missing_entity_offsets",
            )
        try:
            start_int = int(start)
            end_int = int(end)
            confidence_float = float(0.0 if confidence is None else confidence)
        except (TypeError, ValueError) as exc:
            raise PrivacyPolicyViolation(
                "Detected PHI span metadata was invalid",
                reason_code="invalid_entity_offsets",
            ) from exc
        entities.append(
            PrivacyGatewayEntity(
                label=str(label or "UNKNOWN"),
                start=start_int,
                end=end_int,
                confidence=confidence_float,
            )
        )
    return validate_gateway_entities(source_text, entities)


def validate_gateway_entities(
    text: str,
    entities: Sequence[PrivacyGatewayEntity],
) -> list[PrivacyGatewayEntity]:
    """Validate span offsets and reject overlaps before redaction."""
    ordered = sorted(entities, key=lambda entity: (entity.start, entity.end))
    previous_end = 0
    for entity in ordered:
        if entity.start < 0 or entity.end <= entity.start or entity.end > len(text):
            raise PrivacyPolicyViolation(
                "Detected PHI span offsets were outside the prompt",
                reason_code="invalid_entity_offsets",
            )
        if entity.start < previous_end:
            raise PrivacyPolicyViolation(
                "Detected PHI spans overlapped",
                reason_code="overlapping_entity_spans",
            )
        previous_end = entity.end
    return ordered


def entity_counts(entities: Sequence[PrivacyGatewayEntity]) -> dict[str, int]:
    """Return safe entity counts by normalized category."""
    counts: Counter[str] = Counter(entity.category for entity in entities)
    return dict(sorted(counts.items()))


def normalize_entity_category(value: Any) -> str:
    """Normalize detector labels for placeholders and audit counters."""
    category = str(value or "UNKNOWN").strip().upper()
    if len(category) > 2 and category[1] == "-" and category[0] in {"B", "I", "E", "S"}:
        category = category[2:]
    category = re.sub(r"[^A-Z0-9]+", "_", category).strip("_")
    return category or "UNKNOWN"


def coerce_request_id(request_id: Optional[str]) -> str:
    """Return a generated or validated request id for in-memory map lookup."""
    if request_id is None:
        return str(uuid.uuid4())
    if not _REQUEST_ID_PATTERN.match(request_id):
        raise ValueError("request_id contains unsupported characters")
    return request_id


def coerce_transport_response_text(response: Any) -> str:
    """Extract text from common transport response shapes."""
    if isinstance(response, PrivacyGatewayTransportResponse):
        return response.text
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        for key in ("text", "response", "content", "completion", "output"):
            value = response.get(key)
            if isinstance(value, str):
                return value
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                message = first.get("message")
                if isinstance(message, Mapping) and isinstance(
                    message.get("content"), str
                ):
                    return str(message["content"])
                if isinstance(first.get("text"), str):
                    return str(first["text"])
    raise PrivacyTransportError(
        "External LLM transport did not return text",
        reason_code="invalid_transport_response",
    )


def transport_audit_metadata(transport: Any) -> dict[str, Any]:
    """Return safe transport metadata without raw endpoint values."""
    metadata = getattr(transport, "audit_metadata", None)
    if callable(metadata):
        value = metadata()
        if isinstance(value, Mapping):
            return dict(value)
    return {"type": transport.__class__.__name__}


def sha256_text(value: str) -> str:
    """Return the SHA-256 hex digest for a string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record_hash(record: Mapping[str, Any]) -> str:
    material = {key: value for key, value in record.items() if key != "record_hash"}
    return sha256_text(
        json.dumps(material, separators=(",", ":"), sort_keys=True, default=str)
    )


def _entity_value(entity: Any, *names: str) -> Any:
    if isinstance(entity, Mapping):
        for name in names:
            if name in entity:
                return entity[name]
        return None
    for name in names:
        value = getattr(entity, name, None)
        if value is not None:
            return value
    return None


__all__ = [
    "DEFAULT_PRIVACY_GATEWAY_MIN_CONFIDENCE",
    "DEFAULT_PRIVACY_GATEWAY_POLICY",
    "HttpExternalLLMTransport",
    "InMemoryReidentificationStore",
    "PrivacyGateway",
    "PrivacyGatewayAuditTrail",
    "PrivacyGatewayConfigurationError",
    "PrivacyGatewayEntity",
    "PrivacyGatewayError",
    "PrivacyGatewayPolicy",
    "PrivacyGatewayResult",
    "PrivacyGatewayTransportResponse",
    "PrivacyPolicyViolation",
    "PrivacyReidentificationError",
    "PrivacyTransportError",
    "PrivacyTripwireViolation",
    "SERVICE_PRIVACY_GATEWAY_ENDPOINT_ENV_VAR",
    "build_placeholder_token",
    "coerce_gateway_entities",
    "redact_text",
    "reidentify_placeholders",
]
