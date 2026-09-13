"""Deterministic, content-free result envelopes for multimodal providers.

The envelope contains only bounded metadata supplied by a provider boundary.
It never accepts generated text, transcripts, pixel or waveform data, DICOM
values, paths, URLs, prompts, credentials, or arbitrary error messages.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, cast

__all__ = [
    "MAX_PROVIDER_RESULT_JSON_BYTES",
    "PROVIDER_RESULT_SCHEMA_VERSION",
    "ProviderAbstentionCode",
    "ProviderResultEnvelope",
    "ProviderResultError",
    "ProviderResultOutcome",
]

PROVIDER_RESULT_SCHEMA_VERSION: Final = "openmed.multimodal.provider_result.v1"
MAX_PROVIDER_RESULT_JSON_BYTES: Final = 64 * 1024

_MAX_COUNT: Final = (1 << 63) - 1
_MAX_DURATION_MS: Final = 86_400_000.0
_IDENTIFIER: Final = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?\Z")
_DIGEST: Final = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN_IDENTIFIER_PARTS: Final = frozenset(
    {"bearer", "credential", "mrn", "password", "patient", "prompt", "secret", "token"}
)
_COUNT_FIELDS: Final = frozenset(
    {
        "detection_count",
        "frame_count",
        "input_bytes",
        "input_items",
        "output_items",
        "page_count",
        "sample_count",
        "segment_count",
        "token_count",
    }
)
_REQUIRED_FIELDS: Final = frozenset(
    {
        "schema_version",
        "provider_id",
        "model_id",
        "input_digest",
        "outcome",
        "duration_ms",
    }
)
_OPTIONAL_FIELDS: Final = frozenset(
    {"output_digest", "abstention_code", "count_metadata"}
)


class ProviderResultError(ValueError):
    """Raised when provider-result metadata violates the safe envelope."""


class ProviderResultOutcome(str, Enum):
    """Closed set of terminal provider outcomes."""

    SUCCESS = "success"
    ABSTENTION = "abstention"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    VALIDATION_FAILURE = "validation_failure"


class ProviderAbstentionCode(str, Enum):
    """Content-free reasons for a provider abstention."""

    UNSUPPORTED_MEDIA = "unsupported_media"
    MALFORMED_MEDIA = "malformed_media"
    RESOURCE_LIMIT = "resource_limit"
    LOW_QUALITY = "low_quality"
    PHI_UNCERTAINTY = "phi_uncertainty"
    SPEAKER_UNCERTAINTY = "speaker_uncertainty"
    TEMPORAL_INSTABILITY = "temporal_instability"


@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderResultEnvelope:
    """Immutable metadata result returned by a multimodal provider boundary.

    Args:
        provider_id: Bounded lowercase identifier for the provider adapter.
        model_id: Bounded lowercase identifier for the selected model.
        input_digest: Lowercase SHA-256 digest of the provider input.
        outcome: One of the four terminal provider outcomes.
        duration_ms: Finite elapsed time from 0 through 86,400,000 milliseconds.
        count_metadata: Closed, bounded mapping of aggregate counters.
        output_digest: Required only for a successful result.
        abstention_code: Required only for an abstention result.
        schema_version: Exact supported envelope schema identifier.
    """

    provider_id: str
    model_id: str
    input_digest: str
    outcome: ProviderResultOutcome
    duration_ms: float
    count_metadata: Mapping[str, int] = field(default_factory=dict)
    output_digest: str | None = None
    abstention_code: ProviderAbstentionCode | None = None
    schema_version: str = PROVIDER_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != PROVIDER_RESULT_SCHEMA_VERSION
        ):
            raise ProviderResultError("provider result schema is unsupported")

        provider_id = _identifier(self.provider_id, "provider")
        model_id = _identifier(self.model_id, "model")
        input_digest = _digest(self.input_digest, required=True)
        outcome = _outcome(self.outcome)
        duration_ms = _duration(self.duration_ms)
        count_metadata = _counts(self.count_metadata)
        output_digest = _digest(self.output_digest, required=False)
        abstention_code = _abstention_code(self.abstention_code)

        if outcome is ProviderResultOutcome.SUCCESS:
            if output_digest is None:
                raise ProviderResultError("successful result requires output digest")
        elif output_digest is not None:
            raise ProviderResultError("output digest is invalid for result outcome")

        if outcome is ProviderResultOutcome.ABSTENTION:
            if abstention_code is None:
                raise ProviderResultError("abstention result requires abstention code")
        elif abstention_code is not None:
            raise ProviderResultError("abstention code is invalid for result outcome")

        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "input_digest", input_digest)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "duration_ms", duration_ms)
        object.__setattr__(self, "count_metadata", count_metadata)
        object.__setattr__(self, "output_digest", output_digest)
        object.__setattr__(self, "abstention_code", abstention_code)

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh, fixed-shape dictionary containing safe metadata."""

        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "outcome": self.outcome.value,
            "abstention_code": (
                None if self.abstention_code is None else self.abstention_code.value
            ),
            "duration_ms": self.duration_ms,
            "count_metadata": dict(self.count_metadata),
        }

    def to_json(self) -> str:
        """Serialize to deterministic compact JSON."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, payload: object) -> ProviderResultEnvelope:
        """Parse a strict dictionary without reflecting submitted content."""

        if type(payload) is not dict or any(type(key) is not str for key in payload):
            raise ProviderResultError("provider result must be an object")
        fields = frozenset(payload)
        if not _REQUIRED_FIELDS <= fields or not fields <= (
            _REQUIRED_FIELDS | _OPTIONAL_FIELDS
        ):
            raise ProviderResultError("provider result fields are invalid")
        return cls(
            schema_version=payload["schema_version"],
            provider_id=payload["provider_id"],
            model_id=payload["model_id"],
            input_digest=payload["input_digest"],
            output_digest=payload.get("output_digest"),
            outcome=payload["outcome"],
            abstention_code=payload.get("abstention_code"),
            duration_ms=payload["duration_ms"],
            count_metadata=payload.get("count_metadata", {}),
        )

    @classmethod
    def from_json(cls, payload: str) -> ProviderResultEnvelope:
        """Parse bounded strict JSON with duplicate-key rejection."""

        if type(payload) is not str or len(payload) > MAX_PROVIDER_RESULT_JSON_BYTES:
            raise ProviderResultError("provider result JSON is invalid")
        try:
            if len(payload.encode("utf-8")) > MAX_PROVIDER_RESULT_JSON_BYTES:
                raise ProviderResultError("provider result JSON is invalid")
            decoded = json.loads(
                payload,
                object_pairs_hook=_strict_object,
                parse_int=_parse_integer,
                parse_constant=_reject_constant,
            )
        except (UnicodeError, ValueError, RecursionError):
            pass
        else:
            return cls.from_dict(decoded)
        raise ProviderResultError("provider result JSON is invalid")


def _identifier(value: object, kind: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ProviderResultError(f"{kind} identifier is invalid")
    parts = frozenset(re.split(r"[._-]+", value))
    if parts & _FORBIDDEN_IDENTIFIER_PARTS:
        raise ProviderResultError(f"{kind} identifier is invalid")
    return value


def _digest(value: object, *, required: bool) -> str | None:
    if value is None and not required:
        return None
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ProviderResultError("SHA-256 digest is invalid")
    return value


def _outcome(value: object) -> ProviderResultOutcome:
    try:
        if not isinstance(value, (str, ProviderResultOutcome)):
            raise ValueError
        return ProviderResultOutcome(value)
    except ValueError:
        pass
    raise ProviderResultError("provider result outcome is unsupported")


def _abstention_code(value: object) -> ProviderAbstentionCode | None:
    if value is None:
        return None
    try:
        if not isinstance(value, (str, ProviderAbstentionCode)):
            raise ValueError
        return ProviderAbstentionCode(value)
    except ValueError:
        pass
    raise ProviderResultError("provider abstention code is unsupported")


def _duration(value: object) -> float:
    if type(value) not in (int, float):
        raise ProviderResultError("provider duration is invalid")
    if not 0 <= cast(int | float, value) <= _MAX_DURATION_MS:
        raise ProviderResultError("provider duration is invalid")
    duration = float(cast(int | float, value))
    if not math.isfinite(duration) or not 0.0 <= duration <= _MAX_DURATION_MS:
        raise ProviderResultError("provider duration is invalid")
    return duration


def _counts(value: object) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise ProviderResultError("provider count metadata is invalid")
    try:
        counts = dict(value)
    except Exception:
        raise ProviderResultError("provider count metadata is invalid") from None
    if any(type(name) is not str or name not in _COUNT_FIELDS for name in counts):
        raise ProviderResultError("provider count metadata contains unsupported fields")
    if any(
        type(count) is not int or not 0 <= count <= _MAX_COUNT
        for count in counts.values()
    ):
        raise ProviderResultError("provider count metadata is invalid")
    return MappingProxyType(dict(sorted(counts.items())))


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProviderResultError("provider result contains duplicate fields")
        result[key] = value
    return result


def _parse_integer(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        raise ProviderResultError("provider result integer is invalid")
    parsed = int(value)
    if not -_MAX_COUNT <= parsed <= _MAX_COUNT:
        raise ProviderResultError("provider result integer is invalid")
    return parsed


def _reject_constant(value: str) -> None:
    raise ProviderResultError("provider result number is invalid")
