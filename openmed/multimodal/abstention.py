"""Stable, metadata-only abstention records for multimodal pipelines.

The boundary in this module deliberately accepts only a stage and a reason
code.  It has no free-text field, so callers cannot accidentally serialize OCR
text, transcripts, DICOM values, file paths, URLs, or model prompts while
explaining why processing stopped.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

__all__ = [
    "ABSTENTION_SCHEMA_VERSION",
    "AbstentionReason",
    "AbstentionRecord",
    "AbstentionStage",
    "AbstentionValidationError",
]


ABSTENTION_SCHEMA_VERSION: Final = 1


class AbstentionValidationError(ValueError):
    """Raised when an abstention record fails strict, PHI-safe validation."""


class AbstentionStage(str, Enum):
    """Pipeline stage at which multimodal processing stopped."""

    PREFLIGHT = "preflight"
    DECODE = "decode"
    INFERENCE = "inference"
    POST_PROCESS = "post_process"


class AbstentionReason(str, Enum):
    """Stable reason why a multimodal pipeline declined to continue."""

    UNSUPPORTED_MEDIA = "unsupported_media"
    MALFORMED_MEDIA = "malformed_media"
    RESOURCE_LIMIT = "resource_limit"
    LOW_QUALITY = "low_quality"
    PHI_UNCERTAINTY = "phi_uncertainty"
    SPEAKER_UNCERTAINTY = "speaker_uncertainty"
    TEMPORAL_INSTABILITY = "temporal_instability"
    PROVIDER_UNAVAILABLE = "provider_unavailable"


_ALLOWED_REASONS: Final = {
    AbstentionStage.PREFLIGHT: frozenset(
        {
            AbstentionReason.UNSUPPORTED_MEDIA,
            AbstentionReason.RESOURCE_LIMIT,
            AbstentionReason.PROVIDER_UNAVAILABLE,
        }
    ),
    AbstentionStage.DECODE: frozenset(
        {
            AbstentionReason.MALFORMED_MEDIA,
            AbstentionReason.RESOURCE_LIMIT,
            AbstentionReason.LOW_QUALITY,
        }
    ),
    AbstentionStage.INFERENCE: frozenset(
        {
            AbstentionReason.RESOURCE_LIMIT,
            AbstentionReason.LOW_QUALITY,
            AbstentionReason.PHI_UNCERTAINTY,
            AbstentionReason.SPEAKER_UNCERTAINTY,
            AbstentionReason.TEMPORAL_INSTABILITY,
            AbstentionReason.PROVIDER_UNAVAILABLE,
        }
    ),
    AbstentionStage.POST_PROCESS: frozenset(
        {
            AbstentionReason.RESOURCE_LIMIT,
            AbstentionReason.LOW_QUALITY,
            AbstentionReason.PHI_UNCERTAINTY,
            AbstentionReason.SPEAKER_UNCERTAINTY,
            AbstentionReason.TEMPORAL_INSTABILITY,
        }
    ),
}

_REQUIRED_FIELDS: Final = frozenset({"schema_version", "stage", "reason"})


def _stage(value: object) -> AbstentionStage:
    try:
        return AbstentionStage(value)
    except Exception:
        raise AbstentionValidationError("stage is unsupported") from None


def _reason(value: object) -> AbstentionReason:
    try:
        return AbstentionReason(value)
    except Exception:
        raise AbstentionValidationError("reason is unsupported") from None


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise AbstentionValidationError("payload fields are invalid")
    return fields


@dataclass(frozen=True, slots=True)
class AbstentionRecord:
    """A deterministic explanation that contains no source-media content.

    Args:
        stage: Pipeline stage at which processing stopped.
        reason: Stable reason code valid for that stage.
        schema_version: Serialization contract version. Only version 1 is
            currently accepted.
    """

    stage: AbstentionStage
    reason: AbstentionReason
    schema_version: int = ABSTENTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        stage = _stage(self.stage)
        reason = _reason(self.reason)
        if type(self.schema_version) is not int or (
            self.schema_version != ABSTENTION_SCHEMA_VERSION
        ):
            raise AbstentionValidationError("schema_version is unsupported")
        if reason not in _ALLOWED_REASONS[stage]:
            raise AbstentionValidationError("reason is not valid for stage")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "reason", reason)

    def to_dict(self) -> dict[str, int | str]:
        """Return the fixed-shape metadata-only representation."""

        return {
            "schema_version": self.schema_version,
            "stage": self.stage.value,
            "reason": self.reason.value,
        }

    def to_json(self) -> str:
        """Serialize with deterministic key order and no insignificant space."""

        return json.dumps(self.to_dict(), ensure_ascii=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AbstentionRecord":
        """Parse a strict mapping without reflecting submitted values."""

        if not isinstance(payload, Mapping):
            raise AbstentionValidationError("payload must be an object")
        try:
            fields = dict(payload)
        except Exception:
            raise AbstentionValidationError("payload could not be read") from None
        if frozenset(fields) != _REQUIRED_FIELDS:
            raise AbstentionValidationError("payload fields are invalid")
        return cls(
            schema_version=fields["schema_version"],
            stage=fields["stage"],
            reason=fields["reason"],
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "AbstentionRecord":
        """Parse strict JSON without including rejected content in errors."""

        try:
            parsed = json.loads(payload, object_pairs_hook=_strict_object)
        except AbstentionValidationError:
            raise
        except Exception:
            raise AbstentionValidationError("payload is not valid JSON") from None
        return cls.from_dict(parsed)
