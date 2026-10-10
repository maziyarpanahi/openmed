"""Versioned, patient-value-free notices bound to multimodal review outputs.

Review wrappers carry only an output digest. Protected clinical content stays
with its producer; this layer does not invent measurement or draft schemas.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import MISSING, dataclass, fields
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, cast

__all__ = [
    "NOTICE_CATALOG",
    "NOTICE_RESULT_TYPES",
    "NoticeKind",
    "MultimodalNotice",
    "MultimodalNoticeError",
    "NoticeBoundOutput",
    "MeasurementReviewResult",
    "VisualDescriptionResult",
    "DraftReviewResult",
    "validate_notice_registry",
]

MAX_NOTICE_RESULT_JSON_BYTES = 64 * 1024


class MultimodalNoticeError(ValueError):
    """Value-free failure of the multimodal notice or review contract."""


class NoticeKind(str, Enum):
    """Clinical meaning of a review-only multimodal output."""

    MEASUREMENT = "measurement_for_review"
    VISUAL_DESCRIPTION = "visual_description"
    DRAFT = "draft_for_review"


# Published IDs are immutable: revised wording must get a new versioned ID.
_CATALOG = MappingProxyType(
    {
        NoticeKind.MEASUREMENT: (
            "openmed.multimodal.measurement_for_review.v1",
            "Measurement candidate for clinician review only. This output is not a "
            "diagnosis. A qualified clinician must independently review the source "
            "and explicitly confirm before any consequential use. This output must "
            "never automatically trigger a clinical decision.",
        ),
        NoticeKind.VISUAL_DESCRIPTION: (
            "openmed.multimodal.visual_description.v1",
            "Visual description for clinician review only. This output is not a "
            "diagnosis. A qualified clinician must independently review the source "
            "and explicitly confirm before any consequential use. This output must "
            "never automatically trigger a clinical decision.",
        ),
        NoticeKind.DRAFT: (
            "openmed.multimodal.draft_for_review.v1",
            "Draft for clinician review only. This output is not a diagnosis. A "
            "qualified clinician must independently review the source and explicitly "
            "confirm before any consequential use. This output must never "
            "automatically trigger a clinical decision.",
        ),
    }
)


@dataclass(frozen=True)
class MultimodalNotice:
    """An exact identifier/text pair from the published notice catalog."""

    identifier: str
    text: str

    def __post_init__(self) -> None:
        if (self.identifier, self.text) not in _CATALOG.values():
            raise MultimodalNoticeError("notice identifier and text must match catalog")

    def to_dict(self) -> dict[str, str]:
        """Serialize a validated notice without caller-supplied interpolation."""
        self.__post_init__()
        return {"identifier": self.identifier, "text": self.text}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MultimodalNotice:
        """Reject missing, unknown or altered notice fields."""
        if not isinstance(data, Mapping) or set(data) != {"identifier", "text"}:
            raise MultimodalNoticeError("notice fields are invalid")
        return cls(identifier=data["identifier"], text=data["text"])


NOTICE_CATALOG = MappingProxyType(
    {
        kind: MultimodalNotice(identifier=identifier, text=text)
        for kind, (identifier, text) in _CATALOG.items()
    }
)


class NoticeBoundOutput:
    """Shared construction, wire and reviewer-confirmation guards.

    Concrete frozen dataclasses must declare their own ``NOTICE_KIND`` and a
    required ``notice`` field, and call ``_validate_notice`` at construction.
    JSON is a protected result surface, never a license to log clinical text.
    """

    NOTICE_KIND: ClassVar[NoticeKind]
    SCHEMA_VERSION: ClassVar[str] = "openmed.multimodal.review_result.v1"
    requires_reviewer_confirmation: ClassVar[bool] = True
    is_diagnostic: ClassVar[bool] = False
    notice: MultimodalNotice

    def __post_init__(self) -> None:
        self._validate_notice()

    def _validate_notice(self) -> None:
        if type(self.notice) is not MultimodalNotice:
            raise MultimodalNoticeError("result requires its catalog notice")
        self.notice.__post_init__()
        if self.notice != NOTICE_CATALOG[self.NOTICE_KIND]:
            raise MultimodalNoticeError("result notice kind does not match")

    def require_reviewer_confirmation(self, *, reviewer_confirmed: bool) -> None:
        """Require explicit confirmation before caller-controlled consequential use.

        This guard performs no action and does not establish reviewer identity,
        clinical validity or an authorization receipt; hosts enforce those.
        """
        self._validate_notice()
        if reviewer_confirmed is not True:
            raise MultimodalNoticeError("explicit reviewer confirmation required")

    def to_dict(self) -> dict[str, Any]:
        """Serialize all result fields with the validated notice and safety flags."""
        self.__post_init__()
        result = {
            field.name: getattr(self, field.name) for field in fields(cast(Any, self))
        }
        result["notice"] = self.notice.to_dict()
        for key, value in result.items():
            if isinstance(value, tuple):
                result[key] = list(value)
        return {
            "schema_version": self.SCHEMA_VERSION,
            "requires_reviewer_confirmation": True,
            "is_diagnostic": False,
            **result,
        }

    def to_json(self) -> str:
        """Return deterministic JSON retaining the mandatory notice."""
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Any:
        """Deserialize strictly; never repair a missing notice or safety flag."""
        expected = {field.name for field in fields(cast(Any, cls))} | {
            "schema_version",
            "requires_reviewer_confirmation",
            "is_diagnostic",
        }
        if not isinstance(data, Mapping) or set(data) != expected:
            raise MultimodalNoticeError("result fields are invalid")
        if (
            data["schema_version"] != cls.SCHEMA_VERSION
            or data["requires_reviewer_confirmation"] is not True
            or data["is_diagnostic"] is not False
        ):
            raise MultimodalNoticeError("result safety contract is invalid")
        values = {field.name: data[field.name] for field in fields(cast(Any, cls))}
        values["notice"] = MultimodalNotice.from_dict(values["notice"])
        return cls(**values)

    @classmethod
    def from_json(cls, payload: str | bytes) -> Any:
        """Parse bounded JSON with duplicate-key and patient-value-free errors."""
        if not isinstance(payload, (str, bytes)):
            raise MultimodalNoticeError("result JSON is invalid")
        invalid = False
        try:
            if (
                len(payload.encode("utf-8") if isinstance(payload, str) else payload)
                > MAX_NOTICE_RESULT_JSON_BYTES
            ):
                raise MultimodalNoticeError("result JSON exceeds size limit")
            data = json.loads(payload, object_pairs_hook=_unique_keys)
        except (ValueError, UnicodeError):
            invalid = True
        if invalid:
            raise MultimodalNoticeError("result JSON is invalid")
        return cls.from_dict(data)

    def render_text(self) -> str:
        """Render the notice and digest-only reference for CLI/review displays."""
        output_digest = self.to_dict().get("output_digest")
        if output_digest is None:
            raise MultimodalNoticeError(
                "digest-only rendering requires an output digest"
            )
        return f"[{self.notice.identifier}] {self.notice.text}\nOutput digest: {output_digest}"

    def __str__(self) -> str:
        return self.render_text()


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MultimodalNoticeError("duplicate result JSON key")
        result[key] = value
    return result


@dataclass(frozen=True)
class MeasurementReviewResult(NoticeBoundOutput):
    """Digest-bound measurement candidate, including ECG or WSI aggregates."""

    NOTICE_KIND: ClassVar[NoticeKind] = NoticeKind.MEASUREMENT
    output_digest: str
    notice: MultimodalNotice

    def __post_init__(self) -> None:
        _validate_review_result(self)


@dataclass(frozen=True)
class VisualDescriptionResult(NoticeBoundOutput):
    """Digest-bound visual description or image-quality finding for review."""

    NOTICE_KIND: ClassVar[NoticeKind] = NoticeKind.VISUAL_DESCRIPTION
    output_digest: str
    notice: MultimodalNotice

    def __post_init__(self) -> None:
        _validate_review_result(self)


@dataclass(frozen=True)
class DraftReviewResult(NoticeBoundOutput):
    """Digest-bound ambient draft for explicit clinician review."""

    NOTICE_KIND: ClassVar[NoticeKind] = NoticeKind.DRAFT
    output_digest: str
    notice: MultimodalNotice

    def __post_init__(self) -> None:
        _validate_review_result(self)


def _validate_review_result(result: Any) -> None:
    result._validate_notice()
    if not isinstance(result.output_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", result.output_digest
    ):
        raise MultimodalNoticeError("output digest must be lowercase SHA-256")


NOTICE_RESULT_TYPES = MappingProxyType(
    {
        "measurement_for_review": MeasurementReviewResult,
        "visual_description": VisualDescriptionResult,
        "draft_for_review": DraftReviewResult,
    }
)


def validate_notice_registry(result_types: Iterable[type]) -> None:
    """Fail closed when a covered output lacks a declared mandatory notice."""
    for result_type in result_types:
        if (
            not issubclass(result_type, NoticeBoundOutput)
            or result_type.__dict__.get("NOTICE_KIND") not in NOTICE_CATALOG
        ):
            raise MultimodalNoticeError("covered result must declare its notice kind")
        notice_fields = [
            field for field in fields(cast(Any, result_type)) if field.name == "notice"
        ]
        if (
            len(notice_fields) != 1
            or notice_fields[0].default is not MISSING
            or notice_fields[0].default_factory is not MISSING
        ):
            raise MultimodalNoticeError("covered result must require its notice")
