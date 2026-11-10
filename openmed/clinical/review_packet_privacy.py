"""Final privacy gate for rendered clinical review packets.

The caller supplies a configured local leakage detector. This module scans the
exact rendered content before export or persistence and reduces detector output
to entity classes, character offsets, and hashes computed from the rendered
span. Raw detector values are never copied into reports or exceptions.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol, TypeVar

from openmed.core.audit import hash_text
from openmed.core.review_workflow import critical_labels as default_critical_labels

REVIEW_PACKET_PRIVACY_SCHEMA_VERSION: Final = 1

_SAFE_ENTITY_CLASS = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_CRITICAL_LEVELS = frozenset({"blocking", "critical", "high"})
_T = TypeVar("_T")


class LeakageDetector(Protocol):
    """Protocol for a configured local detector over rendered text."""

    def __call__(self, text: str) -> Any:
        """Return findings for ``text``."""


class ReviewPacketPrivacyError(RuntimeError):
    """Base error for a review-packet privacy scan."""


class ReviewPacketPrivacyScanError(ReviewPacketPrivacyError):
    """Raised when a configured detector cannot produce a valid safe report."""


class ReviewPacketPrivacyBlocked(ReviewPacketPrivacyError):
    """Raised when critical leakage blocks packet output."""

    def __init__(self, report: "ReviewPacketPrivacyReport") -> None:
        self.report = report
        super().__init__(
            "clinical review packet output blocked by critical privacy findings"
        )


@dataclass(frozen=True)
class ReviewPacketPrivacyFinding:
    """Value-free location and hash for one detector finding."""

    entity_class: str
    start: int
    end: int
    text_hash: str
    _critical: bool = field(default=False, repr=False)

    @property
    def is_critical(self) -> bool:
        """Whether this finding blocks packet output."""

        return self._critical

    def to_dict(self) -> dict[str, Any]:
        """Return only the entity class, offsets, and rendered-span hash."""

        return {
            "entity_class": self.entity_class,
            "start": self.start,
            "end": self.end,
            "text_hash": self.text_hash,
        }


@dataclass(frozen=True)
class ReviewPacketPrivacyReport:
    """Deterministic, value-free final scan report."""

    findings: tuple[ReviewPacketPrivacyFinding, ...]
    content_hash: str
    schema_version: int = REVIEW_PACKET_PRIVACY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REVIEW_PACKET_PRIVACY_SCHEMA_VERSION:
            raise ValueError("unsupported review packet privacy schema version")
        ordered = tuple(
            sorted(
                set(self.findings),
                key=lambda finding: (
                    finding.start,
                    finding.end,
                    finding.entity_class,
                    finding.text_hash,
                    finding.is_critical,
                ),
            )
        )
        object.__setattr__(self, "findings", ordered)

    @property
    def critical_count(self) -> int:
        """Return the number of findings classified as critical."""

        return sum(finding.is_critical for finding in self.findings)

    @property
    def blocked(self) -> bool:
        """Whether at least one critical finding blocks output."""

        return self.critical_count > 0

    def to_dict(self) -> dict[str, Any]:
        """Return a report containing no rendered or detector-supplied values."""

        return {
            "schema_version": self.schema_version,
            "blocked": self.blocked,
            "finding_count": len(self.findings),
            "critical_count": self.critical_count,
            "content_hash": self.content_hash,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def scan_review_packet_privacy(
    rendered_content: str,
    detector: LeakageDetector | Any,
    *,
    critical_entity_classes: Iterable[str] | None = None,
) -> ReviewPacketPrivacyReport:
    """Run a configured detector over the exact rendered packet content.

    Detector findings may be mappings or objects exposing ``start``, ``end``,
    and an entity class through ``entity_class``, ``canonical_label``, ``label``,
    or ``entity_type``. A finding is critical when the detector marks it
    ``critical``, gives it a ``critical``/``high``/``blocking`` severity, or its
    entity class belongs to ``critical_entity_classes``. The default class set
    is OpenMed's direct-identifier/high-risk review set.
    """

    if not isinstance(rendered_content, str):
        raise TypeError("rendered_content must be a string")
    critical = _critical_classes(critical_entity_classes)

    try:
        raw_result = _run_detector(detector, rendered_content)
        raw_findings = _extract_findings(raw_result)
        findings = tuple(
            _normalize_finding(item, rendered_content, critical)
            for item in raw_findings
        )
    except ReviewPacketPrivacyScanError:
        raise
    except Exception:
        raise ReviewPacketPrivacyScanError(
            "configured review packet leakage detector failed"
        ) from None

    return ReviewPacketPrivacyReport(
        findings=findings,
        content_hash=hash_text(rendered_content),
    )


def enforce_review_packet_privacy(
    rendered_content: str,
    detector: LeakageDetector | Any,
    *,
    critical_entity_classes: Iterable[str] | None = None,
) -> ReviewPacketPrivacyReport:
    """Return a safe report or block output when a critical leak is detected."""

    report = scan_review_packet_privacy(
        rendered_content,
        detector,
        critical_entity_classes=critical_entity_classes,
    )
    if report.blocked:
        raise ReviewPacketPrivacyBlocked(report)
    return report


def export_review_packet(
    rendered_content: str,
    detector: LeakageDetector | Any,
    exporter: Callable[[str], _T],
    *,
    critical_entity_classes: Iterable[str] | None = None,
) -> tuple[_T, ReviewPacketPrivacyReport]:
    """Scan first, then pass approved content to an exporter callback."""

    if not callable(exporter):
        raise TypeError("exporter must be callable")
    report = enforce_review_packet_privacy(
        rendered_content,
        detector,
        critical_entity_classes=critical_entity_classes,
    )
    return exporter(rendered_content), report


def persist_review_packet(
    destination: str | Path,
    rendered_content: str,
    detector: LeakageDetector | Any,
    *,
    critical_entity_classes: Iterable[str] | None = None,
    encoding: str = "utf-8",
) -> ReviewPacketPrivacyReport:
    """Scan first, then persist approved rendered content to a local path."""

    report = enforce_review_packet_privacy(
        rendered_content,
        detector,
        critical_entity_classes=critical_entity_classes,
    )
    Path(destination).write_text(rendered_content, encoding=encoding)
    return report


def _run_detector(detector: Any, rendered_content: str) -> Any:
    if callable(detector):
        return detector(rendered_content)
    detect = getattr(detector, "detect", None)
    if callable(detect):
        return detect(rendered_content)
    raise ReviewPacketPrivacyScanError(
        "configured review packet leakage detector must be callable"
    )


def _extract_findings(result: Any) -> tuple[Any, ...]:
    if result is None:
        return ()
    for key in ("findings", "entities", "pii_entities"):
        value = _get(result, key)
        if value is not None:
            result = value
            break
    if isinstance(result, (str, bytes, bytearray, Mapping)):
        raise ReviewPacketPrivacyScanError(
            "configured review packet leakage detector returned invalid findings"
        )
    try:
        return tuple(result)
    except TypeError:
        raise ReviewPacketPrivacyScanError(
            "configured review packet leakage detector returned invalid findings"
        ) from None


def _normalize_finding(
    finding: Any,
    rendered_content: str,
    critical_classes: frozenset[str],
) -> ReviewPacketPrivacyFinding:
    start = _get(finding, "start")
    end = _get(finding, "end")
    if type(start) is not int or type(end) is not int:
        raise ReviewPacketPrivacyScanError(
            "review packet privacy finding offsets must be integers"
        )
    if start < 0 or end <= start or end > len(rendered_content):
        raise ReviewPacketPrivacyScanError(
            "review packet privacy finding offsets are outside rendered content"
        )

    entity_class = _entity_class(finding)
    return ReviewPacketPrivacyFinding(
        entity_class=entity_class,
        start=start,
        end=end,
        text_hash=hash_text(rendered_content[start:end]),
        _critical=_is_critical(finding, entity_class, critical_classes),
    )


def _entity_class(finding: Any) -> str:
    value = None
    for key in ("entity_class", "canonical_label", "label", "entity_type"):
        candidate = _get(finding, key)
        if candidate is not None:
            value = candidate
            break
    if not isinstance(value, str):
        return "UNKNOWN"
    normalized = re.sub(r"[^A-Z0-9]+", "_", value.strip().upper()).strip("_")
    if not _SAFE_ENTITY_CLASS.fullmatch(normalized):
        return "UNKNOWN"
    return normalized


def _is_critical(
    finding: Any,
    entity_class: str,
    critical_classes: frozenset[str],
) -> bool:
    explicit = _get(finding, "critical")
    if explicit is True:
        return True
    for key in ("severity", "risk_level"):
        level = _get(finding, key)
        if isinstance(level, str) and level.strip().casefold() in _CRITICAL_LEVELS:
            return True
    return entity_class in critical_classes


def _critical_classes(values: Iterable[str] | None) -> frozenset[str]:
    source = default_critical_labels() if values is None else values
    if isinstance(source, (str, bytes, bytearray)):
        raise TypeError("critical_entity_classes must be an iterable of class names")
    try:
        return frozenset(_safe_configured_class(value) for value in source)
    except TypeError:
        raise TypeError(
            "critical_entity_classes must contain class-name strings"
        ) from None


def _safe_configured_class(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError
    normalized = re.sub(r"[^A-Z0-9]+", "_", value.strip().upper()).strip("_")
    if not _SAFE_ENTITY_CLASS.fullmatch(normalized):
        raise ValueError("critical entity class is invalid")
    return normalized


def _get(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


__all__ = [
    "REVIEW_PACKET_PRIVACY_SCHEMA_VERSION",
    "LeakageDetector",
    "ReviewPacketPrivacyBlocked",
    "ReviewPacketPrivacyError",
    "ReviewPacketPrivacyFinding",
    "ReviewPacketPrivacyReport",
    "ReviewPacketPrivacyScanError",
    "enforce_review_packet_privacy",
    "export_review_packet",
    "persist_review_packet",
    "scan_review_packet_privacy",
]
