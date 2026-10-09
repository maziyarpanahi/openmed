"""Fail-closed, local post-generation screening with digest-only evidence."""

from __future__ import annotations

import hashlib
import json
import math
import queue
import threading
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import islice
from typing import Any

from .provider_result import (
    ProviderAbstentionCode,
    ProviderResultEnvelope,
    ProviderResultOutcome,
)

__all__ = [
    "AssetIdentifierDigest",
    "GeneratedTextGuard",
    "GuardedTextResult",
    "GuardPolicy",
    "PHILabel",
    "PHISpan",
    "identifier_digest",
]


class PHILabel(str, Enum):
    """Controlled identifier labels accepted by the release boundary."""

    NAME = "name"
    MRN = "mrn"
    DATE = "date"
    IDENTIFIER = "identifier"


def _hash(value: str) -> str:
    try:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
    except UnicodeError:
        raise ValueError("digest input is invalid") from None


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalized(value: str) -> str:
    # Per-code-point normalization makes the original offset map unambiguous.
    return "".join(
        part
        for char in value
        for part in unicodedata.normalize("NFKC", char).casefold()
        if not part.isspace()
    )


def identifier_digest(value: str) -> str:
    """Hash a transient identifier using v1 case/whitespace normalization.

    Args:
        value: Caller-held identifier; never retained by this function.

    Returns:
        Lowercase SHA-256 digest compatible with asset evidence matching.
    """
    return _hash(_normalized(value))


@dataclass(frozen=True, slots=True)
class AssetIdentifierDigest:
    """Same-asset evidence, with no identifier value retained.

    Args:
        asset_digest: Digest of the asset supplying OCR/header/region evidence.
        label: Controlled PHI label.
        digest: Digest from ``identifier_digest``.
        normalized_length: Length after v1 normalization (1 through 256).
    """

    asset_digest: str
    label: PHILabel
    digest: str
    normalized_length: int

    def __post_init__(self) -> None:
        if (
            not _valid_digest(self.asset_digest)
            or not _valid_digest(self.digest)
            or not isinstance(self.label, PHILabel)
            or type(self.normalized_length) is not int
            or not 1 <= self.normalized_length <= 256
        ):
            raise ValueError("identifier evidence is invalid")


def _valid_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


@dataclass(frozen=True, slots=True)
class PHISpan:
    """Value-free local detector finding in Python code-point offsets.

    Args:
        start: Inclusive offset in the generated text.
        end: Exclusive offset in the generated text.
        label: Controlled PHI label.
        resolved: False when a critical finding cannot be localized safely.
    """

    start: int
    end: int
    label: PHILabel
    resolved: bool = True


@dataclass(frozen=True, slots=True)
class GuardPolicy:
    """Bounded release policy; unsafe or incomplete findings always abstain.

    Args:
        action: ``redact``, ``withhold``, or ``abstain`` on resolved findings.
        timeout_seconds: Maximum detector wait, at most 60 seconds.
        max_characters: Maximum generation size, at most 8192 code points.
    """

    action: str = "redact"
    timeout_seconds: float = 1.0
    max_characters: int = 8192

    def __post_init__(self) -> None:
        if (
            type(self.action) is not str
            or self.action not in {"redact", "withhold", "abstain"}
            or type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= 60
            or type(self.max_characters) is not int
            or not 1 <= self.max_characters <= 8192
        ):
            raise ValueError("guard policy is invalid")

    @property
    def digest(self) -> str:
        """Return the deterministic policy and normalization contract digest."""
        return _hash(
            _canonical(
                {
                    "version": 1,
                    "use_policy": "non_diagnostic_review_required",
                    "normalization": "per-codepoint-nfkc-casefold-no-space",
                    "action": self.action,
                    "timeout_seconds": float(self.timeout_seconds),
                    "max_characters": self.max_characters,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class GuardedTextResult:
    """Released text separate from content-free qualification metadata.

    Args:
        text: Redacted or clean released text; None on abstention/withholding.
        envelope: Provider result binding the decision and evidence digests.
        report_json: Deterministic counts, controlled labels and digests only.
    """

    text: str | None = field(repr=False)
    envelope: ProviderResultEnvelope
    report_json: str

    @property
    def requires_reviewer_confirmation(self) -> bool:
        """Require explicit human confirmation before consequential use."""
        return True

    @property
    def non_diagnostic_notice(self) -> str:
        """Return the fixed notice bound by the report use-policy label."""
        return "Not diagnostic. Explicit reviewer confirmation is required."


class GeneratedTextGuard:
    """Screen buffered generations using an injected offline detector.

    Args:
        detector: Local callable returning only PHISpan findings. None abstains.
        detector_version: Caller-controlled version, stored only as a digest.
        policy: Release action and resource bounds.

    A timeout cannot terminate arbitrary caller code. A daemon worker discards
    late findings; this guard rejects concurrent calls and stays unavailable
    until that worker finishes, bounding each instance to one outstanding call.
    Detectors must not log, persist, or transmit the transient text.
    """

    def __init__(
        self,
        detector: Callable[[str], Sequence[PHISpan]] | None,
        *,
        detector_version: str = "unavailable",
        policy: GuardPolicy = GuardPolicy(),
    ) -> None:
        if (
            type(detector_version) is not str
            or not detector_version
            or len(detector_version) > 256
        ):
            raise ValueError("detector version is invalid")
        if not isinstance(policy, GuardPolicy):
            raise ValueError("guard policy is invalid")
        self._detector = detector
        self._version_digest = _hash(detector_version)
        self.policy = policy
        self._busy = threading.Lock()

    def release(
        self,
        text: str,
        *,
        asset_digest: str,
        identifiers: Sequence[AssetIdentifierDigest] = (),
        provider_id: str = "local-vlm",
        model_id: str = "local-vlm",
        duration_ms: float = 0.0,
    ) -> GuardedTextResult:
        """Return guarded text and a provider envelope; never raise detector errors.

        Args:
            text: Buffered generation, never serialized in metadata.
            asset_digest: Digest identifying the same input asset as the evidence.
            identifiers: OCR/header/region commitments using v1 normalization.
            provider_id: Controlled adapter identifier for the provider envelope.
            model_id: Controlled model identifier for the provider envelope.
            duration_ms: Caller-supplied bounded provider duration.

        Returns:
            Clean/redacted text or phi_uncertainty abstention with safe evidence.
        """
        spans: list[PHISpan] = []
        decision = "abstain"
        released = None
        valid = type(text) is str and len(text) <= self.policy.max_characters
        try:
            identifiers = tuple(islice(identifiers, 129))
            valid = (
                valid
                and len(identifiers) <= 128
                and all(
                    isinstance(item, AssetIdentifierDigest)
                    and item.asset_digest == asset_digest
                    for item in identifiers
                )
            )
            if valid:
                detected = self._detect(text)
                if (
                    detected is not None
                    and len(detected) <= 4096
                    and all(
                        type(span) is PHISpan
                        and type(span.start) is int
                        and type(span.end) is int
                        and 0 <= span.start < span.end <= len(text)
                        and isinstance(span.label, PHILabel)
                        and type(span.resolved) is bool
                        and span.resolved
                        for span in detected
                    )
                ):
                    spans = list(detected) + _matches(text, identifiers)
                    if len(spans) > 4096:
                        raise ValueError("guard finding limit exceeded")
                    decision = self.policy.action if spans else "pass"
                    if decision == "pass":
                        released = text
                    elif decision == "redact":
                        released = _redact(text, spans)
        except Exception:
            # Never propagate a detector, malformed finding or payload exception.
            valid = False
            spans = []
            decision = "abstain"
            released = None
        counts = {
            label.value: sum(s.label is label for s in spans) for label in PHILabel
        }
        report = _canonical(
            {
                "decision": decision,
                "use_policy": "non_diagnostic_review_required",
                "detection_count": len(spans),
                "label_counts": counts,
                "detector_version_digest": self._version_digest,
                "policy_digest": self.policy.digest,
                "identifier_digests": sorted({s.digest for s in identifiers})
                if valid
                else [],
            }
        )
        envelope = ProviderResultEnvelope(
            provider_id=provider_id,
            model_id=model_id,
            input_digest=asset_digest,
            outcome=ProviderResultOutcome.SUCCESS
            if released is not None
            else ProviderResultOutcome.ABSTENTION,
            output_digest=_hash(released) if released is not None else None,
            abstention_code=None
            if released is not None
            else ProviderAbstentionCode.PHI_UNCERTAINTY,
            duration_ms=duration_ms,
            count_metadata={"detection_count": len(spans)},
            guard_metadata={
                "decision": decision,
                "detector_version_digest": self._version_digest,
                "policy_digest": self.policy.digest,
                "report_digest": _hash(report),
            },
        )
        return GuardedTextResult(released, envelope, report)

    def _detect(self, text: str) -> Sequence[PHISpan] | None:
        if self._detector is None or not self._busy.acquire(blocking=False):
            return None
        results: queue.Queue[Sequence[PHISpan] | None] = queue.Queue(maxsize=1)

        def run() -> None:
            try:
                # Materialize inside the deadline, including lazy detector results.
                value = tuple(islice(self._detector(text), 4097))  # type: ignore[misc]
            except BaseException:
                value = None
            finally:
                self._busy.release()
            results.put(value)

        threading.Thread(target=run, daemon=True).start()
        try:
            return results.get(timeout=self.policy.timeout_seconds)
        except queue.Empty:
            return None


def _matches(text: str, identifiers: Sequence[AssetIdentifierDigest]) -> list[PHISpan]:
    normalized = ""
    offsets = []
    for index, char in enumerate(text):
        part = _normalized(char)
        normalized += part
        offsets.extend([index] * len(part))
    spans = []
    by_length: dict[int, dict[str, set[PHILabel]]] = {}
    for item in identifiers:
        by_length.setdefault(item.normalized_length, {}).setdefault(
            item.digest, set()
        ).add(item.label)
    for length, digests in by_length.items():
        for start in range(len(normalized) - length + 1):
            labels = digests.get(_hash(normalized[start : start + length]), ())
            for label in sorted(labels, key=lambda x: x.value):
                spans.append(
                    PHISpan(offsets[start], offsets[start + length - 1] + 1, label)
                )
                if len(spans) > 4096:
                    raise ValueError("guard finding limit exceeded")
    return spans


def _redact(text: str, spans: Sequence[PHISpan]) -> str:
    merged: list[tuple[int, int]] = []
    for span in sorted(spans, key=lambda s: (s.start, s.end)):
        if merged and span.start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], span.end))
        else:
            merged.append((span.start, span.end))
    for start, end in reversed(merged):
        text = text[:start] + "[REDACTED]" + text[end:]
    return text
