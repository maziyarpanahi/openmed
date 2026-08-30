"""Deterministic prompt-injection screening for untrusted tool input.

The guard is intentionally lexical and local.  It identifies a small,
reviewable set of instruction-override, tool-spoofing, delimiter, and
data-exfiltration cues.  It does not attempt to decide whether a document is
clinically true or safe, and it is not a replacement for model-level controls.

Findings contain only a stable pattern identifier, an original-text offset,
and a severity.  The input text is never included in a finding or its safe
serialization.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Pattern

InjectionGuardMode = Literal["allow", "strict"]
FindingSeverity = Literal["high", "critical"]

DEFAULT_INJECTION_GUARD_MODE: InjectionGuardMode = "strict"
INJECTION_GUARD_MODE_ENV = "OPENMED_INJECTION_GUARD_MODE"
QUARANTINE_MARKER = "[OPENMED_QUARANTINED_PROMPT_INJECTION:{patterns}]"


@dataclass(frozen=True)
class _PatternDefinition:
    pattern_id: str
    severity: FindingSeverity
    expressions: tuple[Pattern[str], ...]


def _compile(expression: str) -> Pattern[str]:
    return re.compile(expression, re.IGNORECASE | re.MULTILINE)


_PATTERN_DEFINITIONS: tuple[_PatternDefinition, ...] = (
    _PatternDefinition(
        pattern_id="instruction_override",
        severity="high",
        expressions=(
            _compile(
                r"\b(?:ignore|disregard|forget|override|bypass)\s+"
                r"(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier|"
                r"system)\s+(?:instructions?|rules?|directions?|prompt)\b"
            ),
            _compile(
                r"\b(?:new|updated|real|actual)\s+(?:system|developer)\s+"
                r"instructions?\b"
            ),
            _compile(
                r"\b(?:you\s+are|act\s+as|assume\s+the\s+role\s+of)\s+"
                r"(?:now\s+)?(?:an?\s+)?(?:unrestricted|unfiltered|"
                r"different|system|developer)\b"
            ),
        ),
    ),
    _PatternDefinition(
        pattern_id="tool_name_spoofing",
        severity="high",
        expressions=(
            _compile(r"<\s*(?:tool_call|function_call|function|tool)\b[^>]*>"),
            _compile(
                r"\b(?:call|invoke|run|execute|use)\s+(?:the\s+)?"
                r"(?:mcp\s+)?(?:tool|function)\s*[:=]?\s*[\w.-]+"
            ),
            _compile(
                r"\b(?:call|invoke|run|execute|use)\s+(?:the\s+)?"
                r"(?:openmed|mcp)[\w.-]*\b"
            ),
            _compile(
                r"\b(?:recipient|to)\s*=\s*[\"']?"
                r"(?:functions|tools)\.[\w.-]+"
            ),
            _compile(
                r"[\"'](?:name|tool|function)[\"']\s*:\s*[\"']"
                r"(?:openmed|mcp)[\w.-]*[\"']"
            ),
        ),
    ),
    _PatternDefinition(
        pattern_id="delimiter_breakout",
        severity="high",
        expressions=(
            _compile(
                r"(?:^|\n)[ \t]*(?:#{1,6}\s*)?"
                r"(?:system|developer|assistant|user|instruction|instructions)\s*:"
            ),
            _compile(
                r"<\s*/?\s*(?:system|developer|assistant|user|instruction|"
                r"instructions|prompt|message)\b[^>]*>"
            ),
            _compile(r"\[\s*/?(?:inst|sys|system|instruction|instructions)\s*\]"),
            _compile(r"<<\s*/?(?:sys|system)\s*>>"),
            _compile(
                r"\b(?:begin|end)\s+(?:system\s+)?"
                r"(?:prompt|instructions?)\s*:?"
            ),
            _compile(
                r"(?:^|\n)\s*#{2,}\s*"
                r"(?:system|instruction|developer|assistant)\b"
            ),
        ),
    ),
    _PatternDefinition(
        pattern_id="data_exfiltration",
        severity="critical",
        expressions=(
            _compile(
                r"\b(?:reveal|show|print|repeat|dump|disclose|expose|return|"
                r"output)\s+(?:the\s+)?(?:hidden|secret|system|developer)?\s*"
                r"(?:prompt|instructions?|context|messages?|rules?)\b"
            ),
            _compile(
                r"\b(?:exfiltrate|export|send|upload|forward|transmit|email|post)"
                r"\s+(?:all\s+|the\s+|any\s+)?(?:raw\s+)?"
                r"(?:phi|pii|patient|clinical|medical|health|private|sensitive|"
                r"records?|data|document|text)\b"
            ),
            _compile(
                r"\b(?:return|include|output|print|dump)\s+(?:the\s+)?"
                r"(?:raw|original|unredacted|unmasked)\s+"
                r"(?:phi|pii|patient(?:\s+data|\s+records?)?|"
                r"clinical(?:\s+notes?)?|document|text|record)\b"
            ),
            _compile(
                r"\b(?:do\s+not|don't)\s+(?:redact|mask|deidentify|"
                r"de-identify|sanitize|remove)\b.*\b"
                r"(?:phi|pii|patient|record|text|data)\b"
            ),
        ),
    ),
)


@dataclass(frozen=True)
class InjectionFinding:
    """PHI-safe description of one detected prompt-injection pattern."""

    pattern_id: str
    start: int
    end: int
    severity: FindingSeverity

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("finding offsets must describe a non-empty span")
        if self.severity not in {"high", "critical"}:
            raise ValueError("finding severity must be high or critical")

    def to_dict(self) -> dict[str, Any]:
        """Return the safe, text-free finding representation."""
        return {
            "pattern_id": self.pattern_id,
            "start": self.start,
            "end": self.end,
            "severity": self.severity,
        }

    @property
    def offsets(self) -> tuple[int, int]:
        """Return the original-text span as a ``(start, end)`` pair."""
        return self.start, self.end


@dataclass(frozen=True)
class InjectionScan:
    """Result of scanning one text value.

    ``quarantined_text`` is intended for dispatch only.  Use ``to_dict`` for
    audit or error responses because it never serializes either text value.
    """

    text: str
    findings: tuple[InjectionFinding, ...]
    quarantined_text: str
    mode: InjectionGuardMode

    @property
    def flagged(self) -> bool:
        """Return whether any known injection pattern was found."""
        return bool(self.findings)

    @property
    def is_flagged(self) -> bool:
        """Alias for callers that prefer an explicit predicate name."""
        return self.flagged

    @property
    def sanitized_text(self) -> str:
        """Return the inert text prepared for allow-mode dispatch."""
        return self.quarantined_text

    def to_dict(self) -> dict[str, Any]:
        """Return a PHI-safe scan summary without input or output text."""
        return {
            "flagged": self.flagged,
            "mode": self.mode,
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class GuardedInput:
    """A recursively copied input prepared for safe tool dispatch."""

    value: Any
    findings: tuple[InjectionFinding, ...]

    @property
    def flagged(self) -> bool:
        """Return whether any nested text value was quarantined."""
        return bool(self.findings)

    def finding_dicts(self) -> list[dict[str, Any]]:
        """Return finding records suitable for a PHI-safe error envelope."""
        return [finding.to_dict() for finding in self.findings]


class PromptInjectionDetected(ValueError):
    """Raised by strict mode before a flagged tool input reaches a handler."""

    def __init__(self, findings: Iterable[InjectionFinding]) -> None:
        self.findings = tuple(findings)
        super().__init__("Prompt-injection patterns were detected in tool input.")

    def to_dict(self) -> dict[str, Any]:
        """Return a safe error payload without including the rejected input."""
        return {
            "code": "prompt_injection_detected",
            "message": "The tool input was quarantined by the prompt-injection guard.",
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class _NormalizedText:
    text: str
    source_spans: tuple[tuple[int, int], ...]


def _normalize_with_offsets(text: str) -> _NormalizedText:
    normalized: list[str] = []
    source_spans: list[tuple[int, int]] = []
    for index, character in enumerate(text):
        # NFKC handles full-width delimiters and letters.  Format characters
        # such as zero-width joiners are omitted so they cannot split a cue.
        if unicodedata.category(character) == "Cf":
            continue
        folded = unicodedata.normalize("NFKC", character).casefold()
        for item in folded:
            normalized.append(" " if item.isspace() else item)
            source_spans.append((index, index + 1))
    return _NormalizedText("".join(normalized), tuple(source_spans))


def _original_span(
    normalized: _NormalizedText,
    start: int,
    end: int,
) -> tuple[int, int]:
    spans = normalized.source_spans[start:end]
    if not spans:
        raise ValueError("cannot map an empty normalized match")
    return min(span[0] for span in spans), max(span[1] for span in spans)


def _scan_text(text: str) -> tuple[InjectionFinding, ...]:
    normalized = _normalize_with_offsets(text)
    findings: set[InjectionFinding] = set()
    for definition in _PATTERN_DEFINITIONS:
        for expression in definition.expressions:
            for match in expression.finditer(normalized.text):
                start, end = _original_span(
                    normalized,
                    match.start(),
                    match.end(),
                )
                # Heading patterns may include a line break or indentation;
                # report the meaningful cue while preserving its source span.
                while start < end and text[start].isspace():
                    start += 1
                if start < end:
                    findings.add(
                        InjectionFinding(
                            pattern_id=definition.pattern_id,
                            start=start,
                            end=end,
                            severity=definition.severity,
                        )
                    )
    return tuple(
        sorted(
            findings,
            key=lambda finding: (
                finding.start,
                finding.end,
                finding.pattern_id,
                finding.severity,
            ),
        )
    )


def quarantine_text(
    text: str,
    findings: Sequence[InjectionFinding],
) -> str:
    """Replace flagged spans with inert markers while preserving other text."""

    if not findings:
        return text
    ordered = sorted(findings, key=lambda finding: (finding.start, finding.end))
    chunks: list[str] = []
    cursor = 0
    group_start = -1
    group_end = -1
    group_ids: set[str] = set()

    def flush_group() -> None:
        nonlocal cursor, group_start, group_end, group_ids
        if group_start < 0:
            return
        chunks.append(text[cursor:group_start])
        patterns = ",".join(sorted(group_ids))
        chunks.append(QUARANTINE_MARKER.format(patterns=patterns))
        cursor = group_end
        group_start = -1
        group_end = -1
        group_ids = set()

    for finding in ordered:
        if finding.start < cursor or finding.end > len(text):
            raise ValueError("finding offsets do not match the supplied text")
        if group_start < 0:
            group_start = finding.start
            group_end = finding.end
            group_ids.add(finding.pattern_id)
            continue
        if finding.start <= group_end:
            group_end = max(group_end, finding.end)
            group_ids.add(finding.pattern_id)
            continue
        flush_group()
        group_start = finding.start
        group_end = finding.end
        group_ids.add(finding.pattern_id)
    flush_group()
    chunks.append(text[cursor:])
    return "".join(chunks)


_STRUCTURAL_FIELDS = frozenset(
    {
        "bundle_type",
        "category",
        "date_shift_days",
        "doc_id",
        "from_handle",
        "from_step",
        "key_id",
        "lang",
        "language",
        "limit",
        "max_candidates",
        "method",
        "model_name",
        "path",
        "pii_language",
        "session_id",
        "stages",
        "tool",
        "workflow_id",
    }
)


def _guard_value(
    value: Any,
    guard: "InjectionGuard",
    field_name: str | None = None,
) -> tuple[Any, list[InjectionFinding]]:
    if isinstance(value, str):
        if field_name is not None and field_name.casefold() in _STRUCTURAL_FIELDS:
            return value, []
        scan = guard.scan(value)
        return scan.quarantined_text, list(scan.findings)
    if isinstance(value, Mapping):
        guarded: dict[Any, Any] = {}
        findings: list[InjectionFinding] = []
        for key, child in value.items():
            guarded_child, child_findings = _guard_value(
                child,
                guard,
                str(key),
            )
            guarded[key] = guarded_child
            findings.extend(child_findings)
        return guarded, findings
    if isinstance(value, list):
        guarded_items: list[Any] = []
        findings = []
        for item in value:
            guarded_item, item_findings = _guard_value(item, guard, field_name)
            guarded_items.append(guarded_item)
            findings.extend(item_findings)
        return guarded_items, findings
    if isinstance(value, tuple):
        guarded_items = []
        findings = []
        for item in value:
            guarded_item, item_findings = _guard_value(item, guard, field_name)
            guarded_items.append(guarded_item)
            findings.extend(item_findings)
        return tuple(guarded_items), findings
    return value, []


def _dedupe_findings(
    findings: Iterable[InjectionFinding],
) -> tuple[InjectionFinding, ...]:
    return tuple(
        sorted(
            set(findings),
            key=lambda finding: (
                finding.start,
                finding.end,
                finding.pattern_id,
                finding.severity,
            ),
        )
    )


class InjectionGuard:
    """Scan and quarantine untrusted text using a strict or allow policy.

    ``strict`` is fail-closed: a finding raises
    :class:`PromptInjectionDetected` before dispatch.  ``allow`` still
    quarantines every finding, then permits the copied value to reach the
    handler with the matched span replaced by an inert marker.
    """

    def __init__(self, mode: InjectionGuardMode | str = DEFAULT_INJECTION_GUARD_MODE):
        normalized_mode = str(mode).casefold()
        if normalized_mode not in {"allow", "strict"}:
            raise ValueError("injection guard mode must be 'allow' or 'strict'")
        self.mode: InjectionGuardMode = normalized_mode  # type: ignore[assignment]

    @classmethod
    def from_env(
        cls,
        env_var: str = INJECTION_GUARD_MODE_ENV,
    ) -> "InjectionGuard":
        """Build a guard from an environment variable, defaulting to strict."""
        return cls(os.getenv(env_var, DEFAULT_INJECTION_GUARD_MODE))

    def scan(self, text: str) -> InjectionScan:
        """Return findings and a quarantined copy without raising."""
        if not isinstance(text, str):
            raise TypeError("injection guard input must be text")
        findings = _scan_text(text)
        return InjectionScan(
            text=text,
            findings=findings,
            quarantined_text=quarantine_text(text, findings),
            mode=self.mode,
        )

    def guard_text(self, text: str) -> str:
        """Return safe text or raise in strict mode when a cue is found."""
        scan = self.scan(text)
        if scan.flagged and self.mode == "strict":
            raise PromptInjectionDetected(scan.findings)
        return scan.quarantined_text

    def guard_input(self, value: Any) -> GuardedInput:
        """Recursively copy and guard nested mapping, list, and tuple values."""
        guarded, findings = _guard_value(value, self)
        ordered_findings = _dedupe_findings(findings)
        if ordered_findings and self.mode == "strict":
            raise PromptInjectionDetected(ordered_findings)
        return GuardedInput(value=guarded, findings=ordered_findings)

    def guard_arguments(self, arguments: Mapping[str, Any]) -> GuardedInput:
        """Guard a JSON-like MCP argument mapping before handler dispatch."""
        if not isinstance(arguments, Mapping):
            raise TypeError("MCP tool arguments must be an object")
        return self.guard_input(arguments)


def scan_text(text: str) -> InjectionScan:
    """Scan text with the default strict-policy metadata, without raising."""
    return InjectionGuard().scan(text)


def guard_text(
    text: str,
    *,
    mode: InjectionGuardMode | str = DEFAULT_INJECTION_GUARD_MODE,
) -> str:
    """Guard one text value using the requested strict or allow policy."""
    return InjectionGuard(mode=mode).guard_text(text)


__all__ = [
    "DEFAULT_INJECTION_GUARD_MODE",
    "FindingSeverity",
    "GuardedInput",
    "INJECTION_GUARD_MODE_ENV",
    "InjectionFinding",
    "InjectionGuard",
    "InjectionGuardMode",
    "InjectionScan",
    "PromptInjectionDetected",
    "QUARANTINE_MARKER",
    "guard_text",
    "quarantine_text",
    "scan_text",
]
