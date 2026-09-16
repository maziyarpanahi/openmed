"""Deterministic local preflight checks for agent context and tool outputs.

The gate scans string leaves in caller-supplied payloads before dispatch.  It
uses OpenMed's deterministic safety sweep rather than loading a model, and its
findings contain only a category, offsets, and a one-way surface hash.  A
caller can either fail closed or receive a structurally equivalent payload with
the detected spans replaced by stable redaction tokens.

This is a local privacy guard, not a complete PHI detector or a compliance
certification.  Deployments with domain-specific identifiers can provide a
local scanner that returns offset-only findings.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from ..core.safety_sweep import safety_sweep
from ..processing.text import validate_pii_input

FAIL_CLOSED_POLICY = "fail_closed"
REDACT_THEN_CONTINUE_POLICY = "redact_then_continue"

FAIL_CLOSED = FAIL_CLOSED_POLICY
REDACT_THEN_CONTINUE = REDACT_THEN_CONTINUE_POLICY

PreflightPolicy: TypeAlias = Literal[
    "fail_closed",
    "redact_then_continue",
]
PreflightScanner: TypeAlias = Callable[[str], Iterable[Any] | Any]

_CHANNELS = frozenset({"context", "tool_output"})
_CATEGORY_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_POLICY_ALIASES = {
    "block": FAIL_CLOSED_POLICY,
    "fail-closed": FAIL_CLOSED_POLICY,
    "fail_closed": FAIL_CLOSED_POLICY,
    "redact": REDACT_THEN_CONTINUE_POLICY,
    "redact-then-continue": REDACT_THEN_CONTINUE_POLICY,
    "redact_then_continue": REDACT_THEN_CONTINUE_POLICY,
}
_MAX_PAYLOAD_DEPTH = 32
_MAX_PAYLOAD_LEAVES = 4096
_MAX_PAYLOAD_BYTES = 8 * 1024 * 1024


def _normalize_policy(policy: str) -> PreflightPolicy:
    if not isinstance(policy, str):
        raise ValueError("preflight policy is invalid")
    normalized = _POLICY_ALIASES.get(policy.strip().lower())
    if normalized is None:
        raise ValueError("preflight policy is invalid")
    return normalized  # type: ignore[return-value]


def _normalize_category(category: Any) -> str:
    if not isinstance(category, str):
        raise ValueError("preflight finding category is invalid")
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", category.strip()).strip("_").upper()
    if not _CATEGORY_RE.fullmatch(normalized):
        raise ValueError("preflight finding category is invalid")
    return normalized


def _surface_hash(text: str, start: int, end: int) -> str:
    surface = text[start:end]
    digest = hashlib.sha256(surface.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True)
class PreflightFinding:
    """A PHI finding with no matched source text.

    ``start`` and ``end`` are character offsets in the individual string leaf
    that was scanned.  ``payload_index`` identifies that leaf in a stable
    traversal of the context or tool-output payload; it is not a source path.
    """

    category: str
    start: int
    end: int
    channel: str = "context"
    payload_index: int = 0
    text_hash: str = ""

    def __post_init__(self) -> None:
        """Validate fields so reports remain safe and deterministic."""

        object.__setattr__(self, "category", _normalize_category(self.category))
        if not isinstance(self.channel, str) or self.channel not in _CHANNELS:
            raise ValueError("preflight finding channel is invalid")
        if type(self.start) is not int or type(self.end) is not int:
            raise ValueError("preflight finding offsets are invalid")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("preflight finding offsets are invalid")
        if type(self.payload_index) is not int or self.payload_index < 0:
            raise ValueError("preflight finding payload index is invalid")
        if not isinstance(self.text_hash, str) or (
            self.text_hash and not _HASH_RE.fullmatch(self.text_hash)
        ):
            raise ValueError("preflight finding hash is invalid")

    @property
    def label(self) -> str:
        """Return the category under the common entity-label name."""

        return self.category

    @property
    def source(self) -> str:
        """Return the payload channel under the common source name."""

        return self.channel

    @property
    def length(self) -> int:
        """Return the matched span length without exposing its surface."""

        return self.end - self.start

    @property
    def offsets(self) -> tuple[int, int]:
        """Return the inclusive/exclusive offset pair."""

        return self.start, self.end

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe finding without the matched source text."""

        return {
            "category": self.category,
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "channel": self.channel,
            "payload_index": self.payload_index,
            "text_hash": self.text_hash,
        }


@dataclass(frozen=True)
class PreflightReport:
    """PHI-free report produced by a preflight scan."""

    policy: PreflightPolicy
    allowed: bool
    redacted: bool
    findings: tuple[PreflightFinding, ...] = ()
    payload_count: int = 0

    def __post_init__(self) -> None:
        """Normalize report collections and validate safe scalar fields."""

        object.__setattr__(self, "policy", _normalize_policy(self.policy))
        findings = tuple(self.findings)
        if not all(isinstance(finding, PreflightFinding) for finding in findings):
            raise ValueError("preflight report findings are invalid")
        object.__setattr__(self, "findings", findings)
        if type(self.allowed) is not bool or type(self.redacted) is not bool:
            raise ValueError("preflight report state is invalid")
        if type(self.payload_count) is not int or self.payload_count < 0:
            raise ValueError("preflight report payload count is invalid")

    @property
    def finding_count(self) -> int:
        """Return the number of detected spans."""

        return len(self.findings)

    @property
    def finding_categories(self) -> tuple[str, ...]:
        """Return unique finding categories in deterministic order."""

        return tuple(sorted({finding.category for finding in self.findings}))

    @property
    def finding_counts(self) -> dict[str, int]:
        """Return category counts without any source content."""

        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.category] = counts.get(finding.category, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report suitable for logs and audit records."""

        return {
            "policy": self.policy,
            "allowed": self.allowed,
            "redacted": self.redacted,
            "payload_count": self.payload_count,
            "finding_count": self.finding_count,
            "finding_categories": list(self.finding_categories),
            "finding_counts": self.finding_counts,
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class PreflightResult:
    """Result of a preflight scan and its safe payload for dispatch.

    The sanitized payload fields are intentionally excluded from ``repr`` and
    ``to_dict``.  Callers should pass them directly to the next local stage;
    only :attr:`report` belongs in logs or audit artifacts.
    """

    report: PreflightReport
    sanitized_context: Any = field(repr=False)
    sanitized_tool_outputs: Any = field(repr=False)

    @property
    def allowed(self) -> bool:
        """Return whether dispatch may proceed under the selected policy."""

        return self.report.allowed

    @property
    def blocked(self) -> bool:
        """Return whether the selected policy stopped dispatch."""

        return not self.allowed

    @property
    def policy(self) -> PreflightPolicy:
        """Return the normalized policy name."""

        return self.report.policy

    @property
    def findings(self) -> tuple[PreflightFinding, ...]:
        """Return offset-only findings."""

        return self.report.findings

    @property
    def finding_categories(self) -> tuple[str, ...]:
        """Return unique detected categories."""

        return self.report.finding_categories

    @property
    def redacted(self) -> bool:
        """Return whether a finding was replaced in the dispatch payload."""

        return self.report.redacted

    @property
    def context(self) -> Any:
        """Return the payload safe for dispatch under this result."""

        return self.sanitized_context

    @property
    def tool_outputs(self) -> Any:
        """Return tool outputs safe for dispatch under this result."""

        return self.sanitized_tool_outputs

    def to_dict(self) -> dict[str, Any]:
        """Return only the PHI-free report fields."""

        return self.report.to_dict()


class PreflightError(RuntimeError):
    """Base error for local preflight failures."""


class PreflightBlockedError(PreflightError):
    """Raised when fail-closed policy detects one or more findings."""

    def __init__(self, result: PreflightResult) -> None:
        self.result = result
        self.report = result.report
        super().__init__("agent context blocked by the local PHI preflight gate")


class PreflightInputError(PreflightError):
    """Raised when a supplied payload cannot be safely traversed."""


class PreflightScanError(PreflightError):
    """Raised when a scanner fails or returns an invalid offset."""


@dataclass(frozen=True)
class _TextLeaf:
    index: int
    text: str


def _collect_text_leaves(payload: Any) -> tuple[_TextLeaf, ...]:
    leaves: list[_TextLeaf] = []
    active: set[int] = set()
    total_bytes = 0

    def visit(value: Any, depth: int) -> None:
        nonlocal total_bytes
        if isinstance(value, (str, bytes, bytearray, memoryview)):
            try:
                text = validate_pii_input(value)
                encoded_length = len(text.encode("utf-8"))
            except Exception:
                raise PreflightInputError(
                    "agent payload contains invalid text"
                ) from None
            total_bytes += encoded_length
            if total_bytes > _MAX_PAYLOAD_BYTES:
                raise PreflightInputError(
                    "agent payload exceeds the configured byte limit"
                )
            if len(leaves) >= _MAX_PAYLOAD_LEAVES:
                raise PreflightInputError(
                    "agent payload exceeds the configured leaf limit"
                )
            leaves.append(_TextLeaf(index=len(leaves), text=text))
            return

        if value is None or isinstance(value, (bool, int, float, complex)):
            return
        if not isinstance(value, (Mapping, list, tuple)):
            return
        if depth >= _MAX_PAYLOAD_DEPTH:
            raise PreflightInputError(
                "agent payload exceeds the configured nesting limit"
            )
        object_id = id(value)
        if object_id in active:
            raise PreflightInputError("agent payload contains a recursive structure")
        active.add(object_id)
        try:
            if isinstance(value, Mapping):
                items = value.items()
                for _, item in items:
                    visit(item, depth + 1)
            else:
                for item in value:
                    visit(item, depth + 1)
        except PreflightError:
            raise
        except Exception:
            raise PreflightInputError("agent payload cannot be traversed") from None
        finally:
            active.remove(object_id)

    visit(payload, 0)
    return tuple(leaves)


def _scanner_items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (PreflightFinding, Mapping)):
        return (value,)
    if isinstance(value, (str, bytes, bytearray)):
        raise PreflightScanError("preflight scanner returned an invalid finding")
    if isinstance(value, Sequence) and len(value) in {2, 3}:
        first = value[0]
        last = value[-1]
        if isinstance(first, (str, int)) and isinstance(last, (str, int)):
            return (value,)
    try:
        return tuple(value)
    except Exception:
        return (value,)


def _scanner_candidate(candidate: Any) -> tuple[Any, Any, Any]:
    if isinstance(candidate, PreflightFinding):
        return candidate.category, candidate.start, candidate.end
    if isinstance(candidate, Mapping):
        category = candidate.get(
            "category",
            candidate.get("label", candidate.get("entity_type", "PHI")),
        )
        return category, candidate.get("start"), candidate.get("end")
    if isinstance(candidate, Sequence) and not isinstance(
        candidate, (str, bytes, bytearray)
    ):
        if len(candidate) == 2:
            return "PHI", candidate[0], candidate[1]
        if len(candidate) == 3:
            if isinstance(candidate[0], str):
                return candidate[0], candidate[1], candidate[2]
            if isinstance(candidate[2], str):
                return candidate[2], candidate[0], candidate[1]
    category = getattr(candidate, "category", getattr(candidate, "label", "PHI"))
    return category, getattr(candidate, "start", None), getattr(candidate, "end", None)


def _normalize_scanner_findings(
    raw_findings: Any,
    *,
    text: str,
    channel: str,
    payload_index: int,
) -> tuple[PreflightFinding, ...]:
    candidates: list[PreflightFinding] = []
    try:
        items = _scanner_items(raw_findings)
        for candidate in items:
            category, start, end = _scanner_candidate(candidate)
            if type(start) is not int or type(end) is not int:
                raise ValueError
            if start < 0 or end <= start or end > len(text):
                raise ValueError
            normalized_category = _normalize_category(category)
            candidates.append(
                PreflightFinding(
                    category=normalized_category,
                    start=start,
                    end=end,
                    channel=channel,
                    payload_index=payload_index,
                    text_hash=_surface_hash(text, start, end),
                )
            )
    except PreflightError:
        raise
    except Exception:
        raise PreflightScanError(
            "preflight scanner returned an invalid finding"
        ) from None

    selected: list[PreflightFinding] = []
    seen: set[tuple[str, int, int]] = set()
    for finding in sorted(
        candidates,
        key=lambda item: (
            item.start,
            -(item.end - item.start),
            item.category,
            item.end,
        ),
    ):
        identity = (finding.category, finding.start, finding.end)
        if identity in seen:
            continue
        if selected and finding.start < selected[-1].end:
            continue
        seen.add(identity)
        selected.append(finding)
    return tuple(selected)


def _default_scanner(text: str) -> list[Any]:
    """Run the local structured-identifier sweep without model inference."""

    return safety_sweep(text, ())


def _scan_payload(
    payload: Any,
    *,
    channel: str,
    scanner: PreflightScanner,
) -> tuple[tuple[PreflightFinding, ...], int]:
    leaves = _collect_text_leaves(payload)
    findings: list[PreflightFinding] = []
    for leaf in leaves:
        scanner_failed = False
        try:
            raw_findings = scanner(leaf.text)
        except Exception:
            scanner_failed = True
        if scanner_failed:
            raise PreflightScanError("preflight scanner failed")

        invalid_finding = False
        try:
            findings.extend(
                _normalize_scanner_findings(
                    raw_findings,
                    text=leaf.text,
                    channel=channel,
                    payload_index=leaf.index,
                )
            )
        except Exception:
            invalid_finding = True
        if invalid_finding:
            raise PreflightScanError("preflight scanner returned an invalid finding")
    return tuple(findings), len(leaves)


def _redact_text(value: Any, findings: Sequence[PreflightFinding]) -> Any:
    if not findings:
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        text = validate_pii_input(value)
    elif isinstance(value, str):
        text = value
    else:
        return value

    parts: list[str] = []
    cursor = 0
    for finding in findings:
        parts.append(text[cursor : finding.start])
        parts.append(f"[OPENMED_REDACTED_{finding.category}]")
        cursor = finding.end
    parts.append(text[cursor:])
    redacted = "".join(parts)
    if isinstance(value, bytes):
        return redacted.encode("utf-8")
    if isinstance(value, bytearray):
        return bytearray(redacted.encode("utf-8"))
    if isinstance(value, memoryview):
        return memoryview(redacted.encode("utf-8"))
    return redacted


def _redact_payload(
    payload: Any,
    findings_by_index: Mapping[int, Sequence[PreflightFinding]],
) -> Any:
    leaf_index = 0
    active: set[int] = set()

    def visit(value: Any, depth: int) -> Any:
        nonlocal leaf_index
        if isinstance(value, (str, bytes, bytearray, memoryview)):
            findings = findings_by_index.get(leaf_index, ())
            leaf_index += 1
            return _redact_text(value, findings)
        if value is None or isinstance(value, (bool, int, float, complex)):
            return value
        if not isinstance(value, (Mapping, list, tuple)):
            return value
        if depth >= _MAX_PAYLOAD_DEPTH:
            raise PreflightInputError(
                "agent payload exceeds the configured nesting limit"
            )
        object_id = id(value)
        if object_id in active:
            raise PreflightInputError("agent payload contains a recursive structure")
        active.add(object_id)
        try:
            if isinstance(value, Mapping):
                return {key: visit(item, depth + 1) for key, item in value.items()}
            if isinstance(value, list):
                return [visit(item, depth + 1) for item in value]
            return tuple(visit(item, depth + 1) for item in value)
        except PreflightError:
            raise
        except Exception:
            raise PreflightInputError("agent payload cannot be redacted") from None
        finally:
            active.remove(object_id)

    redacted = visit(payload, 0)
    if leaf_index != len(_collect_text_leaves(payload)):
        raise PreflightInputError("agent payload changed during redaction")
    return redacted


def preflight_context(
    context: Any,
    tool_outputs: Any = (),
    *,
    policy: PreflightPolicy | str = FAIL_CLOSED_POLICY,
    scanner: PreflightScanner | None = None,
    raise_on_block: bool = True,
) -> PreflightResult:
    """Scan context and tool outputs before an agent dispatch.

    Args:
        context: Caller-supplied context. Nested mappings, lists, tuples, and
            string or UTF-8 byte leaves are scanned in traversal order.
        tool_outputs: Tool result payloads to scan using the same rules.
        policy: ``"fail_closed"`` (default) blocks when findings are present;
            ``"redact_then_continue"`` replaces findings with stable tokens.
            ``"block"`` and ``"redact"`` are accepted aliases.
        scanner: Optional local scanner. It receives one string leaf at a time
            and returns mappings or tuples containing category and offsets.
        raise_on_block: Raise :class:`PreflightBlockedError` for blocked
            fail-closed results. Set to ``False`` to inspect the safe report.

    Returns:
        A result whose payload fields are safe for the selected policy. The
        report and findings never contain matched source text.

    Raises:
        PreflightBlockedError: If fail-closed policy finds sensitive spans and
            ``raise_on_block`` is true.
        PreflightError: If payload traversal or scanning cannot complete safely.
    """

    normalized_policy = _normalize_policy(policy)
    if type(raise_on_block) is not bool:
        raise ValueError("preflight raise_on_block is invalid")
    selected_scanner = scanner if scanner is not None else _default_scanner
    if not callable(selected_scanner):
        raise ValueError("preflight scanner is invalid")

    context_findings, context_count = _scan_payload(
        context,
        channel="context",
        scanner=selected_scanner,
    )
    output_findings, output_count = _scan_payload(
        tool_outputs,
        channel="tool_output",
        scanner=selected_scanner,
    )
    findings = tuple(
        sorted(
            (*context_findings, *output_findings),
            key=lambda item: (
                item.channel,
                item.payload_index,
                item.start,
                item.end,
                item.category,
            ),
        )
    )
    has_findings = bool(findings)
    blocked = normalized_policy == FAIL_CLOSED_POLICY and has_findings
    redacted = normalized_policy == REDACT_THEN_CONTINUE_POLICY and has_findings

    sanitized_context = context
    sanitized_tool_outputs = tool_outputs
    if redacted:
        context_by_index: dict[int, list[PreflightFinding]] = defaultdict(list)
        output_by_index: dict[int, list[PreflightFinding]] = defaultdict(list)
        for finding in context_findings:
            context_by_index[finding.payload_index].append(finding)
        for finding in output_findings:
            output_by_index[finding.payload_index].append(finding)
        sanitized_context = _redact_payload(context, context_by_index)
        sanitized_tool_outputs = _redact_payload(tool_outputs, output_by_index)
    elif blocked:
        sanitized_context = None
        sanitized_tool_outputs = None

    report = PreflightReport(
        policy=normalized_policy,
        allowed=not blocked,
        redacted=redacted,
        findings=findings,
        payload_count=context_count + output_count,
    )
    result = PreflightResult(
        report=report,
        sanitized_context=sanitized_context,
        sanitized_tool_outputs=sanitized_tool_outputs,
    )
    if blocked and raise_on_block:
        raise PreflightBlockedError(result)
    return result


def inspect_context(
    context: Any,
    tool_outputs: Any = (),
    *,
    scanner: PreflightScanner | None = None,
) -> PreflightResult:
    """Inspect payloads and return a non-raising fail-closed result."""

    return preflight_context(
        context,
        tool_outputs,
        policy=FAIL_CLOSED_POLICY,
        scanner=scanner,
        raise_on_block=False,
    )


def scan_context(
    context: Any,
    tool_outputs: Any = (),
    *,
    scanner: PreflightScanner | None = None,
) -> PreflightResult:
    """Compatibility alias for :func:`inspect_context`."""

    return inspect_context(context, tool_outputs, scanner=scanner)


def preflight(
    context: Any,
    tool_outputs: Any = (),
    *,
    policy: PreflightPolicy | str = FAIL_CLOSED_POLICY,
    scanner: PreflightScanner | None = None,
    raise_on_block: bool = True,
) -> PreflightResult:
    """Short alias for :func:`preflight_context`."""

    return preflight_context(
        context,
        tool_outputs,
        policy=policy,
        scanner=scanner,
        raise_on_block=raise_on_block,
    )


class PreflightGate:
    """Reusable configured gate for repeated agent dispatch checks."""

    def __init__(
        self,
        *,
        policy: PreflightPolicy | str = FAIL_CLOSED_POLICY,
        scanner: PreflightScanner | None = None,
    ) -> None:
        self.policy = _normalize_policy(policy)
        self.scanner = scanner

    def check(
        self,
        context: Any,
        tool_outputs: Any = (),
        *,
        raise_on_block: bool = True,
    ) -> PreflightResult:
        """Check one pair of context and tool-output payloads."""

        return preflight_context(
            context,
            tool_outputs,
            policy=self.policy,
            scanner=self.scanner,
            raise_on_block=raise_on_block,
        )

    __call__ = check


__all__ = [
    "FAIL_CLOSED",
    "FAIL_CLOSED_POLICY",
    "REDACT_THEN_CONTINUE",
    "REDACT_THEN_CONTINUE_POLICY",
    "PreflightBlockedError",
    "PreflightError",
    "PreflightFinding",
    "PreflightGate",
    "PreflightInputError",
    "PreflightPolicy",
    "PreflightReport",
    "PreflightResult",
    "PreflightScanError",
    "PreflightScanner",
    "inspect_context",
    "preflight",
    "preflight_context",
    "scan_context",
]
