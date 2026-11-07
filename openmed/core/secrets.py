"""Deterministic, raw-value-free detection of credentials and secret tokens.

The detector is deliberately local and conservative.  It recognizes credential
shapes that are useful to catch in agent traces, headers, configuration payloads,
and environment files without attempting to contact a provider or infer a
credential from high-entropy text alone.

Only :class:`SecretFinding` records leave the detector.  A finding contains the
category, the half-open character offsets of the sensitive value, and a stable
SHA-256 fingerprint for deduplication.  The matched value is never retained in
the finding or its serialized representation.

Example:
    >>> from openmed.core.secrets import detect_secrets
    >>> findings = detect_secrets("Authorization: Bearer " + "A" * 24)
    >>> findings[0].category
    'authorization_header'
"""

from __future__ import annotations

import hashlib
import re
from bisect import bisect_left
from dataclasses import dataclass
from typing import Final

AUTHORIZATION_HEADER: Final = "authorization_header"
"""Category for values found in authorization-style HTTP headers."""

ACCESS_TOKEN: Final = "access_token"
"""Category for recognizable access-token shapes."""

API_KEY: Final = "api_key"
"""Category for recognizable API-key shapes."""

ACCESS_KEY: Final = "access_key"
"""Category for recognizable provider access-key identifiers."""

PRIVATE_KEY: Final = "private_key"
"""Category for PEM or PGP private-key material."""

ENVIRONMENT_SECRET: Final = "environment_secret"
"""Category for secret-looking values assigned to configuration names."""

_SECRET_CATEGORIES = frozenset(
    {
        ACCESS_KEY,
        ACCESS_TOKEN,
        API_KEY,
        AUTHORIZATION_HEADER,
        ENVIRONMENT_SECRET,
        PRIVATE_KEY,
    }
)
_FINGERPRINT_PREFIX = "sha256:"
_FINGERPRINT_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MIN_SECRET_VALUE_LENGTH = 8
_MAX_ENVIRONMENT_VALUE_LENGTH = 4096


@dataclass(frozen=True, slots=True)
class SecretFinding:
    """A raw-value-free secret finding.

    Attributes:
        category: Stable detector category.
        offset: Half-open ``(start, end)`` character offsets into the input.
        fingerprint: One-way SHA-256 fingerprint of the sensitive value.

    The source value is intentionally not an attribute of this class.  The
    ``start`` and ``end`` properties are convenience accessors for callers that
    use the same span convention as the rest of OpenMed.
    """

    category: str
    offset: tuple[int, int]
    fingerprint: str

    def __post_init__(self) -> None:
        """Validate and normalize the safe, serialized fields."""
        try:
            start, end = self.offset
        except (TypeError, ValueError):
            raise ValueError("offset must contain exactly two integers") from None
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise ValueError("offset must contain a non-empty, non-negative span")
        if self.category not in _SECRET_CATEGORIES:
            raise ValueError("category must be a supported secret category")
        if (
            not isinstance(self.fingerprint, str)
            or _FINGERPRINT_PATTERN.fullmatch(self.fingerprint) is None
        ):
            raise ValueError("fingerprint must use the sha256 format")
        object.__setattr__(self, "offset", (start, end))

    @property
    def start(self) -> int:
        """Return the inclusive start offset."""
        return self.offset[0]

    @property
    def end(self) -> int:
        """Return the exclusive end offset."""
        return self.offset[1]

    def to_dict(self) -> dict[str, object]:
        """Return the only safe fields emitted by the detector."""
        return {
            "category": self.category,
            "offset": list(self.offset),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class _Candidate:
    """An internal span candidate before overlap resolution."""

    start: int
    end: int
    category: str
    priority: int


@dataclass(frozen=True, slots=True)
class _TokenPattern:
    """A known token shape and its safe output category."""

    pattern: re.Pattern[str]
    category: str
    priority: int


def _token_pattern(pattern: str, category: str, priority: int) -> _TokenPattern:
    return _TokenPattern(re.compile(pattern), category, priority)


_TOKEN_PATTERNS: tuple[_TokenPattern, ...] = (
    _token_pattern(
        r"(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9_]{20,255}(?![A-Za-z0-9_])",
        ACCESS_TOKEN,
        80,
    ),
    _token_pattern(
        r"(?<![A-Za-z0-9_])gh[pousr]_[A-Za-z0-9]{20,255}(?![A-Za-z0-9_])",
        ACCESS_TOKEN,
        80,
    ),
    _token_pattern(
        r"(?<![A-Za-z0-9_])xox[baprs]-[A-Za-z0-9-]{10,255}(?![A-Za-z0-9_])",
        ACCESS_TOKEN,
        80,
    ),
    _token_pattern(
        r"(?<![A-Za-z0-9_])xapp-[A-Za-z0-9-]{10,255}(?![A-Za-z0-9_])",
        ACCESS_TOKEN,
        80,
    ),
    _token_pattern(
        r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{20,255}(?![A-Za-z0-9_])",
        ACCESS_TOKEN,
        80,
    ),
    _token_pattern(
        r"(?<![A-Za-z0-9_])ya29\.[A-Za-z0-9_-]{20,255}(?![A-Za-z0-9_])",
        ACCESS_TOKEN,
        80,
    ),
    _token_pattern(
        r"(?<![A-Za-z0-9_])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_])",
        ACCESS_TOKEN,
        82,
    ),
    _token_pattern(
        r"(?<![A-Za-z0-9_])AKIA[0-9A-Z]{16}(?![A-Za-z0-9_])",
        ACCESS_KEY,
        84,
    ),
    _token_pattern(
        r"(?<![A-Za-z0-9_])ASIA[0-9A-Z]{16}(?![A-Za-z0-9_])",
        ACCESS_KEY,
        84,
    ),
    _token_pattern(
        r"(?<![A-Za-z0-9_])AIza[0-9A-Za-z_-]{30,40}(?![A-Za-z0-9_])",
        API_KEY,
        82,
    ),
    _token_pattern(
        r"(?<![A-Za-z0-9_])(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,255}(?![A-Za-z0-9_])",
        API_KEY,
        82,
    ),
    _token_pattern(
        r"(?<![A-Za-z0-9_])sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,255}(?![A-Za-z0-9_])",
        API_KEY,
        82,
    ),
    _token_pattern(
        r"(?<![A-Za-z0-9_])npm_[A-Za-z0-9]{20,255}(?![A-Za-z0-9_])",
        API_KEY,
        80,
    ),
    _token_pattern(
        r"(?<![A-Za-z0-9_])pypi-[A-Za-z0-9_-]{20,255}(?![A-Za-z0-9_])",
        API_KEY,
        80,
    ),
)

_PRIVATE_KEY_BOUNDARY_PATTERN = re.compile(
    r"-----(?P<action>BEGIN|END) "
    r"(?P<kind>(?:(?:RSA|EC|DSA|OPENSSH|ENCRYPTED) )?PRIVATE KEY|"
    r"PGP PRIVATE KEY BLOCK)-----"
)

_AUTHORIZATION_HEADER_PATTERN = re.compile(
    r"(?ix)"
    r"(?<![A-Za-z0-9_-])"
    r"(?:authorization|proxy-authorization|x-api-key|x-auth-token|"
    r"x-access-token|api-key|api-token|client-secret)"
    r"\s*:\s*"
    r"(?:(?:bearer|basic|token)\s+)?"
    r"(?:\"[^\r\n\"]*\"|'[^\r\n']*'|[^\s,;\]}#]+)"
)
_PARAMETERIZED_AUTHORIZATION_HEADER_PATTERN = re.compile(
    r"(?imx)"
    r"(?<![A-Za-z0-9_-])"
    r"(?:authorization|proxy-authorization)"
    r"\s*:\s*"
    r"(?P<value>(?:aws4-hmac-sha256|digest|signature)\s+[^\r\n]+)"
)

_ENVIRONMENT_ASSIGNMENT_PATTERN = re.compile(
    r"(?ix)"
    r"(?<![A-Za-z0-9_-])"
    r"[A-Za-z][A-Za-z0-9_-]{0,95}"
    r"\s*[:=]\s*"
    r"(?:\"[^\r\n\"]*\"|'[^\r\n']*'|\[[^\r\n\]]*\]|[^\s,;\]}#]+)"
)

_SECRET_NAME_PARTS = frozenset(
    {
        "api",
        "auth",
        "access",
        "bearer",
        "client",
        "credential",
        "database",
        "db",
        "encryption",
        "jwt",
        "key",
        "passphrase",
        "password",
        "passwd",
        "private",
        "secret",
        "signing",
        "token",
    }
)
_EXACT_PLACEHOLDERS = frozenset(
    {
        "",
        "null",
        "none",
        "false",
        "true",
        "changeme",
        "change-me",
        "example",
        "example-value",
        "placeholder",
        "redacted",
        "masked",
        "sample",
        "test",
        "testing",
        "dummy",
        "secret",
        "token",
        "value",
        "your-token",
        "your_token",
        "your-secret",
        "your_secret",
        "<token>",
        "<secret>",
        "<value>",
        "<api-key>",
        "<api_key>",
        "[api-key]",
        "[api_key]",
        "[masked]",
        "[redacted]",
        "[secret]",
        "[token]",
        "[value]",
    }
)
_PLACEHOLDER_PREFIXES = (
    "change-me",
    "changeme",
    "example-",
    "example_",
    "placeholder-",
    "placeholder_",
    "replace-me",
    "replace_me",
    "your-",
    "your_",
)
_RUNTIME_VALUE_PATTERN = re.compile(
    r"(?ix)^(?:\$\{|\$[A-Za-z_]|\{\{|os\.environ|os\.getenv|"
    r"getenv\(|process\.env|secrets\.)"
)


def _is_placeholder(value: str) -> bool:
    """Return whether a value is an explicit example or runtime reference."""
    normalized = value.strip().casefold()
    if normalized in _EXACT_PLACEHOLDERS:
        return True
    if normalized.startswith(_PLACEHOLDER_PREFIXES):
        return True
    if _RUNTIME_VALUE_PATTERN.match(normalized):
        return True
    compact = re.sub(r"[\s_-]", "", normalized)
    if compact and len(compact) >= 4 and set(compact) <= {"x", "*", "#"}:
        return True
    return False


def _looks_like_secret(value: str) -> bool:
    """Apply the shared minimum-shape and placeholder checks."""
    normalized = value.strip()
    if not (
        _MIN_SECRET_VALUE_LENGTH <= len(normalized) <= _MAX_ENVIRONMENT_VALUE_LENGTH
    ):
        return False
    if _is_placeholder(normalized):
        return False
    return any(character.isalnum() for character in normalized)


def _is_secret_name(name: str) -> bool:
    """Return whether a configuration name strongly implies secret material."""
    normalized = name.casefold().replace("-", "_")
    parts = tuple(part for part in normalized.split("_") if part)
    if not parts:
        return False
    if normalized in {
        "api_key",
        "access_key",
        "access_token",
        "auth_token",
        "authorization",
        "bearer_token",
        "client_secret",
        "credential",
        "database_url",
        "db_url",
        "encryption_key",
        "jwt",
        "passphrase",
        "password",
        "passwd",
        "private_key",
        "secret",
        "secret_key",
        "secret_token",
        "signing_key",
        "token",
    }:
        return True
    if parts[-1] in {"password", "passwd", "passphrase", "secret", "token"}:
        return True
    if parts[-2:] in {
        ("api", "key"),
        ("access", "key"),
        ("access", "token"),
        ("auth", "token"),
        ("bearer", "token"),
        ("client", "secret"),
        ("database", "url"),
        ("db", "url"),
        ("encryption", "key"),
        ("private", "key"),
        ("secret", "access", "key"),
        ("secret", "key"),
        ("secret", "token"),
        ("signing", "key"),
    }:
        return True
    return bool(set(parts) & _SECRET_NAME_PARTS) and any(
        part in {"secret", "token", "password", "passwd", "passphrase"}
        for part in parts
    )


def _trim_value_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim syntax delimiters without changing the source text or offsets."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1

    if end - start >= 2 and text[start] == text[end - 1] and text[start] in "\"'":
        start += 1
        end -= 1

    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _candidate(
    text: str,
    start: int,
    end: int,
    category: str,
    priority: int,
    *,
    allow_short: bool = False,
) -> _Candidate | None:
    """Build a candidate after validating only local, non-sensitive metadata."""
    if start < 0 or end <= start or end > len(text):
        return None
    surface = text[start:end]
    if not allow_short and not _looks_like_secret(surface):
        return None
    return _Candidate(start, end, category, priority)


def _header_candidate(text: str, match: re.Match[str]) -> _Candidate | None:
    """Extract the value portion of an authorization-style header."""
    matched = text[match.start() : match.end()]
    separator = re.search(r":\s*", matched)
    if separator is None:
        return None
    start = match.start() + separator.end()
    end = match.end()
    start, end = _trim_value_bounds(text, start, end)
    if start >= end:
        return None

    value_text = text[start:end]
    scheme = re.match(r"(?is)(?:bearer|basic|token)\s+", value_text)
    if scheme is not None:
        start += scheme.end()
    start, end = _trim_value_bounds(text, start, end)
    return _candidate(
        text,
        start,
        end,
        AUTHORIZATION_HEADER,
        92,
    )


def _environment_candidate(text: str, match: re.Match[str]) -> _Candidate | None:
    """Extract a value from a secret-looking configuration assignment."""
    assignment = text[match.start() : match.end()]
    separator = re.search(r"\s*[:=]\s*", assignment)
    if separator is None:
        return None
    name = assignment[: separator.start()].strip(" \t\"'")
    if not _is_secret_name(name):
        return None

    value_start = match.start() + separator.end()
    value_start, value_end = _trim_value_bounds(text, value_start, match.end())
    return _candidate(
        text,
        value_start,
        value_end,
        ENVIRONMENT_SECRET,
        60,
    )


def _parameterized_header_candidate(
    text: str,
    match: re.Match[str],
) -> _Candidate | None:
    """Extract a complete parameterized Authorization value."""
    start, end = match.span("value")
    start, end = _trim_value_bounds(text, start, end)
    return _candidate(
        text,
        start,
        end,
        AUTHORIZATION_HEADER,
        94,
    )


def _private_key_candidates(text: str) -> list[_Candidate]:
    """Find private-key blocks and fail closed on unmatched begin markers."""
    candidates: list[_Candidate] = []
    pending: dict[str, list[tuple[int, int]]] = {}

    for match in _PRIVATE_KEY_BOUNDARY_PATTERN.finditer(text):
        kind = match.group("kind")
        if match.group("action") == "BEGIN":
            pending.setdefault(kind, []).append(match.span())
            continue

        begins = pending.pop(kind, ())
        if begins:
            candidates.append(
                _Candidate(
                    start=begins[0][0],
                    end=match.end(),
                    category=PRIVATE_KEY,
                    priority=100,
                )
            )

    for spans in pending.values():
        if spans:
            candidates.append(_Candidate(spans[0][0], len(text), PRIVATE_KEY, 99))
    return candidates


def _collect_candidates(text: str) -> list[_Candidate]:
    """Collect deterministic candidates without retaining matched surfaces."""
    candidates: list[_Candidate] = []

    for token_pattern in _TOKEN_PATTERNS:
        for match in token_pattern.pattern.finditer(text):
            candidate = _candidate(
                text,
                *match.span(),
                token_pattern.category,
                token_pattern.priority,
            )
            if candidate is not None:
                candidates.append(candidate)

    candidates.extend(_private_key_candidates(text))

    for match in _AUTHORIZATION_HEADER_PATTERN.finditer(text):
        candidate = _header_candidate(text, match)
        if candidate is not None:
            candidates.append(candidate)

    for match in _PARAMETERIZED_AUTHORIZATION_HEADER_PATTERN.finditer(text):
        candidate = _parameterized_header_candidate(text, match)
        if candidate is not None:
            candidates.append(candidate)

    for match in _ENVIRONMENT_ASSIGNMENT_PATTERN.finditer(text):
        candidate = _environment_candidate(text, match)
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def _select_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
    """Resolve overlaps by confidence priority, then by stable source order."""
    ranked = sorted(
        candidates,
        key=lambda item: (
            -item.priority,
            item.start,
            -(item.end - item.start),
            item.category,
        ),
    )
    selected: list[_Candidate] = []
    selected_starts: list[int] = []
    for candidate in ranked:
        insertion_index = bisect_left(selected_starts, candidate.start)
        overlaps_previous = (
            insertion_index > 0 and selected[insertion_index - 1].end > candidate.start
        )
        overlaps_next = (
            insertion_index < len(selected)
            and selected[insertion_index].start < candidate.end
        )
        if overlaps_previous or overlaps_next:
            continue
        selected.insert(insertion_index, candidate)
        selected_starts.insert(insertion_index, candidate.start)
    return selected


def _fingerprint(surface: str) -> str:
    """Return the stable, one-way fingerprint used for deduplication."""
    digest = hashlib.sha256(surface.encode("utf-8", errors="surrogatepass"))
    return f"{_FINGERPRINT_PREFIX}{digest.hexdigest()}"


class SecretDetector:
    """Scan text for high-confidence credentials using local rules only."""

    def scan(self, text: str) -> list[SecretFinding]:
        """Return deterministic, raw-value-free findings for ``text``.

        Args:
            text: Input text to inspect.  It is used only during this call to
                calculate offsets and fingerprints.

        Raises:
            TypeError: If ``text`` is not a string.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return [
            SecretFinding(
                category=candidate.category,
                offset=(candidate.start, candidate.end),
                fingerprint=_fingerprint(text[candidate.start : candidate.end]),
            )
            for candidate in _select_candidates(_collect_candidates(text))
        ]

    def detect(self, text: str) -> list[SecretFinding]:
        """Alias for :meth:`scan` for detector-style call sites."""
        return self.scan(text)

    def __call__(self, text: str) -> list[SecretFinding]:
        """Allow a detector instance to be used as a callable scan pass."""
        return self.scan(text)


def detect_secrets(text: str) -> list[SecretFinding]:
    """Detect credentials and secret tokens without returning raw values."""
    return SecretDetector().scan(text)


def scan_secrets(text: str) -> list[SecretFinding]:
    """Compatibility alias for :func:`detect_secrets`."""
    return detect_secrets(text)


def find_secrets(text: str) -> list[SecretFinding]:
    """Compatibility alias for :func:`detect_secrets`."""
    return detect_secrets(text)


SecretMatch = SecretFinding


__all__ = [
    "ACCESS_KEY",
    "ACCESS_TOKEN",
    "API_KEY",
    "AUTHORIZATION_HEADER",
    "ENVIRONMENT_SECRET",
    "PRIVATE_KEY",
    "SecretDetector",
    "SecretFinding",
    "SecretMatch",
    "detect_secrets",
    "find_secrets",
    "scan_secrets",
]
