"""Fail-closed limits for untrusted operational inputs.

Checks in this module run before caller-selected parsers, decompression, model
execution, or storage work.  Errors expose stable reason codes and aggregate
counts only; input values and archive member names are never returned.
"""

from __future__ import annotations

import codecs
import io
import json
import math
import re
import stat
import zipfile
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from openmed.interop.archive_safety import (
    ArchiveMember,
    ArchiveSafetyPolicy,
    ArchiveSafetyReport,
    inspect_archive_members,
)

OPERATIONAL_LIMIT_SCHEMA_VERSION: Final = "1.0.0"
OPERATIONAL_LIMIT_COMPATIBILITY: Final = "same_major"
_LIMIT_CODES: Final = frozenset(
    {
        "archive_allowed",
        "archive_encrypted",
        "archive_invalid",
        "archive_limit_denied",
        "archive_type_invalid",
        "page_size_allowed",
        "page_size_exceeded",
        "page_size_invalid",
        "payload_size_allowed",
        "payload_size_exceeded",
        "payload_size_invalid",
    }
)


class LimitState(str, Enum):
    """Typed outcomes for a pre-work resource-limit decision."""

    SUCCESS = "success"
    DENIED = "denied"
    UNSUPPORTED = "unsupported"
    FAILURE = "failure"


class OperationalLimitError(ValueError):
    """Content-free exception raised before expensive work begins."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


JSON_STRUCTURE_EXCEEDED: Final = "json_structure_exceeded"


class JsonStructureCategory(str, Enum):
    """Structural JSON budget that a scanned document exceeded."""

    DEPTH = "depth"
    NODES = "nodes"
    STRING_BYTES = "string_bytes"


class JsonStructureLimitError(OperationalLimitError):
    """Content-free structural violation carrying only category and ceiling.

    Attributes:
        category: The exceeded budget.
        ceiling: The configured maximum for that budget.
    """

    def __init__(self, category: JsonStructureCategory | str, ceiling: int) -> None:
        self.category = JsonStructureCategory(category)
        self.ceiling = int(ceiling)
        super().__init__(JSON_STRUCTURE_EXCEEDED)

    def to_dict(self) -> dict[str, object]:
        """Return the limit category and ceiling, never document content."""

        return {"category": self.category.value, "ceiling": self.ceiling}


@dataclass(frozen=True, slots=True)
class OperationalLimits:
    """Versioned upper bounds for request, parser, and archive work."""

    max_request_bytes: int = 16 * 1024 * 1024
    max_json_depth: int = 32
    max_json_nodes: int = 100_000
    max_json_string_bytes: int = 4 * 1024 * 1024
    max_page_size: int = 100
    max_archive_entries: int = 1_024
    max_archive_member_bytes: int = 128 * 1024 * 1024
    max_archive_total_bytes: int = 512 * 1024 * 1024
    max_archive_expansion_ratio: float = 100.0
    schema_version: str = OPERATIONAL_LIMIT_SCHEMA_VERSION
    compatibility_policy: str = OPERATIONAL_LIMIT_COMPATIBILITY

    def __post_init__(self) -> None:
        integer_limits = {
            "max_request_bytes": (1, 256 * 1024 * 1024),
            "max_json_depth": (1, 128),
            "max_json_nodes": (1, 1_000_000),
            "max_json_string_bytes": (1, 64 * 1024 * 1024),
            "max_page_size": (1, 10_000),
            "max_archive_entries": (1, 10_000),
            "max_archive_member_bytes": (1, 128 * 1024 * 1024),
            "max_archive_total_bytes": (1, 512 * 1024 * 1024),
        }
        for name, (minimum, maximum) in integer_limits.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError("operational limit is outside the safe range")
        ratio = self.max_archive_expansion_ratio
        if type(ratio) not in {int, float} or not 1 <= ratio <= 100:
            raise ValueError("archive expansion limit is outside the safe range")
        if self.schema_version != OPERATIONAL_LIMIT_SCHEMA_VERSION:
            raise ValueError("unsupported operational-limit schema version")
        if self.compatibility_policy != OPERATIONAL_LIMIT_COMPATIBILITY:
            raise ValueError("unsupported operational-limit compatibility policy")
        object.__setattr__(self, "max_archive_expansion_ratio", float(ratio))

    def archive_policy(self) -> ArchiveSafetyPolicy:
        """Return the existing metadata-only archive policy for these limits."""

        return ArchiveSafetyPolicy(
            max_entries=self.max_archive_entries,
            max_total_uncompressed_bytes=self.max_archive_total_bytes,
            max_member_uncompressed_bytes=self.max_archive_member_bytes,
            max_expansion_ratio=self.max_archive_expansion_ratio,
        )


@dataclass(frozen=True, slots=True)
class LimitDecision:
    """Value-free resource decision with aggregate observations."""

    state: LimitState
    code: str
    observed_count: int = 0
    schema_version: str = OPERATIONAL_LIMIT_SCHEMA_VERSION
    compatibility_policy: str = OPERATIONAL_LIMIT_COMPATIBILITY

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", LimitState(self.state))
        if self.code not in _LIMIT_CODES:
            raise ValueError("limit code is unsupported")
        if type(self.observed_count) is not int or self.observed_count < 0:
            raise ValueError("limit observation must be a non-negative count")
        if self.schema_version != OPERATIONAL_LIMIT_SCHEMA_VERSION:
            raise ValueError("unsupported limit-decision schema version")
        if self.compatibility_policy != OPERATIONAL_LIMIT_COMPATIBILITY:
            raise ValueError("unsupported limit-decision compatibility policy")

    @property
    def allowed(self) -> bool:
        return self.state is LimitState.SUCCESS

    def to_dict(self) -> dict[str, object]:
        """Return controlled metadata without any input values."""

        return {
            "code": self.code,
            "compatibility_policy": self.compatibility_policy,
            "observed_count": self.observed_count,
            "schema_version": self.schema_version,
            "state": self.state.value,
        }


@dataclass(frozen=True, slots=True)
class ArchiveLimitDecision:
    """Archive decision containing aggregate metadata only."""

    state: LimitState
    code: str
    report: ArchiveSafetyReport | None = None
    schema_version: str = OPERATIONAL_LIMIT_SCHEMA_VERSION
    compatibility_policy: str = OPERATIONAL_LIMIT_COMPATIBILITY

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", LimitState(self.state))
        if self.code not in _LIMIT_CODES:
            raise ValueError("archive-limit code is unsupported")
        if self.schema_version != OPERATIONAL_LIMIT_SCHEMA_VERSION:
            raise ValueError("unsupported archive-limit schema version")
        if self.compatibility_policy != OPERATIONAL_LIMIT_COMPATIBILITY:
            raise ValueError("unsupported archive-limit compatibility policy")

    @property
    def allowed(self) -> bool:
        return self.state is LimitState.SUCCESS

    def to_dict(self) -> dict[str, object]:
        """Return counts only, never archive paths or payload bytes."""

        return {
            "code": self.code,
            "compatibility_policy": self.compatibility_policy,
            "report": None if self.report is None else self.report.to_dict(),
            "schema_version": self.schema_version,
            "state": self.state.value,
        }


def validate_payload_size(
    byte_count: int,
    *,
    limits: OperationalLimits | None = None,
) -> LimitDecision:
    """Reject a declared or observed byte count before parsing."""

    active = limits or OperationalLimits()
    if type(byte_count) is not int or byte_count < 0:
        return LimitDecision(LimitState.FAILURE, "payload_size_invalid")
    if byte_count > active.max_request_bytes:
        return LimitDecision(
            LimitState.DENIED,
            "payload_size_exceeded",
            active.max_request_bytes,
        )
    return LimitDecision(LimitState.SUCCESS, "payload_size_allowed", byte_count)


def validate_page_size(
    page_size: int,
    *,
    limits: OperationalLimits | None = None,
) -> LimitDecision:
    """Reject unbounded or malformed pagination before a query executes."""

    active = limits or OperationalLimits()
    if type(page_size) is not int or page_size < 1:
        return LimitDecision(LimitState.FAILURE, "page_size_invalid")
    if page_size > active.max_page_size:
        return LimitDecision(
            LimitState.DENIED,
            "page_size_exceeded",
            active.max_page_size,
        )
    return LimitDecision(LimitState.SUCCESS, "page_size_allowed", page_size)


def parse_bounded_json(
    payload: bytes,
    *,
    limits: OperationalLimits | None = None,
) -> Any:
    """Parse JSON only after byte, node, depth, and string budgets pass."""

    active = limits or OperationalLimits()
    if not isinstance(payload, bytes):
        raise OperationalLimitError("payload_type_invalid")
    decision = validate_payload_size(len(payload), limits=active)
    if not decision.allowed:
        raise OperationalLimitError(decision.code)
    _preflight_json(payload, active)
    try:
        value = json.loads(payload, parse_constant=_reject_json_constant)
    except (RecursionError, UnicodeError, ValueError):
        raise OperationalLimitError("json_invalid") from None
    node_count = 0
    string_bytes = 0
    stack = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        node_count += 1
        if node_count > active.max_json_nodes:
            raise OperationalLimitError("json_node_limit_exceeded")
        if depth > active.max_json_depth:
            raise OperationalLimitError("json_depth_limit_exceeded")
        if isinstance(current, str):
            string_bytes += _utf8_length(current)
        elif isinstance(current, dict):
            for key, item in current.items():
                string_bytes += _utf8_length(str(key))
                stack.append((item, depth + 1))
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
        elif isinstance(current, float) and not math.isfinite(current):
            raise OperationalLimitError("json_invalid")
        if string_bytes > active.max_json_string_bytes:
            raise OperationalLimitError("json_string_limit_exceeded")
    return value


def _reject_json_constant(_: str) -> None:
    raise OperationalLimitError("json_invalid")


def _utf8_length(value: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError:
        raise OperationalLimitError("json_invalid") from None


_PREFLIGHT_CODES: Final = {
    JsonStructureCategory.DEPTH: "json_depth_limit_exceeded",
    JsonStructureCategory.NODES: "json_node_limit_exceeded",
    JsonStructureCategory.STRING_BYTES: "json_string_limit_exceeded",
}


def _preflight_json(payload: bytes, limits: OperationalLimits) -> None:
    """Bound JSON structure lexically before invoking the decoder."""

    scanner = JsonStructureScanner(limits, cumulative_strings=True)
    try:
        scanner.feed(payload)
        scanner.finish()
    except JsonStructureLimitError as error:
        raise OperationalLimitError(_PREFLIGHT_CODES[error.category]) from None
    if not scanner.is_complete:
        raise OperationalLimitError("json_invalid")


_OPEN_OBJECT: Final = 0x7B
_QUOTE: Final = 0x22
# Longest escape that may need lookahead: a surrogate pair is 12 bytes.
_MAX_ESCAPE_BYTES: Final = 12
# String content is matched in bounded windows so the regex engine never keeps
# per-escape backtracking state for more than this many bytes at a time.
_STRING_WINDOW_BYTES: Final = 16 * 1024
# Counted tokens are string starts, container opens and closes, and new
# scalars. Valid JSON has at most three per value (a key, the value and a
# close), so this budget is never reached before the node ceiling; it bounds
# CPU for malformed floods such as keys without values.
_TOKENS_PER_NODE: Final = 4
_TOKEN_SLACK: Final = 64
# Group 1: separators. Group 2: an optional escape-free object key with its
# colon. Then one of: 3 escape-free string completed in this buffer, 4 any
# other string start, 5 container open, 6 run of container closes, 7 scalar
# run (number or literal). Long strings are matched in C within fixed windows,
# so Python work is one match per token.
_TOKEN: Final = re.compile(
    rb'([ \t\n\r,:]*)(?:"([^"\\]*)"[ \t\n\r]*:[ \t\n\r]*)?'
    rb'(?:"([^"\\]*)"|(")|([\[{])|([\]}]+)|([^ \t\n\r,:\[\]{}"]+))?'
)
_STRING_CONTENT: Final = re.compile(
    rb'(?:[^"\\]+'
    rb'|\\["\\/bfnrt]'
    rb"|\\u[dD][89abAB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2}"
    rb"|\\u[dD][89abAB][0-9a-fA-F]{2}"
    rb'(?=[^\\]|\\["\\/bfnrt]|\\u(?:[0-9a-cA-CeEfF]|[dD][0-9abAB]))'
    rb"|\\u(?![dD][89abAB])[0-9a-fA-F]{4})*"
)
_LENIENT_STRING_DECODER: Final = json.JSONDecoder(strict=False)
_NON_ASCII_BYTES: Final = bytes(range(0x80, 0x100))


def _decoded_utf8_width(fragment: bytes) -> int:
    """Return the UTF-8 length of a whole-escape JSON string fragment."""

    if b"\\" not in fragment:
        return len(fragment)
    # Latin-1 maps each raw byte to one character, so the C decoder resolves
    # escapes without rejecting a UTF-8 sequence split across fragments. Raw
    # non-ASCII bytes then encode as two bytes each and are corrected below.
    decoded = _LENIENT_STRING_DECODER.decode('"' + fragment.decode("latin-1") + '"')
    non_ascii = len(fragment) - len(fragment.translate(None, _NON_ASCII_BYTES))
    return len(decoded.encode("utf-8", "surrogatepass")) - non_ascii


class JsonStructureScanner:
    """Single-pass, non-recursive structural scan of a streamed JSON body.

    The scanner bounds the work a JSON parser would do without building any
    value. Feed body chunks in order, then call :meth:`finish`. The first
    violation raises :class:`JsonStructureLimitError` and every later call
    raises it again. Budgets follow :class:`OperationalLimits`:

    * ``max_json_depth`` bounds container nesting; ``[]`` has depth one and
      scalars add no depth.
    * ``max_json_nodes`` bounds JSON values: every object, array, string,
      number, ``true``, ``false`` and ``null`` (including the root) counts
      once. Object keys are not values.
    * ``max_json_string_bytes`` bounds the decoded UTF-8 length of the longest
      single string or object key, with escapes such as ``\\u00e9`` measured
      after decoding. ``cumulative_strings=True`` bounds the total of all
      strings instead.

    UTF-16 and UTF-32 bodies are transcoded first, matching the encodings
    ``json.loads`` accepts for bytes. Input the JSON grammar cannot accept,
    such as an invalid escape, halts the scan without raising so the parser
    reports its own syntax error; :attr:`is_complete` is then false.
    """

    def __init__(
        self,
        limits: OperationalLimits | None = None,
        *,
        cumulative_strings: bool = False,
    ) -> None:
        """Create a scanner.

        Args:
            limits: Structural budgets; defaults to ``OperationalLimits()``.
            cumulative_strings: Bound the total of all string bytes instead of
                the longest single string.
        """

        self._limits = limits or OperationalLimits()
        self._cumulative_strings = cumulative_strings
        self._token_budget = (
            _TOKENS_PER_NODE * self._limits.max_json_nodes + _TOKEN_SLACK
        )
        self._tokens = 0
        self._prefix = b""
        self._encoding_known = False
        self._decoder: codecs.IncrementalDecoder | None = None
        self._carry = b""
        self._stack: list[int] = []
        self._expect_key = False
        self._awaiting_colon = False
        self._in_string = False
        self._in_scalar = False
        self._string_bytes = 0
        self._total_string_bytes = 0
        self._longest_string_bytes = 0
        self._nodes = 0
        self._max_depth = 0
        self._underflow = False
        self._halted = False
        self._finished = False
        self._violation: tuple[JsonStructureCategory, int] | None = None

    @property
    def node_count(self) -> int:
        """Number of JSON values seen so far."""

        return self._nodes

    @property
    def max_depth(self) -> int:
        """Deepest container nesting seen so far."""

        return self._max_depth

    @property
    def longest_string_bytes(self) -> int:
        """Decoded UTF-8 length of the longest string or key seen so far."""

        return self._longest_string_bytes

    @property
    def is_complete(self) -> bool:
        """Whether a finished scan saw balanced, fully terminated structure."""

        return (
            self._finished
            and self._violation is None
            and not self._halted
            and not self._underflow
            and not self._stack
            and not self._in_string
            and not self._carry
        )

    def feed(self, chunk: bytes) -> None:
        """Scan the next body chunk.

        Args:
            chunk: The next bytes of the body, in arrival order.

        Raises:
            JsonStructureLimitError: A structural budget is exceeded.
            ValueError: The scan was already finished.
        """

        self._raise_recorded_violation()
        if self._finished:
            raise ValueError("JSON structure scan is already finished")
        if chunk and not self._halted:
            self._guarded(bytes(chunk), final=False)

    def finish(self) -> None:
        """Flush buffered bytes after the last chunk.

        Raises:
            JsonStructureLimitError: A structural budget is exceeded.
        """

        self._raise_recorded_violation()
        if self._finished:
            return
        self._finished = True
        if not self._halted:
            self._guarded(b"", final=True)

    def _raise_recorded_violation(self) -> None:
        if self._violation is not None:
            raise JsonStructureLimitError(*self._violation)

    def _guarded(self, data: bytes, *, final: bool) -> None:
        try:
            self._feed_bytes(data, final=final)
        except JsonStructureLimitError as error:
            self._violation = (error.category, error.ceiling)
            raise

    def _feed_bytes(self, data: bytes, *, final: bool) -> None:
        if not self._encoding_known:
            data = self._prefix + data
            if len(data) < 4 and not final:
                self._prefix = data
                return
            self._prefix = b""
            self._encoding_known = True
            encoding = json.detect_encoding(data)
            if encoding == "utf-8-sig":
                data = data[len(codecs.BOM_UTF8) :]
            elif encoding != "utf-8":
                self._decoder = codecs.getincrementaldecoder(encoding)("surrogatepass")
        if self._decoder is not None:
            try:
                text = self._decoder.decode(data, final)
            except UnicodeDecodeError:
                self._halted = True
                return
            data = text.encode("utf-8", "surrogatepass")
        self._scan(data, final=final)

    def _scan(self, data: bytes, *, final: bool) -> None:
        if self._carry:
            data = self._carry + data
            self._carry = b""
        end = len(data)
        position = 0
        # Hot-loop state lives in locals and is written back on every exit.
        stack = self._stack
        nodes = self._nodes
        tokens = self._tokens
        expect_key = self._expect_key
        awaiting_colon = self._awaiting_colon
        in_scalar = self._in_scalar
        max_nodes = self._limits.max_json_nodes
        token_budget = self._token_budget
        match_token = _TOKEN.match
        try:
            while position < end:
                if self._in_string:
                    position = self._scan_string(data, position, end, final=final)
                    if self._halted:
                        return
                    continue
                match = match_token(data, position)
                separators_end = match.end(1)
                kind = match.lastindex
                if separators_end > position:
                    in_scalar = False
                    if (
                        stack
                        and stack[-1] == _OPEN_OBJECT
                        and data.find(b",", position, separators_end) != -1
                    ):
                        expect_key = True
                    if awaiting_colon:
                        awaiting_colon = data.find(b":", position, separators_end) == -1
                position = match.end()
                if kind == 1:
                    continue
                if kind == 7 and in_scalar:
                    # The rest of a number or literal split across chunks.
                    in_scalar = position == end
                    continue
                tokens += 1
                if tokens > token_budget or awaiting_colon:
                    if awaiting_colon:
                        # A key without a colon; every JSON parser stops here.
                        self._halted = True
                        return
                    self._exceeded(JsonStructureCategory.NODES)
                in_scalar = False
                if match.start(2) >= 0:
                    # ``"key":`` matched as one token, so the key needs no colon.
                    if not (expect_key and stack and stack[-1] == _OPEN_OBJECT):
                        self._halted = True
                        return
                    expect_key = False
                    self._observe_string(match.end(2) - match.start(2))
                    if kind == 6 or (kind == 2 and position < end):
                        # A close or separator where the key's value belongs.
                        self._halted = True
                        return
                    if kind == 2:
                        continue
                if kind == 3 or kind == 4:
                    if expect_key and stack and stack[-1] == _OPEN_OBJECT:
                        expect_key = False
                        awaiting_colon = True
                    else:
                        nodes += 1
                        if nodes > max_nodes:
                            self._exceeded(JsonStructureCategory.NODES)
                    if kind == 3:
                        self._observe_string(match.end(3) - match.start(3))
                    else:
                        self._string_bytes = 0
                        self._in_string = True
                elif kind == 5:
                    nodes += 1
                    if nodes > max_nodes:
                        self._exceeded(JsonStructureCategory.NODES)
                    opened = data[position - 1]
                    stack.append(opened)
                    if len(stack) > self._max_depth:
                        self._max_depth = len(stack)
                        if len(stack) > self._limits.max_json_depth:
                            self._exceeded(JsonStructureCategory.DEPTH)
                    expect_key = opened == _OPEN_OBJECT
                elif kind == 6:
                    closed = position - match.start(6)
                    if closed > len(stack):
                        # An unmatched close; every JSON parser stops here too.
                        self._underflow = True
                        self._halted = True
                        return
                    del stack[len(stack) - closed :]
                    expect_key = False
                else:
                    nodes += 1
                    if nodes > max_nodes:
                        self._exceeded(JsonStructureCategory.NODES)
                    in_scalar = position == end
        finally:
            self._nodes = nodes
            self._tokens = tokens
            self._expect_key = expect_key
            self._awaiting_colon = awaiting_colon
            self._in_scalar = in_scalar

    def _scan_string(self, data: bytes, position: int, end: int, *, final: bool) -> int:
        window_end = min(end, position + _STRING_WINDOW_BYTES)
        stop = _STRING_CONTENT.match(data, position, window_end).end()
        if stop > position:
            self._add_string_bytes(_decoded_utf8_width(data[position:stop]))
        if stop < end and data[stop] == _QUOTE:
            self._in_string = False
            return stop + 1
        if stop > position or stop == end:
            return stop
        # A backslash that could not be matched: either an escape cut by the
        # end of this chunk, or an escape the JSON grammar rejects.
        if not final and end - stop < _MAX_ESCAPE_BYTES:
            self._carry = data[stop:end]
            return end
        self._halted = True
        return end

    def _observe_string(self, width: int) -> None:
        self._string_bytes = 0
        self._add_string_bytes(width)

    def _add_string_bytes(self, width: int) -> None:
        self._string_bytes += width
        self._total_string_bytes += width
        if self._string_bytes > self._longest_string_bytes:
            self._longest_string_bytes = self._string_bytes
        observed = (
            self._total_string_bytes if self._cumulative_strings else self._string_bytes
        )
        if observed > self._limits.max_json_string_bytes:
            self._exceeded(JsonStructureCategory.STRING_BYTES)

    def _exceeded(self, category: JsonStructureCategory) -> None:
        ceilings = {
            JsonStructureCategory.DEPTH: self._limits.max_json_depth,
            JsonStructureCategory.NODES: self._limits.max_json_nodes,
            JsonStructureCategory.STRING_BYTES: self._limits.max_json_string_bytes,
        }
        raise JsonStructureLimitError(category, ceilings[category])


def inspect_zip_payload(
    payload: bytes,
    *,
    limits: OperationalLimits | None = None,
) -> ArchiveLimitDecision:
    """Inspect ZIP central-directory metadata without opening any member."""

    active = limits or OperationalLimits()
    if not isinstance(payload, bytes):
        return ArchiveLimitDecision(LimitState.FAILURE, "archive_type_invalid")
    size = validate_payload_size(len(payload), limits=active)
    if not size.allowed:
        return ArchiveLimitDecision(size.state, size.code)
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = archive.infolist()
    except (OSError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile):
        return ArchiveLimitDecision(LimitState.FAILURE, "archive_invalid")
    if any(info.flag_bits & 0x1 for info in infos):
        return ArchiveLimitDecision(LimitState.DENIED, "archive_encrypted")
    members = (
        ArchiveMember(
            path=info.filename,
            compressed_size=info.compress_size,
            uncompressed_size=info.file_size,
            kind=_zip_member_kind(info),
        )
        for info in infos
    )
    report = inspect_archive_members(members, active.archive_policy())
    if not report.allowed:
        return ArchiveLimitDecision(LimitState.DENIED, "archive_limit_denied", report)
    return ArchiveLimitDecision(LimitState.SUCCESS, "archive_allowed", report)


def _zip_member_kind(info: zipfile.ZipInfo) -> str:
    if info.is_dir():
        return "directory"
    mode = (info.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(mode):
        return "symlink"
    return "file"


__all__ = [
    "ArchiveLimitDecision",
    "JSON_STRUCTURE_EXCEEDED",
    "JsonStructureCategory",
    "JsonStructureLimitError",
    "JsonStructureScanner",
    "LimitDecision",
    "LimitState",
    "OPERATIONAL_LIMIT_COMPATIBILITY",
    "OPERATIONAL_LIMIT_SCHEMA_VERSION",
    "OperationalLimitError",
    "OperationalLimits",
    "inspect_zip_payload",
    "parse_bounded_json",
    "validate_page_size",
    "validate_payload_size",
]
