"""Fail-closed limits for untrusted operational inputs.

Checks in this module run before caller-selected parsers, decompression, model
execution, or storage work.  Errors expose stable reason codes and aggregate
counts only; input values and archive member names are never returned.
"""

from __future__ import annotations

import io
import json
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
        value = json.loads(payload)
    except (RecursionError, UnicodeError, json.JSONDecodeError):
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
            string_bytes += len(current.encode("utf-8"))
        elif isinstance(current, dict):
            for key, item in current.items():
                string_bytes += len(str(key).encode("utf-8"))
                stack.append((item, depth + 1))
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
        if string_bytes > active.max_json_string_bytes:
            raise OperationalLimitError("json_string_limit_exceeded")
    return value


def _preflight_json(payload: bytes, limits: OperationalLimits) -> None:
    """Bound JSON structure lexically before invoking the decoder."""

    depth = 0
    structural_units = 1
    string_bytes = 0
    in_string = False
    escaped = False
    for byte in payload:
        if in_string:
            if escaped:
                escaped = False
                string_bytes += 1
            elif byte == 0x5C:  # backslash
                escaped = True
                string_bytes += 1
            elif byte == 0x22:  # double quote
                in_string = False
            else:
                string_bytes += 1
            if string_bytes > limits.max_json_string_bytes:
                raise OperationalLimitError("json_string_limit_exceeded")
            continue
        if byte == 0x22:
            in_string = True
        elif byte in {0x5B, 0x7B}:  # [ {
            depth += 1
            structural_units += 1
            if depth > limits.max_json_depth:
                raise OperationalLimitError("json_depth_limit_exceeded")
        elif byte in {0x5D, 0x7D}:  # ] }
            depth -= 1
            if depth < 0:
                raise OperationalLimitError("json_invalid")
        elif byte == 0x2C:  # comma
            structural_units += 1
        if structural_units > limits.max_json_nodes:
            raise OperationalLimitError("json_node_limit_exceeded")
    if in_string or escaped or depth != 0:
        raise OperationalLimitError("json_invalid")


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
