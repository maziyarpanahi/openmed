"""Offline safety assessment for archive-member metadata.

The evaluator only examines metadata supplied by the caller.  It never opens,
extracts, follows, or otherwise reads archive content.  This keeps archive
inspection suitable for a local-first redaction workflow where an archive
must be classified before any extraction is allowed.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

__all__ = [
    "ArchiveDecision",
    "ArchiveMember",
    "ArchiveSafetyDecision",
    "ArchiveSafetyPolicy",
    "ArchiveSafetyReason",
    "ArchiveSafetyReport",
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_MAX_EXPANSION_RATIO",
    "DEFAULT_MAX_MEMBER_UNCOMPRESSED_BYTES",
    "DEFAULT_MAX_PATH_LENGTH",
    "DEFAULT_MAX_TOTAL_UNCOMPRESSED_BYTES",
    "Decision",
    "assess_archive_members",
    "check_archive_safety",
    "evaluate_archive_members",
    "inspect_archive_members",
]


DEFAULT_MAX_ENTRIES: Final = 10_000
DEFAULT_MAX_TOTAL_UNCOMPRESSED_BYTES: Final = 512 * 1024 * 1024
DEFAULT_MAX_MEMBER_UNCOMPRESSED_BYTES: Final = 128 * 1024 * 1024
DEFAULT_MAX_EXPANSION_RATIO: Final = 100.0
DEFAULT_MAX_PATH_LENGTH: Final = 4_096

_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:")
_MISSING = object()


class ArchiveDecision(str, Enum):
    """Deterministic action for an archive before content extraction."""

    ALLOW = "allow"
    QUARANTINE = "quarantine"
    REJECT = "reject"


ArchiveSafetyDecision = ArchiveDecision
Decision = ArchiveDecision


class ArchiveSafetyReason(str, Enum):
    """Stable, PHI-free reason codes emitted in safety reports."""

    INVALID_METADATA = "invalid_metadata"
    ABSOLUTE_PATH = "absolute_path"
    PATH_TRAVERSAL = "path_traversal"
    PATH_TOO_LONG = "path_too_long"
    LINK = "link"
    DUPLICATE_PATH = "duplicate_path"
    ENTRY_LIMIT = "entry_limit"
    MEMBER_SIZE_LIMIT = "member_size_limit"
    TOTAL_SIZE_LIMIT = "total_size_limit"
    EXPANSION_RATIO = "expansion_ratio"


@dataclass(frozen=True, slots=True, repr=False)
class ArchiveMember:
    """Metadata for one archive member.

    ``path`` is retained only in the caller-owned input object.  The custom
    representation deliberately omits it so accidental debug output cannot
    disclose a sensitive archive name.

    Args:
        path: Archive-relative member name.
        compressed_size: Number of compressed bytes reported by the archive.
        uncompressed_size: Number of bytes produced if this member is read.
        kind: ``"file"``, ``"directory"``, ``"symlink"``, or ``"hardlink"``.
        link_target: Optional link target; any link target makes the member
            unsafe regardless of its target value.
    """

    path: str
    compressed_size: int
    uncompressed_size: int
    kind: str = "file"
    link_target: str | None = None

    def __repr__(self) -> str:
        """Return a representation that contains metadata shape only."""

        return "ArchiveMember(<metadata>)"

    @classmethod
    def from_mapping(cls, metadata: Mapping[str, Any]) -> "ArchiveMember":
        """Build a member from common archive metadata key names.

        Missing or malformed values are preserved as opaque values so the
        evaluator can return a deterministic ``reject`` report instead of
        echoing input data in an exception.
        """

        if not isinstance(metadata, Mapping):
            raise TypeError("archive member metadata must be a mapping")

        path = _first_value(metadata, ("path", "name"))
        compressed_size = _first_value(
            metadata,
            ("compressed_size", "compressed_bytes", "compressed"),
        )
        uncompressed_size = _first_value(
            metadata,
            ("uncompressed_size", "uncompressed_bytes", "size"),
        )
        kind = _first_value(metadata, ("kind", "type"), default="file")
        link_target = _first_value(
            metadata,
            ("link_target", "target"),
            default=None,
        )

        link_flag = any(
            metadata.get(key) is True
            for key in ("is_link", "is_symlink", "is_hardlink")
        )
        if link_flag:
            kind = "link"

        return cls(
            path=path,  # type: ignore[arg-type]
            compressed_size=compressed_size,  # type: ignore[arg-type]
            uncompressed_size=uncompressed_size,  # type: ignore[arg-type]
            kind=kind,  # type: ignore[arg-type]
            link_target=link_target,  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ArchiveSafetyPolicy:
    """Limits applied while classifying archive-member metadata.

    Resource-limit findings produce ``quarantine``.  Structural findings such
    as traversal, links, or malformed metadata produce ``reject``.  A
    quarantined archive must not be extracted until a separate caller-owned
    review step approves it.
    """

    max_entries: int = DEFAULT_MAX_ENTRIES
    max_total_uncompressed_bytes: int = DEFAULT_MAX_TOTAL_UNCOMPRESSED_BYTES
    max_member_uncompressed_bytes: int = DEFAULT_MAX_MEMBER_UNCOMPRESSED_BYTES
    max_expansion_ratio: float = DEFAULT_MAX_EXPANSION_RATIO
    max_path_length: int = DEFAULT_MAX_PATH_LENGTH

    def __post_init__(self) -> None:
        for value in (
            self.max_entries,
            self.max_total_uncompressed_bytes,
            self.max_member_uncompressed_bytes,
            self.max_path_length,
        ):
            if not _is_nonnegative_int(value):
                raise ValueError("archive safety limits must be non-negative integers")
        if (
            isinstance(self.max_expansion_ratio, bool)
            or not isinstance(self.max_expansion_ratio, (int, float))
            or not math.isfinite(float(self.max_expansion_ratio))
            or self.max_expansion_ratio <= 0
        ):
            raise ValueError("max_expansion_ratio must be a finite positive number")

    @property
    def max_total_size(self) -> int:
        """Return the aggregate uncompressed-byte limit."""

        return self.max_total_uncompressed_bytes

    @property
    def max_member_size(self) -> int:
        """Return the per-member uncompressed-byte limit."""

        return self.max_member_uncompressed_bytes


@dataclass(frozen=True, slots=True)
class ArchiveSafetyReport:
    """PHI-safe result containing counts and aggregate metadata only."""

    decision: ArchiveDecision
    entry_count: int
    total_compressed_bytes: int
    total_uncompressed_bytes: int
    reason_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.decision, ArchiveDecision):
            object.__setattr__(self, "decision", ArchiveDecision(self.decision))
        counts = {
            str(reason): int(count)
            for reason, count in self.reason_counts.items()
            if int(count) > 0
        }
        object.__setattr__(
            self,
            "reason_counts",
            MappingProxyType(dict(sorted(counts.items()))),
        )

    @property
    def counts(self) -> Mapping[str, int]:
        """Return stable reason counts without any member names."""

        return self.reason_counts

    @property
    def diagnostics(self) -> Mapping[str, int]:
        """Return the counts-only diagnostic mapping."""

        return self.reason_counts

    @property
    def allowed(self) -> bool:
        """Return whether extraction may proceed under this policy."""

        return self.decision is ArchiveDecision.ALLOW

    @property
    def quarantined(self) -> bool:
        """Return whether a separate review is required before extraction."""

        return self.decision is ArchiveDecision.QUARANTINE

    @property
    def rejected(self) -> bool:
        """Return whether extraction must not proceed."""

        return self.decision is ArchiveDecision.REJECT

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, counts-only serializable representation."""

        return {
            "decision": self.decision.value,
            "entry_count": self.entry_count,
            "total_compressed_bytes": self.total_compressed_bytes,
            "total_uncompressed_bytes": self.total_uncompressed_bytes,
            "reason_counts": dict(self.reason_counts),
        }


def inspect_archive_members(
    members: Iterable[ArchiveMember | Mapping[str, Any]],
    policy: ArchiveSafetyPolicy | None = None,
) -> ArchiveSafetyReport:
    """Classify archive-member metadata without reading archive content.

    The scan is bounded at one item beyond ``policy.max_entries``.  Once that
    boundary is crossed, the report records an entry-limit finding and does
    not consume additional metadata.  This prevents an untrusted metadata
    iterator from becoming an unbounded local workload.

    Args:
        members: ArchiveMember objects or mappings with ``path``,
            ``compressed_size``, and ``uncompressed_size`` fields.
        policy: Optional immutable safety policy.  Defaults to
            :class:`ArchiveSafetyPolicy`.

    Returns:
        A deterministic report.  It contains no member paths, link targets, or
        other member-provided strings.
    """

    active_policy = policy or ArchiveSafetyPolicy()
    if not isinstance(active_policy, ArchiveSafetyPolicy):
        raise TypeError("policy must be an ArchiveSafetyPolicy")
    try:
        iterator = iter(members)
    except TypeError:
        raise TypeError("members must be an iterable of archive metadata") from None

    reason_counts = {reason.value: 0 for reason in ArchiveSafetyReason}
    total_compressed_bytes = 0
    total_uncompressed_bytes = 0
    entry_count = 0
    seen_paths: set[str] = set()

    for raw_member in iterator:
        entry_count += 1
        if entry_count > active_policy.max_entries:
            reason_counts[ArchiveSafetyReason.ENTRY_LIMIT.value] = 1
            break

        member = _coerce_member(raw_member)
        if member is None:
            reason_counts[ArchiveSafetyReason.INVALID_METADATA.value] += 1
            continue

        canonical_path, path_reasons = _canonical_path(
            member.path,
            max_path_length=active_policy.max_path_length,
        )
        for reason in path_reasons:
            reason_counts[reason.value] += 1

        if canonical_path is not None:
            duplicate_key = canonical_path.casefold()
            if duplicate_key in seen_paths:
                reason_counts[ArchiveSafetyReason.DUPLICATE_PATH.value] += 1
            else:
                seen_paths.add(duplicate_key)

        if _is_link(member):
            reason_counts[ArchiveSafetyReason.LINK.value] += 1

        if not _valid_kind(member.kind) or not _valid_link_target(member.link_target):
            reason_counts[ArchiveSafetyReason.INVALID_METADATA.value] += 1

        if not _valid_sizes(member.compressed_size, member.uncompressed_size):
            reason_counts[ArchiveSafetyReason.INVALID_METADATA.value] += 1
            continue

        compressed_size = member.compressed_size
        uncompressed_size = member.uncompressed_size
        total_compressed_bytes += compressed_size
        total_uncompressed_bytes += uncompressed_size

        if uncompressed_size > active_policy.max_member_uncompressed_bytes:
            reason_counts[ArchiveSafetyReason.MEMBER_SIZE_LIMIT.value] += 1

        if total_uncompressed_bytes > active_policy.max_total_uncompressed_bytes:
            reason_counts[ArchiveSafetyReason.TOTAL_SIZE_LIMIT.value] = 1

        if _exceeds_expansion_ratio(
            compressed_size,
            uncompressed_size,
            active_policy.max_expansion_ratio,
        ):
            reason_counts[ArchiveSafetyReason.EXPANSION_RATIO.value] += 1

    decision = _decision_for(reason_counts)
    return ArchiveSafetyReport(
        decision=decision,
        entry_count=entry_count,
        total_compressed_bytes=total_compressed_bytes,
        total_uncompressed_bytes=total_uncompressed_bytes,
        reason_counts=reason_counts,
    )


def evaluate_archive_members(
    members: Iterable[ArchiveMember | Mapping[str, Any]],
    policy: ArchiveSafetyPolicy | None = None,
) -> ArchiveSafetyReport:
    """Alias for :func:`inspect_archive_members`."""

    return inspect_archive_members(members, policy)


def assess_archive_members(
    members: Iterable[ArchiveMember | Mapping[str, Any]],
    policy: ArchiveSafetyPolicy | None = None,
) -> ArchiveSafetyReport:
    """Alias for :func:`inspect_archive_members`."""

    return inspect_archive_members(members, policy)


def check_archive_safety(
    members: Iterable[ArchiveMember | Mapping[str, Any]],
    policy: ArchiveSafetyPolicy | None = None,
) -> ArchiveSafetyReport:
    """Alias for :func:`inspect_archive_members`."""

    return inspect_archive_members(members, policy)


def _coerce_member(raw_member: Any) -> ArchiveMember | None:
    if isinstance(raw_member, ArchiveMember):
        return raw_member
    if isinstance(raw_member, Mapping):
        try:
            return ArchiveMember.from_mapping(raw_member)
        except (TypeError, ValueError):
            return None
    return None


def _first_value(
    metadata: Mapping[str, Any],
    keys: tuple[str, ...],
    *,
    default: Any = _MISSING,
) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    if default is not _MISSING:
        return default
    return None


def _canonical_path(
    path: Any,
    *,
    max_path_length: int,
) -> tuple[str | None, tuple[ArchiveSafetyReason, ...]]:
    if not isinstance(path, str) or not path:
        return None, (ArchiveSafetyReason.INVALID_METADATA,)
    if len(path) > max_path_length:
        return None, (ArchiveSafetyReason.PATH_TOO_LONG,)
    normalized = unicodedata.normalize("NFKC", path).replace("\\", "/")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in normalized):
        return None, (ArchiveSafetyReason.INVALID_METADATA,)

    reasons: list[ArchiveSafetyReason] = []
    if normalized.startswith("/") or _DRIVE_PATH_RE.match(normalized):
        reasons.append(ArchiveSafetyReason.ABSOLUTE_PATH)
    segments = normalized.split("/")
    if any(segment == ".." for segment in segments):
        reasons.append(ArchiveSafetyReason.PATH_TRAVERSAL)
    if reasons:
        return None, tuple(reasons)

    safe_segments = [segment for segment in segments if segment not in {"", "."}]
    if not safe_segments:
        return None, (ArchiveSafetyReason.INVALID_METADATA,)
    return "/".join(safe_segments), ()


def _valid_sizes(compressed_size: Any, uncompressed_size: Any) -> bool:
    return _is_nonnegative_int(compressed_size) and _is_nonnegative_int(
        uncompressed_size
    )


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_kind(kind: Any) -> bool:
    if not isinstance(kind, str):
        return False
    return kind.strip().lower().replace("_", "-") in {
        "file",
        "regular",
        "regular-file",
        "directory",
        "dir",
        "symlink",
        "symbolic-link",
        "hardlink",
        "hard-link",
        "link",
    }


def _valid_link_target(link_target: Any) -> bool:
    return link_target is None or isinstance(link_target, str)


def _is_link(member: ArchiveMember) -> bool:
    if member.link_target is not None:
        return True
    if not isinstance(member.kind, str):
        return False
    return member.kind.strip().lower().replace("_", "-") in {
        "symlink",
        "symbolic-link",
        "hardlink",
        "hard-link",
        "link",
    }


def _exceeds_expansion_ratio(
    compressed_size: int,
    uncompressed_size: int,
    max_expansion_ratio: float,
) -> bool:
    if compressed_size == 0:
        return uncompressed_size > 0
    return (uncompressed_size / compressed_size) > max_expansion_ratio


def _decision_for(reason_counts: Mapping[str, int]) -> ArchiveDecision:
    structural = (
        ArchiveSafetyReason.INVALID_METADATA.value,
        ArchiveSafetyReason.ABSOLUTE_PATH.value,
        ArchiveSafetyReason.PATH_TRAVERSAL.value,
        ArchiveSafetyReason.PATH_TOO_LONG.value,
        ArchiveSafetyReason.LINK.value,
    )
    if any(reason_counts.get(reason, 0) for reason in structural):
        return ArchiveDecision.REJECT
    if any(count for count in reason_counts.values()):
        return ArchiveDecision.QUARANTINE
    return ArchiveDecision.ALLOW
