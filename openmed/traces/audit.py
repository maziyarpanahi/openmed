"""Counts-only, local trace privacy inventories.

This module is intentionally an aggregation boundary.  A caller may inspect a
trace with a separate local scanner, but the audit surface accepts only store,
category, file, and byte-offset metadata.  Unknown mapping fields (including
values, snippets, and exception details) are ignored and never reach a report.
The module does not open files or make network requests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, TypeVar

AUDIT_SCHEMA_VERSION = 1
TRACE_STATUSES = ("scanned", "skipped", "unreadable", "unsupported")
TraceAuditStatus: TypeAlias = Literal["scanned", "skipped", "unreadable", "unsupported"]

_DEFAULT_STORE = "unknown"
_DEFAULT_FILE = "<unknown>"
_SAFE_STORE_LABELS = frozenset(
    {"claude", "codex", "cursor", "custom", "other", "unknown"}
)
_SAFE_CATEGORY_LABELS = frozenset(
    {
        "assistant",
        "completion",
        "credential",
        "environment",
        "identifier",
        "message",
        "other",
        "path",
        "prompt",
        "response",
        "secret",
        "system",
        "tool-call",
        "tool-output",
        "tool-result",
        "unknown",
        "user",
    }
)
_T = TypeVar("_T")
_MISSING = object()
_HASHED_LABEL = re.compile(r"^(?P<prefix>store|category|file)_sha256_[0-9a-f]{16}$")


def _metadata_text(value: object, *, fallback: str) -> str:
    """Normalize a report dimension without interpolating it into errors."""

    if isinstance(value, os.PathLike):
        try:
            value = os.fspath(value)
        except Exception:  # noqa: BLE001 - untrusted metadata must fail closed
            return fallback
    if not isinstance(value, str):
        return fallback

    text = value.strip().replace("\\", "/")
    if not text:
        return fallback
    # A report is safe to print as one terminal row even when a caller's
    # metadata came from an unusual filename or store label.
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def _store_name(value: object) -> str:
    text = _metadata_text(value, fallback=_DEFAULT_STORE).lower()
    if text in _SAFE_STORE_LABELS:
        return text
    return _hashed_label(text, prefix="store", fallback=_DEFAULT_STORE)


def _file_name(value: object) -> str:
    text = _metadata_text(value, fallback=_DEFAULT_FILE)
    if text == _DEFAULT_FILE:
        return text

    # A basename may itself contain a patient name, encounter identifier, or
    # other PHI. Hash every caller-provided file label so absolute and relative
    # paths remain useful as deterministic grouping keys without being echoed.
    return _hashed_label(text, prefix="file", fallback=_DEFAULT_FILE)


def _category_name(value: object) -> str:
    text = _metadata_text(value, fallback="unknown").lower()
    if text in _SAFE_CATEGORY_LABELS:
        return text
    return _hashed_label(text, prefix="category", fallback="unknown")


def _hashed_label(value: str, *, prefix: str, fallback: str) -> str:
    if value == fallback:
        return fallback
    match = _HASHED_LABEL.fullmatch(value)
    if match is not None and match.group("prefix") == prefix:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_sha256_{digest}"


def _safe_items(value: Iterable[_T], *, error: str) -> Iterator[_T]:
    """Iterate caller input without propagating value-bearing exceptions."""

    try:
        items = iter(value)
    except Exception:  # noqa: BLE001 - input may expose PHI in errors
        raise TypeError(error) from None
    while True:
        try:
            yield next(items)
        except StopIteration:
            return
        except Exception:  # noqa: BLE001 - input may expose PHI in errors
            raise ValueError(error) from None


def _mapping_value(
    value: Mapping[str, Any],
    *keys: str,
    default: Any,
    error: str,
) -> Any:
    """Read the first present mapping key without exposing lookup failures."""

    for key in keys:
        try:
            item = value.get(key, _MISSING)
        except Exception:  # noqa: BLE001 - mapping errors may contain PHI
            raise ValueError(error) from None
        if item is not _MISSING:
            return item
    return default


def _offset(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("trace byte offsets must be non-negative integers")
    return value


def _positive_count(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("trace finding count must be a positive integer")
    return value


def _nonnegative_count(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("trace status count must be a non-negative integer")
    return value


def _status(value: object) -> TraceAuditStatus:
    if not isinstance(value, str):
        raise ValueError("trace scan status is unsupported")
    normalized = value.strip().lower()
    if normalized not in TRACE_STATUSES:
        raise ValueError("trace scan status is unsupported")
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, order=True)
class ByteRange:
    """A half-open byte range retained by a counts-only report."""

    start: int
    end: int

    def __post_init__(self) -> None:
        start = _offset(self.start)
        end = _offset(self.end)
        if end < start:
            raise ValueError("trace byte range end must not precede its start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def length(self) -> int:
        """Return the number of bytes in this half-open range."""

        return self.end - self.start

    def to_dict(self) -> dict[str, int]:
        """Return the value-free JSON representation of the range."""

        return {"start": self.start, "end": self.end}


@dataclass(frozen=True, slots=True)
class TraceFinding:
    """One value-free trace finding.

    ``start`` and ``end`` use the usual half-open byte-offset convention.
    ``count`` is useful when an upstream scanner has already coalesced
    identical metadata records.  No matched value or snippet is accepted.
    """

    store: str
    category: str
    file: str
    start: int
    end: int
    count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "store", _store_name(self.store))
        object.__setattr__(self, "category", _category_name(self.category))
        object.__setattr__(self, "file", _file_name(self.file))
        start = _offset(self.start)
        end = _offset(self.end)
        if end < start:
            raise ValueError("trace byte range end must not precede its start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "count", _positive_count(self.count))

    @property
    def byte_range(self) -> ByteRange:
        """Return the finding's byte range."""

        return ByteRange(self.start, self.end)

    @property
    def path(self) -> str:
        """Return the normalized file label under the common path spelling."""

        return self.file

    def to_dict(self) -> dict[str, Any]:
        """Return only dimensions, counts, and byte ranges."""

        return {
            "store": self.store,
            "category": self.category,
            "file": self.file,
            "count": self.count,
            "byte_ranges": [self.byte_range.to_dict()],
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        default_store: object = _DEFAULT_STORE,
        default_file: object = _DEFAULT_FILE,
    ) -> "TraceFinding":
        """Build a finding while ignoring all unknown mapping fields.

        Accepted offset spellings are ``start``/``end``,
        ``byte_start``/``byte_end``, and a ``byte_range`` pair or mapping.
        ``path`` and ``file_path`` are accepted as aliases for ``file``.
        """

        if not isinstance(value, Mapping):
            raise TypeError("trace finding must be a mapping")

        byte_range = _mapping_value(
            value,
            "byte_range",
            default=None,
            error="trace finding metadata could not be read",
        )
        start = _mapping_value(
            value,
            "start",
            "byte_start",
            default=None,
            error="trace finding metadata could not be read",
        )
        end = _mapping_value(
            value,
            "end",
            "byte_end",
            default=None,
            error="trace finding metadata could not be read",
        )
        if isinstance(byte_range, Mapping):
            start = _mapping_value(
                byte_range,
                "start",
                "byte_start",
                default=start,
                error="trace finding byte range could not be read",
            )
            end = _mapping_value(
                byte_range,
                "end",
                "byte_end",
                default=end,
                error="trace finding byte range could not be read",
            )
        elif isinstance(byte_range, (list, tuple)) and len(byte_range) == 2:
            start, end = byte_range
        if start is None or end is None:
            raise ValueError("trace finding requires a byte range")

        return cls(
            store=_mapping_value(
                value,
                "store",
                "store_type",
                default=default_store,
                error="trace finding metadata could not be read",
            ),
            category=_mapping_value(
                value,
                "category",
                "kind",
                default="unknown",
                error="trace finding metadata could not be read",
            ),
            file=_mapping_value(
                value,
                "file",
                "path",
                "file_path",
                default=default_file,
                error="trace finding metadata could not be read",
            ),
            start=start,
            end=end,
            count=_mapping_value(
                value,
                "count",
                default=1,
                error="trace finding metadata could not be read",
            ),
        )


@dataclass(frozen=True, slots=True)
class TraceScan:
    """One scanned-file status and its optional value-free findings."""

    store: str = _DEFAULT_STORE
    file: str = _DEFAULT_FILE
    status: TraceAuditStatus = "scanned"
    findings: tuple[TraceFinding, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "store", _store_name(self.store))
        object.__setattr__(self, "file", _file_name(self.file))
        object.__setattr__(self, "status", _status(self.status))
        normalized = tuple(
            item
            if isinstance(item, TraceFinding)
            else TraceFinding.from_mapping(
                item,
                default_store=self.store,
                default_file=self.file,
            )
            for item in _safe_items(
                self.findings,
                error="trace scan findings could not be consumed",
            )
        )
        object.__setattr__(self, "findings", normalized)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TraceScan":
        """Build a scan record without retaining status reasons or values."""

        if not isinstance(value, Mapping):
            raise TypeError("trace scan must be a mapping")
        raw_findings = _mapping_value(
            value,
            "findings",
            default=(),
            error="trace scan metadata could not be read",
        )
        if raw_findings is None:
            raw_findings = ()
        if isinstance(raw_findings, Mapping) or isinstance(raw_findings, str):
            raise TypeError("trace scan findings must be an iterable")
        if not isinstance(raw_findings, Iterable):
            raise TypeError("trace scan findings must be an iterable")
        return cls(
            store=_mapping_value(
                value,
                "store",
                "store_type",
                default=_DEFAULT_STORE,
                error="trace scan metadata could not be read",
            ),
            file=_mapping_value(
                value,
                "file",
                "path",
                "file_path",
                default=_DEFAULT_FILE,
                error="trace scan metadata could not be read",
            ),
            status=_mapping_value(
                value,
                "status",
                default="scanned",
                error="trace scan metadata could not be read",
            ),
            findings=tuple(
                _safe_items(
                    raw_findings,
                    error="trace scan findings could not be consumed",
                )
            ),
        )


@dataclass
class _Aggregate:
    count: int = 0
    byte_ranges: set[ByteRange] = field(default_factory=set)
    located_byte_ranges: set[tuple[str, str, ByteRange]] = field(default_factory=set)
    stores: set[str] = field(default_factory=set)
    categories: set[str] = field(default_factory=set)
    files: set[tuple[str, str]] = field(default_factory=set)

    def add(self, finding: TraceFinding) -> None:
        self.count += finding.count
        self.byte_ranges.add(finding.byte_range)
        self.located_byte_ranges.add((finding.store, finding.file, finding.byte_range))
        self.stores.add(finding.store)
        self.categories.add(finding.category)
        self.files.add((finding.store, finding.file))

    def ranges(self) -> list[dict[str, int]]:
        return [item.to_dict() for item in sorted(self.byte_ranges)]

    def byte_count(self) -> int:
        return sum(item.length for _, _, item in self.located_byte_ranges)


@dataclass(frozen=True, slots=True)
class TraceAuditReport:
    """Deterministic, counts-only inventory of local trace findings."""

    totals: Mapping[str, int]
    stores: tuple[Mapping[str, Any], ...]
    categories: tuple[Mapping[str, Any], ...]
    files: tuple[Mapping[str, Any], ...]
    findings: tuple[Mapping[str, Any], ...]
    schema_version: int = AUDIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            totals = MappingProxyType(
                {
                    status: _nonnegative_count(self.totals.get(status, 0))
                    for status in TRACE_STATUSES
                }
            )
            stores = _normalize_report_rows(self.stores, kind="store")
            categories = _normalize_report_rows(self.categories, kind="category")
            files = _normalize_report_rows(self.files, kind="file")
            findings = _normalize_report_rows(self.findings, kind="finding")
            if type(self.schema_version) is not int or self.schema_version < 1:
                raise ValueError
        except Exception:  # noqa: BLE001 - report inputs may contain PHI
            raise ValueError("trace audit report is invalid") from None
        object.__setattr__(self, "totals", totals)
        object.__setattr__(self, "stores", stores)
        object.__setattr__(self, "categories", categories)
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "findings", findings)

    @property
    def status_counts(self) -> dict[str, int]:
        """Return status totals in canonical order."""

        return {status: int(self.totals.get(status, 0)) for status in TRACE_STATUSES}

    @property
    def finding_count(self) -> int:
        """Return the total number of finding occurrences."""

        return sum(int(item.get("count", 0)) for item in self.findings)

    @property
    def scanned(self) -> int:
        """Return the number of scanned files."""

        return self.status_counts["scanned"]

    @property
    def skipped(self) -> int:
        """Return the number of skipped files."""

        return self.status_counts["skipped"]

    @property
    def unreadable(self) -> int:
        """Return the number of unreadable files."""

        return self.status_counts["unreadable"]

    @property
    def unsupported(self) -> int:
        """Return the number of unsupported files."""

        return self.status_counts["unsupported"]

    @property
    def by_store(self) -> dict[str, Mapping[str, Any]]:
        """Return store aggregates keyed by their safe store label."""

        return {str(item["store"]): item for item in self.stores}

    @property
    def by_category(self) -> dict[str, Mapping[str, Any]]:
        """Return category aggregates keyed by category label."""

        return {str(item["category"]): item for item in self.categories}

    @property
    def by_file(self) -> dict[tuple[str, str], Mapping[str, Any]]:
        """Return file aggregates keyed by ``(store, file)``."""

        return {(str(item["store"]), str(item["file"])): item for item in self.files}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible payload containing no finding values."""

        return {
            "schema_version": self.schema_version,
            "totals": self.status_counts,
            "finding_count": self.finding_count,
            "stores": [dict(item) for item in self.stores],
            "categories": [dict(item) for item in self.categories],
            "files": [dict(item) for item in self.files],
            "findings": [dict(item) for item in self.findings],
        }

    def as_dict(self) -> dict[str, Any]:
        """Return :meth:`to_dict` under a descriptive compatibility alias."""

        return self.to_dict()

    def __getitem__(self, key: str) -> Any:
        """Allow convenient read-only access to serialized report sections."""

        return self.to_dict()[key]

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return deterministic JSON with stable key and row ordering."""

        if indent is None:
            return json.dumps(
                self.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )

    def to_terminal(self) -> str:
        """Return a deterministic terminal summary with no raw values."""

        lines = [
            "Trace privacy inventory",
            "",
            "Totals",
            *[f"  {status}: {self.status_counts[status]}" for status in TRACE_STATUSES],
            f"  findings: {self.finding_count}",
            "",
            "By store",
        ]
        lines.extend(_terminal_rows(self.stores, "store"))
        lines.extend(["", "By category"])
        lines.extend(_terminal_rows(self.categories, "category"))
        lines.extend(["", "By file"])
        for item in self.files:
            lines.append(
                "  "
                f"{item['store']} / {item['file']}: "
                f"{item['count']} findings, ranges={_format_ranges(item['byte_ranges'])}"
            )
        if not self.files:
            lines.append("  none")
        return "\n".join(lines) + "\n"

    def to_text(self) -> str:
        """Return the terminal representation under a common text alias."""

        return self.to_terminal()

    def __str__(self) -> str:
        return self.to_terminal()


def _terminal_rows(rows: Iterable[Mapping[str, Any]], label_key: str) -> list[str]:
    result: list[str] = []
    for item in rows:
        result.append(
            f"  {item[label_key]}: {item['count']} findings, "
            f"ranges={_format_ranges(item['byte_ranges'])}"
        )
    if not result:
        result.append("  none")
    return result


def _format_ranges(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    ranges: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            start = item.get("start")
            end = item.get("end")
            if type(start) is int and type(end) is int:
                ranges.append(f"{start}-{end}")
    return ",".join(ranges) if ranges else "none"


_ReportRowKind: TypeAlias = Literal["store", "category", "file", "finding"]


def _normalize_report_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    kind: _ReportRowKind,
) -> tuple[Mapping[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    for row in _safe_items(rows, error="trace audit rows could not be consumed"):
        if not isinstance(row, Mapping):
            raise TypeError("trace audit rows must be mappings")
        ranges = _normalize_report_ranges(row.get("byte_ranges", ()))
        minimum_byte_count = sum(item["end"] - item["start"] for item in ranges)
        byte_count = _nonnegative_count(row.get("byte_count", minimum_byte_count))
        if byte_count < minimum_byte_count:
            raise ValueError("trace audit byte count is inconsistent")
        base: dict[str, Any] = {
            "count": _nonnegative_count(row.get("count", 0)),
            "byte_count": byte_count,
            "byte_ranges": ranges,
        }
        if kind == "store":
            normalized.append(
                {
                    "store": _store_name(row.get("store", _DEFAULT_STORE)),
                    **base,
                    "category_count": _nonnegative_count(row.get("category_count", 0)),
                    "file_count": _nonnegative_count(row.get("file_count", 0)),
                }
            )
        elif kind == "category":
            normalized.append(
                {
                    "category": _category_name(row.get("category", "unknown")),
                    **base,
                    "file_count": _nonnegative_count(row.get("file_count", 0)),
                    "store_count": _nonnegative_count(row.get("store_count", 0)),
                }
            )
        elif kind == "file":
            normalized.append(
                {
                    "store": _store_name(row.get("store", _DEFAULT_STORE)),
                    "file": _file_name(row.get("file", _DEFAULT_FILE)),
                    **base,
                    "category_count": _nonnegative_count(row.get("category_count", 0)),
                }
            )
        else:
            normalized.append(
                {
                    "store": _store_name(row.get("store", _DEFAULT_STORE)),
                    "category": _category_name(row.get("category", "unknown")),
                    "file": _file_name(row.get("file", _DEFAULT_FILE)),
                    **base,
                }
            )

    sort_keys = {
        "store": lambda row: (str(row["store"]),),
        "category": lambda row: (str(row["category"]),),
        "file": lambda row: (str(row["store"]), str(row["file"])),
        "finding": lambda row: (
            str(row["store"]),
            str(row["category"]),
            str(row["file"]),
        ),
    }
    normalized.sort(key=sort_keys[kind])
    return tuple(MappingProxyType(row) for row in normalized)


def _normalize_report_ranges(value: object) -> list[dict[str, int]]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise TypeError("trace audit byte ranges must be an iterable")
    ranges: set[ByteRange] = set()
    for item in _safe_items(
        value,
        error="trace audit byte ranges could not be consumed",
    ):
        if not isinstance(item, Mapping):
            raise TypeError("trace audit byte ranges must be mappings")
        start = _offset(item.get("start"))
        end = _offset(item.get("end"))
        ranges.add(ByteRange(start, end))
    return [item.to_dict() for item in sorted(ranges)]


class TraceAudit:
    """Mutable collector for local scan statuses and value-free findings."""

    def __init__(self) -> None:
        self._statuses: dict[tuple[str, str], TraceAuditStatus] = {}
        self._manual_status_counts = {status: 0 for status in TRACE_STATUSES}
        self._findings: list[TraceFinding] = []

    def record_scan(
        self,
        store: object = _DEFAULT_STORE,
        file: object = _DEFAULT_FILE,
        *,
        status: TraceAuditStatus = "scanned",
    ) -> None:
        """Record one file status, deduplicating repeated file metadata."""

        safe_store = _store_name(store)
        safe_file = _file_name(file)
        safe_status = _status(status)
        key = (safe_store, safe_file)
        previous = self._statuses.get(key)
        if previous == safe_status:
            return
        if previous is not None:
            self._statuses[key] = safe_status
            return
        self._statuses[key] = safe_status

    def record_status(self, status: TraceAuditStatus, count: int = 1) -> None:
        """Record status totals that do not have a file identity."""

        safe_status = _status(status)
        self._manual_status_counts[safe_status] += _nonnegative_count(count)

    def add_finding(self, finding: TraceFinding | Mapping[str, Any]) -> TraceFinding:
        """Add one finding and implicitly mark its file as scanned."""

        normalized = (
            finding
            if isinstance(finding, TraceFinding)
            else TraceFinding.from_mapping(finding)
        )
        if (normalized.store, normalized.file) not in self._statuses:
            self.record_scan(normalized.store, normalized.file)
        self._findings.append(normalized)
        return normalized

    def add_scan(self, scan: TraceScan | Mapping[str, Any]) -> None:
        """Add a scan record and its findings without retaining extra fields."""

        normalized = (
            scan if isinstance(scan, TraceScan) else TraceScan.from_mapping(scan)
        )
        self.record_scan(normalized.store, normalized.file, status=normalized.status)
        for finding in normalized.findings:
            if (finding.store, finding.file) not in self._statuses:
                self.record_scan(finding.store, finding.file)
            self._findings.append(finding)

    def extend(
        self,
        findings: Iterable[TraceFinding | Mapping[str, Any]],
    ) -> None:
        """Add findings from an iterable."""

        for finding in _safe_items(
            findings,
            error="trace findings could not be consumed",
        ):
            self.add_finding(finding)

    def report(self) -> TraceAuditReport:
        """Build an immutable snapshot with deterministic aggregate ordering."""

        aggregates: dict[tuple[str, ...], _Aggregate] = {}
        for finding in self._findings:
            for key in (
                ("store", finding.store),
                ("category", finding.category),
                ("file", finding.store, finding.file),
                ("finding", finding.store, finding.category, finding.file),
            ):
                aggregates.setdefault(key, _Aggregate()).add(finding)

        status_counts = {
            status: self._manual_status_counts[status] for status in TRACE_STATUSES
        }
        for status in self._statuses.values():
            status_counts[status] += 1

        stores = tuple(
            _store_row(key[1], aggregate)
            for key, aggregate in sorted(
                (
                    (key, value)
                    for key, value in aggregates.items()
                    if key[0] == "store"
                ),
                key=lambda item: item[0][1],
            )
        )
        categories = tuple(
            _category_row(key[1], aggregate)
            for key, aggregate in sorted(
                (
                    (key, value)
                    for key, value in aggregates.items()
                    if key[0] == "category"
                ),
                key=lambda item: item[0][1],
            )
        )
        files = tuple(
            _file_row(key[1], key[2], aggregate)
            for key, aggregate in sorted(
                ((key, value) for key, value in aggregates.items() if key[0] == "file"),
                key=lambda item: (item[0][1], item[0][2]),
            )
        )
        findings = tuple(
            _finding_row(key[1], key[2], key[3], aggregate)
            for key, aggregate in sorted(
                (
                    (key, value)
                    for key, value in aggregates.items()
                    if key[0] == "finding"
                ),
                key=lambda item: (item[0][1], item[0][2], item[0][3]),
            )
        )
        return TraceAuditReport(
            totals=status_counts,
            stores=stores,
            categories=categories,
            files=files,
            findings=findings,
        )

    build = report
    snapshot = report


def _base_row(aggregate: _Aggregate) -> dict[str, Any]:
    return {
        "count": aggregate.count,
        "byte_count": aggregate.byte_count(),
        "byte_ranges": aggregate.ranges(),
    }


def _store_row(store: str, aggregate: _Aggregate) -> dict[str, Any]:
    return {
        "store": store,
        **_base_row(aggregate),
        "category_count": len(aggregate.categories),
        "file_count": len(aggregate.files),
    }


def _category_row(category: str, aggregate: _Aggregate) -> dict[str, Any]:
    return {
        "category": category,
        **_base_row(aggregate),
        "file_count": len(aggregate.files),
        "store_count": len(aggregate.stores),
    }


def _file_row(store: str, file: str, aggregate: _Aggregate) -> dict[str, Any]:
    return {
        "store": store,
        "file": file,
        **_base_row(aggregate),
        "category_count": len(aggregate.categories),
    }


def _finding_row(
    store: str,
    category: str,
    file: str,
    aggregate: _Aggregate,
) -> dict[str, Any]:
    return {
        "store": store,
        "category": category,
        "file": file,
        **_base_row(aggregate),
    }


def _status_inputs(
    collector: TraceAudit,
    status: TraceAuditStatus,
    value: object,
) -> None:
    if type(value) is int:
        collector.record_status(status, value)
        return
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError("trace status input must be a count or iterable")
    if not isinstance(value, Iterable):
        raise TypeError("trace status input must be a count or iterable")
    for _ in _safe_items(
        value,
        error="trace status input could not be consumed",
    ):
        collector.record_status(status)


def build_trace_audit(
    findings: Iterable[TraceFinding | Mapping[str, Any]] | None = None,
    *,
    scans: Iterable[TraceScan | Mapping[str, Any]] | None = None,
    records: Iterable[TraceScan | Mapping[str, Any]] | None = None,
    statuses: Mapping[str, int] | Iterable[str] | None = None,
    scanned: int | Iterable[object] | None = None,
    skipped: int | Iterable[object] | None = None,
    unreadable: int | Iterable[object] | None = None,
    unsupported: int | Iterable[object] | None = None,
) -> TraceAuditReport:
    """Build a deterministic counts-only report from local scan metadata.

    ``findings`` may contain :class:`TraceFinding` objects or mappings.  The
    optional ``scans``/``records`` iterable records file statuses and may carry
    findings with those records.  Status keyword arguments accept either a
    count or an iterable of opaque file records; the iterable items are counted
    but never serialized.  No input file is opened or modified.
    """

    if scans is not None and records is not None:
        raise TypeError("pass either scans or records, not both")
    collector = TraceAudit()

    selected_scans = scans if scans is not None else records
    if selected_scans is None:
        selected_scans = ()
    for scan in _safe_items(
        selected_scans,
        error="trace scans could not be consumed",
    ):
        collector.add_scan(scan)
    collector.extend(findings if findings is not None else ())

    if statuses is not None:
        if isinstance(statuses, Mapping):
            for name in _safe_items(
                statuses,
                error="trace statuses could not be consumed",
            ):
                safe_status = _status(name)
                count = _mapping_value(
                    statuses,
                    name,
                    default=0,
                    error="trace statuses could not be read",
                )
                collector.record_status(safe_status, _nonnegative_count(count))
        elif isinstance(statuses, (str, bytes)):
            raise TypeError("trace statuses must be a mapping or iterable")
        else:
            for name in _safe_items(
                statuses,
                error="trace statuses could not be consumed",
            ):
                collector.record_status(_status(name))

    status_inputs: tuple[tuple[TraceAuditStatus, object], ...] = (
        ("scanned", scanned),
        ("skipped", skipped),
        ("unreadable", unreadable),
        ("unsupported", unsupported),
    )
    for name, value in status_inputs:
        if value is not None:
            _status_inputs(collector, name, value)
    return collector.report()


def format_terminal_report(report: TraceAuditReport) -> str:
    """Render a report using its deterministic terminal format."""

    return report.to_terminal()


def format_json_report(report: TraceAuditReport, *, indent: int | None = 2) -> str:
    """Render a report using its deterministic JSON format."""

    return report.to_json(indent=indent)


# Descriptive aliases keep the small public surface convenient for callers that
# refer to the operation as an inventory or a scan report.
TraceAuditFinding = TraceFinding
TraceScanRecord = TraceScan
TraceAuditCollector = TraceAudit
build_audit_report = build_trace_audit
build_trace_audit_report = build_trace_audit
generate_trace_audit = build_trace_audit
inventory_trace = build_trace_audit
scan_trace = build_trace_audit
render_terminal_report = format_terminal_report
render_json_report = format_json_report
format_trace_audit_terminal = format_terminal_report
format_trace_audit_json = format_json_report


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "TRACE_STATUSES",
    "ByteRange",
    "TraceAudit",
    "TraceAuditCollector",
    "TraceAuditFinding",
    "TraceAuditReport",
    "TraceAuditStatus",
    "TraceFinding",
    "TraceScan",
    "TraceScanRecord",
    "audit_trace",
    "build_audit_report",
    "build_trace_audit",
    "build_trace_audit_report",
    "format_json_report",
    "format_terminal_report",
    "format_trace_audit_json",
    "format_trace_audit_terminal",
    "generate_trace_audit",
    "inventory_trace",
    "render_json_report",
    "render_terminal_report",
    "scan_trace",
]


audit_trace = build_trace_audit
