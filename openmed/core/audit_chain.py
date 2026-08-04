"""Tamper-evident audit chains for de-identification runs.

The chain stores only canonical labels, offsets, and cryptographic hashes from
an :class:`~openmed.core.audit.AuditReport`. It deliberately excludes source
text, replacements, detector evidence, context, and reversible mappings.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audit import AuditReport, AuditSpan, stable_hash
from .labels import CANONICAL_LABELS, normalize_label

CHAIN_FORMAT = "openmed.audit-chain"
CHAIN_VERSION = 1
GENESIS_HASH = stable_hash(
    {"format": CHAIN_FORMAT, "version": CHAIN_VERSION, "sequence": -1}
)

_CHAIN_FIELDS = {
    "format",
    "version",
    "genesis_hash",
    "entry_count",
    "head_hash",
    "entries",
}
_ENTRY_FIELDS = {
    "sequence",
    "previous_hash",
    "report_hash",
    "input_hash",
    "deidentified_text_hash",
    "spans",
    "entry_hash",
}
_SPAN_FIELDS = {"start", "end", "label", "text_hash"}


def _canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _is_sha256_hash(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def _require_sha256_hash(value: Any, field_name: str) -> str:
    if not _is_sha256_hash(value):
        raise ValueError(f"{field_name} must be a sha256:<hex> hash")
    return value


def _reject_unknown_fields(
    data: Mapping[str, Any],
    allowed: set[str],
    object_name: str,
) -> None:
    if any(key not in allowed for key in data):
        raise ValueError(f"{object_name} contains unsupported fields")


@dataclass(frozen=True)
class AuditChainSpan:
    """PHI-safe span material committed to an audit-chain entry."""

    start: int
    end: int
    label: str
    text_hash: str

    def __post_init__(self) -> None:
        if type(self.start) is not int or self.start < 0:
            raise ValueError("audit chain span start must be a non-negative integer")
        if type(self.end) is not int or self.end < self.start:
            raise ValueError("audit chain span end must be at or after start")
        if not isinstance(self.label, str) or self.label not in CANONICAL_LABELS:
            raise ValueError("audit chain span label must be canonical")
        _require_sha256_hash(self.text_hash, "audit chain span text_hash")

    @classmethod
    def from_audit_span(cls, span: AuditSpan) -> "AuditChainSpan":
        """Reduce an audit span to canonical label, offsets, and text hash."""
        return cls(
            start=span.start,
            end=span.end,
            label=normalize_label(span.canonical_label or span.label),
            text_hash=span.text_hash,
        )

    def to_dict(self) -> dict[str, int | str]:
        """Return the JSON-compatible safe span payload."""
        return {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "text_hash": self.text_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditChainSpan":
        """Load a safe span while rejecting fields that could hide raw PHI."""
        _reject_unknown_fields(data, _SPAN_FIELDS, "audit chain span")
        return cls(
            start=data.get("start"),  # type: ignore[arg-type]
            end=data.get("end"),  # type: ignore[arg-type]
            label=data.get("label"),  # type: ignore[arg-type]
            text_hash=data.get("text_hash"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class AuditChainEntry:
    """One de-identification run committed to the preceding entry hash."""

    sequence: int
    previous_hash: str
    report_hash: str
    input_hash: str
    deidentified_text_hash: str
    spans: tuple[AuditChainSpan, ...]
    entry_hash: str

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("audit chain sequence must be a non-negative integer")
        _require_sha256_hash(self.previous_hash, "audit chain previous_hash")
        _require_sha256_hash(self.report_hash, "audit chain report_hash")
        _require_sha256_hash(self.input_hash, "audit chain input_hash")
        _require_sha256_hash(
            self.deidentified_text_hash,
            "audit chain deidentified_text_hash",
        )
        _require_sha256_hash(self.entry_hash, "audit chain entry_hash")
        if not all(isinstance(span, AuditChainSpan) for span in self.spans):
            raise TypeError("audit chain spans must contain AuditChainSpan values")

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "previous_hash": self.previous_hash,
            "report_hash": self.report_hash,
            "input_hash": self.input_hash,
            "deidentified_text_hash": self.deidentified_text_hash,
            "spans": [span.to_dict() for span in self.spans],
        }

    def compute_hash(self) -> str:
        """Return the deterministic hash this entry should carry."""
        return stable_hash(self._hash_payload())

    def matches_report(self, report: AuditReport) -> bool:
        """Return whether this entry commits to the supplied audit report."""
        try:
            expected = _entry_from_report(
                report,
                sequence=self.sequence,
                previous_hash=self.previous_hash,
            )
        except (TypeError, ValueError):
            return False
        return self._hash_payload() == expected._hash_payload()

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-compatible entry payload."""
        return {**self._hash_payload(), "entry_hash": self.entry_hash}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditChainEntry":
        """Load an entry while rejecting uncommitted or PHI-bearing fields."""
        _reject_unknown_fields(data, _ENTRY_FIELDS, "audit chain entry")
        spans = data.get("spans")
        if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes)):
            raise TypeError("audit chain entry spans must be a list")
        loaded_spans: list[AuditChainSpan] = []
        for span in spans:
            if not isinstance(span, Mapping):
                raise TypeError("audit chain spans must contain objects")
            loaded_spans.append(AuditChainSpan.from_dict(span))
        return cls(
            sequence=data.get("sequence"),  # type: ignore[arg-type]
            previous_hash=data.get("previous_hash"),  # type: ignore[arg-type]
            report_hash=data.get("report_hash"),  # type: ignore[arg-type]
            input_hash=data.get("input_hash"),  # type: ignore[arg-type]
            deidentified_text_hash=data.get(  # type: ignore[arg-type]
                "deidentified_text_hash"
            ),
            spans=tuple(loaded_spans),
            entry_hash=data.get("entry_hash"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ChainVerificationResult:
    """Detailed result of an offline audit-chain verification."""

    valid: bool
    checked_entries: int
    reason: str
    entry_index: int | None = None

    def __bool__(self) -> bool:
        return self.valid

    @property
    def is_valid(self) -> bool:
        """Compatibility alias for callers that prefer an ``is_valid`` field."""
        return self.valid

    @property
    def error(self) -> str | None:
        """Return the failure reason, or ``None`` for a valid chain."""
        return None if self.valid else self.reason


def _entry_from_report(
    report: AuditReport,
    *,
    sequence: int,
    previous_hash: str,
) -> AuditChainEntry:
    if not isinstance(report, AuditReport):
        raise TypeError("audit chain entries require an AuditReport")
    if not report.repro_hash_matches():
        raise ValueError("cannot append an audit report with an invalid repro_hash")

    safe_spans = tuple(
        sorted(
            (AuditChainSpan.from_audit_span(span) for span in report.spans),
            key=lambda span: (
                span.start,
                span.end,
                span.label,
                span.text_hash,
            ),
        )
    )
    material = {
        "sequence": sequence,
        "previous_hash": _require_sha256_hash(
            previous_hash, "audit chain previous_hash"
        ),
        "report_hash": _require_sha256_hash(
            report.repro_hash, "audit chain report_hash"
        ),
        "input_hash": _require_sha256_hash(report.input_hash, "audit chain input_hash"),
        "deidentified_text_hash": _require_sha256_hash(
            report.deidentified_text_hash,
            "audit chain deidentified_text_hash",
        ),
        "spans": [span.to_dict() for span in safe_spans],
    }
    return AuditChainEntry(
        sequence=sequence,
        previous_hash=previous_hash,
        report_hash=report.repro_hash,
        input_hash=report.input_hash,
        deidentified_text_hash=report.deidentified_text_hash,
        spans=safe_spans,
        entry_hash=stable_hash(material),
    )


class AuditChain:
    """Append-only sequence of PHI-safe de-identification audit entries."""

    GENESIS_HASH = GENESIS_HASH

    def __init__(
        self,
        entries: Sequence[AuditChainEntry] | None = None,
        *,
        declared_entry_count: int | None = None,
        declared_head_hash: str | None = None,
    ) -> None:
        self._entries = list(entries or ())
        self._declared_entry_count = (
            len(self._entries) if declared_entry_count is None else declared_entry_count
        )
        current_head = (
            self._entries[-1].entry_hash if self._entries else self.GENESIS_HASH
        )
        self._declared_head_hash = (
            current_head if declared_head_hash is None else declared_head_hash
        )

    @property
    def entries(self) -> tuple[AuditChainEntry, ...]:
        """Return the entries in insertion order as an immutable tuple."""
        return tuple(self._entries)

    @property
    def head_hash(self) -> str:
        """Return the retained head hash that anchors the serialized chain."""
        return self._declared_head_hash

    def append(self, report: AuditReport) -> AuditChainEntry:
        """Verify and append one audit report without retaining raw PHI fields."""
        current = self.verify()
        if not current.valid:
            raise ValueError(
                f"cannot append to an invalid audit chain: {current.reason}"
            )
        previous_hash = (
            self._entries[-1].entry_hash if self._entries else self.GENESIS_HASH
        )
        entry = _entry_from_report(
            report,
            sequence=len(self._entries),
            previous_hash=previous_hash,
        )
        self._entries.append(entry)
        self._declared_entry_count = len(self._entries)
        self._declared_head_hash = entry.entry_hash
        return entry

    append_report = append

    def verify(self) -> ChainVerificationResult:
        """Verify count, order, links, entry hashes, and retained head offline."""
        actual_count = len(self._entries)
        if actual_count > self._declared_entry_count:
            return ChainVerificationResult(
                False,
                0,
                "insertion detected: chain contains more entries than declared",
            )
        if actual_count < self._declared_entry_count:
            return ChainVerificationResult(
                False,
                0,
                "deletion detected: chain contains fewer entries than declared",
            )

        sequences = [entry.sequence for entry in self._entries]
        expected_sequences = list(range(actual_count))
        if sequences != expected_sequences:
            if sorted(sequences) == expected_sequences:
                reason = "reordering detected: entry sequence is out of order"
            else:
                reason = (
                    "insertion or deletion detected: entry sequence is not contiguous"
                )
            return ChainVerificationResult(False, 0, reason)

        previous_hash = self.GENESIS_HASH
        for index, entry in enumerate(self._entries):
            if entry.previous_hash != previous_hash:
                return ChainVerificationResult(
                    False,
                    index,
                    f"reordering or mutation detected: broken link at entry {index}",
                    index,
                )
            if entry.entry_hash != entry.compute_hash():
                return ChainVerificationResult(
                    False,
                    index,
                    f"mutation detected: hash mismatch at entry {index}",
                    index,
                )
            previous_hash = entry.entry_hash

        if self._declared_head_hash != previous_hash:
            return ChainVerificationResult(
                False,
                actual_count,
                "deletion or mutation detected: retained head hash does not match",
            )
        return ChainVerificationResult(
            True,
            actual_count,
            "chain is valid",
        )

    def contains_report(self, report: AuditReport) -> bool:
        """Return whether a valid chain contains the supplied audit report."""
        return bool(self.verify()) and any(
            entry.matches_report(report) for entry in self._entries
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned chain envelope and retained head anchor."""
        return {
            "format": CHAIN_FORMAT,
            "version": CHAIN_VERSION,
            "genesis_hash": self.GENESIS_HASH,
            "entry_count": self._declared_entry_count,
            "head_hash": self._declared_head_hash,
            "entries": [entry.to_dict() for entry in self._entries],
        }

    def to_json(self) -> str:
        """Serialize the chain deterministically."""
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditChain":
        """Load a versioned audit chain from a mapping."""
        _reject_unknown_fields(data, _CHAIN_FIELDS, "audit chain")
        if data.get("format") != CHAIN_FORMAT:
            raise ValueError(f"audit chain format must be {CHAIN_FORMAT!r}")
        if data.get("version") != CHAIN_VERSION:
            raise ValueError("unsupported audit chain version")
        if data.get("genesis_hash") != GENESIS_HASH:
            raise ValueError("audit chain genesis_hash does not match this format")
        declared_count = data.get("entry_count")
        if type(declared_count) is not int or declared_count < 0:
            raise ValueError("audit chain entry_count must be a non-negative integer")
        declared_head = _require_sha256_hash(
            data.get("head_hash"), "audit chain head_hash"
        )
        entries = data.get("entries")
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise TypeError("audit chain entries must be a list")
        loaded_entries: list[AuditChainEntry] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise TypeError("audit chain entries must contain objects")
            loaded_entries.append(AuditChainEntry.from_dict(entry))
        return cls(
            loaded_entries,
            declared_entry_count=declared_count,
            declared_head_hash=declared_head,
        )

    @classmethod
    def from_json(cls, data: str | bytes) -> "AuditChain":
        """Load an audit chain from JSON."""
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON for AuditChain: {exc}") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError("AuditChain JSON must contain an object")
        return cls.from_dict(parsed)

    @classmethod
    def load(cls, path: str | Path) -> "AuditChain":
        """Load a UTF-8 audit-chain document from disk."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def write(self, path: str | Path) -> Path:
        """Atomically persist the complete chain document."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=destination.parent,
            encoding="utf-8",
            prefix=f".{destination.name}.",
        ) as handle:
            handle.write(self.to_json())
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, destination)
        return destination


def verify_chain(chain: AuditChain | Mapping[str, Any]) -> ChainVerificationResult:
    """Verify an audit chain supplied as an object or decoded JSON mapping."""
    if not isinstance(chain, AuditChain):
        chain = AuditChain.from_dict(chain)
    return chain.verify()


def append_to_chain_file(path: str | Path, report: AuditReport) -> AuditChainEntry:
    """Append one report to a new or existing chain file and persist atomically."""
    destination = Path(path)
    chain = AuditChain.load(destination) if destination.exists() else AuditChain()
    entry = chain.append(report)
    chain.write(destination)
    return entry


__all__ = [
    "AuditChain",
    "AuditChainEntry",
    "AuditChainSpan",
    "CHAIN_FORMAT",
    "CHAIN_VERSION",
    "ChainVerificationResult",
    "GENESIS_HASH",
    "append_to_chain_file",
    "verify_chain",
]
