"""Counts-only privacy audit artifacts for local trace stores.

The artifact contract intentionally has no field for source text, prompts,
tool results, replacement maps, paths, or arbitrary scanner metadata. Callers
provide already-computed hashes and aggregate counts; local file helpers only
emit SHA-256 fingerprints. This keeps JSON and Markdown exports useful as
evidence without making them a second trace store.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
ARTIFACT_NAME = "trace_privacy_audit"
_HASH_PREFIX = "sha256:"
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_SAFE_HASH_RE = re.compile(r"^sha256:[A-Za-z0-9_-]{1,128}$")
_FINGERPRINT_KEYS = ("fingerprint", "file_fingerprint", "hash")


class TraceAuditError(ValueError):
    """Raised when a counts-only trace audit cannot be validated or written."""


def hash_bytes(value: bytes) -> str:
    """Return a stable SHA-256 digest for bytes without retaining the bytes."""

    if not isinstance(value, bytes):
        raise TraceAuditError("audit hash input must be bytes")
    return f"{_HASH_PREFIX}{hashlib.sha256(value).hexdigest()}"


def hash_policy(policy: str | bytes) -> str:
    """Hash policy content for use as the artifact's policy reference.

    The policy content is consumed only to calculate the digest and is never
    stored in the returned artifact.
    """

    if isinstance(policy, str):
        policy = policy.encode("utf-8")
    return hash_bytes(policy)


def fingerprint_file(path: str | Path) -> str:
    """Return a SHA-256 fingerprint of one local file.

    Only file bytes are read. The path is not included in the returned value,
    and failures use a value-free exception so sensitive path components do
    not enter logs or error reports.
    """

    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except Exception:
        raise TraceAuditError("unable to fingerprint trace file") from None
    return f"{_HASH_PREFIX}{digest.hexdigest()}"


def count_categories(categories: Iterable[str]) -> dict[str, int]:
    """Count safe category labels without retaining any finding details."""

    if isinstance(categories, str):
        categories = (categories,)
    try:
        values = tuple(categories)
    except (TypeError, ValueError):
        raise TraceAuditError("category labels must be iterable") from None

    counts: dict[str, int] = {}
    for category in values:
        label = _safe_token(category, "category labels")
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


@dataclass(frozen=True)
class TraceAuditArtifact:
    """Deterministic, counts-only evidence for one trace privacy scan.

    ``file_fingerprints`` contains content fingerprints only; it never
    contains file names or paths. ``category_counts`` contains label-to-count
    pairs only. The fixed field set is the privacy boundary: source values,
    replacement mappings, prompts, tool outputs, and arbitrary metadata have
    nowhere to be stored or serialized.
    """

    scanner_version: str
    policy_hash: str
    file_fingerprints: tuple[str, ...]
    category_counts: Mapping[str, int]
    disposition: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise TraceAuditError("unsupported trace audit schema version")

        object.__setattr__(
            self,
            "scanner_version",
            _safe_token(self.scanner_version, "scanner version"),
        )
        object.__setattr__(
            self,
            "policy_hash",
            _safe_hash(self.policy_hash, "policy hash"),
        )
        object.__setattr__(
            self,
            "file_fingerprints",
            _coerce_fingerprints(self.file_fingerprints),
        )
        object.__setattr__(
            self,
            "category_counts",
            _coerce_category_counts(self.category_counts),
        )
        object.__setattr__(
            self,
            "disposition",
            _safe_token(self.disposition, "disposition"),
        )

    @property
    def file_count(self) -> int:
        """Return the number of fingerprints in the artifact."""

        return len(self.file_fingerprints)

    @property
    def finding_count(self) -> int:
        """Return the aggregate number of findings represented by counts."""

        return sum(self.category_counts.values())

    def to_dict(self) -> dict[str, Any]:
        """Return the allowlisted JSON-compatible artifact payload."""

        return {
            "artifact": ARTIFACT_NAME,
            "category_counts": dict(self.category_counts),
            "disposition": self.disposition,
            "file_fingerprints": list(self.file_fingerprints),
            "policy_hash": self.policy_hash,
            "scanner_version": self.scanner_version,
            "schema_version": self.schema_version,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the artifact as deterministic JSON."""

        if indent is not None and (type(indent) is not int or indent < 0):
            raise TraceAuditError("JSON indentation must be a non-negative integer")
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TraceAuditArtifact":
        """Build an artifact from its allowlisted serialized fields.

        Unknown fields are ignored rather than copied. This permits a scanner
        summary to carry internal details at its boundary without allowing
        those details into the audit artifact.
        """

        if not isinstance(payload, Mapping):
            raise TraceAuditError("trace audit JSON must contain an object")
        if payload.get("artifact", ARTIFACT_NAME) != ARTIFACT_NAME:
            raise TraceAuditError("invalid trace audit artifact type")
        return cls(
            scanner_version=payload.get("scanner_version"),
            policy_hash=payload.get("policy_hash"),
            file_fingerprints=payload.get("file_fingerprints", ()),
            category_counts=payload.get("category_counts", {}),
            disposition=payload.get("disposition"),
            schema_version=payload.get("schema_version", SCHEMA_VERSION),
        )

    @classmethod
    def from_json(cls, payload: str | bytes) -> "TraceAuditArtifact":
        """Parse deterministic JSON without echoing malformed input."""

        try:
            parsed = json.loads(payload)
        except (TypeError, ValueError, UnicodeError):
            raise TraceAuditError("invalid JSON for trace audit artifact") from None
        return cls.from_dict(parsed)

    @classmethod
    def read_json(cls, path: str | Path) -> "TraceAuditArtifact":
        """Read one local JSON artifact with value-free failures."""

        try:
            payload = Path(path).read_bytes()
        except Exception:
            raise TraceAuditError("unable to read trace audit artifact") from None
        return cls.from_json(payload)

    def write_json(self, path: str | Path, *, indent: int | None = 2) -> Path:
        """Write deterministic JSON to a local path and return that path."""

        try:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        except TraceAuditError:
            raise
        except Exception:
            raise TraceAuditError("unable to write trace audit artifact") from None
        return output_path

    def to_markdown(self) -> str:
        """Serialize the artifact as deterministic, counts-only Markdown."""

        lines = [
            "# Trace Privacy Audit",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Scanner version | `{_markdown_cell(self.scanner_version)}` |",
            f"| Policy hash | `{_markdown_cell(self.policy_hash)}` |",
            f"| Disposition | `{_markdown_cell(self.disposition)}` |",
            f"| Files fingerprinted | {self.file_count} |",
            f"| Findings counted | {self.finding_count} |",
            "",
            "## File fingerprints",
            "",
            "| Fingerprint |",
            "|---|",
        ]
        if self.file_fingerprints:
            lines.extend(
                f"| `{_markdown_cell(fingerprint)}` |"
                for fingerprint in self.file_fingerprints
            )
        else:
            lines.append("| _None_ |")

        lines.extend(
            [
                "",
                "## Category counts",
                "",
                "| Category | Count |",
                "|---|---:|",
            ]
        )
        if self.category_counts:
            lines.extend(
                f"| `{_markdown_cell(category)}` | {count} |"
                for category, count in self.category_counts.items()
            )
        else:
            lines.append("| _None_ | 0 |")
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown to a local path and return that path."""

        try:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(self.to_markdown(), encoding="utf-8")
        except Exception:
            raise TraceAuditError("unable to write trace audit Markdown") from None
        return output_path

    @classmethod
    def from_scan_summary(
        cls,
        summary: Mapping[str, Any],
        *,
        scanner_version: str | None = None,
        policy_hash: str | None = None,
        disposition: str | None = None,
    ) -> "TraceAuditArtifact":
        """Extract only safe fields from a scanner summary mapping."""

        if not isinstance(summary, Mapping):
            raise TraceAuditError("trace scan summary must contain an object")
        return cls(
            scanner_version=(
                summary.get("scanner_version")
                if scanner_version is None
                else scanner_version
            ),
            policy_hash=(
                policy_hash if policy_hash is not None else summary.get("policy_hash")
            ),
            file_fingerprints=summary.get("file_fingerprints", ()),
            category_counts=summary.get("category_counts", {}),
            disposition=(
                summary.get("disposition") if disposition is None else disposition
            ),
        )

    from_summary = from_scan_summary

    @classmethod
    def from_files(
        cls,
        scanner_version: str,
        policy_hash: str,
        files: Iterable[str | Path] | str | Path,
        category_counts: Mapping[str, int],
        disposition: str = "unknown",
    ) -> "TraceAuditArtifact":
        """Create an artifact by fingerprinting explicitly supplied local files."""

        return cls(
            scanner_version=scanner_version,
            policy_hash=policy_hash,
            file_fingerprints=_fingerprints_for_paths(files),
            category_counts=category_counts,
            disposition=disposition,
        )


def build_trace_audit(
    scanner_version: str,
    policy_hash: str,
    file_fingerprints: object = (),
    category_counts: Mapping[str, int] | None = None,
    disposition: str = "unknown",
    *,
    files: Iterable[str | Path] | str | Path | None = None,
) -> TraceAuditArtifact:
    """Build a counts-only artifact from safe summaries and local file paths.

    ``file_fingerprints`` accepts precomputed SHA-256 references. If ``files``
    is supplied, each explicitly named local file is hashed and only the
    resulting fingerprint is retained.
    """

    fingerprints = list(_coerce_fingerprints(file_fingerprints))
    if files is not None:
        fingerprints.extend(_fingerprints_for_paths(files))
    return TraceAuditArtifact(
        scanner_version=scanner_version,
        policy_hash=policy_hash,
        file_fingerprints=fingerprints,
        category_counts={} if category_counts is None else category_counts,
        disposition=disposition,
    )


def render_trace_audit_json(
    artifact: TraceAuditArtifact, *, indent: int | None = 2
) -> str:
    """Render one validated trace audit artifact as JSON."""

    if not isinstance(artifact, TraceAuditArtifact):
        raise TraceAuditError("trace audit renderer requires a trace audit artifact")
    return artifact.to_json(indent=indent)


def render_trace_audit_markdown(artifact: TraceAuditArtifact) -> str:
    """Render one validated trace audit artifact as Markdown."""

    if not isinstance(artifact, TraceAuditArtifact):
        raise TraceAuditError("trace audit renderer requires a trace audit artifact")
    return artifact.to_markdown()


def _safe_token(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _SAFE_TOKEN_RE.fullmatch(value):
        raise TraceAuditError(f"{field_name} must be a safe identifier")
    return value


def _safe_hash(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _SAFE_HASH_RE.fullmatch(value):
        raise TraceAuditError(f"{field_name} must be a SHA-256 reference")
    return value


def _coerce_fingerprint(value: object) -> str:
    if isinstance(value, bytes):
        return hash_bytes(value)
    if isinstance(value, Path):
        return fingerprint_file(value)
    if isinstance(value, Mapping):
        for key in _FINGERPRINT_KEYS:
            if key in value:
                return _coerce_fingerprint(value[key])
        raise TraceAuditError("file fingerprint entry is missing a fingerprint")
    return _safe_hash(value, "file fingerprint")


def _coerce_fingerprints(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        if any(key in value for key in _FINGERPRINT_KEYS):
            values: Iterable[object] = (value,)
        else:
            values = value.values()
    elif isinstance(value, (str, bytes, Path)):
        values = (value,)
    else:
        try:
            values = tuple(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise TraceAuditError("file fingerprints must be iterable") from None
    fingerprints = tuple(sorted(_coerce_fingerprint(item) for item in values))
    return fingerprints


def _coerce_category_counts(value: object) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TraceAuditError("category counts must be a mapping")
    counts: dict[str, int] = {}
    for category, count in value.items():
        label = _safe_token(category, "category")
        if type(count) is not int or count < 0:
            raise TraceAuditError("category counts must be non-negative integers")
        counts[label] = count
    return dict(sorted(counts.items()))


def _fingerprints_for_paths(
    paths: Iterable[str | Path] | str | Path,
) -> tuple[str, ...]:
    if isinstance(paths, (str, Path)):
        paths = (paths,)
    try:
        values = tuple(paths)
    except (TypeError, ValueError):
        raise TraceAuditError("trace files must be iterable") from None
    fingerprints: list[str] = []
    for path in values:
        if not isinstance(path, (str, Path)):
            raise TraceAuditError("trace files must be local paths")
        fingerprints.append(fingerprint_file(path))
    return tuple(sorted(fingerprints))


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("`", "\\`")


TraceAudit = TraceAuditArtifact
TracePrivacyAudit = TraceAuditArtifact


__all__ = [
    "ARTIFACT_NAME",
    "SCHEMA_VERSION",
    "TraceAudit",
    "TraceAuditArtifact",
    "TraceAuditError",
    "TracePrivacyAudit",
    "build_trace_audit",
    "count_categories",
    "fingerprint_file",
    "hash_bytes",
    "hash_policy",
    "render_trace_audit_json",
    "render_trace_audit_markdown",
]
