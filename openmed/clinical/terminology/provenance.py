"""Value-free provenance and expiry reporting for local terminology snapshots.

The helpers in this module record only snapshot metadata: a caller-supplied
source name and version, a SHA-256 checksum, import time, and an explicit
expiry policy. Snapshot bytes and terminology values are accepted only long
enough to calculate a checksum and are never retained in a manifest, report,
exception, or cache artifact.

All report builders are offline and deterministic. A reference time is
required when evaluating freshness so the same manifest and reference time
produce byte-identical reports regardless of the machine clock.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROVENANCE_SCHEMA_VERSION = 1
_SHA256_HEX_LENGTH = 64
_CHECKSUM_CHUNK_SIZE = 1024 * 1024
_UTC = timezone.utc

EXPIRY_STATUS_FRESH = "fresh"
EXPIRY_STATUS_EXPIRING = "expiring"
EXPIRY_STATUS_EXPIRED = "expired"
EXPIRY_STATUS_NO_POLICY = "no_expiry_policy"
EXPIRY_STATUS_FUTURE = "not_yet_imported"

_EXPIRY_ACTIONS = frozenset({"allow", "report", "reject"})
_POLICY_KEYS = frozenset(
    {
        "action",
        "expires_at",
        "max_age_days",
        "on_expiry",
        "reject_expired",
        "ttl_days",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "checksum",
        "expiry_policy",
        "imported_at",
        "schema_version",
        "source_name",
        "source_version",
    }
)


class SnapshotManifestError(ValueError):
    """Raised when value-free terminology provenance is malformed."""


class SnapshotExpiredError(SnapshotManifestError):
    """Raised when an expired snapshot is disallowed by its policy."""


def _invalid(field_name: str) -> SnapshotManifestError:
    """Return a safe validation error that never echoes caller data."""

    return SnapshotManifestError(f"{field_name} is invalid")


def _non_empty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise _invalid(field_name)
    normalized = value.strip()
    if not normalized or any(ord(character) < 32 for character in normalized):
        raise _invalid(field_name)
    return normalized


def _normalize_checksum(value: object) -> str:
    if not isinstance(value, str):
        raise _invalid("checksum")
    normalized = value.strip().lower()
    if normalized.startswith("sha256:"):
        normalized = normalized.removeprefix("sha256:")
    if len(normalized) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise _invalid("checksum")
    return f"sha256:{normalized}"


def _coerce_datetime(value: object, field_name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time())
    elif isinstance(value, str):
        normalized = value.strip()
        try:
            parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            raise _invalid(field_name) from None
    else:
        raise _invalid(field_name)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_UTC)
    try:
        return parsed.astimezone(_UTC)
    except (OverflowError, ValueError):
        raise _invalid(field_name) from None


def _format_datetime(value: datetime) -> str:
    normalized = value.astimezone(_UTC)
    timespec = "microseconds" if normalized.microsecond else "seconds"
    return normalized.isoformat(timespec=timespec).replace("+00:00", "Z")


def _canonical_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        indent=indent,
        separators=(",", ":") if indent is None else None,
        sort_keys=True,
    )


def _round_days(value: float) -> float:
    rounded = round(value, 6)
    return int(rounded) if rounded.is_integer() else rounded


@dataclass(frozen=True)
class ExpiryPolicy:
    """Rules for determining and enforcing a terminology snapshot expiry.

    ``max_age_days`` is measured from the manifest's ``imported_at`` value.
    Alternatively, callers may provide an absolute ``expires_at`` timestamp.
    Setting ``reject_expired`` (or ``action='reject'`` when loading a mapping)
    makes :func:`require_fresh_snapshot` fail closed after expiry.
    """

    max_age_days: int | None = None
    reject_expired: bool = False
    expires_at: str | datetime | None = None

    def __post_init__(self) -> None:
        if self.max_age_days is not None and (
            type(self.max_age_days) is not int or self.max_age_days < 0
        ):
            raise _invalid("expiry_policy.max_age_days")
        if type(self.reject_expired) is not bool:
            raise _invalid("expiry_policy.reject_expired")
        if self.expires_at is not None:
            if self.max_age_days is not None:
                raise _invalid("expiry_policy")
            object.__setattr__(
                self,
                "expires_at",
                _format_datetime(_coerce_datetime(self.expires_at, "expires_at")),
            )

    @classmethod
    def from_value(cls, value: object) -> "ExpiryPolicy":
        """Normalize a policy object or JSON-compatible policy mapping."""

        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if type(value) is int:
            return cls(max_age_days=value)
        if not isinstance(value, Mapping):
            raise _invalid("expiry_policy")
        if any(key not in _POLICY_KEYS for key in value):
            raise _invalid("expiry_policy")

        max_age_days = value.get("max_age_days", value.get("ttl_days"))
        action = value.get("action", value.get("on_expiry"))
        reject_expired = value.get("reject_expired", False)
        if action is not None:
            if (
                not isinstance(action, str)
                or action.strip().lower() not in _EXPIRY_ACTIONS
            ):
                raise _invalid("expiry_policy.action")
            action_rejects = action.strip().lower() == "reject"
            if "reject_expired" in value and reject_expired is not action_rejects:
                raise _invalid("expiry_policy")
            reject_expired = action_rejects

        return cls(
            max_age_days=max_age_days,
            reject_expired=reject_expired,
            expires_at=value.get("expires_at"),
        )

    @property
    def configured(self) -> bool:
        """Return whether this policy defines an expiry boundary."""

        return self.max_age_days is not None or self.expires_at is not None

    def expiry_datetime(self, imported_at: str | datetime) -> datetime | None:
        """Return the effective UTC expiry timestamp for an imported snapshot."""

        if self.expires_at is not None:
            return _coerce_datetime(self.expires_at, "expiry_policy.expires_at")
        if self.max_age_days is None:
            return None
        imported = _coerce_datetime(imported_at, "imported_at")
        return imported + timedelta(days=self.max_age_days)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation of this policy."""

        payload: dict[str, Any] = {
            "max_age_days": self.max_age_days,
            "reject_expired": self.reject_expired,
        }
        if self.expires_at is not None:
            payload["expires_at"] = self.expires_at
        return payload


@dataclass(frozen=True)
class SnapshotManifest:
    """Value-free identity and freshness policy for one local snapshot."""

    source_name: str
    source_version: str
    checksum: str
    imported_at: str | datetime
    expiry_policy: ExpiryPolicy | Mapping[str, Any] | int | None = None
    schema_version: int = PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROVENANCE_SCHEMA_VERSION:
            raise _invalid("schema_version")
        object.__setattr__(
            self, "source_name", _non_empty_text(self.source_name, "source_name")
        )
        object.__setattr__(
            self,
            "source_version",
            _non_empty_text(self.source_version, "source_version"),
        )
        object.__setattr__(self, "checksum", _normalize_checksum(self.checksum))
        object.__setattr__(
            self,
            "imported_at",
            _format_datetime(_coerce_datetime(self.imported_at, "imported_at")),
        )
        object.__setattr__(
            self, "expiry_policy", ExpiryPolicy.from_value(self.expiry_policy)
        )

    @property
    def version(self) -> str:
        """Alias for the source release version."""

        return self.source_version

    @property
    def import_time(self) -> str:
        """Alias for the canonical import timestamp."""

        return self.imported_at

    @property
    def source_checksum(self) -> str:
        """Alias for the canonical source checksum."""

        return self.checksum

    @property
    def expires_at(self) -> str | None:
        """Return the effective expiry timestamp, if the policy has one."""

        expiry = self.expiry_policy.expiry_datetime(self.imported_at)
        return _format_datetime(expiry) if expiry is not None else None

    def is_expired(self, as_of: str | datetime | date) -> bool:
        """Return whether the snapshot is at or beyond its expiry boundary."""

        expiry = self.expiry_policy.expiry_datetime(self.imported_at)
        return expiry is not None and _coerce_datetime(as_of, "as_of") >= expiry

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-ready value-free manifest."""

        return {
            "checksum": self.checksum,
            "expiry_policy": self.expiry_policy.to_dict(),
            "imported_at": self.imported_at,
            "schema_version": self.schema_version,
            "source_name": self.source_name,
            "source_version": self.source_version,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize this manifest deterministically as JSON."""

        return _canonical_json(self.to_dict(), indent=indent)

    def write_json(self, path: str | Path, *, indent: int | None = 2) -> Path:
        """Write this manifest to a local JSON file without network access."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SnapshotManifest":
        """Load and validate one manifest mapping."""

        if not isinstance(value, Mapping) or set(value) != _MANIFEST_KEYS:
            raise _invalid("snapshot manifest")
        return cls(
            source_name=value["source_name"],
            source_version=value["source_version"],
            checksum=value["checksum"],
            imported_at=value["imported_at"],
            expiry_policy=value["expiry_policy"],
            schema_version=value["schema_version"],
        )

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "SnapshotManifest":
        """Load and validate one manifest JSON document."""

        try:
            payload = json.loads(value)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            raise _invalid("snapshot manifest JSON") from None
        if not isinstance(payload, Mapping):
            raise _invalid("snapshot manifest JSON")
        return cls.from_mapping(payload)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Any,
        *,
        source_name: str | None = None,
        source_version: str | None = None,
        imported_at: str | datetime,
        expiry_policy: ExpiryPolicy | Mapping[str, Any] | int | None = None,
    ) -> "SnapshotManifest":
        """Build a manifest from an already validated local snapshot object.

        The object is read for metadata only. Its terminology values are not
        copied into the resulting manifest.
        """

        resolved_name = source_name or getattr(snapshot, "system_uri", None)
        resolved_version = source_version or getattr(snapshot, "release_version", None)
        resolved_checksum = getattr(snapshot, "content_hash", None)
        if (
            resolved_name is None
            or resolved_version is None
            or resolved_checksum is None
        ):
            raise _invalid("snapshot")
        return cls(
            source_name=resolved_name,
            source_version=resolved_version,
            checksum=resolved_checksum,
            imported_at=imported_at,
            expiry_policy=expiry_policy,
        )

    def __getitem__(self, key: str) -> Any:
        """Provide mapping-style access to the serialized fields."""

        return self.to_dict()[key]


TerminologySnapshotManifest = SnapshotManifest


def checksum_bytes(snapshot: bytes | bytearray | memoryview) -> str:
    """Return a canonical SHA-256 checksum without retaining snapshot bytes."""

    if not isinstance(snapshot, (bytes, bytearray, memoryview)):
        raise _invalid("snapshot")
    return f"sha256:{hashlib.sha256(bytes(snapshot)).hexdigest()}"


def checksum_file(path: str | Path) -> str:
    """Return the canonical checksum of one local snapshot file."""

    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as snapshot_file:
            while chunk := snapshot_file.read(_CHECKSUM_CHUNK_SIZE):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"
    except (OSError, TypeError, ValueError):
        raise _invalid("snapshot path") from None


def build_snapshot_manifest(
    source_name: str,
    source_version: str,
    snapshot: bytes | bytearray | memoryview | str | Path | None = None,
    *,
    checksum: str | None = None,
    imported_at: str | datetime,
    expiry_policy: ExpiryPolicy | Mapping[str, Any] | int | None = None,
) -> SnapshotManifest:
    """Build a value-free manifest from bytes or a caller-supplied checksum.

    ``snapshot`` may be bytes-like data, a text payload, or a local path. The
    payload is used only for hashing. Supplying ``checksum`` is useful when the
    snapshot has already been verified by a local loader and avoids rereading
    it. Exactly one of ``snapshot`` and ``checksum`` is required.
    """

    if snapshot is not None and checksum is not None:
        raise _invalid("snapshot/checksum")
    if checksum is None:
        if snapshot is None:
            raise _invalid("snapshot")
        if isinstance(snapshot, Path):
            checksum = checksum_file(snapshot)
        elif isinstance(snapshot, str):
            checksum = checksum_bytes(snapshot.encode("utf-8"))
        else:
            checksum = checksum_bytes(snapshot)
    return SnapshotManifest(
        source_name=source_name,
        source_version=source_version,
        checksum=checksum,
        imported_at=imported_at,
        expiry_policy=expiry_policy,
    )


def save_snapshot_manifest(
    manifest: SnapshotManifest | Mapping[str, Any],
    path: str | Path,
    *,
    indent: int | None = 2,
) -> Path:
    """Validate and write one local snapshot manifest."""

    return _coerce_manifest(manifest).write_json(path, indent=indent)


def load_snapshot_manifest(path: str | Path) -> SnapshotManifest:
    """Load one local snapshot manifest without making a network request."""

    try:
        return SnapshotManifest.from_json(Path(path).read_bytes())
    except SnapshotManifestError:
        raise
    except (OSError, TypeError, ValueError):
        raise _invalid("snapshot manifest path") from None


@dataclass(frozen=True)
class SnapshotFreshness:
    """Deterministic freshness evaluation for one snapshot manifest."""

    source_name: str
    source_version: str
    checksum: str
    imported_at: str
    expires_at: str | None
    status: str
    age_days: float
    days_remaining: float | None
    reject_expired: bool

    @property
    def rejection_required(self) -> bool:
        """Return whether this record requires fail-closed enforcement."""

        return self.status == EXPIRY_STATUS_EXPIRED and self.reject_expired

    @property
    def expired(self) -> bool:
        """Return whether the snapshot is expired."""

        return self.status == EXPIRY_STATUS_EXPIRED

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready freshness record."""

        return {
            "age_days": self.age_days,
            "checksum": self.checksum,
            "days_remaining": self.days_remaining,
            "expires_at": self.expires_at,
            "imported_at": self.imported_at,
            "rejection_required": self.rejection_required,
            "reject_expired": self.reject_expired,
            "source_name": self.source_name,
            "source_version": self.source_version,
            "status": self.status,
        }

    def __getitem__(self, key: str) -> Any:
        """Provide mapping-style access to the serialized fields."""

        return self.to_dict()[key]


@dataclass(frozen=True)
class FreshnessReport:
    """Deterministic freshness report for an ordered set of manifests."""

    as_of: str
    snapshots: tuple[SnapshotFreshness, ...]
    expiring_within_days: int
    schema_version: int = PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshots", tuple(self.snapshots))

    @property
    def entries(self) -> tuple[SnapshotFreshness, ...]:
        """Alias for the report's per-snapshot records."""

        return self.snapshots

    @property
    def expired(self) -> tuple[SnapshotFreshness, ...]:
        """Return expired records in deterministic report order."""

        return tuple(record for record in self.snapshots if record.expired)

    @property
    def rejection_required(self) -> bool:
        """Return whether any snapshot policy requires rejection."""

        return any(record.rejection_required for record in self.snapshots)

    @property
    def ok(self) -> bool:
        """Return whether no configured policy requires rejection."""

        return not self.rejection_required

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready freshness report."""

        counts = {
            "expired": sum(
                record.status == EXPIRY_STATUS_EXPIRED for record in self.snapshots
            ),
            "expiring": sum(
                record.status == EXPIRY_STATUS_EXPIRING for record in self.snapshots
            ),
            "fresh": sum(
                record.status == EXPIRY_STATUS_FRESH for record in self.snapshots
            ),
            "not_yet_imported": sum(
                record.status == EXPIRY_STATUS_FUTURE for record in self.snapshots
            ),
            "no_expiry_policy": sum(
                record.status == EXPIRY_STATUS_NO_POLICY for record in self.snapshots
            ),
            "rejection_required": sum(
                record.rejection_required for record in self.snapshots
            ),
            "total": len(self.snapshots),
        }
        return {
            "as_of": self.as_of,
            "expiring_within_days": self.expiring_within_days,
            "ok": self.ok,
            "schema_version": self.schema_version,
            "snapshots": [record.to_dict() for record in self.snapshots],
            "summary": counts,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the freshness report deterministically as JSON."""

        return _canonical_json(self.to_dict(), indent=indent)

    def write_json(self, path: str | Path, *, indent: int | None = 2) -> Path:
        """Write the freshness report to a local JSON file."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Render a deterministic Markdown freshness report."""

        lines = [
            "# Terminology Snapshot Freshness Report",
            "",
            f"As of: `{self.as_of}`",
            "",
            "| Source | Version | Checksum | Imported At | Expires At | Status | Reject |",
            "|---|---|---|---|---|---|---|",
        ]
        for record in self.snapshots:
            lines.append(
                "| "
                + " | ".join(
                    (
                        _markdown_cell(record.source_name),
                        _markdown_cell(record.source_version),
                        f"`{record.checksum}`",
                        f"`{record.imported_at}`",
                        f"`{record.expires_at or 'none'}`",
                        f"`{record.status}`",
                        "yes" if record.rejection_required else "no",
                    )
                )
                + " |"
            )
        summary = self.to_dict()["summary"]
        lines.extend(
            [
                "",
                "## Summary",
                "",
                f"- Total snapshots: {summary['total']}",
                f"- Fresh: {summary['fresh']}",
                f"- Expiring: {summary['expiring']}",
                f"- Expired: {summary['expired']}",
                f"- Rejection required: {'yes' if self.rejection_required else 'no'}",
            ]
        )
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write the freshness report to a local Markdown file."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def __getitem__(self, key: str) -> Any:
        """Provide mapping-style access to the serialized report."""

        return self.to_dict()[key]


@dataclass(frozen=True)
class ProvenanceReport:
    """Deterministic report of value-free snapshot provenance."""

    snapshots: tuple[SnapshotManifest, ...]
    schema_version: int = PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshots", tuple(self.snapshots))

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready provenance report."""

        return {
            "schema_version": self.schema_version,
            "snapshot_count": len(self.snapshots),
            "snapshots": [manifest.to_dict() for manifest in self.snapshots],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the provenance report deterministically as JSON."""

        return _canonical_json(self.to_dict(), indent=indent)

    def write_json(self, path: str | Path, *, indent: int | None = 2) -> Path:
        """Write the provenance report to a local JSON file."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Render a deterministic Markdown provenance report."""

        lines = [
            "# Terminology Snapshot Provenance Report",
            "",
            "| Source | Version | Checksum | Imported At | Expiry Policy |",
            "|---|---|---|---|---|",
        ]
        for manifest in self.snapshots:
            policy = _canonical_json(manifest.expiry_policy.to_dict())
            lines.append(
                "| "
                + " | ".join(
                    (
                        _markdown_cell(manifest.source_name),
                        _markdown_cell(manifest.source_version),
                        f"`{manifest.checksum}`",
                        f"`{manifest.imported_at}`",
                        f"`{_markdown_cell(policy)}`",
                    )
                )
                + " |"
            )
        lines.extend(["", f"Snapshot count: {len(self.snapshots)}", ""])
        return "\n".join(lines)

    def write_markdown(self, path: str | Path) -> Path:
        """Write the provenance report to a local Markdown file."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def __getitem__(self, key: str) -> Any:
        """Provide mapping-style access to the serialized report."""

        return self.to_dict()[key]


def _coerce_manifest(value: SnapshotManifest | Mapping[str, Any]) -> SnapshotManifest:
    if isinstance(value, SnapshotManifest):
        return value
    if isinstance(value, Mapping):
        return SnapshotManifest.from_mapping(value)
    raise _invalid("snapshot manifest")


def _sorted_manifests(
    manifests: Iterable[SnapshotManifest | Mapping[str, Any]],
) -> tuple[SnapshotManifest, ...]:
    normalized = tuple(_coerce_manifest(manifest) for manifest in manifests)
    return tuple(
        sorted(
            normalized,
            key=lambda manifest: (
                manifest.source_name,
                manifest.source_version,
                manifest.checksum,
            ),
        )
    )


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _freshness_for(
    manifest: SnapshotManifest,
    *,
    as_of: datetime,
    expiring_within_days: int,
) -> SnapshotFreshness:
    imported = _coerce_datetime(manifest.imported_at, "imported_at")
    expiry = manifest.expiry_policy.expiry_datetime(manifest.imported_at)
    age_days = _round_days((as_of - imported).total_seconds() / 86400)
    if as_of < imported:
        status = EXPIRY_STATUS_FUTURE
    elif expiry is None:
        status = EXPIRY_STATUS_NO_POLICY
    elif as_of >= expiry:
        status = EXPIRY_STATUS_EXPIRED
    else:
        days_remaining = (expiry - as_of).total_seconds() / 86400
        status = (
            EXPIRY_STATUS_EXPIRING
            if days_remaining <= expiring_within_days
            else EXPIRY_STATUS_FRESH
        )
    days_remaining = (
        _round_days((expiry - as_of).total_seconds() / 86400)
        if expiry is not None
        else None
    )
    return SnapshotFreshness(
        source_name=manifest.source_name,
        source_version=manifest.source_version,
        checksum=manifest.checksum,
        imported_at=manifest.imported_at,
        expires_at=manifest.expires_at,
        status=status,
        age_days=age_days,
        days_remaining=days_remaining,
        reject_expired=manifest.expiry_policy.reject_expired,
    )


def _resolve_as_of(
    as_of: str | datetime | date | None,
    now: str | datetime | date | None,
) -> datetime:
    if as_of is not None and now is not None:
        raise _invalid("as_of/now")
    reference = as_of if as_of is not None else now
    if reference is None:
        raise SnapshotManifestError(
            "as_of is required for deterministic freshness evaluation"
        )
    return _coerce_datetime(reference, "as_of")


def build_freshness_report(
    manifests: Iterable[SnapshotManifest | Mapping[str, Any]],
    as_of: str | datetime | date | None = None,
    *,
    now: str | datetime | date | None = None,
    expiring_within_days: int = 30,
) -> FreshnessReport:
    """Evaluate manifests at a fixed reference time in stable sort order."""

    if type(expiring_within_days) is not int or expiring_within_days < 0:
        raise _invalid("expiring_within_days")
    reference = _resolve_as_of(as_of, now)
    ordered = _sorted_manifests(manifests)
    return FreshnessReport(
        as_of=_format_datetime(reference),
        snapshots=tuple(
            _freshness_for(
                manifest,
                as_of=reference,
                expiring_within_days=expiring_within_days,
            )
            for manifest in ordered
        ),
        expiring_within_days=expiring_within_days,
    )


freshness_report = build_freshness_report


def build_provenance_report(
    manifests: Iterable[SnapshotManifest | Mapping[str, Any]],
) -> ProvenanceReport:
    """Build a value-free provenance report in deterministic order."""

    return ProvenanceReport(snapshots=_sorted_manifests(manifests))


provenance_report = build_provenance_report


def _render_format(format_name: str) -> str:
    if not isinstance(format_name, str):
        raise _invalid("format")
    normalized = format_name.strip().lower()
    if normalized not in {"json", "markdown", "md"}:
        raise _invalid("format")
    return "json" if normalized == "json" else "markdown"


def render_freshness_report(
    manifests: Iterable[SnapshotManifest | Mapping[str, Any]],
    as_of: str | datetime | date | None = None,
    *,
    now: str | datetime | date | None = None,
    expiring_within_days: int = 30,
    format: str = "markdown",
) -> str:
    """Render a deterministic JSON or Markdown freshness report."""

    report = build_freshness_report(
        manifests,
        as_of,
        now=now,
        expiring_within_days=expiring_within_days,
    )
    return (
        report.to_json() if _render_format(format) == "json" else report.to_markdown()
    )


def render_provenance_report(
    manifests: Iterable[SnapshotManifest | Mapping[str, Any]],
    *,
    format: str = "markdown",
) -> str:
    """Render a deterministic JSON or Markdown provenance report."""

    report = build_provenance_report(manifests)
    return (
        report.to_json() if _render_format(format) == "json" else report.to_markdown()
    )


def is_snapshot_expired(
    manifest: SnapshotManifest | Mapping[str, Any],
    as_of: str | datetime | date,
) -> bool:
    """Return whether a manifest is expired at a fixed reference time."""

    return _coerce_manifest(manifest).is_expired(as_of)


def require_fresh_snapshot(
    manifest: SnapshotManifest | Mapping[str, Any],
    as_of: str | datetime | date | None = None,
    *,
    now: str | datetime | date | None = None,
) -> SnapshotManifest:
    """Return a manifest or reject it when its policy requires freshness."""

    normalized = _coerce_manifest(manifest)
    reference = _resolve_as_of(as_of, now)
    if normalized.expiry_policy.reject_expired and normalized.is_expired(reference):
        raise SnapshotExpiredError(
            "terminology snapshot is expired and its policy requires rejection"
        )
    return normalized


__all__ = [
    "EXPIRY_STATUS_EXPIRED",
    "EXPIRY_STATUS_EXPIRING",
    "EXPIRY_STATUS_FRESH",
    "EXPIRY_STATUS_FUTURE",
    "EXPIRY_STATUS_NO_POLICY",
    "PROVENANCE_SCHEMA_VERSION",
    "ExpiryPolicy",
    "FreshnessReport",
    "ProvenanceReport",
    "SnapshotExpiredError",
    "SnapshotFreshness",
    "SnapshotManifest",
    "SnapshotManifestError",
    "TerminologySnapshotManifest",
    "build_freshness_report",
    "build_provenance_report",
    "build_snapshot_manifest",
    "checksum_bytes",
    "checksum_file",
    "freshness_report",
    "is_snapshot_expired",
    "load_snapshot_manifest",
    "provenance_report",
    "render_freshness_report",
    "render_provenance_report",
    "require_fresh_snapshot",
    "save_snapshot_manifest",
]
