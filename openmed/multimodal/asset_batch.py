"""Privacy-safe multimodal asset batches for preflight.

A batch groups an opaque batch identifier with an ordered tuple of
:class:`~openmed.multimodal.asset_manifest.AssetManifest` records so clinical
packets that contain several media assets can be ordered, de-duplicated, and
summarized before any asset is opened. The batch carries only the bounded,
non-identifying facts that the manifests already hold. It rejects duplicate
asset identifiers, repeated content digests, non-canonical ordering, oversized
batches, aggregate overflow, and declared totals that disagree with the
manifests, and it never records paths, filenames, extracted text, or bytes.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from .asset_manifest import (
    MAX_MANIFEST_BYTE_SIZE,
    MAX_MANIFEST_COUNT,
    MAX_MANIFEST_DURATION_SECONDS,
    AssetManifest,
    AssetManifestError,
)

__all__ = [
    "BATCH_VERSION",
    "MAX_BATCH_ASSETS",
    "AssetBatch",
    "AssetBatchError",
    "BatchFinding",
    "validate_asset_batch",
]

BATCH_VERSION: Final = 1
MAX_BATCH_ASSETS: Final = 10_000

_BATCH_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_PATH_OR_URL_RE: Final = re.compile(r"://|(^|[A-Za-z]):[\\/]|[\\/]|~")
_DURATION_ABSOLUTE_TOLERANCE: Final = 1e-9

_COUNT_AGGREGATES: Final = ("asset_count", "total_bytes", "total_pages", "total_frames")
_AGGREGATE_FIELDS: Final = (*_COUNT_AGGREGATES, "total_duration_seconds")
_AGGREGATE_LIMITS: Final = {
    "total_bytes": MAX_MANIFEST_BYTE_SIZE,
    "total_pages": MAX_MANIFEST_COUNT,
    "total_frames": MAX_MANIFEST_COUNT,
    "total_duration_seconds": MAX_MANIFEST_DURATION_SECONDS,
}
_REQUIRED_FIELDS: Final = frozenset({"batch_id", "assets"})
_ALLOWED_FIELDS: Final = frozenset(
    {"version", "batch_id", "assets", *_AGGREGATE_FIELDS}
)
_ORDERED_FIELDS: Final = ("version", "batch_id", *_AGGREGATE_FIELDS, "assets")

_REASON_CODES: Final = frozenset(
    {
        "aggregate_invalid",
        "aggregate_mismatch",
        "aggregate_overflow",
        "batch_too_large",
        "duplicate_asset_id",
        "duplicate_sha256",
        "empty_batch",
        "invalid_asset",
        "invalid_assets",
        "invalid_batch",
        "invalid_batch_id",
        "invalid_version",
        "missing_required",
        "order_not_canonical",
        "unknown_field",
    }
)


class AssetBatchError(ValueError):
    """Raised when a privacy-safe asset batch fails validation."""


@dataclass(frozen=True, slots=True)
class BatchFinding:
    """A deterministic, value-free finding from asset batch validation.

    ``field_name`` is restricted to the fixed batch schema and ``position`` is
    the zero-based index of the offending asset, so a finding can never carry
    an unknown key, a path, or manifest content.
    """

    reason_code: str
    field_name: str | None = None
    position: int | None = None

    def __post_init__(self) -> None:
        if type(self.reason_code) is not str or self.reason_code not in _REASON_CODES:
            raise AssetBatchError("finding reason_code is unsupported")
        if self.field_name is not None and (
            type(self.field_name) is not str or self.field_name not in _ALLOWED_FIELDS
        ):
            raise AssetBatchError("finding field_name is unsupported")
        if self.position is not None and (
            type(self.position) is not int or self.position < 0
        ):
            raise AssetBatchError("finding position must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        """Return the finding as a deterministic metadata-only dictionary."""
        data: dict[str, Any] = {"reason_code": self.reason_code}
        if self.field_name is not None:
            data["field_name"] = self.field_name
        if self.position is not None:
            data["position"] = self.position
        return data


@dataclass(frozen=True, slots=True)
class AssetBatch:
    """Versioned, privacy-safe batch of multimodal asset manifests.

    Assets are stored in canonical order, strictly ascending by ``asset_id``,
    so equal batches always serialize identically. Aggregate totals are derived
    from the manifests rather than stored, which keeps the Python object free
    of inconsistent state; declared totals in serialized payloads are checked
    against the derived values when a batch is loaded.
    """

    batch_id: str
    assets: tuple[AssetManifest, ...]
    version: int = BATCH_VERSION

    def __post_init__(self) -> None:
        findings = _structural_findings(self.version, self.batch_id, self.assets)
        if findings:
            raise AssetBatchError(_describe(findings))

    @property
    def asset_count(self) -> int:
        """Number of assets in the batch."""
        return len(self.assets)

    @property
    def total_bytes(self) -> int:
        """Sum of manifest byte sizes."""
        return _aggregates(self.assets)["total_bytes"]

    @property
    def total_pages(self) -> int:
        """Sum of declared page counts across page-bearing assets."""
        return _aggregates(self.assets)["total_pages"]

    @property
    def total_frames(self) -> int:
        """Sum of declared frame counts across frame-bearing assets."""
        return _aggregates(self.assets)["total_frames"]

    @property
    def total_duration_seconds(self) -> float:
        """Sum of declared durations across duration-bearing assets."""
        return _aggregates(self.assets)["total_duration_seconds"]

    @classmethod
    def build(cls, batch_id: str, manifests: Iterable[AssetManifest]) -> "AssetBatch":
        """Build a canonically ordered batch from validated manifests."""
        try:
            items = tuple(manifests)
        except Exception:
            raise AssetBatchError("batch assets could not be read") from None
        if any(not isinstance(item, AssetManifest) for item in items):
            raise AssetBatchError("batch assets must be AssetManifest records")
        ordered = tuple(sorted(items, key=lambda manifest: manifest.asset_id))
        return cls(batch_id=batch_id, assets=ordered)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        max_assets: int = MAX_BATCH_ASSETS,
        allow_empty: bool = False,
    ) -> "AssetBatch":
        """Build and validate a batch from a strict mapping.

        Declared aggregate fields are optional in the input; when present they
        must match the totals derived from the manifests.
        """
        _validate_policy(max_assets, allow_empty)
        findings, assets = _validate_mapping(data)
        if assets is not None:
            findings.extend(_policy_findings(len(assets), max_assets, allow_empty))
        if findings or assets is None:
            raise AssetBatchError(_describe(findings))
        fields = dict(data)
        return cls(
            version=fields.get("version", BATCH_VERSION),
            batch_id=fields["batch_id"],
            assets=assets,
        )

    @classmethod
    def from_json(
        cls,
        payload: str | bytes | bytearray,
        *,
        max_assets: int = MAX_BATCH_ASSETS,
        allow_empty: bool = False,
    ) -> "AssetBatch":
        """Build and validate a batch from a JSON object."""
        try:
            data = json.loads(payload, object_pairs_hook=_strict_json_object)
        except AssetBatchError:
            raise
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError, ValueError):
            raise AssetBatchError("batch JSON is malformed") from None
        return cls.from_dict(data, max_assets=max_assets, allow_empty=allow_empty)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary with derived aggregate totals."""
        values: dict[str, Any] = {
            "version": self.version,
            "batch_id": self.batch_id,
            "asset_count": self.asset_count,
            **_aggregates(self.assets),
            "assets": [manifest.to_dict() for manifest in self.assets],
        }
        return {field_name: values[field_name] for field_name in _ORDERED_FIELDS}

    def to_json(self) -> str:
        """Return stable compact JSON with sorted keys and canonical asset order."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def validate_asset_batch(
    batch: Mapping[str, Any] | AssetBatch,
    *,
    max_assets: int = MAX_BATCH_ASSETS,
    allow_empty: bool = False,
) -> list[BatchFinding]:
    """Validate a batch payload deterministically without opening any asset.

    Structural findings cover unknown or missing fields, invalid identifiers,
    invalid manifests, duplicate identifiers and digests, non-canonical order,
    aggregate overflow, and declared totals that disagree with the manifests.
    Policy findings cover empty batches and batches above ``max_assets``. The
    result is sorted and de-duplicated so repeated runs are byte-stable.
    """
    _validate_policy(max_assets, allow_empty)
    if isinstance(batch, AssetBatch):
        findings: list[BatchFinding] = []
        assets: tuple[AssetManifest, ...] | None = batch.assets
    else:
        findings, assets = _validate_mapping(batch)
    if assets is not None:
        findings.extend(_policy_findings(len(assets), max_assets, allow_empty))
    return sorted(set(findings), key=_finding_sort_key)


def _validate_policy(max_assets: Any, allow_empty: Any) -> None:
    if type(max_assets) is not int or not 1 <= max_assets <= MAX_BATCH_ASSETS:
        raise AssetBatchError("max_assets must be a positive integer within the limit")
    if type(allow_empty) is not bool:
        raise AssetBatchError("allow_empty must be a boolean")


def _policy_findings(
    count: int, max_assets: int, allow_empty: bool
) -> list[BatchFinding]:
    findings: list[BatchFinding] = []
    if count == 0 and not allow_empty:
        findings.append(BatchFinding("empty_batch", field_name="assets"))
    if count > max_assets:
        findings.append(BatchFinding("batch_too_large", field_name="assets"))
    return findings


def _validate_mapping(
    data: Any,
) -> tuple[list[BatchFinding], tuple[AssetManifest, ...] | None]:
    if not isinstance(data, Mapping):
        return [BatchFinding("invalid_batch")], None
    try:
        fields = dict(data)
    except Exception:
        return [BatchFinding("invalid_batch")], None

    findings: list[BatchFinding] = []
    if set(fields) - _ALLOWED_FIELDS:
        findings.append(BatchFinding("unknown_field"))
    for field_name in sorted(_REQUIRED_FIELDS - set(fields)):
        findings.append(BatchFinding("missing_required", field_name=field_name))
    if findings:
        return findings, None

    version = fields.get("version", BATCH_VERSION)
    assets, asset_findings = _parse_assets(fields["assets"])
    findings.extend(asset_findings)
    if assets is None:
        findings.extend(_scalar_findings(version, fields["batch_id"]))
        return findings, None

    findings.extend(_structural_findings(version, fields["batch_id"], assets))
    findings.extend(_aggregate_findings(fields, assets))
    return findings, assets


def _parse_assets(
    raw: Any,
) -> tuple[tuple[AssetManifest, ...] | None, list[BatchFinding]]:
    if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
        return None, [BatchFinding("invalid_assets", field_name="assets")]
    manifests: list[AssetManifest] = []
    findings: list[BatchFinding] = []
    for position, item in enumerate(raw):
        if isinstance(item, AssetManifest):
            manifests.append(item)
            continue
        try:
            manifests.append(AssetManifest.from_dict(item))
        except AssetManifestError:
            findings.append(BatchFinding("invalid_asset", position=position))
    if findings:
        return None, findings
    return tuple(manifests), []


def _scalar_findings(version: Any, batch_id: Any) -> list[BatchFinding]:
    findings: list[BatchFinding] = []
    if type(version) is not int or version != BATCH_VERSION:
        findings.append(BatchFinding("invalid_version", field_name="version"))
    if (
        not isinstance(batch_id, str)
        or _PATH_OR_URL_RE.search(batch_id)
        or _BATCH_ID_RE.fullmatch(batch_id) is None
    ):
        findings.append(BatchFinding("invalid_batch_id", field_name="batch_id"))
    return findings


def _structural_findings(
    version: Any, batch_id: Any, assets: Any
) -> list[BatchFinding]:
    findings = _scalar_findings(version, batch_id)
    if type(assets) is not tuple:
        findings.append(BatchFinding("invalid_assets", field_name="assets"))
        return findings
    invalid = [
        BatchFinding("invalid_asset", position=position)
        for position, item in enumerate(assets)
        if not isinstance(item, AssetManifest)
    ]
    if invalid:
        return findings + invalid
    if len(assets) > MAX_BATCH_ASSETS:
        findings.append(BatchFinding("batch_too_large", field_name="assets"))

    seen_ids: set[str] = set()
    seen_digests: set[str] = set()
    for position, manifest in enumerate(assets):
        if manifest.asset_id in seen_ids:
            findings.append(BatchFinding("duplicate_asset_id", position=position))
        if position > 0 and manifest.asset_id < assets[position - 1].asset_id:
            findings.append(BatchFinding("order_not_canonical", position=position))
        if manifest.sha256 in seen_digests:
            findings.append(BatchFinding("duplicate_sha256", position=position))
        seen_ids.add(manifest.asset_id)
        seen_digests.add(manifest.sha256)

    totals = _aggregates(assets)
    for field_name, limit in _AGGREGATE_LIMITS.items():
        if totals[field_name] > limit:
            findings.append(BatchFinding("aggregate_overflow", field_name=field_name))
    return findings


def _aggregate_findings(
    fields: Mapping[str, Any], assets: tuple[AssetManifest, ...]
) -> list[BatchFinding]:
    findings: list[BatchFinding] = []
    computed: dict[str, int | float] = {
        "asset_count": len(assets),
        **_aggregates(assets),
    }
    for field_name in _AGGREGATE_FIELDS:
        if field_name not in fields:
            continue
        declared = fields[field_name]
        if field_name in _COUNT_AGGREGATES:
            if type(declared) is not int:
                findings.append(
                    BatchFinding("aggregate_invalid", field_name=field_name)
                )
            elif declared != computed[field_name]:
                findings.append(
                    BatchFinding("aggregate_mismatch", field_name=field_name)
                )
            continue
        if type(declared) not in (int, float) or not math.isfinite(declared):
            findings.append(BatchFinding("aggregate_invalid", field_name=field_name))
        elif not math.isclose(
            declared,
            computed[field_name],
            rel_tol=0.0,
            abs_tol=_DURATION_ABSOLUTE_TOLERANCE,
        ):
            findings.append(BatchFinding("aggregate_mismatch", field_name=field_name))
    return findings


def _aggregates(assets: tuple[AssetManifest, ...]) -> dict[str, Any]:
    return {
        "total_bytes": sum(manifest.byte_size for manifest in assets),
        "total_pages": sum(
            manifest.pages for manifest in assets if manifest.pages is not None
        ),
        "total_frames": sum(
            manifest.frames for manifest in assets if manifest.frames is not None
        ),
        "total_duration_seconds": math.fsum(
            manifest.duration_seconds
            for manifest in assets
            if manifest.duration_seconds is not None
        ),
    }


def _finding_sort_key(finding: BatchFinding) -> tuple[str, str, int]:
    return (
        finding.reason_code,
        finding.field_name or "",
        -1 if finding.position is None else finding.position,
    )


def _describe(findings: Sequence[BatchFinding]) -> str:
    codes = sorted({finding.reason_code for finding in findings})
    return "asset batch is invalid: " + ", ".join(codes)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise AssetBatchError("batch contains duplicate fields")
    return fields
