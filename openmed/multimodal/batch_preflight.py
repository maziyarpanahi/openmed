"""Ordered batch composition of existing content-free multimodal preflight."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import islice
from types import MappingProxyType
from typing import BinaryIO

from .asset_batch import MAX_BATCH_ASSETS
from .asset_limits import DESKTOP_V1, MAX_PIXEL_PRODUCT, LimitProfile
from .asset_manifest import (
    MAX_MANIFEST_BYTE_SIZE,
    MAX_MANIFEST_COUNT,
    MAX_MANIFEST_DURATION_SECONDS,
    AssetManifest,
)
from .preflight import PreflightError, PreflightReport, PreflightStatus, preflight_asset

_FIELDS = ("bytes", "pages", "pixels", "frames", "duration", "rejections")
_MAXIMUM = {
    "bytes": MAX_MANIFEST_BYTE_SIZE,
    "pages": MAX_MANIFEST_COUNT,
    "pixels": MAX_PIXEL_PRODUCT,
    "frames": MAX_MANIFEST_COUNT,
    "duration": MAX_MANIFEST_DURATION_SECONDS,
    "rejections": MAX_BATCH_ASSETS,
}
_REASONS = frozenset(
    {
        "aggregate_limit_exceeded",
        "aggregate_overflow",
        "aggregate_insufficient_metadata",
        "duplicate_asset_id",
        "duplicate_content",
        "asset_abstained",
        "source_error",
    }
)
_POLICIES = frozenset({"continue", "hold", "fail_fast"})


@dataclass(frozen=True, slots=True)
class BatchPreflightAsset:
    """A validated manifest and caller-owned source; source never enters repr/report."""

    manifest: AssetManifest
    source: bytes | bytearray | memoryview | BinaryIO = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, AssetManifest):
            raise ValueError("batch preflight requires validated manifests")
        if not isinstance(self.source, (bytes, bytearray, memoryview)) and not callable(
            getattr(self.source, "read", None)
        ):
            raise ValueError("invalid batch preflight source")


@dataclass(frozen=True, slots=True)
class BatchPreflightLimits:
    """Caller-declared inclusive aggregate ceilings; no clinical policy is inferred."""

    bytes: int
    pages: int
    pixels: int
    frames: int
    duration: int | float
    rejections: int

    def __post_init__(self) -> None:
        for name in _FIELDS:
            value = getattr(self, name)
            if name == "duration":
                valid = (
                    type(value) in (int, float)
                    and math.isfinite(value)
                    and 0 <= value <= _MAXIMUM[name]
                )
            else:
                valid = type(value) is int and 0 <= value <= _MAXIMUM[name]
            if not valid:
                raise ValueError("invalid aggregate preflight ceiling")


@dataclass(frozen=True, slots=True)
class BatchPreflightFinding:
    """A controlled reason at an input position, with an optional resource field."""

    position: int
    reason: str
    resource: str | None = None

    def __post_init__(self) -> None:
        if type(self.position) is not int or not 0 <= self.position < MAX_BATCH_ASSETS:
            raise ValueError("invalid batch preflight position")
        if not isinstance(self.reason, str) or self.reason not in _REASONS:
            raise ValueError("invalid batch preflight reason")
        if self.reason.startswith("aggregate_"):
            if not isinstance(self.resource, str) or self.resource not in _FIELDS:
                raise ValueError("invalid batch preflight resource")
        elif self.resource is not None:
            raise ValueError("unexpected batch preflight resource")

    def to_dict(self) -> dict[str, object]:
        """Return only position, controlled reason and controlled resource."""
        return {
            "position": self.position,
            "reason": self.reason,
            "resource": self.resource,
        }


@dataclass(frozen=True, slots=True)
class BatchAssetResult:
    """One ordered asset result, preserving a real shared preflight report."""

    position: int
    status: str
    preflight: PreflightReport | None = None

    def __post_init__(self) -> None:
        if type(self.position) is not int or not 0 <= self.position < MAX_BATCH_ASSETS:
            raise ValueError("invalid batch preflight result position")
        if not isinstance(self.status, str) or self.status not in {
            "accept",
            "abstain",
            "error",
            "not_evaluated",
        }:
            raise ValueError("invalid batch preflight result status")
        if self.status in {"accept", "abstain"}:
            if (
                not isinstance(self.preflight, PreflightReport)
                or self.preflight.status.value != self.status
            ):
                raise ValueError("inconsistent batch asset preflight result")
        elif self.preflight is not None:
            raise ValueError("unexpected batch asset preflight report")

    def to_dict(self) -> dict[str, object]:
        """Return position and validated metadata only; never the source."""
        return {
            "position": self.position,
            "status": self.status,
            "preflight": None if self.preflight is None else self.preflight.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BatchPreflightReport:
    """A bounded ordered report; any finding holds the batch from acceptance."""

    policy: str
    limits: BatchPreflightLimits
    assets: tuple[BatchAssetResult, ...]
    findings: tuple[BatchPreflightFinding, ...]
    totals: Mapping[str, int | float | None]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.policy, str)
            or self.policy not in _POLICIES
            or not isinstance(self.limits, BatchPreflightLimits)
        ):
            raise ValueError("invalid batch preflight policy")
        if (
            not isinstance(self.assets, (tuple, list))
            or not 1 <= len(self.assets) <= MAX_BATCH_ASSETS
            or any(
                not isinstance(item, BatchAssetResult) or item.position != index
                for index, item in enumerate(self.assets)
            )
        ):
            raise ValueError("invalid ordered batch preflight results")
        if (
            not isinstance(self.findings, (tuple, list))
            or len(self.findings) > MAX_BATCH_ASSETS * 16
            or any(
                not isinstance(item, BatchPreflightFinding)
                or item.position >= len(self.assets)
                for item in self.findings
            )
        ):
            raise ValueError("invalid batch preflight findings")
        if not isinstance(self.totals, Mapping) or set(self.totals) != set(_FIELDS):
            raise ValueError("invalid batch preflight totals")
        totals = dict(self.totals)
        for name, value in totals.items():
            if value is not None and (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= _MAXIMUM[name]
                or (name != "duration" and type(value) is not int)
            ):
                raise ValueError("invalid batch preflight total")
        if not self.findings and any(
            value is None or value > getattr(self.limits, name)
            for name, value in totals.items()
        ):
            raise ValueError("inconsistent accepted batch aggregate totals")
        if not self.findings and any(item.status != "accept" for item in self.assets):
            raise ValueError("inconsistent accepted batch preflight")
        object.__setattr__(self, "assets", tuple(self.assets))
        object.__setattr__(
            self,
            "findings",
            tuple(
                sorted(
                    set(self.findings),
                    key=lambda item: (item.position, item.reason, item.resource or ""),
                )
            ),
        )
        object.__setattr__(self, "totals", MappingProxyType(totals))

    @property
    def outcome(self) -> str:
        """Return accept, hold, or fail_fast; continue processes all, never overrides findings."""
        return (
            "accept"
            if not self.findings
            else "fail_fast"
            if self.policy == "fail_fast"
            else "hold"
        )

    def to_dict(self) -> dict[str, object]:
        """Export metadata-only batch status and ordered single-asset results."""
        return {
            "schema_version": 1,
            "policy": self.policy,
            "limits": {name: getattr(self.limits, name) for name in _FIELDS},
            "outcome": self.outcome,
            "totals": dict(self.totals),
            "assets": [item.to_dict() for item in self.assets],
            "findings": [item.to_dict() for item in self.findings],
        }

    def to_json(self) -> str:
        """Serialize byte-stable metadata; source bytes and source errors are excluded."""
        return (
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        )


def preflight_asset_batch(
    assets: Iterable[BatchPreflightAsset],
    *,
    limits: BatchPreflightLimits,
    policy: str = "continue",
    asset_limit_profile: LimitProfile = DESKTOP_V1,
) -> BatchPreflightReport:
    """Preflight an ordered batch using actual shared single-asset contracts.

    Args:
        assets: Bounded ordered validated manifests with caller-owned local sources.
        limits: Explicit aggregate byte/page/pixel/frame/duration/rejection ceilings.
        policy: Continue examines all byte-admitted sources; hold and fail_fast stop source
            reads after the first rejected position, retaining later metadata.
        asset_limit_profile: Existing per-asset pre-decode resource policy.

    Returns:
        One result per input position. Aggregate and duplicate metadata checks
        continue for every asset even when source reads stop. Unknown applicable
        geometry or duration never contributes an assumed zero. Any finding
        prevents batch acceptance; this does not decode or judge clinical data.
    """
    if (
        not isinstance(limits, BatchPreflightLimits)
        or not isinstance(asset_limit_profile, LimitProfile)
        or not isinstance(policy, str)
        or policy not in _POLICIES
    ):
        raise ValueError("invalid batch preflight configuration")
    if isinstance(assets, (str, bytes)):
        raise ValueError("invalid batch preflight asset collection")
    try:
        assets = tuple(islice(assets, MAX_BATCH_ASSETS + 1))
    except Exception:
        raise ValueError("invalid batch preflight asset collection") from None
    if not 1 <= len(assets) <= MAX_BATCH_ASSETS or any(
        not isinstance(item, BatchPreflightAsset) for item in assets
    ):
        raise ValueError("invalid bounded batch preflight assets")
    totals: dict[str, int | float | None] = dict.fromkeys(_FIELDS, 0)
    findings, results = [], []
    ids, digests = set(), set()
    stopped = False
    durations = []
    for index, asset in enumerate(assets):
        start = len(findings)
        manifest = asset.manifest
        if manifest.asset_id in ids:
            findings.append(BatchPreflightFinding(index, "duplicate_asset_id"))
        if manifest.sha256 in digests:
            findings.append(BatchPreflightFinding(index, "duplicate_content"))
        ids.add(manifest.asset_id)
        digests.add(manifest.sha256)
        values: dict[str, int | float | None] = {
            "bytes": manifest.byte_size,
            "pages": manifest.pages or 0,
            "frames": manifest.frames or 0,
            "pixels": 0,
            "duration": 0,
        }
        if (
            manifest.media_type.startswith("image/")
            or manifest.media_type == "application/dicom"
        ):
            values["pixels"] = (
                None
                if manifest.width is None
                or manifest.height is None
                or (
                    manifest.media_type == "application/dicom"
                    and manifest.frames is None
                )
                else manifest.width * manifest.height * (manifest.frames or 1)
            )
        elif manifest.media_type == "application/pdf":
            values["pixels"] = None
            values["pages"] = manifest.pages
        elif manifest.media_type.startswith("audio/"):
            values["duration"] = manifest.duration_seconds
        for name, value in values.items():
            if value is None:
                totals[name] = None
                findings.append(
                    BatchPreflightFinding(
                        index, "aggregate_insufficient_metadata", name
                    )
                )
            elif totals[name] is not None:
                if name == "duration":
                    durations.append(value)
                    total = math.fsum(durations)
                else:
                    total = totals[name] + value
                if total > _MAXIMUM[name]:
                    totals[name] = None
                    findings.append(
                        BatchPreflightFinding(index, "aggregate_overflow", name)
                    )
                else:
                    totals[name] = total
                    if total > getattr(limits, name):
                        findings.append(
                            BatchPreflightFinding(
                                index, "aggregate_limit_exceeded", name
                            )
                        )
        byte_admission_failed = (
            totals["bytes"] is None or totals["bytes"] > limits.bytes
        )
        if (
            stopped
            or byte_admission_failed
            or (policy != "continue" and len(findings) > start)
        ):
            result = BatchAssetResult(index, "not_evaluated")
        else:
            try:
                report = preflight_asset(
                    manifest, asset.source, limit_profile=asset_limit_profile
                )
                result = BatchAssetResult(index, report.status.value, report)
                if report.status is PreflightStatus.ABSTAIN:
                    findings.append(BatchPreflightFinding(index, "asset_abstained"))
            except PreflightError:
                result = BatchAssetResult(index, "error")
                findings.append(BatchPreflightFinding(index, "source_error"))
        rejected = len(findings) > start
        if rejected:
            totals["rejections"] += 1
            if totals["rejections"] > limits.rejections:
                findings.append(
                    BatchPreflightFinding(
                        index, "aggregate_limit_exceeded", "rejections"
                    )
                )
            if policy != "continue":
                stopped = True
        results.append(result)
    return BatchPreflightReport(policy, limits, tuple(results), tuple(findings), totals)
