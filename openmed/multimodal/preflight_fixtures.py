"""Small synthetic negative fixtures for existing multimodal preflight contracts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .asset_batch import validate_asset_batch
from .preflight import preflight_asset

MAX_FIXTURE_BYTES = 65536
MAX_CASES = 64
MAX_HEADER_BYTES = 132
_CODE = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MEDIA = frozenset(
    {
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/tiff",
        "application/dicom",
        "audio/wav",
        "image/unknown",
    }
)
_CATEGORIES = frozenset(
    {
        "truncation",
        "type_mismatch",
        "invalid_dimensions",
        "oversized_counts",
        "bad_digest",
        "duplicate_asset",
        "unknown_media",
        "resource_limit",
    }
)
_EXPECTED = frozenset(
    {
        "manifest.malformed_manifest",
        "media_type.unknown",
        "media_type.mismatch",
        "metadata.missing_required",
        "metadata.inapplicable_present",
        "limits.insufficient_metadata",
        "limits.limit_exceeded",
        "digest.sha256_mismatch",
        "digest.byte_count_mismatch",
        "digest.not_evaluated",
        "batch.duplicate_asset_id",
        "batch.duplicate_sha256",
        "batch.order_not_canonical",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "version",
        "asset_id",
        "media_type",
        "sha256",
        "byte_size",
        "pages",
        "width",
        "height",
        "frames",
        "duration_seconds",
    }
)


@dataclass(frozen=True, slots=True)
class PreflightFixtureAsset:
    """Immutable controlled metadata plus at most 132 synthetic header bytes."""

    manifest: Mapping[str, object]
    header: bytes

    def __post_init__(self) -> None:
        if (
            not isinstance(self.manifest, Mapping)
            or set(self.manifest) - _MANIFEST_FIELDS
            or not {"asset_id", "media_type", "sha256", "byte_size"}
            <= set(self.manifest)
        ):
            raise ValueError("invalid preflight fixture manifest fields")
        manifest = dict(self.manifest)
        if not isinstance(manifest["asset_id"], str) or not _CODE.fullmatch(
            manifest["asset_id"]
        ):
            raise ValueError("invalid preflight fixture asset code")
        if (
            not isinstance(manifest["media_type"], str)
            or manifest["media_type"] not in _MEDIA
        ):
            raise ValueError("invalid preflight fixture media type")
        if not isinstance(manifest["sha256"], str) or not _DIGEST.fullmatch(
            manifest["sha256"]
        ):
            raise ValueError("invalid preflight fixture digest")
        for key in set(manifest) - {"asset_id", "media_type", "sha256"}:
            value = manifest[key]
            if type(value) is not int or not 0 <= value <= 2**63:
                raise ValueError("invalid preflight fixture numeric metadata")
        if (
            type(self.header) is not bytes
            or not 1 <= len(self.header) <= MAX_HEADER_BYTES
        ):
            raise ValueError("invalid preflight fixture header size")
        object.__setattr__(self, "manifest", MappingProxyType(manifest))


@dataclass(frozen=True, slots=True)
class PreflightFixtureCase:
    """A bounded named negative case with a closed expected-finding inventory."""

    case_id: str
    category: str
    assets: tuple[PreflightFixtureAsset, ...]
    expected_findings: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not _CODE.fullmatch(self.case_id):
            raise ValueError("invalid preflight fixture case code")
        if not isinstance(self.category, str) or self.category not in _CATEGORIES:
            raise ValueError("invalid preflight fixture category")
        if (
            not isinstance(self.assets, (tuple, list))
            or not 1 <= len(self.assets) <= 2
            or any(
                not isinstance(asset, PreflightFixtureAsset) for asset in self.assets
            )
        ):
            raise ValueError("invalid preflight fixture asset collection")
        if (self.category == "duplicate_asset") != (len(self.assets) == 2):
            raise ValueError("invalid preflight fixture batch category")
        if (
            not isinstance(self.expected_findings, (tuple, list))
            or not 1 <= len(self.expected_findings) <= len(_EXPECTED)
            or any(
                not isinstance(code, str) or code not in _EXPECTED
                for code in self.expected_findings
            )
        ):
            raise ValueError("invalid preflight fixture expected findings")
        if len(set(self.expected_findings)) != len(self.expected_findings):
            raise ValueError("duplicate preflight fixture expected finding")
        object.__setattr__(self, "assets", tuple(self.assets))
        object.__setattr__(
            self, "expected_findings", tuple(sorted(self.expected_findings))
        )

    def actual_findings(self) -> tuple[str, ...]:
        """Execute the existing preflight or batch validator and return codes only."""
        if self.category == "duplicate_asset":
            findings = validate_asset_batch(
                {
                    "batch_id": "fixture",
                    "assets": [dict(asset.manifest) for asset in self.assets],
                }
            )
            return tuple(
                sorted({"batch." + finding.reason_code for finding in findings})
            )
        asset = self.assets[0]
        report = preflight_asset(dict(asset.manifest), asset.header)
        return tuple(
            sorted(
                {
                    finding.check + "." + finding.reason_code
                    for finding in report.findings
                }
            )
        )


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate preflight fixture JSON field")
        result[key] = value
    return result


def load_preflight_fixtures(
    payload: bytes | str | None = None,
) -> tuple[PreflightFixtureCase, ...]:
    """Load the bundled synthetic header pack or a bounded strict JSON payload.

    Args:
        payload: Optional synthetic JSON bytes or text; omitted loads the package
            fixture. No filename or external transport is accepted.

    Returns:
        Immutable cases sorted by unique case ID. Loading validates schema and
        expectations; ``actual_findings`` separately exercises production checks.
    """
    if payload is None:
        with (Path(__file__).parent / "fixtures/malformed_asset_preflight.json").open(
            "rb"
        ) as source:
            payload = source.read(MAX_FIXTURE_BYTES + 1)
    if isinstance(payload, str):
        if len(payload) > MAX_FIXTURE_BYTES:
            raise ValueError("preflight fixture JSON size exceeded")
        payload = payload.encode("utf-8")
    if not isinstance(payload, bytes) or len(payload) > MAX_FIXTURE_BYTES:
        raise ValueError("invalid preflight fixture JSON size")
    try:
        data = json.loads(payload, object_pairs_hook=_object)
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("invalid preflight fixture JSON") from None
    if (
        not isinstance(data, dict)
        or set(data) != {"schema_version", "synthetic", "cases"}
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or data["synthetic"] is not True
    ):
        raise ValueError("unsupported preflight fixture schema")
    if not isinstance(data["cases"], list) or not 1 <= len(data["cases"]) <= MAX_CASES:
        raise ValueError("invalid preflight fixture case count")
    cases = []
    for item in data["cases"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"case_id", "category", "assets", "expected_findings"}
            or not isinstance(item["assets"], list)
            or not 1 <= len(item["assets"]) <= 2
        ):
            raise ValueError("invalid preflight fixture case fields")
        assets = []
        for asset in item["assets"]:
            if (
                not isinstance(asset, dict)
                or set(asset) != {"manifest", "header_hex"}
                or not isinstance(asset["header_hex"], str)
                or len(asset["header_hex"]) > MAX_HEADER_BYTES * 2
            ):
                raise ValueError("invalid preflight fixture asset fields")
            try:
                header = bytes.fromhex(asset["header_hex"])
            except ValueError:
                raise ValueError("invalid preflight fixture header encoding") from None
            assets.append(PreflightFixtureAsset(asset["manifest"], header))
        cases.append(
            PreflightFixtureCase(
                item["case_id"],
                item["category"],
                tuple(assets),
                item["expected_findings"],
            )
        )
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("duplicate preflight fixture case code")
    return tuple(sorted(cases, key=lambda case: case.case_id))
