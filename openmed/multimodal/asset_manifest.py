"""Privacy-safe asset manifests for multimodal preflight.

The manifest records only bounded, non-identifying facts that downstream image,
PDF, DICOM, and audio handlers can share before decoding an asset. It rejects
paths, URLs, free-text payload fields, and unknown keys so callers do not
accidentally persist PHI-bearing source metadata.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

MANIFEST_VERSION = 1

_ASSET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_MEDIA_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*$")
_PATH_OR_URL_RE = re.compile(r"://|(^|[A-Za-z]):[\\/]|[\\/]|~")

_ALLOWED_FIELDS = {
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

_ORDERED_FIELDS = (
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
)

_SUPPORTED_EXACT_MEDIA_TYPES = {
    "application/dicom",
    "application/pdf",
    "application/dicom+json",
}
_SUPPORTED_MEDIA_PREFIXES = ("image/", "audio/")


class AssetManifestError(ValueError):
    """Raised when a privacy-safe asset manifest fails validation."""


@dataclass(frozen=True)
class AssetManifest:
    """Versioned, privacy-safe description of a multimodal input asset."""

    asset_id: str
    media_type: str
    sha256: str
    byte_size: int
    version: int = MANIFEST_VERSION
    pages: int | None = None
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        _validate_version(self.version)
        _validate_opaque_string("asset_id", self.asset_id, _ASSET_ID_RE)
        _validate_media_type(self.media_type)
        _validate_opaque_string("sha256", self.sha256, _SHA256_RE)
        _validate_positive_int("byte_size", self.byte_size)
        for field_name in ("pages", "width", "height", "frames"):
            value = getattr(self, field_name)
            if value is not None:
                _validate_positive_int(field_name, value)
        if self.duration_seconds is not None:
            _validate_positive_number("duration_seconds", self.duration_seconds)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AssetManifest":
        """Build and validate a manifest from a strict mapping."""
        if not isinstance(data, Mapping):
            raise AssetManifestError("manifest must be a mapping")

        unknown = set(data) - _ALLOWED_FIELDS
        if unknown:
            raise AssetManifestError("manifest contains unknown fields")

        missing = {"asset_id", "media_type", "sha256", "byte_size"} - set(data)
        if missing:
            raise AssetManifestError("manifest is missing required fields")

        version = data.get("version", MANIFEST_VERSION)
        return cls(
            version=version,
            asset_id=data["asset_id"],
            media_type=data["media_type"],
            sha256=data["sha256"],
            byte_size=data["byte_size"],
            pages=data.get("pages"),
            width=data.get("width"),
            height=data.get("height"),
            frames=data.get("frames"),
            duration_seconds=data.get("duration_seconds"),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "AssetManifest":
        """Build and validate a manifest from a JSON object."""
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
            raise AssetManifestError("manifest JSON is malformed") from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary without unset optional fields."""
        data: dict[str, Any] = {}
        for field_name in _ORDERED_FIELDS:
            value = getattr(self, field_name)
            if value is not None:
                data[field_name] = value
        return data

    def to_json(self) -> str:
        """Return stable compact JSON with sorted keys."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _validate_version(value: Any) -> None:
    if value != MANIFEST_VERSION or isinstance(value, bool):
        raise AssetManifestError("version must match the supported manifest version")


def _validate_opaque_string(
    field_name: str, value: Any, pattern: re.Pattern[str]
) -> None:
    if not isinstance(value, str):
        raise AssetManifestError(f"{field_name} must be a string")
    if _PATH_OR_URL_RE.search(value):
        raise AssetManifestError(f"{field_name} must be an opaque value")
    if pattern.fullmatch(value) is None:
        raise AssetManifestError(f"{field_name} has an invalid format")


def _validate_media_type(value: Any) -> None:
    if not isinstance(value, str):
        raise AssetManifestError("media_type must be a string")
    media_type = value.lower()
    if value != media_type:
        raise AssetManifestError("media_type has an invalid format")
    if _MEDIA_TYPE_RE.fullmatch(media_type) is None:
        raise AssetManifestError("media_type has an invalid format")
    if media_type in _SUPPORTED_EXACT_MEDIA_TYPES:
        return
    if media_type.startswith(_SUPPORTED_MEDIA_PREFIXES):
        return
    raise AssetManifestError("media_type is unsupported")


def _validate_positive_int(field_name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AssetManifestError(f"{field_name} must be a positive integer")
    if value <= 0:
        raise AssetManifestError(f"{field_name} must be a positive integer")


def _validate_positive_number(field_name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AssetManifestError(f"{field_name} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise AssetManifestError(f"{field_name} must be a positive finite number")
