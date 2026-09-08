"""Pre-decode limit profiles for multimodal assets.

One admission policy, evaluated over a privacy-safe asset manifest before any
decoder, tensor, or model is loaded. A profile is an immutable set of inclusive
ceilings for bytes, pages, pixels per image/page/frame, total pixels, frames,
and audio duration. Evaluation reads manifest metadata only and returns
deterministic findings that carry a field name, a reason code, the ceiling,
and the observed number (or ``None`` when a rule could not be evaluated).

A rule that cannot be evaluated is reported, never assumed safe: missing
evidence must not become acceptance. The module never opens, decodes, or
downsamples media, and it never infers raster geometry that the manifest does
not carry.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Final

from .asset_manifest import (
    MAX_MANIFEST_BYTE_SIZE,
    MAX_MANIFEST_COUNT,
    MAX_MANIFEST_DURATION_SECONDS,
    AssetManifest,
)

__all__ = [
    "DESKTOP_V1",
    "LIMIT_FIELDS",
    "LIMIT_REASON_CODES",
    "MAX_PIXEL_PRODUCT",
    "MOBILE_V1",
    "AssetLimitError",
    "LimitFinding",
    "LimitProfile",
    "evaluate_asset_limits",
]

_MODALITIES: Final = frozenset({"image", "pdf", "dicom", "audio"})

# The documented, deterministic order of findings.
LIMIT_FIELDS: Final = (
    "byte_size",
    "pages",
    "pixels",
    "total_pixels",
    "frames",
    "duration_seconds",
)
_FIELD_ORDER: Final = {name: index for index, name in enumerate(LIMIT_FIELDS)}
_FIELD_SET: Final = frozenset(LIMIT_FIELDS)

LIMIT_REASON_CODES: Final = frozenset({"limit_exceeded", "insufficient_metadata"})
_REASON_ORDER: Final = {"limit_exceeded": 0, "insufficient_metadata": 1}

# Which rules a modality can express. A rule that is applicable but whose
# inputs are absent yields ``insufficient_metadata``; a rule that is not
# applicable yields nothing. PDF geometry is deliberately not representable
# under the manifest contract, so its pixel rules are always unevaluable.
_APPLICABLE_RULES: Final = {
    "image": frozenset({"byte_size", "pixels", "total_pixels"}),
    "pdf": frozenset({"byte_size", "pages", "pixels", "total_pixels"}),
    "dicom": frozenset({"byte_size", "pixels", "total_pixels", "frames"}),
    "audio": frozenset({"byte_size", "duration_seconds"}),
}

_INTEGER_INPUTS: Final = frozenset({"byte_size", "pages", "width", "height", "frames"})
_MANIFEST_INPUTS: Final = (
    "byte_size",
    "pages",
    "width",
    "height",
    "frames",
    "duration_seconds",
)

# Checked arithmetic bound: three bounded counts multiplied. Python integers do
# not wrap, so the check guards the contract rather than the machine; any
# product beyond it means an input escaped validation.
MAX_PIXEL_PRODUCT: Final = MAX_MANIFEST_COUNT**3

_PROFILE_NAME_MAX: Final = 32


class AssetLimitError(ValueError):
    """Raised when a limit profile, finding, or evaluation input is invalid."""


def _require_bounded_int(value: Any, name: str, maximum: int) -> None:
    if type(value) is not int or not 0 < value <= maximum:
        raise AssetLimitError(f"{name} must be a bounded positive integer")


def _require_positive_number(value: Any, name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise AssetLimitError(f"{name} must be a bounded positive finite number")


@dataclass(frozen=True, slots=True)
class LimitFinding:
    """One deterministic, privacy-safe finding from a limit evaluation.

    ``limit`` is the inclusive ceiling that applied. ``observed`` is the
    number the manifest yielded, or ``None`` when the rule could not be
    evaluated. Nothing else crosses this boundary: no identifiers, paths,
    digests, or content.
    """

    field_name: str
    reason_code: str
    limit: int | float
    observed: int | float | None

    def __post_init__(self) -> None:
        if type(self.field_name) is not str or self.field_name not in _FIELD_SET:
            raise AssetLimitError("finding field_name is unsupported")
        if (
            type(self.reason_code) is not str
            or self.reason_code not in LIMIT_REASON_CODES
        ):
            raise AssetLimitError("finding reason_code is unsupported")
        _require_positive_number(self.limit, "finding limit")
        if self.observed is not None:
            if type(self.observed) is bool or type(self.observed) not in (int, float):
                raise AssetLimitError("finding observed must be numeric or null")
            if not math.isfinite(self.observed) or self.observed < 0:
                raise AssetLimitError(
                    "finding observed must be a finite non-negative number"
                )
        if self.reason_code == "insufficient_metadata" and self.observed is not None:
            raise AssetLimitError("an unevaluable rule cannot carry an observed value")

    def to_dict(self) -> dict[str, Any]:
        """Return the finding as a JSON-ready mapping with a fixed key order."""
        return {
            "field_name": self.field_name,
            "reason_code": self.reason_code,
            "limit": self.limit,
            "observed": self.observed,
        }


@dataclass(frozen=True, slots=True)
class LimitProfile:
    """An immutable set of inclusive ceilings (equal passes, above fails)."""

    name: str
    version: str
    max_byte_size: int
    max_pages: int
    max_pixels: int
    max_total_pixels: int
    max_frames: int
    max_duration_seconds: int | float

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or len(self.name) > _PROFILE_NAME_MAX
            or not self.name.replace("-", "").replace("_", "").isalnum()
            or not self.name.isascii()
            or self.name != self.name.lower()
        ):
            raise AssetLimitError("profile name is invalid")
        if type(self.version) is not str or self.version != "1.0":
            raise AssetLimitError("profile version is unsupported")
        _require_bounded_int(
            self.max_byte_size, "max_byte_size", MAX_MANIFEST_BYTE_SIZE
        )
        _require_bounded_int(self.max_pages, "max_pages", MAX_MANIFEST_COUNT)
        _require_bounded_int(self.max_pixels, "max_pixels", MAX_PIXEL_PRODUCT)
        _require_bounded_int(
            self.max_total_pixels, "max_total_pixels", MAX_PIXEL_PRODUCT
        )
        _require_bounded_int(self.max_frames, "max_frames", MAX_MANIFEST_COUNT)
        _require_positive_number(self.max_duration_seconds, "max_duration_seconds")
        if self.max_duration_seconds > MAX_MANIFEST_DURATION_SECONDS:
            raise AssetLimitError("max_duration_seconds is out of range")
        if self.max_pixels > self.max_total_pixels:
            raise AssetLimitError("max_pixels cannot exceed max_total_pixels")

    def limit_for(self, field_name: str) -> int | float:
        """Return the ceiling that applies to a limit field."""
        if field_name not in _FIELD_SET:
            raise AssetLimitError("limit field_name is unsupported")
        return getattr(self, _LIMIT_ATTRIBUTES[field_name])

    def with_limits(self, **overrides: Any) -> LimitProfile:
        """Return a new, fully re-validated profile with some ceilings replaced."""
        unknown = set(overrides) - set(_LIMIT_ATTRIBUTES.values())
        if unknown:
            raise AssetLimitError("unknown limit override")
        return replace(self, **overrides)

    def to_dict(self) -> dict[str, Any]:
        """Return the profile as a JSON-ready mapping with a fixed key order."""
        return {
            "name": self.name,
            "version": self.version,
            "max_byte_size": self.max_byte_size,
            "max_pages": self.max_pages,
            "max_pixels": self.max_pixels,
            "max_total_pixels": self.max_total_pixels,
            "max_frames": self.max_frames,
            "max_duration_seconds": self.max_duration_seconds,
        }


_LIMIT_ATTRIBUTES: Final = {
    "byte_size": "max_byte_size",
    "pages": "max_pages",
    "pixels": "max_pixels",
    "total_pixels": "max_total_pixels",
    "frames": "max_frames",
    "duration_seconds": "max_duration_seconds",
}

# Initial admission-policy defaults. These are explicit policy choices, not
# measured device memory budgets, and not a guarantee that decoding or
# inference fits. The desktop page and pixel ceilings match the redacted-PDF
# renderer's long-standing defaults; the other values are unbenchmarked.
MOBILE_V1 = LimitProfile(
    name="mobile",
    version="1.0",
    max_byte_size=64 * 1024**2,
    max_pages=10,
    max_pixels=10_000_000,
    max_total_pixels=25_000_000,
    max_frames=128,
    max_duration_seconds=300,
)

DESKTOP_V1 = LimitProfile(
    name="desktop",
    version="1.0",
    max_byte_size=256 * 1024**2,
    max_pages=100,
    max_pixels=40_000_000,
    max_total_pixels=100_000_000,
    max_frames=1_024,
    max_duration_seconds=1_800,
)


def evaluate_asset_limits(
    profile: LimitProfile,
    manifest: Mapping[str, Any] | AssetManifest,
    modality: str,
) -> list[LimitFinding]:
    """Evaluate a manifest against a profile for one modality.

    Returns findings in the documented order (``LIMIT_FIELDS``); an empty list
    means every applicable rule was evaluated and passed. Rules the modality
    cannot express produce nothing. Applicable rules whose inputs are absent
    produce ``insufficient_metadata`` with ``observed`` ``None``; the checker
    never assumes the asset is safe.

    Mapping callers remain responsible for running the structural
    asset-manifest validator first; this function reads only the bounded
    numeric fields and rejects values outside the manifest contract.
    """
    if not isinstance(profile, LimitProfile):
        raise TypeError("profile must be a LimitProfile")
    if type(modality) is not str or modality not in _MODALITIES:
        raise AssetLimitError("modality is unsupported")
    inputs = _read_inputs(manifest)
    applicable = _APPLICABLE_RULES[modality]
    findings: list[LimitFinding] = []

    def check(field_name: str, observed: int | float | None) -> None:
        limit = profile.limit_for(field_name)
        if observed is None:
            findings.append(
                LimitFinding(field_name, "insufficient_metadata", limit, None)
            )
        elif observed > limit:
            findings.append(LimitFinding(field_name, "limit_exceeded", limit, observed))

    if "byte_size" in applicable:
        check("byte_size", inputs.get("byte_size"))
    if "pages" in applicable:
        check("pages", inputs.get("pages"))
    if "pixels" in applicable:
        check("pixels", _pixels(inputs, modality, total=False))
    if "total_pixels" in applicable:
        check("total_pixels", _pixels(inputs, modality, total=True))
    if "frames" in applicable:
        check("frames", inputs.get("frames"))
    if "duration_seconds" in applicable:
        check("duration_seconds", inputs.get("duration_seconds"))

    findings.sort(
        key=lambda f: (_FIELD_ORDER[f.field_name], _REASON_ORDER[f.reason_code])
    )
    return findings


def _pixels(inputs: Mapping[str, Any], modality: str, *, total: bool) -> int | None:
    # PDF geometry is not representable under the manifest contract (the PDF
    # profile marks width and height inapplicable), and raster pixels are never
    # inferred from a page count: both pixel rules stay unevaluable.
    if modality == "pdf":
        return None
    width, height = inputs.get("width"), inputs.get("height")
    if width is None or height is None:
        return None
    if modality == "dicom" and total:
        frames = inputs.get("frames")
        if frames is None:
            return None
        return _checked_product(width, height, frames)
    return _checked_product(width, height)


def _checked_product(*values: int) -> int:
    product = 1
    for value in values:
        if type(value) is not int or not 0 < value <= MAX_MANIFEST_COUNT:
            raise AssetLimitError("pixel arithmetic received an unbounded input")
        product *= value
        if product > MAX_PIXEL_PRODUCT:
            raise AssetLimitError("pixel arithmetic exceeded its bound")
    return product


def _read_inputs(manifest: Mapping[str, Any] | AssetManifest) -> dict[str, int | float]:
    if isinstance(manifest, AssetManifest):
        fields = manifest.to_dict()
    elif isinstance(manifest, Mapping):
        try:
            fields = dict(manifest)
        except Exception:
            raise AssetLimitError("manifest metadata could not be read") from None
    else:
        raise TypeError("manifest must be a mapping or AssetManifest")

    inputs: dict[str, int | float] = {}
    for field_name in _MANIFEST_INPUTS:
        if field_name not in fields or fields[field_name] is None:
            continue
        value = fields[field_name]
        if field_name in _INTEGER_INPUTS:
            bound = (
                MAX_MANIFEST_BYTE_SIZE
                if field_name == "byte_size"
                else MAX_MANIFEST_COUNT
            )
            _require_bounded_int(value, field_name, bound)
        else:
            _require_positive_number(value, field_name)
            if value > MAX_MANIFEST_DURATION_SECONDS:
                raise AssetLimitError(f"{field_name} is out of range")
        inputs[field_name] = value
    return inputs
