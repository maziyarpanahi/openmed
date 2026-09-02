"""Content-free cache keys for deterministic multimodal processing."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, cast

from .digest import AssetDigest

__all__ = [
    "CACHE_KEY_SCHEMA_VERSION",
    "CATEGORICAL_PREPROCESSING_OPTIONS",
    "NUMERIC_PREPROCESSING_OPTIONS",
    "MultimodalCacheKeyError",
    "build_multimodal_cache_key",
]

CACHE_KEY_SCHEMA_VERSION: Final = 1
_CACHE_KEY_PREFIX: Final = "openmed-multimodal-v1"

CATEGORICAL_PREPROCESSING_OPTIONS: Final[Mapping[str, frozenset[str]]] = (
    MappingProxyType(
        {
            "channel_mode": frozenset({"mono", "native", "stereo"}),
            "color_mode": frozenset({"grayscale", "native", "rgb", "rgba"}),
            "orientation_mode": frozenset({"normalize", "preserve"}),
            "resize_mode": frozenset({"fill", "fit", "none"}),
        }
    )
)

# Each numeric rule is ``(kind, minimum, maximum)``. Bounds keep canonical JSON
# finite and prevent accidental use of unbounded measurements as option values.
NUMERIC_PREPROCESSING_OPTIONS: Final[
    Mapping[str, tuple[type[int] | type[float], float, float]]
] = MappingProxyType(
    {
        "clip_duration_seconds": (float, 0.0, 86_400.0),
        "frame_stride": (int, 1.0, 1_000_000.0),
        "render_dpi": (int, 1.0, 9_600.0),
        "sample_rate_hz": (int, 1.0, 768_000.0),
        "target_height": (int, 1.0, 65_536.0),
        "target_width": (int, 1.0, 65_536.0),
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MEDIA_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
_SUPPORTED_EXACT_MEDIA_TYPES = frozenset(
    {"application/dicom", "application/dicom+json", "application/pdf"}
)
_SUPPORTED_MEDIA_PREFIXES = ("audio/", "image/")


class MultimodalCacheKeyError(ValueError):
    """Raised when cache-key metadata is unsafe or unsupported."""


def build_multimodal_cache_key(
    *,
    asset_digest: str | AssetDigest,
    media_type: str,
    provider_version: str,
    model_version: str,
    policy_version: str,
    preprocessing_options: Mapping[str, object] | None = None,
) -> str:
    """Return a versioned SHA-256 key from content-free processing metadata.

    This function performs no I/O and never hashes source bytes, paths, text,
    prompts, credentials, or arbitrary option values. Callers must compute the
    asset digest separately and declare every preprocessing choice explicitly.
    """

    digest = _asset_sha256(asset_digest)
    normalized_media_type = _media_type(media_type)
    versions = {
        "model_version": _version(model_version),
        "policy_version": _version(policy_version),
        "provider_version": _version(provider_version),
    }
    options = _preprocessing_options(preprocessing_options)
    payload = {
        "asset_sha256": digest,
        "media_type": normalized_media_type,
        "preprocessing_options": options,
        "schema_version": CACHE_KEY_SCHEMA_VERSION,
        **versions,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return f"{_CACHE_KEY_PREFIX}:{hashlib.sha256(encoded).hexdigest()}"


def _asset_sha256(value: object) -> str:
    if isinstance(value, AssetDigest):
        return value.sha256
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise MultimodalCacheKeyError("asset digest is invalid")
    return value


def _media_type(value: object) -> str:
    if type(value) is not str or _MEDIA_TYPE_RE.fullmatch(value) is None:
        raise MultimodalCacheKeyError("media type is invalid")
    if value in _SUPPORTED_EXACT_MEDIA_TYPES or value.startswith(
        _SUPPORTED_MEDIA_PREFIXES
    ):
        return value
    raise MultimodalCacheKeyError("media type is unsupported")


def _version(value: object) -> str:
    if type(value) is not str or _VERSION_RE.fullmatch(value) is None:
        raise MultimodalCacheKeyError("processing version is invalid")
    return value


def _preprocessing_options(
    value: Mapping[str, object] | None,
) -> dict[str, int | float | str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise MultimodalCacheKeyError("preprocessing options are invalid")
    try:
        options = dict(value)
    except Exception:
        raise MultimodalCacheKeyError("preprocessing options are invalid") from None

    known = CATEGORICAL_PREPROCESSING_OPTIONS.keys() | (
        NUMERIC_PREPROCESSING_OPTIONS.keys()
    )
    if any(type(name) is not str or name not in known for name in options):
        raise MultimodalCacheKeyError("preprocessing option is unsupported")

    normalized: dict[str, int | float | str] = {}
    for name in sorted(options):
        option = options[name]
        categories = CATEGORICAL_PREPROCESSING_OPTIONS.get(name)
        if categories is not None:
            if type(option) is not str or option not in categories:
                raise MultimodalCacheKeyError(
                    "categorical preprocessing option is invalid"
                )
            normalized[name] = option
            continue
        normalized[name] = _numeric_option(
            option,
            NUMERIC_PREPROCESSING_OPTIONS[name],
        )
    return normalized


def _numeric_option(
    value: object,
    rule: tuple[type[int] | type[float], float, float],
) -> int | float:
    kind, minimum, maximum = rule
    if kind is int:
        if type(value) is not int or not minimum <= value <= maximum:
            raise MultimodalCacheKeyError("numeric preprocessing option is invalid")
        return value
    if type(value) not in (int, float):
        raise MultimodalCacheKeyError("numeric preprocessing option is invalid")
    normalized = float(cast(int | float, value))
    if not math.isfinite(normalized) or not minimum < normalized <= maximum:
        raise MultimodalCacheKeyError("numeric preprocessing option is invalid")
    return normalized
