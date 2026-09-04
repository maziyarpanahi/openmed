from __future__ import annotations

import re
from collections.abc import Iterator, Mapping

import pytest

from openmed.multimodal.cache_key import (
    CACHE_KEY_SCHEMA_VERSION,
    CATEGORICAL_PREPROCESSING_OPTIONS,
    NUMERIC_PREPROCESSING_OPTIONS,
    MultimodalCacheKeyError,
    build_multimodal_cache_key,
)
from openmed.multimodal.digest import AssetDigest

DIGEST = "a" * 64
BASE = {
    "asset_digest": DIGEST,
    "media_type": "image/png",
    "provider_version": "doctr-1.0",
    "model_version": "vision-2.1",
    "policy_version": "redact-v3",
}


def _key(**overrides: object) -> str:
    inputs = dict(BASE)
    inputs.update(overrides)
    return build_multimodal_cache_key(**inputs)  # type: ignore[arg-type]


def test_cache_key_is_versioned_content_free_sha256() -> None:
    key = _key()

    assert CACHE_KEY_SCHEMA_VERSION == 1
    assert re.fullmatch(r"openmed-multimodal-v1:[0-9a-f]{64}", key)
    for value in BASE.values():
        assert value not in key


def test_equivalent_option_mappings_produce_identical_keys() -> None:
    first = _key(
        preprocessing_options={
            "resize_mode": "fit",
            "target_width": 512,
            "target_height": 256,
            "clip_duration_seconds": 5,
        }
    )
    second = _key(
        preprocessing_options={
            "clip_duration_seconds": 5.0,
            "target_height": 256,
            "target_width": 512,
            "resize_mode": "fit",
        }
    )

    assert first == second


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("asset_digest", "b" * 64),
        ("media_type", "image/jpeg"),
        ("provider_version", "doctr-1.1"),
        ("model_version", "vision-2.2"),
        ("policy_version", "redact-v4"),
    ],
)
def test_every_declared_processing_version_changes_the_key(
    field: str,
    replacement: str,
) -> None:
    assert _key() != _key(**{field: replacement})


@pytest.mark.parametrize(
    "version",
    ["ocr-v2", "vision-2.1.0", "policy-3.0.0-rc1"],
)
def test_processing_versions_use_a_bounded_version_grammar(version: str) -> None:
    assert _key(provider_version=version)


def test_preprocessing_option_changes_invalidate_the_key() -> None:
    baseline = _key(preprocessing_options={"resize_mode": "fit"})

    assert baseline != _key(preprocessing_options={"resize_mode": "fill"})
    assert baseline != _key(
        preprocessing_options={"resize_mode": "fit", "target_width": 512}
    )
    assert _key() == _key(preprocessing_options={})


def test_asset_digest_result_and_hex_string_are_equivalent() -> None:
    digest = AssetDigest(sha256=DIGEST, byte_count=42)

    assert _key(asset_digest=digest) == _key(asset_digest=DIGEST)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("channel_mode", "mono"),
        ("color_mode", "rgb"),
        ("orientation_mode", "normalize"),
        ("resize_mode", "none"),
        ("clip_duration_seconds", 0.25),
        ("frame_stride", 2),
        ("render_dpi", 300),
        ("sample_rate_hz", 16_000),
        ("target_height", 512),
        ("target_width", 512),
    ],
)
def test_every_allowlisted_option_is_accepted(name: str, value: object) -> None:
    assert re.fullmatch(
        r"openmed-multimodal-v1:[0-9a-f]{64}",
        _key(preprocessing_options={name: value}),
    )


def test_exported_option_allowlists_are_read_only() -> None:
    with pytest.raises(TypeError):
        CATEGORICAL_PREPROCESSING_OPTIONS["unsafe"] = frozenset({"value"})  # type: ignore[index]
    with pytest.raises(TypeError):
        NUMERIC_PREPROCESSING_OPTIONS["unsafe"] = (int, 0.0, 1.0)  # type: ignore[index]


@pytest.mark.parametrize(
    "asset_digest",
    [
        "A" * 64,
        "a" * 63,
        "g" * 64,
        b"raw asset bytes",
        "/srv/charts/patient.dcm",
    ],
)
def test_invalid_digests_fail_without_echoing_values(asset_digest: object) -> None:
    with pytest.raises(MultimodalCacheKeyError) as error:
        _key(asset_digest=asset_digest)
    assert str(asset_digest) not in str(error.value)


@pytest.mark.parametrize(
    "media_type",
    [
        "Image/PNG",
        "text/plain",
        "image/png; patient=synthetic",
        "https://internal.invalid/image/png",
        "/srv/charts/patient.png",
        b"image/png",
    ],
)
def test_invalid_media_types_fail_without_echoing_values(media_type: object) -> None:
    with pytest.raises(MultimodalCacheKeyError) as error:
        _key(media_type=media_type)
    assert str(media_type) not in str(error.value)


@pytest.mark.parametrize(
    "unsafe",
    [
        "Patient Jane synthetic chart text",
        "/srv/charts/patient-123",
        "https://internal.invalid/model",
        "Bearer synthetic-credential",
        "prompt: extract diagnosis",
        "Patient_Jane_Doe",
        "MRN_123456",
        "sk-proj-secret-token",
        "patient-jane-1",
        b"raw-version-bytes",
        "",
        "v" * 65,
    ],
)
@pytest.mark.parametrize(
    "field",
    ["provider_version", "model_version", "policy_version"],
)
def test_unsafe_processing_versions_are_rejected_without_echo(
    field: str,
    unsafe: object,
) -> None:
    with pytest.raises(MultimodalCacheKeyError) as error:
        _key(**{field: unsafe})
    assert not isinstance(unsafe, str) or not unsafe or unsafe not in str(error.value)


@pytest.mark.parametrize(
    "options",
    [
        {"unknown_option": 1},
        {"resize_mode": "Patient Jane chart text"},
        {"resize_mode": "/srv/charts/patient.png"},
        {"resize_mode": "https://internal.invalid/prompt"},
        {"resize_mode": b"raw bytes"},
        {"resize_mode": {"nested": "payload"}},
        {"target_width": True},
        {"target_width": 0},
        {"target_width": 65_537},
        {"target_width": 512.0},
        {"clip_duration_seconds": float("nan")},
        {"clip_duration_seconds": float("inf")},
        {"clip_duration_seconds": 0},
        {"clip_duration_seconds": 86_401},
    ],
)
def test_unsafe_or_unbounded_options_fail_closed(
    options: Mapping[str, object],
) -> None:
    sentinel = next(iter(options.values()))

    with pytest.raises(MultimodalCacheKeyError) as error:
        _key(preprocessing_options=options)
    assert str(sentinel) not in str(error.value)


def test_options_mapping_is_not_mutated() -> None:
    options = {"resize_mode": "fit", "target_width": 512}
    original = dict(options)

    _key(preprocessing_options=options)

    assert options == original


def test_unreadable_option_mapping_does_not_leak_upstream_errors() -> None:
    sentinel = "SYNTHETIC_PATIENT_VALUE_FROM_MAPPING"

    class BrokenMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise RuntimeError(sentinel)

        def __iter__(self) -> Iterator[str]:
            raise RuntimeError(sentinel)

        def __len__(self) -> int:
            return 1

    with pytest.raises(MultimodalCacheKeyError) as error:
        _key(preprocessing_options=BrokenMapping())
    assert sentinel not in str(error.value)


def test_helper_rejects_non_mapping_options() -> None:
    sentinel = "SYNTHETIC_PATIENT_TEXT"

    with pytest.raises(MultimodalCacheKeyError) as error:
        _key(preprocessing_options=[sentinel])
    assert sentinel not in str(error.value)
