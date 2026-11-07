"""Tests for normalized device-capability profiles and the reference adapters."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from openmed.core import device_capabilities as dc
from openmed.core import execution_plan as ep
from openmed.core import portable_bundle as pb

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "device_capabilities"
VECTORS = json.loads((FIXTURES / "vectors.json").read_text(encoding="utf-8"))
NORMALIZERS = {
    "android": dc.profile_from_android_probe,
    "apple": dc.profile_from_apple_probe,
    "web": dc.profile_from_web_probe,
}
PROFILE_SCHEMA = Draft202012Validator(
    ep.load_execution_plan_schema("device_capability_profile")
)
_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+){0,3}")


def _profile(name: str) -> dict[str, Any]:
    return json.loads(json.dumps(VECTORS["profiles"][name]["document"]))


@pytest.mark.parametrize(
    "item",
    VECTORS["normalization"],
    ids=[item["id"] for item in VECTORS["normalization"]],
)
def test_reference_adapter_matches_golden_profile(item: dict[str, Any]) -> None:
    profile = NORMALIZERS[item["kit"]](item["probe"])
    assert profile.to_dict() == item["expected"]


@pytest.mark.parametrize("name", sorted(VECTORS["profiles"]))
def test_profile_documents_round_trip_to_identical_bytes(name: str) -> None:
    entry = VECTORS["profiles"][name]
    profile = dc.DeviceCapabilityProfile.from_mapping(entry["document"])
    assert profile.to_dict() == entry["document"]
    assert profile.canonical_bytes() == pb.canonical_json_bytes(entry["document"])
    assert profile.digest() == entry["digest"]
    PROFILE_SCHEMA.validate(entry["document"])


def test_vocabulary_block_matches_reference_constants() -> None:
    vocabulary = VECTORS["vocabulary"]
    assert vocabulary["accelerators"] == sorted(dc.ACCELERATORS)
    assert vocabulary["features"] == sorted(dc.FEATURES)
    assert vocabulary["cpu_architectures"] == sorted(dc.CPU_ARCHITECTURES)
    assert vocabulary["model_tiers"] == list(dc.MODEL_TIERS)
    assert vocabulary["network_states"] == sorted(dc.NETWORK_STATES)
    assert vocabulary["provider_accelerators"] == dict(dc.PROVIDER_ACCELERATORS)
    assert vocabulary["platform_providers"] == {
        key: sorted(value) for key, value in dc.PLATFORM_PROVIDERS.items()
    }
    assert vocabulary["budget_ladder"] == {
        "max_bytes": dc.MAX_BUDGET_BYTES,
        "min_bytes": dc.MIN_BUDGET_BYTES,
    }
    assert set(dc.PLATFORM_PROVIDERS) == set(pb.PLATFORMS)
    assert set().union(*dc.PLATFORM_PROVIDERS.values()) == dc.PROVIDERS


def test_profile_schema_enums_match_constants() -> None:
    properties = ep.load_execution_plan_schema("device_capability_profile")[
        "properties"
    ]
    assert properties["providers"]["items"]["enum"] == sorted(dc.PROVIDERS)
    assert properties["features"]["items"]["enum"] == sorted(dc.FEATURES)
    assert properties["accelerators"]["items"]["enum"] == sorted(dc.ACCELERATORS)
    ladder = properties["memory_budget_bytes"]["oneOf"][1]["enum"]
    assert ladder == [0] + [1 << bit for bit in range(24, 41)]
    assert all(dc.budget_bucket(value) == value for value in ladder)


@pytest.mark.parametrize(
    ("measured", "bucket"),
    [
        (None, None),
        (0, 0),
        (dc.MIN_BUDGET_BYTES - 1, 0),
        (dc.MIN_BUDGET_BYTES, dc.MIN_BUDGET_BYTES),
        (150 * dc.MIB, 128 * dc.MIB),
        (900 * dc.MIB, 512 * dc.MIB),
        (3_000_000_000, 2 << 30),
        (dc.MAX_BUDGET_BYTES * 4, dc.MAX_BUDGET_BYTES),
    ],
)
def test_budget_bucket_rounds_down_to_the_ladder(
    measured: int | None, bucket: int | None
) -> None:
    assert dc.budget_bucket(measured) == bucket


@pytest.mark.parametrize("value", [-1, 1.5, True, "1024"])
def test_budget_bucket_rejects_non_integers(value: Any) -> None:
    with pytest.raises(dc.DeviceCapabilityError, match="^probe_invalid$"):
        dc.budget_bucket(value)


def test_coarse_os_version_keeps_major_and_minor() -> None:
    assert dc.coarse_os_version("17.4.1") == "17.4"
    assert dc.coarse_os_version("34") == "34"
    assert dc.coarse_os_version(None) is None
    with pytest.raises(dc.DeviceCapabilityError):
        dc.coarse_os_version("17.4-beta")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc.update(device_id="SENTINEL-SERIAL-0042"),
        lambda doc: doc.pop("network"),
        lambda doc: doc.update(schema="openmed.other"),
        lambda doc: doc.update(schema_version=True),
        lambda doc: doc.update(accelerators=["cpu"]),
        lambda doc: doc.update(providers=["cpu", "webgpu"]),
        lambda doc: doc.update(runtimes=["coreml"]),
        lambda doc: doc.update(memory_budget_bytes=3_000_000_000),
        lambda doc: doc.update(memory_budget_bytes=True),
        lambda doc: doc.update(os_version="17.4.1"),
        lambda doc: doc.update(sdk_name="SENTINEL-DEVICE-NAME"),
        lambda doc: doc.update(cpu_architecture="SENTINEL-GPU-ADAPTER"),
        lambda doc: doc.update(features=["SENTINEL-VENDOR-ID"]),
        lambda doc: doc.update(operator_sets={"onnx": 0}),
        lambda doc: doc.update(operator_sets={"SENTINEL": 1}),
        lambda doc: doc.update(max_model_tier="huge"),
        lambda doc: doc.update(max_sequence_length=0),
        lambda doc: doc.update(network="SENTINEL-USER-AGENT"),
    ],
)
def test_profile_documents_are_closed(mutate: Any) -> None:
    document = _profile("android_flagship")
    mutate(document)
    with pytest.raises(dc.DeviceCapabilityError, match="^profile_invalid$"):
        dc.DeviceCapabilityProfile.from_mapping(document)


def test_collections_are_sets_with_one_canonical_encoding() -> None:
    document = _profile("web_full")
    shuffled = dict(document)
    shuffled["providers"] = list(reversed(document["providers"])) * 2
    shuffled["precisions"] = list(reversed(document["precisions"]))
    left = dc.DeviceCapabilityProfile.from_mapping(document)
    right = dc.DeviceCapabilityProfile.from_mapping(shuffled)
    assert left.canonical_bytes() == right.canonical_bytes()
    assert left.digest() == right.digest()


def test_network_state_is_excluded_from_the_digest() -> None:
    online = dc.DeviceCapabilityProfile.from_mapping(_profile("web_full"))
    offline = dc.DeviceCapabilityProfile.from_mapping(_profile("web_full_offline"))
    assert online.network != offline.network
    assert online.digest() == offline.digest()
    assert online.canonical_bytes() != offline.canonical_bytes()


@pytest.mark.parametrize(
    ("kit", "probe"),
    [
        ("web", {"wasm": "yes", "sdk_version": "4.0.0"}),
        ("web", {"wasm": True, "sdk_version": "4.0.0", "hardware_concurrency": -1}),
        ("web", {"wasm": True}),
        ("web", {"wasm": True, "sdk_version": "4.0.0", "network": "SENTINEL"}),
        (
            "android",
            {
                "sdk_int": 34,
                "abi": "arm64-v8a",
                "sdk_version": "4.0.0",
                "available_providers": ["CPU", "SENTINEL-GPU-ADAPTER"],
            },
        ),
        ("android", {"sdk_int": "34", "abi": "arm64-v8a", "sdk_version": "4.0.0"}),
        (
            "android",
            {
                "sdk_int": 34,
                "abi": "arm64-v8a",
                "sdk_version": "4.0.0",
                "device_tier": "SENTINEL",
            },
        ),
        (
            "apple",
            {
                "platform": "tvOS",
                "os_version": "17.0",
                "sdk_version": "4.0.0",
                "configuration": {},
            },
        ),
        (
            "apple",
            {
                "platform": "iOS",
                "os_version": "17.0",
                "sdk_version": "4.0.0",
                "configuration": {
                    "maximum_tier": "tiny",
                    "maximum_resident_memory_mb": 350,
                    "maximum_sequence_length": 512,
                    "requires_int8": "no",
                },
            },
        ),
    ],
)
def test_reference_adapters_reject_malformed_probes(
    kit: str, probe: dict[str, Any]
) -> None:
    with pytest.raises(dc.DeviceCapabilityError, match="^probe_invalid$"):
        NORMALIZERS[kit](probe)


def test_android_adapter_mirrors_accelerator_session_gating() -> None:
    base = {"abi": "arm64-v8a", "sdk_version": "4.0.0"}
    old = dc.profile_from_android_probe(
        {**base, "sdk_int": 26, "available_providers": ["NNAPI", "QNN"]}
    )
    new = dc.profile_from_android_probe(
        {**base, "sdk_int": 27, "available_providers": ["NNAPI"]}
    )
    assert old.providers == ("cpu", "qnn")
    assert new.providers == ("cpu", "nnapi")
    assert old.accelerators == ("cpu", "npu")


def test_apple_adapter_links_mlx_only_where_the_package_does() -> None:
    watch = next(
        item["probe"]
        for item in VECTORS["normalization"]
        if item["id"] == "apple_watchos"
    )
    assert watch["mlx_available"] is True
    profile = dc.profile_from_apple_probe(watch)
    assert "metal" not in profile.providers and "mlx" not in profile.runtimes
    assert profile.precisions == ("int8",)
    assert profile.memory_budget_bytes == 128 * dc.MIB
    assert profile.max_model_tier == "nano"


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [*value.keys(), *(s for item in value.values() for s in _strings(item))]
    if isinstance(value, list):
        return [s for item in value for s in _strings(item)]
    return []


def test_profiles_carry_no_identity_values() -> None:
    sentinels = VECTORS["identity_sentinels"]
    probes = [item["probe"] for item in VECTORS["normalization"]]
    assert all(any(s in json.dumps(probe) for s in sentinels) for probe in probes)
    vocabulary = (
        set(dc.PROVIDERS)
        | set(dc.ACCELERATORS)
        | set(dc.FEATURES)
        | set(dc.CPU_ARCHITECTURES)
        | set(dc.MODEL_TIERS)
        | set(dc.NETWORK_STATES)
        | set(pb.PLATFORMS)
        | set(pb.PRECISIONS)
        | set(pb.FORMAT_PLATFORMS)
        | set(dc.SDK_NAMES.values())
        | {"openmed-python", dc.DEVICE_CAPABILITIES_SCHEMA}
        | set(dc.DeviceCapabilityProfile.from_mapping(_profile("web_full")).to_dict())
    )
    for entry in VECTORS["profiles"].values():
        text = json.dumps(entry["document"])
        assert not any(sentinel in text for sentinel in sentinels)
        for value in _strings(entry["document"]):
            assert value in vocabulary or _VERSION.fullmatch(value), value
