"""Normalized device-capability profiles shared by every OpenMed kit.

A device-capability profile is the one record a resolver needs to choose an
artifact from a signed portable bundle (see
:mod:`openmed.core.portable_bundle`). Swift, Kotlin, and TypeScript kits keep
their native probes (Core ML and MLX availability, ONNX Runtime execution
providers, WebGPU and WebAssembly feature detection) and map the results into
this record. The Python reference normalizers below define that mapping, and
the shared golden vectors in ``tests/fixtures/device_capabilities/`` hold every
kit to it.

Profiles are deliberately coarse. They contain only controlled vocabulary,
dotted versions, and budgets rounded down to a power of two mebibytes, so a
profile cannot carry a device name, serial number, vendor string, user agent,
or exact memory size. Unknown members are rejected, which keeps the record
closed: a value that does not fit the vocabulary has nowhere to go.

Probing is outside this module. Nothing here opens a network connection,
loads a model, or runs inference.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .portable_bundle import (
    FORMAT_PLATFORMS,
    MAX_SAFE_INTEGER,
    PLATFORMS,
    PRECISIONS,
    canonical_json_bytes,
)

DEVICE_CAPABILITIES_SCHEMA = "openmed.device_capabilities"
DEVICE_CAPABILITIES_SCHEMA_VERSION = 1

PROVIDER_ACCELERATORS: Mapping[str, str] = {
    "coreml": "npu",
    "cpu": "cpu",
    "metal": "gpu",
    "nnapi": "npu",
    "qnn": "npu",
    "wasm": "cpu",
    "webgpu": "gpu",
}
"""Accelerator kind behind each normalized execution provider."""

PROVIDERS = frozenset(PROVIDER_ACCELERATORS)
ACCELERATORS = frozenset(PROVIDER_ACCELERATORS.values())

PLATFORM_PROVIDERS: Mapping[str, frozenset[str]] = {
    "android": frozenset({"cpu", "nnapi", "qnn"}),
    "ios": frozenset({"coreml", "cpu", "metal"}),
    "linux": frozenset({"cpu"}),
    "macos": frozenset({"coreml", "cpu", "metal"}),
    "visionos": frozenset({"coreml", "cpu"}),
    "watchos": frozenset({"coreml", "cpu"}),
    "web": frozenset({"wasm", "webgpu"}),
    "windows": frozenset({"cpu"}),
}
"""Execution providers a profile may report on each platform."""

CPU_ARCHITECTURES = frozenset({"arm64", "armv7", "unknown", "wasm32", "x86_64"})
FEATURES = frozenset(
    {
        "cpu_avx2",
        "cpu_fp16",
        "cpu_neon",
        "wasm_simd",
        "wasm_threads",
        "webgpu_shader_f16",
    }
)
MODEL_TIERS: tuple[str, ...] = ("nano", "tiny", "base")
"""Model tiers in increasing capacity order."""

NETWORK_STATES = frozenset({"offline", "online", "unknown"})

MIB = 1 << 20
MIN_BUDGET_BYTES = 16 * MIB
MAX_BUDGET_BYTES = 1 << 40
MAX_SEQUENCE_LENGTH = 1 << 20
MAX_OPERATOR_SET = 1000

SDK_NAMES: Mapping[str, str] = {
    "android": "openmedkit-android",
    "apple": "openmedkit-swift",
    "web": "openmedkit-web",
}
"""SDK name each kit adapter reports; these match bundle ``compatibility``."""

_TOKEN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_VERSION = re.compile(r"(?:0|[1-9][0-9]{0,8})(?:\.(?:0|[1-9][0-9]{0,8})){0,3}")
_COARSE_OS = re.compile(r"(?:0|[1-9][0-9]{0,8})(?:\.(?:0|[1-9][0-9]{0,8}))?")

_PROFILE_KEYS = frozenset(
    {
        "accelerators",
        "cpu_architecture",
        "features",
        "max_model_tier",
        "max_sequence_length",
        "memory_budget_bytes",
        "network",
        "operator_sets",
        "os_version",
        "platform",
        "precisions",
        "providers",
        "runtimes",
        "schema",
        "schema_version",
        "sdk_name",
        "sdk_version",
        "storage_budget_bytes",
    }
)


class DeviceCapabilityReason(str, Enum):
    """Value-free reason codes raised for malformed profiles or probes."""

    PROFILE_INVALID = "profile_invalid"
    PROBE_INVALID = "probe_invalid"


class DeviceCapabilityError(ValueError):
    """Raised for an invalid profile or probe; the message is the code only.

    Attributes:
        reason: The controlled :class:`DeviceCapabilityReason`.
    """

    def __init__(self, reason: DeviceCapabilityReason) -> None:
        """Store the controlled reason and use its value as the message."""

        self.reason = reason
        super().__init__(reason.value)


_INVALID = DeviceCapabilityReason.PROFILE_INVALID


def _is_uint(value: object, maximum: int = MAX_SAFE_INTEGER) -> bool:
    return type(value) is int and 0 <= value <= maximum


def budget_bucket(value: int | None) -> int | None:
    """Round a measured byte budget down to the coarse budget ladder.

    The ladder is ``0`` followed by every power of two from 16 MiB to 1 TiB.
    Rounding down is conservative: a profile never claims more memory or
    storage than was measured.

    Args:
        value: Measured budget in bytes, or ``None`` when unknown.

    Returns:
        The ladder value, or ``None`` when ``value`` is ``None``.

    Raises:
        DeviceCapabilityError: ``probe_invalid`` for a negative or non-integer
            budget.
    """

    if value is None:
        return None
    if not _is_uint(value):
        raise DeviceCapabilityError(DeviceCapabilityReason.PROBE_INVALID)
    if value < MIN_BUDGET_BYTES:
        return 0
    return 1 << (min(value, MAX_BUDGET_BYTES).bit_length() - 1)


def _is_budget(value: object) -> bool:
    if value is None:
        return True
    if not _is_uint(value):
        return False
    assert isinstance(value, int)
    return value == budget_bucket(value)


def coarse_os_version(value: str | None) -> str | None:
    """Keep only the major and minor components of a dotted OS version.

    Args:
        value: Dotted numeric OS version such as ``17.4.1``, or ``None``.

    Returns:
        ``MAJOR`` or ``MAJOR.MINOR``, or ``None``.

    Raises:
        DeviceCapabilityError: ``probe_invalid`` when ``value`` is not a
            dotted numeric version.
    """

    if value is None:
        return None
    if not isinstance(value, str) or _VERSION.fullmatch(value) is None:
        raise DeviceCapabilityError(DeviceCapabilityReason.PROBE_INVALID)
    return ".".join(value.split(".")[:2])


def model_tier_rank(tier: str) -> int:
    """Return the capacity rank of a model tier (``nano`` is 0).

    Args:
        tier: One of :data:`MODEL_TIERS`.

    Returns:
        The tier's index in :data:`MODEL_TIERS`.
    """

    return MODEL_TIERS.index(tier)


def _string_set(value: object, allowed: Iterable[str]) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise DeviceCapabilityError(_INVALID)
    allowed_set = frozenset(allowed)
    items = list(value)
    if not all(isinstance(item, str) and item in allowed_set for item in items):
        raise DeviceCapabilityError(_INVALID)
    return tuple(sorted(set(items)))


@dataclass(frozen=True)
class DeviceCapabilityProfile:
    """Coarse, closed description of what a device can execute.

    Collections are sets: they are sorted and de-duplicated on construction so
    equivalent profiles encode to identical canonical bytes.

    Args:
        platform: One of the portable-bundle platforms.
        sdk_name: Reading SDK, matched against bundle ``compatibility``.
        sdk_version: Dotted numeric SDK version.
        os_version: ``MAJOR`` or ``MAJOR.MINOR`` (Android uses the API level),
            or ``None`` when the platform has no OS version (web) or it is
            unknown.
        cpu_architecture: One of :data:`CPU_ARCHITECTURES`.
        providers: Available execution providers, each valid on ``platform``.
        runtimes: Bundle runtime formats the SDK can load on ``platform``.
        precisions: Numeric precisions the device can execute.
        features: Optional hardware or browser features (:data:`FEATURES`).
        operator_sets: Highest supported operator-set version per runtime
            format (ONNX opset, Core ML specification version).
        memory_budget_bytes: Memory the runtime may use, on the budget ladder,
            or ``None`` when unknown.
        storage_budget_bytes: Storage the model cache may use, on the budget
            ladder, or ``None`` when unknown.
        max_sequence_length: Longest token sequence the platform accepts, or
            ``None`` when unknown.
        max_model_tier: Largest model tier the platform allows, or ``None``
            when the platform sets no tier ceiling.
        network: ``offline``, ``online``, or ``unknown``. Informational only:
            resolution never uses the network, so this field never changes a
            plan and is excluded from :meth:`digest`.
    """

    platform: str
    sdk_name: str
    sdk_version: str
    os_version: str | None = None
    cpu_architecture: str = "unknown"
    providers: tuple[str, ...] = ()
    runtimes: tuple[str, ...] = ()
    precisions: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    operator_sets: Mapping[str, int] = field(default_factory=dict)
    memory_budget_bytes: int | None = None
    storage_budget_bytes: int | None = None
    max_sequence_length: int | None = None
    max_model_tier: str | None = None
    network: str = "unknown"

    def __post_init__(self) -> None:
        """Validate every member against the closed vocabulary."""

        if (
            self.platform not in PLATFORMS
            or not isinstance(self.sdk_name, str)
            or _TOKEN.fullmatch(self.sdk_name) is None
            or not isinstance(self.sdk_version, str)
            or _VERSION.fullmatch(self.sdk_version) is None
            or (
                self.os_version is not None
                and (
                    not isinstance(self.os_version, str)
                    or _COARSE_OS.fullmatch(self.os_version) is None
                )
            )
            or self.cpu_architecture not in CPU_ARCHITECTURES
            or not _is_budget(self.memory_budget_bytes)
            or not _is_budget(self.storage_budget_bytes)
            or (
                self.max_sequence_length is not None
                and not (
                    _is_uint(self.max_sequence_length, MAX_SEQUENCE_LENGTH)
                    and self.max_sequence_length > 0
                )
            )
            or (
                self.max_model_tier is not None
                and self.max_model_tier not in MODEL_TIERS
            )
            or self.network not in NETWORK_STATES
        ):
            raise DeviceCapabilityError(_INVALID)
        providers = _string_set(self.providers, PLATFORM_PROVIDERS[self.platform])
        runtimes = _string_set(
            self.runtimes,
            (fmt for fmt, where in FORMAT_PLATFORMS.items() if self.platform in where),
        )
        precisions = _string_set(self.precisions, PRECISIONS)
        features = _string_set(self.features, FEATURES)
        if not isinstance(self.operator_sets, Mapping) or not all(
            isinstance(key, str)
            and key in FORMAT_PLATFORMS
            and _is_uint(value, MAX_OPERATOR_SET)
            and value > 0
            for key, value in self.operator_sets.items()
        ):
            raise DeviceCapabilityError(_INVALID)
        object.__setattr__(self, "providers", providers)
        object.__setattr__(self, "runtimes", runtimes)
        object.__setattr__(self, "precisions", precisions)
        object.__setattr__(self, "features", features)
        object.__setattr__(
            self, "operator_sets", dict(sorted(self.operator_sets.items()))
        )

    @property
    def accelerators(self) -> tuple[str, ...]:
        """Return the sorted accelerator kinds behind :attr:`providers`."""

        return tuple(sorted({PROVIDER_ACCELERATORS[item] for item in self.providers}))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DeviceCapabilityProfile:
        """Parse a profile document (``device_capability_profile``).

        Args:
            data: Profile mapping with every member present.

        Returns:
            The validated profile.

        Raises:
            DeviceCapabilityError: ``profile_invalid`` for unknown or missing
                members, a foreign schema, or values outside the vocabulary.
        """

        if (
            not isinstance(data, Mapping)
            or set(data) != _PROFILE_KEYS
            or data["schema"] != DEVICE_CAPABILITIES_SCHEMA
            or type(data["schema_version"]) is not int
            or data["schema_version"] != DEVICE_CAPABILITIES_SCHEMA_VERSION
        ):
            raise DeviceCapabilityError(_INVALID)
        profile = cls(
            platform=data["platform"],
            sdk_name=data["sdk_name"],
            sdk_version=data["sdk_version"],
            os_version=data["os_version"],
            cpu_architecture=data["cpu_architecture"],
            providers=data["providers"],
            runtimes=data["runtimes"],
            precisions=data["precisions"],
            features=data["features"],
            operator_sets=data["operator_sets"],
            memory_budget_bytes=data["memory_budget_bytes"],
            storage_budget_bytes=data["storage_budget_bytes"],
            max_sequence_length=data["max_sequence_length"],
            max_model_tier=data["max_model_tier"],
            network=data["network"],
        )
        if _string_set(data["accelerators"], ACCELERATORS) != profile.accelerators:
            raise DeviceCapabilityError(_INVALID)
        return profile

    def to_dict(self) -> dict[str, Any]:
        """Return the profile document (``device_capability_profile``)."""

        return {
            "accelerators": list(self.accelerators),
            "cpu_architecture": self.cpu_architecture,
            "features": list(self.features),
            "max_model_tier": self.max_model_tier,
            "max_sequence_length": self.max_sequence_length,
            "memory_budget_bytes": self.memory_budget_bytes,
            "network": self.network,
            "operator_sets": dict(self.operator_sets),
            "os_version": self.os_version,
            "platform": self.platform,
            "precisions": list(self.precisions),
            "providers": list(self.providers),
            "runtimes": list(self.runtimes),
            "schema": DEVICE_CAPABILITIES_SCHEMA,
            "schema_version": DEVICE_CAPABILITIES_SCHEMA_VERSION,
            "sdk_name": self.sdk_name,
            "sdk_version": self.sdk_version,
            "storage_budget_bytes": self.storage_budget_bytes,
        }

    def canonical_bytes(self) -> bytes:
        """Return the RFC 8785 canonical encoding of :meth:`to_dict`."""

        return canonical_json_bytes(self.to_dict())

    def digest(self) -> str:
        """Return the ``sha256:`` digest of every resolution-relevant member.

        ``network`` is excluded because it never affects resolution.

        Returns:
            The profile digest used in plan invalidation keys.
        """

        value = self.to_dict()
        del value["network"]
        return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


# --------------------------------------------------------------------------
# Reference normalizers for the kit adapters
# --------------------------------------------------------------------------

_ANDROID_PROVIDERS = {"CPU": "cpu", "NNAPI": "nnapi", "QNN": "qnn"}
_ANDROID_ABIS = {
    "arm64-v8a": "arm64",
    "armeabi-v7a": "armv7",
    "x86": "unknown",
    "x86_64": "x86_64",
}
_ANDROID_DEVICE_TIERS = {
    "DEVICE_FARM_STUB": None,
    "ENTRY": "nano",
    "FLAGSHIP": "base",
    "MID_RANGE": "tiny",
}
_NNAPI_MIN_SDK = 27
_APPLE_PLATFORMS = {
    "iOS": "ios",
    "macOS": "macos",
    "visionOS": "visionos",
    "watchOS": "watchos",
}
_APPLE_ARCHITECTURES = {"arm64": "arm64", "x86_64": "x86_64"}
_MLX_PLATFORMS = frozenset({"ios", "macos"})
_MIN_THREADED_WASM_CONCURRENCY = 2


def _lookup(table: Mapping[str, Any], key: object, default: Any = None) -> Any:
    return table.get(key, default) if isinstance(key, str) else default


def _probe_bool(probe: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = probe.get(key, default)
    if not isinstance(value, bool):
        raise DeviceCapabilityError(DeviceCapabilityReason.PROBE_INVALID)
    return value


def _probe_optional_uint(probe: Mapping[str, Any], key: str) -> int | None:
    value = probe.get(key)
    if value is not None and not _is_uint(value):
        raise DeviceCapabilityError(DeviceCapabilityReason.PROBE_INVALID)
    return value


def _common(probe: Mapping[str, Any]) -> dict[str, Any]:
    operator_sets = probe.get("operator_sets", {})
    network = probe.get("network", "unknown")
    if not isinstance(operator_sets, Mapping) or network not in NETWORK_STATES:
        raise DeviceCapabilityError(DeviceCapabilityReason.PROBE_INVALID)
    return {
        "sdk_version": probe.get("sdk_version"),
        "operator_sets": dict(operator_sets),
        "storage_budget_bytes": budget_bucket(
            _probe_optional_uint(probe, "storage_free_bytes")
        ),
        "network": network,
    }


def _build(**members: Any) -> DeviceCapabilityProfile:
    try:
        return DeviceCapabilityProfile(**members)
    except DeviceCapabilityError:
        raise DeviceCapabilityError(DeviceCapabilityReason.PROBE_INVALID) from None


def profile_from_web_probe(probe: Mapping[str, Any]) -> DeviceCapabilityProfile:
    """Normalize an OpenMedKit web capability probe (reference mapping).

    Mirrors ``webDeviceCapabilityProfile`` in the ``openmed`` npm package. The
    probe carries the ``OrtWebCapabilityProfile`` fields plus caller options.
    Any other member, such as a user agent or GPU adapter description, is
    ignored and never reaches the profile.

    Args:
        probe: ``webgpu``, ``wasm``, ``wasm_simd``, ``shared_array_buffer``,
            ``cross_origin_isolated``, ``hardware_concurrency``,
            ``webgpu_shader_f16``, ``sdk_version``, and optional ``runtimes``,
            ``memory_budget_bytes``, ``storage_free_bytes``,
            ``max_sequence_length``, ``operator_sets``, and ``network``.

    Returns:
        The normalized profile.

    Raises:
        DeviceCapabilityError: ``probe_invalid`` for malformed probe values.
    """

    webgpu = _probe_bool(probe, "webgpu")
    wasm = _probe_bool(probe, "wasm")
    simd = wasm and _probe_bool(probe, "wasm_simd")
    concurrency = probe.get("hardware_concurrency", 1)
    if not _is_uint(concurrency):
        raise DeviceCapabilityError(DeviceCapabilityReason.PROBE_INVALID)
    threads = (
        simd
        and _probe_bool(probe, "shared_array_buffer")
        and _probe_bool(probe, "cross_origin_isolated")
        and concurrency >= _MIN_THREADED_WASM_CONCURRENCY
    )
    shader_f16 = webgpu and _probe_bool(probe, "webgpu_shader_f16")
    providers = [name for name, ok in (("wasm", wasm), ("webgpu", webgpu)) if ok]
    precisions: list[str] = []
    if providers:
        precisions = ["fp32", "int4", "int8"] + (["fp16"] if webgpu else [])
    features = [
        name
        for name, ok in (
            ("wasm_simd", simd),
            ("wasm_threads", threads),
            ("webgpu_shader_f16", shader_f16),
        )
        if ok
    ]
    return _build(
        platform="web",
        sdk_name=SDK_NAMES["web"],
        os_version=None,
        cpu_architecture="wasm32",
        providers=providers,
        runtimes=probe.get("runtimes", ["onnx"]),
        precisions=precisions,
        features=features,
        memory_budget_bytes=budget_bucket(
            _probe_optional_uint(probe, "memory_budget_bytes")
        ),
        max_sequence_length=probe.get("max_sequence_length"),
        max_model_tier=None,
        **_common(probe),
    )


def profile_from_android_probe(probe: Mapping[str, Any]) -> DeviceCapabilityProfile:
    """Normalize an OpenMedKit Android capability probe (reference mapping).

    Mirrors ``AndroidCapabilityProbe.profile`` in OpenMedKit for Android. The
    provider list is the ``AcceleratorProvider`` set that
    ``AcceleratorSession`` discovers; NNAPI counts only from API 27, as in the
    session. Build model, fingerprint, and serial members are ignored.

    Args:
        probe: ``sdk_int``, ``abi``, ``available_providers`` (``CPU``,
            ``NNAPI``, ``QNN``), ``sdk_version``, and optional
            ``memory_class_mb``, ``storage_free_bytes``, ``device_tier``
            (``AndroidDeviceTier`` name), ``max_sequence_length``,
            ``operator_sets``, and ``network``.

    Returns:
        The normalized profile.

    Raises:
        DeviceCapabilityError: ``probe_invalid`` for malformed probe values.
    """

    sdk_int = probe.get("sdk_int")
    abi = probe.get("abi")
    available = probe.get("available_providers", [])
    tier_name = probe.get("device_tier")
    memory_class = _probe_optional_uint(probe, "memory_class_mb")
    if (
        not _is_uint(sdk_int)
        or not isinstance(abi, str)
        or not isinstance(available, (list, tuple))
        or not all(item in _ANDROID_PROVIDERS for item in available)
        or (tier_name is not None and tier_name not in _ANDROID_DEVICE_TIERS)
    ):
        raise DeviceCapabilityError(DeviceCapabilityReason.PROBE_INVALID)
    assert isinstance(sdk_int, int)
    providers = {"cpu"}
    if "QNN" in available:
        providers.add("qnn")
    if "NNAPI" in available and sdk_int >= _NNAPI_MIN_SDK:
        providers.add("nnapi")
    architecture = _ANDROID_ABIS.get(abi, "unknown")
    return _build(
        platform="android",
        sdk_name=SDK_NAMES["android"],
        os_version=str(sdk_int),
        cpu_architecture=architecture,
        providers=providers,
        runtimes=["onnx", "ort"],
        precisions=["fp16", "fp32", "int4", "int8"],
        features=["cpu_neon"] if architecture == "arm64" else [],
        memory_budget_bytes=budget_bucket(
            None if memory_class is None else memory_class * MIB
        ),
        max_sequence_length=probe.get("max_sequence_length"),
        max_model_tier=None if tier_name is None else _ANDROID_DEVICE_TIERS[tier_name],
        **_common(probe),
    )


def profile_from_apple_probe(probe: Mapping[str, Any]) -> DeviceCapabilityProfile:
    """Normalize an OpenMedKit Apple capability probe (reference mapping).

    Mirrors ``DeviceCapabilityProfile.apple(...)`` in OpenMedKit for Swift. The
    memory budget, sequence limit, tier ceiling, and INT8 requirement come from
    ``PlatformModelConfiguration``; MLX counts only on iOS and macOS, where the
    package links it. Device names and vendor identifiers are ignored.

    Args:
        probe: ``platform`` (``OpenMedApplePlatform`` raw value),
            ``os_version``, ``architecture``, ``mlx_available``,
            ``configuration`` (``maximum_tier``,
            ``maximum_resident_memory_mb``, ``maximum_sequence_length``,
            ``requires_int8``), ``sdk_version``, and optional
            ``storage_free_bytes``, ``operator_sets``, and ``network``.

    Returns:
        The normalized profile.

    Raises:
        DeviceCapabilityError: ``probe_invalid`` for malformed probe values.
    """

    platform = _lookup(_APPLE_PLATFORMS, probe.get("platform"))
    configuration = probe.get("configuration")
    if platform is None or not isinstance(configuration, Mapping):
        raise DeviceCapabilityError(DeviceCapabilityReason.PROBE_INVALID)
    memory_mb = configuration.get("maximum_resident_memory_mb")
    sequence = configuration.get("maximum_sequence_length")
    tier = configuration.get("maximum_tier")
    if not _is_uint(memory_mb) or not _is_uint(sequence) or tier not in MODEL_TIERS:
        raise DeviceCapabilityError(DeviceCapabilityReason.PROBE_INVALID)
    assert isinstance(memory_mb, int)
    requires_int8 = _probe_bool(configuration, "requires_int8")
    mlx = _probe_bool(probe, "mlx_available") and platform in _MLX_PLATFORMS
    architecture = _lookup(_APPLE_ARCHITECTURES, probe.get("architecture"), "unknown")
    if requires_int8:
        precisions = ["int8"]
    else:
        precisions = ["fp16", "fp32", "int4", "int8"]
        if mlx:
            precisions += ["bf16", "int2"]
    return _build(
        platform=platform,
        sdk_name=SDK_NAMES["apple"],
        os_version=coarse_os_version(probe.get("os_version")),
        cpu_architecture=architecture,
        providers=["coreml", "cpu"] + (["metal"] if mlx else []),
        runtimes=["coreml"] + (["mlx"] if mlx else []),
        precisions=precisions,
        features=["cpu_neon"] if architecture == "arm64" else [],
        memory_budget_bytes=budget_bucket(memory_mb * MIB),
        max_sequence_length=sequence,
        max_model_tier=tier,
        **_common(probe),
    )


__all__ = [
    "ACCELERATORS",
    "CPU_ARCHITECTURES",
    "DEVICE_CAPABILITIES_SCHEMA",
    "DEVICE_CAPABILITIES_SCHEMA_VERSION",
    "FEATURES",
    "MAX_BUDGET_BYTES",
    "MAX_OPERATOR_SET",
    "MAX_SEQUENCE_LENGTH",
    "MIB",
    "MIN_BUDGET_BYTES",
    "MODEL_TIERS",
    "NETWORK_STATES",
    "PLATFORM_PROVIDERS",
    "PROVIDERS",
    "PROVIDER_ACCELERATORS",
    "SDK_NAMES",
    "DeviceCapabilityError",
    "DeviceCapabilityProfile",
    "DeviceCapabilityReason",
    "budget_bucket",
    "coarse_os_version",
    "model_tier_rank",
    "profile_from_android_probe",
    "profile_from_apple_probe",
    "profile_from_web_probe",
]
