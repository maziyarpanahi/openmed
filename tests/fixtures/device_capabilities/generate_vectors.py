"""Regenerate the device-capability and execution-plan golden vectors.

Usage::

    python tests/fixtures/device_capabilities/generate_vectors.py          # write
    python tests/fixtures/device_capabilities/generate_vectors.py --check  # verify

Every probe, profile, bundle, and file here is synthetic. Bundle files are
tiny text placeholders, so no model weights, patient data, credentials, or
restricted assets are involved. Probes carry planted ``SENTINEL`` identity
values (device names, serials, user agents, GPU adapter strings); the
normalizers must drop them, and every kit asserts that none reaches a profile
or plan. Verification results are written out directly, so no signing key is
used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openmed.core import device_capabilities as dc  # noqa: E402
from openmed.core import execution_plan as ep  # noqa: E402
from openmed.core import portable_bundle as pb  # noqa: E402

VECTORS_PATH = Path(__file__).with_name("vectors.json")
MIB = dc.MIB

SENTINELS = (
    "SENTINEL-DEVICE-NAME",
    "SENTINEL-SERIAL-0042",
    "SENTINEL-VENDOR-ID",
    "SENTINEL-USER-AGENT",
    "SENTINEL-GPU-ADAPTER",
    "SENTINEL-BUILD-FINGERPRINT",
)
"""Planted identity values that must never reach a profile or a plan."""

TREE = {
    "LICENSE": "Synthetic capability-matrix fixture.\nLicensed Apache-2.0.\n",
    "config.json": '{"model_type":"bert","num_labels":3,"synthetic":true}\n',
    "coreml/base.mlmodel": "synthetic coreml base fp16 placeholder\n",
    "coreml/nano.mlmodel": "synthetic coreml nano int8 placeholder\n",
    "coreml/tiny.mlmodel": "synthetic coreml tiny int8 placeholder\n",
    "evidence/coreml-parity.json": '{"cases":0,"synthetic":true}\n',
    "evidence/cpu-parity.json": '{"cases":0,"synthetic":true}\n',
    "evidence/webgpu-parity.json": '{"cases":0,"synthetic":true}\n',
    "id2label.json": '{"0":"O","1":"B-NAME","2":"I-NAME"}\n',
    "onnx/android-int8.onnx": "synthetic android int8 onnx placeholder\n",
    "onnx/web-fp16.onnx": "synthetic webgpu fp16 onnx placeholder\n",
    "onnx/web-int8.onnx": "synthetic wasm int8 onnx placeholder\n",
    "policy/default.json": '{"policy_version":1,"synthetic":true}\n',
    "tokenizer.json": '{"synthetic":true,"version":"1.0"}\n',
}


def _role(path: str) -> str:
    if path == "LICENSE":
        return "license"
    if path.startswith(("coreml/", "onnx/")):
        return "graph"
    if path.startswith("evidence/"):
        return "evidence"
    return {
        "config.json": "config",
        "id2label.json": "labels",
        "policy/default.json": "policy",
        "tokenizer.json": "tokenizer",
    }[path]


def _media(path: str) -> str:
    if path.endswith(".json"):
        return "application/json"
    if path.endswith(".onnx"):
        return "application/onnx"
    if path.endswith(".mlmodel"):
        return "application/vnd.apple.coreml"
    return "text/plain"


def digest(path: str) -> str:
    """Return the manifest digest of a synthetic tree file."""

    return "sha256:" + hashlib.sha256(TREE[path].encode("utf-8")).hexdigest()


def _platform(name: str, min_os: str | None) -> dict[str, Any]:
    return {"max_os_exclusive": None, "min_os": min_os, "platform": name}


def _runtime(
    runtime_id: str,
    runtime_format: str,
    precision: str,
    entrypoint: str,
    platforms: list[dict[str, Any]],
    memory_mib: int,
    storage_mib: int,
    fallback: str | None,
) -> dict[str, Any]:
    return {
        "entrypoint": entrypoint,
        "fallback": fallback,
        "files": sorted({"config.json", entrypoint}),
        "format": runtime_format,
        "id": runtime_id,
        "min_memory_bytes": memory_mib * MIB,
        "min_storage_bytes": storage_mib * MIB,
        "platforms": platforms,
        "precision": precision,
    }


def _meta(
    providers: list[str],
    evidence: list[str],
    *,
    features: list[str] = (),
    max_sequence_length: int | None = 512,
    min_operator_set: int | None = None,
    tier: str | None = None,
) -> dict[str, Any]:
    return {
        "evidence": list(evidence),
        "features": list(features),
        "max_sequence_length": max_sequence_length,
        "min_operator_set": min_operator_set,
        "providers": list(providers),
        "tier": tier,
    }


def matrix_manifest() -> dict[str, Any]:
    """Return the capability-matrix manifest covering web, Android, and Apple."""

    apple = [_platform("ios", "17.0"), _platform("macos", "14.0")]
    return {
        "bundle": {
            "expires_at": "2027-10-01T00:00:00Z",
            "id": "openmed.synthetic.capability-matrix",
            "issued_at": "2026-10-01T00:00:00Z",
            "sequence": 1,
            "version": "1.0.0",
        },
        "calibration": None,
        "compatibility": {
            "sdks": [
                {"max_exclusive": "5.0.0", "min": "4.0.0", "name": name}
                for name in (
                    "openmed-python",
                    "openmedkit-android",
                    "openmedkit-swift",
                    "openmedkit-web",
                )
            ]
        },
        "critical_extensions": [],
        "decoder": {"contract_version": 1, "kind": "bio"},
        "evidence": [
            {"kind": "coreml_parity", "path": "evidence/coreml-parity.json"},
            {"kind": "cpu_parity", "path": "evidence/cpu-parity.json"},
            {"kind": "webgpu_parity", "path": "evidence/webgpu-parity.json"},
        ],
        "extensions": {
            ep.EXECUTION_EXTENSION: {
                "runtimes": {
                    "coreml-base-fp16": _meta(
                        ["coreml"], ["coreml_parity"], tier="base"
                    ),
                    "coreml-nano-int8": _meta(
                        ["coreml", "cpu"],
                        ["coreml_parity"],
                        max_sequence_length=256,
                        tier="nano",
                    ),
                    "coreml-tiny-int8": _meta(
                        ["coreml"], ["coreml_parity"], tier="tiny"
                    ),
                    "onnx-android-int8": _meta(["qnn", "nnapi", "cpu"], ["cpu_parity"]),
                    "onnx-wasm-int8": _meta(["wasm"], ["cpu_parity"]),
                    "onnx-webgpu-fp16": _meta(
                        ["webgpu"],
                        ["webgpu_parity"],
                        features=["webgpu_shader_f16"],
                        min_operator_set=17,
                    ),
                },
                "schema_version": ep.EXECUTION_EXTENSION_VERSION,
            }
        },
        "files": [
            {
                "executable": False,
                "media_type": _media(path),
                "path": path,
                "role": _role(path),
                "sha256": digest(path),
                "size_bytes": len(TREE[path].encode("utf-8")),
            }
            for path in sorted(TREE, key=lambda item: item.encode("utf-8"))
        ],
        "labels": {"count": 3, "path": "id2label.json", "scheme": "bio"},
        "licenses": [{"files": ["LICENSE"], "id": "apache-2.0"}],
        "model": {
            "architecture": "bert",
            "family": "pii",
            "id": "OpenMed/synthetic-capability-matrix",
            "languages": ["en"],
            "revision": None,
            "task": "token-classification",
        },
        "offsets": {"contract_version": 1, "unit": "unicode_scalar"},
        "policy": {
            "max_version_exclusive": 2,
            "min_version": 1,
            "path": "policy/default.json",
        },
        "provenance": {"export_record": None, "source_digest": None},
        "runtimes": [
            _runtime(
                "onnx-webgpu-fp16",
                "onnx",
                "fp16",
                "onnx/web-fp16.onnx",
                [_platform("web", None)],
                512,
                64,
                "onnx-wasm-int8",
            ),
            _runtime(
                "onnx-wasm-int8",
                "onnx",
                "int8",
                "onnx/web-int8.onnx",
                [_platform("web", None)],
                256,
                32,
                None,
            ),
            _runtime(
                "onnx-android-int8",
                "onnx",
                "int8",
                "onnx/android-int8.onnx",
                [_platform("android", "26")],
                256,
                32,
                None,
            ),
            _runtime(
                "coreml-base-fp16",
                "coreml",
                "fp16",
                "coreml/base.mlmodel",
                apple,
                512,
                64,
                "coreml-tiny-int8",
            ),
            _runtime(
                "coreml-tiny-int8",
                "coreml",
                "int8",
                "coreml/tiny.mlmodel",
                [*apple, _platform("visionos", "1.0")],
                128,
                32,
                "coreml-nano-int8",
            ),
            _runtime(
                "coreml-nano-int8",
                "coreml",
                "int8",
                "coreml/nano.mlmodel",
                [*apple, _platform("visionos", "1.0"), _platform("watchos", "10.0")],
                96,
                16,
                None,
            ),
        ],
        "schema": pb.PORTABLE_BUNDLE_SCHEMA,
        "schema_version": pb.PORTABLE_BUNDLE_SCHEMA_VERSION,
        "segmenter": None,
        "tokenizer": {
            "contract_version": 1,
            "files": ["tokenizer.json"],
            "kind": "wordpiece",
        },
    }


def _ext(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest["extensions"][ep.EXECUTION_EXTENSION]["runtimes"]


def manifests() -> dict[str, bytes]:
    """Return every manifest variant as canonical bytes."""

    variants: dict[str, dict[str, Any]] = {"matrix": matrix_manifest()}

    value = matrix_manifest()
    value["critical_extensions"] = [ep.EXECUTION_EXTENSION]
    variants["critical_extension"] = value

    value = matrix_manifest()
    _ext(value)["onnx-android-int8"]["providers"] = ["nnapi"]
    variants["android_nnapi_only"] = value

    value = matrix_manifest()
    value["runtimes"][3]["fallback"] = None
    variants["apple_no_tier_fallback"] = value

    value = matrix_manifest()
    value["extensions"] = {}
    variants["no_extension"] = value

    value = matrix_manifest()
    _ext(value)["onnx-wasm-int8"]["providers"] = ["remote"]
    variants["metadata_remote_provider"] = value

    value = matrix_manifest()
    _ext(value)["coreml-base-fp16"]["providers"] = ["webgpu"]
    variants["metadata_provider_format_mismatch"] = value

    value = matrix_manifest()
    _ext(value)["onnx-wasm-int8"]["evidence"] = ["clinical_validation"]
    variants["metadata_unknown_evidence"] = value

    value = matrix_manifest()
    _ext(value)["ghost-runtime"] = _meta(["cpu"], [])
    variants["metadata_unknown_runtime"] = value

    return {
        name: pb.PortableBundleManifest.from_mapping(
            item, understood_extensions=item["critical_extensions"]
        ).canonical_bytes
        for name, item in variants.items()
    }


IDENTITY_WEB = {
    "gpu_adapter": {
        "architecture": "SENTINEL-GPU-ADAPTER",
        "description": "SENTINEL-GPU-ADAPTER",
        "vendor": "SENTINEL-VENDOR-ID",
    },
    "user_agent": "SENTINEL-USER-AGENT",
}
IDENTITY_ANDROID = {
    "fingerprint": "SENTINEL-BUILD-FINGERPRINT",
    "model": "SENTINEL-DEVICE-NAME",
    "serial": "SENTINEL-SERIAL-0042",
}
IDENTITY_APPLE = {
    "device_name": "SENTINEL-DEVICE-NAME",
    "identifier_for_vendor": "SENTINEL-VENDOR-ID",
    "serial": "SENTINEL-SERIAL-0042",
}


def _web(**overrides: Any) -> dict[str, Any]:
    probe = {
        "cross_origin_isolated": True,
        "hardware_concurrency": 8,
        "memory_budget_bytes": 3_000_000_000,
        "network": "online",
        "operator_sets": {"onnx": 21},
        "runtimes": ["onnx"],
        "sdk_version": "4.0.0",
        "shared_array_buffer": True,
        "storage_free_bytes": 10_000_000_000,
        "wasm": True,
        "wasm_simd": True,
        "webgpu": True,
        "webgpu_shader_f16": True,
        **IDENTITY_WEB,
    }
    probe.update(overrides)
    return {key: value for key, value in probe.items() if value is not None}


def _android(**overrides: Any) -> dict[str, Any]:
    probe = {
        "abi": "arm64-v8a",
        "available_providers": ["CPU", "NNAPI", "QNN"],
        "device_tier": "FLAGSHIP",
        "memory_class_mb": 512,
        "network": "offline",
        "sdk_int": 34,
        "sdk_version": "4.0.0",
        "storage_free_bytes": 4_000_000_000,
        **IDENTITY_ANDROID,
    }
    probe.update(overrides)
    return {key: value for key, value in probe.items() if value is not None}


def _apple(platform: str, os_version: str, configuration: dict[str, Any]) -> dict:
    return {
        "architecture": "arm64",
        "configuration": configuration,
        "mlx_available": True,
        "network": "unknown",
        "os_version": os_version,
        "platform": platform,
        "sdk_version": "4.1.0",
        "storage_free_bytes": 50_000_000_000,
        **IDENTITY_APPLE,
    }


# PlatformModelConfiguration.configuration(for:) values in PlatformModel.swift.
APPLE_CONFIGURATIONS = {
    "iOS": {
        "maximum_resident_memory_mb": 350,
        "maximum_sequence_length": 512,
        "maximum_tier": "tiny",
        "requires_int8": False,
    },
    "macOS": {
        "maximum_resident_memory_mb": 900,
        "maximum_sequence_length": 512,
        "maximum_tier": "base",
        "requires_int8": False,
    },
    "visionOS": {
        "maximum_resident_memory_mb": 150,
        "maximum_sequence_length": 256,
        "maximum_tier": "nano",
        "requires_int8": True,
    },
    "watchOS": {
        "maximum_resident_memory_mb": 150,
        "maximum_sequence_length": 256,
        "maximum_tier": "nano",
        "requires_int8": True,
    },
}

PROBES: dict[str, tuple[str, dict[str, Any]]] = {
    "web_full": ("web", _web()),
    "web_full_offline": ("web", _web(network="offline")),
    "web_no_webgpu": (
        "web",
        _web(
            webgpu=False,
            webgpu_shader_f16=False,
            shared_array_buffer=False,
            cross_origin_isolated=False,
            hardware_concurrency=4,
            memory_budget_bytes=None,
            storage_free_bytes=None,
        ),
    ),
    "web_no_shader_f16": ("web", _web(webgpu_shader_f16=False)),
    "web_no_operator_sets": ("web", _web(operator_sets={})),
    "web_old_operator_set": ("web", _web(operator_sets={"onnx": 13})),
    "web_basic_wasm": (
        "web",
        _web(
            webgpu=False,
            wasm_simd=False,
            hardware_concurrency=1,
            memory_budget_bytes=None,
        ),
    ),
    "web_none": ("web", _web(webgpu=False, wasm=False, wasm_simd=False)),
    "web_sdk_too_new": ("web", _web(sdk_version="5.0.0")),
    "android_flagship": ("android", _android()),
    "android_cpu_only": (
        "android",
        _android(
            abi="x86_64",
            available_providers=["CPU"],
            device_tier=None,
            memory_class_mb=256,
            sdk_int=30,
        ),
    ),
    "android_cpu_only_os_update": (
        "android",
        _android(
            abi="x86_64",
            available_providers=["CPU"],
            device_tier=None,
            memory_class_mb=256,
            sdk_int=31,
        ),
    ),
    "android_api26_nnapi": (
        "android",
        _android(
            abi="armeabi-v7a",
            available_providers=["CPU", "NNAPI"],
            device_tier="ENTRY",
            sdk_int=26,
        ),
    ),
    "android_api25": ("android", _android(sdk_int=25, device_tier="MID_RANGE")),
    "android_low_storage": ("android", _android(storage_free_bytes=20 * MIB)),
    "apple_macos": ("apple", _apple("macOS", "15.1.0", APPLE_CONFIGURATIONS["macOS"])),
    "apple_ios": ("apple", _apple("iOS", "17.4.1", APPLE_CONFIGURATIONS["iOS"])),
    "apple_ios16": ("apple", _apple("iOS", "16.7", APPLE_CONFIGURATIONS["iOS"])),
    "apple_watchos": (
        "apple",
        _apple("watchOS", "10.2", APPLE_CONFIGURATIONS["watchOS"]),
    ),
    "apple_visionos": (
        "apple",
        _apple("visionOS", "1.1", APPLE_CONFIGURATIONS["visionOS"]),
    ),
}

NORMALIZERS = {
    "android": dc.profile_from_android_probe,
    "apple": dc.profile_from_apple_probe,
    "web": dc.profile_from_web_probe,
}


def profiles() -> dict[str, dc.DeviceCapabilityProfile]:
    """Return every named profile: normalized probes plus direct records."""

    result = {name: NORMALIZERS[kit](probe) for name, (kit, probe) in PROBES.items()}
    flagship = result["android_flagship"].to_dict()
    flagship["os_version"] = None
    result["android_os_unknown"] = dc.DeviceCapabilityProfile.from_mapping(flagship)
    result["linux_python"] = dc.DeviceCapabilityProfile(
        platform="linux",
        sdk_name="openmed-python",
        sdk_version="4.0.0",
        os_version=None,
        cpu_architecture="x86_64",
        providers=("cpu",),
        runtimes=("onnx", "pytorch"),
        precisions=("fp32", "int8"),
        features=("cpu_avx2",),
        memory_budget_bytes=8 << 30,
        storage_budget_bytes=64 << 30,
    )
    result["macos_python"] = dc.DeviceCapabilityProfile(
        platform="macos",
        sdk_name="openmed-python",
        sdk_version="4.0.0",
        os_version="15.1",
        cpu_architecture="arm64",
        providers=("coreml", "cpu"),
        runtimes=("coreml",),
        precisions=("fp16", "fp32", "int8"),
        features=("cpu_neon",),
        memory_budget_bytes=8 << 30,
        storage_budget_bytes=64 << 30,
    )
    return result


def constraints() -> dict[str, ep.ExecutionConstraints]:
    """Return every named caller-constraint set."""

    return {
        "default": ep.ExecutionConstraints(),
        "low_power": ep.ExecutionConstraints(energy="low_power"),
        "no_webgpu": ep.ExecutionConstraints(disallowed_providers=("webgpu",)),
        "no_qnn": ep.ExecutionConstraints(disallowed_providers=("qnn",)),
        "fp16_only": ep.ExecutionConstraints(allowed_precisions=("fp16",)),
        "int4_only": ep.ExecutionConstraints(allowed_precisions=("int4",)),
        "webgpu_evidence": ep.ExecutionConstraints(
            required_evidence=("webgpu_parity",)
        ),
        "clinical_evidence": ep.ExecutionConstraints(
            required_evidence=("clinical_validation",)
        ),
        "sequence_384": ep.ExecutionConstraints(min_sequence_length=384),
        "sequence_1024": ep.ExecutionConstraints(min_sequence_length=1024),
        "revoke_wasm": ep.ExecutionConstraints(
            revoked_artifacts=(digest("onnx/web-int8.onnx"),)
        ),
        "memory_300mib": ep.ExecutionConstraints(max_memory_bytes=300 * MIB),
        "one_step": ep.ExecutionConstraints(max_steps=1),
        "remote": ep.ExecutionConstraints(local_only=False),
        "manifest_only": ep.ExecutionConstraints(require_verified_files=False),
        "deadline": ep.ExecutionConstraints(deadline_ms=1500),
    }


def _verification(
    manifest: bytes,
    kind: str,
    runtime_count: int,
    other: bytes,
) -> pb.PortableBundleVerification:
    identity = pb.content_id(manifest)
    every = tuple(range(runtime_count))
    files = len(TREE)
    if kind == "ok":
        return pb.PortableBundleVerification(
            True, None, identity, 1, every, files, True
        )
    if kind == "manifest_only":
        return pb.PortableBundleVerification(True, None, identity, 1, every, 0, False)
    if kind == "signature_invalid":
        return pb.PortableBundleVerification(False, "signature_invalid", identity, 0)
    if kind == "bundle_revoked":
        return pb.PortableBundleVerification(False, "bundle_revoked", identity, 0)
    if kind == "content_mismatch":
        return pb.PortableBundleVerification(
            True, None, pb.content_id(other), 1, every, files, True
        )
    if kind == "web_wasm_only":
        return pb.PortableBundleVerification(True, None, identity, 1, (1,), files, True)
    raise KeyError(kind)


# (id, manifest, verification, profile, constraints, expected steps or reason)
CASES: tuple[tuple[str, str, str, str, str, Any], ...] = (
    (
        "web_webgpu_then_wasm",
        "matrix",
        "ok",
        "web_full",
        "default",
        [
            ("onnx-webgpu-fp16", "webgpu", "selected"),
            ("onnx-wasm-int8", "wasm", "artifact_fallback"),
        ],
    ),
    (
        "web_offline_same_plan",
        "matrix",
        "ok",
        "web_full_offline",
        "default",
        [
            ("onnx-webgpu-fp16", "webgpu", "selected"),
            ("onnx-wasm-int8", "wasm", "artifact_fallback"),
        ],
    ),
    (
        "web_without_webgpu_uses_wasm",
        "matrix",
        "ok",
        "web_no_webgpu",
        "default",
        [("onnx-wasm-int8", "wasm", "selected")],
    ),
    (
        "web_basic_wasm",
        "matrix",
        "ok",
        "web_basic_wasm",
        "default",
        [("onnx-wasm-int8", "wasm", "selected")],
    ),
    (
        "web_without_shader_f16",
        "matrix",
        "ok",
        "web_no_shader_f16",
        "default",
        [("onnx-wasm-int8", "wasm", "selected")],
    ),
    (
        "web_operator_set_unknown",
        "matrix",
        "ok",
        "web_no_operator_sets",
        "default",
        [("onnx-wasm-int8", "wasm", "selected")],
    ),
    (
        "web_operator_set_too_old",
        "matrix",
        "ok",
        "web_old_operator_set",
        "default",
        [("onnx-wasm-int8", "wasm", "selected")],
    ),
    (
        "web_low_power_skips_gpu",
        "matrix",
        "ok",
        "web_full",
        "low_power",
        [("onnx-wasm-int8", "wasm", "selected")],
    ),
    (
        "web_webgpu_disallowed",
        "matrix",
        "ok",
        "web_full",
        "no_webgpu",
        [("onnx-wasm-int8", "wasm", "selected")],
    ),
    (
        "web_fp16_only_has_no_fallback",
        "matrix",
        "ok",
        "web_full",
        "fp16_only",
        [("onnx-webgpu-fp16", "webgpu", "selected")],
    ),
    (
        "web_int4_only_unsupported",
        "matrix",
        "ok",
        "web_full",
        "int4_only",
        "no_eligible_runtime",
    ),
    (
        "web_requires_webgpu_evidence",
        "matrix",
        "ok",
        "web_full",
        "webgpu_evidence",
        [("onnx-webgpu-fp16", "webgpu", "selected")],
    ),
    (
        "web_unqualified_unsupported",
        "matrix",
        "ok",
        "web_full",
        "clinical_evidence",
        "no_eligible_runtime",
    ),
    (
        "web_sequence_384",
        "matrix",
        "ok",
        "web_full",
        "sequence_384",
        [
            ("onnx-webgpu-fp16", "webgpu", "selected"),
            ("onnx-wasm-int8", "wasm", "artifact_fallback"),
        ],
    ),
    (
        "web_sequence_1024_unsupported",
        "matrix",
        "ok",
        "web_full",
        "sequence_1024",
        "no_eligible_runtime",
    ),
    (
        "web_sequence_undeclared_unsupported",
        "no_extension",
        "ok",
        "web_full",
        "sequence_384",
        "no_eligible_runtime",
    ),
    (
        "web_revoked_wasm_artifact",
        "matrix",
        "ok",
        "web_full",
        "revoke_wasm",
        [("onnx-webgpu-fp16", "webgpu", "selected")],
    ),
    (
        "web_revoked_wasm_without_webgpu",
        "matrix",
        "ok",
        "web_no_webgpu",
        "revoke_wasm",
        "no_eligible_runtime",
    ),
    (
        "web_memory_ceiling",
        "matrix",
        "ok",
        "web_full",
        "memory_300mib",
        [("onnx-wasm-int8", "wasm", "selected")],
    ),
    (
        "web_one_step_bound",
        "matrix",
        "ok",
        "web_full",
        "one_step",
        [("onnx-webgpu-fp16", "webgpu", "selected")],
    ),
    (
        "web_deadline_recorded",
        "matrix",
        "ok",
        "web_full",
        "deadline",
        [
            ("onnx-webgpu-fp16", "webgpu", "selected"),
            ("onnx-wasm-int8", "wasm", "artifact_fallback"),
        ],
    ),
    (
        "web_remote_refused",
        "matrix",
        "ok",
        "web_full",
        "remote",
        "remote_execution_unsupported",
    ),
    (
        "web_sdk_incompatible",
        "matrix",
        "ok",
        "web_sdk_too_new",
        "default",
        "sdk_incompatible",
    ),
    (
        "web_signature_invalid",
        "matrix",
        "signature_invalid",
        "web_full",
        "default",
        "bundle_unverified",
    ),
    (
        "web_bundle_revoked",
        "matrix",
        "bundle_revoked",
        "web_full",
        "default",
        "bundle_revoked",
    ),
    (
        "web_content_id_mismatch",
        "matrix",
        "content_mismatch",
        "web_full",
        "default",
        "bundle_unverified",
    ),
    (
        "web_files_unverified",
        "matrix",
        "manifest_only",
        "web_full",
        "default",
        "bundle_files_unverified",
    ),
    (
        "web_manifest_only_allowed",
        "matrix",
        "manifest_only",
        "web_full",
        "manifest_only",
        [
            ("onnx-webgpu-fp16", "webgpu", "selected"),
            ("onnx-wasm-int8", "wasm", "artifact_fallback"),
        ],
    ),
    (
        "web_runtime_not_verified_eligible",
        "matrix",
        "web_wasm_only",
        "web_full",
        "default",
        [("onnx-wasm-int8", "wasm", "selected")],
    ),
    ("web_no_providers", "matrix", "ok", "web_none", "default", "no_eligible_runtime"),
    (
        "web_metadata_remote_provider",
        "metadata_remote_provider",
        "ok",
        "web_full",
        "default",
        "execution_metadata_invalid",
    ),
    (
        "web_metadata_provider_format_mismatch",
        "metadata_provider_format_mismatch",
        "ok",
        "web_full",
        "default",
        "execution_metadata_invalid",
    ),
    (
        "web_metadata_unknown_evidence",
        "metadata_unknown_evidence",
        "ok",
        "web_full",
        "default",
        "execution_metadata_invalid",
    ),
    (
        "web_metadata_unknown_runtime",
        "metadata_unknown_runtime",
        "ok",
        "web_full",
        "default",
        "execution_metadata_invalid",
    ),
    (
        "web_no_extension_baseline",
        "no_extension",
        "ok",
        "web_full",
        "default",
        [
            ("onnx-webgpu-fp16", "wasm", "selected"),
            ("onnx-wasm-int8", "wasm", "artifact_fallback"),
        ],
    ),
    (
        "web_critical_extension",
        "critical_extension",
        "ok",
        "web_full",
        "default",
        [
            ("onnx-webgpu-fp16", "webgpu", "selected"),
            ("onnx-wasm-int8", "wasm", "artifact_fallback"),
        ],
    ),
    (
        "android_provider_chain",
        "matrix",
        "ok",
        "android_flagship",
        "default",
        [
            ("onnx-android-int8", "qnn", "selected"),
            ("onnx-android-int8", "nnapi", "provider_fallback"),
            ("onnx-android-int8", "cpu", "provider_fallback"),
        ],
    ),
    (
        "android_cpu_only_device",
        "matrix",
        "ok",
        "android_cpu_only",
        "default",
        [("onnx-android-int8", "cpu", "selected")],
    ),
    (
        "android_cpu_only_after_os_update",
        "matrix",
        "ok",
        "android_cpu_only_os_update",
        "default",
        [("onnx-android-int8", "cpu", "selected")],
    ),
    (
        "android_nnapi_needs_api_27",
        "matrix",
        "ok",
        "android_api26_nnapi",
        "default",
        [("onnx-android-int8", "cpu", "selected")],
    ),
    (
        "android_qnn_disallowed",
        "matrix",
        "ok",
        "android_flagship",
        "no_qnn",
        [
            ("onnx-android-int8", "nnapi", "selected"),
            ("onnx-android-int8", "cpu", "provider_fallback"),
        ],
    ),
    (
        "android_nnapi_only_bundle_on_cpu_device",
        "android_nnapi_only",
        "ok",
        "android_cpu_only",
        "default",
        "no_eligible_runtime",
    ),
    (
        "android_undeclared_providers_stay_on_cpu",
        "no_extension",
        "ok",
        "android_flagship",
        "default",
        [("onnx-android-int8", "cpu", "selected")],
    ),
    (
        "android_api_25_unsupported",
        "matrix",
        "ok",
        "android_api25",
        "default",
        "no_eligible_runtime",
    ),
    (
        "android_os_unknown",
        "matrix",
        "ok",
        "android_os_unknown",
        "default",
        "no_eligible_runtime",
    ),
    (
        "android_low_storage",
        "matrix",
        "ok",
        "android_low_storage",
        "default",
        "no_eligible_runtime",
    ),
    (
        "android_one_step_bound",
        "matrix",
        "ok",
        "android_flagship",
        "one_step",
        [("onnx-android-int8", "qnn", "selected")],
    ),
    (
        "apple_macos_tier_chain",
        "matrix",
        "ok",
        "apple_macos",
        "default",
        [
            ("coreml-base-fp16", "coreml", "selected"),
            ("coreml-tiny-int8", "coreml", "artifact_fallback"),
            ("coreml-nano-int8", "coreml", "artifact_fallback"),
            ("coreml-nano-int8", "cpu", "provider_fallback"),
        ],
    ),
    (
        "apple_ios_tier_fallback",
        "matrix",
        "ok",
        "apple_ios",
        "default",
        [
            ("coreml-tiny-int8", "coreml", "selected"),
            ("coreml-nano-int8", "coreml", "artifact_fallback"),
            ("coreml-nano-int8", "cpu", "provider_fallback"),
        ],
    ),
    (
        "apple_ios_sequence_384",
        "matrix",
        "ok",
        "apple_ios",
        "sequence_384",
        [("coreml-tiny-int8", "coreml", "selected")],
    ),
    (
        "apple_watchos_nano",
        "matrix",
        "ok",
        "apple_watchos",
        "default",
        [
            ("coreml-nano-int8", "coreml", "selected"),
            ("coreml-nano-int8", "cpu", "provider_fallback"),
        ],
    ),
    (
        "apple_visionos_tier_ceiling",
        "matrix",
        "ok",
        "apple_visionos",
        "default",
        [
            ("coreml-nano-int8", "coreml", "selected"),
            ("coreml-nano-int8", "cpu", "provider_fallback"),
        ],
    ),
    (
        "apple_watchos_sequence_unsupported",
        "matrix",
        "ok",
        "apple_watchos",
        "sequence_384",
        "no_eligible_runtime",
    ),
    (
        "apple_tier_fallback_only_when_declared",
        "apple_no_tier_fallback",
        "ok",
        "apple_macos",
        "default",
        [("coreml-base-fp16", "coreml", "selected")],
    ),
    (
        "apple_ios16_unsupported",
        "matrix",
        "ok",
        "apple_ios16",
        "default",
        "no_eligible_runtime",
    ),
    (
        "macos_python_tier_limit_unknown",
        "matrix",
        "ok",
        "macos_python",
        "default",
        [
            ("coreml-base-fp16", "coreml", "selected"),
            ("coreml-tiny-int8", "coreml", "artifact_fallback"),
            ("coreml-nano-int8", "coreml", "artifact_fallback"),
            ("coreml-nano-int8", "cpu", "provider_fallback"),
        ],
    ),
    (
        "linux_python_no_runtime",
        "matrix",
        "ok",
        "linux_python",
        "default",
        "no_eligible_runtime",
    ),
)

# (plan from case, current inputs from case, changed components)
INVALIDATIONS: tuple[tuple[str, str, list[str]], ...] = (
    ("web_webgpu_then_wasm", "web_webgpu_then_wasm", []),
    ("web_webgpu_then_wasm", "web_offline_same_plan", []),
    ("web_webgpu_then_wasm", "web_without_webgpu_uses_wasm", ["profile"]),
    ("web_webgpu_then_wasm", "web_low_power_skips_gpu", ["constraints"]),
    ("web_webgpu_then_wasm", "web_critical_extension", ["bundle", "verification"]),
    ("web_webgpu_then_wasm", "web_bundle_revoked", ["verification"]),
    ("web_webgpu_then_wasm", "web_files_unverified", ["verification"]),
    ("android_cpu_only_device", "android_cpu_only_after_os_update", ["profile"]),
    ("android_provider_chain", "android_qnn_disallowed", ["constraints"]),
    ("apple_ios_tier_fallback", "apple_ios16_unsupported", ["profile"]),
    (
        "apple_macos_tier_chain",
        "apple_tier_fallback_only_when_declared",
        ["bundle", "verification"],
    ),
)


def vocabulary() -> dict[str, Any]:
    """Return the shared vocabulary every kit must reproduce exactly."""

    return {
        "accelerators": sorted(dc.ACCELERATORS),
        "assumptions": list(ep.ASSUMPTIONS),
        "budget_ladder": {
            "max_bytes": dc.MAX_BUDGET_BYTES,
            "min_bytes": dc.MIN_BUDGET_BYTES,
        },
        "cpu_architectures": sorted(dc.CPU_ARCHITECTURES),
        "default_providers": {
            key: list(value) for key, value in sorted(ep.DEFAULT_PROVIDERS.items())
        },
        "energy_postures": sorted(ep.ENERGY_POSTURES),
        "execution_extension": ep.EXECUTION_EXTENSION,
        "features": sorted(dc.FEATURES),
        "format_providers": {
            key: sorted(value) for key, value in sorted(ep.FORMAT_PROVIDERS.items())
        },
        "invalidation_components": list(ep.INVALIDATION_COMPONENTS),
        "model_tiers": list(dc.MODEL_TIERS),
        "network_states": sorted(dc.NETWORK_STATES),
        "platform_providers": {
            key: sorted(value) for key, value in sorted(dc.PLATFORM_PROVIDERS.items())
        },
        "provider_accelerators": dict(sorted(dc.PROVIDER_ACCELERATORS.items())),
        "reason_codes": list(ep.REASON_CODES),
        "resolver_version": ep.RESOLVER_VERSION,
        "sdk_names": dict(sorted(dc.SDK_NAMES.items())),
        "step_roles": list(ep.STEP_ROLES),
    }


def _check_expectation(case_id: str, plan: ep.ExecutionPlan, expected: Any) -> None:
    if isinstance(expected, str):
        assert plan.status == "unsupported", case_id
        assert plan.reason == expected, (case_id, plan.reason)
        return
    assert plan.status == "planned", (case_id, plan.reason)
    observed = [(step.runtime_id, step.provider, step.role) for step in plan.steps]
    assert observed == expected, (case_id, observed)


def build() -> dict[str, Any]:
    """Build the full vector document and check every expectation."""

    manifest_bytes = manifests()
    profile_set = profiles()
    constraint_set = constraints()
    runtime_count = len(matrix_manifest()["runtimes"])
    other = manifest_bytes["no_extension"]

    cases = []
    by_id: dict[str, tuple[Any, ...]] = {}
    for case_id, manifest, kind, profile, constraint, expected in CASES:
        verification = _verification(
            manifest_bytes[manifest], kind, runtime_count, other
        )
        plan = ep.resolve_execution_plan(
            manifest_bytes[manifest],
            verification,
            profile_set[profile],
            constraint_set[constraint],
        )
        _check_expectation(case_id, plan, expected)
        by_id[case_id] = (
            manifest_bytes[manifest],
            verification,
            profile_set[profile],
            constraint_set[constraint],
            plan,
        )
        cases.append(
            {
                "constraints": constraint,
                "expected": plan.to_dict(),
                "id": case_id,
                "manifest": manifest,
                "profile": profile,
                "verification": verification.to_dict(),
            }
        )

    first = by_id["web_webgpu_then_wasm"][4].to_dict()
    second = by_id["web_offline_same_plan"][4].to_dict()
    assert first == second, "network state must not change a plan"

    covered = {case["expected"]["reason"] for case in cases}
    for case in cases:
        covered |= {item["reason"] for item in case["expected"]["skipped"]}
    missing = set(ep.REASON_CODES) - covered
    assert not missing, sorted(missing)
    assumed = {item for case in cases for item in case["expected"]["assumptions"]}
    assert assumed == set(ep.ASSUMPTIONS), sorted(set(ep.ASSUMPTIONS) - assumed)

    invalidations = []
    for plan_case, inputs_case, expected in INVALIDATIONS:
        plan = by_id[plan_case][4]
        manifest, verification, profile, constraint, _ = by_id[inputs_case]
        changed = list(
            ep.plan_invalidation(plan, manifest, verification, profile, constraint)
        )
        assert changed == expected, (plan_case, inputs_case, changed)
        invalidations.append(
            {"expected": changed, "inputs": inputs_case, "plan": plan_case}
        )

    normalization = []
    for name, (kit, probe) in PROBES.items():
        normalization.append(
            {
                "expected": profile_set[name].to_dict(),
                "id": name,
                "kit": kit,
                "probe": copy.deepcopy(probe),
            }
        )

    return {
        "cases": cases,
        "constraints": {
            name: {"digest": item.digest(), "document": item.to_dict()}
            for name, item in constraint_set.items()
        },
        "identity_sentinels": list(SENTINELS),
        "invalidations": invalidations,
        "manifests": {
            name: {"content_id": pb.content_id(data), "text": data.decode("utf-8")}
            for name, data in manifest_bytes.items()
        },
        "normalization": normalization,
        "notice": (
            "Synthetic device-capability and execution-plan vectors. Probes carry "
            "planted SENTINEL identity values that must never reach a profile or "
            "plan. Regenerate with "
            "tests/fixtures/device_capabilities/generate_vectors.py."
        ),
        "profiles": {
            name: {"digest": item.digest(), "document": item.to_dict()}
            for name, item in profile_set.items()
        },
        "schema": "openmed.device_capabilities.vectors",
        "schema_version": 1,
        "synthetic_only": True,
        "vocabulary": vocabulary(),
    }


def render() -> str:
    """Return the vector document as stable, indented JSON text."""

    return json.dumps(build(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    """Write or check ``vectors.json``."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    if args.check:
        current = VECTORS_PATH.read_text(encoding="utf-8")
        if current != text:
            print("device-capability vectors are stale", file=sys.stderr)
            return 1
        return 0
    VECTORS_PATH.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
