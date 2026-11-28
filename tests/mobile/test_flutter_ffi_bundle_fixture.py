"""Offline checks for the Flutter FFI bundle fixture, contracts, and CI lanes."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from openmed.core import portable_bundle as pb

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "js" / "openmedkit-flutter"
FIXTURE_DIR = Path(__file__).with_name("fixtures")
WORKFLOW = ROOT / ".github" / "workflows" / "flutter-ci.yml"

_SPEC = importlib.util.spec_from_file_location(
    "generate_flutter_ffi_bundle",
    FIXTURE_DIR / "generate_flutter_ffi_bundle.py",
)
assert _SPEC is not None and _SPEC.loader is not None
generator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = generator
_SPEC.loader.exec_module(generator)

FIXTURE: dict[str, Any] = json.loads(
    (FIXTURE_DIR / "flutter_ffi_bundle.json").read_text(encoding="utf-8")
)


def _materialize(root: Path) -> Path:
    for item in FIXTURE["tree"]:
        target = root.joinpath(*item["path"].split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(item["text"].encode("utf-8"))
    (root / pb.MANIFEST_FILENAME).write_bytes(FIXTURE["manifest"].encode("utf-8"))
    (root / pb.ENVELOPE_FILENAME).write_bytes(FIXTURE["envelope"].encode("utf-8"))
    return root


def _context(platform: str, os_version: str | None) -> pb.VerificationContext:
    return pb.VerificationContext(
        now=FIXTURE["now"],
        target=pb.VerificationTarget(
            platform=platform,
            sdk_name="openmedkit-flutter",
            sdk_version="4.0.0",
            runtimes=("onnx", "ort") if platform in {"android", "ios"} else ("onnx",),
            os_version=os_version,
        ),
    )


def test_committed_fixture_regenerates_byte_for_byte() -> None:
    committed = (FIXTURE_DIR / "flutter_ffi_bundle.json").read_text(encoding="utf-8")
    assert generator.render() == committed


def test_fixture_is_synthetic_and_test_only() -> None:
    assert FIXTURE["synthetic_only"] is True
    assert FIXTURE["test_only_keys"] is True
    assert "TEST-ONLY" in FIXTURE["notice"]
    assert all(len(item["text"]) < 256 for item in FIXTURE["tree"])
    assert FIXTURE["content_id"] == pb.content_id(FIXTURE["manifest"].encode())


@pytest.mark.parametrize(
    ("platform", "os_version"),
    [
        ("android", "26"),
        ("ios", "13.0"),
        ("linux", None),
        ("macos", None),
        ("windows", None),
    ],
)
def test_python_reference_accepts_the_fixture_for_flutter_targets(
    tmp_path: Path, platform: str, os_version: str | None
) -> None:
    root = _materialize(tmp_path)
    result = pb.verify_portable_bundle_dir(
        root,
        trust_policy=pb.TrustPolicy.from_mapping(FIXTURE["trust_policy"]),
        context=_context(platform, os_version),
    )
    assert result.ok, result.reason
    assert result.eligible_for_activation
    assert result.eligible_runtimes == (0, 1)
    assert result.files_checked == len(FIXTURE["tree"])


def test_python_reference_refuses_a_tampered_fixture_model(tmp_path: Path) -> None:
    root = _materialize(tmp_path)
    model = root / "onnx" / "model_int8.onnx"
    model.write_bytes(model.read_bytes().replace(b"int8", b"INT8"))
    result = pb.verify_portable_bundle_dir(
        root,
        trust_policy=pb.TrustPolicy.from_mapping(FIXTURE["trust_policy"]),
        context=_context("linux", None),
    )
    assert result.reason == "file_digest_mismatch"
    assert not result.eligible_for_activation


def test_dart_reason_vocabulary_matches_the_python_reference() -> None:
    source = (PACKAGE_DIR / "lib" / "src" / "portable_bundle_types.dart").read_text(
        encoding="utf-8"
    )
    block = source.split("const List<String> portableBundleReasons", 1)[1]
    block = block.split("];", 1)[0]
    dart_reasons = re.findall(r"'([a-z_]+)'", block)
    assert dart_reasons == [reason.value for reason in pb.PortableBundleReason]


def test_native_and_dart_abi_versions_agree() -> None:
    header = (PACKAGE_DIR / "native" / "openmed_ffi.h").read_text(encoding="utf-8")
    types = (PACKAGE_DIR / "lib" / "src" / "types.dart").read_text(encoding="utf-8")
    native = re.search(r"#define OPENMED_FFI_ABI_VERSION (\d+)", header)
    dart = re.search(r"const int openMedKitFfiAbiVersion = (\d+);", types)
    assert native is not None and dart is not None
    assert native.group(1) == dart.group(1) == "2"


def _jobs() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def test_fixture_lane_never_claims_a_real_runtime() -> None:
    job = _jobs()["ffi-test"]
    text = json.dumps(job)
    assert "fixture" in job["name"].lower()
    assert "OPENMED_FLUTTER_REAL_BUNDLE_DIR" not in text
    assert "-DOPENMED_FFI_ENABLE_TEST_SESSION=ON" in text
    assert "OPENMED_ONNXRUNTIME_ROOT" not in text
    run_steps = [
        step for step in job["steps"] if "OPENMED_FFI_TEST_SESSION" in str(step)
    ]
    assert run_steps and all(
        step["env"]["OPENMED_FFI_TEST_SESSION"] == "1" for step in run_steps
    )


def test_real_artifact_lane_is_opt_in_and_never_uses_the_fixture() -> None:
    job = _jobs()["real-artifact"]
    text = json.dumps(job)
    assert "opt-in" in job["name"]
    assert "workflow_dispatch" in job["if"]
    assert "inputs.real_artifact" in job["if"]
    assert "-DOPENMED_FFI_ENABLE_TEST_SESSION=OFF" in text
    assert "unset OPENMED_FFI_TEST_SESSION" in text
    assert "OPENMED_FFI_TEST_SESSION: " not in text
    assert "ffi_real_artifact_test.dart" in text
    assert all(
        "${{ inputs." not in str(step.get("run", "")) for step in job["steps"]
    ), "inputs reach shell steps only through environment variables"


def test_real_artifact_suite_refuses_the_fixture_session() -> None:
    source = (
        ROOT / "tests" / "mobile" / "test_flutter_ffi_real_artifact.dart"
    ).read_text(encoding="utf-8")
    assert "OPENMED_FFI_TEST_SESSION" in source
    assert "OpenMedSessionKind.onnxRuntime" in source
    assert "isNot(contains(OpenMedSessionKind.fixture))" in source
