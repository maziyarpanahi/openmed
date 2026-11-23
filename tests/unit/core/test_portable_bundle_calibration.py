"""Tests for decision calibration profiles signed inside portable bundles."""

from __future__ import annotations

import copy
import importlib.util
import json
import socket
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from openmed.core import portable_bundle as pb
from openmed.core import portable_bundle_calibration as pc

FIXTURES = (
    Path(__file__).resolve().parents[2] / "fixtures" / "portable_bundle_calibration"
)
VECTORS = json.loads((FIXTURES / "vectors.json").read_text(encoding="utf-8"))
EXT = pc.DECISION_CALIBRATION_EXTENSION
GOLDEN = VECTORS["manifests"]["golden"]["text"].encode("utf-8")
GOLDEN_VALUE = json.loads(GOLDEN)


def _generator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "portable_bundle_calibration_vector_generator", FIXTURES / "generate_vectors.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(key: str) -> bytes:
    entry = VECTORS["manifests"][key]
    if "text" in entry:
        return entry["text"].encode("utf-8")
    base = VECTORS["manifests"][entry["base"]]["text"]
    text = base[: entry["at"]] + entry["insert"] + base[entry["at"] + entry["delete"] :]
    return text.encode("utf-8")


def _verification(raw: dict[str, Any]) -> pb.PortableBundleVerification:
    return pb.PortableBundleVerification(
        ok=raw["ok"],
        reason=raw["reason"],
        content_id=raw["content_id"],
        valid_signatures=raw["valid_signatures"],
        eligible_runtimes=tuple(raw["eligible_runtimes"]),
        files_checked=raw["files_checked"],
        files_verified=raw["files_verified"],
    )


def _tree(case: dict[str, Any] | None = None) -> dict[str, bytes]:
    files = {item["path"]: item["text"].encode("utf-8") for item in VECTORS["tree"]}
    for entry in (case or {}).get("tree_set", []):
        files[entry["path"]] = entry["text"].encode("utf-8")
    return files


def _context(understood: list[str]) -> pb.VerificationContext:
    return pb.VerificationContext(
        now=VECTORS["now"],
        target=pb.VerificationTarget(**VECTORS["target"]),
        understood_extensions=frozenset(understood),
    )


def _policy() -> pb.TrustPolicy:
    return pb.TrustPolicy.from_mapping(VECTORS["trust_policy"])


def _verify(case: dict[str, Any]) -> pb.PortableBundleVerification:
    return pb.verify_portable_bundle(
        _manifest(case["manifest"]),
        case["envelope"].encode("utf-8"),
        trust_policy=_policy(),
        context=_context(case["understood_extensions"]),
        files=_tree(case),
    )


def _golden_verification() -> pb.PortableBundleVerification:
    result = pb.verify_portable_bundle(
        GOLDEN,
        VECTORS["verification_cases"][0]["envelope"].encode("utf-8"),
        trust_policy=_policy(),
        context=_context([EXT]),
        files=_tree(),
    )
    assert result.eligible_for_activation
    return result


def test_committed_vectors_match_the_generator() -> None:
    assert (FIXTURES / "vectors.json").read_text(encoding="utf-8") == (
        _generator().render()
    )


def test_vectors_are_synthetic_test_only_and_share_the_vocabulary() -> None:
    assert VECTORS["synthetic_only"] is True
    assert VECTORS["test_only_keys"] is True
    assert "TEST-ONLY" in VECTORS["notice"]
    assert VECTORS["extension"] == EXT
    assert VECTORS["extension_version"] == pc.DECISION_CALIBRATION_EXTENSION_VERSION
    assert VECTORS["reason_codes"] == list(pc.REASON_CODES)
    for key in VECTORS["trust_policy"]["keys"]:
        assert key["kid"].startswith("openmed-test-")
    for item in VECTORS["tree"]:
        assert "synthetic" in item["text"].lower() or item["path"] == "id2label.json"


@pytest.mark.parametrize(
    "case",
    VECTORS["verification_cases"],
    ids=[case["id"] for case in VECTORS["verification_cases"]],
)
def test_verification_case_matches_reference(case: dict[str, Any]) -> None:
    payload = _verify(case).to_dict()
    payload.pop("schema")
    payload.pop("schema_version")
    assert payload == case["expected"]


@pytest.mark.parametrize(
    "case",
    VECTORS["read_cases"],
    ids=[case["id"] for case in VECTORS["read_cases"]],
)
def test_read_case_matches_reference(case: dict[str, Any]) -> None:
    result = pc.read_decision_calibration(
        _manifest(case["manifest"]), _verification(case["verification"])
    )
    assert {
        "content_id": result.content_id,
        "ok": result.ok,
        "profiles": [
            item.canonical_bytes().decode("utf-8") for item in result.profiles
        ],
        "reason": result.reason,
    } == case["expected"]


@pytest.mark.parametrize(
    "case",
    VECTORS["resolve_cases"],
    ids=[case["id"] for case in VECTORS["resolve_cases"]],
)
def test_resolve_case_matches_reference(case: dict[str, Any]) -> None:
    expected = case["expected"]
    if "error" in expected:
        with pytest.raises(pc.DecisionCalibrationError) as error:
            pc.DecisionCalibrationRequest(**case["request"])
        assert str(error.value) == expected["error"]
        return
    selection = pc.resolve_decision_calibration(
        _manifest(case["manifest"]),
        _verification(case["verification"]),
        pc.DecisionCalibrationRequest(**case["request"]),
    )
    assert {
        "content_id": selection.content_id,
        "ok": selection.ok,
        "profile_digest": (
            None if selection.profile is None else selection.profile.profile_digest
        ),
        "reason": selection.reason,
    } == expected


def test_every_calibration_reason_has_a_vector() -> None:
    covered = {
        case["expected"].get("reason") or case["expected"].get("error")
        for case in VECTORS["read_cases"] + VECTORS["resolve_cases"]
    } - {None}
    assert covered == set(pc.REASON_CODES)
    verification = {
        case["expected"]["reason"] for case in VECTORS["verification_cases"]
    }
    assert "extension_invalid" in verification
    assert None in verification


def test_reason_codes_are_unique_and_value_free() -> None:
    assert len(set(pc.REASON_CODES)) == len(pc.REASON_CODES)
    for code in pc.REASON_CODES:
        assert code.startswith("calibration_")
        assert all(char.isalpha() or char == "_" for char in code)
    assert "extension_invalid" in pb.REASON_CODES
    assert pb.REASON_CODES.index("extension_invalid") == (
        pb.REASON_CODES.index("unknown_critical_extension") + 1
    )
    assert EXT in pb.BUILTIN_EXTENSIONS


def test_golden_digests_and_profiles_round_trip() -> None:
    golden = VECTORS["golden"]
    assert pb.content_id(GOLDEN) == golden["content_id"]
    for runtime_id, digest in golden["runtime_digests"].items():
        assert pc.calibration_runtime_digest(GOLDEN_VALUE, runtime_id) == digest
    profiles = pc.validate_decision_calibration_extension(GOLDEN_VALUE)
    raw = GOLDEN_VALUE["extensions"][EXT]["profiles"]
    assert [item.profile_digest for item in profiles] == golden["profile_digests"]
    for profile, item in zip(profiles, raw, strict=True):
        assert profile.to_dict() == item
        assert pc.calibration_profile_digest(item) == profile.profile_digest
    triage, quantized, screening, labels = profiles
    assert triage.minimum_confidence == 0.7
    assert triage.minimum_margin == 0.15
    assert triage.runtime_ids == ("coreml-fp16", "onnx-fp32")
    assert triage.runtime_digest("onnx-int8") is None
    assert quantized.temperature_ppm == 1250000
    assert quantized.parameters == {"temperature_ppm": 1250000}
    assert (screening.platt_slope_ppm, screening.platt_intercept_ppm) == (
        -1800000,
        250000,
    )
    assert labels.isotonic_points[1] == (250000, 100000)
    assert labels.evaluation_path is None
    assert triage.evaluation_path == "evidence/decision-calibration.json"


def test_schema_accepts_golden_and_rejects_shape_errors() -> None:
    schema = pc.load_decision_calibration_schema()
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    golden = GOLDEN_VALUE["extensions"][EXT]
    validator.validate(golden)
    for edit in (
        lambda value: value.update(schema_version=2),
        lambda value: value.update(profiles=[]),
        lambda value: value["profiles"][0].update(method="histogram_binning"),
        lambda value: value["profiles"][0]["thresholds"].update(
            minimum_confidence_ppm=1000001
        ),
        lambda value: value["profiles"][1]["parameters"].update(temperature_ppm=0),
        lambda value: value["profiles"][0].update(parameters={"temperature_ppm": 1}),
        lambda value: value["profiles"][0]["scope"].update(kind="labelset"),
    ):
        broken = copy.deepcopy(golden)
        edit(broken)
        assert not validator.is_valid(broken)
    manifest_schema = Draft202012Validator(pb.load_portable_bundle_schema())
    manifest_schema.validate(GOLDEN_VALUE)


def test_builder_reproduces_the_golden_manifest() -> None:
    generator = _generator()
    bare = generator.bare_manifest()
    profiles = generator.golden_profiles(bare)
    built = pc.with_decision_calibration(bare, profiles)
    assert pb.encode_portable_manifest(built, understood_extensions=[EXT]) == GOLDEN
    assert built["critical_extensions"] == [EXT]
    noncritical = pc.with_decision_calibration(built, profiles, critical=False)
    assert noncritical["critical_extensions"] == []
    assert pb.PortableBundleManifest.from_mapping(noncritical).bundle_id == (
        "openmed.synthetic.tiny-decision"
    )


def test_builder_refuses_inconsistent_profiles() -> None:
    generator = _generator()
    bare = generator.bare_manifest()
    profiles = generator.golden_profiles(bare)
    stale = copy.deepcopy(profiles)
    stale[0]["thresholds"]["minimum_confidence_ppm"] = 500000
    with pytest.raises(pc.DecisionCalibrationError) as error:
        pc.with_decision_calibration(bare, stale)
    assert error.value.reason is pc.DecisionCalibrationReason.DIGEST_MISMATCH
    with pytest.raises(pc.DecisionCalibrationError) as error:
        pc.with_decision_calibration(bare, [])
    assert str(error.value) == "calibration_malformed"
    with pytest.raises(KeyError):
        pc.build_calibration_profile(
            bare,
            calibration_id="openmed.synthetic.triage.v1",
            version="1.0.0",
            runtime_ids=["tflite-int8"],
            scope_kind="options",
            scope_digest=pc.options_scope_digest(["no", "yes"]),
            decision_schema_min="1.0.0",
            decision_schema_max_exclusive="2.0.0",
            dataset_digest=VECTORS["golden"]["profile_digests"][0],
            in_domain_digest=VECTORS["golden"]["profile_digests"][0],
            out_of_domain_digest=VECTORS["golden"]["profile_digests"][0],
            minimum_confidence_ppm=700000,
            minimum_margin_ppm=150000,
            multi_label_threshold_ppm=700000,
        )


def test_signed_directory_bundle_resolves_and_fails_closed(tmp_path: Path) -> None:
    generator = _generator()
    for path, data in _tree().items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    envelope = pb.sign_portable_manifest(
        GOLDEN, [("openmed-test-root-a", generator.BASE.seed("openmed-test-root-a"))]
    )
    pb.write_portable_bundle(tmp_path, GOLDEN, envelope)
    result = pb.verify_portable_bundle_dir(
        tmp_path, trust_policy=_policy(), context=_context([EXT])
    )
    assert result.eligible_for_activation
    request = pc.DecisionCalibrationRequest(
        calibration_id="openmed.synthetic.triage.v1",
        runtime_id="onnx-int8",
        decision_schema_version="1.1.0",
    )
    selection = pc.resolve_decision_calibration(GOLDEN, result, request)
    assert selection.ok and selection.profile is not None
    assert selection.profile.version == "1.1.0"
    assert selection.profile.method == "temperature_scaling"
    (tmp_path / "onnx" / "model_int8.onnx").write_text("re-exported\n")
    stale = pb.verify_portable_bundle_dir(
        tmp_path, trust_policy=_policy(), context=_context([EXT])
    )
    assert stale.reason == "file_size_mismatch"
    refused = pc.resolve_decision_calibration(GOLDEN, stale, request)
    assert refused.reason == "calibration_bundle_unverified"
    assert refused.profile is None


def test_reader_rejects_wrong_argument_types() -> None:
    with pytest.raises(TypeError):
        pc.read_decision_calibration(GOLDEN, {"ok": True})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        pc.resolve_decision_calibration(
            GOLDEN,
            _golden_verification(),
            {"calibration_id": "x"},  # type: ignore[arg-type]
        )


def test_results_and_errors_are_value_free() -> None:
    sentinel = "OpenMed/SENTINEL-PRIVATE-MODEL"
    manifest = copy.deepcopy(GOLDEN_VALUE)
    profile = manifest["extensions"][EXT]["profiles"][0]
    profile["model"]["id"] = sentinel
    profile["profile_digest"] = pc.calibration_profile_digest(profile)
    data = pb.canonical_json_bytes(manifest)
    verification = pb.PortableBundleVerification(
        ok=True, reason=None, content_id=pb.content_id(data), files_verified=True
    )
    read = pc.read_decision_calibration(data, verification)
    assert read.reason == "calibration_model_mismatch"
    assert sentinel not in repr(read)
    with pytest.raises(pc.DecisionCalibrationError) as error:
        pc.validate_decision_calibration_extension(manifest)
    assert str(error.value) == "calibration_model_mismatch"
    assert sentinel not in repr(error.value)
    with pytest.raises(pc.DecisionCalibrationError) as error:
        pc.DecisionCalibrationRequest(
            calibration_id="openmed.synthetic.triage.v1",
            runtime_id="onnx-fp32",
            decision_schema_version="1.1.0",
            model_revision="SENTINEL-REVISION",
        )
    assert "SENTINEL" not in str(error.value)


def test_reading_and_resolving_never_open_sockets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verification = _golden_verification()

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    for case in VECTORS["verification_cases"]:
        _verify(case)
    assert pc.read_decision_calibration(GOLDEN, verification).ok
    selection = pc.resolve_decision_calibration(
        GOLDEN,
        verification,
        pc.DecisionCalibrationRequest(
            calibration_id="openmed.synthetic.labels.v1",
            runtime_id="onnx-fp32",
            decision_schema_version="1.0.0",
        ),
    )
    assert selection.ok
