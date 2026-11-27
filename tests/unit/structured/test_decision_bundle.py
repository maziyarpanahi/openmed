"""Decision runtime use of calibration profiles signed inside bundles."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from openmed.core import portable_bundle as pb
from openmed.core import portable_bundle_calibration as pc
from openmed.structured.decision import (
    DEFAULT_CALIBRATION_PROFILE,
    DecisionBackendIdentity,
    DecisionBackendKind,
    DecisionCalibrationProfile,
    DecisionMode,
    DecisionQuestion,
    DecisionRequest,
    decide,
)
from openmed.structured.decision_bundle import (
    bundle_profile_from_decision,
    decision_profile_from_bundle,
    decision_request_scope,
    resolve_bundle_decision_profile,
)

FIXTURES = (
    Path(__file__).resolve().parents[2] / "fixtures" / "portable_bundle_calibration"
)
VECTORS = json.loads((FIXTURES / "vectors.json").read_text(encoding="utf-8"))
GOLDEN = VECTORS["manifests"]["golden"]["text"].encode("utf-8")
GOLDEN_VALUE = json.loads(GOLDEN)
VERIFIED = pb.PortableBundleVerification(
    ok=True,
    reason=None,
    content_id=pb.content_id(GOLDEN),
    valid_signatures=1,
    eligible_runtimes=(1, 2),
    files_checked=len(VECTORS["tree"]),
    files_verified=True,
)
MODEL = GOLDEN_VALUE["model"]


def _generator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "portable_bundle_calibration_vector_generator", FIXTURES / "generate_vectors.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _triage(options: tuple[str, ...] = ("no", "unsure", "yes")) -> DecisionRequest:
    return DecisionRequest(
        DecisionMode.FIXED_CHOICE,
        "Synthetic intake note asks whether a follow-up is needed.",
        options=options,
        calibration_id="openmed.synthetic.triage.v1",
    )


def _backend(revision: str = MODEL["revision"]) -> DecisionBackendIdentity:
    return DecisionBackendIdentity(
        backend_id="synthetic-bundle-backend",
        kind=DecisionBackendKind.ENCODER,
        runtime="onnx",
        model_id=MODEL["id"],
        revision=revision,
        license_id="Apache-2.0",
    )


def test_resolved_profile_drives_decide() -> None:
    profile = resolve_bundle_decision_profile(
        GOLDEN,
        VERIFIED,
        request=_triage(),
        runtime_id="onnx-fp32",
        backend=_backend(),
        profile_digest=VECTORS["golden"]["profile_digests"][0],
    )
    assert isinstance(profile, DecisionCalibrationProfile)
    assert (profile.calibration_id, profile.version) == (
        "openmed.synthetic.triage.v1",
        "1.0.0",
    )
    assert (profile.minimum_confidence, profile.minimum_margin) == (0.7, 0.15)
    result = decide(_triage(), calibration_profiles={profile.calibration_id: profile})
    assert result.calibration["calibration_id"] == "openmed.synthetic.triage.v1"
    assert result.calibration["minimum_confidence"] == 0.7


def test_quantized_runtime_gets_its_own_thresholds() -> None:
    profile = resolve_bundle_decision_profile(
        GOLDEN, VERIFIED, request=_triage(), runtime_id="onnx-int8"
    )
    assert profile.version == "1.1.0"
    assert profile.method == "temperature_scaling"
    assert profile.minimum_confidence == 0.75


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"backend": _backend("f" * 40)}, "calibration_model_mismatch"),
        ({"backend": _backend("main")}, "calibration_model_mismatch"),
        ({"request": _triage(("yes", "no"))}, "calibration_scope_mismatch"),
        ({"runtime_id": "coreml-fp32"}, "calibration_runtime_unbound"),
        (
            {"profile_digest": VECTORS["golden"]["profile_digests"][1]},
            "calibration_pin_mismatch",
        ),
    ],
)
def test_mismatched_profile_fails_closed(kwargs: dict[str, Any], reason: str) -> None:
    arguments: dict[str, Any] = {
        "request": _triage(),
        "runtime_id": "onnx-fp32",
        **kwargs,
    }
    with pytest.raises(pc.DecisionCalibrationError) as error:
        resolve_bundle_decision_profile(GOLDEN, VERIFIED, **arguments)
    assert str(error.value) == reason


def test_unverified_bundle_fails_closed() -> None:
    manifest_only = pb.PortableBundleVerification(
        ok=True, reason=None, content_id=pb.content_id(GOLDEN)
    )
    with pytest.raises(pc.DecisionCalibrationError) as error:
        resolve_bundle_decision_profile(
            GOLDEN, manifest_only, request=_triage(), runtime_id="onnx-fp32"
        )
    assert error.value.reason is pc.DecisionCalibrationReason.BUNDLE_UNVERIFIED


def test_question_set_scope_matches_the_signed_digest() -> None:
    generator = _generator()
    questions = tuple(DecisionQuestion.from_dict(item) for item in generator.QUESTIONS)
    request = DecisionRequest(
        DecisionMode.QUESTION_SET,
        "Synthetic intake note.",
        questions=questions,
        calibration_id="openmed.synthetic.screening.v1",
    )
    kind, digest = decision_request_scope(request)
    assert kind == "questions"
    assert digest == pc.questions_scope_digest(generator.QUESTIONS)
    profile = resolve_bundle_decision_profile(
        GOLDEN, VERIFIED, request=request, runtime_id="onnx-fp32"
    )
    assert profile.method == "platt_scaling"
    reordered = DecisionRequest(
        DecisionMode.QUESTION_SET,
        "Synthetic intake note.",
        questions=tuple(reversed(questions)),
        calibration_id="openmed.synthetic.screening.v1",
    )
    with pytest.raises(pc.DecisionCalibrationError) as error:
        resolve_bundle_decision_profile(
            GOLDEN, VERIFIED, request=reordered, runtime_id="onnx-fp32"
        )
    assert str(error.value) == "calibration_scope_mismatch"


def test_qualified_profile_round_trips_through_a_bundle() -> None:
    generator = _generator()
    bare = generator.bare_manifest()
    entry = bundle_profile_from_decision(
        DEFAULT_CALIBRATION_PROFILE,
        bare,
        runtime_ids=["onnx-fp32"],
        scope_kind="options",
        scope_digest=pc.options_scope_digest(["yes", "no"]),
        dataset_digest=VECTORS["golden"]["profile_digests"][0],
    )
    manifest = pc.with_decision_calibration(bare, [entry])
    data = pb.canonical_json_bytes(manifest)
    verification = pb.PortableBundleVerification(
        ok=True, reason=None, content_id=pb.content_id(data), files_verified=True
    )
    read = pc.read_decision_calibration(data, verification)
    assert read.ok
    assert decision_profile_from_bundle(read.profiles[0]) == DEFAULT_CALIBRATION_PROFILE


def test_inexact_threshold_is_refused() -> None:
    generator = _generator()
    profile = DecisionCalibrationProfile(
        minimum_confidence=0.1234567, minimum_margin=0.1
    )
    with pytest.raises(pc.DecisionCalibrationError) as error:
        bundle_profile_from_decision(
            profile,
            generator.bare_manifest(),
            runtime_ids=["onnx-fp32"],
            scope_kind="labels",
            dataset_digest=VECTORS["golden"]["profile_digests"][0],
        )
    assert str(error.value) == "calibration_malformed"


def test_bridge_rejects_wrong_argument_types() -> None:
    wrong: Any = DEFAULT_CALIBRATION_PROFILE
    with pytest.raises(TypeError):
        decision_profile_from_bundle(wrong)
    with pytest.raises(TypeError):
        decision_request_scope({"mode": "fixed_choice"})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        resolve_bundle_decision_profile(
            GOLDEN,
            VERIFIED,
            request=_triage(),
            runtime_id="onnx-fp32",
            backend={"model_id": "x"},  # type: ignore[arg-type]
        )
