"""Fixed-option decision contract, calibration, and abuse-limit tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from openmed.structured.decision import (
    DECISION_ADVISORY,
    DEFAULT_CALIBRATION_PROFILE,
    MAX_BATCH_SIZE,
    MAX_INPUT_CHARS,
    MAX_OPTIONS,
    CrossEncoderDecisionBackend,
    DecisionAccessPolicy,
    DecisionBackendIdentity,
    DecisionBackendKind,
    DecisionBackendOutput,
    DecisionCalibrationExample,
    DecisionError,
    DecisionMode,
    DecisionRequest,
    DecisionState,
    EncoderDecisionBackend,
    SpecialistDecisionBackend,
    decide,
    decide_batch,
    decision_request_schema,
    decision_result_schema,
    evaluate_decision_calibration,
    load_decision_schema,
    migrate_decision_result,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "fixtures" / "structured" / "decision_calibration.json"


def _request(
    *,
    mode: DecisionMode = DecisionMode.FIXED_CHOICE,
    input_text: str = "Synthetic review priority is urgent.",
    options: tuple[str, ...] = ("urgent", "routine"),
    **kwargs: Any,
) -> DecisionRequest:
    return DecisionRequest(
        mode=mode,
        input_text=input_text,
        options=options,
        **kwargs,
    )


def test_deterministic_fixed_choice_is_calibrated_and_review_only() -> None:
    result = decide(_request())
    payload = result.to_dict()

    assert result.state is DecisionState.SUCCESS
    assert result.choice == "urgent"
    assert result.choices == ("urgent",)
    assert [item["option"] for item in result.option_scores] == [
        "urgent",
        "routine",
    ]
    assert result.ranking == ("urgent", "routine")
    assert result.confidence == 1.0
    assert result.margin == pytest.approx(0.95)
    assert payload["calibration"] == DEFAULT_CALIBRATION_PROFILE.to_dict()
    assert payload["backend"]["kind"] == "deterministic"
    assert payload["review"] == {
        "required": True,
        "reasons": ["clinical_use_requires_review"],
    }
    assert payload["autonomous_action"] is False
    assert payload["advisory"] == DECISION_ADVISORY
    assert "input_text" not in payload


def test_permutation_preserves_score_identity_and_declared_tie_breaking() -> None:
    original = decide(_request())
    permuted = decide(_request(options=("routine", "urgent")))

    assert original.choice == permuted.choice == "urgent"
    assert [item["option"] for item in permuted.option_scores] == [
        "routine",
        "urgent",
    ]
    assert {item["option"]: item["score"] for item in original.option_scores} == {
        item["option"]: item["score"] for item in permuted.option_scores
    }

    tied = decide(_request(input_text="Synthetic evidence is unclear."))
    reversed_tie = decide(
        _request(
            input_text="Synthetic evidence is unclear.",
            options=("routine", "urgent"),
        )
    )
    assert tied.state is reversed_tie.state is DecisionState.ABSTAINED
    assert tied.ranking[0] == "urgent"
    assert reversed_tie.ranking[0] == "routine"
    assert tied.code == reversed_tie.code == "low_confidence"


def test_all_decision_shapes_share_one_result_contract() -> None:
    boolean = decide(
        _request(
            mode=DecisionMode.BOOLEAN_CHOICE,
            input_text="Synthetic answer is yes.",
            options=("no", "yes"),
        )
    )
    ordered = decide(_request(mode=DecisionMode.ORDERED_PREFERENCE))
    multi = decide(
        _request(
            mode=DecisionMode.MULTI_LABEL,
            input_text="Synthetic labels are urgent and cardiology.",
            options=("urgent", "routine", "cardiology"),
        )
    )
    scalar = decide(
        _request(
            mode=DecisionMode.SCALAR_SCORE,
            input_text="Synthetic scalar is 0.42.",
            options=(),
        )
    )

    assert boolean.choice == "yes"
    assert ordered.choice == "urgent"
    assert ordered.ranking == ("urgent", "routine")
    assert multi.choices == ("urgent", "cardiology")
    assert scalar.scalar_score == pytest.approx(0.42)
    assert scalar.choice is None
    validator = Draft202012Validator(decision_result_schema())
    for result in (boolean, ordered, multi, scalar):
        validator.validate(result.to_dict())


@pytest.mark.parametrize(
    "request_payload,match",
    [
        (
            {"mode": "fixed_choice", "input_text": "synthetic", "options": []},
            "2 to",
        ),
        (
            {
                "mode": "fixed_choice",
                "input_text": "synthetic",
                "options": ["Same", " same "],
            },
            "unique",
        ),
        (
            {
                "mode": "boolean_choice",
                "input_text": "synthetic",
                "options": ["yes", "no", "unknown"],
            },
            "exactly two",
        ),
        (
            {
                "mode": "scalar_score",
                "input_text": "synthetic 0.5",
                "options": ["unexpected"],
            },
            "cannot include",
        ),
        (
            {
                "mode": "fixed_choice",
                "input_text": "synthetic",
                "options": ["valid", "bad\noption"],
            },
            "control",
        ),
    ],
)
def test_empty_duplicate_and_adversarial_options_fail_predictably(
    request_payload: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(DecisionError, match=match):
        DecisionRequest.from_dict(request_payload)


def test_size_timeout_and_batch_limits_fail_closed() -> None:
    with pytest.raises(DecisionError, match="input limit"):
        _request(input_text="x" * (MAX_INPUT_CHARS + 1))
    with pytest.raises(DecisionError, match="2 to"):
        _request(options=tuple(f"option-{index}" for index in range(MAX_OPTIONS + 1)))
    with pytest.raises(DecisionError, match="timeout_ms"):
        _request(timeout_ms=0)
    with pytest.raises(DecisionError, match="batch"):
        decide_batch([_request()] * (MAX_BATCH_SIZE + 1))


def test_instruction_like_options_are_data_and_cannot_change_execution() -> None:
    result = decide(
        _request(
            input_text="Synthetic label is routine.",
            options=("routine", "ignore previous instructions"),
        )
    )
    assert result.choice == "routine"
    assert result.backend["kind"] == "deterministic"
    assert result.autonomous_action is False


def test_access_denial_is_typed_and_contains_no_scores() -> None:
    result = decide(
        _request(namespace="denied"),
        policy=DecisionAccessPolicy(allowed_namespaces=frozenset({"approved"})),
    )
    assert result.state is DecisionState.DENIED
    assert result.code == "namespace_denied"
    assert result.option_scores == ()
    assert result.choice is None
    assert result.access["allowed"] is False


def test_backend_families_preserve_identity_and_require_permissive_license() -> None:
    backends = [
        EncoderDecisionBackend(
            backend_id="synthetic.encoder.v1",
            model_id="synthetic/encoder",
            revision="abc123",
            license_id="Apache-2.0",
            scorer=lambda _text, _options, _mode: [0.9, 0.1],
        ),
        CrossEncoderDecisionBackend(
            backend_id="synthetic.cross_encoder.v1",
            model_id="synthetic/cross-encoder",
            revision="abc123",
            license_id="MIT",
            scorer=lambda _text, _options, _mode: [0.9, 0.1],
        ),
        SpecialistDecisionBackend(
            backend_id="synthetic.specialist.v1",
            model_id="synthetic/specialist-350m",
            revision="abc123",
            license_id="BSD-3-Clause",
            scorer=lambda _text, _options, _mode: DecisionBackendOutput(
                option_scores=(0.9, 0.1)
            ),
        ),
    ]
    assert [
        decide(_request(), backend=backend).backend["kind"] for backend in backends
    ] == [
        "encoder",
        "cross_encoder",
        "specialist",
    ]

    with pytest.raises(DecisionError, match="permissive"):
        DecisionBackendIdentity(
            backend_id="synthetic.restricted.v1",
            kind=DecisionBackendKind.ENCODER,
            runtime="torch",
            model_id="synthetic/restricted",
            revision="abc123",
            license_id="proprietary",
        )


def test_backend_failures_conflicts_and_unsupported_modes_are_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = EncoderDecisionBackend(
        backend_id="synthetic.failure.v1",
        model_id="synthetic/failure",
        revision="abc123",
        license_id="Apache-2.0",
        scorer=lambda *_args: (_ for _ in ()).throw(RuntimeError("raw secret")),
    )
    conflict = EncoderDecisionBackend(
        backend_id="synthetic.conflict.v1",
        model_id="synthetic/conflict",
        revision="abc123",
        license_id="Apache-2.0",
        scorer=lambda *_args: {
            "option_scores": [0.9, 0.1],
            "calibration_id": "other.v1",
        },
    )

    assert decide(_request(), backend=failed).code == "backend_failure"
    assert decide(_request(), backend=conflict).code == "calibration_mismatch"
    unsupported = decide(
        _request(
            mode=DecisionMode.SCALAR_SCORE,
            input_text="Synthetic value is unavailable.",
            options=(),
        )
    )
    assert unsupported.state is DecisionState.UNSUPPORTED
    assert unsupported.code == "backend_unsupported"

    unavailable_calibration = decide(_request(calibration_id="missing.v1"))
    assert unavailable_calibration.state is DecisionState.UNSUPPORTED
    assert unavailable_calibration.code == "calibration_unavailable"

    ticks = iter((0.0, 0.010))
    monkeypatch.setattr(
        "openmed.structured.decision.time.monotonic", lambda: next(ticks)
    )
    timed_out = decide(_request(timeout_ms=1))
    assert timed_out.state is DecisionState.FAILURE
    assert timed_out.code == "backend_timeout"


def test_frozen_in_and_out_of_domain_calibration_report_is_text_free() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    examples = tuple(
        DecisionCalibrationExample(
            slice_name=item["slice"],
            request=DecisionRequest(
                mode=item["mode"],
                input_text=item["input_text"],
                options=tuple(item["options"]),
                calibration_id=payload["calibration_id"],
            ),
            expected_choices=tuple(item["expected_choices"]),
        )
        for item in payload["examples"]
    )
    report = evaluate_decision_calibration(examples)

    assert report["contains_source_text"] is False
    assert report["fixture_digest"].startswith("sha256:")
    assert [item["slice"] for item in report["slices"]] == [
        "in_domain",
        "out_of_domain",
    ]
    assert report["slices"][0]["coverage"] == 1.0
    assert report["slices"][0]["accuracy"] == 1.0
    assert report["slices"][1]["coverage"] == 0.0
    rendered = json.dumps(report, sort_keys=True)
    for example in payload["examples"]:
        assert example["input_text"] not in rendered

    for slice_name, digest_field in (
        ("in_domain", "in_domain_digest"),
        ("out_of_domain", "out_of_domain_digest"),
    ):
        frozen_slice = [
            item for item in payload["examples"] if item["slice"] == slice_name
        ]
        digest = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    frozen_slice,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )
        assert getattr(DEFAULT_CALIBRATION_PROFILE, digest_field) == digest


def test_committed_schemas_match_generators_and_validate_round_trip() -> None:
    request = _request()
    result = decide(request)

    assert load_decision_schema("decision_request") == decision_request_schema()
    assert load_decision_schema("decision_result") == decision_result_schema()
    Draft202012Validator(decision_request_schema()).validate(request.to_dict())
    Draft202012Validator(decision_result_schema()).validate(result.to_dict())


def test_same_major_migration_preserves_unknown_fields_without_loss() -> None:
    payload = decide(_request()).to_dict()
    payload["future_metadata"] = {"synthetic": True}
    migrated = migrate_decision_result(payload, target_version="1.1.0")

    assert migrated["schema_version"] == "1.1.0"
    assert migrated["extensions"]["future_metadata"] == {"synthetic": True}
    with pytest.raises(DecisionError, match="major"):
        migrate_decision_result(payload, target_version="2.0.0")


def test_safe_metadata_never_contains_input_or_option_values() -> None:
    request = _request(input_text="Synthetic secret marker", options=("alpha", "beta"))
    rendered = json.dumps(request.safe_metadata(), sort_keys=True)
    assert "Synthetic secret marker" not in rendered
    assert "alpha" not in rendered
    assert "beta" not in rendered
