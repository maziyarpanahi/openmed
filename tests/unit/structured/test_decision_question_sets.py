"""Question sets, per-question abstention and text-free audit output."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from openmed.structured.decision import (
    DEFAULT_CALIBRATION_PROFILE,
    MAX_QUESTIONS,
    DecisionAccessPolicy,
    DecisionBackendIdentity,
    DecisionBackendKind,
    DecisionBackendOutput,
    DecisionCalibrationProfile,
    DecisionError,
    DecisionMode,
    DecisionQuestion,
    DecisionQuestionSetBackend,
    DecisionRequest,
    DecisionState,
    EncoderDecisionBackend,
    decide,
    decision_audit_record,
    decision_request_schema,
    decision_result_schema,
)

INPUT_SENTINEL = "Synthetic sentinel input about urgent cardiology review."
INSTRUCTION_SENTINEL = "Synthetic sentinel instruction wording."
DESCRIPTION_SENTINEL = "Synthetic sentinel option description."
IDENTITY = DecisionBackendIdentity(
    backend_id="synthetic.question_set.v1",
    kind=DecisionBackendKind.SPECIALIST,
    runtime="python",
    model_id="synthetic/question-set",
    revision="abc123",
    license_id="Apache-2.0",
)


def _questions() -> list[dict[str, Any]]:
    return [
        {
            "question_id": "priority",
            "mode": "fixed_choice",
            "options": ["urgent", "routine", "deferred"],
            "option_descriptions": [DESCRIPTION_SENTINEL, None, None],
            "instructions": INSTRUCTION_SENTINEL,
        },
        {
            "question_id": "specialty",
            "mode": "boolean_choice",
            "options": ["true", "false"],
        },
        {
            "question_id": "severity",
            "mode": "scalar_score",
            "levels": ["Low", "Medium", "High"],
            "instructions": "Synthetic severity scale.",
        },
    ]


def _request(**overrides: Any) -> DecisionRequest:
    payload: dict[str, Any] = {
        "mode": "question_set",
        "input_text": INPUT_SENTINEL,
        "questions": _questions(),
    }
    payload.update(overrides)
    return DecisionRequest.from_dict(payload)


class _SetBackend:
    """Fake backend that scores a whole question set in one call."""

    identity = IDENTITY

    def __init__(self, outputs: Any) -> None:
        self.outputs = outputs
        self.calls = 0

    def score(self, request: DecisionRequest) -> DecisionBackendOutput:
        raise AssertionError("question sets must use score_questions")

    def score_questions(self, request: DecisionRequest) -> Mapping[str, Any]:
        self.calls += 1
        if isinstance(self.outputs, Exception):
            raise self.outputs
        return self.outputs


def _output(*scores: float, **kwargs: Any) -> DecisionBackendOutput:
    return DecisionBackendOutput(option_scores=scores, **kwargs)


def test_question_set_request_shape_and_schema() -> None:
    request = _request()

    assert request.mode is DecisionMode.QUESTION_SET
    assert request.schema_version == "1.1.0"
    assert [item.question_id for item in request.questions] == [
        "priority",
        "specialty",
        "severity",
    ]
    assert request.questions[0].option_descriptions == (
        DESCRIPTION_SENTINEL,
        None,
        None,
    )
    assert DecisionRequest.from_dict(request.to_dict()) == request
    Draft202012Validator(decision_request_schema()).validate(request.to_dict())
    sub_request = request.for_question(request.questions[2])
    assert sub_request.mode is DecisionMode.SCALAR_SCORE
    assert sub_request.levels == ("Low", "Medium", "High")
    assert sub_request.input_text == INPUT_SENTINEL
    assert sub_request.question_id == "severity"


def test_one_abstaining_or_failing_question_does_not_hide_the_others() -> None:
    backend = _SetBackend(
        {
            "priority": _output(0.9, 0.05, 0.05),
            "specialty": _output(0.55, 0.45),
            "severity": _output(0.1, 0.2),
        }
    )
    assert isinstance(backend, DecisionQuestionSetBackend)

    result = decide(_request(), backend=backend)
    payload = result.to_dict()
    by_id = {item.question_id: item for item in result.questions}

    assert backend.calls == 1
    assert result.state is DecisionState.PARTIAL
    assert result.code == "question_set_partial"
    assert by_id["priority"].state is DecisionState.SUCCESS
    assert by_id["priority"].choice == "urgent"
    assert by_id["specialty"].state is DecisionState.ABSTAINED
    assert by_id["specialty"].code == "low_confidence"
    assert by_id["specialty"].choice is None
    assert [item["score"] for item in by_id["specialty"].option_scores] == [
        0.55,
        0.45,
    ]
    assert by_id["severity"].state is DecisionState.FAILURE
    assert by_id["severity"].code == "backend_contract_invalid"
    assert payload["autonomous_action"] is False
    assert payload["review"]["required"] is True
    assert payload["choice"] is None
    Draft202012Validator(decision_result_schema()).validate(payload)


def test_each_question_uses_the_shared_calibration_profile() -> None:
    strict = DecisionCalibrationProfile(
        calibration_id="synthetic.strict.v1",
        minimum_confidence=0.95,
        minimum_margin=0.5,
    )
    outputs = {
        "priority": _output(0.96, 0.02, 0.02, calibration_id="synthetic.strict.v1"),
        "specialty": _output(0.9, 0.1, calibration_id="synthetic.strict.v1"),
        "severity": _output(0.0, 0.01, 0.99, calibration_id="other.v1"),
    }
    result = decide(
        _request(calibration_id="synthetic.strict.v1"),
        backend=_SetBackend(outputs),
        calibration_profiles={"synthetic.strict.v1": strict},
    )
    states = {item.question_id: (item.state, item.code) for item in result.questions}

    assert result.calibration == strict.to_dict()
    assert states == {
        "priority": (DecisionState.SUCCESS, None),
        "specialty": (DecisionState.ABSTAINED, "low_confidence"),
        "severity": (DecisionState.CONFLICT, "calibration_mismatch"),
    }

    default = decide(
        _request(),
        backend=_SetBackend(
            {
                "priority": _output(0.96, 0.02, 0.02),
                "specialty": _output(0.9, 0.1),
                "severity": _output(0.0, 0.01, 0.99),
            }
        ),
    )
    assert default.state is DecisionState.SUCCESS
    assert default.calibration == DEFAULT_CALIBRATION_PROFILE.to_dict()
    severity = default.questions[2]
    assert severity.expected_level == pytest.approx(1.99)
    assert severity.scalar_score == pytest.approx(0.995)


def test_per_question_fallback_scores_each_question_independently() -> None:
    seen: list[tuple[tuple[str, ...], DecisionMode]] = []

    def scorer(
        _text: str,
        options: tuple[str, ...],
        mode: DecisionMode,
    ) -> list[float]:
        seen.append((options, mode))
        if mode is DecisionMode.BOOLEAN_CHOICE:
            raise RuntimeError("synthetic failure")
        if mode is DecisionMode.SCALAR_SCORE:
            return [0.05, 0.05, 0.9]
        return [0.9, 0.05, 0.05]

    backend = EncoderDecisionBackend(
        backend_id="synthetic.encoder.v1",
        model_id="synthetic/encoder",
        revision="abc123",
        license_id="Apache-2.0",
        scorer=scorer,
    )
    result = decide(_request(), backend=backend)

    assert seen == [
        (("urgent", "routine", "deferred"), DecisionMode.FIXED_CHOICE),
        (("true", "false"), DecisionMode.BOOLEAN_CHOICE),
        (("Low", "Medium", "High"), DecisionMode.SCALAR_SCORE),
    ]
    assert [(item.state, item.code) for item in result.questions] == [
        (DecisionState.SUCCESS, None),
        (DecisionState.FAILURE, "backend_failure"),
        (DecisionState.SUCCESS, None),
    ]
    assert result.state is DecisionState.PARTIAL


def test_deterministic_backend_scores_question_sets_per_question() -> None:
    result = decide(
        {
            "mode": "question_set",
            "input_text": "Synthetic priority is urgent. Synthetic severity is high.",
            "questions": [
                {
                    "question_id": "priority",
                    "mode": "fixed_choice",
                    "options": ["urgent", "routine"],
                },
                {
                    "question_id": "severity",
                    "mode": "scalar_score",
                    "levels": ["Low", "High"],
                },
            ],
        }
    )

    assert result.state is DecisionState.SUCCESS
    assert result.code is None
    assert result.questions[0].choice == "urgent"
    assert result.questions[1].expected_level == pytest.approx(1.0 / 1.05)


@pytest.mark.parametrize(
    ("outputs", "state", "code"),
    [
        (DecisionError("synthetic"), DecisionState.FAILURE, "backend_failure"),
        (TimeoutError(), DecisionState.FAILURE, "backend_timeout"),
        ({"priority": _output(0.9, 0.05, 0.05)}, DecisionState.FAILURE, None),
        (["not", "a", "mapping"], DecisionState.FAILURE, None),
    ],
)
def test_whole_set_backend_failures_report_every_question(
    outputs: Any,
    state: DecisionState,
    code: str | None,
) -> None:
    result = decide(_request(), backend=_SetBackend(outputs))
    expected_code = code or "backend_contract_invalid"

    assert result.state is state
    assert result.code == expected_code
    assert [item.question_id for item in result.questions] == [
        "priority",
        "specialty",
        "severity",
    ]
    assert {(item.state, item.code) for item in result.questions} == {
        (state, expected_code)
    }
    Draft202012Validator(decision_result_schema()).validate(result.to_dict())


def test_denial_and_missing_calibration_report_every_question() -> None:
    denied = decide(
        _request(namespace="restricted"),
        policy=DecisionAccessPolicy(allowed_namespaces=frozenset({"approved"})),
    )
    unavailable = decide(_request(calibration_id="missing.v1"))

    for result, state, code in (
        (denied, DecisionState.DENIED, "namespace_denied"),
        (unavailable, DecisionState.UNSUPPORTED, "calibration_unavailable"),
    ):
        assert result.state is state
        assert result.code == code
        assert len(result.questions) == 3
        assert all(item.state is state for item in result.questions)
        assert all(not item.option_scores for item in result.questions)


def test_question_set_timeout_budget_covers_the_whole_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((0.0, 0.0, 0.001, 0.010, 0.010))
    monkeypatch.setattr(
        "openmed.structured.decision.time.monotonic", lambda: next(ticks)
    )
    result = decide(_request(timeout_ms=5))

    assert [(item.state, item.code) for item in result.questions][1:] == [
        (DecisionState.FAILURE, "backend_timeout"),
        (DecisionState.FAILURE, "backend_timeout"),
    ]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda q: q + [dict(q[0])], "unique"),
        (lambda q: [{**q[0], "question_id": "bad id"}], "question identifier"),
        (lambda q: [{**q[0], "mode": "question_set"}], "nested"),
        (lambda q: [{**q[0], "option_descriptions": [None]}], "align"),
        (lambda q: [{**q[1], "levels": ["Low", "High"]}], "scalar_score"),
        (lambda q: [{**q[2], "levels": ["Low"]}], "2 to"),
        (lambda q: [{**q[2], "levels": ["Low", " low "]}], "unique"),
        (lambda q: [{**q[0], "instructions": "   "}], "non-blank"),
        (lambda q: [{**q[0], "instructions": "bad\x00text"}], "control"),
        (lambda q: [{**q[0], "unexpected": True}], "unsupported"),
        (lambda q: [], "1 to"),
        (lambda q: [dict(q[0], question_id=f"q{i}") for i in range(33)], "1 to"),
        (
            lambda q: [
                {
                    **q[0],
                    "question_id": f"q{i}",
                    "option_descriptions": ["x" * 1000] * 3,
                }
                for i in range(30)
            ],
            "aggregate",
        ),
    ],
)
def test_invalid_question_sets_fail_closed(change: Any, match: str) -> None:
    with pytest.raises((DecisionError, TypeError), match=match):
        _request(questions=change(_questions()))


def test_question_set_rejects_top_level_question_fields() -> None:
    with pytest.raises(DecisionError, match="inside questions"):
        _request(options=["urgent", "routine"])
    question = DecisionQuestion(
        "priority",
        DecisionMode.FIXED_CHOICE,
        options=("urgent", "routine"),
    )
    with pytest.raises(DecisionError, match="question_set"):
        DecisionRequest(
            mode=DecisionMode.FIXED_CHOICE,
            input_text=INPUT_SENTINEL,
            options=("urgent", "routine"),
            questions=(question,),
        )
    assert MAX_QUESTIONS == 32


def test_safe_metadata_and_audit_record_are_text_free() -> None:
    request = _request()
    result = decide(
        request,
        backend=_SetBackend(
            {
                "priority": _output(0.9, 0.05, 0.05),
                "specialty": _output(0.55, 0.45),
                "severity": _output(0.0, 0.1, 0.9),
            }
        ),
    )
    metadata = request.safe_metadata()
    record = decision_audit_record(request, result)
    rendered = json.dumps([metadata, record], sort_keys=True)

    for sentinel in (
        INPUT_SENTINEL,
        INSTRUCTION_SENTINEL,
        DESCRIPTION_SENTINEL,
        "urgent",
        "routine",
        "Medium",
    ):
        assert sentinel not in rendered
    assert record["contains_source_text"] is False
    assert [item["question_id"] for item in metadata["questions"]] == [
        "priority",
        "specialty",
        "severity",
    ]
    assert [item["mode"] for item in metadata["questions"]] == [
        "fixed_choice",
        "boolean_choice",
        "scalar_score",
    ]
    assert [item["option_count"] for item in metadata["questions"]] == [3, 2, 0]
    assert metadata["questions"][2]["level_count"] == 3
    assert all(
        item["definition_digest"].startswith("sha256:")
        for item in metadata["questions"]
    )
    assert record["result"]["questions"][1] == {
        "question_id": "specialty",
        "mode": "boolean_choice",
        "state": "abstained",
        "code": "low_confidence",
        "option_count": 2,
        "level_count": 0,
    }
    changed = _request(
        questions=[{**_questions()[0], "instructions": "Other wording."}]
    )
    assert (
        changed.safe_metadata()["questions"][0]["definition_digest"]
        != metadata["questions"][0]["definition_digest"]
    )
