"""System One and OpenAI-compatible Decisions converters."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from openmed.structured.decision import (
    DecisionBackendIdentity,
    DecisionBackendKind,
    DecisionBackendOutput,
    DecisionConflictError,
    DecisionError,
    DecisionInputFormat,
    DecisionMode,
    DecisionRequest,
    DecisionState,
    DecisionUnsupportedError,
    decide,
    decision_request_schema,
    decision_result_schema,
)
from openmed.structured.decision_interop import (
    BOOLEAN_OPTIONS,
    from_openai_decisions_request,
    from_openai_decisions_response,
    from_system_one_request,
    from_system_one_response,
    to_openai_decisions_request,
    to_openai_decisions_response,
    to_system_one_request,
    to_system_one_response,
)

MODEL = "synthetic/decision-model"
SYSTEM_ONE_REQUEST: dict[str, Any] = {
    "state": {
        "ticket": "Synthetic ticket: checkout fails for every customer.",
        "plan": "enterprise",
        "attempts": [1, 2.5, None, True],
    },
    "questions": {
        "priority": {
            "type": "choice",
            "instructions": "Which synthetic priority applies?",
            "criteria": {
                "p1": "Synthetic path fully down",
                "p2": "Synthetic degraded path",
                "p3": None,
            },
        },
        "all_users": {
            "type": "noul",
            "instructions": "The synthetic problem affects every customer.",
            "criteria": {"true": "Affects everyone", "false": "Affects some"},
        },
        "urgency": {
            "type": "score",
            "instructions": None,
            "criteria": ["Low", "Medium", "High"],
        },
    },
}
SYSTEM_ONE_RESPONSE: dict[str, Any] = {
    "model": MODEL,
    "answers": {
        "all_users": {"type": "noul", "noul": 0.875},
        "priority": {
            "type": "choice",
            "choice": "p1",
            "confidence": 0.9,
            "probabilities": {"p1": 0.875, "p2": 0.0625, "p3": 0.0625},
        },
        "urgency": {
            "type": "score",
            "legend": {"0": "Low", "1": "Medium", "2": "High"},
            "score": 1.75,
            "confidence": 0.8125,
            "probabilities": {"0": 0.0625, "1": 0.125, "2": 0.8125},
        },
    },
    "probability_source": "native",
    "usage": {"input_tokens": 12, "output_tokens": 3},
}
OPENAI_REQUEST: dict[str, Any] = {
    "model": MODEL,
    "input": "Synthetic ticket: checkout fails for every customer.",
    "questions": [
        {
            "type": "predicate",
            "name": "outage",
            "instructions": "Is this a synthetic full outage?",
        },
        {
            "type": "choice",
            "name": "team",
            "instructions": "Which synthetic team should take it?",
            "choices": [
                {"value": "payments", "description": "Payments"},
                {"value": "web"},
                {"value": "support", "description": "Customer support"},
            ],
        },
        {
            "type": "score",
            "name": "urgency",
            "levels": [{"label": "Low"}, {"label": "Medium"}, {"label": "High"}],
        },
    ],
}
OPENAI_RESPONSE: dict[str, Any] = {
    "id": "synthetic-decision",
    "model": MODEL,
    "answers": [
        {"type": "predicate", "name": "outage", "probability": 0.875},
        {
            "type": "choice",
            "name": "team",
            "choice": "payments",
            "confidence": 0.8,
            "probabilities": [
                {"value": "payments", "probability": 0.875},
                {"value": "web", "probability": 0.0625},
                {"value": "support", "probability": 0.0625},
            ],
        },
        {
            "type": "score",
            "name": "urgency",
            "score": 1.75,
            "confidence": 0.8125,
            "probabilities": [
                {"label": "Low", "value": 0, "probability": 0.0625},
                {"label": "Medium", "value": 1, "probability": 0.125},
                {"label": "High", "value": 2, "probability": 0.8125},
            ],
        },
    ],
}


class _Bridge:
    """In-process stand-in for a loopback decision server."""

    identity = DecisionBackendIdentity(
        backend_id="synthetic.loopback.v1",
        kind=DecisionBackendKind.SPECIALIST,
        runtime="http",
        model_id=MODEL,
        revision="abc123",
        license_id="Apache-2.0",
    )

    def __init__(
        self,
        render: Callable[[DecisionRequest], Any],
        parse: Callable[[DecisionRequest, Any], Mapping[str, DecisionBackendOutput]],
        response: Any,
    ) -> None:
        self.render = render
        self.parse = parse
        self.response = response
        self.sent: list[Any] = []

    def score(self, request: DecisionRequest) -> DecisionBackendOutput:
        raise AssertionError("question sets use score_questions")

    def score_questions(
        self,
        request: DecisionRequest,
    ) -> Mapping[str, DecisionBackendOutput]:
        self.sent.append(self.render(request))
        return self.parse(request, copy.deepcopy(self.response))


def test_three_question_system_one_request_round_trips_without_loss() -> None:
    request = from_system_one_request(SYSTEM_ONE_REQUEST)

    assert request.mode is DecisionMode.QUESTION_SET
    assert request.input_format is DecisionInputFormat.JSON
    assert [(item.question_id, item.mode) for item in request.questions] == [
        ("priority", DecisionMode.FIXED_CHOICE),
        ("all_users", DecisionMode.BOOLEAN_CHOICE),
        ("urgency", DecisionMode.SCALAR_SCORE),
    ]
    assert request.questions[1].options == BOOLEAN_OPTIONS
    assert request.questions[2].levels == ("Low", "Medium", "High")
    assert to_system_one_request(request) == SYSTEM_ONE_REQUEST
    assert DecisionRequest.from_dict(request.to_dict()) == request
    Draft202012Validator(decision_request_schema()).validate(request.to_dict())

    text_state = {
        "state": "Synthetic plain-text state.",
        "questions": {
            "flag": {"type": "noul", "instructions": None},
            "level": {"type": "score", "instructions": None, "criteria": ["a", "b"]},
        },
    }
    text_request = from_system_one_request(text_state)
    assert text_request.input_format is DecisionInputFormat.TEXT
    assert text_request.questions[0].option_descriptions is None
    assert to_system_one_request(text_request) == text_state


def test_three_question_openai_decisions_request_round_trips_without_loss() -> None:
    request = from_openai_decisions_request(OPENAI_REQUEST)

    assert [(item.question_id, item.mode) for item in request.questions] == [
        ("outage", DecisionMode.BOOLEAN_CHOICE),
        ("team", DecisionMode.FIXED_CHOICE),
        ("urgency", DecisionMode.SCALAR_SCORE),
    ]
    assert request.questions[1].option_descriptions == (
        "Payments",
        None,
        "Customer support",
    )
    assert to_openai_decisions_request(request, model=MODEL) == OPENAI_REQUEST
    assert DecisionRequest.from_dict(request.to_dict()) == request


def test_requests_cross_between_wire_shapes() -> None:
    system_one = from_system_one_request(SYSTEM_ONE_REQUEST)
    as_openai = to_openai_decisions_request(system_one, model=MODEL)

    assert json.loads(as_openai["input"]) == SYSTEM_ONE_REQUEST["state"]
    noul = as_openai["questions"][1]
    assert noul["type"] == "choice"
    assert noul["choices"] == [
        {"value": "true", "description": "Affects everyone"},
        {"value": "false", "description": "Affects some"},
    ]
    assert as_openai["questions"][2] == {
        "type": "score",
        "name": "urgency",
        "levels": [{"label": "Low"}, {"label": "Medium"}, {"label": "High"}],
    }

    single = DecisionRequest(
        mode=DecisionMode.BOOLEAN_CHOICE,
        input_text="Synthetic flag is true.",
        options=BOOLEAN_OPTIONS,
    )
    assert to_system_one_request(single) == {
        "state": "Synthetic flag is true.",
        "questions": {"decision": {"type": "noul", "instructions": None}},
    }
    assert to_openai_decisions_request(single, model=MODEL)["questions"] == [
        {"type": "predicate", "name": "decision"}
    ]


def test_system_one_response_decides_per_question_and_renders_back() -> None:
    request = from_system_one_request(SYSTEM_ONE_REQUEST)
    outputs = from_system_one_response(request, SYSTEM_ONE_RESPONSE)

    assert outputs["priority"].option_scores == (0.875, 0.0625, 0.0625)
    assert outputs["all_users"].option_scores == (0.875, 0.125)
    assert outputs["urgency"].option_scores == (0.0625, 0.125, 0.8125)

    bridge = _Bridge(
        to_system_one_request, from_system_one_response, SYSTEM_ONE_RESPONSE
    )
    result = decide(request, backend=bridge)
    assert bridge.sent == [SYSTEM_ONE_REQUEST]
    assert result.state is DecisionState.SUCCESS
    Draft202012Validator(decision_result_schema()).validate(result.to_dict())

    rendered = to_system_one_response(request, result)
    assert rendered["model"] == MODEL
    expected = SYSTEM_ONE_RESPONSE["answers"]
    answers = rendered["answers"]
    assert list(answers) == ["priority", "all_users", "urgency"]
    assert answers["all_users"] == expected["all_users"]
    assert answers["priority"]["choice"] == "p1"
    assert answers["priority"]["probabilities"] == expected["priority"]["probabilities"]
    assert answers["urgency"]["legend"] == expected["urgency"]["legend"]
    assert answers["urgency"]["score"] == pytest.approx(1.75)
    assert answers["urgency"]["probabilities"] == expected["urgency"]["probabilities"]
    assert from_system_one_response(request, rendered) == outputs


def test_openai_response_decides_per_question_and_renders_back() -> None:
    request = from_openai_decisions_request(OPENAI_REQUEST)
    bridge = _Bridge(
        lambda item: to_openai_decisions_request(item, model=MODEL),
        from_openai_decisions_response,
        OPENAI_RESPONSE,
    )
    result = decide(request, backend=bridge)

    assert bridge.sent == [OPENAI_REQUEST]
    assert result.state is DecisionState.SUCCESS
    rendered = to_openai_decisions_response(request, result)
    expected = OPENAI_RESPONSE["answers"]
    assert rendered["model"] == MODEL
    assert rendered["answers"][0] == expected[0]
    assert rendered["answers"][1]["choice"] == "payments"
    assert rendered["answers"][1]["probabilities"] == expected[1]["probabilities"]
    assert rendered["answers"][2]["score"] == pytest.approx(1.75)
    assert rendered["answers"][2]["probabilities"] == expected[2]["probabilities"]
    assert from_openai_decisions_response(
        request, rendered
    ) == from_openai_decisions_response(request, OPENAI_RESPONSE)


def test_non_success_questions_are_rendered_without_fabricated_answers() -> None:
    request = from_system_one_request(SYSTEM_ONE_REQUEST)
    response = copy.deepcopy(SYSTEM_ONE_RESPONSE)
    response["answers"]["priority"]["probabilities"] = {
        "p1": 0.4,
        "p2": 0.35,
        "p3": 0.25,
    }
    response["answers"]["all_users"]["noul"] = 0.5
    result = decide(
        request,
        backend=_Bridge(to_system_one_request, from_system_one_response, response),
    )
    answers = to_system_one_response(request, result)["answers"]

    assert result.state is DecisionState.PARTIAL
    assert answers["priority"]["choice"] is None
    assert answers["priority"]["state"] == "abstained"
    assert answers["priority"]["code"] == "low_confidence"
    assert answers["all_users"] == {
        "type": "noul",
        "noul": 0.5,
        "state": "abstained",
        "code": "low_confidence",
    }
    assert "state" not in answers["urgency"]

    openai_request = from_openai_decisions_request(OPENAI_REQUEST)
    failed = decide(
        openai_request,
        backend=_Bridge(
            lambda item: to_openai_decisions_request(item, model=MODEL),
            from_openai_decisions_response,
            {"answers": []},
        ),
    )
    rendered = to_openai_decisions_response(openai_request, failed)["answers"]
    assert failed.state is DecisionState.CONFLICT
    assert rendered[0] == {
        "type": "predicate",
        "name": "outage",
        "probability": None,
        "state": "conflict",
        "code": "backend_conflict",
    }
    assert rendered[1]["choice"] is None
    assert "probabilities" not in rendered[1]
    assert rendered[2]["score"] is None


def _system_one_answers(change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    response = copy.deepcopy(SYSTEM_ONE_RESPONSE)
    change(response["answers"])
    return response


@pytest.mark.parametrize(
    "change",
    [
        lambda answers: answers.pop("urgency"),
        lambda answers: answers.update({"extra": {"type": "noul", "noul": 0.5}}),
        lambda answers: answers["all_users"].update({"type": "choice"}),
        lambda answers: answers["all_users"].update({"noul": 1.5}),
        lambda answers: answers["all_users"].update({"noul": None}),
        lambda answers: answers["priority"].update({"choice": "p9"}),
        lambda answers: answers["priority"]["probabilities"].pop("p3"),
        lambda answers: answers["priority"]["probabilities"].update({"p1": True}),
        lambda answers: answers["priority"].update({"confidence": -0.1}),
        lambda answers: answers["urgency"]["legend"].update({"2": "Critical"}),
        lambda answers: answers["urgency"]["probabilities"].update({"3": 0.0}),
        lambda answers: answers["urgency"].update(
            {"probabilities": None, "score": None}
        ),
        lambda answers: answers["urgency"].update({"probabilities": None, "score": 3}),
        lambda answers: answers["urgency"].update({"legend": ["Low", "Medium"]}),
        lambda answers: answers["urgency"].update({"legend": "Low"}),
        lambda answers: answers["urgency"].update({"level": 3}),
        lambda answers: answers["urgency"].update({"level": True}),
        lambda answers: answers["urgency"].update({"level": "Critical"}),
        lambda answers: answers["urgency"].update({"probabilities": [0.5, 0.5]}),
        lambda answers: answers["urgency"].update(
            {"probabilities": {"Low": 0.1, "Medium": 0.2, "Critical": 0.7}}
        ),
        lambda answers: answers["urgency"].update(
            {"probabilities": None, "score": None, "expected_level": 2.5}
        ),
    ],
)
def test_malformed_system_one_answers_are_conflicts(
    change: Callable[[dict[str, Any]], None],
) -> None:
    request = from_system_one_request(SYSTEM_ONE_REQUEST)
    response = _system_one_answers(change)
    with pytest.raises(DecisionConflictError):
        from_system_one_response(request, response)

    result = decide(
        request,
        backend=_Bridge(to_system_one_request, from_system_one_response, response),
    )
    assert result.state is DecisionState.CONFLICT
    assert result.code == "backend_conflict"
    assert all(item.choice is None for item in result.questions)


@pytest.mark.parametrize(
    "urgency",
    [
        # Legend as a label list (shape left open by some model cards).
        {"legend": ["Low", "Medium", "High"]},
        # Probabilities keyed by level label, or listed in level order.
        {"probabilities": {"Low": 0.0625, "Medium": 0.125, "High": 0.8125}},
        {"probabilities": [0.0625, 0.125, 0.8125]},
        # ``level`` and ``expected_level`` instead of ``legend`` and ``score``.
        {
            "legend": None,
            "score": None,
            "confidence": None,
            "level": 2,
            "expected_level": 1.75,
        },
        {"legend": None, "level": "High"},
        {"legend": None, "level": "2"},
    ],
)
def test_published_score_answer_variants_parse_identically(
    urgency: dict[str, Any],
) -> None:
    request = from_system_one_request(SYSTEM_ONE_REQUEST)
    expected = from_system_one_response(request, SYSTEM_ONE_RESPONSE)
    response = _system_one_answers(lambda answers: answers["urgency"].update(urgency))

    outputs = from_system_one_response(request, response)

    assert outputs == expected


def test_score_answers_without_probabilities_use_the_reported_level() -> None:
    request = from_system_one_request(SYSTEM_ONE_REQUEST)
    response = _system_one_answers(
        lambda answers: answers["urgency"].pop("probabilities")
    )
    outputs = from_system_one_response(request, response)
    assert outputs["urgency"].scalar_score == pytest.approx(1.75 / 2)
    assert outputs["urgency"].confidence == 0.8125

    result = decide(
        request,
        backend=_Bridge(to_system_one_request, from_system_one_response, response),
    )
    urgency = result.questions[2]
    assert urgency.expected_level == pytest.approx(1.75)
    assert urgency.level_scores == ()


@pytest.mark.parametrize(
    "answers",
    [
        [],
        [{"type": "predicate", "name": "outage", "probability": 0.5}] * 2,
        [
            {"type": "predicate", "name": "outage", "probability": 0.5},
            {"type": "choice", "name": "team", "probabilities": []},
            {"type": "score", "name": "urgency", "score": 1},
        ],
        [
            {"type": "predicate", "name": "outage", "probability": 0.5},
            {
                "type": "choice",
                "name": "team",
                "probabilities": [
                    {"value": "payments", "probability": 0.5},
                    {"value": "web", "probability": 0.25},
                    {"value": "support", "probability": 0.25},
                ],
            },
            {
                "type": "score",
                "name": "urgency",
                "probabilities": [
                    {"label": "Low", "value": 0, "probability": 0.5},
                    {"label": "High", "value": 1, "probability": 0.5},
                    {"label": "High", "value": 2, "probability": 0.0},
                ],
            },
        ],
        [
            {"type": "predicate", "name": "outage", "probability": 0.5},
            {"type": "choice", "name": "team", "probabilities": []},
            {"type": "score", "name": "urgency", "score": 7},
        ],
    ],
)
def test_malformed_openai_answers_are_conflicts(answers: list[Any]) -> None:
    request = from_openai_decisions_request(OPENAI_REQUEST)
    with pytest.raises(DecisionConflictError):
        from_openai_decisions_response(request, {"answers": answers})


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ({"state": "x"}, DecisionError),
        ({**SYSTEM_ONE_REQUEST, "effort": "fast"}, DecisionError),
        ({"state": 3, "questions": {}}, DecisionError),
        ({"state": "x", "questions": []}, DecisionError),
        ({"state": "x", "questions": {}}, DecisionError),
        (
            {"state": "x", "questions": {"q": {"type": "rank"}}},
            DecisionUnsupportedError,
        ),
        (
            {
                "state": "x",
                "questions": {"q": {"type": "noul", "criteria": {"yes": "y"}}},
            },
            DecisionError,
        ),
        (
            {
                "state": "x",
                "questions": {"q": {"type": "score", "criteria": [" a", "b"]}},
            },
            DecisionError,
        ),
        (
            {"state": "x", "questions": {"bad id": {"type": "noul"}}},
            DecisionError,
        ),
        (
            {"state": {"value": float("nan")}, "questions": {"q": {"type": "noul"}}},
            DecisionError,
        ),
    ],
)
def test_invalid_system_one_requests_fail_closed(
    body: dict[str, Any],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        from_system_one_request(body)


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ({**OPENAI_REQUEST, "temperature": 0}, DecisionError),
        ({key: OPENAI_REQUEST[key] for key in ("input", "questions")}, DecisionError),
        (
            {**OPENAI_REQUEST, "input": [{"type": "input_text", "text": "x"}]},
            DecisionUnsupportedError,
        ),
        (
            {**OPENAI_REQUEST, "questions": [{"type": "rank", "name": "q"}]},
            DecisionUnsupportedError,
        ),
        (
            {
                **OPENAI_REQUEST,
                "questions": [{"type": "predicate", "name": "q", "choices": []}],
            },
            DecisionError,
        ),
        (
            {
                **OPENAI_REQUEST,
                "questions": [
                    {"type": "score", "name": "q", "levels": [{"label": "a", "x": 1}]}
                ],
            },
            DecisionError,
        ),
    ],
)
def test_invalid_openai_requests_fail_closed(
    body: dict[str, Any],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        from_openai_decisions_request(body)


@pytest.mark.parametrize(
    "request_payload",
    [
        {
            "mode": "ordered_preference",
            "input_text": "Synthetic ranking.",
            "options": ["a", "b"],
        },
        {
            "mode": "multi_label",
            "input_text": "Synthetic labels.",
            "options": ["a", "b"],
        },
        {"mode": "scalar_score", "input_text": "Synthetic scalar 0.5."},
    ],
)
def test_modes_without_a_wire_equivalent_are_unsupported(
    request_payload: dict[str, Any],
) -> None:
    request = DecisionRequest.from_dict(request_payload)
    with pytest.raises(DecisionUnsupportedError):
        to_system_one_request(request)
    with pytest.raises(DecisionUnsupportedError):
        to_openai_decisions_request(request, model=MODEL)


def test_converter_errors_never_echo_input_or_question_text() -> None:
    sentinel = "Synthetic private sentinel"
    failures: list[Exception] = []
    for body in (
        {
            "state": sentinel,
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": sentinel,
                    "criteria": {sentinel: sentinel},
                }
            },
        },
        {"state": sentinel, "questions": {sentinel: {"type": "noul"}}},
        {"state": sentinel, "questions": {"q": {"type": sentinel}}},
    ):
        with pytest.raises(DecisionError) as caught:
            from_system_one_request(body)
        failures.append(caught.value)
    request = from_system_one_request(SYSTEM_ONE_REQUEST)
    with pytest.raises(DecisionConflictError) as caught:
        from_system_one_response(
            request,
            _system_one_answers(
                lambda answers: answers["priority"].update({"choice": sentinel})
            ),
        )
    failures.append(caught.value)

    for failure in failures:
        assert sentinel not in str(failure)


def test_render_requires_a_matching_result() -> None:
    request = from_system_one_request(SYSTEM_ONE_REQUEST)
    other = decide(
        DecisionRequest(
            mode=DecisionMode.FIXED_CHOICE,
            input_text="Synthetic priority is urgent.",
            options=("urgent", "routine"),
        )
    )
    with pytest.raises(DecisionError, match="does not match"):
        to_system_one_response(request, other)
    with pytest.raises(TypeError):
        to_system_one_request(SYSTEM_ONE_REQUEST)  # type: ignore[arg-type]
