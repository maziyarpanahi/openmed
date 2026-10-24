"""Converters between OpenMed decisions and decision-model wire shapes.

Two request shapes used by decision-model servers are supported:

* **System One** (``POST /v1/systemone``): one ``state`` (text or JSON) and a
  mapping of named questions of type ``choice``, ``noul`` (yes/no) or
  ``score``. Answers carry a probability for every option.
* **OpenAI-compatible Decisions** (``POST /v1/decisions``): one text ``input``
  and a list of ``predicate``, ``choice`` or ``score`` questions.

The mapping onto the OpenMed contract is fixed:

============  =================  ==========================================
System One    OpenAI-compatible  OpenMed question
============  =================  ==========================================
``choice``    ``choice``         ``fixed_choice``; options in declared order
``noul``      ``predicate``      ``boolean_choice`` with ``("true", "false")``
``score``     ``score``          ``scalar_score`` with ordered ``levels``
============  =================  ==========================================

Requests convert both ways without loss for canonical bodies: System One
questions always carry ``instructions`` (``null`` when unset), OpenAI-compatible
questions omit unset ``instructions`` and option descriptions, and option values
and level labels carry no surrounding whitespace. Policy fields that the wire
shapes do not carry (namespace, purpose, calibration and timeout) are supplied
by the caller.

Responses parse into one :class:`~openmed.structured.decision.DecisionBackendOutput`
per question, so OpenMed applies its own calibration and abstention instead of
trusting a server's choice. OpenMed results render back into either shape; a
question that did not succeed is rendered with a ``null`` answer plus its
OpenMed ``state`` and ``code`` and is never given a fabricated answer.

Nothing here performs network I/O, and error messages never echo the input,
instructions, option values or descriptions.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

from .decision import (
    DECISION_SCHEMA_VERSION,
    DEFAULT_CALIBRATION_ID,
    DecisionBackendOutput,
    DecisionConflictError,
    DecisionError,
    DecisionInputFormat,
    DecisionMode,
    DecisionQuestion,
    DecisionQuestionResult,
    DecisionRequest,
    DecisionResult,
    DecisionState,
    DecisionUnsupportedError,
)

BOOLEAN_OPTIONS: Final = ("true", "false")
SYSTEM_ONE_QUESTION_TYPES: Final = ("choice", "noul", "score")
OPENAI_DECISIONS_QUESTION_TYPES: Final = ("predicate", "choice", "score")

_CHOICE: Final = "choice"
_BOOLEAN: Final = "boolean"
_SCORE: Final = "score"
_SYSTEM_ONE_TYPE: Final = {_CHOICE: "choice", _BOOLEAN: "noul", _SCORE: "score"}


def from_system_one_request(
    body: Mapping[str, Any],
    *,
    namespace: str = "default",
    purpose: str = "care_review",
    calibration_id: str = DEFAULT_CALIBRATION_ID,
    timeout_ms: int = 5000,
    max_input_chars: int | None = None,
) -> DecisionRequest:
    """Convert a System One request body into an OpenMed question set.

    Args:
        body: ``{"state": str | dict | list, "questions": {id: question}}``.
        namespace: Access-policy namespace for the OpenMed request.
        purpose: Access-policy purpose for the OpenMed request.
        calibration_id: Calibration profile applied to every question.
        timeout_ms: Backend budget for the whole request.
        max_input_chars: Optional input bound; 8,192 characters by default.

    Returns:
        A ``question_set`` request whose questions keep the body's order.

    Raises:
        DecisionError: If the body is malformed or violates a bound.
        DecisionUnsupportedError: If a question type is not supported.
    """

    payload = _mapping(body, "System One request")
    _only(payload, {"state", "questions"}, "System One request")
    if "state" not in payload or "questions" not in payload:
        raise DecisionError("System One requests require state and questions")
    input_text, input_format = _state_to_input(payload["state"])
    questions = _mapping(payload["questions"], "System One questions")
    return DecisionRequest(
        mode=DecisionMode.QUESTION_SET,
        input_text=input_text,
        namespace=namespace,
        purpose=purpose,
        calibration_id=calibration_id,
        timeout_ms=timeout_ms,
        schema_version=DECISION_SCHEMA_VERSION,
        questions=tuple(
            _system_one_question(question_id, spec)
            for question_id, spec in questions.items()
        ),
        input_format=input_format,
        max_input_chars=max_input_chars,
    )


def to_system_one_request(request: DecisionRequest) -> dict[str, Any]:
    """Render an OpenMed request as a System One request body.

    A single-question request becomes one question named by its
    ``question_id`` or ``decision``.

    Args:
        request: An OpenMed decision request.

    Returns:
        A System One request body.

    Raises:
        DecisionUnsupportedError: If a question has no System One equivalent
            (``ordered_preference``, ``multi_label`` or a ``scalar_score``
            without levels).
    """

    _require_request(request)
    questions: dict[str, Any] = {}
    for question in request.as_questions():
        kind = _question_kind(question)
        entry: dict[str, Any] = {
            "type": _SYSTEM_ONE_TYPE[kind],
            "instructions": question.instructions,
        }
        if kind == _CHOICE:
            entry["criteria"] = dict(
                zip(question.options, _descriptions(question), strict=True)
            )
        elif kind == _BOOLEAN:
            if question.option_descriptions is not None:
                entry["criteria"] = dict(
                    zip(BOOLEAN_OPTIONS, question.option_descriptions, strict=True)
                )
        else:
            entry["criteria"] = list(question.levels)
        questions[question.question_id] = entry
    return {"state": _input_to_state(request), "questions": questions}


def from_system_one_response(
    request: DecisionRequest,
    body: Mapping[str, Any],
) -> dict[str, DecisionBackendOutput]:
    """Parse a System One response into per-question backend outputs.

    Server metadata such as ``model``, ``usage`` or ``probability_source`` is
    ignored. Server-reported ``choice``, ``confidence``, ``level`` and ``score``
    values are checked for shape only; OpenMed recomputes them under its
    calibration profile from the probabilities.

    Score answers differ between published servers, so they are read
    tolerantly: ``legend`` may be ``{"0": label, ...}`` or a label list;
    ``probabilities`` may be keyed by level index or level label, or listed in
    level order; and the expected level may be reported as ``score`` or
    ``expected_level``. Every variant must still match the declared levels.

    Args:
        request: The request the server answered.
        body: ``{"answers": {id: answer}, ...}``.

    Returns:
        One output per question identifier, ready for :func:`decide`.

    Raises:
        DecisionConflictError: If the answers do not match the declared
            questions, options, levels or probability bounds.
    """

    _require_request(request)
    payload = _mapping(body, "System One response", DecisionConflictError)
    answers = _mapping(
        payload.get("answers"), "System One answers", DecisionConflictError
    )
    questions = request.as_questions()
    if set(answers) != {question.question_id for question in questions}:
        raise DecisionConflictError(
            "System One answers do not match the declared questions"
        )
    return {
        question.question_id: _system_one_output(
            question,
            answers[question.question_id],
            request.calibration_id,
        )
        for question in questions
    }


def to_system_one_response(
    request: DecisionRequest,
    result: DecisionResult,
) -> dict[str, Any]:
    """Render an OpenMed result as a System One response body.

    Args:
        request: The request that produced *result*.
        result: Its decision result.

    Returns:
        ``{"model": ..., "answers": {id: answer}}``. Questions that did not
        succeed carry a ``null`` answer plus OpenMed ``state`` and ``code``.
    """

    answers: dict[str, Any] = {}
    for question, outcome in _paired_results(request, result):
        kind = _question_kind(question)
        answer: dict[str, Any] = {"type": _SYSTEM_ONE_TYPE[kind]}
        if kind == _CHOICE:
            answer["choice"] = outcome.choice
            answer["confidence"] = outcome.confidence
            if outcome.option_scores:
                answer["probabilities"] = {
                    item["option"]: item["score"] for item in outcome.option_scores
                }
        elif kind == _BOOLEAN:
            answer["noul"] = _true_score(outcome)
        else:
            answer["legend"] = {
                str(index): level for index, level in enumerate(question.levels)
            }
            answer["score"] = outcome.expected_level
            answer["confidence"] = outcome.confidence
            if outcome.level_scores:
                answer["probabilities"] = {
                    str(item["index"]): item["score"] for item in outcome.level_scores
                }
        _mark_non_success(answer, outcome)
        answers[question.question_id] = answer
    return {"model": result.backend["model_id"], "answers": answers}


def from_openai_decisions_request(
    body: Mapping[str, Any],
    *,
    namespace: str = "default",
    purpose: str = "care_review",
    calibration_id: str = DEFAULT_CALIBRATION_ID,
    timeout_ms: int = 5000,
    max_input_chars: int | None = None,
) -> DecisionRequest:
    """Convert an OpenAI-compatible Decisions request into a question set.

    ``model`` is validated as routing metadata and is not part of the OpenMed
    request; pass it back to :func:`to_openai_decisions_request`.

    Args:
        body: ``{"model": str, "input": str, "questions": [question]}``.
        namespace: Access-policy namespace for the OpenMed request.
        purpose: Access-policy purpose for the OpenMed request.
        calibration_id: Calibration profile applied to every question.
        timeout_ms: Backend budget for the whole request.
        max_input_chars: Optional input bound; 8,192 characters by default.

    Returns:
        A ``question_set`` request whose questions keep the body's order.

    Raises:
        DecisionError: If the body is malformed or violates a bound.
        DecisionUnsupportedError: If the input is not text or a question type
            is not supported.
    """

    payload = _mapping(body, "Decisions request")
    _only(payload, {"model", "input", "questions"}, "Decisions request")
    if not {"model", "input", "questions"} <= set(payload):
        raise DecisionError("Decisions requests require model, input and questions")
    _model_id(payload["model"])
    if not isinstance(payload["input"], str):
        raise DecisionUnsupportedError(
            "Decisions requests support text input only, not messages or media"
        )
    return DecisionRequest(
        mode=DecisionMode.QUESTION_SET,
        input_text=payload["input"],
        namespace=namespace,
        purpose=purpose,
        calibration_id=calibration_id,
        timeout_ms=timeout_ms,
        schema_version=DECISION_SCHEMA_VERSION,
        questions=tuple(
            _openai_question(spec)
            for spec in _sequence(payload["questions"], "Decisions questions")
        ),
        max_input_chars=max_input_chars,
    )


def to_openai_decisions_request(
    request: DecisionRequest,
    *,
    model: str,
) -> dict[str, Any]:
    """Render an OpenMed request as an OpenAI-compatible Decisions body.

    A ``boolean_choice`` question over ``("true", "false")`` without option
    descriptions becomes a ``predicate``; with descriptions it becomes a
    two-option ``choice`` so the descriptions are kept. JSON inputs are sent
    as their JSON text.

    Args:
        request: An OpenMed decision request.
        model: Served model identifier, usually the backend's ``model_id``.

    Returns:
        An OpenAI-compatible Decisions request body.

    Raises:
        DecisionUnsupportedError: If a question has no Decisions equivalent.
    """

    _require_request(request)
    questions = []
    for question in request.as_questions():
        kind = _openai_kind(question)
        entry: dict[str, Any] = {"type": kind, "name": question.question_id}
        if question.instructions is not None:
            entry["instructions"] = question.instructions
        if kind == "choice":
            entry["choices"] = [
                {"value": option}
                if description is None
                else {"value": option, "description": description}
                for option, description in zip(
                    question.options, _descriptions(question), strict=True
                )
            ]
        elif kind == "score":
            entry["levels"] = [{"label": level} for level in question.levels]
        questions.append(entry)
    return {
        "model": _model_id(model),
        "input": request.input_text,
        "questions": questions,
    }


def from_openai_decisions_response(
    request: DecisionRequest,
    body: Mapping[str, Any],
) -> dict[str, DecisionBackendOutput]:
    """Parse an OpenAI-compatible Decisions response into backend outputs.

    The body must list one answer per question under ``answers``; other
    top-level fields are ignored.

    Args:
        request: The request the server answered.
        body: ``{"answers": [answer], ...}``.

    Returns:
        One output per question identifier, ready for :func:`decide`.

    Raises:
        DecisionConflictError: If the answers do not match the declared
            questions, options, levels or probability bounds.
    """

    _require_request(request)
    payload = _mapping(body, "Decisions response", DecisionConflictError)
    by_name: dict[str, Mapping[str, Any]] = {}
    for item in _sequence(
        payload.get("answers"), "Decisions answers", DecisionConflictError
    ):
        answer = _mapping(item, "Decisions answer", DecisionConflictError)
        name = answer.get("name")
        if not isinstance(name, str) or name in by_name:
            raise DecisionConflictError("Decisions answers must have unique names")
        by_name[name] = answer
    questions = request.as_questions()
    if set(by_name) != {question.question_id for question in questions}:
        raise DecisionConflictError(
            "Decisions answers do not match the declared questions"
        )
    return {
        question.question_id: _openai_output(
            question,
            by_name[question.question_id],
            request.calibration_id,
        )
        for question in questions
    }


def to_openai_decisions_response(
    request: DecisionRequest,
    result: DecisionResult,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """Render an OpenMed result as an OpenAI-compatible Decisions response.

    Args:
        request: The request that produced *result*.
        result: Its decision result.
        model: Model identifier to report; the backend ``model_id`` by default.

    Returns:
        ``{"model": ..., "answers": [answer]}`` in question order. Questions
        that did not succeed carry a ``null`` answer plus OpenMed ``state`` and
        ``code``.
    """

    answers = []
    for question, outcome in _paired_results(request, result):
        kind = _openai_kind(question)
        answer: dict[str, Any] = {"type": kind, "name": question.question_id}
        if kind == "predicate":
            answer["probability"] = _true_score(outcome)
        elif kind == "choice":
            answer["choice"] = outcome.choice
            answer["confidence"] = outcome.confidence
            if outcome.option_scores:
                answer["probabilities"] = [
                    {"value": item["option"], "probability": item["score"]}
                    for item in outcome.option_scores
                ]
        else:
            answer["score"] = outcome.expected_level
            answer["confidence"] = outcome.confidence
            if outcome.level_scores:
                answer["probabilities"] = [
                    {
                        "label": item["level"],
                        "value": item["index"],
                        "probability": item["score"],
                    }
                    for item in outcome.level_scores
                ]
        _mark_non_success(answer, outcome)
        answers.append(answer)
    return {
        "model": _model_id(model if model is not None else result.backend["model_id"]),
        "answers": answers,
    }


def _system_one_question(question_id: Any, spec: Any) -> DecisionQuestion:
    question = _mapping(spec, "System One question")
    _only(question, {"type", "instructions", "criteria"}, "System One question")
    kind = question.get("type")
    instructions = question.get("instructions")
    criteria = question.get("criteria")
    if kind == "choice":
        options = _mapping(criteria, "choice criteria")
        return DecisionQuestion(
            question_id=question_id,
            mode=DecisionMode.FIXED_CHOICE,
            options=tuple(_exact_label(item) for item in options),
            option_descriptions=tuple(options.values()),
            instructions=instructions,
        )
    if kind == "noul":
        descriptions = None
        if criteria is not None:
            described = _mapping(criteria, "noul criteria")
            if set(described) != set(BOOLEAN_OPTIONS):
                raise DecisionError("noul criteria must describe true and false")
            descriptions = tuple(described[item] for item in BOOLEAN_OPTIONS)
        return DecisionQuestion(
            question_id=question_id,
            mode=DecisionMode.BOOLEAN_CHOICE,
            options=BOOLEAN_OPTIONS,
            option_descriptions=descriptions,
            instructions=instructions,
        )
    if kind == "score":
        return DecisionQuestion(
            question_id=question_id,
            mode=DecisionMode.SCALAR_SCORE,
            levels=tuple(
                _exact_label(item) for item in _sequence(criteria, "score criteria")
            ),
            instructions=instructions,
        )
    raise DecisionUnsupportedError("System One question type is unsupported")


def _openai_question(spec: Any) -> DecisionQuestion:
    question = _mapping(spec, "Decisions question")
    kind = question.get("type")
    allowed = {"type", "name", "instructions"}
    if kind == "choice":
        allowed.add("choices")
    elif kind == "score":
        allowed.add("levels")
    elif kind != "predicate":
        raise DecisionUnsupportedError("Decisions question type is unsupported")
    _only(question, allowed, "Decisions question")
    name = question.get("name")
    instructions = question.get("instructions")
    if kind == "predicate":
        return DecisionQuestion(
            question_id=name,
            mode=DecisionMode.BOOLEAN_CHOICE,
            options=BOOLEAN_OPTIONS,
            instructions=instructions,
        )
    if kind == "choice":
        options: list[str] = []
        descriptions: list[Any] = []
        for item in _sequence(question.get("choices"), "Decisions choices"):
            choice = _mapping(item, "Decisions choice")
            _only(choice, {"value", "description"}, "Decisions choice")
            if "value" not in choice:
                raise DecisionError("Decisions choices require a value")
            options.append(_exact_label(choice["value"]))
            descriptions.append(choice.get("description"))
        described = any(item is not None for item in descriptions)
        return DecisionQuestion(
            question_id=name,
            mode=DecisionMode.FIXED_CHOICE,
            options=tuple(options),
            option_descriptions=tuple(descriptions) if described else None,
            instructions=instructions,
        )
    levels: list[str] = []
    for item in _sequence(question.get("levels"), "Decisions levels"):
        level = _mapping(item, "Decisions level")
        _only(level, {"label"}, "Decisions level")
        if "label" not in level:
            raise DecisionError("Decisions levels require a label")
        levels.append(_exact_label(level["label"]))
    return DecisionQuestion(
        question_id=name,
        mode=DecisionMode.SCALAR_SCORE,
        levels=tuple(levels),
        instructions=instructions,
    )


def _system_one_output(
    question: DecisionQuestion,
    value: Any,
    calibration_id: str,
) -> DecisionBackendOutput:
    answer = _mapping(value, "System One answer", DecisionConflictError)
    kind = _question_kind(question)
    expected = _SYSTEM_ONE_TYPE[kind]
    if answer.get("type", expected) != expected:
        raise DecisionConflictError("System One answer type does not match")
    if kind == _BOOLEAN:
        probability = _wire_probability(answer.get("noul"))
        return DecisionBackendOutput(
            option_scores=(probability, 1.0 - probability),
            calibration_id=calibration_id,
        )
    _optional_wire_probability(answer.get("confidence"))
    if kind == _CHOICE:
        choice = answer.get("choice")
        if choice is not None and choice not in question.options:
            raise DecisionConflictError("System One choice is not a declared option")
        probabilities = _mapping(
            answer.get("probabilities"),
            "System One probabilities",
            DecisionConflictError,
        )
        return DecisionBackendOutput(
            option_scores=_aligned(probabilities, question.options),
            calibration_id=calibration_id,
        )
    legend = answer.get("legend")
    if legend is not None and _legend_levels(legend, question) != question.levels:
        raise DecisionConflictError("System One legend does not match the levels")
    level = answer.get("level")
    if level is not None and not _is_level_reference(level, question):
        raise DecisionConflictError("System One level is not a declared level")
    probabilities = answer.get("probabilities")
    if probabilities is not None:
        return DecisionBackendOutput(
            option_scores=_level_probabilities(probabilities, question),
            calibration_id=calibration_id,
        )
    return _scalar_output(question, answer, calibration_id)


def _legend_levels(legend: Any, question: DecisionQuestion) -> tuple[Any, ...]:
    """Return a legend's labels in index order.

    Servers publish the legend as ``{"0": label, ...}``; a plain label list in
    index order is accepted too. Anything else cannot match the levels.
    """

    if isinstance(legend, Mapping):
        keys = tuple(str(index) for index in range(len(question.levels)))
        if set(legend) != set(keys):
            return ()
        return tuple(legend[key] for key in keys)
    if isinstance(legend, list):
        return tuple(legend)
    return ()


def _is_level_reference(level: Any, question: DecisionQuestion) -> bool:
    """Return whether a reported ``level`` names a declared level.

    The most likely level is reported as an index, an index string or a label.
    """

    indices = range(len(question.levels))
    if isinstance(level, bool):
        return False
    if isinstance(level, int):
        return level in indices
    return isinstance(level, str) and (
        level in question.levels or level in {str(index) for index in indices}
    )


def _level_probabilities(value: Any, question: DecisionQuestion) -> tuple[float, ...]:
    """Align score-level probabilities with the declared levels.

    Probabilities keyed by level index (``{"0": p, ...}``) are the published
    System One shape. Probabilities keyed by level label, or listed in level
    order, are accepted as well; keys that are exactly the level indices are
    always read as indices.
    """

    if isinstance(value, Mapping):
        keys = tuple(str(index) for index in range(len(question.levels)))
        if set(value) == set(keys):
            return _aligned(value, keys)
        return _aligned(value, question.levels)
    values = _sequence(value, "System One probabilities", DecisionConflictError)
    if len(values) != len(question.levels):
        raise DecisionConflictError("answer probabilities do not match the declaration")
    return tuple(_wire_probability(item) for item in values)


def _openai_output(
    question: DecisionQuestion,
    answer: Mapping[str, Any],
    calibration_id: str,
) -> DecisionBackendOutput:
    kind = _openai_kind(question)
    if answer.get("type", kind) != kind:
        raise DecisionConflictError("Decisions answer type does not match")
    if kind == "predicate":
        probability = _wire_probability(answer.get("probability"))
        return DecisionBackendOutput(
            option_scores=(probability, 1.0 - probability),
            calibration_id=calibration_id,
        )
    _optional_wire_probability(answer.get("confidence"))
    if kind == "choice":
        choice = answer.get("choice")
        if choice is not None and choice not in question.options:
            raise DecisionConflictError("Decisions choice is not a declared option")
        scores: dict[str, Any] = {}
        for item in _sequence(
            answer.get("probabilities"),
            "Decisions probabilities",
            DecisionConflictError,
        ):
            entry = _mapping(item, "Decisions probability", DecisionConflictError)
            option = entry.get("value")
            if not isinstance(option, str) or option in scores:
                raise DecisionConflictError("Decisions probabilities must be unique")
            scores[option] = entry.get("probability")
        return DecisionBackendOutput(
            option_scores=_aligned(scores, question.options),
            calibration_id=calibration_id,
        )
    probabilities = answer.get("probabilities")
    if probabilities is None:
        return _scalar_output(question, answer, calibration_id)
    by_index: dict[str, Any] = {}
    for item in _sequence(
        probabilities, "Decisions probabilities", DecisionConflictError
    ):
        entry = _mapping(item, "Decisions probability", DecisionConflictError)
        index = entry.get("value")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(question.levels)
            or str(index) in by_index
        ):
            raise DecisionConflictError("Decisions level indices are invalid")
        label = entry.get("label")
        if label is not None and label != question.levels[index]:
            raise DecisionConflictError("Decisions level labels do not match")
        by_index[str(index)] = entry.get("probability")
    keys = tuple(str(index) for index in range(len(question.levels)))
    return DecisionBackendOutput(
        option_scores=_aligned(by_index, keys),
        calibration_id=calibration_id,
    )


def _scalar_output(
    question: DecisionQuestion,
    answer: Mapping[str, Any],
    calibration_id: str,
) -> DecisionBackendOutput:
    highest = len(question.levels) - 1
    # Some servers report the expected level as ``expected_level``.
    score = answer.get("score")
    if score is None:
        score = answer.get("expected_level")
    if (
        isinstance(score, bool)
        or not isinstance(score, int | float)
        or not math.isfinite(score)
        or not 0 <= score <= highest
    ):
        raise DecisionConflictError("score answers require probabilities or a level")
    return DecisionBackendOutput(
        scalar_score=float(score) / highest,
        confidence=_optional_wire_probability(answer.get("confidence")),
        calibration_id=calibration_id,
    )


def _question_kind(question: DecisionQuestion) -> str:
    if question.mode is DecisionMode.FIXED_CHOICE:
        return _CHOICE
    if question.mode is DecisionMode.BOOLEAN_CHOICE:
        return _BOOLEAN if question.options == BOOLEAN_OPTIONS else _CHOICE
    if question.mode is DecisionMode.SCALAR_SCORE and question.levels:
        return _SCORE
    raise DecisionUnsupportedError(
        f"{question.mode.value} questions have no decision-server equivalent"
    )


def _openai_kind(question: DecisionQuestion) -> str:
    kind = _question_kind(question)
    if kind == _BOOLEAN:
        return "predicate" if question.option_descriptions is None else "choice"
    return kind


def _paired_results(
    request: DecisionRequest,
    result: DecisionResult,
) -> tuple[tuple[DecisionQuestion, DecisionQuestionResult], ...]:
    _require_request(request)
    if not isinstance(result, DecisionResult):
        raise TypeError("result must be a DecisionResult")
    questions = request.as_questions()
    outcomes = result.question_results(questions[0].question_id)
    if [item.question_id for item in outcomes] != [
        item.question_id for item in questions
    ] or any(
        outcome.mode is not question.mode
        for question, outcome in zip(questions, outcomes, strict=True)
    ):
        raise DecisionError("decision result does not match the request questions")
    return tuple(zip(questions, outcomes, strict=True))


def _mark_non_success(answer: dict[str, Any], outcome: DecisionQuestionResult) -> None:
    if outcome.state is not DecisionState.SUCCESS:
        answer["state"] = outcome.state.value
        answer["code"] = outcome.code


def _true_score(outcome: DecisionQuestionResult) -> float | None:
    for item in outcome.option_scores:
        if item["option"] == BOOLEAN_OPTIONS[0]:
            return item["score"]
    return None


def _descriptions(question: DecisionQuestion) -> tuple[str | None, ...]:
    if question.option_descriptions is None:
        return (None,) * len(question.options)
    return question.option_descriptions


def _state_to_input(state: Any) -> tuple[str, DecisionInputFormat]:
    if isinstance(state, str):
        return state, DecisionInputFormat.TEXT
    if isinstance(state, (Mapping, list)):
        try:
            text = json.dumps(
                state,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError, RecursionError):
            raise DecisionError("System One state must be text or plain JSON") from None
        return text, DecisionInputFormat.JSON
    raise DecisionError("System One state must be text, an object or an array")


def _input_to_state(request: DecisionRequest) -> Any:
    if request.input_format is DecisionInputFormat.JSON:
        return json.loads(request.input_text)
    return request.input_text


def _aligned(scores: Mapping[str, Any], keys: Sequence[str]) -> tuple[float, ...]:
    if set(scores) != set(keys):
        raise DecisionConflictError("answer probabilities do not match the declaration")
    return tuple(_wire_probability(scores[key]) for key in keys)


def _wire_probability(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise DecisionConflictError("answer probabilities must lie in [0, 1]")
    return float(value)


def _optional_wire_probability(value: Any) -> float | None:
    return None if value is None else _wire_probability(value)


def _exact_label(value: Any) -> Any:
    if isinstance(value, str) and value != value.strip():
        raise DecisionError(
            "option values and level labels cannot have surrounding whitespace"
        )
    return value


def _model_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise DecisionError("model must be a bounded identifier")
    return value


def _require_request(request: Any) -> None:
    if not isinstance(request, DecisionRequest):
        raise TypeError("request must be a DecisionRequest")


def _mapping(
    value: Any,
    name: str,
    error: type[DecisionError] = DecisionError,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error(f"{name} must be an object")
    return value


def _sequence(
    value: Any,
    name: str,
    error: type[DecisionError] = DecisionError,
) -> Sequence[Any]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise error(f"{name} must be an array")
    return value


def _only(payload: Mapping[str, Any], allowed: set[str], name: str) -> None:
    if set(payload) - allowed:
        raise DecisionError(f"{name} contains unsupported fields")


__all__ = [
    "BOOLEAN_OPTIONS",
    "OPENAI_DECISIONS_QUESTION_TYPES",
    "SYSTEM_ONE_QUESTION_TYPES",
    "from_openai_decisions_request",
    "from_openai_decisions_response",
    "from_system_one_request",
    "from_system_one_response",
    "to_openai_decisions_request",
    "to_openai_decisions_response",
    "to_system_one_request",
    "to_system_one_response",
]
