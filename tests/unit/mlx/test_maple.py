"""Tests for Maple's bounded clinical task contract."""

from __future__ import annotations

import ast
import inspect

import pytest

from openmed.mlx import maple as maple_module
from openmed.mlx.maple import (
    MAPLE_MEDICAL_DISCLAIMER,
    MAPLE_RESPONSE_ERROR_CODES,
    MapleClinicalAssistant,
    MapleRelation,
    MapleResponseError,
    MapleSpan,
    MapleTask,
    build_maple_task_messages,
    parse_maple_task_response,
    redact_maple_spans,
    visible_maple_response,
)


def test_build_messages_preserves_exact_source_and_offset_contract():
    source = "Patient A\u0301da called 555-0100."

    messages = build_maple_task_messages("pii_removal", source)

    assert messages[0]["role"] == "system"
    assert "Unicode-scalar offsets" in messages[0]["content"]
    assert messages[1]["content"].endswith(source)
    assert "discards" in messages[0]["content"]
    assert "after validation" in messages[0]["content"]


def test_parse_pii_snaps_grapheme_and_redacts_without_surface_metadata():
    source = "Patient A\u0301da called 555-0100."
    name_start = source.index("A")
    phone_start = source.index("555")
    response = (
        "<think>private scratch text</think>\n"
        '{"spans":['
        f'{{"start":{name_start},"end":{name_start + 1},"label":"NAME","text":"Á"}},'
        f'{{"start":{phone_start},"end":{phone_start + 8},"label":"PHONE","text":"555-0100"}}'
        "]}"
    )

    result = parse_maple_task_response(MapleTask.PII, response, source)

    assert result.entities[0].surface(source) == "A\u0301"
    assert result.redacted_text == "Patient [NAME]da called [PHONE]."
    assert result.entities[0].to_dict() == {
        "start": name_start,
        "end": name_start + 2,
        "label": "NAME",
    }
    assert "A\u0301" not in repr(result.entities[0].to_dict())
    assert result.review_required is True
    assert result.disclaimer == MAPLE_MEDICAL_DISCLAIMER


def test_parse_relation_response_validates_entity_references():
    source = "Aspirin treats pain."
    response = (
        '{"entities":['
        '{"start":0,"end":7,"label":"medication","text":"Aspirin"},'
        '{"start":15,"end":19,"label":"condition","text":"pain"}'
        '],"relations":[{"source":0,"target":1,"label":"treats"}]}'
    )

    result = parse_maple_task_response("relation_extraction", response, source)

    assert [entity.label for entity in result.entities] == [
        "MEDICATION",
        "CONDITION",
    ]
    assert result.relations[0].to_dict() == {
        "source": 0,
        "target": 1,
        "label": "TREATS",
    }


def test_parse_relation_response_resolves_exact_text_endpoints():
    response = (
        '{"entities":['
        '{"label":"MEDICATION","text":"Aspirin"},'
        '{"label":"CONDITION","text":"pain"}'
        '],"relations":['
        '{"source":"Aspirin","target":"pain","label":"TREATS"}'
        "]}"
    )

    result = parse_maple_task_response("relations", response, "Aspirin treats pain")

    assert result.relations == (MapleRelation(0, 1, "TREATS"),)


def test_parse_relation_response_rejects_unknown_entity_reference():
    response = (
        '{"entities":[{"start":0,"end":7,"label":"MEDICATION","text":"Aspirin"}],'
        '"relations":[{"source":0,"target":2,"label":"TREATS"}]}'
    )

    with pytest.raises(MapleResponseError, match="unknown entity"):
        parse_maple_task_response("relations", response, "Aspirin")


def test_parse_reasoning_response_keeps_answer_and_bounded_evidence():
    source = "[NAME] takes aspirin daily."
    response = (
        "private analysis that is ignored</think>\n"
        '{"answer":"The note reports daily aspirin use.",'
        '"uncertainties":["Dose is not documented."],'
        '"evidence":[{"start":13,"end":26,"text":"aspirin daily"}]}'
    )

    result = parse_maple_task_response("reasoning", response, source)

    assert result.answer == "The note reports daily aspirin use."
    assert result.uncertainties == ("Dose is not documented.",)
    assert result.evidence == (MapleSpan(13, 26, "EVIDENCE"),)


def test_parser_rejects_unrequested_keys_and_out_of_bounds_spans():
    with pytest.raises(MapleResponseError, match="unexpected copied_text"):
        parse_maple_task_response(
            "pii",
            '{"spans":[],"copied_text":"secret"}',
            "secret",
        )

    with pytest.raises(MapleResponseError, match="not allowed"):
        parse_maple_task_response(
            "pii",
            '{"spans":[{"label":"CONDITION","text":"asthma"}]}',
            "asthma",
        )

    with pytest.raises(MapleResponseError, match="not allowed"):
        parse_maple_task_response(
            "relations",
            '{"entities":['
            '{"label":"MEDICATION","text":"Aspirin"},'
            '{"label":"CONDITION","text":"pain"}'
            '],"relations":['
            '{"source":"Aspirin","target":"pain","label":"INVENTED"}'
            "]}",
            "Aspirin treats pain",
        )

    with pytest.raises(MapleResponseError, match="one exact"):
        parse_maple_task_response(
            "entities",
            '{"spans":[{"start":0,"end":99,"label":"CONDITION","text":"other"}]}',
            "short",
        )


def test_parser_repairs_unique_surface_offset_and_rejects_ambiguous_repair():
    repaired = parse_maple_task_response(
        "entities",
        '{"spans":[{"start":0,"end":1,"label":"MEDICATION","text":"metformin"}]}',
        "Takes metformin daily.",
    )

    assert repaired.entities == (MapleSpan(6, 15, "MEDICATION"),)

    with pytest.raises(MapleResponseError, match="one exact"):
        parse_maple_task_response(
            "entities",
            '{"spans":[{"start":99,"end":100,"label":"MEDICATION","text":"aspirin"}]}',
            "aspirin then aspirin",
        )


def test_parser_derives_text_only_offsets_in_document_order():
    result = parse_maple_task_response(
        "entities",
        '{"spans":['
        '{"label":"MEDICATION","text":"aspirin"},'
        '{"label":"MEDICATION","text":"aspirin"}'
        "]}",
        "aspirin then aspirin",
    )

    assert result.entities == (
        MapleSpan(0, 7, "MEDICATION"),
        MapleSpan(13, 20, "MEDICATION"),
    )


def test_overlapping_redactions_merge_without_exposing_original_text():
    source = "Casey Example"
    spans = (MapleSpan(0, 5, "NAME"), MapleSpan(0, 13, "OTHER_PII"))

    assert redact_maple_spans(source, spans) == "[PII]"


def test_visible_response_never_exposes_hidden_reasoning():
    assert visible_maple_response("<think>secret chain</think>Final answer") == (
        "Final answer"
    )
    assert visible_maple_response("<think>unfinished secret") == ""
    assert visible_maple_response("unfinished implicit reasoning") == ""


def test_parser_rejects_truncated_implicit_reasoning_with_schema_example():
    response = (
        'I should return {"spans":[{"start":0,"end":1,"label":"NAME","text":"A"}]} '
        "after I finish checking offsets"
    )

    with pytest.raises(MapleResponseError, match="before emitting"):
        parse_maple_task_response("pii", response, "A")


def test_assistant_uses_injected_runner_for_structured_task_and_chat():
    class Runner:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return '{"spans":[{"start":0,"end":5,"label":"NAME","text":"Casey"}]}'
            return "<think>hidden</think>Visible answer"

    runner = Runner()
    assistant = MapleClinicalAssistant(runner=runner)

    result = assistant.complete_task("pii", "Casey called.")
    answer = assistant.chat([{"role": "user", "content": "Summarise safely."}])

    assert result.redacted_text == "[NAME] called."
    assert runner.calls[0]["temp"] == 0.0
    assert runner.calls[0]["max_tokens"] == 1_024
    assert answer == "Visible answer"
    assert runner.calls[1]["messages"][0]["role"] == "system"


def test_non_reasoning_tasks_reject_question(subtests):
    for task in ["pii", "entities", "relations"]:
        with subtests.test(task=task):
            with pytest.raises(ValueError, match="only accepted"):
                build_maple_task_messages(task, "synthetic note", question="Why?")


_ASPIRIN_ENTITIES = (
    '{"entities":['
    '{"start":0,"end":7,"label":"MEDICATION","text":"Aspirin"},'
    '{"start":15,"end":19,"label":"CONDITION","text":"pain"}'
    "],"
)
_SENTINEL = "MODEL_OUTPUT_SENTINEL"
_PARSER_CODE_CASES = [
    ("empty_response", "pii", "   ", "A"),
    ("unfinished_reasoning", "pii", "I should return a JSON object", "A"),
    ("invalid_json", "pii", "</think>{not json", "A"),
    ("trailing_text", "pii", '{"spans":[]} trailing words', "A"),
    ("non_object_json", "pii", "</think>[1]", "A"),
    ("invalid_keys", "pii", '{"spans":[],"' + _SENTINEL + '":1}', "A"),
    (
        "invalid_answer",
        "reasoning",
        '{"answer":"","uncertainties":[],"evidence":[]}',
        "A",
    ),
    ("invalid_span", "pii", '{"spans":{}}', "A"),
    ("invalid_span", "pii", '{"spans":[1]}', "A"),
    ("invalid_span", "pii", '{"spans":[{"label":"NAME","text":""}]}', "A"),
    ("unmatched_span", "pii", '{"spans":[{"label":"NAME","text":"Zed"}]}', "A"),
    ("invalid_label", "pii", '{"spans":[{"label":5,"text":"A"}]}', "A"),
    ("invalid_label", "pii", '{"spans":[{"label":"9bad","text":"A"}]}', "A"),
    ("label_not_allowed", "pii", '{"spans":[{"label":"CONDITION","text":"A"}]}', "A"),
    (
        "invalid_relation",
        "relations",
        _ASPIRIN_ENTITIES + '"relations":{}}',
        "Aspirin treats pain.",
    ),
    (
        "invalid_relation",
        "relations",
        _ASPIRIN_ENTITIES + '"relations":[1]}',
        "Aspirin treats pain.",
    ),
    (
        "invalid_relation",
        "relations",
        _ASPIRIN_ENTITIES + '"relations":[{"source":0,"target":0,"label":"TREATS"}]}',
        "Aspirin treats pain.",
    ),
    (
        "invalid_label",
        "relations",
        _ASPIRIN_ENTITIES + '"relations":[{"source":0,"target":1,"label":7}]}',
        "Aspirin treats pain.",
    ),
    (
        "label_not_allowed",
        "relations",
        _ASPIRIN_ENTITIES
        + '"relations":[{"source":0,"target":1,"label":"'
        + _SENTINEL
        + '"}]}',
        "Aspirin treats pain.",
    ),
    (
        "unknown_relation_endpoint",
        "relations",
        _ASPIRIN_ENTITIES + '"relations":[{"source":0,"target":9,"label":"TREATS"}]}',
        "Aspirin treats pain.",
    ),
    (
        "unknown_relation_endpoint",
        "relations",
        _ASPIRIN_ENTITIES
        + '"relations":[{"source":"'
        + _SENTINEL
        + '","target":1,"label":"TREATS"}]}',
        "Aspirin treats pain.",
    ),
    (
        "invalid_uncertainty",
        "reasoning",
        '{"answer":"x","uncertainties":{},"evidence":[]}',
        "A",
    ),
    (
        "invalid_uncertainty",
        "reasoning",
        '{"answer":"x","uncertainties":[1],"evidence":[]}',
        "A",
    ),
]


class _FixedRunner:
    def __init__(self, response):
        self.response = response

    def generate(self, **kwargs):
        return self.response


def _runner_code_cases():
    return [
        (
            "non_text_response",
            lambda: MapleClinicalAssistant(runner=_FixedRunner(42)).complete_task(
                "pii", "A"
            ),
        ),
        (
            "non_text_response",
            lambda: MapleClinicalAssistant(runner=_FixedRunner(None)).chat(
                [{"role": "user", "content": "Hi"}]
            ),
        ),
        (
            "unfinished_reasoning",
            lambda: MapleClinicalAssistant(
                runner=_FixedRunner("<think>" + _SENTINEL)
            ).chat([{"role": "user", "content": "Hi"}]),
        ),
    ]


@pytest.mark.parametrize(
    ("code", "task", "response", "source"),
    _PARSER_CODE_CASES,
    ids=[f"{case[0]}-{index}" for index, case in enumerate(_PARSER_CODE_CASES)],
)
def test_parser_errors_carry_stable_reason_codes(code, task, response, source):
    with pytest.raises(MapleResponseError) as exc_info:
        parse_maple_task_response(task, response, source)

    assert exc_info.value.code == code
    assert code in MAPLE_RESPONSE_ERROR_CODES


def test_runtime_errors_carry_stable_reason_codes(subtests):
    for code, call in _runner_code_cases():
        with subtests.test(code=code):
            with pytest.raises(MapleResponseError) as exc_info:
                call()
            assert exc_info.value.code == code


def test_every_reason_code_except_fallback_is_reachable():
    reached = {case[0] for case in _PARSER_CODE_CASES}
    reached |= {code for code, _ in _runner_code_cases()}

    assert reached == MAPLE_RESPONSE_ERROR_CODES - {"invalid_response"}


def test_every_raise_site_sets_a_literal_code_from_the_closed_set():
    tree = ast.parse(inspect.getsource(maple_module))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "MapleResponseError"
    ]

    assert len(calls) >= 25
    for call in calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert "code" in keywords, ast.unparse(call)
        value = keywords["code"]
        assert isinstance(value, ast.Constant), ast.unparse(call)
        assert value.value in MAPLE_RESPONSE_ERROR_CODES, ast.unparse(call)


def test_reason_codes_never_contain_model_output():
    for code, task, response, source in _PARSER_CODE_CASES:
        with pytest.raises(MapleResponseError) as exc_info:
            parse_maple_task_response(task, response, source)
        assert _SENTINEL not in exc_info.value.code
        assert _SENTINEL.lower() not in exc_info.value.code


def test_unknown_or_missing_code_falls_back_to_invalid_response():
    assert MapleResponseError("legacy message").code == "invalid_response"
    assert MapleResponseError("x", code=_SENTINEL).code == "invalid_response"
    error = MapleResponseError("kept message", code="trailing_text")
    assert str(error) == "kept message"
    assert isinstance(error, ValueError)


@pytest.mark.parametrize("args", [(), ("message",), ("message", 7)])
def test_response_error_preserves_value_error_arguments(args):
    import pickle

    error = MapleResponseError(*args, code="invalid_json")
    assert error.args == ValueError(*args).args
    assert str(error) == str(ValueError(*args))
    restored = pickle.loads(pickle.dumps(error))
    assert restored.args == args
    assert restored.code == "invalid_json"


@pytest.mark.parametrize("code", [[], {}, 7])
def test_non_string_reason_codes_fall_back(code):
    assert MapleResponseError("message", code=code).code == "invalid_response"
