"""Tests for Maple's bounded clinical task contract."""

from __future__ import annotations

import pytest

from openmed.mlx.maple import (
    MAPLE_MEDICAL_DISCLAIMER,
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


@pytest.mark.parametrize("task", ["pii", "entities", "relations"])
def test_non_reasoning_tasks_reject_question(task):
    with pytest.raises(ValueError, match="only accepted"):
        build_maple_task_messages(task, "synthetic note", question="Why?")
