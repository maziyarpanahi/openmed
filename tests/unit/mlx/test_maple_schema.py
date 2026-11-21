"""Contract tests for the Maple structured task result JSON Schema."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from openmed.mlx.maple import MapleTask, MapleTaskResult, parse_maple_task_response

SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "openmed"
    / "core"
    / "schemas"
    / "json"
    / "maple_task_result.schema.json"
)


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validate(schema: dict, payload: dict) -> None:
    """Validate against the schema plus the span-order rule.

    JSON Schema cannot compare two sibling properties, so ``end > start`` is
    checked here on top of the schema's per-field bounds.
    """

    Draft202012Validator(schema).validate(payload)
    for span in [*payload["entities"], *payload["evidence"]]:
        if span["end"] <= span["start"]:
            raise ValidationError(f"span end <= start: {span}")


_PII_SOURCE = "Patient Ada called 555-0100."
_REL_SOURCE = "Aspirin treats pain."
_REASON_SOURCE = "[NAME] takes aspirin daily."

_FIXTURES = {
    "pii": (
        MapleTask.PII,
        '{"spans":['
        '{"start":8,"end":11,"label":"NAME","text":"Ada"},'
        '{"start":19,"end":27,"label":"PHONE","text":"555-0100"}]}',
        _PII_SOURCE,
    ),
    "entities": (
        MapleTask.ENTITIES,
        '{"spans":[{"start":0,"end":7,"label":"medication","text":"Aspirin"}]}',
        _REL_SOURCE,
    ),
    "relations": (
        MapleTask.RELATIONS,
        '{"entities":['
        '{"start":0,"end":7,"label":"medication","text":"Aspirin"},'
        '{"start":15,"end":19,"label":"condition","text":"pain"}'
        '],"relations":[{"source":0,"target":1,"label":"treats"}]}',
        _REL_SOURCE,
    ),
    "reasoning": (
        MapleTask.REASONING,
        '{"answer":"The note reports daily aspirin use.",'
        '"uncertainties":["Dose is not documented."],'
        '"evidence":[{"start":13,"end":26,"text":"aspirin daily"}]}',
        _REASON_SOURCE,
    ),
    "summarize": (
        MapleTask.SUMMARIZE,
        '{"answer":"Daily aspirin.","uncertainties":[],'
        '"evidence":[{"start":13,"end":26,"text":"aspirin daily"}]}',
        _REASON_SOURCE,
    ),
}


def _build(name: str) -> MapleTaskResult:
    task, response, source = _FIXTURES[name]
    return parse_maple_task_response(task, response, source)


def test_schema_is_a_valid_closed_draft_2020_12_document(schema):
    Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(MapleTaskResult(task=MapleTask.PII).to_dict())
    assert set(schema["properties"]) == set(schema["required"])
    assert schema["properties"]["task"]["enum"] == [task.value for task in MapleTask]


@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_parsed_results_validate(schema, name):
    payload = _build(name).to_dict()

    _validate(schema, payload)
    # The payload must also survive a JSON round trip unchanged.
    _validate(schema, json.loads(json.dumps(payload)))


def test_default_result_validates(schema):
    _validate(schema, MapleTaskResult(task=MapleTask.PII).to_dict())


def test_extra_top_level_key_fails(schema):
    payload = _build("reasoning").to_dict()
    payload["source_text"] = "leaked"

    with pytest.raises(ValidationError):
        _validate(schema, payload)


def test_extra_span_key_fails(schema):
    payload = _build("pii").to_dict()
    payload["entities"][0]["text"] = "Ada"

    with pytest.raises(ValidationError):
        _validate(schema, payload)


def test_missing_required_key_fails(schema):
    payload = _build("pii").to_dict()
    del payload["disclaimer"]

    with pytest.raises(ValidationError):
        _validate(schema, payload)


def test_unknown_task_fails(schema):
    payload = _build("pii").to_dict()
    payload["task"] = "diagnose"

    with pytest.raises(ValidationError):
        _validate(schema, payload)


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1, 3), (3, -1), (5, 5), (7, 3)],
    ids=["negative-start", "negative-end", "empty", "reversed"],
)
@pytest.mark.parametrize("field", ["entities", "evidence"])
def test_invalid_span_offsets_fail(schema, field, start, end):
    payload = copy.deepcopy(_build("reasoning").to_dict())
    payload[field] = [{"start": start, "end": end, "label": "X"}]

    with pytest.raises(ValidationError):
        _validate(schema, payload)


def test_relation_with_negative_index_fails(schema):
    payload = _build("relations").to_dict()
    payload["relations"][0]["source"] = -1

    with pytest.raises(ValidationError):
        _validate(schema, payload)
