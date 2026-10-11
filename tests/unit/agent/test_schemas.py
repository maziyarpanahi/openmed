"""Offline contract tests for agent governance JSON Schema exports."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from dataclasses import fields
from typing import Any, NoReturn

import pytest
from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry, Resource

from openmed.agent.correlation import (
    ACTION_ID_PREFIX,
    CORRELATION_SCHEMA_VERSION,
    CORRELATION_TOKEN_BYTES,
    RUN_ID_PREFIX,
    ActionCorrelation,
    ActionId,
    RunId,
)
from openmed.agent.outcomes import (
    OUTCOME_SCHEMA_VERSION,
    OutcomeClass,
    WorkflowOutcome,
    allowed_reason_codes,
)
from openmed.agent.run_summary import (
    _MAX_DURATION_SECONDS,
    _MAX_EVENTS,
    _MAX_SUMMARY_DIGESTS,
    _MAX_TOOL_CALLS,
    _MAX_WORKFLOWS,
    RUN_SUMMARY_SCHEMA_VERSION,
    RunEvent,
    RunSummary,
)
from openmed.agent.schemas import (
    AGENT_SCHEMA_DIALECT,
    build_agent_schema,
    build_agent_schema_catalog,
    list_agent_schema_names,
    render_agent_schema,
)
from openmed.agent.timing import ActionTiming, AgentRunTiming, RunTiming

SCHEMA_NAMES = ("correlation", "outcome", "run_summary", "timing")
SCHEMA_SHA256 = {
    "correlation": "c36133f0bd6d7a39e35b4486ba81b0b847ab8bfedd83fcedce085148c70454a0",
    "outcome": "91ec366ce715e460e6edc6bd385bc6ce9d972091e08c53024e76c0735f925360",
    "run_summary": "5e0fb6c541b163a898fef5575b47c1aab6ac926c0898d2cf4391615bb1fed95a",
    "timing": "772aea820547fce3cdc8afa571f2f096d01c08bfefd36c004a6672d9768e87ca",
}
RUN_ID = RUN_ID_PREFIX + "a" * (CORRELATION_TOKEN_BYTES * 2)
ACTION_ID = ACTION_ID_PREFIX + "b" * (CORRELATION_TOKEN_BYTES * 2)
PARENT_ID = ACTION_ID_PREFIX + "c" * (CORRELATION_TOKEN_BYTES * 2)
DIGEST = "sha256:" + "d" * 64


def _no_remote(uri: str) -> NoReturn:
    raise AssertionError(f"unexpected remote schema resolution: {uri}")


def _validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(
        build_agent_schema(name), registry=Registry(retrieve=_no_remote)
    )


def _outcome() -> dict[str, Any]:
    return WorkflowOutcome(OutcomeClass.SUCCESS, "completed").to_dict()


def _correlation() -> dict[str, Any]:
    return ActionCorrelation(RunId(RUN_ID), ActionId(ACTION_ID)).to_dict()


def _timing() -> dict[str, Any]:
    return AgentRunTiming(
        run=RunTiming(start_ns=10, end_ns=30, correlation_id="run-opaque"),
        actions=(
            ActionTiming(action_id="parent", start_ns=12, end_ns=27),
            ActionTiming(
                action_id="child",
                start_ns=15,
                end_ns=20,
                parent_action_id="parent",
                correlation_id="act-opaque",
            ),
        ),
    ).to_dict()


def _summary() -> dict[str, Any]:
    return RunSummary.from_events(
        [
            RunEvent(
                workflow_id="intake",
                outcome=WorkflowOutcome(OutcomeClass.SUCCESS, "completed"),
                tool_call_count=2,
                duration_seconds=1.25,
                artifact_digests=(DIGEST,),
            ),
            RunEvent(
                workflow_id="review",
                outcome=WorkflowOutcome(OutcomeClass.REVIEW_REQUIRED, "human_gate"),
            ),
        ]
    ).to_dict()


EXAMPLES: dict[str, Callable[[], dict[str, Any]]] = {
    "outcome": _outcome,
    "correlation": _correlation,
    "timing": _timing,
    "run_summary": _summary,
}


def test_catalog_has_unique_valid_ids_defs_and_only_local_references() -> None:
    catalog = build_agent_schema_catalog()
    assert list_agent_schema_names() == SCHEMA_NAMES
    assert tuple(catalog) == SCHEMA_NAMES

    ids: set[str] = set()
    definitions: set[str] = set()
    resources = []
    for name, schema in catalog.items():
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == AGENT_SCHEMA_DIALECT
        assert schema["$id"].startswith("urn:openmed:agent:")
        assert schema["$id"] not in ids
        ids.add(schema["$id"])
        for definition in schema["$defs"]:
            assert definition not in definitions
            definitions.add(definition)
        resources.append((schema["$id"], Resource.from_contents(schema)))

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "$ref":
                        assert child.startswith("#/$defs/")
                        assert child.split("/")[-1] in schema["$defs"]
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(schema)

    registry = Registry(retrieve=_no_remote).with_resources(resources)
    for name, schema in catalog.items():
        Draft202012Validator({"$ref": schema["$id"]}, registry=registry).validate(
            EXAMPLES[name]()
        )


@pytest.mark.parametrize("outcome_class", list(OutcomeClass))
def test_every_public_outcome_and_allowed_reason_validates(
    outcome_class: OutcomeClass,
) -> None:
    for reason in allowed_reason_codes(outcome_class):
        payload = WorkflowOutcome(outcome_class, reason).to_dict()
        _validator("outcome").validate(payload)


def test_public_correlation_timing_and_summary_payloads_validate() -> None:
    for name in ("correlation", "timing", "run_summary"):
        _validator(name).validate(EXAMPLES[name]())

    child = ActionCorrelation(RunId(RUN_ID), ActionId(ACTION_ID), ActionId(PARENT_ID))
    _validator("correlation").validate(child.to_dict())
    _validator("timing").validate(AgentRunTiming(RunTiming(0, 0)).to_dict())
    _validator("run_summary").validate(RunSummary.from_events([]).to_dict())


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("outcome", lambda p: p.update({"message": "synthetic patient text"})),
        ("outcome", lambda p: p.update({"schema_version": "v0"})),
        ("outcome", lambda p: p.update({"outcome_class": "unknown"})),
        ("outcome", lambda p: p.update({"reason_code": "human_gate"})),
        ("outcome", lambda p: p.pop("reason_code")),
        ("correlation", lambda p: p.update({"run_id": ACTION_ID})),
        ("correlation", lambda p: p.update({"action_id": ACTION_ID.upper()})),
        ("correlation", lambda p: p.update({"parent_action_id": RUN_ID})),
        ("correlation", lambda p: p.update({"schema_version": "v0"})),
        ("correlation", lambda p: p.update({"run_id": RUN_ID + "\n"})),
        ("correlation", lambda p: p.update({"prompt": "synthetic content"})),
        ("correlation", lambda p: p.pop("parent_action_id")),
        ("timing", lambda p: p["run"].update({"start_ns": -1})),
        ("timing", lambda p: p["run"].update({"duration_ns": True})),
        ("timing", lambda p: p["actions"][0].update({"action_id": "a/b"})),
        ("timing", lambda p: p["actions"][0].update({"args": {"x": 1}})),
        ("timing", lambda p: p["run"].pop("duration_ns")),
        ("timing", lambda p: p.pop("actions")),
        ("run_summary", lambda p: p.update({"schema_version": "v0"})),
        ("run_summary", lambda p: p["outcome_counts"].pop("success")),
        ("run_summary", lambda p: p["outcome_counts"].update({"unknown": 1})),
        ("run_summary", lambda p: p.update({"tool_call_count": True})),
        ("run_summary", lambda p: p.update({"duration_seconds": -1})),
        ("run_summary", lambda p: p["workflow_ids"].append("private/path")),
        ("run_summary", lambda p: p["artifact_digests"].append(DIGEST + "\n")),
        ("run_summary", lambda p: p.update({"raw_events": []})),
        ("run_summary", lambda p: p.pop("workflow_ids")),
    ],
)
def test_malformed_or_content_bearing_records_fail(
    name: str, mutate: Callable[[dict[str, Any]], object]
) -> None:
    payload = copy.deepcopy(EXAMPLES[name]())
    mutate(payload)
    with pytest.raises(ValidationError):
        _validator(name).validate(payload)


def test_bounds_and_nested_closure_track_run_summary_source() -> None:
    schema = build_agent_schema("run_summary")
    props = schema["properties"]
    defs = schema["$defs"]

    assert props["workflow_ids"]["maxItems"] == _MAX_WORKFLOWS
    assert props["artifact_digests"]["maxItems"] == _MAX_SUMMARY_DIGESTS
    assert props["tool_call_count"]["maximum"] == _MAX_TOOL_CALLS
    assert props["duration_seconds"]["maximum"] == _MAX_DURATION_SECONDS
    assert set(defs["summary_outcome_counts"]["properties"]) == {
        item.value for item in OutcomeClass
    }
    assert all(
        prop["maximum"] == _MAX_EVENTS
        for prop in defs["summary_outcome_counts"]["properties"].values()
    )
    assert defs["summary_outcome_counts"]["additionalProperties"] is False
    assert props["artifact_digests"]["uniqueItems"] is True
    assert set(props) == {field.name for field in fields(RunSummary)}
    assert set(schema["required"]) == set(props)

    invalid = RunSummary.from_events([]).to_dict()
    invalid["workflow_ids"] = ["x"] * (_MAX_WORKFLOWS + 1)
    with pytest.raises(ValidationError):
        _validator("run_summary").validate(invalid)


def test_outcome_and_correlation_versions_enums_and_fields_track_source() -> None:
    outcome = build_agent_schema("outcome")
    correlation = build_agent_schema("correlation")

    assert outcome["properties"]["schema_version"]["const"] == OUTCOME_SCHEMA_VERSION
    assert outcome["properties"]["outcome_class"]["enum"] == sorted(
        item.value for item in OutcomeClass
    )
    for item in OutcomeClass:
        assert outcome["$defs"][f"outcome_{item.value}"]["properties"]["reason_code"][
            "enum"
        ] == sorted(allowed_reason_codes(item))
    assert set(outcome["properties"]) == {
        field.name for field in fields(WorkflowOutcome)
    }
    assert set(outcome["required"]) == set(outcome["properties"])

    assert correlation["properties"]["schema_version"]["const"] == (
        CORRELATION_SCHEMA_VERSION
    )
    assert set(correlation["properties"]) == {
        field.name for field in fields(ActionCorrelation)
    }
    assert set(correlation["required"]) == set(correlation["properties"])
    assert correlation["$defs"]["correlation_run_id"]["minLength"] == (
        len(RUN_ID_PREFIX) + 2 * CORRELATION_TOKEN_BYTES
    )
    assert correlation["$defs"]["correlation_action_id"]["minLength"] == (
        len(ACTION_ID_PREFIX) + 2 * CORRELATION_TOKEN_BYTES
    )


@pytest.mark.parametrize("outcome_class", list(OutcomeClass))
def test_outcome_definition_is_closed_when_resolved_independently(
    outcome_class: OutcomeClass,
) -> None:
    schema = build_agent_schema("outcome")
    fragment = schema["$defs"][f"outcome_{outcome_class.value}"]
    validator = Draft202012Validator(fragment)
    payload = WorkflowOutcome(
        outcome_class, min(allowed_reason_codes(outcome_class))
    ).to_dict()
    validator.validate(payload)
    with pytest.raises(ValidationError):
        validator.validate({**payload, "note": "synthetic private content"})
    with pytest.raises(ValidationError):
        validator.validate({**payload, "schema_version": "v0"})


def test_timing_schema_matches_serialized_fields_without_inventing_version() -> None:
    schema = build_agent_schema("timing")
    assert set(schema["properties"]) == set(_timing())
    assert "schema_version" not in schema["properties"]
    assert set(schema["$defs"]["timing_run"]["properties"]) == set(
        RunTiming(0, 1, correlation_id="run-id").to_dict()
    )
    assert set(schema["$defs"]["timing_action"]["properties"]) == set(
        ActionTiming("a", 0, 1, parent_action_id="b", correlation_id="act-id").to_dict()
    )
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["timing_run"]["additionalProperties"] is False
    assert schema["$defs"]["timing_action"]["additionalProperties"] is False


def test_rendering_is_byte_stable_and_returns_fresh_mappings() -> None:
    first = build_agent_schema_catalog()
    first["outcome"]["properties"].clear()
    first["timing"]["$defs"]["timing_action"]["properties"].clear()

    for name in SCHEMA_NAMES:
        encoded = render_agent_schema(name)
        assert encoded == render_agent_schema(name)
        assert json.loads(encoded) == build_agent_schema(name)
        assert encoded.isascii()
        assert (
            hashlib.sha256(encoded.encode("ascii")).hexdigest() == SCHEMA_SHA256[name]
        )
    assert build_agent_schema("outcome")["properties"]
    assert build_agent_schema("timing")["$defs"]["timing_action"]["properties"]


@pytest.mark.parametrize("bad_name", ["", "other", "outcome\nsecret", None, []])
def test_unknown_schema_names_fail_without_echo(bad_name: object) -> None:
    with pytest.raises(ValueError, match="^schema: unknown_name$") as caught:
        build_agent_schema(bad_name)  # type: ignore[arg-type]
    if bad_name:
        assert str(bad_name) not in str(caught.value)


def test_catalog_is_available_through_public_agent_module() -> None:
    import openmed.agent as agent

    assert agent.list_agent_schema_names() == SCHEMA_NAMES
    assert agent.build_agent_schema("outcome") == build_agent_schema("outcome")
    assert agent.render_agent_schema("timing") == render_agent_schema("timing")
