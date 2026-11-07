"""The fault-plan schema is versioned, seeded, value-free and reproducible."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from tests.faults.plan import (
    CATEGORIES,
    FAULTS_BY_CATEGORY,
    MANIFEST_SCHEMA_PATH,
    OUTCOMES,
    PLAN_SCHEMA_PATH,
    SERVICE_PLAN_PATH,
    FaultPlan,
    FaultPlanError,
    load_json_schema,
)
from tests.faults.scenarios import COMPONENTS, component_for

CANARY = "SYNTHETIC-PHI-CANARY-7731"


def _raw_plan() -> dict[str, Any]:
    return json.loads(SERVICE_PLAN_PATH.read_text(encoding="utf-8"))


def _one_case_plan(**overrides: Any) -> dict[str, Any]:
    case = {
        "case_id": "unit.case",
        "component": "service.jobs.local_store",
        "category": "disk",
        "fault": "enospc",
        "injection_point": "disk.write",
        "phase": "before_commit",
        "trigger": {"at_call": 1},
        "expected_outcome": "rolled_back",
    }
    case.update(overrides)
    return {
        "schema_version": "1.0.0",
        "plan_id": "unit.plan",
        "seed": 7,
        "cases": [case],
    }


def test_bundled_schemas_are_valid_draft_2020_12() -> None:
    for path in (PLAN_SCHEMA_PATH, MANIFEST_SCHEMA_PATH):
        schema = load_json_schema(path)
        Draft202012Validator.check_schema(schema)
        assert schema["properties"]["schema_version"] == {"const": "1.0.0"}


def test_service_plan_validates_and_covers_every_category_and_fault() -> None:
    raw = _raw_plan()
    Draft202012Validator(load_json_schema(PLAN_SCHEMA_PATH)).validate(raw)
    plan = FaultPlan.load()

    assert {case.category for case in plan.cases} == set(CATEGORIES)
    assert {case.fault for case in plan.cases} == {
        fault for faults in FAULTS_BY_CATEGORY.values() for fault in faults
    }
    assert {case.expected_outcome for case in plan.cases} == set(OUTCOMES)
    for case in plan.cases:
        component = component_for(case)
        assert component.name in COMPONENTS
    assert plan.to_mapping() == raw


def test_plan_rejections_use_fixed_codes_without_echoing_values(subtests) -> None:
    duplicate = _one_case_plan()
    duplicate["cases"].append(copy.deepcopy(duplicate["cases"][0]))
    rejected = {
        "fault_case_duplicate": duplicate,
        "fault_trigger_range_invalid": _one_case_plan(
            trigger={"seeded": {"min_call": 5, "max_call": 2}}
        ),
    }
    schema_invalid = {
        "category_mismatch": _one_case_plan(category="disk", fault="clock_skew"),
        "unknown_field": _one_case_plan(note=CANARY),
        "free_text_parameter": _one_case_plan(parameters={"operation": CANARY}),
        "both_triggers": _one_case_plan(
            trigger={"at_call": 1, "seeded": {"min_call": 1, "max_call": 2}}
        ),
        "unknown_outcome": _one_case_plan(expected_outcome="eventually_fine"),
        "unversioned": {**_one_case_plan(), "schema_version": "0.9.0"},
    }
    for code, document in rejected.items():
        with subtests.test(code=code):
            with pytest.raises(FaultPlanError) as error:
                FaultPlan.from_mapping(document)
            assert error.value.code == code
    for name, document in schema_invalid.items():
        with subtests.test(name=name):
            with pytest.raises(FaultPlanError) as error:
                FaultPlan.from_mapping(document)
            assert error.value.code == "fault_plan_schema_invalid"
            assert CANARY not in str(error.value)


def test_seeded_triggers_and_case_identities_are_reproducible() -> None:
    plan = FaultPlan.load()
    first = plan.resolved()
    second = FaultPlan.load().resolved()

    assert [item.case_digest for item in first] == [item.case_digest for item in second]
    assert [item.fault_call for item in first] == [item.fault_call for item in second]
    assert len({item.case_digest for item in first}) == len(first)

    reseeded = plan.with_seed(plan.seed + 1).resolved()
    assert all(
        left.case_digest != right.case_digest
        for left, right in zip(first, reseeded, strict=True)
    )

    seeded = plan.case("jobs-store.partial-write")
    trigger = seeded.trigger
    assert trigger.min_call is not None and trigger.max_call is not None
    calls = {plan.with_seed(seed).resolve(seeded).fault_call for seed in range(40)}
    assert calls <= set(range(trigger.min_call, trigger.max_call + 1))
    assert len(calls) > 1


def test_case_identity_changes_with_its_declaration() -> None:
    raw = _one_case_plan()
    baseline = FaultPlan.from_mapping(raw).resolved()[0]
    changed = copy.deepcopy(raw)
    changed["cases"][0]["expected_outcome"] = "retryable"

    assert FaultPlan.from_mapping(changed).resolved()[0].case_digest != (
        baseline.case_digest
    )
    assert FaultPlan.from_mapping(raw).resolved()[0] == baseline


def test_registry_rejects_unknown_components_and_unsupported_seams(subtests) -> None:
    cases = {
        "fault_component_unknown": _one_case_plan(component="service.unknown"),
        "fault_case_unsupported": _one_case_plan(injection_point="sqlite.commit"),
    }
    for code, document in cases.items():
        with subtests.test(code=code):
            case = FaultPlan.from_mapping(document).cases[0]
            with pytest.raises(FaultPlanError) as error:
                component_for(case)
            assert error.value.code == code
