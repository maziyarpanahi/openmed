"""Evidence-grounded patient and cohort query planner tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jsonschema.validators import validator_for

from openmed.agent.workflows.evidence_query import (
    EVIDENCE_QUERY_ANSWER_SCHEMA_NAME,
    EVIDENCE_QUERY_PLAN_SCHEMA_NAME,
    AccessOutcome,
    BoundedEvidenceQuery,
    BoundedQueryOperation,
    EvidenceAnswerState,
    EvidenceFact,
    EvidenceQueryAnswer,
    EvidenceQueryConflictError,
    EvidenceQueryError,
    EvidenceTool,
    EvidenceToolResult,
    EvidenceUncertainty,
    QueryIntent,
    QueryPlanState,
    QueryScope,
    ToolResultState,
    compose_evidence_answer,
    load_evidence_query_schema,
    make_evidence_citation,
    make_query_access_decision,
    plan_evidence_query,
)
from openmed.clinical.journey_contracts import canonical_digest, derived_opaque_id
from openmed.guard.query_safety import (
    quote_untrusted_scalar,
    validate_bounded_read_only_sql,
)

FIXTURES = Path(__file__).parents[2] / "fixtures" / "agent"
SUBJECT_ID = derived_opaque_id("patient", "evidence-query-subject")
SNAPSHOT_ID = derived_opaque_id("snapshot", "evidence-query-snapshot")
SNAPSHOT_DIGEST = canonical_digest({"snapshot": 7})


def _query(
    *,
    query_text: str = "Show the current synthetic condition evidence.",
    intent: QueryIntent = QueryIntent.EVIDENCE_RETRIEVAL,
    operations: tuple[BoundedQueryOperation, ...] | None = None,
    scope: QueryScope = QueryScope.PATIENT,
) -> BoundedEvidenceQuery:
    return BoundedEvidenceQuery(
        query_id=derived_opaque_id("query", query_text, intent.value, scope.value),
        query_text=query_text,
        intent=intent,
        scope=scope,
        namespace="clinical",
        purpose="care_review",
        subject_id=SUBJECT_ID if scope is QueryScope.PATIENT else None,
        operations=operations
        or (
            BoundedQueryOperation(
                operation_id="read_journey",
                tool=EvidenceTool.JOURNEY,
                resource_id="current",
                fields=("condition",),
                limit=5,
            ),
        ),
        max_results=20,
    )


def _allow():  # type: ignore[no-untyped-def]
    return make_query_access_decision(
        outcome=AccessOutcome.ALLOW,
        namespace="clinical",
        purpose="care_review",
        policy_id="query_policy",
        policy_version="1.0.0",
        reason_code="authorized",
    )


def _deny():  # type: ignore[no-untyped-def]
    return make_query_access_decision(
        outcome=AccessOutcome.DENY,
        namespace="clinical",
        purpose="care_review",
        policy_id="query_policy",
        policy_version="1.0.0",
        reason_code="purpose_denied",
    )


def _citation(*, snapshot: bool = True):  # type: ignore[no-untyped-def]
    return make_evidence_citation(
        resource_kind="journey",
        resource_id="current",
        resource_version="7",
        evidence_ids=(derived_opaque_id("evidence", "synthetic-source"),),
        snapshot_id=SNAPSHOT_ID if snapshot else None,
        snapshot_digest=SNAPSHOT_DIGEST if snapshot else None,
    )


def _success_result(plan, value="synthetic condition"):  # type: ignore[no-untyped-def]
    citation = _citation()
    return EvidenceToolResult(
        call_id=plan.tool_calls[0].call_id,
        state=ToolResultState.SUCCESS,
        access_decision=_allow(),
        facts=(
            EvidenceFact(
                fact_id=derived_opaque_id("fact", "synthetic-condition", value),
                field_name="condition",
                value=value,
                citation_ids=(citation.citation_id,),
            ),
        ),
        citations=(citation,),
    )


def test_ready_plan_emits_only_bounded_read_only_typed_calls() -> None:
    operations = tuple(
        BoundedQueryOperation(
            operation_id=f"read_{name}",
            tool=tool,
            resource_id=f"synthetic_{name}",
            fields=("status",),
            limit=10,
        )
        for name, tool in (
            ("journey", EvidenceTool.JOURNEY),
            ("cohort", EvidenceTool.COHORT),
            ("measure", EvidenceTool.MEASURE),
            ("registry", EvidenceTool.REGISTRY),
        )
    )
    query = _query(operations=operations)
    plan = plan_evidence_query(query)
    assert plan.state is QueryPlanState.READY
    assert len(plan.tool_calls) == 4
    assert all(item.read_only and item.limit == 10 for item in plan.tool_calls)
    assert all(item.subject_id == SUBJECT_ID for item in plan.tool_calls)
    serialized = json.dumps(plan.to_dict(), sort_keys=True)
    assert query.query_text not in serialized
    assert "SELECT " not in serialized

    schema = load_evidence_query_schema(EVIDENCE_QUERY_PLAN_SCHEMA_NAME)
    validator_for(schema).check_schema(schema)
    assert not list(validator_for(schema)(schema).iter_errors(plan.to_dict()))


@pytest.mark.parametrize(
    ("sql", "error"),
    (
        ("DELETE FROM journey_facts LIMIT 5", "sql_not_read_only"),
        ("SELECT fact_id FROM journey_facts", "sql_unbounded"),
        ("SELECT * FROM journey_facts LIMIT 5", "sql_wildcard_rejected"),
        (
            "SELECT fact_id FROM private_records LIMIT 5",
            "sql_view_not_allowed",
        ),
        (
            "SELECT fact_id FROM evil.journey_facts LIMIT 5",
            "sql_view_not_allowed",
        ),
        (
            "SELECT pg_read_file(path) FROM journey_facts LIMIT 5",
            "sql_function_rejected",
        ),
        (
            "SELECT fact_id FROM journey_facts LIMIT 101",
            "sql_row_limit_exceeded",
        ),
        (
            "SELECT fact_id FROM journey_facts LIMIT 5 OFFSET 101",
            "sql_offset_exceeded",
        ),
        (
            "SELECT fact_id FROM journey_facts LIMIT 5; DROP TABLE journey_facts",
            "sql_multiple_statements",
        ),
    ),
)
def test_sql_guard_rejects_mutation_and_unbounded_scans(sql: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        validate_bounded_read_only_sql(sql, max_rows=100)


def test_sql_literals_are_inert_and_raw_sql_is_not_serialized() -> None:
    sql = (
        "SELECT fact_id FROM journey_facts "
        "WHERE label = 'DROP TABLE journey_facts; --' LIMIT 5"
    )
    operation = BoundedQueryOperation(
        operation_id="read_sql",
        tool=EvidenceTool.SQL,
        resource_id="journey_facts",
        fields=("fact_id",),
        limit=5,
        sql=sql,
    )
    plan = plan_evidence_query(_query(operations=(operation,)))
    assert plan.state is QueryPlanState.READY
    assert plan.tool_calls[0].sql == sql
    assert plan.tool_calls[0].sql_digest is not None
    assert sql not in plan.to_json()

    cte_sql = (
        "WITH bounded AS ("
        "SELECT fact_id FROM journey_facts LIMIT 5"
        ") SELECT fact_id FROM bounded LIMIT 5"
    )
    assert validate_bounded_read_only_sql(cte_sql, max_rows=5) == cte_sql


@given(limit=st.integers(min_value=1, max_value=100))
def test_sql_limit_property_accepts_only_the_declared_bound(limit: int) -> None:
    sql = f"SELECT fact_id FROM journey_facts LIMIT {limit}"
    assert validate_bounded_read_only_sql(sql, max_rows=limit) == sql
    if limit > 1:
        with pytest.raises(ValueError, match="sql_row_limit_exceeded"):
            validate_bounded_read_only_sql(sql, max_rows=limit - 1)


@pytest.mark.parametrize(
    ("query_text", "intent", "reason"),
    (
        (
            "What medication should the patient take?",
            QueryIntent.EVIDENCE_RETRIEVAL,
            "unsupported_clinical_advice",
        ),
        (
            "Show evidence and then contact the patient.",
            QueryIntent.EVIDENCE_RETRIEVAL,
            "state_change_requested",
        ),
        (
            "Show current evidence.",
            QueryIntent.STATE_CHANGE,
            "state_change_requested",
        ),
        (
            "Diagnose this synthetic record.",
            QueryIntent.CLINICAL_ADVICE,
            "unsupported_clinical_advice",
        ),
    ),
)
def test_advice_and_state_changes_are_refused_without_tool_calls(
    query_text: str, intent: QueryIntent, reason: str
) -> None:
    plan = plan_evidence_query(_query(query_text=query_text, intent=intent))
    assert plan.state is QueryPlanState.REFUSED
    assert plan.refusal_code == reason
    assert plan.tool_calls == ()
    answer = compose_evidence_answer(plan, ())
    assert answer.state is EvidenceAnswerState.REFUSED
    assert answer.reason_code == reason
    assert answer.statements == ()


def test_successful_patient_answer_has_citations_snapshot_access_and_schema() -> None:
    plan = plan_evidence_query(_query())
    result = _success_result(plan)
    answer = compose_evidence_answer(plan, (result,))
    assert answer.state is EvidenceAnswerState.ANSWERED
    assert answer.uncertainty is EvidenceUncertainty.CERTAIN
    assert len(answer.statements) == 1
    assert answer.statements[0].citation_ids == (answer.citations[0].citation_id,)
    assert answer.citations[0].snapshot_id == SNAPSHOT_ID
    assert answer.citations[0].snapshot_digest == SNAPSHOT_DIGEST
    assert answer.access_decisions[0].outcome is AccessOutcome.ALLOW
    assert answer.reason_code is None
    assert EvidenceQueryAnswer.from_dict(answer.to_dict()) == answer

    schema = load_evidence_query_schema(EVIDENCE_QUERY_ANSWER_SCHEMA_NAME)
    validator_for(schema).check_schema(schema)
    assert not list(validator_for(schema)(schema).iter_errors(answer.to_dict()))


def test_cohort_answer_accepts_cited_aggregate_without_patient_snapshot() -> None:
    operation = BoundedQueryOperation(
        operation_id="read_cohort",
        tool=EvidenceTool.COHORT,
        resource_id="synthetic_cohort",
        fields=("member_count",),
        limit=1,
    )
    plan = plan_evidence_query(_query(operations=(operation,), scope=QueryScope.COHORT))
    citation = make_evidence_citation(
        resource_kind="cohort",
        resource_id="synthetic_cohort",
        resource_version="3",
        evidence_ids=(derived_opaque_id("evidence", "cohort-count"),),
    )
    result = EvidenceToolResult(
        call_id=plan.tool_calls[0].call_id,
        state=ToolResultState.SUCCESS,
        access_decision=_allow(),
        facts=(
            EvidenceFact(
                fact_id=derived_opaque_id("fact", "cohort-count"),
                field_name="member_count",
                value=12,
                citation_ids=(citation.citation_id,),
                patient_level=False,
            ),
        ),
        citations=(citation,),
    )
    answer = compose_evidence_answer(plan, (result,))
    assert answer.state is EvidenceAnswerState.ANSWERED
    assert answer.citations[0].snapshot_id is None
    assert answer.statements[0].patient_level is False


def test_missing_snapshot_empty_denied_failure_and_missing_result_fail_closed() -> None:
    plan = plan_evidence_query(_query())
    citation = _citation(snapshot=False)
    missing_snapshot = EvidenceToolResult(
        call_id=plan.tool_calls[0].call_id,
        state=ToolResultState.SUCCESS,
        access_decision=_allow(),
        facts=(
            EvidenceFact(
                fact_id=derived_opaque_id("fact", "missing-snapshot"),
                field_name="condition",
                value="synthetic condition",
                citation_ids=(citation.citation_id,),
            ),
        ),
        citations=(citation,),
    )
    answer = compose_evidence_answer(plan, (missing_snapshot,))
    assert answer.state is EvidenceAnswerState.INSUFFICIENT_DATA
    assert answer.reason_code == "patient_snapshot_missing"
    assert answer.statements == ()

    empty = EvidenceToolResult(
        call_id=plan.tool_calls[0].call_id,
        state=ToolResultState.EMPTY,
        access_decision=_allow(),
        code="no_evidence",
    )
    assert compose_evidence_answer(plan, (empty,)).state is (
        EvidenceAnswerState.INSUFFICIENT_DATA
    )

    unknown = EvidenceToolResult(
        call_id=plan.tool_calls[0].call_id,
        state=ToolResultState.UNKNOWN,
        access_decision=_allow(),
        code="evidence_unknown",
    )
    assert compose_evidence_answer(plan, (unknown,)).reason_code == (
        "evidence_empty_or_unknown"
    )

    partial_source = _success_result(plan)
    partial = EvidenceToolResult(
        call_id=partial_source.call_id,
        state=ToolResultState.PARTIAL,
        access_decision=partial_source.access_decision,
        facts=partial_source.facts,
        citations=partial_source.citations,
        code="evidence_partial",
    )
    partial_answer = compose_evidence_answer(plan, (partial,))
    assert partial_answer.state is EvidenceAnswerState.INSUFFICIENT_DATA
    assert partial_answer.statements == ()

    unsupported = EvidenceToolResult(
        call_id=plan.tool_calls[0].call_id,
        state=ToolResultState.UNSUPPORTED,
        access_decision=_allow(),
        code="tool_unsupported",
    )
    assert compose_evidence_answer(plan, (unsupported,)).state is (
        EvidenceAnswerState.REFUSED
    )

    denied = EvidenceToolResult(
        call_id=plan.tool_calls[0].call_id,
        state=ToolResultState.DENIED,
        access_decision=_deny(),
        code="purpose_denied",
    )
    denied_answer = compose_evidence_answer(plan, (denied,))
    assert denied_answer.state is EvidenceAnswerState.ACCESS_DENIED
    assert denied_answer.statements == ()

    failed = EvidenceToolResult(
        call_id=plan.tool_calls[0].call_id,
        state=ToolResultState.FAILURE,
        access_decision=_allow(),
        code="tool_failure",
    )
    assert compose_evidence_answer(plan, (failed,)).state is EvidenceAnswerState.FAILURE
    assert compose_evidence_answer(plan, ()).reason_code == (
        "tool_result_missing_or_unexpected"
    )


def test_contradictory_or_uncited_tool_facts_cannot_be_answered() -> None:
    operations = (
        BoundedQueryOperation(
            operation_id="read_one",
            tool=EvidenceTool.JOURNEY,
            resource_id="current",
            fields=("status",),
            limit=2,
        ),
        BoundedQueryOperation(
            operation_id="read_two",
            tool=EvidenceTool.MEASURE,
            resource_id="synthetic_measure",
            fields=("status",),
            limit=2,
        ),
    )
    plan = plan_evidence_query(_query(operations=operations))
    citation = _citation()
    results = tuple(
        EvidenceToolResult(
            call_id=call.call_id,
            state=ToolResultState.SUCCESS,
            access_decision=_allow(),
            facts=(
                EvidenceFact(
                    fact_id=derived_opaque_id("fact", "conflict", index),
                    field_name="status",
                    value=value,
                    citation_ids=(citation.citation_id,),
                    comparison_key="synthetic_status",
                ),
            ),
            citations=(citation,),
        )
        for index, (call, value) in enumerate(
            zip(plan.tool_calls, ("active", "inactive"), strict=True)
        )
    )
    answer = compose_evidence_answer(plan, results)
    assert answer.state is EvidenceAnswerState.CONFLICT
    assert answer.statements == ()

    with pytest.raises(EvidenceQueryError, match="uncited fact"):
        EvidenceToolResult(
            call_id=plan.tool_calls[0].call_id,
            state=ToolResultState.SUCCESS,
            access_decision=_allow(),
            facts=(
                EvidenceFact(
                    fact_id=derived_opaque_id("fact", "uncited"),
                    field_name="status",
                    value="active",
                    citation_ids=(citation.citation_id,),
                ),
            ),
            citations=(),
        )


def test_query_and_answer_identifiers_are_self_verifying() -> None:
    plan = plan_evidence_query(_query())
    with pytest.raises(EvidenceQueryConflictError, match="plan identifier"):
        replace(plan, plan_id=derived_opaque_id("queryplan", "changed"))

    answer = compose_evidence_answer(plan, (_success_result(plan),))
    with pytest.raises(EvidenceQueryConflictError, match="answer identifier"):
        replace(answer, answer_id=derived_opaque_id("queryanswer", "changed"))

    changed = answer.to_dict()
    changed["answer_digest"] = "sha256:" + "0" * 64
    with pytest.raises(EvidenceQueryConflictError, match="answer digest"):
        EvidenceQueryAnswer.from_dict(changed)


def test_golden_prompt_injection_cases_remain_inert_and_bounded() -> None:
    cases = json.loads(
        (FIXTURES / "evidence_query_cases.json").read_text(encoding="utf-8")
    )
    assert len(cases) == 4
    for case in cases:
        intent = QueryIntent(case["intent"])
        plan = plan_evidence_query(_query(query_text=case["query_text"], intent=intent))
        assert plan.state.value == case["expected_plan_state"]
        if plan.state is QueryPlanState.REFUSED:
            answer = compose_evidence_answer(plan, ())
        else:
            assert len(plan.tool_calls) == 1
            assert plan.tool_calls[0].tool is EvidenceTool.JOURNEY
            answer = compose_evidence_answer(
                plan, (_success_result(plan, case["tool_value"]),)
            )
            rendered = answer.statements[0].text
            assert "\n" not in rendered
            assert quote_untrusted_scalar(case["tool_value"]) in rendered
            context = answer.to_explanation_context()
            assert "untrusted data" in context["instruction"]
            assert context["allowed_statements"] == [answer.statements[0].to_dict()]
        assert answer.state.value == case["expected_answer_state"]


@settings(max_examples=40, deadline=None)
@given(
    value=st.text(
        alphabet=st.characters(
            blacklist_categories=("Cs",),
            blacklist_characters=("\x00",),
        ),
        max_size=64,
    )
)
def test_tool_value_property_never_creates_an_uncited_statement(value: str) -> None:
    plan = plan_evidence_query(_query())
    answer = compose_evidence_answer(plan, (_success_result(plan, value),))
    assert answer.state is EvidenceAnswerState.ANSWERED
    assert len(answer.statements) == 1
    assert answer.statements[0].citation_ids
    assert not any(
        marker in answer.statements[0].text
        for marker in ("\n", "\r", "\u2028", "\u2029")
    )
    assert quote_untrusted_scalar(value) in answer.statements[0].text
