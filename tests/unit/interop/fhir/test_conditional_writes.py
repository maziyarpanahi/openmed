"""Offline tests for FHIR conditional write planning."""

from __future__ import annotations

import socket
from uuid import UUID

import pytest

from openmed.interop.fhir.conditional_writes import (
    ConditionalWriteKind,
    MatchDecision,
    MatchDisposition,
    SearchPredicate,
    assess_matches,
    build_conditional_write_plan,
)

SECRET = b"synthetic-deployment-secret-32-bytes-long"
OPERATION_ID = UUID("00000000-0000-4000-8000-000000000001")
PARAMETERS = {"identifier": "urn:synthetic|private-marker", "status": "final"}


def _plan(kind: ConditionalWriteKind = ConditionalWriteKind.CREATE):
    return build_conditional_write_plan(
        kind,
        "Observation",
        PARAMETERS,
        operation_id=OPERATION_ID,
        secret=SECRET,
    )


def test_canonical_query_and_key_are_stable_across_input_order() -> None:
    first = _plan()
    reversed_parameters = dict(reversed(list(PARAMETERS.items())))
    second = build_conditional_write_plan(
        ConditionalWriteKind.CREATE,
        "Observation",
        reversed_parameters,
        operation_id=OPERATION_ID,
        secret=SECRET,
    )

    assert first == second
    assert first.predicate.canonical_query == (
        "identifier=urn%3Asynthetic%7Cprivate-marker&status=final"
    )
    assert first.idempotency_key.startswith("fhir-cw-v1-")
    assert len(first.idempotency_key) == len("fhir-cw-v1-") + 64
    assert _plan(ConditionalWriteKind.UPDATE).idempotency_key != first.idempotency_key
    assert (
        build_conditional_write_plan(
            ConditionalWriteKind.CREATE,
            "Patient",
            PARAMETERS,
            operation_id=OPERATION_ID,
            secret=SECRET,
        ).idempotency_key
        != first.idempotency_key
    )
    assert (
        build_conditional_write_plan(
            ConditionalWriteKind.CREATE,
            "Observation",
            PARAMETERS,
            operation_id=OPERATION_ID,
            secret=b"another-synthetic-deployment-secret-32-bytes",
        ).idempotency_key
        != first.idempotency_key
    )
    assert (
        build_conditional_write_plan(
            ConditionalWriteKind.CREATE,
            "Observation",
            PARAMETERS,
            operation_id=UUID("00000000-0000-4000-8000-000000000002"),
            secret=SECRET,
        ).idempotency_key
        != first.idempotency_key
    )


def test_planning_never_opens_a_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_socket(*args, **kwargs):
        raise AssertionError("planning attempted network access")

    monkeypatch.setattr(socket, "socket", fail_socket)
    assert _plan().resource_type == "Observation"


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"_count": "1"},
        {"identifier": ""},
        {"identifier": "one,two"},
        {"identifier": "a\nb"},
        {"bad?name": "value"},
    ],
)
def test_invalid_predicates_fail_closed_without_echoing_values(parameters) -> None:
    with pytest.raises(ValueError) as error:
        build_conditional_write_plan(
            ConditionalWriteKind.CREATE,
            "Observation",
            parameters,
            operation_id=OPERATION_ID,
            secret=SECRET,
        )
    assert "one,two" not in str(error.value)
    assert "a\nb" not in str(error.value)


def test_duplicate_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate search parameter"):
        SearchPredicate((("identifier", "first"), ("identifier", "second")))


def test_invalid_plan_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="resource type is invalid"):
        build_conditional_write_plan(
            ConditionalWriteKind.CREATE,
            "../Observation",
            PARAMETERS,
            operation_id=OPERATION_ID,
            secret=SECRET,
        )
    with pytest.raises(ValueError, match="secret must contain"):
        build_conditional_write_plan(
            ConditionalWriteKind.CREATE,
            "Observation",
            PARAMETERS,
            operation_id=OPERATION_ID,
            secret=b"short",
        )
    with pytest.raises(TypeError, match="kind must be"):
        build_conditional_write_plan(
            "create",
            "Observation",
            PARAMETERS,
            operation_id=OPERATION_ID,
            secret=SECRET,
        )
    with pytest.raises(TypeError, match="operation_id must be UUID"):
        build_conditional_write_plan(
            ConditionalWriteKind.CREATE,
            "Observation",
            PARAMETERS,
            operation_id="private-marker",
            secret=SECRET,
        )


def test_repr_and_decisions_never_expose_predicate_values() -> None:
    plan = _plan()
    assert "private-marker" not in repr(plan)
    assert "private-marker" not in repr(plan.predicate)
    assert plan.idempotency_key not in repr(plan)

    decision = assess_matches(plan, 2, search_complete=True)
    assert decision.disposition is MatchDisposition.REVIEW
    assert decision.reason_code == "ambiguous_matches"
    assert decision.is_ready is False
    with pytest.raises(ValueError, match="ambiguous_matches") as error:
        decision.require_ready()
    assert "private-marker" not in str(error.value)
    assert "private-marker" not in repr(decision)


@pytest.mark.parametrize(
    ("kind", "count", "complete", "disposition", "reason"),
    [
        (
            ConditionalWriteKind.CREATE,
            0,
            True,
            MatchDisposition.READY,
            "no_existing_match",
        ),
        (
            ConditionalWriteKind.CREATE,
            1,
            True,
            MatchDisposition.NO_OP,
            "existing_match",
        ),
        (
            ConditionalWriteKind.CREATE,
            2,
            True,
            MatchDisposition.REVIEW,
            "ambiguous_matches",
        ),
        (
            ConditionalWriteKind.UPDATE,
            0,
            True,
            MatchDisposition.REVIEW,
            "update_match_missing",
        ),
        (ConditionalWriteKind.UPDATE, 1, True, MatchDisposition.READY, "unique_match"),
        (
            ConditionalWriteKind.UPDATE,
            3,
            True,
            MatchDisposition.REVIEW,
            "ambiguous_matches",
        ),
        (
            ConditionalWriteKind.CREATE,
            0,
            False,
            MatchDisposition.REVIEW,
            "search_incomplete",
        ),
        (
            ConditionalWriteKind.UPDATE,
            1,
            False,
            MatchDisposition.REVIEW,
            "search_incomplete",
        ),
    ],
)
def test_match_decisions(kind, count, complete, disposition, reason) -> None:
    decision = assess_matches(_plan(kind), count, search_complete=complete)
    assert (decision.disposition, decision.reason_code) == (disposition, reason)
    if disposition is MatchDisposition.READY:
        decision.require_ready()
    else:
        with pytest.raises(ValueError):
            decision.require_ready()


@pytest.mark.parametrize("count", [-1, True, 1.5, "1"])
def test_invalid_match_counts_fail_closed(count) -> None:
    with pytest.raises(ValueError, match="match count is invalid"):
        assess_matches(_plan(), count, search_complete=True)


def test_match_decision_cannot_hold_a_free_form_reason() -> None:
    with pytest.raises(ValueError, match="match decision is invalid") as error:
        MatchDecision(MatchDisposition.REVIEW, "private-marker")
    assert "private-marker" not in str(error.value)
