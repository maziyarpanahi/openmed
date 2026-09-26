"""Tests for privacy-safe agent policy decision matrices."""

from __future__ import annotations

import json
from dataclasses import fields

import pytest

from openmed.agent.identifiers import CapabilityId, PolicyId, PurposeId, ToolId
from openmed.agent.policy_matrix import (
    POLICY_MATRIX_SCHEMA_VERSION,
    PolicyDecisionMatrix,
    PolicyDecisionRow,
    PolicyMatrixError,
)


def _row(
    suffix: str,
    outcome_reason_code: str,
    *,
    reviewer_required: bool = False,
) -> PolicyDecisionRow:
    return PolicyDecisionRow(
        policy_version=PolicyId(f"policy:org.openmed/access-{suffix}@1.0.0"),
        capability_id=CapabilityId(f"capability:org.openmed/read-{suffix}"),
        purpose_id=PurposeId(f"purpose:org.openmed/care-{suffix}"),
        tool_id=ToolId(f"tool:org.openmed/search-{suffix}"),
        outcome_reason_code=outcome_reason_code,
        reviewer_required=reviewer_required,
    )


@pytest.mark.parametrize(
    "rows",
    [
        (),
        (_row("allow", "completed"),),
        (_row("deny", "phi_policy"),),
        (_row("abstain", "insufficient_evidence"),),
        (_row("review", "human_gate", reviewer_required=True),),
    ],
    ids=["empty", "allow", "deny", "abstain", "review-required"],
)
def test_synthetic_matrices_render_byte_stably(
    rows: tuple[PolicyDecisionRow, ...],
) -> None:
    matrix = PolicyDecisionMatrix.from_rows(reversed(rows))

    assert matrix.to_json().encode() == matrix.to_json().encode()
    assert matrix.to_markdown().encode() == matrix.to_markdown().encode()
    assert PolicyDecisionMatrix.from_json(matrix.to_json()) == matrix


def test_rows_are_sorted_by_the_complete_decision_key() -> None:
    rows = (
        _row("zulu", "completed"),
        _row("alpha", "phi_policy"),
        _row("middle", "human_gate", reviewer_required=True),
    )

    forward = PolicyDecisionMatrix.from_rows(rows)
    backward = PolicyDecisionMatrix.from_rows(reversed(rows))

    assert forward.to_json() == backward.to_json()
    assert forward.to_markdown() == backward.to_markdown()
    assert [row.policy_version.local_name for row in forward.rows] == [
        "access-alpha",
        "access-middle",
        "access-zulu",
    ]


def test_duplicate_decision_keys_fail_closed_without_echo() -> None:
    sentinel = "phi_policy"
    rows = (
        _row("duplicate", "completed"),
        _row("duplicate", sentinel),
    )

    with pytest.raises(PolicyMatrixError) as exc_info:
        PolicyDecisionMatrix.from_rows(rows)

    assert exc_info.value.code == "duplicate_decision_key"
    assert sentinel not in str(exc_info.value)


@pytest.mark.parametrize(
    "value",
    ["approved", "Patient Jane Doe", "Bearer synthetic-token", "/private/chart"],
)
def test_unknown_outcome_codes_fail_closed_without_echo(value: str) -> None:
    with pytest.raises(PolicyMatrixError) as exc_info:
        _row("invalid", value)

    assert exc_info.value.code == "unknown_outcome_code"
    assert value not in str(exc_info.value)


def test_matrix_schema_cannot_carry_sensitive_content() -> None:
    forbidden_fields = {
        "prompt",
        "arguments",
        "output",
        "evidence_text",
        "bearer",
        "filesystem_path",
    }
    sentinel_values = {
        "prompt": "Patient Jane Doe has synthetic condition Z99.999",
        "arguments": {"record": "synthetic-123"},
        "output": "synthetic clinical output",
        "evidence_text": "synthetic supporting note",
        "bearer": "Bearer synthetic-secret",
        "filesystem_path": "/private/synthetic/chart.json",
    }
    matrix = PolicyDecisionMatrix.from_rows([_row("safe", "completed")])

    assert {field.name for field in fields(PolicyDecisionRow)}.isdisjoint(
        forbidden_fields
    )
    assert {field.name for field in fields(PolicyDecisionMatrix)}.isdisjoint(
        forbidden_fields
    )
    rendered = matrix.to_json() + matrix.to_markdown()
    assert all(str(value) not in rendered for value in sentinel_values.values())

    row_payload = matrix.to_dict()["rows"][0]
    for field_name, sentinel in sentinel_values.items():
        unsafe = dict(row_payload)
        unsafe[field_name] = sentinel
        with pytest.raises(PolicyMatrixError) as exc_info:
            PolicyDecisionRow.from_dict(unsafe)
        assert exc_info.value.code == "unknown_field"
        assert str(sentinel) not in str(exc_info.value)


def test_json_and_markdown_have_an_exact_metadata_only_shape() -> None:
    matrix = PolicyDecisionMatrix.from_rows(
        [_row("review", "safety_review", reviewer_required=True)]
    )

    assert json.loads(matrix.to_json()) == {
        "schema_version": POLICY_MATRIX_SCHEMA_VERSION,
        "rows": [
            {
                "policy_version": "policy:org.openmed/access-review@1.0.0",
                "capability_id": "capability:org.openmed/read-review",
                "purpose_id": "purpose:org.openmed/care-review",
                "tool_id": "tool:org.openmed/search-review",
                "outcome_reason_code": "safety_review",
                "reviewer_required": True,
            }
        ],
    }
    assert matrix.to_markdown() == (
        "# Agent Policy Decision Matrix\n"
        "\n"
        "Schema: `openmed.agent.policy_matrix.v1`\n"
        "\n"
        "| Policy version | Capability | Purpose | Tool | Outcome reason | "
        "Reviewer required |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| `policy:org.openmed/access-review@1.0.0` | "
        "`capability:org.openmed/read-review` | "
        "`purpose:org.openmed/care-review` | "
        "`tool:org.openmed/search-review` | `safety_review` | yes |\n"
    )


def test_unversioned_policy_and_non_boolean_review_flag_are_rejected() -> None:
    with pytest.raises(PolicyMatrixError, match="version_required"):
        PolicyDecisionRow(
            policy_version=PolicyId("policy:org.openmed/access"),
            capability_id=CapabilityId("capability:org.openmed/read"),
            purpose_id=PurposeId("purpose:org.openmed/care"),
            tool_id=ToolId("tool:org.openmed/search"),
            outcome_reason_code="completed",
            reviewer_required=False,
        )

    payload = _row("typed", "completed").to_dict()
    payload["reviewer_required"] = 1
    with pytest.raises(PolicyMatrixError, match="invalid_boolean"):
        PolicyDecisionRow.from_dict(payload)


def test_parser_rejects_wrong_identifier_kinds_and_duplicate_json_fields() -> None:
    payload = _row("typed", "completed").to_dict()
    payload["tool_id"] = "purpose:org.openmed/not-a-tool"

    with pytest.raises(PolicyMatrixError) as exc_info:
        PolicyDecisionRow.from_dict(payload)
    assert exc_info.value.code == "invalid_identifier"
    assert payload["tool_id"] not in str(exc_info.value)

    duplicate = (
        '{"schema_version":"openmed.agent.policy_matrix.v1","rows":[],"rows":[]}'
    )
    with pytest.raises(PolicyMatrixError) as exc_info:
        PolicyDecisionMatrix.from_json(duplicate)
    assert exc_info.value.code == "duplicate_field"


def test_direct_matrix_construction_requires_canonical_order() -> None:
    rows = (_row("zulu", "completed"), _row("alpha", "completed"))

    with pytest.raises(PolicyMatrixError, match="not_sorted"):
        PolicyDecisionMatrix(rows=rows)


def test_policy_matrix_contract_is_available_from_public_agent_api() -> None:
    import openmed.agent as agent

    assert agent.PolicyDecisionMatrix is PolicyDecisionMatrix
    assert agent.PolicyDecisionRow is PolicyDecisionRow
    assert agent.PolicyMatrixError is PolicyMatrixError
    assert agent.POLICY_MATRIX_SCHEMA_VERSION == POLICY_MATRIX_SCHEMA_VERSION
