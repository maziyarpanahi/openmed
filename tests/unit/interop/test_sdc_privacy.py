"""Offline tests for the FHIR SDC QuestionnaireResponse privacy projection."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from openmed.interop.fhir.sdc_privacy import (
    AmbiguousPolicyPathError,
    QuestionnaireResponseChangeSummary,
    project_questionnaire_response,
    project_questionnaire_response_with_summary,
)


def _questionnaire_response() -> dict:
    return {
        "resourceType": "QuestionnaireResponse",
        "id": "qr-synthetic-1",
        "status": "completed",
        "item": [
            {
                "linkId": "section-demographics",
                "item": [
                    {
                        "linkId": "field-display-name",
                        "answer": [{"valueString": "synthetic-name"}],
                    },
                    {
                        "linkId": "field-private-note",
                        "answer": [{"valueString": "synthetic-secret"}],
                    },
                ],
            },
            {
                "linkId": "section-consent",
                "answer": [
                    {"valueBoolean": True},
                    {"valueBoolean": False},
                ],
            },
        ],
    }


def test_projection_keeps_links_shape_and_order_while_removing_values():
    response = _questionnaire_response()
    original = deepcopy(response)
    policy = {
        "fields": {
            "item[0].item[0].answer[0].valueString": "allow",
            "item[1].answer[0].valueBoolean": "allow",
        }
    }

    projected, summary = project_questionnaire_response_with_summary(response, policy)

    assert response == original
    assert [item["linkId"] for item in projected["item"]] == [
        "section-demographics",
        "section-consent",
    ]
    nested = projected["item"][0]["item"]
    assert [item["linkId"] for item in nested] == [
        "field-display-name",
        "field-private-note",
    ]
    assert nested[0]["answer"] == [{"valueString": "synthetic-name"}]
    assert nested[1]["answer"] == []
    assert projected["item"][1]["answer"] == [{"valueBoolean": True}]
    assert summary == QuestionnaireResponseChangeSummary(
        items_seen=4,
        answers_seen=4,
        answers_removed=2,
        values_removed=2,
        changed_paths=(
            "QuestionnaireResponse.item[0].item[1].answer[0].valueString",
            "QuestionnaireResponse.item[1].answer[1].valueBoolean",
        ),
    )


def test_deny_list_keeps_unlisted_answers_and_preserves_answer_order():
    response = _questionnaire_response()
    policy = {
        "default": "allow",
        "deny": ["QuestionnaireResponse.item[1].answer[0].valueBoolean"],
    }

    projected, summary = project_questionnaire_response_with_summary(response, policy)

    assert projected["item"][1]["answer"] == [{"valueBoolean": False}]
    assert summary.answers_removed == 1
    assert summary.values_removed == 1
    assert summary.changed_paths == (
        "QuestionnaireResponse.item[1].answer[0].valueBoolean",
    )


def test_nested_answer_items_keep_evidence_shape_when_parent_value_is_dropped():
    response = {
        "resourceType": "QuestionnaireResponse",
        "item": [
            {
                "linkId": "section-nested",
                "answer": [
                    {
                        "valueString": "synthetic-parent",
                        "item": [
                            {
                                "linkId": "nested-evidence",
                                "answer": [{"valueCode": "synthetic-code"}],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    policy = {
        "item[0].answer[0].item[0].answer[0].valueCode": "allow",
    }

    projected, summary = project_questionnaire_response_with_summary(response, policy)

    answer = projected["item"][0]["answer"][0]
    assert "valueString" not in answer
    assert answer["item"][0]["linkId"] == "nested-evidence"
    assert answer["item"][0]["answer"] == [{"valueCode": "synthetic-code"}]
    assert summary.answers_removed == 0
    assert summary.changed_paths == (
        "QuestionnaireResponse.item[0].answer[0].valueString",
    )


def test_projection_is_deterministic_and_summary_has_no_answer_values():
    response = _questionnaire_response()
    policy = [
        "QuestionnaireResponse.item[1].answer[0].valueBoolean",
        "QuestionnaireResponse.item[0].item[0].answer[0].valueString",
    ]

    first = project_questionnaire_response_with_summary(response, policy)
    second = project_questionnaire_response_with_summary(response, policy)

    assert first == second
    summary_json = json.dumps(first[1].to_dict(), sort_keys=True)
    assert "synthetic-name" not in summary_json
    assert "synthetic-secret" not in summary_json


@pytest.mark.parametrize(
    "path",
    [
        "item.answer.valueString",
        "item[0].answer.valueString",
        "item[0].answer[0].value[x]",
        "item[*].answer[0].valueString",
    ],
)
def test_ambiguous_policy_paths_fail_closed_without_echoing_payload(path):
    with pytest.raises(AmbiguousPolicyPathError) as excinfo:
        project_questionnaire_response(
            _questionnaire_response(),
            {path: "allow"},
        )

    assert "synthetic-secret" not in str(excinfo.value)


def test_keyword_field_policy_alias_and_empty_default_drop_policy():
    response = _questionnaire_response()

    projected = project_questionnaire_response(
        response,
        field_policy={"default": "drop"},
    )

    assert projected["item"][0]["linkId"] == "section-demographics"
    assert projected["item"][0]["item"][0]["answer"] == []
    assert projected["item"][1]["answer"] == []


def test_direct_deny_mapping_can_set_an_explicit_default():
    projected = project_questionnaire_response(
        _questionnaire_response(),
        {
            "default": "allow",
            "item[1].answer[0].valueBoolean": "drop",
        },
    )

    assert projected["item"][0]["item"][1]["answer"] == [
        {"valueString": "synthetic-secret"}
    ]
    assert projected["item"][1]["answer"] == [{"valueBoolean": False}]
