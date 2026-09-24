"""Offline and content-free FHIR transaction compensation tests."""

from __future__ import annotations

import socket

import pytest

from openmed.interop.fhir.compensation_report import (
    CompensationAction,
    EffectClass,
    build_compensation_report,
)

_PRIVATE = "private-marker"


def _request(*methods: str) -> dict:
    return {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {
                "fullUrl": f"urn:uuid:synthetic-{index}",
                "request": {"method": method, "url": f"Observation/{_PRIVATE}"},
                "resource": {"resourceType": "Observation", "valueString": _PRIVATE},
            }
            for index, method in enumerate(methods)
        ],
    }


def _response(*statuses: str, locations: tuple[bool, ...] = ()) -> dict:
    return {
        "resourceType": "Bundle",
        "type": "transaction-response",
        "entry": [
            {
                "response": {
                    "status": status,
                    **(
                        {"location": f"Observation/{_PRIVATE}"}
                        if index < len(locations) and locations[index]
                        else {}
                    ),
                }
            }
            for index, status in enumerate(statuses)
        ],
    }


def test_mixed_result_proposes_review_only_actions_without_content() -> None:
    packet = build_compensation_report(
        _request("POST", "PUT", "DELETE", "PATCH"),
        _response(
            "201 Created",
            "200 OK",
            "204 No Content",
            "412 Precondition Failed",
            locations=(True,),
        ),
    )
    assert packet.has_partial_failure
    assert packet.approval_required
    assert [effect.classification for effect in packet.effects] == [
        EffectClass.REVERSIBLE,
        EffectClass.REVIEW_REQUIRED,
        EffectClass.IRREVERSIBLE,
        EffectClass.REVIEW_REQUIRED,
    ]
    assert [effect.proposed_action for effect in packet.effects] == [
        CompensationAction.VERIFY_AND_CONSIDER_DELETE,
        CompensationAction.REVIEW_PRIOR_VERSION,
        CompensationAction.REVIEW_DELETION,
        CompensationAction.RECONCILE_SERVER_STATE,
    ]
    assert _PRIVATE not in repr(packet)


def test_missing_or_unusable_responses_never_imply_safe_reversal() -> None:
    request = _request("POST", "PUT")
    for response in (None, _response("500 Server Error"), _response("201 Created")):
        packet = build_compensation_report(request, response)
        assert all(
            effect.classification is EffectClass.REVIEW_REQUIRED
            for effect in packet.effects
        )
        assert _PRIVATE not in repr(packet)
    assert not build_compensation_report(request, None).has_partial_failure


def test_ambiguous_correlation_and_malformed_status_require_review() -> None:
    request = _request("POST")
    response = _response("201 Created", locations=(True,))
    response["entry"][0]["fullUrl"] = _PRIVATE
    effect = build_compensation_report(request, response).effects[0]
    assert effect.classification is EffectClass.REVIEW_REQUIRED
    response["entry"][0].pop("fullUrl")
    response["entry"][0]["response"]["status"] = "201\r\n" + _PRIVATE
    effect = build_compensation_report(request, response).effects[0]
    assert effect.classification is EffectClass.REVIEW_REQUIRED
    assert _PRIVATE not in repr(effect)


def test_deterministic_offline_packet(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_socket(*args, **kwargs):
        raise AssertionError("unexpected network access")

    monkeypatch.setattr(socket, "socket", fail_socket)
    request = _request("POST", "DELETE")
    response = _response("201 Created", "503 Unavailable", locations=(True,))
    assert build_compensation_report(request, response) == build_compensation_report(
        request, response
    )


def test_batch_response_uses_the_same_conservative_classification() -> None:
    request = _request("POST", "DELETE")
    response = _response("201 Created", "503 Unavailable", locations=(True,))
    request["type"] = "batch"
    response["type"] = "batch-response"
    packet = build_compensation_report(request, response)
    assert packet.has_partial_failure
    assert [effect.classification for effect in packet.effects] == [
        EffectClass.REVERSIBLE,
        EffectClass.REVIEW_REQUIRED,
    ]


@pytest.mark.parametrize(
    "intended,received",
    [
        ({"resourceType": "Bundle", "type": "transaction", "entry": {}}, None),
        (_request("GET"), None),
        (
            {
                "resourceType": "Bundle",
                "type": "transaction",
                "entry": [{"request": {"method": "POST"}}],
            },
            None,
        ),
        (
            _request("POST"),
            {"resourceType": "Bundle", "type": "batch-response", "entry": []},
        ),
        (_request("POST"), _response("200 OK", "201 Created")),
    ],
)
def test_invalid_shapes_raise_value_free_errors(
    intended: dict, received: dict | None
) -> None:
    with pytest.raises(ValueError) as error:
        build_compensation_report(intended, received)
    assert _PRIVATE not in str(error.value)
