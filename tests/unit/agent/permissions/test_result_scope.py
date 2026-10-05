"""Offline authorization tests with independently corrupted result metadata."""

import json
from dataclasses import replace

import pytest

from openmed.agent.permissions import AccessTicketExpiredError, AccessTicketVerifier
from openmed.agent.permissions.result_scope import (
    ResultQuarantineCode,
    ResultQuarantinedError,
    ResultScope,
    ToolResultPage,
    ToolResultRecord,
    authorize_tool_results,
    dispatch_with_authorized_results,
)
from openmed.agent.tools.data_projection import DataProjectionDeniedError
from tests.fixtures.agent.result_scope import binding, pages, record, schema, selector


def authorize(output, **kwargs):
    ticket, request, scope = binding()
    return authorize_tool_results(
        ticket,
        request,
        AccessTicketVerifier(clock=lambda: 1),
        scope=scope,
        schema=schema(),
        pages=output(scope),
        **kwargs,
    )


def test_projected_copy_preserves_evidence_without_aliasing_protected_fields():
    ticket, request, scope = binding()
    original = pages(scope)
    result = authorize_tool_results(
        ticket,
        request,
        AccessTicketVerifier(clock=lambda: 1),
        scope=scope,
        schema=schema(),
        pages=original,
    )
    assert result == original
    assert result is not original
    for old_page, new_page in zip(original, result):
        for old, new in zip(old_page.records, new_page.records):
            assert new.evidence is old.evidence
            assert new.scope is old.scope
            assert new.fields is not old.fields
            assert new.fields["observations"] is not old.fields["observations"]
    assert result[0].records[0].children[0].evidence is (
        original[0].records[0].children[0].evidence
    )
    original[0].records[0].fields["observations"][0]["text"] = "changed"
    assert result[0].records[0].fields["observations"][0]["text"] == "synthetic finding"


@pytest.mark.parametrize("dimension", ["patient", "encounter", "namespace", "snapshot"])
@pytest.mark.parametrize("location", ["page", "record", "nested"])
def test_independent_scope_corruption_is_quarantined(dimension, location):
    def output(scope):
        wrong = replace(scope, **{dimension: selector(dimension, "synthetic-other")})
        item = record(scope)
        if location == "page":
            return (ToolResultPage(wrong, (item,)),)
        if location == "record":
            item = replace(item, scope=wrong)
        else:
            item = replace(item, children=(record(wrong, 2),))
        return (ToolResultPage(scope, (item,)),)

    with pytest.raises(ResultQuarantinedError) as caught:
        authorize(output)
    assert caught.value.code is ResultQuarantineCode.SCOPE_MISMATCH


@pytest.mark.parametrize("location", ["page", "record", "nested", "later_page"])
def test_missing_scope_never_inherits_parent_or_first_page(location):
    def output(scope):
        item = record(scope)
        if location == "page":
            return (ToolResultPage(None, (item,)),)
        if location == "record":
            item = replace(item, scope=None)
        if location == "nested":
            item = replace(item, children=(replace(record(scope, 2), scope=None),))
        if location == "later_page":
            return (
                ToolResultPage(scope, (item,), 0, False),
                ToolResultPage(None, (record(scope, 2),), 1, True),
            )
        return (ToolResultPage(scope, (item,)),)

    with pytest.raises(ResultQuarantinedError) as caught:
        authorize(output)
    assert caught.value.code is ResultQuarantineCode.MISSING_SCOPE


def test_ambiguous_selector_roles_and_swapped_roles_fail_closed():
    for ambiguous, code in (
        (
            lambda s: replace(s, encounter=s.patient),
            ResultQuarantineCode.AMBIGUOUS_SCOPE,
        ),
        (
            lambda s: replace(s, patient=s.encounter, encounter=s.patient),
            ResultQuarantineCode.SCOPE_MISMATCH,
        ),
    ):
        with pytest.raises(ResultQuarantinedError) as caught:
            authorize(lambda s: (ToolResultPage(s, (record(ambiguous(s)),)),))
        assert caught.value.code is code


@pytest.mark.parametrize(
    "fields",
    [
        {"observations": [{"text": "synthetic finding"}], "extra": "synthetic secret"},
        {"observations": [{"text": "synthetic finding", "extra": "synthetic secret"}]},
        {"observations": [{"text": {"hidden": "synthetic secret"}}]},
        {"observations": [{"text": ["synthetic secret"]}]},
        {
            "observations": [
                {"text": "synthetic finding"},
                {"extra": "synthetic secret"},
            ]
        },
        {"observations": [{}]},
        {},
        {"observations": "synthetic finding"},
        {"observations": [{1: "synthetic secret"}]},
    ],
)
def test_closed_projection_rejects_extra_missing_and_shape_changed_fields(fields):
    with pytest.raises(ResultQuarantinedError):
        authorize(lambda s: (ToolResultPage(s, (replace(record(s), fields=fields),)),))


@pytest.mark.parametrize(
    "output",
    [
        lambda s: (),
        lambda s: [ToolResultPage(s, (record(s),))],
        lambda s: (ToolResultPage(s, (record(s),), 1, True),),
        lambda s: (ToolResultPage(s, (record(s),), False, True),),
        lambda s: (ToolResultPage(s, (record(s),), 0, False),),
        lambda s: (ToolResultPage(s, (), 0, True), ToolResultPage(s, (), 1, True)),
        lambda s: (ToolResultPage(s, (record(s),)), ToolResultPage(s, (), 2, True)),
    ],
)
def test_missing_ambiguous_or_incomplete_pagination_fails_closed(output):
    with pytest.raises(ResultQuarantinedError) as caught:
        authorize(output)
    assert caught.value.code is ResultQuarantineCode.INCOMPLETE_PAGES


@pytest.mark.parametrize(
    "output,code",
    [
        (
            lambda s: (ToolResultPage(s, (replace(record(s), evidence=None),)),),
            ResultQuarantineCode.MISSING_EVIDENCE,
        ),
        (
            lambda s: (ToolResultPage(s, (record(s), record(s))),),
            ResultQuarantineCode.AMBIGUOUS_EVIDENCE,
        ),
        (
            lambda s: (
                ToolResultPage(s, (replace(record(s), children=(record(s),)),)),
            ),
            ResultQuarantineCode.AMBIGUOUS_EVIDENCE,
        ),
        (
            lambda s: (ToolResultPage(s, ({"scope": s},)),),
            ResultQuarantineCode.INVALID_RESULT,
        ),
        (
            lambda s: (ToolResultPage(s, [record(s)]),),
            ResultQuarantineCode.INVALID_RESULT,
        ),
        (
            lambda s: (ToolResultPage(s, (replace(record(s), children=None),)),),
            ResultQuarantineCode.INVALID_RESULT,
        ),
    ],
)
def test_evidence_and_typed_record_boundaries(output, code):
    with pytest.raises(ResultQuarantinedError) as caught:
        authorize(output)
    assert caught.value.code is code


def test_empty_complete_result_still_requires_page_scope():
    assert authorize(lambda s: (ToolResultPage(s, ()),))[0].records == ()


@pytest.mark.parametrize("which", ["pages", "fields", "depth", "cycle"])
def test_bounded_traversal(which):
    def output(scope):
        if which == "pages":
            return tuple(ToolResultPage(scope, (), i, i == 64) for i in range(65))
        if which == "fields":
            item = replace(
                record(scope),
                fields={"observations": [{"text": "synthetic"} for _ in range(10_000)]},
            )
        else:
            item = record(scope, 100)
            if which == "cycle":
                values = []
                values.append(values)
                item = replace(item, fields={"observations": values})
            else:
                for index in range(1, 35):
                    item = replace(record(scope, index), children=(item,))
        return (ToolResultPage(scope, (item,)),)

    with pytest.raises(ResultQuarantinedError):
        authorize(output)


def test_request_and_projection_cannot_expand_to_broader_ticket():
    ticket, request, scope = binding()
    other = selector("patient", "synthetic-other")
    ticket = replace(ticket, record_selectors=(*ticket.record_selectors, other))
    with pytest.raises(ResultQuarantinedError):
        authorize_tool_results(
            ticket,
            request,
            AccessTicketVerifier(clock=lambda: 1),
            scope=scope,
            schema=schema(),
            pages=(ToolResultPage(scope, (record(replace(scope, patient=other)),)),),
        )
    output_schema = schema()
    output_schema["properties"]["observations"]["x-openmed-data-class"] = (
        "data:org.example/extra@1.0.0"
    )
    ticket = replace(
        ticket,
        permitted_data_classes=(
            *ticket.permitted_data_classes,
            "data:org.example/extra@1.0.0",
        ),
    )
    with pytest.raises(DataProjectionDeniedError):
        authorize_tool_results(
            ticket,
            request,
            AccessTicketVerifier(clock=lambda: 1),
            scope=scope,
            schema=output_schema,
            pages=pages(scope),
        )


def test_missing_requested_dimension_and_expected_binding_fail_before_read():
    ticket, request, scope = binding()
    request = replace(request, record_selectors=(scope.patient,))
    calls = []
    with pytest.raises(ResultQuarantinedError) as caught:
        dispatch_with_authorized_results(
            ticket,
            request,
            AccessTicketVerifier(clock=lambda: 1),
            scope=scope,
            schema=schema(),
            read=lambda: calls.append("read"),
            consume=lambda _: None,
        )
    assert caught.value.code is ResultQuarantineCode.INVALID_BINDING
    assert calls == []


def test_read_time_expiry_blocks_next_step():
    ticket, request, scope = binding()
    current_time = [1]
    calls = []

    def read():
        current_time[0] = 100
        return pages(scope)

    with pytest.raises(AccessTicketExpiredError):
        dispatch_with_authorized_results(
            ticket,
            request,
            AccessTicketVerifier(clock=lambda: current_time[0]),
            scope=scope,
            schema=schema(),
            read=read,
            consume=lambda _: calls.append("next"),
        )
    assert calls == []


def test_read_cannot_widen_the_reviewed_schema():
    ticket, request, scope = binding()
    reviewed_schema = schema()

    def read():
        properties = reviewed_schema["properties"]["observations"]["items"][
            "properties"
        ]
        properties["extra"] = properties["text"]
        item = record(scope)
        item.fields["observations"][0]["extra"] = "synthetic secret"
        return (ToolResultPage(scope, (item,)),)

    with pytest.raises(ResultQuarantinedError) as caught:
        dispatch_with_authorized_results(
            ticket,
            request,
            AccessTicketVerifier(clock=lambda: 1),
            scope=scope,
            schema=reviewed_schema,
            read=read,
            consume=lambda _: pytest.fail("exposed"),
        )
    assert caught.value.code is ResultQuarantineCode.OVERBROAD_OUTPUT


def test_quarantine_diagnostics_and_representations_are_content_free(caplog):
    ticket, request, scope = binding()
    secret = "synthetic secret /private/synthetic credential"
    item = replace(record(scope), fields={secret: secret})
    with pytest.raises(ResultQuarantinedError) as caught:
        authorize_tool_results(
            ticket,
            request,
            AccessTicketVerifier(clock=lambda: 1),
            scope=scope,
            schema=schema(),
            pages=(ToolResultPage(scope, (item,)),),
        )
    diagnostic = (
        str(caught.value) + repr(caught.value) + json.dumps(caught.value.to_dict())
    )
    diagnostic += repr(scope) + repr(item) + repr(ToolResultPage(scope, (item,)))
    assert secret not in diagnostic + caplog.text
    assert scope.patient.digest not in diagnostic
    assert item.evidence.artifact_id not in diagnostic
    assert caplog.text == ""


def test_provider_error_is_sanitized():
    ticket, request, scope = binding()

    def read():
        raise RuntimeError("synthetic secret provider error")

    with pytest.raises(ResultQuarantinedError) as caught:
        dispatch_with_authorized_results(
            ticket,
            request,
            AccessTicketVerifier(clock=lambda: 1),
            scope=scope,
            schema=schema(),
            read=read,
            consume=lambda _: pytest.fail("exposed"),
        )
    assert caught.value.code is ResultQuarantineCode.READ_FAILED
    assert "synthetic secret" not in str(caught.value)
    assert caught.value.__suppress_context__
