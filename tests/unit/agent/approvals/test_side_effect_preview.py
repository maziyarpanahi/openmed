"""Offline tests for value-free clinical write previews."""

import dataclasses

import pytest

from openmed.agent.approvals.side_effect_preview import (
    PreviewError,
    ResourceWrite,
    WorkflowState,
    WriteIntent,
    WriteKind,
    render_side_effect_preview,
    require_current_preview,
)

_SECRET = b"local-review-key-with-at-least-32-bytes"
_SENSITIVE = "synthetic-sensitive-clinical-value"


def _intent() -> WriteIntent:
    return WriteIntent(
        action_id="11111111-1111-4111-8111-111111111111",
        writes=(
            ResourceWrite(
                WriteKind.UPDATE,
                "Observation",
                "res_" + "a" * 32,
                {"status": "preliminary", "valueString": _SENSITIVE},
                {"status": "final", "valueString": "synthetic-replacement"},
            ),
            ResourceWrite(
                WriteKind.CREATE,
                "Task",
                "res_" + "b" * 32,
                None,
                {"status": "requested"},
            ),
        ),
        workflow_before=WorkflowState.AWAITING_REVIEW,
        workflow_after=WorkflowState.APPROVED,
    )


def _check(intent: WriteIntent, digest: str, observed: tuple) -> None:
    require_current_preview(
        intent,
        observed_before=observed,
        observed_workflow_state=WorkflowState.AWAITING_REVIEW,
        approved_digest=digest,
        secret=_SECRET,
    )


def test_deterministic_redacted_preview_and_matching_fresh_state() -> None:
    intent = _intent()
    preview = render_side_effect_preview(intent, secret=_SECRET)
    assert preview == render_side_effect_preview(intent, secret=_SECRET)
    assert [resource.resource_type for resource in preview.resources] == [
        "Observation",
        "Task",
    ]
    assert [
        (field.path, field.before, field.after) for field in preview.resources[0].fields
    ] == [
        ("status", "redacted", "redacted"),
        ("valueString", "redacted", "redacted"),
    ]
    assert preview.resources[1].fields[0].before == "absent"
    assert preview.workflow_before is WorkflowState.AWAITING_REVIEW
    assert preview.workflow_after is WorkflowState.APPROVED
    for text in (repr(intent), repr(intent.writes[0]), repr(preview), str(preview)):
        assert _SENSITIVE not in text
    _check(intent, preview.digest, (dict(intent.writes[0].before), None))


@pytest.mark.parametrize(
    "change",
    ["before", "after", "action", "workflow", "order", "key"],
)
def test_preview_commitment_rejects_every_material_change(change: str) -> None:
    intent = _intent()
    preview = render_side_effect_preview(intent, secret=_SECRET)
    before = dict(intent.writes[0].before)
    if change == "before":
        before["status"] = "amended"
    elif change == "after":
        write = intent.writes[0]
        intent = dataclasses.replace(
            intent,
            writes=(
                dataclasses.replace(write, after={"status": "cancelled"}),
                intent.writes[1],
            ),
        )
    elif change == "action":
        intent = dataclasses.replace(
            intent, action_id="22222222-2222-4222-8222-222222222222"
        )
    elif change == "workflow":
        intent = dataclasses.replace(intent, workflow_after=WorkflowState.CANCELLED)
    elif change == "order":
        intent = dataclasses.replace(intent, writes=tuple(reversed(intent.writes)))
    secret = (
        _SECRET if change != "key" else b"another-local-review-key-at-least-32-bytes"
    )
    observed = (before, None) if change != "order" else (None, before)
    with pytest.raises(PreviewError, match="stale_preview"):
        require_current_preview(
            intent,
            observed_before=observed,
            observed_workflow_state=WorkflowState.AWAITING_REVIEW,
            approved_digest=preview.digest,
            secret=secret,
        )


def test_changed_workflow_state_forces_fresh_review() -> None:
    intent = _intent()
    preview = render_side_effect_preview(intent, secret=_SECRET)
    with pytest.raises(PreviewError, match="stale_preview"):
        require_current_preview(
            intent,
            observed_before=(intent.writes[0].before, None),
            observed_workflow_state=WorkflowState.IN_PROGRESS,
            approved_digest=preview.digest,
            secret=_SECRET,
        )


def test_mapping_order_does_not_change_preview_digest() -> None:
    intent = _intent()
    write = intent.writes[0]
    reordered = dataclasses.replace(
        intent,
        writes=(
            dataclasses.replace(
                write,
                before={"valueString": _SENSITIVE, "status": "preliminary"},
                after={"valueString": "synthetic-replacement", "status": "final"},
            ),
            intent.writes[1],
        ),
    )
    assert render_side_effect_preview(
        intent, secret=_SECRET
    ) == render_side_effect_preview(reordered, secret=_SECRET)


def test_create_delete_and_value_free_validation_errors() -> None:
    delete = ResourceWrite(
        WriteKind.DELETE,
        "Task",
        "res_" + "c" * 32,
        {"status": _SENSITIVE},
        None,
    )
    intent = dataclasses.replace(_intent(), writes=(delete,))
    preview = render_side_effect_preview(intent, secret=_SECRET)
    assert preview.resources[0].fields[0].after == "absent"
    assert _SENSITIVE not in repr(preview)
    with pytest.raises(PreviewError) as error:
        ResourceWrite(
            WriteKind.UPDATE,
            "Observation",
            "res_" + "d" * 32,
            {"status": object()},
            {"status": _SENSITIVE},
        )
    assert _SENSITIVE not in str(error.value)
    with pytest.raises(PreviewError, match="invalid_preview_key"):
        render_side_effect_preview(intent, secret=b"short")
    with pytest.raises(PreviewError, match="invalid_approved_digest"):
        _check(intent, _SENSITIVE, (delete.before,))


def test_input_values_are_frozen_after_preview() -> None:
    before = {"status": "preliminary"}
    after = {"status": "final", "code": {"text": "synthetic"}}
    write = ResourceWrite(
        WriteKind.UPDATE,
        "Observation",
        "res_" + "d" * 32,
        before,
        after,
    )
    intent = dataclasses.replace(_intent(), writes=(write,))
    preview = render_side_effect_preview(intent, secret=_SECRET)
    before["status"] = "amended"
    after["code"]["text"] = "changed"
    assert render_side_effect_preview(intent, secret=_SECRET) == preview
    with pytest.raises(TypeError):
        write.after["status"] = "cancelled"
    with pytest.raises(TypeError):
        write.after["code"]["text"] = "changed"
    _check(intent, preview.digest, ({"status": "preliminary"},))
