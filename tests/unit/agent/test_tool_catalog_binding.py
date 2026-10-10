"""Offline catalog binding, replay and content-free negative controls."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from openmed.agent.correlation import RunId
from openmed.agent.tool_catalog_binding import (
    CatalogEligibility,
    RunToolCatalog,
    RunToolCatalogSnapshot,
    ToolCatalogBindingError,
    ToolCatalogReviewRequired,
    ToolImplementationBinding,
    tool_spec_schema_digest,
    tool_spec_side_effect_class,
)
from openmed.agent.tool_inventory import (
    SideEffectClass,
    ToolInventory,
    ToolInventoryRecord,
)
from openmed.mcp.tool_registry import ToolRegistry, ToolSpec

RUN = RunId("run_" + "1" * 32)
TOOL = "tool:org.example/count"
CAPABILITY = "capability:org.example/local-read@1.0.0"
SENTINEL = (
    "Synthetic Patient María 王 https://private.example.test token-secret /private/path"
)


def spec(version="1.0.0", **changes):
    return ToolSpec(
        name="count_local",
        description=SENTINEL,
        version=version,
        input_schema={"type": "object", "properties": {"value": {"type": "integer"}}},
        output_schema={"type": "object", "properties": {"count": {"type": "integer"}}},
        read_only_hint=changes.pop("read_only_hint", True),
        destructive_hint=changes.pop("destructive_hint", False),
        **changes,
    )


def record(contract, tool_id=TOOL):
    return ToolInventoryRecord(
        tool_id,
        contract.version,
        CAPABILITY,
        tool_spec_side_effect_class(contract),
        tool_spec_schema_digest(contract),
    )


def original(value=0):
    return {"count": value + 1}


def replacement(value=0):
    return {"count": value + 100}


def setup_catalog():
    contract = spec()
    registry = ToolRegistry()
    registry.register(contract, handler=original)
    inventory = ToolInventory((record(contract),))
    names = {(TOOL, "1.0.0"): contract.name}
    catalog = RunToolCatalog.capture(RUN, inventory, registry, names)
    return catalog, registry, names


def test_unchanged_snapshot_restores_and_replays_deterministically():
    catalog, registry, names = setup_catalog()
    text = catalog.snapshot.to_json()
    snapshot = RunToolCatalogSnapshot.from_dict(json.loads(text))
    restored = RunToolCatalog.restore(snapshot, registry, names)
    assert snapshot == catalog.snapshot
    assert snapshot.to_json() == text
    assert snapshot.digest == catalog.snapshot.digest
    assert restored.check(registry).to_dict() == catalog.check(registry).to_dict()
    for dispatcher in (catalog, restored):
        assert dispatcher.invoke(registry, TOOL, "1.0.0", {"value": 7}) == {"count": 8}
        assert dispatcher.check(registry).preview_valid
        assert dispatcher.check(registry).resume_eligible


def test_same_name_version_schema_cannot_substitute_new_implementation():
    catalog, _, names = setup_catalog()
    swapped = ToolRegistry()
    swapped.register(spec(), handler=replacement)
    report = catalog.check(swapped)
    assert report.reasons == ("implementation_changed",)
    assert report.to_dict()["status"] == "re-review-required"
    assert not report.preview_valid and not report.resume_eligible
    with pytest.raises(ToolCatalogReviewRequired, match="implementation_changed"):
        catalog.invoke(swapped, TOOL, "1.0.0", {"value": 7})
    with pytest.raises(ToolCatalogReviewRequired, match="implementation_changed"):
        RunToolCatalog.restore(catalog.snapshot, swapped, names)


def test_fresh_registration_of_same_callable_still_requires_review():
    catalog, _, _ = setup_catalog()
    swapped = ToolRegistry()
    swapped.register(spec(), handler=original)
    assert catalog.check(swapped).reasons == ("implementation_changed",)


@pytest.mark.parametrize("replacement_version", [None, "2.0.0"])
def test_removed_tool_or_version_never_falls_back_to_latest(replacement_version):
    catalog, _, names = setup_catalog()
    registry = ToolRegistry()
    if replacement_version:
        registry.register(spec(replacement_version), handler=replacement)
    assert catalog.check(registry).reasons == ("tool_unavailable",)
    with pytest.raises(ToolCatalogReviewRequired, match="tool_unavailable"):
        catalog.invoke(registry, TOOL, "1.0.0", {})
    with pytest.raises(ToolCatalogReviewRequired, match="tool_unavailable"):
        RunToolCatalog.restore(catalog.snapshot, registry, names)


def test_spec_without_registered_handler_is_unavailable():
    catalog, _, names = setup_catalog()
    registry = ToolRegistry((spec(),))
    assert catalog.check(registry).reasons == ("tool_unavailable",)
    with pytest.raises(ToolCatalogReviewRequired, match="tool_unavailable"):
        RunToolCatalog.capture(RUN, ToolInventory((record(spec()),)), registry, names)


@pytest.mark.parametrize("schema_field", ["input_schema", "output_schema"])
def test_nested_schema_mutation_invalidates_preview_and_resume(schema_field):
    catalog, registry, names = setup_catalog()
    getattr(registry.get("count_local"), schema_field)["required"] = ["new_field"]
    assert catalog.check(registry).reasons == ("schema_changed",)
    with pytest.raises(ToolCatalogReviewRequired, match="schema_changed"):
        catalog.invoke(registry, TOOL, "1.0.0", {})
    with pytest.raises(ToolCatalogReviewRequired, match="schema_changed"):
        RunToolCatalog.restore(catalog.snapshot, registry, names)


def test_schema_change_in_replacement_reports_both_drifts():
    catalog, _, _ = setup_catalog()
    changed = spec()
    changed.input_schema["required"] = ["different"]
    registry = ToolRegistry()
    registry.register(changed, handler=replacement)
    assert catalog.check(registry).reasons == (
        "implementation_changed",
        "schema_changed",
    )


def test_side_effect_drift_fails_closed():
    catalog, _, _ = setup_catalog()
    registry = ToolRegistry()
    registry.register(
        spec(read_only_hint=False, destructive_hint=True), handler=original
    )
    assert catalog.check(registry).reasons == (
        "implementation_changed",
        "side_effect_changed",
    )


@pytest.mark.parametrize(
    ("hints", "expected"),
    [
        ({"read_only_hint": True}, SideEffectClass.READ_ONLY),
        (
            {"read_only_hint": False, "destructive_hint": True},
            SideEffectClass.DESTRUCTIVE,
        ),
        (
            {"read_only_hint": False, "idempotent_hint": True},
            SideEffectClass.IDEMPOTENT_WRITE,
        ),
        ({"read_only_hint": False}, SideEffectClass.NON_IDEMPOTENT_WRITE),
    ],
)
def test_registry_side_effect_projection_is_conservative(hints, expected):
    assert tool_spec_side_effect_class(spec(**hints)) is expected


def test_schema_digest_is_canonical_and_covers_both_schemas():
    first = spec()
    reordered = replace(
        first, input_schema=dict(reversed(list(first.input_schema.items())))
    )
    assert tool_spec_schema_digest(first) == tool_spec_schema_digest(reordered)
    assert tool_spec_schema_digest(first) == tool_spec_schema_digest(
        replace(first, description="other")
    )
    changed = replace(first, output_schema={"type": "array"})
    assert tool_spec_schema_digest(first) != tool_spec_schema_digest(changed)


def test_captured_spec_is_detached_from_mutable_registry_contract():
    registry = ToolRegistry()
    registry.register(spec(), handler=original)
    detached, handler, identity = registry.implementation_binding(
        "count_local", "1.0.0"
    )
    detached.input_schema["required"] = ["changed"]
    actual, same_handler, same_id = registry.implementation_binding(
        "count_local", "1.0.0"
    )
    assert "required" not in actual.input_schema
    assert handler is same_handler is original
    assert identity == same_id


def test_registration_and_snapshot_order_do_not_change_review_evidence():
    registry = ToolRegistry()
    registry.register(spec(), handler=original)
    registry.register(spec("2.0.0"), handler=replacement)
    records = (record(spec("2.0.0")), record(spec()))
    names = {(TOOL, version): "count_local" for version in ("1.0.0", "2.0.0")}
    first = RunToolCatalog.capture(RUN, ToolInventory(records), registry, names)
    second = RunToolCatalog.capture(
        RUN, ToolInventory(tuple(reversed(records))), registry, names
    )
    assert first.snapshot.to_json() == second.snapshot.to_json()
    assert first.invoke(registry, TOOL, "1.0.0", {}) == {"count": 1}
    assert first.invoke(registry, TOOL, "2.0.0", {}) == {"count": 100}
    assert (
        replace(first.snapshot, bindings=tuple(reversed(first.snapshot.bindings)))
        == first.snapshot
    )


def test_unreviewed_tool_and_version_are_not_dispatched():
    catalog, registry, _ = setup_catalog()
    for tool_id, version in [(TOOL, "2.0.0"), (SENTINEL, "1.0.0"), ([], "1.0.0")]:
        with pytest.raises(
            ToolCatalogReviewRequired, match="unreviewed_tool"
        ) as caught:
            catalog.invoke(registry, tool_id, version, {})
        assert SENTINEL not in str(caught.value)


def test_empty_catalog_is_stable_and_run_identity_is_bound():
    catalog = RunToolCatalog.capture(RUN, ToolInventory(()), ToolRegistry(), {})
    assert catalog.snapshot.to_json() == (
        '{"bindings":[],"run_id":"run_' + "1" * 32 + '",'
        '"schema_version":"openmed.agent.tool_catalog_binding.v1"}'
    )
    assert (
        replace(catalog.snapshot, run_id=RunId("run_" + "2" * 32)).digest
        != catalog.snapshot.digest
    )


@pytest.mark.parametrize("level", ["snapshot", "binding", "tool"])
@pytest.mark.parametrize(
    "field", ["endpoint", "credential", "arguments", "patient_id", "path", "handler"]
)
def test_content_bearing_fields_are_rejected_at_every_snapshot_level(level, field):
    catalog, _, _ = setup_catalog()
    payload = catalog.snapshot.to_dict()
    target = payload if level == "snapshot" else payload["bindings"][0]
    if level == "tool":
        target = target["tool"]
    target[field] = SENTINEL
    with pytest.raises(ToolCatalogBindingError, match="invalid_snapshot") as caught:
        RunToolCatalogSnapshot.from_dict(payload)
    assert SENTINEL not in str(caught.value) + repr(caught.value)


@pytest.mark.parametrize(
    "field", ["run_id", "schema_version", "implementation_id", "schema_digest"]
)
def test_identifying_snapshot_values_fail_without_echo(field):
    catalog, _, _ = setup_catalog()
    payload = catalog.snapshot.to_dict()
    target = payload
    if field == "implementation_id":
        target = payload["bindings"][0]
    elif field == "schema_digest":
        target = payload["bindings"][0]["tool"]
    target[field] = SENTINEL
    with pytest.raises(ToolCatalogBindingError) as caught:
        RunToolCatalogSnapshot.from_dict(payload)
    assert SENTINEL not in str(caught.value) + repr(caught.value)


def test_representations_and_reports_exclude_runtime_metadata_and_arguments():
    catalog, registry, _ = setup_catalog()
    artifacts = [
        catalog.snapshot.to_json(),
        catalog.snapshot.digest,
        repr(catalog),
        repr(catalog.snapshot),
        repr(catalog.snapshot.bindings[0]),
        json.dumps(catalog.check(registry).to_dict()),
    ]
    for artifact in artifacts:
        assert SENTINEL not in artifact
        assert "count_local" not in artifact
        assert "description" not in artifact
        assert "arguments" not in artifact


def test_handler_failure_has_content_free_exception_and_no_log(caplog):
    def failing(**arguments):
        raise RuntimeError(SENTINEL)

    registry = ToolRegistry()
    registry.register(spec(), handler=failing)
    catalog = RunToolCatalog.capture(
        RUN,
        ToolInventory((record(spec()),)),
        registry,
        {(TOOL, "1.0.0"): "count_local"},
    )
    with pytest.raises(ToolCatalogBindingError, match="invocation_failed") as caught:
        catalog.invoke(registry, TOOL, "1.0.0", {"patient": SENTINEL})
    import traceback

    assert SENTINEL not in "".join(traceback.format_exception(caught.value))
    assert SENTINEL not in caplog.text


def test_invalid_registry_and_schema_are_controlled_failures():
    catalog, registry, _ = setup_catalog()
    assert catalog.check(None).reasons == ("catalog_unavailable",)
    registry.get("count_local").input_schema["default"] = float("nan")
    assert catalog.check(registry).reasons == ("invalid_schema",)


def test_snapshot_and_runtime_selection_are_immutable_and_detached():
    catalog, registry, names = setup_catalog()
    names[(TOOL, "1.0.0")] = SENTINEL
    assert catalog.invoke(registry, TOOL, "1.0.0", {}) == {"count": 1}
    with pytest.raises(FrozenInstanceError):
        catalog.snapshot = None
    with pytest.raises(FrozenInstanceError):
        catalog.snapshot.bindings = ()
    with pytest.raises(TypeError):
        catalog._runtime[(TOOL, "1.0.0")] = None


def test_duplicate_bindings_and_invalid_implementation_ids_are_rejected():
    catalog, _, _ = setup_catalog()
    item = catalog.snapshot.bindings[0]
    with pytest.raises(ToolCatalogBindingError, match="invalid_bindings"):
        replace(catalog.snapshot, bindings=(item, item))
    with pytest.raises(ToolCatalogBindingError, match="invalid_bindings"):
        ToolImplementationBinding(item.tool, SENTINEL)
    with pytest.raises(ToolCatalogBindingError, match="invalid_bindings"):
        CatalogEligibility(catalog.snapshot.digest, (SENTINEL,))


@pytest.mark.parametrize(
    "names", [{}, {(TOOL, "2.0.0"): "count_local"}, {(TOOL, "1.0.0"): 1}]
)
def test_incomplete_or_malformed_selection_is_rejected(names):
    _, registry, _ = setup_catalog()
    with pytest.raises(ToolCatalogBindingError, match="invalid_selection"):
        RunToolCatalog.capture(RUN, ToolInventory((record(spec()),)), registry, names)


def test_capture_rejects_unreviewed_schema_and_risk_class():
    _, registry, names = setup_catalog()
    tool = record(spec())
    for changed, code in [
        (replace(tool, schema_digest="sha256:" + "0" * 64), "schema_changed"),
        (replace(tool, side_effect_class=SideEffectClass.NONE), "side_effect_changed"),
    ]:
        with pytest.raises(ToolCatalogReviewRequired, match=code):
            RunToolCatalog.capture(RUN, ToolInventory((changed,)), registry, names)
