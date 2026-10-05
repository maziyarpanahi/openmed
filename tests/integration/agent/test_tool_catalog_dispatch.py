"""Synthetic local dispatch through reviewed catalogs during reloads."""

import pytest

from openmed.agent.correlation import RunId
from openmed.agent.tool_catalog_binding import (
    RunToolCatalog,
    ToolCatalogReviewRequired,
    tool_spec_schema_digest,
)
from openmed.agent.tool_inventory import (
    SideEffectClass,
    ToolInventory,
    ToolInventoryRecord,
)
from openmed.mcp.tool_registry import ToolRegistry, ToolSpec


def test_reload_between_preview_and_effect_requires_review_before_dispatch():
    calls = []

    def first():
        calls.append("first")
        return {"count": 1}

    def second():
        calls.append("second")
        return {"count": 2}

    spec = ToolSpec(
        "synthetic_effect",
        "local synthetic effect",
        {"type": "object"},
        {"type": "object"},
        destructive_hint=False,
        idempotent_hint=True,
    )
    tool_id = "tool:org.example/synthetic-effect"
    inventory = ToolInventory(
        (
            ToolInventoryRecord(
                tool_id,
                spec.version,
                "capability:org.example/local-write@1.0.0",
                SideEffectClass.IDEMPOTENT_WRITE,
                tool_spec_schema_digest(spec),
            ),
        )
    )
    old = ToolRegistry()
    old.register(spec, handler=first)
    new = ToolRegistry()
    new.register(spec, handler=second)
    names = {(tool_id, spec.version): spec.name}
    catalog = RunToolCatalog.capture(RunId("run_" + "1" * 32), inventory, old, names)
    reviewed_digest = catalog.snapshot.digest
    assert catalog.check(old).preview_valid

    with pytest.raises(ToolCatalogReviewRequired, match="implementation_changed"):
        catalog.invoke(new, tool_id, spec.version, {})
    assert calls == []
    assert not catalog.check(new).resume_eligible
    assert catalog.snapshot.digest == reviewed_digest

    # A new review explicitly binds the new registration; no automatic refresh.
    reviewed = RunToolCatalog.capture(RunId("run_" + "2" * 32), inventory, new, names)
    assert reviewed.invoke(new, tool_id, spec.version, {}) == {"count": 2}
    assert calls == ["second"]


def test_reload_after_final_lookup_cannot_replace_captured_callable():
    calls = []

    def first():
        calls.append("reviewed")
        return {"count": 1}

    def second():
        calls.append("substituted")
        return {"count": 2}

    class ReloadingRegistry(ToolRegistry):
        armed = False

        def implementation_binding(self, name, version):
            binding = super().implementation_binding(name, version)
            if self.armed:
                # Model a catalog change precisely after the availability check.
                self._handlers[(name, version)] = second
            return binding

    spec = ToolSpec(
        "synthetic_read",
        "offline",
        {"type": "object"},
        {"type": "object"},
        read_only_hint=True,
    )
    registry = ReloadingRegistry()
    registry.register(spec, handler=first)
    tool_id = "tool:org.example/synthetic-read"
    inventory = ToolInventory(
        (
            ToolInventoryRecord(
                tool_id,
                spec.version,
                "capability:org.example/local-read@1.0.0",
                SideEffectClass.READ_ONLY,
                tool_spec_schema_digest(spec),
            ),
        )
    )
    catalog = RunToolCatalog.capture(
        RunId("run_" + "3" * 32),
        inventory,
        registry,
        {(tool_id, spec.version): spec.name},
    )
    registry.armed = True
    assert catalog.invoke(registry, tool_id, spec.version, {}) == {"count": 1}
    assert calls == ["reviewed"]
    assert not catalog.check(registry).preview_valid
