"""Run-scoped, content-free bindings to exact registered implementations.

The serializable snapshot contains governance metadata only. Executable
references and runtime names stay in a separate, in-memory dispatch object.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from openmed.agent.correlation import RunId
from openmed.agent.tool_inventory import (
    MAX_TOOL_INVENTORY_RECORDS,
    SideEffectClass,
    ToolInventory,
    ToolInventoryError,
    ToolInventoryRecord,
)
from openmed.mcp.tool_registry import ToolRegistry, ToolSpec

TOOL_CATALOG_BINDING_SCHEMA_VERSION: Final = "openmed.agent.tool_catalog_binding.v1"
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_REASONS = frozenset(
    {
        "invalid_snapshot",
        "invalid_bindings",
        "invalid_selection",
        "invalid_schema",
        "tool_unavailable",
        "implementation_changed",
        "schema_changed",
        "side_effect_changed",
        "catalog_unavailable",
        "unreviewed_tool",
        "invocation_failed",
    }
)


class ToolCatalogBindingError(ValueError):
    """Fail closed with a controlled code and no rejected metadata.

    Args:
        code: One of the module's fixed reason codes.
    """

    def __init__(self, code: str) -> None:
        self.code = (
            code if type(code) is str and code in _REASONS else "invalid_bindings"
        )
        super().__init__(self.code)


class ToolCatalogReviewRequired(ToolCatalogBindingError):
    """A reviewed catalog can no longer dispatch or resume safely."""


@dataclass(frozen=True, slots=True, repr=False)
class ToolImplementationBinding:
    """One reviewed inventory record and opaque registration identity."""

    tool: ToolInventoryRecord
    implementation_id: str

    def __post_init__(self) -> None:
        if type(self.tool) is not ToolInventoryRecord or not _is_digest(
            self.implementation_id
        ):
            raise ToolCatalogBindingError("invalid_bindings")

    def to_dict(self) -> dict[str, Any]:
        """Return metadata only, without runtime names or executable objects."""

        return {
            "tool": self.tool.to_dict(),
            "implementation_id": self.implementation_id,
        }

    def __repr__(self) -> str:
        return "ToolImplementationBinding(<content-free>)"


@dataclass(frozen=True, slots=True, repr=False)
class RunToolCatalogSnapshot:
    """Immutable review evidence for the exact implementations of one run."""

    run_id: RunId
    bindings: tuple[ToolImplementationBinding, ...]
    schema_version: str = TOOL_CATALOG_BINDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.run_id) is not RunId
            or self.schema_version != TOOL_CATALOG_BINDING_SCHEMA_VERSION
            or type(self.bindings) is not tuple
            or len(self.bindings) > MAX_TOOL_INVENTORY_RECORDS
            or not all(
                type(item) is ToolImplementationBinding for item in self.bindings
            )
        ):
            raise ToolCatalogBindingError("invalid_snapshot")
        try:
            inventory = ToolInventory(tuple(item.tool for item in self.bindings))
        except ToolInventoryError:
            raise ToolCatalogBindingError("invalid_bindings") from None
        by_key = {_key(item.tool): item for item in self.bindings}
        object.__setattr__(
            self, "bindings", tuple(by_key[_key(tool)] for tool in inventory.tools)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the fixed content-free snapshot projection."""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id.serialize(),
            "bindings": [item.to_dict() for item in self.bindings],
        }

    def to_json(self) -> str:
        """Return canonical, byte-stable JSON for unchanged snapshots."""

        return _canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        """Return a domain-separated review reference, including the run ID."""

        return _digest("openmed.agent.run-tool-catalog.v1\0" + self.to_json())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunToolCatalogSnapshot:
        """Restore evidence while rejecting every content-bearing extra field.

        Args:
            payload: The exact mapping returned by ``to_dict``.

        Returns:
            Validated immutable evidence. This alone grants no dispatch authority.

        Raises:
            ToolCatalogBindingError: If the mapping is not a supported snapshot.
        """

        try:
            values = _fixed_mapping(payload, {"schema_version", "run_id", "bindings"})
            items = values["bindings"]
            if type(items) is not list or len(items) > MAX_TOOL_INVENTORY_RECORDS:
                raise ToolCatalogBindingError("invalid_snapshot")
            bindings = []
            for item in items:
                entry = _fixed_mapping(item, {"tool", "implementation_id"})
                bindings.append(
                    ToolImplementationBinding(
                        ToolInventoryRecord.from_dict(entry["tool"]),
                        entry["implementation_id"],
                    )
                )
            return cls(
                RunId.parse(values["run_id"]),
                tuple(bindings),
                values["schema_version"],
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise ToolCatalogBindingError("invalid_snapshot") from None

    def __repr__(self) -> str:
        return f"RunToolCatalogSnapshot(tool_count={len(self.bindings)})"


@dataclass(frozen=True, slots=True)
class CatalogEligibility:
    """Deterministic content-free preview and resume eligibility."""

    snapshot_digest: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not _is_digest(self.snapshot_digest)
            or type(self.reasons) is not tuple
            or any(
                type(reason) is not str or reason not in _REASONS
                for reason in self.reasons
            )
        ):
            raise ToolCatalogBindingError("invalid_bindings")
        object.__setattr__(self, "reasons", tuple(sorted(set(self.reasons))))

    @property
    def preview_valid(self) -> bool:
        """Return whether the reviewed implementations remain available."""

        return not self.reasons

    @property
    def resume_eligible(self) -> bool:
        """Return whether the original review may be reused on resume."""

        return self.preview_valid

    def to_dict(self) -> dict[str, Any]:
        """Return controlled codes and a digest, never runtime metadata."""

        return {
            "snapshot_digest": self.snapshot_digest,
            "status": "eligible" if self.preview_valid else "re-review-required",
            "preview_valid": self.preview_valid,
            "resume_eligible": self.resume_eligible,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True, repr=False)
class _RuntimeBinding:
    name: str
    handler: Callable[..., Mapping[str, Any]]


@dataclass(frozen=True, slots=True, repr=False, init=False)
class RunToolCatalog:
    """In-memory dispatch exclusively through a reviewed run snapshot.

    Construct with ``capture`` for review, or ``restore`` with existing review
    evidence. Check against the current registry immediately before dispatch;
    invocation always calls the captured handler, never a fresh name lookup.
    """

    snapshot: RunToolCatalogSnapshot
    _runtime: Mapping[tuple[str, str], _RuntimeBinding]

    @classmethod
    def capture(
        cls,
        run_id: RunId,
        inventory: ToolInventory,
        registry: ToolRegistry,
        tool_names: Mapping[tuple[str, str], str],
    ) -> RunToolCatalog:
        """Pin the inventory to executable registrations for a new review.

        Args:
            run_id: Opaque run correlation identifier, never patient-derived.
            inventory: Reviewed canonical tool metadata and schema digests.
            registry: Existing registry with exact-version executable handlers.
            tool_names: Exact mapping from inventory keys to runtime names.
                Names remain in memory and are excluded from evidence.

        Returns:
            A frozen run catalog retaining only the selected registrations.

        Raises:
            ToolCatalogBindingError: If selection or inventory is invalid.
            ToolCatalogReviewRequired: If an implementation or contract differs.
        """

        if type(run_id) is not RunId or type(inventory) is not ToolInventory:
            raise ToolCatalogBindingError("invalid_snapshot")
        names = _selection(tool_names, tuple(_key(tool) for tool in inventory.tools))
        bindings = []
        runtime = {}
        for tool in inventory.tools:
            spec, handler, identity = _lookup(registry, names[_key(tool)], tool.version)
            reason = _contract_reason(tool, spec)
            if reason:
                raise ToolCatalogReviewRequired(reason)
            bindings.append(ToolImplementationBinding(tool, identity))
            runtime[_key(tool)] = _RuntimeBinding(names[_key(tool)], handler)
        return cls._build(RunToolCatalogSnapshot(run_id, tuple(bindings)), runtime)

    @classmethod
    def restore(
        cls,
        snapshot: RunToolCatalogSnapshot,
        registry: ToolRegistry,
        tool_names: Mapping[tuple[str, str], str],
    ) -> RunToolCatalog:
        """Rebind evidence only when the original registrations still exist.

        Args:
            snapshot: Validated evidence from the original review.
            registry: Current registry; replacement registrations fail closed.
            tool_names: Exact in-memory runtime-name mapping for the snapshot.

        Returns:
            A dispatch catalog using the same reviewed implementation identities.

        Raises:
            ToolCatalogReviewRequired: If the original review cannot be reused.
            ToolCatalogBindingError: If the snapshot or selection is invalid.
        """

        if type(snapshot) is not RunToolCatalogSnapshot:
            raise ToolCatalogBindingError("invalid_snapshot")
        names = _selection(
            tool_names, tuple(_key(item.tool) for item in snapshot.bindings)
        )
        runtime = {}
        for item in snapshot.bindings:
            tool = item.tool
            spec, handler, identity = _lookup(registry, names[_key(tool)], tool.version)
            reason = _contract_reason(tool, spec)
            if reason or identity != item.implementation_id:
                raise ToolCatalogReviewRequired(reason or "implementation_changed")
            runtime[_key(tool)] = _RuntimeBinding(names[_key(tool)], handler)
        return cls._build(snapshot, runtime)

    @classmethod
    def _build(cls, snapshot, runtime) -> RunToolCatalog:
        instance = object.__new__(cls)
        object.__setattr__(instance, "snapshot", snapshot)
        object.__setattr__(instance, "_runtime", MappingProxyType(dict(runtime)))
        return instance

    def check(self, registry: ToolRegistry) -> CatalogEligibility:
        """Check all reviewed bindings against the current registry, offline.

        Args:
            registry: The current catalog after any application-owned reload.

        Returns:
            Deterministic preview and resume eligibility with controlled reasons.
        """

        reasons = []
        for item in self.snapshot.bindings:
            captured = self._runtime[_key(item.tool)]
            try:
                spec, handler, identity = _lookup(
                    registry, captured.name, item.tool.version
                )
                reason = _contract_reason(item.tool, spec)
                if reason:
                    reasons.append(reason)
                if (
                    identity != item.implementation_id
                    or handler is not captured.handler
                ):
                    reasons.append("implementation_changed")
            except ToolCatalogBindingError as exc:
                reasons.append(exc.code)
        return CatalogEligibility(self.snapshot.digest, tuple(reasons))

    def invoke(
        self,
        registry: ToolRegistry,
        tool_id: str,
        version: str,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Check eligibility and call only the captured exact-version handler.

        Args:
            registry: Current registry, checked immediately before dispatch.
            tool_id: Canonical reviewed tool ID.
            version: Exact reviewed version; no implicit upgrade or fallback.
            arguments: Protected runtime arguments, never added to evidence.

        Returns:
            The protected tool result; callers must govern its use separately.

        Raises:
            ToolCatalogReviewRequired: If any reviewed binding is unavailable
                or changed, or the requested key was never reviewed.
            ToolCatalogBindingError: If the handler fails, without echoing it.
        """

        if type(tool_id) is not str or type(version) is not str:
            raise ToolCatalogReviewRequired("unreviewed_tool")
        captured = self._runtime.get((tool_id, version))
        if captured is None:
            raise ToolCatalogReviewRequired("unreviewed_tool")
        eligibility = self.check(registry)
        if not eligibility.preview_valid:
            raise ToolCatalogReviewRequired(eligibility.reasons[0])
        try:
            return captured.handler(**arguments)
        except Exception:
            raise ToolCatalogBindingError("invocation_failed") from None

    def __repr__(self) -> str:
        return f"RunToolCatalog(tool_count={len(self.snapshot.bindings)})"


def tool_spec_schema_digest(spec: ToolSpec) -> str:
    """Digest the exact input/output schemas without exposing their contents.

    Args:
        spec: The registry's tool specification.

    Returns:
        A domain-separated SHA-256 digest of canonical schema JSON.

    Raises:
        ToolCatalogBindingError: If the schema cannot be represented canonically.
    """

    try:
        payload = _canonical_json(
            {"input_schema": spec.input_schema, "output_schema": spec.output_schema}
        )
    except Exception:
        raise ToolCatalogBindingError("invalid_schema") from None
    return _digest("openmed.agent.tool-contract.v1\0" + payload)


def tool_spec_side_effect_class(spec: ToolSpec) -> SideEffectClass:
    """Project registry annotations conservatively into inventory risk classes.

    Args:
        spec: Registered tool specification with behavioral annotations.

    Returns:
        Read-only, destructive, idempotent-write or non-idempotent-write. The
        registry does not declare pure computation, so ``NONE`` is not inferred.
    """

    if spec.read_only_hint:
        return SideEffectClass.READ_ONLY
    if spec.destructive_hint:
        return SideEffectClass.DESTRUCTIVE
    if spec.idempotent_hint:
        return SideEffectClass.IDEMPOTENT_WRITE
    return SideEffectClass.NON_IDEMPOTENT_WRITE


def _contract_reason(tool: ToolInventoryRecord, spec: ToolSpec) -> str | None:
    if spec.version != tool.version:
        return "tool_unavailable"
    if tool_spec_schema_digest(spec) != tool.schema_digest:
        return "schema_changed"
    if tool_spec_side_effect_class(spec) != tool.side_effect_class:
        return "side_effect_changed"
    return None


def _lookup(registry: ToolRegistry, name: str, version: str):
    if not isinstance(registry, ToolRegistry):
        raise ToolCatalogReviewRequired("catalog_unavailable")
    try:
        spec, handler, identity = registry.implementation_binding(name, version)
        if (
            type(spec) is not ToolSpec
            or not callable(handler)
            or not _is_digest(identity)
        ):
            raise ToolCatalogReviewRequired("catalog_unavailable")
        return spec, handler, identity
    except KeyError:
        raise ToolCatalogReviewRequired("tool_unavailable") from None
    except ToolCatalogBindingError:
        raise
    except Exception:
        raise ToolCatalogReviewRequired("catalog_unavailable") from None


def _selection(payload, keys):
    try:
        if not isinstance(payload, Mapping) or len(payload) != len(keys):
            raise ToolCatalogBindingError("invalid_selection")
        names = dict(payload)
        if set(names) != set(keys) or any(
            type(name) is not str for name in names.values()
        ):
            raise ToolCatalogBindingError("invalid_selection")
        return names
    except Exception:
        raise ToolCatalogBindingError("invalid_selection") from None


def _fixed_mapping(payload, fields):
    if not isinstance(payload, Mapping) or len(payload) != len(fields):
        raise ToolCatalogBindingError("invalid_snapshot")
    values = dict(payload)
    if set(values) != fields:
        raise ToolCatalogBindingError("invalid_snapshot")
    return values


def _key(tool: ToolInventoryRecord) -> tuple[str, str]:
    return tool.tool_id, tool.version


def _is_digest(value: Any) -> bool:
    return type(value) is str and _DIGEST_RE.fullmatch(value) is not None


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "TOOL_CATALOG_BINDING_SCHEMA_VERSION",
    "CatalogEligibility",
    "RunToolCatalog",
    "RunToolCatalogSnapshot",
    "ToolCatalogBindingError",
    "ToolCatalogReviewRequired",
    "ToolImplementationBinding",
    "tool_spec_schema_digest",
    "tool_spec_side_effect_class",
]
