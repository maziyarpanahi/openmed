"""Deterministic, value-free checks for nested redaction idempotence.

The checker compares two already-produced structured redaction results.  It
does not redact input, infer clinical meaning, or contact a service.  Reports
contain schema shape, aggregate counts, safe action names, and SHA-256
fingerprints for surrogates and policy metadata; source and replacement values
are never copied into a report or an exception.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, TypeAlias

IDEMPOTENCE_SCHEMA_VERSION = 1

IdempotenceInput: TypeAlias = Mapping[str, Any] | str | Path | Any
ChangeDimension: TypeAlias = Literal[
    "shape",
    "count",
    "action",
    "surrogate",
    "policy_fingerprint",
]
ChangeClassification: TypeAlias = Literal["added", "removed", "changed"]

_DIGEST_RE = re.compile(r"^(?:sha256|hmac-sha256):[0-9a-f]{64}$")
_PATH_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z_-]{0,63}$")
_SAFE_ACTIONS = frozenset(
    {
        "drop",
        "format_preserve",
        "hash",
        "keep",
        "mask",
        "null",
        "redact",
        "remove",
        "replace",
    }
)
_ACTION_ALIASES = (
    "action_counts",
    "applied_action_counts",
    "action_summary",
    "by_action",
)
_ACTION_FIELDS = ("action", "operation", "decision", "redaction_action")
_EVENT_METADATA_FIELDS = (
    "path",
    "field_path",
    "json_path",
    "location",
    *_ACTION_FIELDS,
    "surrogate_fingerprint",
    "surrogate_hash",
    "replacement_fingerprint",
    "surrogate",
    "replacement",
    "redacted_value",
    "replaced_with",
)
_COUNT_SECTIONS = ("counts", "count_summary", "totals")
_COUNT_FIELDS = frozenset(
    {
        "added",
        "array_count",
        "changed",
        "changed_count",
        "changed_value_count",
        "count",
        "detection_count",
        "document_count",
        "errors",
        "hashed",
        "input_count",
        "kept",
        "matched_rule_count",
        "null_preserved_count",
        "nullified_value_count",
        "output_count",
        "processed",
        "redacted",
        "redacted_count",
        "redacted_value_count",
        "redaction_count",
        "removed",
        "removed_field_count",
        "replaced",
        "resource_identifier_count",
        "resource_identifiers_preserved",
        "rule_count",
        "span_count",
        "total",
        "total_count",
        "total_rows",
        "total_spans",
        "warnings",
    }
)
_EVENT_CONTAINERS = (
    "redactions",
    "redaction_events",
    "events",
    "actions",
    "operations",
    "changes",
    "spans",
    "applied_spans",
)
_SURROGATE_CONTAINERS = (
    "surrogates",
    "surrogate_by_path",
    "replacements",
    "replacement_by_path",
)
_METADATA_CONTAINERS = (
    "report",
    "redaction_report",
    "audit_report",
    "metadata",
    "summary",
    "redaction_summary",
    "result_summary",
)
_WRAPPER_KEYS = frozenset(
    {
        "audit_report",
        "counts",
        "data",
        "events",
        "metadata",
        "output",
        "policy",
        "policy_fingerprint",
        "redacted",
        "redactions",
        "redaction_report",
        "report",
        "resource",
        "surrogates",
    }
)
_DIRECT_COUNT_KEYS = _COUNT_FIELDS | {
    "redacted_values",
    "removed_fields",
}
_MISSING = object()


@dataclass(frozen=True)
class ShapeNode:
    """A value-free description of one node in a structured result."""

    path: str
    kind: str
    keys: tuple[str, ...] = ()
    length: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the node without exposing its scalar value."""

        result: dict[str, Any] = {"path": self.path, "kind": self.kind}
        if self.keys:
            result["keys"] = list(self.keys)
        if self.length is not None:
            result["length"] = self.length
        return result

    def signature(self) -> tuple[Any, ...]:
        """Return the comparable shape signature for this node."""

        return (self.kind, self.keys, self.length)


@dataclass(frozen=True)
class RedactionEvent:
    """A raw-value-free event summary at one structured path."""

    path: str
    action: str | None = None
    surrogate_fingerprint: str | None = None
    policy_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return safe event metadata suitable for an audit artifact."""

        return {
            "path": self.path,
            "action": self.action,
            "surrogate_fingerprint": self.surrogate_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
        }


@dataclass(frozen=True)
class RedactionPassSummary:
    """Safe evidence extracted from one structured redaction pass."""

    shape: tuple[ShapeNode, ...]
    shape_fingerprint: str
    action_counts: tuple[tuple[str, int], ...]
    counts: tuple[tuple[str, int], ...]
    events: tuple[RedactionEvent, ...]
    policy_fingerprint: str | None

    @property
    def shape_node_count(self) -> int:
        """Return the number of nodes represented by the shape snapshot."""

        return len(self.shape)

    @property
    def action_count_map(self) -> dict[str, int]:
        """Return action counts in deterministic insertion order."""

        return dict(self.action_counts)

    @property
    def count_map(self) -> dict[str, int]:
        """Return aggregate counts in deterministic insertion order."""

        return dict(self.counts)

    def to_dict(self) -> dict[str, Any]:
        """Return safe JSON-compatible evidence for this pass."""

        return {
            "shape_fingerprint": self.shape_fingerprint,
            "shape_node_count": self.shape_node_count,
            "shape": [node.to_dict() for node in self.shape],
            "action_counts": dict(self.action_counts),
            "counts": dict(self.counts),
            "events": [event.to_dict() for event in self.events],
            "policy_fingerprint": self.policy_fingerprint,
        }


@dataclass(frozen=True)
class IdempotenceDifference:
    """One safe classification of a difference between two passes."""

    dimension: ChangeDimension
    path: str
    before: Any
    after: Any
    classification: ChangeClassification

    def to_dict(self) -> dict[str, Any]:
        """Return the difference without source or replacement values."""

        return {
            "dimension": self.dimension,
            "path": self.path,
            "before": self.before,
            "after": self.after,
            "classification": self.classification,
        }


@dataclass(frozen=True)
class IdempotenceReport:
    """Deterministic, raw-value-free comparison of two redaction passes."""

    first_pass: RedactionPassSummary
    second_pass: RedactionPassSummary
    differences: tuple[IdempotenceDifference, ...]

    @property
    def is_idempotent(self) -> bool:
        """Return whether every checked dimension is unchanged."""

        return not self.differences

    @property
    def passed(self) -> bool:
        """Alias for :attr:`is_idempotent` for gate-style callers."""

        return self.is_idempotent

    @property
    def shape_match(self) -> bool:
        """Return whether the nested output shapes are equal."""

        return not self._has_dimension("shape")

    @property
    def counts_match(self) -> bool:
        """Return whether aggregate counts are equal."""

        return not self._has_dimension("count")

    @property
    def actions_match(self) -> bool:
        """Return whether actions and action counts are equal."""

        return not self._has_dimension("action")

    @property
    def surrogates_match(self) -> bool:
        """Return whether all reported surrogate fingerprints are equal."""

        return not self._has_dimension("surrogate")

    @property
    def policy_fingerprint_match(self) -> bool:
        """Return whether global and per-event policy fingerprints match."""

        return not self._has_dimension("policy_fingerprint")

    @property
    def non_idempotent_paths(self) -> tuple[str, ...]:
        """Return sorted paths at which non-idempotence was classified."""

        return tuple(sorted({difference.path for difference in self.differences}))

    @property
    def summary(self) -> dict[str, Any]:
        """Return aggregate comparison status without resource values."""

        dimensions = {
            "shape": int(self.shape_match is False),
            "count": int(self.counts_match is False),
            "action": int(self.actions_match is False),
            "surrogate": int(self.surrogates_match is False),
            "policy_fingerprint": int(self.policy_fingerprint_match is False),
        }
        return {
            "idempotent": self.is_idempotent,
            "shape_match": self.shape_match,
            "counts_match": self.counts_match,
            "actions_match": self.actions_match,
            "surrogates_match": self.surrogates_match,
            "policy_fingerprint_match": self.policy_fingerprint_match,
            "non_idempotent_paths": list(self.non_idempotent_paths),
            "changes_by_dimension": dimensions,
            "total_changes": len(self.differences),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible report."""

        return {
            "schema_version": IDEMPOTENCE_SCHEMA_VERSION,
            "summary": self.summary,
            "first_pass": self.first_pass.to_dict(),
            "second_pass": self.second_pass.to_dict(),
            "differences": [difference.to_dict() for difference in self.differences],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the report with stable JSON settings."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        """Render a compact, value-free review summary."""

        lines = [
            "## Structured redaction idempotence",
            "",
            f"Idempotent: **{'yes' if self.is_idempotent else 'no'}**",
            "",
            "| Dimension | Changes |",
            "|---|---:|",
        ]
        for dimension, count in self.summary["changes_by_dimension"].items():
            lines.append(f"| {_markdown_cell(dimension)} | {count} |")
        lines.extend(["", "### Non-idempotent paths"])
        if not self.non_idempotent_paths:
            lines.append("None.")
            return "\n".join(lines)
        lines.extend(
            "- `" + _markdown_cell(path) + "`" for path in self.non_idempotent_paths
        )
        return "\n".join(lines)

    def _has_dimension(self, dimension: ChangeDimension) -> bool:
        return any(item.dimension == dimension for item in self.differences)


def check_idempotence(
    first_pass: IdempotenceInput,
    second_pass: IdempotenceInput,
) -> IdempotenceReport:
    """Compare two nested redaction results without retaining raw values.

    Each input may be a nested JSON-compatible resource, a local JSON path, a
    mapping with a ``resource``/``data``/``output`` value and report metadata,
    or an object exposing the equivalent ``resource`` and ``report``
    attributes.  Reports may expose ``redactions``/``events`` with ``path``,
    ``action``, and ``surrogate`` fields, aggregate ``counts``, and policy
    metadata.  A bare resource still receives a deterministic shape check.

    The function performs no network access and never includes scalar resource
    values in the returned report.
    """

    first = _snapshot(first_pass)
    second = _snapshot(second_pass)
    return IdempotenceReport(
        first_pass=first,
        second_pass=second,
        differences=_compare_snapshots(first, second),
    )


def check_redaction_idempotence(
    first_pass: IdempotenceInput,
    second_pass: IdempotenceInput,
) -> IdempotenceReport:
    """Alias for :func:`check_idempotence` with an explicit redaction name."""

    return check_idempotence(first_pass, second_pass)


def compare_structured_redaction(
    first_pass: IdempotenceInput,
    second_pass: IdempotenceInput,
) -> IdempotenceReport:
    """Compatibility alias for :func:`check_idempotence`."""

    return check_idempotence(first_pass, second_pass)


def _snapshot(value: IdempotenceInput) -> RedactionPassSummary:
    resource, metadata = _coerce_pass(value)
    shape = _shape(resource)
    sources = _metadata_sources(metadata)
    policy_fingerprint = _extract_policy_fingerprint(sources)
    events = _extract_events(sources, policy_fingerprint)
    action_counts = _extract_action_counts(sources)
    if not action_counts:
        action_counts = _action_counts_from_events(events)
    counts = _extract_counts(sources)
    if events:
        counts.setdefault("redaction_events", len(events))
    return RedactionPassSummary(
        shape=shape,
        shape_fingerprint=_shape_fingerprint(shape),
        action_counts=tuple(sorted(action_counts.items())),
        counts=tuple(sorted(counts.items())),
        events=events,
        policy_fingerprint=policy_fingerprint,
    )


def _coerce_pass(value: IdempotenceInput) -> tuple[Any, Any]:
    if isinstance(value, Path):
        return _read_json_path(value)
    if isinstance(value, str):
        try:
            path = Path(value)
        except (OSError, ValueError):
            raise TypeError(
                "redaction pass must be a mapping, result object, or local JSON path"
            ) from None
        return _read_json_path(path)

    if isinstance(value, (list, tuple)):
        _validate_tree(value)
        return value, {}

    if isinstance(value, Mapping):
        return _coerce_mapping(value)

    resource = _attribute(value, ("resource", "data", "redacted", "output"))
    if resource is not _MISSING:
        metadata = _attribute(value, ("report", "audit_report", "metadata"))
        if metadata is _MISSING:
            metadata = _call_to_dict(value)
        _validate_tree(resource)
        return resource, metadata if metadata is not _MISSING else {}

    payload = _call_to_dict(value)
    if isinstance(payload, Mapping):
        return _coerce_mapping(payload)
    raise TypeError(
        "redaction pass must be a mapping, result object, or local JSON path"
    )


def _coerce_mapping(value: Mapping[Any, Any]) -> tuple[Any, Any]:
    payload = dict(value)
    metadata = payload
    resource_key = next(
        (
            key
            for key in ("resource", "data", "output", "redacted", "result")
            if key in payload
            and (len(payload) > 1 or key != "resource" or "resourceType" not in payload)
        ),
        None,
    )
    if resource_key is not None:
        resource = payload[resource_key]
    elif any(key in payload for key in _WRAPPER_KEYS) and not _looks_like_resource(
        payload
    ):
        resource = {
            key: item for key, item in payload.items() if key not in _WRAPPER_KEYS
        }
        metadata = payload
    else:
        resource = payload
        metadata = {}
    _validate_tree(resource)
    return resource, metadata


def _read_json_path(path: Path) -> tuple[Any, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("could not read structured redaction JSON") from None
    if not isinstance(payload, (Mapping, list, tuple)):
        raise ValueError("structured redaction JSON must contain an object or array")
    return _coerce_pass(payload)


def _looks_like_resource(value: Mapping[Any, Any]) -> bool:
    return any(key in value for key in ("resourceType", "id", "meta", "entry"))


def _attribute(value: Any, names: Sequence[str]) -> Any:
    for name in names:
        try:
            candidate = getattr(value, name)
        except Exception:
            continue
        if candidate is not None and not callable(candidate):
            return candidate
    return _MISSING


def _call_to_dict(value: Any) -> Any:
    try:
        method = getattr(value, "to_dict", None)
        if callable(method):
            result = method()
            if isinstance(result, Mapping):
                return result
    except Exception:
        return _MISSING
    return _MISSING


def _validate_tree(value: Any, *, seen: set[int] | None = None) -> None:
    if seen is None:
        seen = set()
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in seen:
            raise ValueError("structured redaction result must be acyclic")
        seen.add(marker)
        for key, child in value.items():
            if not isinstance(key, (str, int, float, bool)):
                raise TypeError("structured redaction object keys must be scalar")
            _validate_tree(child, seen=seen)
        seen.remove(marker)
        return
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in seen:
            raise ValueError("structured redaction result must be acyclic")
        seen.add(marker)
        for child in value:
            _validate_tree(child, seen=seen)
        seen.remove(marker)
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise TypeError("structured redaction result must contain JSON-compatible values")


def _shape(value: Any) -> tuple[ShapeNode, ...]:
    nodes: list[ShapeNode] = []

    def visit(current: Any, path: tuple[str | int, ...]) -> None:
        rendered = _render_path(path)
        if isinstance(current, Mapping):
            keys = tuple(sorted(_safe_key(key) for key in current))
            nodes.append(ShapeNode(rendered, "object", keys=keys))
            for key in sorted(current, key=lambda item: _safe_key(item)):
                visit(current[key], path + (str(key),))
        elif isinstance(current, (list, tuple)):
            nodes.append(ShapeNode(rendered, "array", length=len(current)))
            for index, child in enumerate(current):
                visit(child, path + (index,))
        else:
            nodes.append(ShapeNode(rendered, _scalar_kind(current)))

    visit(value, ())
    return tuple(nodes)


def _scalar_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "string"


def _shape_fingerprint(shape: Sequence[ShapeNode]) -> str:
    payload = [node.to_dict() for node in shape]
    return _digest(payload)


def _metadata_sources(metadata: Any) -> tuple[Mapping[Any, Any], ...]:
    root = _as_mapping(metadata)
    if root is None:
        return ()
    sources: list[Mapping[Any, Any]] = []
    seen: set[int] = set()
    queue: list[Mapping[Any, Any]] = [root]
    while queue:
        current = queue.pop(0)
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        sources.append(current)
        for key in _METADATA_CONTAINERS:
            nested = _as_mapping(current.get(key, _MISSING))
            if nested is not None:
                queue.append(nested)
    return tuple(sources)


def _as_mapping(value: Any) -> Mapping[Any, Any] | None:
    if isinstance(value, Mapping):
        return value
    payload = _call_to_dict(value)
    return payload if isinstance(payload, Mapping) else None


def _extract_policy_fingerprint(sources: Sequence[Mapping[Any, Any]]) -> str | None:
    for source in sources:
        for key in ("policy_fingerprint", "policy_hash"):
            if key in source:
                fingerprint = _fingerprint(source[key])
                if fingerprint is not None:
                    return fingerprint
    for source in sources:
        for key in ("policy", "policy_name", "policy_profile"):
            if key in source and source[key] is not None:
                return _fingerprint(source[key])
    return None


def _extract_counts(sources: Sequence[Mapping[Any, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in sources:
        for key in _COUNT_SECTIONS:
            value = source.get(key)
            if isinstance(value, Mapping):
                _collect_count_leaves(value, counts)
        for raw_key, value in source.items():
            key = str(raw_key).strip().lower()
            if key in _DIRECT_COUNT_KEYS and not isinstance(value, Mapping):
                counts[_safe_count_key(key)] = _as_count(value)
    return counts


def _collect_count_leaves(
    value: Mapping[Any, Any],
    counts: dict[str, int],
    prefix: str = "",
) -> None:
    for raw_key, child in value.items():
        key = str(raw_key).strip().lower()
        if key in _ACTION_ALIASES:
            continue
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, Mapping):
            _collect_count_leaves(child, counts, path)
        elif _is_number(child):
            counts[_safe_count_key(path)] = _as_count(child)


def _extract_action_counts(sources: Sequence[Mapping[Any, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in sources:
        for key in _ACTION_ALIASES:
            value = source.get(key)
            if isinstance(value, Mapping):
                parsed = _numeric_action_mapping(value)
                if parsed:
                    counts.update(parsed)
        for section_key in _COUNT_SECTIONS:
            section = source.get(section_key)
            if not isinstance(section, Mapping):
                continue
            for key in _ACTION_ALIASES:
                value = section.get(key)
                if isinstance(value, Mapping):
                    parsed = _numeric_action_mapping(value)
                    if parsed:
                        counts.update(parsed)
        actions = source.get("actions")
        if isinstance(actions, Mapping):
            parsed = _numeric_action_mapping(actions)
            if parsed:
                counts.update(parsed)
    return counts


def _numeric_action_mapping(value: Mapping[Any, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for raw_key, child in value.items():
        if _is_number(child):
            result[_safe_action(raw_key)] = _as_count(child)
    return result


def _extract_events(
    sources: Sequence[Mapping[Any, Any]],
    global_policy: str | None,
) -> tuple[RedactionEvent, ...]:
    events: list[RedactionEvent] = []
    for source in sources:
        if any(key in source for key in _EVENT_METADATA_FIELDS):
            direct = _event_from_mapping(source, global_policy)
            if direct is not None:
                events.append(direct)
        for key in _EVENT_CONTAINERS:
            container = source.get(key)
            if container is not None:
                _events_from_container(container, events, global_policy)
        for key in _SURROGATE_CONTAINERS:
            container = source.get(key)
            if isinstance(container, Mapping):
                _events_from_path_mapping(
                    container,
                    events,
                    global_policy,
                    surrogate=True,
                )
        applied_paths = source.get("applied_paths")
        if isinstance(applied_paths, Sequence) and not isinstance(
            applied_paths, (str, bytes)
        ):
            for path in applied_paths:
                if isinstance(path, (str, Sequence)) and not isinstance(
                    path, (bytes, bytearray)
                ):
                    events.append(
                        RedactionEvent(
                            path=_render_path_value(path),
                            policy_fingerprint=global_policy,
                        )
                    )
    return _coalesce_events(events)


def _events_from_container(
    container: Any,
    events: list[RedactionEvent],
    global_policy: str | None,
    *,
    fallback_path: str | None = None,
) -> None:
    if isinstance(container, Mapping):
        direct = _event_from_mapping(container, global_policy, fallback_path)
        if direct is not None:
            events.append(direct)
            return
        for raw_path, child in container.items():
            path = _render_path_value(raw_path)
            if isinstance(child, Mapping):
                event = _event_from_mapping(child, global_policy, path)
                if event is not None:
                    events.append(event)
            elif isinstance(child, str) and _safe_action_or_none(child) is not None:
                events.append(
                    RedactionEvent(
                        path=path,
                        action=_safe_action(child),
                        policy_fingerprint=global_policy,
                    )
                )
        return
    if isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
        for child in container:
            if isinstance(child, Mapping):
                event = _event_from_mapping(child, global_policy, fallback_path)
                if event is not None:
                    events.append(event)
            elif isinstance(child, str):
                events.append(
                    RedactionEvent(
                        path=_render_path_value(child),
                        policy_fingerprint=global_policy,
                    )
                )


def _events_from_path_mapping(
    container: Mapping[Any, Any],
    events: list[RedactionEvent],
    global_policy: str | None,
    *,
    surrogate: bool,
) -> None:
    for raw_path, child in container.items():
        path = _render_path_value(raw_path)
        fingerprint = _fingerprint(child) if surrogate and child is not None else None
        action = _safe_action(child) if not surrogate else None
        if isinstance(child, Mapping):
            event = _event_from_mapping(
                child,
                global_policy,
                path,
                force_surrogate=surrogate,
            )
        else:
            event = RedactionEvent(
                path=path,
                action=action,
                surrogate_fingerprint=fingerprint,
                policy_fingerprint=global_policy,
            )
        events.append(event)


def _event_from_mapping(
    value: Mapping[Any, Any],
    global_policy: str | None,
    fallback_path: str | None = None,
    *,
    force_surrogate: bool = False,
) -> RedactionEvent | None:
    path_value = _first_value(value, ("path", "field_path", "json_path", "location"))
    if path_value is _MISSING:
        start = value.get("start")
        end = value.get("end")
        if isinstance(start, Integral) and isinstance(end, Integral):
            path = f"@span[{int(start)}:{int(end)}]"
        else:
            path = fallback_path or "$"
    else:
        path = _render_path_value(path_value)
    action_value = _first_value(value, _ACTION_FIELDS)
    action = None if action_value is _MISSING else _safe_action(action_value)
    surrogate_value = _first_value(
        value,
        (
            "surrogate_fingerprint",
            "surrogate_hash",
            "replacement_fingerprint",
            "surrogate",
            "replacement",
            "redacted_value",
            "replaced_with",
        ),
    )
    surrogate = (
        None
        if surrogate_value is _MISSING or surrogate_value is None
        else _fingerprint(surrogate_value)
    )
    if force_surrogate and surrogate is None:
        surrogate = _fingerprint(value)
    local_policy = _extract_policy_fingerprint((value,)) or global_policy
    if (
        path_value is _MISSING
        and action_value is _MISSING
        and surrogate_value is _MISSING
        and not force_surrogate
        and local_policy is None
        and fallback_path is None
    ):
        return None
    return RedactionEvent(
        path=path,
        action=action,
        surrogate_fingerprint=surrogate,
        policy_fingerprint=local_policy,
    )


def _coalesce_events(events: Sequence[RedactionEvent]) -> tuple[RedactionEvent, ...]:
    merged: list[RedactionEvent] = []
    for event in events:
        match_index = next(
            (
                index
                for index, current in enumerate(merged)
                if current.path == event.path and _events_mergeable(current, event)
            ),
            None,
        )
        if match_index is None:
            merged.append(event)
            continue
        current = merged[match_index]
        merged[match_index] = RedactionEvent(
            path=current.path,
            action=current.action or event.action,
            surrogate_fingerprint=(
                current.surrogate_fingerprint or event.surrogate_fingerprint
            ),
            policy_fingerprint=current.policy_fingerprint or event.policy_fingerprint,
        )
    unique = {
        (
            event.path,
            event.action,
            event.surrogate_fingerprint,
            event.policy_fingerprint,
        ): event
        for event in merged
    }
    return tuple(unique[key] for key in sorted(unique))


def _events_mergeable(left: RedactionEvent, right: RedactionEvent) -> bool:
    return all(
        first is None or second is None or first == second
        for first, second in (
            (left.action, right.action),
            (left.surrogate_fingerprint, right.surrogate_fingerprint),
            (left.policy_fingerprint, right.policy_fingerprint),
        )
    )


def _action_counts_from_events(events: Sequence[RedactionEvent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        if event.action is not None:
            counts[event.action] = counts.get(event.action, 0) + 1
    return counts


def _compare_snapshots(
    first: RedactionPassSummary,
    second: RedactionPassSummary,
) -> tuple[IdempotenceDifference, ...]:
    differences: list[IdempotenceDifference] = []
    first_shape = {node.path: node for node in first.shape}
    second_shape = {node.path: node for node in second.shape}
    for path in sorted(set(first_shape) | set(second_shape)):
        before = first_shape.get(path)
        after = second_shape.get(path)
        if before is not None and after is not None:
            if before.signature() == after.signature():
                continue
            before_value: Any = before.to_dict()
            after_value: Any = after.to_dict()
        else:
            before_value = before.to_dict() if before is not None else None
            after_value = after.to_dict() if after is not None else None
        differences.append(_difference("shape", path, before_value, after_value))

    differences.extend(
        _compare_count_maps(first.count_map, second.count_map, "count", "counts")
    )
    differences.extend(
        _compare_count_maps(
            first.action_count_map,
            second.action_count_map,
            "action",
            "actions",
        )
    )
    differences.extend(_compare_events(first.events, second.events))
    if first.policy_fingerprint != second.policy_fingerprint:
        differences.append(
            _difference(
                "policy_fingerprint",
                "$",
                first.policy_fingerprint,
                second.policy_fingerprint,
            )
        )
    return tuple(
        sorted(
            differences,
            key=lambda item: (
                _dimension_order(item.dimension),
                item.path,
                item.classification,
                _canonical_json(item.before),
                _canonical_json(item.after),
            ),
        )
    )


def _compare_count_maps(
    first: Mapping[str, int],
    second: Mapping[str, int],
    dimension: Literal["count", "action"],
    prefix: str,
) -> list[IdempotenceDifference]:
    differences: list[IdempotenceDifference] = []
    for key in sorted(set(first) | set(second)):
        before = first.get(key)
        after = second.get(key)
        if before == after:
            continue
        differences.append(_difference(dimension, f"{prefix}.{key}", before, after))
    return differences


def _compare_events(
    first: Sequence[RedactionEvent],
    second: Sequence[RedactionEvent],
) -> list[IdempotenceDifference]:
    first_by_path = _group_events(first)
    second_by_path = _group_events(second)
    differences: list[IdempotenceDifference] = []
    for path in sorted(set(first_by_path) | set(second_by_path)):
        left = first_by_path.get(path, ())
        right = second_by_path.get(path, ())
        if not left or not right:
            before = [event.to_dict() for event in left] or None
            after = [event.to_dict() for event in right] or None
            differences.append(_difference("action", path, before, after))
            continue
        for index in range(max(len(left), len(right))):
            before = left[index] if index < len(left) else None
            after = right[index] if index < len(right) else None
            if before is None or after is None:
                differences.append(
                    _difference(
                        "action",
                        path,
                        before.to_dict() if before else None,
                        after.to_dict() if after else None,
                    )
                )
                continue
            if before.action != after.action:
                differences.append(
                    _difference("action", path, before.action, after.action)
                )
            if before.surrogate_fingerprint != after.surrogate_fingerprint:
                differences.append(
                    _difference(
                        "surrogate",
                        path,
                        before.surrogate_fingerprint,
                        after.surrogate_fingerprint,
                    )
                )
            if before.policy_fingerprint != after.policy_fingerprint:
                differences.append(
                    _difference(
                        "policy_fingerprint",
                        path,
                        before.policy_fingerprint,
                        after.policy_fingerprint,
                    )
                )
    return differences


def _group_events(
    events: Sequence[RedactionEvent],
) -> dict[str, tuple[RedactionEvent, ...]]:
    grouped: dict[str, list[RedactionEvent]] = {}
    for event in events:
        grouped.setdefault(event.path, []).append(event)
    return {
        path: tuple(sorted(items, key=lambda item: item.to_dict().__repr__()))
        for path, items in grouped.items()
    }


def _difference(
    dimension: ChangeDimension,
    path: str,
    before: Any,
    after: Any,
) -> IdempotenceDifference:
    if before is None:
        classification: ChangeClassification = "added"
    elif after is None:
        classification = "removed"
    else:
        classification = "changed"
    return IdempotenceDifference(dimension, path, before, after, classification)


def _dimension_order(dimension: ChangeDimension) -> int:
    return {
        "shape": 0,
        "count": 1,
        "action": 2,
        "surrogate": 3,
        "policy_fingerprint": 4,
    }[dimension]


def _first_value(value: Mapping[Any, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return _MISSING


def _safe_action(value: Any) -> str:
    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in _SAFE_ACTIONS:
            return candidate
    return "action:" + _digest(_stable_value(value)).removeprefix("sha256:")


def _safe_action_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip().lower() in _SAFE_ACTIONS:
        return value.strip().lower()
    return None


def _safe_count_key(value: Any) -> str:
    candidate = str(value).strip().lower()
    if candidate in _COUNT_FIELDS or candidate in _DIRECT_COUNT_KEYS:
        return candidate
    return "count:" + _digest(candidate).removeprefix("sha256:")


def _safe_key(value: Any) -> str:
    candidate = str(value)
    if _PATH_KEY_RE.fullmatch(candidate):
        return candidate
    return "key:" + _digest(candidate).removeprefix("sha256:")


def _render_path_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        segments: list[str | int] = []
        for segment in value:
            if isinstance(segment, bool):
                segments.append(_safe_key(segment))
            elif isinstance(segment, Integral) and int(segment) >= 0:
                segments.append(int(segment))
            else:
                segments.append(str(segment))
        return _render_path(tuple(segments))
    if not isinstance(value, str):
        return "path:" + _digest(_stable_value(value)).removeprefix("sha256:")
    candidate = value.strip()
    if not candidate or candidate == "$":
        return "$"
    if candidate.startswith("$"):
        candidate = candidate[1:]
    segments: list[str | int] = []
    position = 0
    try:
        while position < len(candidate):
            if candidate[position] in "./":
                position += 1
                if position == len(candidate):
                    raise ValueError
                continue
            if candidate[position] == "[":
                closing = candidate.find("]", position + 1)
                if closing < 0:
                    raise ValueError
                index = candidate[position + 1 : closing]
                if index != "*" and not index.isdigit():
                    raise ValueError
                segments.append(int(index) if index != "*" else "*")
                position = closing + 1
                continue
            start = position
            while position < len(candidate) and candidate[position] not in "./[":
                position += 1
            token = candidate[start:position]
            if not token:
                raise ValueError
            segments.append(token)
    except ValueError:
        return "path:" + _digest(value).removeprefix("sha256:")
    if not segments:
        return "path:" + _digest(value).removeprefix("sha256:")
    return _render_path(tuple(segments))


def _render_path(segments: Sequence[str | int]) -> str:
    if not segments:
        return "$"
    rendered = "$"
    for segment in segments:
        if isinstance(segment, int):
            rendered += f"[{segment}]"
        elif segment == "*":
            rendered += "[*]"
        else:
            rendered += "." + _safe_key(segment)
    return rendered


def _fingerprint(value: Any) -> str | None:
    if value is None:
        return None
    normalized = _normalize_digest(value)
    if normalized is not None:
        return normalized
    return _digest(_stable_value(value))


def _normalize_digest(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    return candidate if _DIGEST_RE.fullmatch(candidate) else None


def _digest(value: Any) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _stable_value(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _stable_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_stable_value(child) for child in value]
    payload = _call_to_dict(value)
    if payload is not _MISSING:
        return _stable_value(payload)
    return {"type": type(value).__name__}


def _is_number(value: Any) -> bool:
    return isinstance(value, (Integral, Real)) and not isinstance(value, bool)


def _as_count(value: Any) -> int:
    if not _is_number(value):
        raise ValueError("redaction counts must be non-negative integers")
    if isinstance(value, Integral):
        numeric = int(value)
        if numeric < 0:
            raise ValueError("redaction counts must be non-negative integers")
        return numeric
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
        raise ValueError("redaction counts must be non-negative integers")
    return int(numeric)


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


RedactionIdempotenceReport = IdempotenceReport
StructuredRedactionIdempotenceReport = IdempotenceReport

__all__ = [
    "ChangeClassification",
    "ChangeDimension",
    "IDEMPOTENCE_SCHEMA_VERSION",
    "IdempotenceDifference",
    "IdempotenceInput",
    "IdempotenceReport",
    "RedactionEvent",
    "RedactionIdempotenceReport",
    "RedactionPassSummary",
    "ShapeNode",
    "StructuredRedactionIdempotenceReport",
    "check_idempotence",
    "check_redaction_idempotence",
    "compare_structured_redaction",
]
