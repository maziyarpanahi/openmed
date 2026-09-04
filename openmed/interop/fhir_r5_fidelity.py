"""Deterministic, privacy-safe diffs for FHIR R5 Bundle round trips.

The helper in this module compares already-exported JSON locally.  It does not
validate a FHIR profile or contact a terminology server.  JSON object member
ordering is naturally ignored, Bundle entries are matched by their stable
``fullUrl`` or ``id`` when possible, and callers can explicitly allow known
serialization differences by path.

Reports contain paths, JSON types, resource metadata, and one-way SHA-256
digests.  They intentionally never include source values because FHIR
identifiers, narrative, coded displays, and references may contain sensitive
data.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

FHIRBundleInput: TypeAlias = Mapping[str, Any] | str | bytes | bytearray | Path
PathSpec: TypeAlias = str | Sequence[str]

_MISSING = object()
_PATH_TOKEN_RE = re.compile(r"[^.\[\]]+|\[[^\]]*\]")


class FHIRR5FidelityError(ValueError):
    """Raised when a supplied value is not a readable FHIR Bundle object."""


@dataclass(frozen=True)
class ResourceIdentifier:
    """Privacy-safe metadata identifying a FHIR resource in a report.

    ``id_hash`` and ``full_url_hash`` are SHA-256 digests prefixed with
    ``sha256:``.  The raw FHIR ``id`` and ``fullUrl`` are never retained in the
    result.
    """

    resource_type: str | None = None
    id_hash: str | None = None
    full_url_hash: str | None = None

    @property
    def resource_id_hash(self) -> str | None:
        """Return the privacy-safe digest of the FHIR resource ``id``."""

        return self.id_hash

    @property
    def resource_id(self) -> str | None:
        """Return the resource id as a privacy-safe digest alias."""

        return self.id_hash

    @property
    def identifier(self) -> str:
        """Return a stable, PHI-safe display identifier."""

        if self.resource_type and self.id_hash:
            return f"{self.resource_type}/{self.id_hash}"
        if self.resource_type and self.full_url_hash:
            return f"{self.resource_type}@{self.full_url_hash}"
        if self.resource_type:
            return self.resource_type
        if self.full_url_hash:
            return f"resource@{self.full_url_hash}"
        return "unknown"

    def to_dict(self) -> dict[str, str | None]:
        """Return JSON-safe metadata without source values."""

        return {
            "resource_type": self.resource_type,
            "id_hash": self.id_hash,
            "full_url_hash": self.full_url_hash,
            "identifier": self.identifier,
        }


@dataclass(frozen=True)
class FHIRR5Difference:
    """One privacy-safe difference between two FHIR Bundle values."""

    path: str
    change_type: str
    before_type: str | None
    after_type: str | None
    before_hash: str | None
    after_hash: str | None
    before_resource: ResourceIdentifier = field(default_factory=ResourceIdentifier)
    after_resource: ResourceIdentifier = field(default_factory=ResourceIdentifier)
    before_resource_type: str | None = None
    after_resource_type: str | None = None

    @property
    def kind(self) -> str:
        """Return ``change_type`` as a concise compatibility alias."""

        return self.change_type

    @property
    def type_changed(self) -> bool:
        """Return whether the JSON or FHIR resource type changed."""

        return self.change_type in {"type_changed", "resource_type_changed"}

    @property
    def before_value_hash(self) -> str | None:
        """Return the digest of the before value, if present."""

        return self.before_hash

    @property
    def after_value_hash(self) -> str | None:
        """Return the digest of the after value, if present."""

        return self.after_hash

    @property
    def resource(self) -> ResourceIdentifier:
        """Return the after resource metadata, or before metadata if removed."""

        if self.after_resource != ResourceIdentifier():
            return self.after_resource
        return self.before_resource

    @property
    def resource_type(self) -> str | None:
        """Return the best available resource type for this difference."""

        return self.resource.resource_type

    @property
    def resource_id_hash(self) -> str | None:
        """Return the best available privacy-safe resource-id digest."""

        return self.resource.id_hash

    @property
    def resource_identifier(self) -> str:
        """Return the best available privacy-safe resource identifier."""

        return self.resource.identifier

    @property
    def json_path(self) -> str:
        """Return the same path using a JSONPath-style ``$`` root."""

        return f"$.{self.path}" if self.path else "$"

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic report record without source values."""

        resource = self.resource
        return {
            "path": self.path,
            "json_path": self.json_path,
            "change_type": self.change_type,
            "kind": self.kind,
            "before_type": self.before_type,
            "after_type": self.after_type,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "before_value_hash": self.before_value_hash,
            "after_value_hash": self.after_value_hash,
            "resource_type": resource.resource_type,
            "resource_id_hash": resource.id_hash,
            "resource_id": resource.resource_id,
            "resource_identifier": resource.identifier,
            "before_resource_type": self.before_resource_type
            or self.before_resource.resource_type,
            "after_resource_type": self.after_resource_type
            or self.after_resource.resource_type,
            "before_resource": self.before_resource.to_dict(),
            "after_resource": self.after_resource.to_dict(),
        }


@dataclass(frozen=True)
class FHIRR5FidelityDiff:
    """Deterministic, PHI-safe comparison result for two FHIR R5 Bundles."""

    changes: tuple[FHIRR5Difference, ...] = ()
    ignored_paths: tuple[str, ...] = ()
    unordered_paths: tuple[str, ...] = ()
    before_digest: str = ""
    after_digest: str = ""

    @property
    def equivalent(self) -> bool:
        """Return whether no non-declared difference was found."""

        return not self.changes

    @property
    def is_faithful(self) -> bool:
        """Return whether the round trip preserved all compared fields."""

        return self.equivalent

    @property
    def ok(self) -> bool:
        """Return whether the comparison passed."""

        return self.equivalent

    @property
    def has_changes(self) -> bool:
        """Return whether any non-declared difference was found."""

        return bool(self.changes)

    @property
    def differences(self) -> tuple[FHIRR5Difference, ...]:
        """Return ``changes`` as a report-oriented compatibility alias."""

        return self.changes

    @property
    def has_differences(self) -> bool:
        """Return whether any non-declared difference was found."""

        return self.has_changes

    @property
    def changed_paths(self) -> tuple[str, ...]:
        """Return changed paths in deterministic report order."""

        return tuple(change.path for change in self.changes)

    @property
    def type_changes(self) -> tuple[FHIRR5Difference, ...]:
        """Return JSON and FHIR resource type changes only."""

        return tuple(change for change in self.changes if change.type_changed)

    @property
    def summary(self) -> dict[str, int]:
        """Return stable counts suitable for logs and dashboards."""

        counts = {
            "changes": len(self.changes),
            "added": 0,
            "removed": 0,
            "changed": 0,
            "type_changed": 0,
            "resource_type_changed": 0,
            "ignored_paths": len(self.ignored_paths),
            "unordered_paths": len(self.unordered_paths),
        }
        for change in self.changes:
            if change.change_type in counts:
                counts[change.change_type] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, privacy-safe diff report."""

        return {
            "equivalent": self.equivalent,
            "summary": self.summary,
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "ignored_paths": list(self.ignored_paths),
            "unordered_paths": list(self.unordered_paths),
            "changes": [change.to_dict() for change in self.changes],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Return deterministic JSON suitable for a local audit artifact."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        """Return a compact Markdown report containing no source values."""

        lines = [
            "## FHIR R5 Round-Trip Fidelity",
            "",
            f"Equivalent: {'yes' if self.equivalent else 'no'}",
            f"Changes: {len(self.changes)}",
            "",
        ]
        if not self.changes:
            lines.append("No non-declared differences.")
            return "\n".join(lines)

        lines.extend(
            [
                "| Path | Resource | Change | Before type | After type |",
                "|---|---|---|---|---|",
            ]
        )
        for change in self.changes:
            lines.append(
                "| "
                f"{_markdown_cell(change.path)} | "
                f"{_markdown_cell(change.resource_identifier)} | "
                f"{_markdown_cell(change.change_type)} | "
                f"{_markdown_cell(change.before_type)} | "
                f"{_markdown_cell(change.after_type)} |"
            )
        return "\n".join(lines)


def diff_fhir_r5_bundles(
    before: FHIRBundleInput,
    after: FHIRBundleInput,
    *,
    ignored_paths: Iterable[PathSpec] = (),
    ignore_paths: Iterable[PathSpec] | None = None,
    allowed_paths: Iterable[PathSpec] = (),
    allowed_differences: Iterable[PathSpec] = (),
    serialization_differences: Iterable[PathSpec] = (),
    allowed_serialization_differences: Iterable[PathSpec] = (),
    unordered_paths: Iterable[PathSpec] = (),
) -> FHIRR5FidelityDiff:
    """Compare two local FHIR R5 Bundle JSON values.

    Object member order and JSON whitespace are ignored automatically.  Bundle
    entries are matched by ``fullUrl`` first, then ``id``, and finally by
    remaining position.  All other differences are compared recursively.

    Args:
        before: Bundle mapping, JSON text, or path to a local JSON file.
        after: Bundle mapping, JSON text, or path to a local JSON file.
        ignored_paths: Exact or wildcard paths whose complete subtrees are
            intentionally excluded from the comparison.
        ignore_paths: Alias for ``ignored_paths``.
        allowed_paths: Additional alias for declared ignored paths.
        allowed_differences: Alias for declared serialization-only paths.
        serialization_differences: Paths for declared serialization-only
            differences.
        allowed_serialization_differences: Explicit alias for declared
            serialization-only differences.
        unordered_paths: Array paths where element order is declared
            insignificant.  Use ``[*]`` for a single array-index wildcard in
            an ignored path, for example ``entry[*].resource.meta.tag``.

    Returns:
        A deterministic diff whose records contain paths, type metadata, and
        digests but never before/after source values.

    Raises:
        FHIRR5FidelityError: If either input is unreadable or is not a Bundle.
    """

    before_bundle = _load_bundle(before)
    after_bundle = _load_bundle(after)
    ignored = _normalize_patterns(
        _combine_path_specs(
            ignored_paths,
            ignore_paths,
            allowed_paths,
            allowed_differences,
            serialization_differences,
            allowed_serialization_differences,
        )
    )
    unordered = _normalize_patterns(_combine_path_specs(unordered_paths))

    before_resource = _resource_identifier(before_bundle)
    after_resource = _resource_identifier(after_bundle)
    changes: list[FHIRR5Difference] = []

    for key in sorted(set(before_bundle) | set(after_bundle)):
        if key == "entry":
            continue
        _diff_value(
            before_bundle.get(key, _MISSING),
            after_bundle.get(key, _MISSING),
            (key,),
            before_resource,
            after_resource,
            ignored,
            unordered,
            changes,
        )

    _diff_bundle_entries(
        _bundle_entries(before_bundle),
        _bundle_entries(after_bundle),
        ignored,
        unordered,
        changes,
    )

    changes.sort(key=_difference_sort_key)
    return FHIRR5FidelityDiff(
        changes=tuple(changes),
        ignored_paths=tuple(_format_path(pattern) for pattern in ignored),
        unordered_paths=tuple(_format_path(pattern) for pattern in unordered),
        before_digest=_digest(before_bundle),
        after_digest=_digest(after_bundle),
    )


def _load_bundle(value: FHIRBundleInput) -> dict[str, Any]:
    payload: Any
    if isinstance(value, Mapping):
        payload = value
    elif isinstance(value, (bytes, bytearray)):
        payload = _parse_json_text(bytes(value))
    elif isinstance(value, Path):
        payload = _read_json_path(value)
    elif isinstance(value, str):
        stripped = value.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            payload = _parse_json_text(value)
        else:
            payload = _read_json_path(Path(value))
    else:
        payload = _call_to_dict(value)

    try:
        materialized = _materialize_json(payload)
    except (TypeError, ValueError):
        raise FHIRR5FidelityError(
            "FHIR R5 bundle contains an unsupported JSON value"
        ) from None
    if not isinstance(materialized, dict):
        raise FHIRR5FidelityError("FHIR R5 bundle must be a JSON object")
    if materialized.get("resourceType") != "Bundle":
        raise FHIRR5FidelityError("FHIR R5 bundle resourceType must be Bundle")
    entries = materialized.get("entry", [])
    if not isinstance(entries, list):
        raise FHIRR5FidelityError("FHIR R5 bundle entry must be an array")
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("resource"), dict):
            raise FHIRR5FidelityError(
                "FHIR R5 bundle entries must contain resource objects"
            )
    return materialized


def _parse_json_text(value: str | bytes) -> Any:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        raise FHIRR5FidelityError("FHIR R5 bundle JSON is invalid") from None


def _read_json_path(path: Path) -> Any:
    try:
        return _parse_json_text(path.read_bytes())
    except (OSError, FHIRR5FidelityError):
        raise FHIRR5FidelityError("FHIR R5 bundle JSON could not be read") from None


def _call_to_dict(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if not callable(to_dict):
        raise FHIRR5FidelityError("FHIR R5 bundle must be JSON text or a mapping")
    try:
        return to_dict()
    except Exception:
        raise FHIRR5FidelityError("FHIR R5 bundle could not be converted") from None


def _materialize_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            result[key] = _materialize_json(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_materialize_json(item) for item in value]
    raise TypeError("unsupported JSON value")


@dataclass(frozen=True)
class _BundleEntry:
    index: int
    value: Mapping[str, Any]
    resource: Mapping[str, Any]


def _bundle_entries(bundle: Mapping[str, Any]) -> tuple[_BundleEntry, ...]:
    return tuple(
        _BundleEntry(
            index=index,
            value=entry,
            resource=entry["resource"],
        )
        for index, entry in enumerate(bundle.get("entry", []))
    )


def _diff_bundle_entries(
    before_entries: Sequence[_BundleEntry],
    after_entries: Sequence[_BundleEntry],
    ignored: Sequence[tuple[str, ...]],
    unordered: Sequence[tuple[str, ...]],
    changes: list[FHIRR5Difference],
) -> None:
    pairs = _pair_entries(before_entries, after_entries)
    for ordinal, (before_entry, after_entry) in enumerate(pairs):
        entry_path = ("entry", f"[{ordinal}]")
        if before_entry is None:
            after_resource = _resource_identifier(
                after_entry.resource,
                full_url=after_entry.value.get("fullUrl"),
            )
            _diff_value(
                _MISSING,
                after_entry.resource,
                (*entry_path, "resource"),
                ResourceIdentifier(),
                after_resource,
                ignored,
                unordered,
                changes,
            )
            continue
        if after_entry is None:
            before_resource = _resource_identifier(
                before_entry.resource,
                full_url=before_entry.value.get("fullUrl"),
            )
            _diff_value(
                before_entry.resource,
                _MISSING,
                (*entry_path, "resource"),
                before_resource,
                ResourceIdentifier(),
                ignored,
                unordered,
                changes,
            )
            continue

        before_resource = _resource_identifier(
            before_entry.resource,
            full_url=before_entry.value.get("fullUrl"),
        )
        after_resource = _resource_identifier(
            after_entry.resource,
            full_url=after_entry.value.get("fullUrl"),
        )
        for key in sorted(set(before_entry.value) | set(after_entry.value)):
            if key == "resource":
                continue
            _diff_value(
                before_entry.value.get(key, _MISSING),
                after_entry.value.get(key, _MISSING),
                (*entry_path, key),
                before_resource,
                after_resource,
                ignored,
                unordered,
                changes,
            )
        _diff_value(
            before_entry.resource,
            after_entry.resource,
            (*entry_path, "resource"),
            before_resource,
            after_resource,
            ignored,
            unordered,
            changes,
        )


def _pair_entries(
    before_entries: Sequence[_BundleEntry],
    after_entries: Sequence[_BundleEntry],
) -> list[tuple[_BundleEntry | None, _BundleEntry | None]]:
    used_before: set[int] = set()
    used_after: set[int] = set()
    pairs: list[tuple[_BundleEntry | None, _BundleEntry | None]] = []

    for field_name in ("fullUrl", "id"):
        before_by_key = _entries_by_key(before_entries, used_before, field_name)
        after_by_key = _entries_by_key(after_entries, used_after, field_name)
        for key in sorted(
            set(before_by_key) & set(after_by_key),
            key=lambda item: _digest(item),
        ):
            for before_entry, after_entry in zip(
                before_by_key[key], after_by_key[key], strict=False
            ):
                used_before.add(before_entry.index)
                used_after.add(after_entry.index)
                pairs.append((before_entry, after_entry))

    remaining_before = [
        entry for entry in before_entries if entry.index not in used_before
    ]
    remaining_after = [
        entry for entry in after_entries if entry.index not in used_after
    ]
    for before_entry, after_entry in zip(
        remaining_before, remaining_after, strict=False
    ):
        used_before.add(before_entry.index)
        used_after.add(after_entry.index)
        pairs.append((before_entry, after_entry))
    pairs.extend(
        (entry, None) for entry in before_entries if entry.index not in used_before
    )
    pairs.extend(
        (None, entry) for entry in after_entries if entry.index not in used_after
    )

    return sorted(pairs, key=_entry_pair_sort_key)


def _entries_by_key(
    entries: Sequence[_BundleEntry],
    used: set[int],
    field_name: str,
) -> dict[str, list[_BundleEntry]]:
    result: dict[str, list[_BundleEntry]] = {}
    for entry in entries:
        if entry.index in used:
            continue
        key = _entry_key(entry, field_name)
        if key is not None:
            result.setdefault(key, []).append(entry)
    return result


def _entry_key(entry: _BundleEntry, field_name: str) -> str | None:
    if field_name == "fullUrl":
        value = entry.value.get("fullUrl")
    else:
        value = entry.resource.get("id")
    return value if isinstance(value, str) and value else None


def _entry_pair_sort_key(
    pair: tuple[_BundleEntry | None, _BundleEntry | None],
) -> tuple[Any, ...]:
    before_entry, after_entry = pair
    entry = after_entry or before_entry
    assert entry is not None
    full_url = entry.value.get("fullUrl")
    resource_id = entry.resource.get("id")
    resource_type = entry.resource.get("resourceType")
    stable_value = full_url if isinstance(full_url, str) else resource_id
    if stable_value is None:
        stable_value = _digest(entry.resource)
    return (
        0 if isinstance(full_url, str) else 1 if isinstance(resource_id, str) else 2,
        _digest(stable_value),
        str(resource_type) if resource_type is not None else "",
        entry.index,
    )


def _resource_identifier(
    resource: Mapping[str, Any],
    *,
    full_url: Any = None,
) -> ResourceIdentifier:
    resource_type = resource.get("resourceType")
    return ResourceIdentifier(
        resource_type=resource_type if isinstance(resource_type, str) else None,
        id_hash=_digest(resource["id"]) if "id" in resource else None,
        full_url_hash=_digest(full_url) if full_url is not None else None,
    )


def _diff_value(
    before: Any,
    after: Any,
    path: tuple[str, ...],
    before_resource: ResourceIdentifier,
    after_resource: ResourceIdentifier,
    ignored: Sequence[tuple[str, ...]],
    unordered: Sequence[tuple[str, ...]],
    changes: list[FHIRR5Difference],
) -> None:
    if _path_is_ignored(path, ignored):
        return
    if before is _MISSING or after is _MISSING:
        if _has_descendant_pattern(path, ignored) or _has_descendant_pattern(
            path, unordered
        ):
            present = after if before is _MISSING else before
            if isinstance(present, Mapping):
                for key in sorted(present):
                    _diff_value(
                        _MISSING if before is _MISSING else before.get(key, _MISSING),
                        after.get(key, _MISSING)
                        if isinstance(after, Mapping)
                        else _MISSING,
                        (*path, key),
                        before_resource,
                        after_resource,
                        ignored,
                        unordered,
                        changes,
                    )
                return
            if _is_array(present):
                for index, item in enumerate(present):
                    _diff_value(
                        _MISSING if before is _MISSING else before[index],
                        after[index] if _is_array(after) else _MISSING,
                        (*path, f"[{index}]"),
                        before_resource,
                        after_resource,
                        ignored,
                        unordered,
                        changes,
                    )
                return
        _record_difference(
            path,
            before,
            after,
            before_resource,
            after_resource,
            changes,
        )
        return

    if _path_matches(path, unordered) and _is_array(before) and _is_array(after):
        if not _sequence_multiset_equal(before, after):
            _record_difference(
                path,
                before,
                after,
                before_resource,
                after_resource,
                changes,
            )
        return

    if isinstance(before, Mapping) and isinstance(after, Mapping):
        for key in sorted(set(before) | set(after)):
            _diff_value(
                before.get(key, _MISSING),
                after.get(key, _MISSING),
                (*path, key),
                before_resource,
                after_resource,
                ignored,
                unordered,
                changes,
            )
        return

    if _is_array(before) and _is_array(after):
        for index in range(max(len(before), len(after))):
            before_item = before[index] if index < len(before) else _MISSING
            after_item = after[index] if index < len(after) else _MISSING
            child_before_resource = before_resource
            child_after_resource = after_resource
            if path and path[-1] == "contained":
                if isinstance(before_item, Mapping):
                    child_before_resource = _resource_identifier(before_item)
                if isinstance(after_item, Mapping):
                    child_after_resource = _resource_identifier(after_item)
            _diff_value(
                before_item,
                after_item,
                (*path, f"[{index}]"),
                child_before_resource,
                child_after_resource,
                ignored,
                unordered,
                changes,
            )
        return

    if _semantic_equal(before, after):
        return
    _record_difference(
        path,
        before,
        after,
        before_resource,
        after_resource,
        changes,
    )


def _record_difference(
    path: tuple[str, ...],
    before: Any,
    after: Any,
    before_resource: ResourceIdentifier,
    after_resource: ResourceIdentifier,
    changes: list[FHIRR5Difference],
) -> None:
    before_type = _json_type(before)
    after_type = _json_type(after)
    if before is _MISSING:
        change_type = "added"
    elif after is _MISSING:
        change_type = "removed"
    elif before_type != after_type:
        change_type = "type_changed"
    elif path[-1:] == ("resourceType",) and before != after:
        change_type = "resource_type_changed"
    else:
        change_type = "changed"

    changes.append(
        FHIRR5Difference(
            path=_format_path(path),
            change_type=change_type,
            before_type=before_type,
            after_type=after_type,
            before_hash=None if before is _MISSING else _digest(before),
            after_hash=None if after is _MISSING else _digest(after),
            before_resource=before_resource,
            after_resource=after_resource,
            before_resource_type=before_resource.resource_type,
            after_resource_type=after_resource.resource_type,
        )
    )


def _normalize_patterns(specs: Iterable[PathSpec]) -> tuple[tuple[str, ...], ...]:
    if isinstance(specs, (str, Path)):
        specs = (str(specs),)
    patterns = {_normalize_path(spec) for spec in specs}
    return tuple(sorted(patterns, key=_format_path))


def _combine_path_specs(
    *groups: Iterable[PathSpec] | PathSpec | None,
) -> tuple[PathSpec, ...]:
    combined: list[PathSpec] = []
    for group in groups:
        if group is None:
            continue
        if isinstance(group, (str, Path)):
            combined.append(str(group))
        else:
            combined.extend(group)
    return tuple(combined)


def _normalize_path(spec: PathSpec) -> tuple[str, ...]:
    if isinstance(spec, (str, Path)):
        text = str(spec).strip()
        if text.startswith("/"):
            tokens = tuple(
                _normalize_path_token(part.replace("~1", "/").replace("~0", "~"))
                for part in text.split("/")[1:]
                if part
            )
        else:
            tokens = tuple(
                _normalize_path_token(match.group(0))
                for match in _PATH_TOKEN_RE.finditer(text.lstrip("$."))
            )
    else:
        tokens = tuple(_normalize_path_token(str(part)) for part in spec)
    if tokens and tokens[0].lower() == "bundle":
        tokens = tokens[1:]
    if not tokens:
        raise ValueError("path specification must not be empty")
    return tokens


def _normalize_path_token(token: str) -> str:
    token = token.strip()
    if token.startswith("[") and token.endswith("]"):
        inside = token[1:-1]
        return "[*]" if inside in {"", "*"} else f"[{inside}]"
    if token.isdigit():
        return f"[{token}]"
    return token


def _format_path(path: Sequence[str]) -> str:
    result = ""
    for token in path:
        if token.startswith("["):
            result += token
        elif result:
            result += f".{token}"
        else:
            result = token
    return result


def _path_is_ignored(
    path: tuple[str, ...], patterns: Sequence[tuple[str, ...]]
) -> bool:
    return any(_pattern_covers_path(pattern, path) for pattern in patterns)


def _path_matches(path: tuple[str, ...], patterns: Sequence[tuple[str, ...]]) -> bool:
    return any(_pattern_matches_exact(pattern, path) for pattern in patterns)


def _has_descendant_pattern(
    path: tuple[str, ...], patterns: Sequence[tuple[str, ...]]
) -> bool:
    return any(_pattern_has_path_prefix(pattern, path) for pattern in patterns)


def _pattern_has_path_prefix(pattern: Sequence[str], path: Sequence[str]) -> bool:
    def match(pattern_index: int, path_index: int) -> bool:
        if path_index == len(path):
            return pattern_index < len(pattern)
        if pattern_index == len(pattern):
            return False
        token = pattern[pattern_index]
        if token == "**":
            return match(pattern_index + 1, path_index) or match(
                pattern_index, path_index + 1
            )
        if token not in {"*", "[*]"} and token != path[path_index]:
            return False
        return match(pattern_index + 1, path_index + 1)

    return match(0, 0)


def _pattern_covers_path(pattern: Sequence[str], path: Sequence[str]) -> bool:
    def match(pattern_index: int, path_index: int) -> bool:
        if pattern_index == len(pattern):
            return True
        token = pattern[pattern_index]
        if token == "**":
            return match(pattern_index + 1, path_index) or (
                path_index < len(path) and match(pattern_index, path_index + 1)
            )
        if path_index == len(path):
            return False
        if token not in {"*", "[*]"} and token != path[path_index]:
            return False
        return match(pattern_index + 1, path_index + 1)

    return match(0, 0)


def _pattern_matches_exact(pattern: Sequence[str], path: Sequence[str]) -> bool:
    if len(pattern) != len(path):
        return False
    return all(
        pattern_token in {"*", "[*]"} or pattern_token == path_token
        for pattern_token, path_token in zip(pattern, path, strict=False)
    )


def _is_array(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _sequence_multiset_equal(before: Sequence[Any], after: Sequence[Any]) -> bool:
    remaining = list(after)
    for value in before:
        for index, candidate in enumerate(remaining):
            if _semantic_equal(value, candidate):
                del remaining[index]
                break
        else:
            return False
    return not remaining


def _semantic_equal(before: Any, after: Any) -> bool:
    if _json_type(before) != _json_type(after):
        return False
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        if set(before) != set(after):
            return False
        return all(_semantic_equal(before[key], after[key]) for key in before)
    if _is_array(before) and _is_array(after):
        return len(before) == len(after) and all(
            _semantic_equal(left, right)
            for left, right in zip(before, after, strict=False)
        )
    return before == after


def _json_type(value: Any) -> str | None:
    if value is _MISSING:
        return None
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if _is_array(value):
        return "array"
    return "unsupported"


def _digest(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise FHIRR5FidelityError(
            "FHIR R5 bundle contains an unsupported value"
        ) from None
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _difference_sort_key(change: FHIRR5Difference) -> tuple[Any, ...]:
    return (
        change.path,
        change.change_type,
        change.resource_identifier,
        change.before_type or "",
        change.after_type or "",
        change.before_hash or "",
        change.after_hash or "",
    )


def _markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


compare_fhir_r5_bundles = diff_fhir_r5_bundles
compare_bundles = diff_fhir_r5_bundles
diff_bundles = diff_fhir_r5_bundles
diff_fhir_r5 = diff_fhir_r5_bundles
FhirR5Difference = FHIRR5Difference
FhirR5FidelityDiff = FHIRR5FidelityDiff
FHIRR5Diff = FHIRR5FidelityDiff
FhirR5Diff = FHIRR5FidelityDiff


__all__ = [
    "FHIRBundleInput",
    "FHIRR5Difference",
    "FHIRR5Diff",
    "FHIRR5FidelityDiff",
    "FHIRR5FidelityError",
    "FhirR5Difference",
    "FhirR5Diff",
    "FhirR5FidelityDiff",
    "PathSpec",
    "ResourceIdentifier",
    "compare_bundles",
    "compare_fhir_r5_bundles",
    "diff_bundles",
    "diff_fhir_r5",
    "diff_fhir_r5_bundles",
]
