"""Deterministic, structure-preserving redaction for nested JSON resources.

The contract in this module is deliberately small: callers select scalar
leaves with explicit paths and choose one scalar action for each selected path.
Mappings keep their insertion order, lists keep both their order and length,
and resource identifier fields are never removed accidentally.  The transform
does not load a model, contact a service, or write a log message.

Reports and exceptions contain schema paths, counts, and digests only.  Raw
resource values are used while transforming the caller-owned in-memory object,
but are never copied into the report or an exception message.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, TypeAlias

REDACTION_CONTRACT_SCHEMA_VERSION: Final = 1
REDACTED_VALUE: Final = "[REDACTED]"

ACTION_KEEP: Final = "keep"
ACTION_REPLACE: Final = "replace"
ACTION_NULL: Final = "null"
ACTION_REMOVE: Final = "remove"
ACTION_MASK: Final = "mask"
ACTION_HASH: Final = "hash"

# The aliases make the contract convenient to use alongside the existing
# tabular redaction actions without making the serialized contract ambiguous.
ACTION_DROP: Final = ACTION_REMOVE
ACTION_REDACT: Final = ACTION_REPLACE

SUPPORTED_REDACTION_ACTIONS: Final = frozenset(
    {
        ACTION_HASH,
        ACTION_KEEP,
        ACTION_MASK,
        ACTION_NULL,
        ACTION_REMOVE,
        ACTION_REPLACE,
    }
)

_ACTION_ALIASES: Final = {
    "clear": ACTION_NULL,
    "drop": ACTION_REMOVE,
    "redact": ACTION_REPLACE,
    "set_null": ACTION_NULL,
}
_SIMPLE_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_MISSING = object()
_REMOVE = object()


class RedactionContractError(ValueError):
    """The path or action contract cannot be applied safely."""


class RedactionInputError(TypeError):
    """The resource is not an acyclic JSON-compatible value."""


@dataclass(frozen=True)
class ArrayWildcard:
    """A path segment matching every element of one array."""

    def __str__(self) -> str:
        """Render the explicit array wildcard notation."""

        return "[*]"


ARRAY_WILDCARD: Final = ArrayWildcard()
PathSegment: TypeAlias = str | int | ArrayWildcard


@dataclass(frozen=True, init=False)
class RedactionPath:
    """A normalized path made of object keys, array indexes, and ``[*]``.

    String paths use dots or slashes between object keys and brackets for array
    indexes.  For example, ``"entry[*].resource.name"`` selects the ``name``
    scalar in every Bundle entry.  A bare ``*`` is intentionally not supported:
    it does not say whether it should match an object key or an array index.
    """

    segments: tuple[PathSegment, ...]

    def __init__(self, value: PathLike) -> None:
        object.__setattr__(self, "segments", _parse_path(value))

    @classmethod
    def parse(cls, value: PathLike) -> "RedactionPath":
        """Parse *value* into a normalized path."""

        return cls(value)

    @property
    def is_root(self) -> bool:
        """Return whether the path selects the resource root."""

        return not self.segments

    def render(self) -> str:
        """Return the canonical, human-readable path notation."""

        if not self.segments:
            return "$"
        rendered = "$"
        for segment in self.segments:
            if isinstance(segment, ArrayWildcard):
                rendered += "[*]"
            elif isinstance(segment, int):
                rendered += f"[{segment}]"
            elif _SIMPLE_KEY_RE.fullmatch(segment):
                rendered += f".{segment}"
            else:
                rendered += f"[{json.dumps(segment, ensure_ascii=False)}]"
        return rendered[2:] if rendered.startswith("$.") else rendered

    def __str__(self) -> str:
        return self.render()


PathLike: TypeAlias = str | Sequence[str | int | ArrayWildcard] | RedactionPath


@dataclass(frozen=True)
class RedactionRule:
    """One scalar action applied to every concrete match of a path.

    Args:
        path: Dotted/slash-delimited path, or a sequence of typed path
            segments.  Array traversal must use ``[*]`` or an integer index.
        action: One of :data:`SUPPORTED_REDACTION_ACTIONS`.
        replacement: Scalar replacement for ``replace`` and ``mask``.  The
            default is ``"[REDACTED]"``.  It is hidden from ``repr`` and
            serialized reports so a caller cannot accidentally log it.
        preserve_null: Override the contract's null-preservation default for
            this rule.  When true, a null input remains null and is not counted
            as a redaction.
    """

    path: RedactionPath | PathLike
    action: str = ACTION_REPLACE
    replacement: Any = field(default=_MISSING, repr=False)
    preserve_null: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", RedactionPath(self.path))
        action = _normalize_action(self.action)
        object.__setattr__(self, "action", action)
        if self.preserve_null is not None and not isinstance(self.preserve_null, bool):
            raise RedactionContractError("preserve_null must be a boolean or null")

        replacement = self.replacement
        if replacement is _MISSING and action in {ACTION_MASK, ACTION_REPLACE}:
            replacement = REDACTED_VALUE
        if replacement is not _MISSING:
            _validate_scalar(replacement, allow_none=True)
            object.__setattr__(self, "replacement", replacement)

    @classmethod
    def from_mapping(cls, path: PathLike, spec: Any) -> "RedactionRule":
        """Build a rule from a compact mapping-style policy value.

        A mapping value with an ``action`` key is interpreted as rule options.
        A scalar value is shorthand for ``action="replace"`` with that value
        as the replacement.
        """

        if isinstance(spec, Mapping):
            if "action" not in spec:
                raise RedactionContractError("a rule mapping must declare action")
            options = {
                key: spec[key]
                for key in ("action", "replacement", "preserve_null")
                if key in spec
            }
            return cls(path, **options)
        if isinstance(spec, str) and _normalize_action_or_none(spec) is not None:
            return cls(path, action=spec)
        return cls(path, action=ACTION_REPLACE, replacement=spec)

    def to_dict(self) -> dict[str, Any]:
        """Serialize policy metadata without serializing a replacement value."""

        result: dict[str, Any] = {
            "path": str(self.path),
            "action": self.action,
            "replacement_provided": self.replacement is not _MISSING,
        }
        if self.preserve_null is not None:
            result["preserve_null"] = self.preserve_null
        return result


@dataclass(frozen=True)
class RedactionContract:
    """Validated rules and invariants for one nested-resource transform."""

    rules: Sequence[RedactionRule] | Mapping[PathLike, Any] = ()
    preserve_null: bool = True
    preserve_resource_identifiers: bool = True
    identifier_keys: Sequence[str] = ("resourceType", "id", "fullUrl")
    preserve_paths: Sequence[PathLike] = ()
    strict_paths: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.preserve_null, bool):
            raise RedactionContractError("preserve_null must be a boolean")
        if not isinstance(self.preserve_resource_identifiers, bool):
            raise RedactionContractError(
                "preserve_resource_identifiers must be a boolean"
            )
        if not isinstance(self.strict_paths, bool):
            raise RedactionContractError("strict_paths must be a boolean")

        if isinstance(self.rules, Mapping):
            normalized_rules = tuple(
                RedactionRule.from_mapping(path, spec)
                for path, spec in self.rules.items()
            )
        else:
            try:
                normalized_rules = tuple(self.rules)
            except TypeError:
                raise RedactionContractError(
                    "rules must be a sequence or mapping"
                ) from None
            if any(not isinstance(rule, RedactionRule) for rule in normalized_rules):
                raise RedactionContractError("rules must contain RedactionRule values")

        if isinstance(self.identifier_keys, (str, bytes)):
            raise RedactionContractError("identifier_keys must be a sequence")
        try:
            identifiers = tuple(self.identifier_keys)
        except TypeError:
            raise RedactionContractError("identifier_keys must be a sequence") from None
        if any(not isinstance(key, str) or not key for key in identifiers):
            raise RedactionContractError("identifier_keys must be non-empty strings")
        if len(set(identifiers)) != len(identifiers):
            raise RedactionContractError("identifier_keys must be unique")

        if isinstance(self.preserve_paths, (str, bytes)):
            raise RedactionContractError("preserve_paths must be a sequence")
        try:
            preserved = tuple(RedactionPath(path) for path in self.preserve_paths)
        except TypeError:
            raise RedactionContractError("preserve_paths must be a sequence") from None
        for left_index, left in enumerate(normalized_rules):
            if left.path.is_root and left.action == ACTION_REMOVE:
                raise RedactionContractError("the resource root cannot be removed")
            if (
                self.preserve_resource_identifiers
                and left.path.segments
                and isinstance(left.path.segments[-1], str)
                and left.path.segments[-1] in identifiers
                and left.action != ACTION_KEEP
            ):
                raise RedactionContractError(
                    "resource identifier fields are structural and cannot be transformed"
                )
            for right in normalized_rules[left_index + 1 :]:
                if _paths_overlap(left.path.segments, right.path.segments):
                    raise RedactionContractError("redaction rules overlap ambiguously")
            if any(
                _paths_overlap(left.path.segments, path.segments) for path in preserved
            ):
                raise RedactionContractError(
                    "a redaction rule conflicts with a preserved path"
                )

        object.__setattr__(self, "rules", normalized_rules)
        object.__setattr__(self, "identifier_keys", identifiers)
        object.__setattr__(self, "preserve_paths", preserved)

    @classmethod
    def from_mapping(
        cls,
        rules: Mapping[PathLike, Any],
        **kwargs: Any,
    ) -> "RedactionContract":
        """Create a contract from ``path -> action/replacement`` entries."""

        return cls(rules=rules, **kwargs)

    @classmethod
    def from_paths(
        cls,
        paths: Sequence[PathLike],
        *,
        action: str = ACTION_REPLACE,
        replacement: Any = _MISSING,
        **kwargs: Any,
    ) -> "RedactionContract":
        """Create one-action rules for a sequence of selected paths."""

        rules = tuple(
            RedactionRule(path, action=action, replacement=replacement)
            for path in paths
        )
        return cls(rules=rules, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the contract without replacement values."""

        return {
            "schema_version": REDACTION_CONTRACT_SCHEMA_VERSION,
            "rules": [rule.to_dict() for rule in self.rules],
            "preserve_null": self.preserve_null,
            "preserve_resource_identifiers": self.preserve_resource_identifiers,
            "identifier_keys": list(self.identifier_keys),
            "preserve_paths": [str(path) for path in self.preserve_paths],
            "strict_paths": self.strict_paths,
        }

    def apply(self, resource: Any, *, strict: bool | None = None) -> "RedactionResult":
        """Apply this contract to a local resource."""

        return redact_resource(resource, self, strict=strict)


@dataclass(frozen=True)
class RedactionReport:
    """Aggregate, raw-value-free evidence for one redaction operation."""

    schema_version: int
    rule_count: int
    matched_rule_count: int
    changed_value_count: int
    null_preserved_count: int
    nullified_value_count: int
    removed_field_count: int
    array_count: int
    array_lengths_preserved: bool
    resource_identifier_count: int
    resource_identifiers_preserved: int
    source_digest: str
    output_digest: str
    applied_paths: tuple[str, ...] = ()

    @property
    def redacted_value_count(self) -> int:
        """Return the number of selected scalar values that changed."""

        return self.changed_value_count

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible report metadata without resource values."""

        return {
            "schema_version": self.schema_version,
            "rule_count": self.rule_count,
            "matched_rule_count": self.matched_rule_count,
            "changed_value_count": self.changed_value_count,
            "redacted_value_count": self.redacted_value_count,
            "null_preserved_count": self.null_preserved_count,
            "nullified_value_count": self.nullified_value_count,
            "removed_field_count": self.removed_field_count,
            "array_count": self.array_count,
            "array_lengths_preserved": self.array_lengths_preserved,
            "resource_identifier_count": self.resource_identifier_count,
            "resource_identifiers_preserved": self.resource_identifiers_preserved,
            "source_digest": self.source_digest,
            "output_digest": self.output_digest,
            "applied_paths": list(self.applied_paths),
        }


@dataclass(frozen=True)
class RedactionResult:
    """The transformed resource plus a raw-value-free :class:`RedactionReport`."""

    resource: Any = field(repr=False)
    report: RedactionReport

    @property
    def data(self) -> Any:
        """Return the transformed resource under a generic data name."""

        return self.resource

    @property
    def redacted(self) -> Any:
        """Return the transformed resource under a redaction-oriented name."""

        return self.resource

    def to_audit_report(self) -> dict[str, Any]:
        """Return only the raw-value-free report."""

        return self.report.to_dict()


@dataclass
class _Stats:
    matched_rule_count: int = 0
    changed_value_count: int = 0
    null_preserved_count: int = 0
    nullified_value_count: int = 0
    removed_field_count: int = 0
    applied_paths: set[str] = field(default_factory=set)


def redact_resource(
    resource: Any,
    contract: RedactionContract | Mapping[PathLike, Any] | Sequence[RedactionRule],
    *,
    strict: bool | None = None,
) -> RedactionResult:
    """Apply a validated contract to a local JSON-compatible resource.

    The input is never mutated.  A missing optional path is ignored unless the
    contract has ``strict_paths=True`` or ``strict=True`` is passed.  Rules may
    select only scalar values; selecting a mapping or list fails closed without
    echoing that value.

    Args:
        resource: An acyclic mapping/list/scalar tree containing JSON values.
        contract: A :class:`RedactionContract`, a path mapping, or a sequence
            of :class:`RedactionRule` objects.
        strict: Optional override for the contract's missing-path behavior.

    Returns:
        A new resource and aggregate evidence.  The result's ``resource``
        attribute is the transformed value; ``report`` contains no raw values.
    """

    resolved = _coerce_contract(contract)
    _validate_json_value(resource, seen=set())
    strict_paths = resolved.strict_paths if strict is None else strict
    if not isinstance(strict_paths, bool):
        raise RedactionContractError("strict must be a boolean or null")

    targets: dict[tuple[str | int, ...], RedactionRule] = {}
    for rule in resolved.rules:
        before_count = len(targets)
        _collect_matches(
            resource,
            rule.path.segments,
            0,
            (),
            rule,
            targets,
        )
        if strict_paths and len(targets) == before_count:
            raise RedactionContractError("a contracted path did not match")

    stats = _Stats()
    transformed = _transform(
        resource,
        (),
        targets=targets,
        contract=resolved,
        stats=stats,
    )
    identifier_before = _identifier_digests(
        resource,
        identifier_keys=resolved.identifier_keys,
    )
    identifier_after = _identifier_digests(
        transformed,
        identifier_keys=resolved.identifier_keys,
    )
    identifier_preserved = sum(
        1
        for path, digest in identifier_before.items()
        if identifier_after.get(path) == digest
    )
    report = RedactionReport(
        schema_version=REDACTION_CONTRACT_SCHEMA_VERSION,
        rule_count=len(resolved.rules),
        matched_rule_count=stats.matched_rule_count,
        changed_value_count=stats.changed_value_count,
        null_preserved_count=stats.null_preserved_count,
        nullified_value_count=stats.nullified_value_count,
        removed_field_count=stats.removed_field_count,
        array_count=_array_count(resource),
        array_lengths_preserved=_array_lengths(resource) == _array_lengths(transformed),
        resource_identifier_count=len(identifier_before),
        resource_identifiers_preserved=identifier_preserved,
        source_digest=_digest(resource),
        output_digest=_digest(transformed),
        applied_paths=tuple(sorted(stats.applied_paths)),
    )
    return RedactionResult(resource=transformed, report=report)


def apply_redaction(
    resource: Any,
    contract: RedactionContract | Mapping[PathLike, Any] | Sequence[RedactionRule],
    *,
    strict: bool | None = None,
) -> RedactionResult:
    """Alias for :func:`redact_resource` for orchestration code."""

    return redact_resource(resource, contract, strict=strict)


def compile_redaction_contract(
    rules: Sequence[RedactionRule] | Mapping[PathLike, Any],
    **kwargs: Any,
) -> RedactionContract:
    """Validate and return a reusable redaction contract."""

    return RedactionContract(rules=rules, **kwargs)


def _coerce_contract(
    contract: RedactionContract | Mapping[PathLike, Any] | Sequence[RedactionRule],
) -> RedactionContract:
    if isinstance(contract, RedactionContract):
        return contract
    if isinstance(contract, Mapping):
        return RedactionContract.from_mapping(contract)
    if isinstance(contract, Sequence) and not isinstance(contract, (str, bytes)):
        return RedactionContract(rules=contract)
    raise RedactionContractError(
        "contract must be a RedactionContract or rule collection"
    )


def _normalize_action(action: Any) -> str:
    normalized = _normalize_action_or_none(action)
    if normalized is None:
        raise RedactionContractError("redaction action is unsupported")
    return normalized


def _normalize_action_or_none(action: Any) -> str | None:
    if not isinstance(action, str):
        return None
    normalized = action.strip().lower()
    normalized = _ACTION_ALIASES.get(normalized, normalized)
    return normalized if normalized in SUPPORTED_REDACTION_ACTIONS else None


def _parse_path(value: PathLike) -> tuple[PathSegment, ...]:
    if isinstance(value, RedactionPath):
        return value.segments
    if isinstance(value, str):
        return _parse_string_path(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        segments: list[PathSegment] = []
        for segment in value:
            if isinstance(segment, ArrayWildcard):
                segments.append(segment)
            elif isinstance(segment, bool):
                raise RedactionContractError(
                    "path indexes must be non-negative integers"
                )
            elif isinstance(segment, int):
                if segment < 0:
                    raise RedactionContractError(
                        "path indexes must be non-negative integers"
                    )
                segments.append(segment)
            elif isinstance(segment, str):
                if segment in {"", ".", "/"}:
                    raise RedactionContractError("path segments must be non-empty")
                if segment in {"[]", "[*]"}:
                    segments.append(ARRAY_WILDCARD)
                elif segment == "*":
                    raise RedactionContractError(
                        "bare wildcard paths are ambiguous; use [*] for arrays"
                    )
                else:
                    segments.append(segment)
            else:
                raise RedactionContractError("path segments must be strings or indexes")
        return tuple(segments)
    raise RedactionContractError("path must be a string or sequence of segments")


def _parse_string_path(value: str) -> tuple[PathSegment, ...]:
    text = value.strip()
    if text in {"", "$"}:
        return ()
    if text.startswith("$.") or text.startswith("$/"):
        text = text[2:]
    elif text.startswith("$"):
        raise RedactionContractError("root paths must use $ followed by a separator")

    segments: list[PathSegment] = []
    token: list[str] = []
    index = 0
    expect_segment = True

    def flush_token() -> None:
        nonlocal expect_segment
        if not token:
            if expect_segment:
                raise RedactionContractError("path contains an empty segment")
            return
        key = "".join(token)
        token.clear()
        if key == "*":
            raise RedactionContractError(
                "bare wildcard paths are ambiguous; use [*] for arrays"
            )
        segments.append(key)
        expect_segment = False

    while index < len(text):
        character = text[index]
        if character in "./":
            flush_token()
            expect_segment = True
            index += 1
            continue
        if character == "[":
            flush_token()
            closing = text.find("]", index + 1)
            if closing < 0:
                raise RedactionContractError("path contains an unclosed bracket")
            contents = text[index + 1 : closing].strip()
            if contents in {"", "*"}:
                segments.append(ARRAY_WILDCARD)
            elif contents.isdigit():
                segments.append(int(contents))
            elif len(contents) >= 2 and contents[0] == contents[-1] in {"'", '"'}:
                key = contents[1:-1]
                if not key:
                    raise RedactionContractError("path segments must be non-empty")
                segments.append(key)
            else:
                raise RedactionContractError("brackets must contain an index or [*]")
            expect_segment = False
            index = closing + 1
            if index < len(text) and text[index] not in ".[/":
                raise RedactionContractError(
                    "path requires a separator after a bracket"
                )
            continue
        token.append(character)
        expect_segment = False
        index += 1

    flush_token()
    if expect_segment:
        raise RedactionContractError("path contains an empty segment")
    return tuple(segments)


def _paths_overlap(
    left: Sequence[PathSegment],
    right: Sequence[PathSegment],
) -> bool:
    if len(left) != len(right):
        return False
    return all(_segments_overlap(a, b) for a, b in zip(left, right))


def _segments_overlap(left: PathSegment, right: PathSegment) -> bool:
    if isinstance(left, ArrayWildcard):
        return isinstance(right, (ArrayWildcard, int))
    if isinstance(right, ArrayWildcard):
        return isinstance(left, (ArrayWildcard, int))
    if isinstance(left, int) or isinstance(right, int):
        return left == right
    return left == right


def _validate_scalar(value: Any, *, allow_none: bool) -> None:
    if value is None:
        if allow_none:
            return
        raise RedactionContractError("scalar value cannot be null")
    if isinstance(value, bool) or isinstance(value, str) or isinstance(value, int):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise RedactionContractError("replacement values must be finite JSON scalars")


def _validate_json_value(value: Any, *, seen: set[int]) -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RedactionInputError("resource contains a non-finite number")
        return
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            raise RedactionInputError("resource must be an acyclic JSON value")
        seen.add(identity)
        for key, child in value.items():
            if not isinstance(key, str):
                raise RedactionInputError("resource object keys must be strings")
            _validate_json_value(child, seen=seen)
        seen.remove(identity)
        return
    if isinstance(value, list):
        identity = id(value)
        if identity in seen:
            raise RedactionInputError("resource must be an acyclic JSON value")
        seen.add(identity)
        for child in value:
            _validate_json_value(child, seen=seen)
        seen.remove(identity)
        return
    raise RedactionInputError(
        "resource must contain only JSON-compatible mappings, lists, and scalars"
    )


def _collect_matches(
    value: Any,
    segments: Sequence[PathSegment],
    position: int,
    concrete_path: tuple[str | int, ...],
    rule: RedactionRule,
    targets: dict[tuple[str | int, ...], RedactionRule],
) -> None:
    if position == len(segments):
        if concrete_path in targets:
            raise RedactionContractError("multiple rules select one scalar path")
        targets[concrete_path] = rule
        return

    segment = segments[position]
    if isinstance(value, Mapping):
        if isinstance(segment, str) and segment in value:
            _collect_matches(
                value[segment],
                segments,
                position + 1,
                concrete_path + (segment,),
                rule,
                targets,
            )
        return

    if isinstance(value, list):
        if isinstance(segment, ArrayWildcard):
            for index, child in enumerate(value):
                _collect_matches(
                    child,
                    segments,
                    position + 1,
                    concrete_path + (index,),
                    rule,
                    targets,
                )
        elif isinstance(segment, int) and segment < len(value):
            _collect_matches(
                value[segment],
                segments,
                position + 1,
                concrete_path + (segment,),
                rule,
                targets,
            )


def _transform(
    value: Any,
    concrete_path: tuple[str | int, ...],
    *,
    targets: Mapping[tuple[str | int, ...], RedactionRule],
    contract: RedactionContract,
    stats: _Stats,
) -> Any:
    rule = targets.get(concrete_path)
    if rule is not None:
        outcome, changed, null_preserved, nullified = _apply_rule(
            value,
            rule,
            preserve_null=(
                contract.preserve_null
                if rule.preserve_null is None
                else rule.preserve_null
            ),
        )
        stats.matched_rule_count += 1
        stats.applied_paths.add(_render_concrete_path(concrete_path))
        if changed:
            stats.changed_value_count += 1
        if null_preserved:
            stats.null_preserved_count += 1
        if nullified:
            stats.nullified_value_count += 1
        return outcome

    if isinstance(value, Mapping):
        transformed: dict[str, Any] = {}
        for key, child in value.items():
            child_path = concrete_path + (key,)
            child_value = _transform(
                child,
                child_path,
                targets=targets,
                contract=contract,
                stats=stats,
            )
            if child_value is _REMOVE:
                if (
                    contract.preserve_resource_identifiers
                    and key in contract.identifier_keys
                ):
                    # Resource identifiers retain their field position.  An
                    # explicit remove request clears the value rather than
                    # deleting the identifier-bearing key.
                    transformed[key] = None
                    stats.nullified_value_count += 1
                else:
                    stats.removed_field_count += 1
                continue
            transformed[key] = child_value
        return transformed

    if isinstance(value, list):
        transformed_list: list[Any] = []
        for index, child in enumerate(value):
            child_value = _transform(
                child,
                concrete_path + (index,),
                targets=targets,
                contract=contract,
                stats=stats,
            )
            transformed_list.append(None if child_value is _REMOVE else child_value)
        return transformed_list

    return value


def _apply_rule(
    value: Any,
    rule: RedactionRule,
    *,
    preserve_null: bool,
) -> tuple[Any, bool, bool, bool]:
    _validate_scalar(value, allow_none=True)
    if value is None and preserve_null:
        return value, False, True, False
    if rule.action == ACTION_KEEP:
        return value, False, False, False
    if rule.action == ACTION_NULL:
        return None, value is not None, False, value is not None
    if rule.action == ACTION_REMOVE:
        return _REMOVE, True, False, False
    if rule.action in {ACTION_MASK, ACTION_REPLACE}:
        replacement = rule.replacement
        if replacement is _MISSING:
            replacement = REDACTED_VALUE
        return (
            replacement,
            replacement != value,
            False,
            value is not None and replacement is None,
        )
    if rule.action == ACTION_HASH:
        hashed = _scalar_digest(value)
        return hashed, hashed != value, False, False
    raise RedactionContractError("redaction action is unsupported")


def _scalar_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any) -> str:
    return _scalar_digest(value)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _canonicalize(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_canonicalize(child) for child in value]
    return value


def _render_concrete_path(path: Sequence[str | int]) -> str:
    rendered = "$"
    for segment in path:
        if isinstance(segment, int):
            rendered += f"[{segment}]"
        elif _SIMPLE_KEY_RE.fullmatch(segment):
            rendered += f".{segment}"
        else:
            rendered += f"[{json.dumps(segment, ensure_ascii=False)}]"
    return rendered[2:] if rendered.startswith("$.") else rendered


def _identifier_digests(
    value: Any,
    *,
    identifier_keys: Sequence[str],
    path: tuple[str | int, ...] = (),
) -> dict[tuple[str | int, ...], str]:
    digests: dict[tuple[str | int, ...], str] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = path + (key,)
            if key in identifier_keys and _is_scalar(child):
                digests[child_path] = _scalar_digest(child)
            digests.update(
                _identifier_digests(
                    child,
                    identifier_keys=identifier_keys,
                    path=child_path,
                )
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            digests.update(
                _identifier_digests(
                    child,
                    identifier_keys=identifier_keys,
                    path=path + (index,),
                )
            )
    return digests


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _array_count(value: Any) -> int:
    if isinstance(value, list):
        return 1 + sum(_array_count(child) for child in value)
    if isinstance(value, Mapping):
        return sum(_array_count(child) for child in value.values())
    return 0


def _array_lengths(value: Any) -> tuple[int, ...]:
    lengths: list[int] = []
    if isinstance(value, list):
        lengths.append(len(value))
        for child in value:
            lengths.extend(_array_lengths(child))
    elif isinstance(value, Mapping):
        for child in value.values():
            lengths.extend(_array_lengths(child))
    return tuple(lengths)


__all__ = [
    "ACTION_DROP",
    "ACTION_HASH",
    "ACTION_KEEP",
    "ACTION_MASK",
    "ACTION_NULL",
    "ACTION_REDACT",
    "ACTION_REMOVE",
    "ACTION_REPLACE",
    "ARRAY_WILDCARD",
    "ArrayWildcard",
    "REDACTED_VALUE",
    "REDACTION_CONTRACT_SCHEMA_VERSION",
    "RedactionContract",
    "RedactionContractError",
    "RedactionInputError",
    "RedactionPath",
    "PathLike",
    "PathSegment",
    "RedactionReport",
    "RedactionResult",
    "RedactionRule",
    "SUPPORTED_REDACTION_ACTIONS",
    "apply_redaction",
    "compile_redaction_contract",
    "redact_resource",
]
