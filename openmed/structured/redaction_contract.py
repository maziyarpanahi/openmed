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
import itertools
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, TypeAlias

REDACTION_CONTRACT_SCHEMA_VERSION: Final = 1
REDACTED_VALUE: Final = "[REDACTED]"
MAX_REDACTION_RULES: Final = 256
MAX_PRESERVE_PATHS: Final = 256
MAX_IDENTIFIER_KEYS: Final = 32
MAX_PATH_SEGMENTS: Final = 64
MAX_PATH_LENGTH: Final = 4_096
MAX_KEY_LENGTH: Final = 256
MAX_CONTAINER_ITEMS: Final = 10_000
MAX_RESOURCE_DEPTH: Final = 64
MAX_RESOURCE_NODES: Final = 100_000
MAX_REDACTION_MATCHES: Final = 10_000
MAX_STRING_CHARS: Final = 1_000_000
MAX_TOTAL_STRING_CHARS: Final = 10_000_000
MAX_REPLACEMENT_STRING_CHARS: Final = 65_536
MAX_JSON_INTEGER: Final = (1 << 63) - 1
MIN_JSON_INTEGER: Final = -(1 << 63)

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
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_RULE_OPTION_KEYS: Final = frozenset({"action", "preserve_null", "replacement"})
_MISSING = object()
_REMOVE = object()


class RedactionContractError(ValueError):
    """The path or action contract cannot be applied safely."""


class RedactionInputError(TypeError):
    """The resource is not an acyclic JSON-compatible value."""


def _bounded_items(
    value: Mapping[Any, Any], *, field_name: str, max_items: int
) -> list[tuple[Any, Any]]:
    try:
        items = list(itertools.islice(value.items(), max_items + 1))
    except Exception:  # noqa: BLE001 - mappings are caller-controlled protocols.
        raise RedactionContractError(f"{field_name} could not be read") from None
    if len(items) > max_items:
        raise RedactionContractError(f"{field_name} exceeds the supported item limit")
    result: list[tuple[Any, Any]] = []
    for item in items:
        if type(item) not in {list, tuple} or len(item) != 2:
            raise RedactionContractError(f"{field_name} contains an invalid entry")
        result.append((item[0], item[1]))
    return result


def _bounded_values(value: Any, *, field_name: str, max_items: int) -> list[Any]:
    if isinstance(value, (str, bytes)):
        raise RedactionContractError(f"{field_name} must be a sequence")
    try:
        iterator = iter(value)
    except Exception:  # noqa: BLE001 - iterables are caller-controlled protocols.
        raise RedactionContractError(f"{field_name} must be a sequence") from None
    try:
        items = list(itertools.islice(iterator, max_items + 1))
    except Exception:  # noqa: BLE001 - iterables are caller-controlled protocols.
        raise RedactionContractError(f"{field_name} could not be read") from None
    if len(items) > max_items:
        raise RedactionContractError(f"{field_name} exceeds the supported item limit")
    return items


def _validate_key(value: Any, *, field_name: str = "path segment") -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > MAX_KEY_LENGTH
        or not value.isprintable()
    ):
        raise RedactionContractError(f"{field_name} must be a safe bounded string")
    return value


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
        segments = _parse_path(value)
        object.__setattr__(self, "segments", segments)
        wildcard_growth = sum(
            len(f"[{MAX_CONTAINER_ITEMS - 1}]") - len("[*]")
            for segment in segments
            if isinstance(segment, ArrayWildcard)
        )
        if len(self.render()) + wildcard_growth > MAX_PATH_LENGTH:
            raise RedactionContractError("path exceeds the supported length limit")

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
        if self.preserve_null is not None and type(self.preserve_null) is not bool:
            raise RedactionContractError("preserve_null must be a boolean or null")

        replacement = self.replacement
        if replacement is _MISSING and action in {ACTION_MASK, ACTION_REPLACE}:
            replacement = REDACTED_VALUE
        elif replacement is not _MISSING and action not in {
            ACTION_MASK,
            ACTION_REPLACE,
        }:
            raise RedactionContractError(
                "replacement is supported only for mask and replace actions"
            )
        if replacement is not _MISSING:
            _validate_scalar(
                replacement,
                allow_none=True,
                max_string_chars=MAX_REPLACEMENT_STRING_CHARS,
            )
            object.__setattr__(self, "replacement", replacement)

    @classmethod
    def from_mapping(cls, path: PathLike, spec: Any) -> "RedactionRule":
        """Build a rule from a compact mapping-style policy value.

        A mapping value with an ``action`` key is interpreted as closed rule
        options. A supported action string is a compact action shorthand.
        Replacement values must use the explicit mapping form.
        """

        if isinstance(spec, Mapping):
            items = _bounded_items(
                spec,
                field_name="rule options",
                max_items=len(_RULE_OPTION_KEYS),
            )
            options: dict[str, Any] = {}
            for key, option_value in items:
                if (
                    type(key) is not str
                    or key not in _RULE_OPTION_KEYS
                    or key in options
                ):
                    raise RedactionContractError(
                        "rule options contain unsupported fields"
                    )
                options[key] = option_value
            if "action" not in options:
                raise RedactionContractError("a rule mapping must declare action")
            return cls(path, **options)
        if type(spec) is str and _normalize_action_or_none(spec) is not None:
            return cls(path, action=spec)
        raise RedactionContractError(
            "rule values must be an action string or closed option mapping"
        )

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
        if type(self.preserve_null) is not bool:
            raise RedactionContractError("preserve_null must be a boolean")
        if type(self.preserve_resource_identifiers) is not bool:
            raise RedactionContractError(
                "preserve_resource_identifiers must be a boolean"
            )
        if type(self.strict_paths) is not bool:
            raise RedactionContractError("strict_paths must be a boolean")

        if isinstance(self.rules, Mapping):
            normalized_rules = tuple(
                RedactionRule.from_mapping(path, spec)
                for path, spec in _bounded_items(
                    self.rules,
                    field_name="rules",
                    max_items=MAX_REDACTION_RULES,
                )
            )
        else:
            normalized_rules = tuple(
                _bounded_values(
                    self.rules,
                    field_name="rules",
                    max_items=MAX_REDACTION_RULES,
                )
            )
            if any(type(rule) is not RedactionRule for rule in normalized_rules):
                raise RedactionContractError("rules must contain RedactionRule values")

        identifiers = tuple(
            _validate_key(key, field_name="identifier key")
            for key in _bounded_values(
                self.identifier_keys,
                field_name="identifier_keys",
                max_items=MAX_IDENTIFIER_KEYS,
            )
        )
        if len(set(identifiers)) != len(identifiers):
            raise RedactionContractError("identifier_keys must be unique")

        preserved = tuple(
            RedactionPath(path)
            for path in _bounded_values(
                self.preserve_paths,
                field_name="preserve_paths",
                max_items=MAX_PRESERVE_PATHS,
            )
        )
        for left_index, left in enumerate(normalized_rules):
            if left.path.is_root and left.action != ACTION_KEEP:
                raise RedactionContractError("the resource root cannot be transformed")
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
            for path in _bounded_values(
                paths,
                field_name="paths",
                max_items=MAX_REDACTION_RULES,
            )
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

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != REDACTION_CONTRACT_SCHEMA_VERSION
        ):
            raise RedactionContractError("report schema version is unsupported")
        count_limits = {
            "rule_count": MAX_REDACTION_RULES,
            "matched_rule_count": MAX_REDACTION_MATCHES,
            "changed_value_count": MAX_REDACTION_MATCHES,
            "null_preserved_count": MAX_REDACTION_MATCHES,
            "nullified_value_count": MAX_REDACTION_MATCHES,
            "removed_field_count": MAX_REDACTION_MATCHES,
            "array_count": MAX_RESOURCE_NODES,
            "resource_identifier_count": MAX_RESOURCE_NODES,
            "resource_identifiers_preserved": MAX_RESOURCE_NODES,
        }
        for name, maximum in count_limits.items():
            value = getattr(self, name)
            if type(value) is not int or not (0 <= value <= maximum):
                raise RedactionContractError(
                    "report counts are outside supported bounds"
                )
        if type(self.array_lengths_preserved) is not bool:
            raise RedactionContractError("report array status must be a boolean")
        if not self.array_lengths_preserved:
            raise RedactionContractError("report array preservation is inconsistent")
        if self.changed_value_count > self.matched_rule_count:
            raise RedactionContractError("report change counts are inconsistent")
        if self.null_preserved_count > self.matched_rule_count:
            raise RedactionContractError("report null counts are inconsistent")
        if self.nullified_value_count > self.changed_value_count:
            raise RedactionContractError("report nullification counts are inconsistent")
        if self.removed_field_count > self.changed_value_count:
            raise RedactionContractError("report removal counts are inconsistent")
        if self.resource_identifiers_preserved > self.resource_identifier_count:
            raise RedactionContractError("report identifier counts are inconsistent")
        if (
            type(self.source_digest) is not str
            or not _DIGEST_RE.fullmatch(self.source_digest)
            or type(self.output_digest) is not str
            or not _DIGEST_RE.fullmatch(self.output_digest)
        ):
            raise RedactionContractError("report digests are invalid")
        if type(self.applied_paths) is not tuple:
            raise RedactionContractError("report paths must be a tuple")
        if len(self.applied_paths) > self.matched_rule_count:
            raise RedactionContractError("report paths exceed matched values")
        if len(self.applied_paths) != self.matched_rule_count:
            raise RedactionContractError("report paths do not match selected values")
        if self.applied_paths != tuple(sorted(set(self.applied_paths))):
            raise RedactionContractError("report paths are not canonical")
        if any(not _is_canonical_concrete_path(path) for path in self.applied_paths):
            raise RedactionContractError("report paths are invalid")
        if self.changed_value_count + self.null_preserved_count > (
            self.matched_rule_count
        ):
            raise RedactionContractError(
                "report selected-value counts are inconsistent"
            )
        if self.nullified_value_count + self.removed_field_count > (
            self.changed_value_count
        ):
            raise RedactionContractError("report changed-value counts are inconsistent")

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

    def __post_init__(self) -> None:
        if type(self.report) is not RedactionReport:
            raise RedactionContractError("result report is invalid")

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
    normalized_resource = _snapshot_json_value(
        resource,
        seen=set(),
        budget=_ResourceBudget(),
        depth=0,
    )
    strict_paths = resolved.strict_paths if strict is None else strict
    if type(strict_paths) is not bool:
        raise RedactionContractError("strict must be a boolean or null")

    targets: dict[tuple[str | int, ...], RedactionRule] = {}
    for rule in resolved.rules:
        before_count = len(targets)
        _collect_matches(
            normalized_resource,
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
        normalized_resource,
        (),
        targets=targets,
        contract=resolved,
        stats=stats,
    )
    identifier_before = _identifier_digests(
        normalized_resource,
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
    if resolved.preserve_resource_identifiers and identifier_before != identifier_after:
        raise RedactionContractError("resource identifier preservation failed")
    report = RedactionReport(
        schema_version=REDACTION_CONTRACT_SCHEMA_VERSION,
        rule_count=len(resolved.rules),
        matched_rule_count=stats.matched_rule_count,
        changed_value_count=stats.changed_value_count,
        null_preserved_count=stats.null_preserved_count,
        nullified_value_count=stats.nullified_value_count,
        removed_field_count=stats.removed_field_count,
        array_count=_array_count(normalized_resource),
        array_lengths_preserved=(
            _array_lengths(normalized_resource) == _array_lengths(transformed)
        ),
        resource_identifier_count=len(identifier_before),
        resource_identifiers_preserved=identifier_preserved,
        source_digest=_digest(normalized_resource),
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
    if type(contract) is RedactionContract:
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
    if type(action) is not str or len(action) > 64:
        return None
    normalized = action.strip().lower()
    normalized = _ACTION_ALIASES.get(normalized, normalized)
    return normalized if normalized in SUPPORTED_REDACTION_ACTIONS else None


def _parse_path(value: PathLike) -> tuple[PathSegment, ...]:
    if type(value) is RedactionPath:
        return value.segments
    if type(value) is str:
        return _parse_string_path(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        segments: list[PathSegment] = []
        for segment in _bounded_values(
            value,
            field_name="path segments",
            max_items=MAX_PATH_SEGMENTS,
        ):
            if type(segment) is ArrayWildcard:
                segments.append(ARRAY_WILDCARD)
            elif type(segment) is int:
                if not (0 <= segment < MAX_CONTAINER_ITEMS):
                    raise RedactionContractError(
                        "path indexes must be bounded non-negative integers"
                    )
                segments.append(segment)
            elif type(segment) is str:
                if segment == "[*]":
                    segments.append(ARRAY_WILDCARD)
                elif segment == "*":
                    raise RedactionContractError(
                        "bare wildcard paths are ambiguous; use [*] for arrays"
                    )
                else:
                    segments.append(_validate_key(segment))
            else:
                raise RedactionContractError("path segments must be strings or indexes")
        return tuple(segments)
    raise RedactionContractError("path must be a string or sequence of segments")


def _parse_string_path(value: str) -> tuple[PathSegment, ...]:
    if len(value) > MAX_PATH_LENGTH:
        raise RedactionContractError("path exceeds the supported length limit")
    text = value.strip()
    if text in {"", "$"}:
        return ()
    if text.startswith("$.") or text.startswith("$/"):
        text = text[2:]
    elif text.startswith("$["):
        text = text[1:]
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
        segments.append(_validate_key(key))
        if len(segments) > MAX_PATH_SEGMENTS:
            raise RedactionContractError("path exceeds the supported segment limit")
        expect_segment = False

    while index < len(text):
        character = text[index]
        if character in "./":
            flush_token()
            expect_segment = True
            index += 1
            continue
        if character == "[":
            if token:
                flush_token()
            elif expect_segment and segments:
                raise RedactionContractError("path contains an empty segment")
            closing = _find_closing_bracket(text, index)
            if closing < 0:
                raise RedactionContractError("path contains an unclosed bracket")
            contents = text[index + 1 : closing].strip()
            if contents == "*":
                segments.append(ARRAY_WILDCARD)
            elif contents.isdigit():
                if len(contents) > 10:
                    raise RedactionContractError(
                        "path index exceeds the supported limit"
                    )
                array_index = int(contents)
                if array_index >= MAX_CONTAINER_ITEMS:
                    raise RedactionContractError(
                        "path index exceeds the supported limit"
                    )
                segments.append(array_index)
            elif len(contents) >= 2 and contents[0] == contents[-1] == '"':
                try:
                    key = json.loads(contents)
                except (TypeError, ValueError):
                    raise RedactionContractError("quoted path key is invalid") from None
                segments.append(_validate_key(key))
            else:
                raise RedactionContractError("brackets must contain an index or [*]")
            if len(segments) > MAX_PATH_SEGMENTS:
                raise RedactionContractError("path exceeds the supported segment limit")
            expect_segment = False
            index = closing + 1
            if index < len(text) and text[index] not in ".[/[":
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


def _find_closing_bracket(value: str, opening: int) -> int:
    quoted = False
    escaped = False
    for index in range(opening + 1, len(value)):
        character = value[index]
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character == "]":
            return index
    return -1


def _is_canonical_concrete_path(value: Any) -> bool:
    if (
        type(value) is not str
        or not value
        or len(value) > MAX_PATH_LENGTH
        or not value.isprintable()
    ):
        return False
    try:
        path = RedactionPath(value)
    except RedactionContractError:
        return False
    return (
        not any(isinstance(segment, ArrayWildcard) for segment in path.segments)
        and str(path) == value
    )


def _paths_overlap(
    left: Sequence[PathSegment],
    right: Sequence[PathSegment],
) -> bool:
    shared_length = min(len(left), len(right))
    return all(
        _segments_overlap(left[index], right[index]) for index in range(shared_length)
    )


def _segments_overlap(left: PathSegment, right: PathSegment) -> bool:
    if isinstance(left, ArrayWildcard):
        return isinstance(right, (ArrayWildcard, int))
    if isinstance(right, ArrayWildcard):
        return isinstance(left, (ArrayWildcard, int))
    if isinstance(left, int) or isinstance(right, int):
        return left == right
    return left == right


def _validate_scalar(
    value: Any,
    *,
    allow_none: bool,
    max_string_chars: int = MAX_STRING_CHARS,
) -> None:
    if value is None:
        if allow_none:
            return
        raise RedactionContractError("scalar value cannot be null")
    if type(value) is bool:
        return
    if type(value) is str:
        if len(value) > max_string_chars:
            raise RedactionContractError("scalar string exceeds the supported limit")
        return
    if type(value) is int:
        if MIN_JSON_INTEGER <= value <= MAX_JSON_INTEGER:
            return
        raise RedactionContractError("integer scalar exceeds the supported limit")
    if type(value) is float and math.isfinite(value):
        return
    raise RedactionContractError("replacement values must be finite JSON scalars")


@dataclass
class _ResourceBudget:
    nodes: int = 0
    string_chars: int = 0


def _snapshot_json_value(
    value: Any,
    *,
    seen: set[int],
    budget: _ResourceBudget,
    depth: int,
) -> Any:
    if depth > MAX_RESOURCE_DEPTH:
        raise RedactionInputError("resource exceeds the supported nesting limit")
    budget.nodes += 1
    if budget.nodes > MAX_RESOURCE_NODES:
        raise RedactionInputError("resource exceeds the supported node limit")

    if value is None or type(value) is bool:
        return value
    if type(value) is str:
        _add_string_to_budget(value, budget=budget)
        return value
    if type(value) is int:
        if not MIN_JSON_INTEGER <= value <= MAX_JSON_INTEGER:
            raise RedactionInputError("resource contains an out-of-range integer")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise RedactionInputError("resource contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            raise RedactionInputError("resource must be an acyclic JSON value")
        seen.add(identity)
        try:
            items = _bounded_resource_items(value)
            result: dict[str, Any] = {}
            for key, child in items:
                _validate_resource_key(key)
                if key in result:
                    raise RedactionInputError("resource object keys must be unique")
                _add_string_to_budget(key, budget=budget)
                result[key] = _snapshot_json_value(
                    child,
                    seen=seen,
                    budget=budget,
                    depth=depth + 1,
                )
            return result
        finally:
            seen.remove(identity)
    if type(value) is list:
        identity = id(value)
        if identity in seen:
            raise RedactionInputError("resource must be an acyclic JSON value")
        seen.add(identity)
        try:
            if len(value) > MAX_CONTAINER_ITEMS:
                raise RedactionInputError(
                    "resource array exceeds the supported item limit"
                )
            return [
                _snapshot_json_value(
                    child,
                    seen=seen,
                    budget=budget,
                    depth=depth + 1,
                )
                for child in value
            ]
        finally:
            seen.remove(identity)
    raise RedactionInputError(
        "resource must contain only JSON-compatible mappings, lists, and scalars"
    )


def _bounded_resource_items(value: Mapping[Any, Any]) -> list[tuple[Any, Any]]:
    try:
        items = list(itertools.islice(value.items(), MAX_CONTAINER_ITEMS + 1))
    except Exception:  # noqa: BLE001 - mappings are caller-controlled protocols.
        raise RedactionInputError("resource object could not be read") from None
    if len(items) > MAX_CONTAINER_ITEMS:
        raise RedactionInputError("resource object exceeds the supported item limit")
    result: list[tuple[Any, Any]] = []
    for item in items:
        if type(item) not in {list, tuple} or len(item) != 2:
            raise RedactionInputError("resource object contains an invalid entry")
        result.append((item[0], item[1]))
    return result


def _validate_resource_key(value: Any) -> None:
    if (
        type(value) is not str
        or not value
        or len(value) > MAX_KEY_LENGTH
        or not value.isprintable()
    ):
        raise RedactionInputError("resource object keys must be safe bounded strings")


def _add_string_to_budget(value: str, *, budget: _ResourceBudget) -> None:
    if len(value) > MAX_STRING_CHARS:
        raise RedactionInputError("resource contains an oversized string")
    budget.string_chars += len(value)
    if budget.string_chars > MAX_TOTAL_STRING_CHARS:
        raise RedactionInputError("resource exceeds the supported string budget")


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
        if len(targets) >= MAX_REDACTION_MATCHES:
            raise RedactionContractError("redaction matches exceed the supported limit")
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
