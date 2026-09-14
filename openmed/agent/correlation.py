"""Opaque correlation identifiers for local agent runs and actions.

Identifiers contain only a fixed kind prefix and 128 random bits. They never
derive from prompts, clinical text, filenames, tool arguments, or operator
identity. Validation fails closed with field names and stable error codes
instead of echoing rejected values.
"""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Final, TypeVar

CORRELATION_SCHEMA_VERSION: Final = "openmed.agent.correlation.v1"
CORRELATION_TOKEN_BYTES: Final = 16
RUN_ID_PREFIX: Final = "run_"
ACTION_ID_PREFIX: Final = "act_"

_TOKEN_HEX_LENGTH: Final = CORRELATION_TOKEN_BYTES * 2
_LOWER_HEX_RE = re.compile(rf"[0-9a-f]{{{_TOKEN_HEX_LENGTH}}}")
_ALLOWED_FIELDS = frozenset(
    {"schema_version", "run_id", "action_id", "parent_action_id"}
)
_ORDERED_FIELDS = (
    "schema_version",
    "run_id",
    "action_id",
    "parent_action_id",
)
_TOKEN_SOURCE_FAILED = object()

_TokenSource = Callable[[int], bytes]
_IdentifierT = TypeVar("_IdentifierT", bound="_OpaqueCorrelationId")


class CorrelationIdError(ValueError):
    """Raised when correlation metadata fails closed validation.

    Args:
        code: Stable machine-readable validation code.
        field_name: Optional public field associated with the failure.
    """

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


@dataclass(frozen=True, slots=True, repr=False)
class _OpaqueCorrelationId:
    """Shared immutable implementation for kind-specific opaque identifiers."""

    value: str
    _prefix: ClassVar[str] = ""
    _field_name: ClassVar[str] = "identifier"

    def __post_init__(self) -> None:
        _validate_identifier(
            self.value,
            prefix=self._prefix,
            field_name=self._field_name,
        )

    @classmethod
    def generate(
        cls: type[_IdentifierT],
        *,
        token_source: _TokenSource | None = None,
    ) -> _IdentifierT:
        """Generate an identifier from 128 random bits.

        Args:
            token_source: Optional byte source for deterministic tests. Runtime
                callers should leave this unset to use :func:`secrets.token_bytes`.

        Returns:
            A canonical opaque identifier of the requested kind.

        Raises:
            CorrelationIdError: If an injected source fails or does not return
                exactly 16 bytes.
        """

        return cls(
            _generate_identifier(
                cls._prefix,
                field_name=cls._field_name,
                token_source=token_source,
            )
        )

    @classmethod
    def parse(cls: type[_IdentifierT], value: Any) -> _IdentifierT:
        """Parse a canonical identifier without normalizing its input."""

        return cls(value)

    def serialize(self) -> str:
        """Return the canonical string representation."""

        return self.value

    def __str__(self) -> str:
        """Return the canonical string representation."""

        return self.value

    def __repr__(self) -> str:
        """Return a value-free representation for diagnostic output."""

        return f"{type(self).__name__}(<opaque>)"


class RunId(_OpaqueCorrelationId):
    """Opaque identifier for one agent run.

    Args:
        value: Canonical ``run_`` identifier containing 32 lowercase hex
            characters after the prefix.
    """

    __slots__ = ()
    _prefix = RUN_ID_PREFIX
    _field_name = "run_id"


class ActionId(_OpaqueCorrelationId):
    """Opaque identifier for one action within an agent run.

    Args:
        value: Canonical ``act_`` identifier containing 32 lowercase hex
            characters after the prefix.
    """

    __slots__ = ()
    _prefix = ACTION_ID_PREFIX
    _field_name = "action_id"


@dataclass(frozen=True, slots=True)
class ActionCorrelation:
    """Typed parent-child correlation metadata for one agent action.

    Args:
        run_id: Opaque identifier for the containing run.
        action_id: Opaque identifier for this action.
        parent_action_id: Optional identifier for another action that directly
            owns this action.
        schema_version: Stable serialization schema version.
    """

    run_id: RunId
    action_id: ActionId
    parent_action_id: ActionId | None = None
    schema_version: str = CORRELATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.run_id) is not RunId:
            raise CorrelationIdError("wrong_identifier_kind", "run_id")
        if type(self.action_id) is not ActionId:
            raise CorrelationIdError("wrong_identifier_kind", "action_id")
        if (
            self.parent_action_id is not None
            and type(self.parent_action_id) is not ActionId
        ):
            raise CorrelationIdError("wrong_identifier_kind", "parent_action_id")
        if self.parent_action_id == self.action_id:
            raise CorrelationIdError("self_parent", "parent_action_id")
        if (
            type(self.schema_version) is not str
            or self.schema_version != CORRELATION_SCHEMA_VERSION
        ):
            raise CorrelationIdError("invalid_schema_version", "schema_version")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionCorrelation":
        """Build typed correlation metadata from a strict mapping.

        Args:
            data: Mapping containing run, action, and optional parent identifiers.

        Returns:
            Validated action correlation metadata.

        Raises:
            CorrelationIdError: If fields are missing, unknown, malformed, or
                use the wrong identifier kind.
        """

        if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
            raise CorrelationIdError("not_a_mapping")

        try:
            fields = set(data)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            fields = None
        if fields is None:
            raise CorrelationIdError("not_a_mapping")
        if fields - _ALLOWED_FIELDS:
            raise CorrelationIdError("unknown_field")
        if "run_id" not in fields or "action_id" not in fields:
            raise CorrelationIdError("missing_field")

        values = _read_mapping_values(data)
        parent_value = values.get("parent_action_id")
        return cls(
            run_id=RunId.parse(values["run_id"]),
            action_id=ActionId.parse(values["action_id"]),
            parent_action_id=(
                None if parent_value is None else _parse_parent_action_id(parent_value)
            ),
            schema_version=values.get(
                "schema_version",
                CORRELATION_SCHEMA_VERSION,
            ),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "ActionCorrelation":
        """Build typed correlation metadata from a strict JSON object."""

        try:
            data = json.loads(payload, object_pairs_hook=_strict_json_object)
        except (
            json.JSONDecodeError,
            CorrelationIdError,
            TypeError,
            UnicodeDecodeError,
        ):
            pass
        else:
            return cls.from_dict(data)
        raise CorrelationIdError("malformed_json")

    @property
    def is_root_action(self) -> bool:
        """Return whether this action has no parent action."""

        return self.parent_action_id is None

    def to_dict(self) -> dict[str, str | None]:
        """Return deterministic metadata-only correlation fields."""

        values: dict[str, str | None] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id.serialize(),
            "action_id": self.action_id.serialize(),
            "parent_action_id": (
                None
                if self.parent_action_id is None
                else self.parent_action_id.serialize()
            ),
        }
        return {field_name: values[field_name] for field_name in _ORDERED_FIELDS}

    def to_json(self) -> str:
        """Return compact JSON with deterministic key ordering."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _generate_identifier(
    prefix: str,
    *,
    field_name: str,
    token_source: _TokenSource | None,
) -> str:
    source = secrets.token_bytes if token_source is None else token_source
    if not callable(source):
        raise CorrelationIdError("invalid_token_source", field_name)

    token: object
    try:
        token = source(CORRELATION_TOKEN_BYTES)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        token = _TOKEN_SOURCE_FAILED

    if type(token) is not bytes or len(token) != CORRELATION_TOKEN_BYTES:
        raise CorrelationIdError("invalid_token_source", field_name)
    return f"{prefix}{token.hex()}"


def _validate_identifier(value: Any, *, prefix: str, field_name: str) -> None:
    if type(value) is not str:
        raise CorrelationIdError("invalid_identifier", field_name)
    token = value.removeprefix(prefix)
    if not value.startswith(prefix) or _LOWER_HEX_RE.fullmatch(token) is None:
        other_prefix = ACTION_ID_PREFIX if prefix == RUN_ID_PREFIX else RUN_ID_PREFIX
        other_token = value.removeprefix(other_prefix)
        if value.startswith(other_prefix) and _LOWER_HEX_RE.fullmatch(other_token):
            raise CorrelationIdError("wrong_identifier_kind", field_name)
        raise CorrelationIdError("invalid_identifier", field_name)


def _parse_parent_action_id(value: Any) -> ActionId:
    _validate_identifier(
        value,
        prefix=ACTION_ID_PREFIX,
        field_name="parent_action_id",
    )
    return ActionId(value)


def _read_mapping_values(data: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return {field_name: data[field_name] for field_name in data}
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        pass
    raise CorrelationIdError("unreadable_mapping")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CorrelationIdError("duplicate_field")
        result[key] = value
    return result


__all__ = [
    "ACTION_ID_PREFIX",
    "CORRELATION_SCHEMA_VERSION",
    "CORRELATION_TOKEN_BYTES",
    "RUN_ID_PREFIX",
    "ActionCorrelation",
    "ActionId",
    "CorrelationIdError",
    "RunId",
]
