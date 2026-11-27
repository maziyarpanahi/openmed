"""Versioned registry of every stable OpenMed error code.

``openmed/core/error_codes.json`` is the single source of truth for the codes
OpenMed reports. Each entry names one code, its transport-neutral category,
whether a client kit can raise it (``portable``), whether retrying the same
request may succeed, the Python classes that carry it, and how each public
surface reports the same failure today: REST status and ``error.code``, gRPC
status and ``openmed-reason`` trailer, GraphQL ``extensions.code``, MCP tool
error code, and CLI exit code. Where surfaces disagree, the entry points at a
recorded divergence instead of hiding the difference.

The registry describes current behaviour; it changes no wire format. Codes are
never reused: a code that stops being emitted moves to ``retired_codes``.

The loader is standard-library only, reads one packaged file, makes no network
call and holds no request content. Every value it returns is an identifier,
status, category or fixed explanation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Optional, Union

__all__ = [
    "CATEGORIES",
    "ERROR_REGISTRY_CONTRACT",
    "ERROR_REGISTRY_SCHEMA_VERSION",
    "RETRYABLE_BASES",
    "SURFACES",
    "CliMapping",
    "ErrorCodeEntry",
    "ErrorCodeRegistry",
    "ErrorDivergence",
    "ErrorRegistryError",
    "GraphQLMapping",
    "GrpcMapping",
    "HttpMapping",
    "McpMapping",
    "load_error_registry",
    "load_error_registry_schema",
    "lookup_error_code",
    "parse_error_registry",
]

ERROR_REGISTRY_CONTRACT: Final = "openmed.errors.code-registry"
ERROR_REGISTRY_SCHEMA_VERSION: Final = 1
ERROR_REGISTRY_SCHEMA_NAME: Final = "error_code_registry"
_DATA_PACKAGE: Final = "openmed.core"
_DATA_FILE: Final = "error_codes.json"
_SCHEMA_PACKAGE: Final = "openmed.core.schemas.json"

#: Transport-neutral categories, shared with ``docs/api/protocol-map.json``.
CATEGORIES: Final = (
    "unauthenticated",
    "permission_denied",
    "invalid_argument",
    "resource_exhausted",
    "deadline_exceeded",
    "cancelled",
    "unavailable",
    "internal",
)
#: Evidence a retryable flag is derived from; ``undetermined`` means unknown.
RETRYABLE_BASES: Final = (
    "retry_after_header",
    "retryable_signal",
    "transient_capacity",
    "request_must_change",
    "configuration_must_change",
    "server_fault",
    "undetermined",
)
_RETRYABLE_TRUE: Final = frozenset(
    {"retry_after_header", "retryable_signal", "transient_capacity"}
)
_RETRYABLE_FALSE: Final = frozenset(
    {"request_must_change", "configuration_must_change", "server_fault"}
)
#: Public surfaces an entry maps, in document order.
SURFACES: Final = ("python", "http", "grpc", "graphql", "mcp", "cli")
_HTTP_APPS: Final = frozenset({"service", "redaction", "privacy_proxy"})
_DELIVERIES: Final = frozenset({"response", "stream_event"})
_GRPC_ORIGINS: Final = frozenset({"server", "transport"})
_MCP_DELIVERIES: Final = frozenset({"tool_error", "failed_result"})
_DIVERGENCE_KINDS: Final = frozenset({"category", "naming", "status", "documentation"})
_EXIT_CODES: Final = frozenset({0, 1, 2})

_CODE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_PYTHON_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$")
_GRPC_STATUS = re.compile(r"^[A-Z][A-Z_]{1,31}$")
_MAX_TEXT = 512


class ErrorRegistryError(ValueError):
    """Raised when the error-code registry document is not valid.

    Messages are fixed strings; they never echo values from the document.
    """


@dataclass(frozen=True)
class HttpMapping:
    """How the REST surface reports a failure.

    Attributes:
        code: ``error.code`` REST sends.
        status: HTTP status, or ``None`` for an in-band stream event.
        delivery: ``response``, or ``stream_event`` once an NDJSON stream
            has started.
        app: The FastAPI app that answers: ``service``, ``redaction`` or
            ``privacy_proxy``.
        retry_after: Whether the response carries ``Retry-After``.
        note: Fixed explanation when one code has several variants.
    """

    code: str
    status: Optional[int] = None
    delivery: str = "response"
    app: str = "service"
    retry_after: bool = False
    note: Optional[str] = None


@dataclass(frozen=True)
class GrpcMapping:
    """How the gRPC surface reports a failure.

    Attributes:
        status: Canonical gRPC status name.
        reason: ``openmed-reason`` trailer, or ``None`` when none is sent.
        details: Code carried in the status details when no reason is sent.
        retryable_metadata: Whether the ``openmed-retryable`` trailer is set.
        origin: ``server``, or ``transport`` when grpcio itself refuses.
        note: Fixed explanation when one code has several variants.
    """

    status: str
    reason: Optional[str] = None
    details: Optional[str] = None
    retryable_metadata: bool = False
    origin: str = "server"
    note: Optional[str] = None


@dataclass(frozen=True)
class GraphQLMapping:
    """How the GraphQL surface reports a failure inside ``errors[]``.

    Attributes:
        code: ``extensions.code``, or ``None`` for an uncoded error.
        http_status: HTTP status of the GraphQL response.
        retry_after: Whether the response carries ``Retry-After``.
        retryable_extension: Whether ``extensions.retryable`` is ``true``.
        note: Fixed explanation when one code has several variants.
    """

    code: Optional[str]
    http_status: int = 200
    retry_after: bool = False
    retryable_extension: bool = False
    note: Optional[str] = None


@dataclass(frozen=True)
class McpMapping:
    """How an MCP tool reports a failure in its structured error result.

    Attributes:
        code: ``structuredContent.error.code``, or the ``error.code`` of a
            failed stage inside a successful result.
        delivery: ``tool_error`` for an ``isError`` result, or
            ``failed_result`` for a successful result whose stage failed.
    """

    code: str
    delivery: str = "tool_error"


@dataclass(frozen=True)
class CliMapping:
    """How the CLI reports a failure.

    Attributes:
        exit_code: Process exit status (0, 1 or 2).
        code: ``error.code`` in ``--json`` output.
    """

    exit_code: int
    code: str


@dataclass(frozen=True)
class ErrorCodeEntry:
    """One registered error code and its per-surface mappings.

    Attributes:
        code: The stable code.
        category: Transport-neutral category from :data:`CATEGORIES`.
        description: One-line, value-free description.
        portable: Whether a client kit (Swift, Kotlin, web) can raise it.
        retryable: Whether retrying the unchanged request may succeed, or
            ``None`` when current behaviour does not establish it.
        retryable_basis: Evidence for ``retryable`` from
            :data:`RETRYABLE_BASES`.
        python: Dotted paths of the Python classes that carry the code.
        http: REST mappings; the first ``response`` mapping is primary.
        grpc: gRPC mappings.
        graphql: GraphQL mappings.
        mcp: MCP mappings.
        cli: CLI mappings.
        divergences: Identifiers of recorded cross-surface divergences.
    """

    code: str
    category: str
    description: str
    portable: bool
    retryable: Optional[bool]
    retryable_basis: str
    python: tuple[str, ...]
    http: tuple[HttpMapping, ...]
    grpc: tuple[GrpcMapping, ...]
    graphql: tuple[GraphQLMapping, ...]
    mcp: tuple[McpMapping, ...]
    cli: tuple[CliMapping, ...]
    divergences: tuple[str, ...]

    @property
    def http_status(self) -> Optional[int]:
        """Return the primary REST status, or ``None`` when REST never answers."""
        for mapping in self.http:
            if mapping.delivery == "response":
                return mapping.status
        return None

    @property
    def cli_exit_code(self) -> Optional[int]:
        """Return the primary CLI exit code, or ``None`` when the CLI never reports."""
        return self.cli[0].exit_code if self.cli else None


@dataclass(frozen=True)
class ErrorDivergence:
    """A recorded difference between surfaces that the registry does not fix.

    Attributes:
        id: Stable identifier.
        kind: ``category``, ``naming``, ``status`` or ``documentation``.
        surfaces: Surfaces the divergence concerns.
        summary: Fixed explanation.
        follow_up: The change that would remove it.
        protocol_map: Matching ``known_divergences`` id in the protocol map.
    """

    id: str
    kind: str
    surfaces: tuple[str, ...]
    summary: str
    follow_up: str
    protocol_map: Optional[str] = None


@dataclass(frozen=True)
class ErrorCodeRegistry:
    """The parsed error-code registry.

    Attributes:
        contract_version: Semantic version of the registry content.
        codes: Entries keyed by code, in sorted order.
        divergences: Recorded divergences keyed by id.
        retired_codes: Codes that must never be reused.
        exit_codes: CLI exit status to its fixed meaning.
    """

    contract_version: str
    codes: Mapping[str, ErrorCodeEntry]
    divergences: Mapping[str, ErrorDivergence]
    retired_codes: tuple[str, ...]
    exit_codes: Mapping[int, str]

    def entry(self, code: str) -> ErrorCodeEntry:
        """Return the entry for ``code``.

        Args:
            code: A registered error code.

        Returns:
            The registry entry.

        Raises:
            KeyError: If the code is not registered.
        """
        try:
            return self.codes[code]
        except KeyError:
            raise KeyError("The error code is not registered") from None

    def portable_codes(self) -> tuple[ErrorCodeEntry, ...]:
        """Return the entries a client kit can raise, sorted by code."""
        return tuple(entry for entry in self.codes.values() if entry.portable)


def parse_error_registry(payload: Any) -> ErrorCodeRegistry:
    """Validate and parse a registry document.

    Args:
        payload: The decoded JSON document.

    Returns:
        The parsed registry.

    Raises:
        ErrorRegistryError: If the document is structurally invalid, repeats
            or reuses a code, or references an unknown divergence.
    """
    document = _mapping(payload)
    if document.get("schema_version") != ERROR_REGISTRY_SCHEMA_VERSION:
        raise ErrorRegistryError("The error registry schema_version is unsupported")
    if document.get("contract") != ERROR_REGISTRY_CONTRACT:
        raise ErrorRegistryError("The error registry contract is unknown")
    version = document.get("contract_version")
    if not isinstance(version, str) or _SEMVER.fullmatch(version) is None:
        raise ErrorRegistryError("The error registry contract_version is invalid")
    if tuple(_sequence(document.get("categories"))) != CATEGORIES:
        raise ErrorRegistryError("The error registry categories are not canonical")

    divergences: dict[str, ErrorDivergence] = {}
    for item in _sequence(document.get("divergences")):
        divergence = _parse_divergence(item)
        if divergence.id in divergences:
            raise ErrorRegistryError("The error registry repeats a divergence")
        divergences[divergence.id] = divergence

    retired = tuple(_code(value) for value in _sequence(document.get("retired_codes")))
    entries: dict[str, ErrorCodeEntry] = {}
    folded: set[str] = {code.casefold() for code in retired}
    if len(folded) != len(retired):
        raise ErrorRegistryError("The error registry repeats a retired code")
    for item in _sequence(document.get("codes")):
        entry = _parse_entry(item)
        key = entry.code.casefold()
        if key in folded:
            raise ErrorRegistryError("The error registry repeats or reuses a code")
        folded.add(key)
        for divergence_id in entry.divergences:
            if divergence_id not in divergences:
                raise ErrorRegistryError("An error code names an unknown divergence")
        entries[entry.code] = entry

    used = {name for entry in entries.values() for name in entry.divergences}
    if set(divergences) - used:
        raise ErrorRegistryError("A recorded divergence names no error code")

    surfaces = _mapping(document.get("surfaces"))
    cli = _mapping(surfaces.get("cli"))
    exit_codes: dict[int, str] = {}
    for key, meaning in _mapping(cli.get("exit_codes")).items():
        if not isinstance(key, str) or not key.isdigit() or int(key) not in _EXIT_CODES:
            raise ErrorRegistryError("The error registry CLI exit codes are invalid")
        exit_codes[int(key)] = _text(meaning)
    if set(exit_codes) != _EXIT_CODES:
        raise ErrorRegistryError("The error registry CLI exit codes are incomplete")

    return ErrorCodeRegistry(
        contract_version=version,
        codes=MappingProxyType(dict(sorted(entries.items()))),
        divergences=MappingProxyType(divergences),
        retired_codes=retired,
        exit_codes=MappingProxyType(dict(sorted(exit_codes.items()))),
    )


def load_error_registry(path: Union[Path, str, None] = None) -> ErrorCodeRegistry:
    """Load and validate the packaged registry, or one at ``path``.

    Args:
        path: Optional registry document to load instead of the packaged one.

    Returns:
        The parsed registry. The packaged registry is cached.

    Raises:
        ErrorRegistryError: If the document cannot be read or is invalid.
    """
    if path is None:
        return _packaged_registry()
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ErrorRegistryError("The error registry could not be read") from exc
    return parse_error_registry(_decode(text))


def lookup_error_code(code: str) -> ErrorCodeEntry:
    """Return the packaged registry entry for ``code``.

    Args:
        code: A registered error code.

    Returns:
        The registry entry.

    Raises:
        KeyError: If the code is not registered.
    """
    return load_error_registry().entry(code)


def load_error_registry_schema() -> dict[str, Any]:
    """Return the packaged JSON Schema of the registry document."""
    resource = resources.files(_SCHEMA_PACKAGE).joinpath(
        f"{ERROR_REGISTRY_SCHEMA_NAME}.schema.json"
    )
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _packaged_registry() -> ErrorCodeRegistry:
    resource = resources.files(_DATA_PACKAGE).joinpath(_DATA_FILE)
    try:
        text = resource.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ErrorRegistryError("The error registry could not be read") from exc
    return parse_error_registry(_decode(text))


def _decode(text: str) -> Any:
    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ErrorRegistryError("The error registry repeats a JSON key")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=unique_object)
    except json.JSONDecodeError as exc:
        raise ErrorRegistryError("The error registry is not valid JSON") from exc


def _parse_entry(raw: Any) -> ErrorCodeEntry:
    item = _mapping(raw)
    code = _code(item.get("code"))
    category = item.get("category")
    if category not in CATEGORIES:
        raise ErrorRegistryError("An error code has an unknown category")
    basis = item.get("retryable_basis")
    retryable = item.get("retryable")
    if basis in _RETRYABLE_TRUE:
        expected: Optional[bool] = True
    elif basis in _RETRYABLE_FALSE:
        expected = False
    elif basis == "undetermined":
        expected = None
    else:
        raise ErrorRegistryError("An error code has an unknown retryable basis")
    if retryable is not expected:
        raise ErrorRegistryError("An error code's retryable flag contradicts its basis")
    portable = _flag(item.get("portable"))
    python = tuple(_python_path(value) for value in _sequence(item.get("python")))
    http = tuple(_http(value) for value in _sequence(item.get("http")))
    grpc = tuple(_grpc(value) for value in _sequence(item.get("grpc")))
    graphql = tuple(_graphql(value) for value in _sequence(item.get("graphql")))
    mcp = tuple(_mcp(value) for value in _sequence(item.get("mcp")))
    cli = tuple(_cli(value) for value in _sequence(item.get("cli")))
    if portable and (retryable is None or not python):
        raise ErrorRegistryError("A portable error code needs a class and a flag")
    divergences = tuple(_identifier(v) for v in _sequence(item.get("divergences")))
    if len(set(divergences)) != len(divergences):
        raise ErrorRegistryError("An error code repeats a divergence")
    return ErrorCodeEntry(
        code=code,
        category=str(category),
        description=_text(item.get("description")),
        portable=portable,
        retryable=retryable,
        retryable_basis=str(basis),
        python=python,
        http=http,
        grpc=grpc,
        graphql=graphql,
        mcp=mcp,
        cli=cli,
        divergences=divergences,
    )


def _http(raw: Any) -> HttpMapping:
    item = _mapping(raw)
    delivery = item.get("delivery", "response")
    if delivery not in _DELIVERIES:
        raise ErrorRegistryError("An HTTP mapping has an unknown delivery")
    status = item.get("status")
    if delivery == "response":
        status = _http_status(status)
    elif status is not None:
        raise ErrorRegistryError("A stream-event mapping cannot carry a status")
    app = item.get("app", "service")
    if app not in _HTTP_APPS:
        raise ErrorRegistryError("An HTTP mapping names an unknown app")
    return HttpMapping(
        code=_code(item.get("code")),
        status=status,
        delivery=str(delivery),
        app=str(app),
        retry_after=_flag(item.get("retry_after", False)),
        note=_optional_text(item.get("note")),
    )


def _grpc(raw: Any) -> GrpcMapping:
    item = _mapping(raw)
    status = item.get("status")
    if not isinstance(status, str) or _GRPC_STATUS.fullmatch(status) is None:
        raise ErrorRegistryError("A gRPC mapping has an invalid status")
    reason = item.get("reason")
    details = item.get("details")
    origin = item.get("origin", "server")
    if origin not in _GRPC_ORIGINS:
        raise ErrorRegistryError("A gRPC mapping has an unknown origin")
    if reason is not None and details is not None:
        raise ErrorRegistryError("A gRPC mapping carries both a reason and details")
    return GrpcMapping(
        status=status,
        reason=None if reason is None else _code(reason),
        details=None if details is None else _code(details),
        retryable_metadata=_flag(item.get("retryable_metadata", False)),
        origin=str(origin),
        note=_optional_text(item.get("note")),
    )


def _graphql(raw: Any) -> GraphQLMapping:
    item = _mapping(raw)
    if "code" not in item:
        raise ErrorRegistryError("A GraphQL mapping is missing its code")
    code = item.get("code")
    return GraphQLMapping(
        code=None if code is None else _code(code),
        http_status=_http_status(item.get("http_status", 200)),
        retry_after=_flag(item.get("retry_after", False)),
        retryable_extension=_flag(item.get("retryable_extension", False)),
        note=_optional_text(item.get("note")),
    )


def _mcp(raw: Any) -> McpMapping:
    item = _mapping(raw)
    delivery = item.get("delivery", "tool_error")
    if delivery not in _MCP_DELIVERIES:
        raise ErrorRegistryError("An MCP mapping has an unknown delivery")
    return McpMapping(code=_code(item.get("code")), delivery=str(delivery))


def _cli(raw: Any) -> CliMapping:
    item = _mapping(raw)
    exit_code = item.get("exit_code")
    if type(exit_code) is not int or exit_code not in _EXIT_CODES - {0}:
        raise ErrorRegistryError("A CLI mapping has an invalid exit code")
    return CliMapping(exit_code=exit_code, code=_code(item.get("code")))


def _parse_divergence(raw: Any) -> ErrorDivergence:
    item = _mapping(raw)
    kind = item.get("kind")
    if kind not in _DIVERGENCE_KINDS:
        raise ErrorRegistryError("A divergence has an unknown kind")
    surfaces = tuple(_sequence(item.get("surfaces")))
    if not surfaces or any(surface not in SURFACES for surface in surfaces):
        raise ErrorRegistryError("A divergence names an unknown surface")
    protocol_map = item.get("protocol_map")
    return ErrorDivergence(
        id=_identifier(item.get("id")),
        kind=str(kind),
        surfaces=tuple(str(surface) for surface in surfaces),
        summary=_text(item.get("summary")),
        follow_up=_text(item.get("follow_up")),
        protocol_map=None if protocol_map is None else _identifier(protocol_map),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ErrorRegistryError("The error registry has a malformed object")
    return value


def _sequence(value: Any) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ErrorRegistryError("The error registry has a malformed list")
    return value


def _code(value: Any) -> str:
    if not isinstance(value, str) or _CODE.fullmatch(value) is None:
        raise ErrorRegistryError("The error registry has an invalid code")
    return value


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ErrorRegistryError("The error registry has an invalid identifier")
    return value


def _python_path(value: Any) -> str:
    if not isinstance(value, str) or _PYTHON_PATH.fullmatch(value) is None:
        raise ErrorRegistryError("The error registry has an invalid class path")
    return value


def _http_status(value: Any) -> int:
    if type(value) is not int or not 100 <= value <= 599:
        raise ErrorRegistryError("The error registry has an invalid HTTP status")
    return value


def _flag(value: Any) -> bool:
    if type(value) is not bool:
        raise ErrorRegistryError("The error registry has an invalid flag")
    return value


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_TEXT:
        raise ErrorRegistryError("The error registry is missing an explanation")
    return value


def _optional_text(value: Any) -> Optional[str]:
    return None if value is None else _text(value)
