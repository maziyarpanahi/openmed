"""REST, GraphQL and gRPC protocol map and its offline drift check.

``docs/api/protocol-map.json`` is the versioned contract for the operations the
three service transports share. For each operation it names the REST route,
GraphQL field and gRPC method, the request and response fields that correspond,
and the operations a transport deliberately does not offer. For each controlled
outcome it names the canonical, transport-neutral category and how each
transport reports it: REST status and error code, GraphQL ``extensions.code``
(or the REST envelope when HTTP middleware refuses ``/graphql``), and gRPC
status with its ``openmed-reason`` trailer. Genuine differences are recorded as
known divergences with an owning follow-up instead of being hidden.

:func:`check_protocol_map` compares the document with what each transport can
emit, read from local source files and committed contracts: the REST error
codes, the REST stream error events, the GraphQL extension codes, the gRPC
failure table, the gRPC methods, the GraphQL query fields, the route scopes and
the request fields of ``openapi.json``, ``graphql-schema.graphql`` and the
protobuf descriptors. It makes no network call and loads no model. Reports
contain only identifiers, codes, statuses and field paths.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional, Union

from .api_compatibility import (
    DEFAULT_CONTRACT_PATH,
    REPOSITORY_ROOT,
    discover_service_error_categories,
)

SCHEMA_VERSION = 1
PROTOCOL_MAP_CONTRACT = "openmed.service.protocol-map"
DEFAULT_PROTOCOL_MAP_PATH = REPOSITORY_ROOT / "docs" / "api" / "protocol-map.json"
DEFAULT_GRAPHQL_SDL_PATH = REPOSITORY_ROOT / "docs" / "api" / "graphql-schema.graphql"
SERVICE_SOURCE_ROOT = REPOSITORY_ROOT / "openmed" / "service"

TRANSPORTS = ("rest", "graphql", "grpc")
CATEGORIES = (
    "unauthenticated",
    "permission_denied",
    "invalid_argument",
    "resource_exhausted",
    "deadline_exceeded",
    "cancelled",
    "unavailable",
    "internal",
)
ORIGINS = ("server", "transport", "client")
DELIVERIES = ("response", "stream_event")
ENVELOPE_GRAPHQL = "graphql"
ENVELOPE_REST = "rest"
DIVERGENCE_KINDS = ("category", "naming")
NORMALIZERS = ("code", "count", "digest", "offset", "opaque", "score")

_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_OPERATION_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_FIELD_SEGMENT = r"[A-Za-z][A-Za-z0-9_]*(\[\])?"
_FIELD_PATH = re.compile(rf"^{_FIELD_SEGMENT}(\.{_FIELD_SEGMENT})*$")
_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_GRPC_STATUS = re.compile(r"^[A-Z][A-Z_]{1,31}$")
_HTTP_OK = 200
_STREAM_SOURCE = "streaming.py"
_GRAPHQL_SOURCE_GLOB = "graphql_*.py"
_WELL_KNOWN_PROTO_PACKAGE = "google.protobuf."
_MAX_PATH_DEPTH = 3

Status = Union[int, str, None]


class ProtocolMapError(ValueError):
    """Raised when the protocol map document is not structurally valid.

    Messages are fixed strings; they never echo values from the document.
    """


class ProtocolDriftError(RuntimeError):
    """Raised when a transport and the protocol map have drifted apart.

    Attributes:
        report: The value-free drift report.
    """

    def __init__(self, report: "ProtocolDriftReport") -> None:
        self.report = report
        message = "; ".join(issue.describe() for issue in report.issues)
        super().__init__(message or "Protocol map drift check failed")


@dataclass(frozen=True)
class OutcomeVariant:
    """One way a transport reports an outcome.

    Attributes:
        transport: ``rest``, ``graphql`` or ``grpc``.
        status: HTTP status, gRPC status name, or ``None`` when the client
            abandoned the request and nothing is delivered.
        code: REST error code, GraphQL extension code (or REST code under the
            REST envelope), or gRPC ``openmed-reason``; ``None`` when the
            transport carries no code.
        envelope: For GraphQL, ``graphql`` for an ``errors[]`` entry or
            ``rest`` for an HTTP middleware refusal; ``None`` otherwise.
        delivery: ``response``, or ``stream_event`` for an in-band REST NDJSON
            error event after the stream has started.
        origin: ``server``, ``transport`` (grpcio itself) or ``client``.
    """

    transport: str
    status: Status
    code: Optional[str] = None
    envelope: Optional[str] = None
    delivery: str = "response"
    origin: str = "server"

    def matches(self, observed: "ObservedOutcome") -> bool:
        """Return whether an observed result is this variant."""
        return (
            observed.transport == self.transport
            and observed.status == self.status
            and observed.code == self.code
            and observed.envelope == self.envelope
            and observed.delivery == self.delivery
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free variant mapping."""
        payload: dict[str, Any] = {
            "transport": self.transport,
            "status": self.status,
            "code": self.code,
            "origin": self.origin,
        }
        if self.envelope is not None:
            payload["envelope"] = self.envelope
        if self.delivery != "response":
            payload["delivery"] = self.delivery
        return payload


@dataclass(frozen=True)
class ObservedOutcome:
    """A transport result reduced to status, code and envelope only.

    Attributes:
        transport: ``rest``, ``graphql`` or ``grpc``.
        status: HTTP status, gRPC status name, or ``None`` for no response.
        code: The error code, GraphQL extension code or ``openmed-reason``.
        envelope: For GraphQL, ``graphql`` or ``rest``; ``None`` otherwise.
        delivery: ``response`` or ``stream_event``.
    """

    transport: str
    status: Status
    code: Optional[str] = None
    envelope: Optional[str] = None
    delivery: str = "response"

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free observation."""
        payload: dict[str, Any] = {
            "transport": self.transport,
            "status": self.status,
            "code": self.code,
        }
        if self.envelope is not None:
            payload["envelope"] = self.envelope
        if self.delivery != "response":
            payload["delivery"] = self.delivery
        return payload


@dataclass(frozen=True)
class ProtocolOutcome:
    """A controlled outcome and how each transport reports it.

    Attributes:
        id: Stable outcome identifier.
        category: Canonical transport-neutral category.
        operations: Operations the outcome applies to; empty means all.
        variants: Per-transport report variants.
        not_applicable: Per-transport reason when a transport has no variant.
    """

    id: str
    category: str
    operations: tuple[str, ...] = ()
    variants: Mapping[str, tuple[OutcomeVariant, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    not_applicable: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def applies_to(self, operation: str) -> bool:
        """Return whether this outcome is declared for ``operation``."""
        return not self.operations or operation in self.operations

    def expects(self, observed: ObservedOutcome) -> bool:
        """Return whether ``observed`` is one of this outcome's variants."""
        return any(
            variant.matches(observed)
            for variant in self.variants.get(observed.transport, ())
        )


@dataclass(frozen=True)
class FieldMapping:
    """Corresponding field paths on each transport.

    Attributes:
        field: Canonical field path.
        paths: Transport path, or ``None`` when the transport has no such field.
        normalize: How a conformance check compares values: ``code``,
            ``count``, ``digest``, ``offset``, ``opaque`` or ``score``.
    """

    field: str
    paths: Mapping[str, Optional[str]]
    normalize: Optional[str] = None


@dataclass(frozen=True)
class ProtocolOperation:
    """One shared service operation and its transport bindings.

    Attributes:
        id: Stable operation identifier.
        scopes: Scopes every transport requires.
        bindings: Binding details for each supporting transport.
        unsupported: Reason for each transport that deliberately lacks it.
        request_fields: Corresponding request fields.
        response_fields: Corresponding response fields.
    """

    id: str
    scopes: tuple[str, ...]
    bindings: Mapping[str, Mapping[str, Any]]
    unsupported: Mapping[str, str]
    request_fields: tuple[FieldMapping, ...]
    response_fields: tuple[FieldMapping, ...]

    def supports(self, transport: str) -> bool:
        """Return whether ``transport`` offers this operation."""
        return transport in self.bindings


@dataclass(frozen=True)
class CodeEntry:
    """Status and canonical category of one transport code."""

    status: Status
    category: str


@dataclass(frozen=True)
class KnownDivergence:
    """A recorded, owned difference between transports.

    Attributes:
        id: Stable divergence identifier.
        kind: ``category`` (a variant reports a different category than its
            outcome) or ``naming`` (code naming conventions differ).
        transport: The transport that diverges.
        outcomes: Outcome ids covered by a ``category`` divergence.
        codes: Codes covered by a ``naming`` divergence.
        follow_up: The issue or contract item that owns the fix.
    """

    id: str
    kind: str
    transport: str
    outcomes: tuple[str, ...] = ()
    codes: tuple[str, ...] = ()
    follow_up: str = ""


@dataclass(frozen=True)
class ProtocolMap:
    """Parsed, validated protocol map.

    Attributes:
        contract_version: Semantic version of the mapping content.
        operations: Shared operations.
        outcomes: Controlled outcomes.
        codes: Per-transport code tables.
        rest_only_codes: REST codes emitted only outside shared operations,
            grouped by route family.
        unmapped_operations: Per-transport operations outside the map, with
            a reason each.
        divergences: Known, owned divergences.
        grpc_status_categories: Category of gRPC statuses that carry no
            ``openmed-reason`` (transport- or client-originated).
        graphql_uncoded_category: Category of GraphQL errors with no code.
    """

    contract_version: str
    operations: tuple[ProtocolOperation, ...]
    outcomes: tuple[ProtocolOutcome, ...]
    codes: Mapping[str, Mapping[str, CodeEntry]]
    rest_only_codes: Mapping[str, tuple[str, ...]]
    unmapped_operations: Mapping[str, Mapping[str, str]]
    divergences: tuple[KnownDivergence, ...]
    grpc_status_categories: Mapping[str, str]
    graphql_uncoded_category: str

    def operation(self, operation_id: str) -> ProtocolOperation:
        """Return the operation with ``operation_id``.

        Raises:
            KeyError: If the operation is not mapped.
        """
        for operation in self.operations:
            if operation.id == operation_id:
                return operation
        raise KeyError("unknown protocol operation")

    def outcome(self, outcome_id: str) -> ProtocolOutcome:
        """Return the outcome with ``outcome_id``.

        Raises:
            KeyError: If the outcome is not mapped.
        """
        for outcome in self.outcomes:
            if outcome.id == outcome_id:
                return outcome
        raise KeyError("unknown protocol outcome")

    def category_of(self, variant: Union[OutcomeVariant, ObservedOutcome]) -> str:
        """Return the canonical category a transport result reports.

        Args:
            variant: A declared variant or an observation.

        Returns:
            The category from the code tables, or from the transport's status
            categories when the result carries no code.

        Raises:
            KeyError: If the code or status is not mapped.
        """
        if variant.status is None:
            return "cancelled"
        if variant.transport == "grpc":
            if variant.code is None:
                return self.grpc_status_categories[str(variant.status)]
            return self.codes["grpc"][variant.code].category
        if variant.transport == "graphql" and variant.envelope == ENVELOPE_GRAPHQL:
            if variant.code is None:
                return self.graphql_uncoded_category
            return self.codes["graphql"][variant.code].category
        if variant.code is None:
            raise KeyError("unmapped protocol code")
        return self.codes["rest"][variant.code].category

    def classify(self, observed: ObservedOutcome) -> dict[str, Any]:
        """Return a value-free conformance record for one observation.

        Returns:
            The observation plus its category and the outcome ids it matches.
        """
        try:
            category: Optional[str] = self.category_of(observed)
        except KeyError:
            category = None
        record = observed.to_dict()
        record["category"] = category
        record["outcomes"] = sorted(
            outcome.id for outcome in self.outcomes if outcome.expects(observed)
        )
        return record


@dataclass(frozen=True)
class TransportSurface:
    """What each transport can emit and accept, read from source and contracts.

    Attributes:
        rest_codes: REST envelope codes the service source can emit.
        rest_stream_event_codes: Codes of in-band REST NDJSON error events.
        graphql_codes: GraphQL ``extensions.code`` values in GraphQL source.
        grpc_failures: gRPC reason to ``(status name, category)``.
        rest_route_scopes: Default scopes per REST ``(method, path)``.
        graphql_field_scopes: Scopes per GraphQL query field.
        grpc_routes: gRPC method to the REST ``(method, path)`` it mirrors.
        rest_request_fields: Request properties per REST ``(method, path)``.
        graphql_arguments: Argument name to input type per GraphQL query field.
        graphql_input_fields: Input fields per GraphQL query field.
        graphql_response_paths: Response paths per GraphQL query field.
        grpc_request_paths: Request paths per gRPC method.
        grpc_response_paths: Response paths per gRPC method.
    """

    rest_codes: frozenset[str]
    rest_stream_event_codes: frozenset[str]
    graphql_codes: frozenset[str]
    grpc_failures: Mapping[str, tuple[str, str]]
    rest_route_scopes: Mapping[tuple[str, str], tuple[str, ...]]
    graphql_field_scopes: Mapping[str, tuple[str, ...]]
    grpc_routes: Mapping[str, tuple[str, str]]
    rest_request_fields: Mapping[tuple[str, str], frozenset[str]]
    graphql_arguments: Mapping[str, Mapping[str, str]]
    graphql_input_fields: Mapping[str, frozenset[str]]
    graphql_response_paths: Mapping[str, frozenset[str]]
    grpc_request_paths: Mapping[str, frozenset[str]]
    grpc_response_paths: Mapping[str, frozenset[str]]


@dataclass(frozen=True)
class ProtocolDriftIssue:
    """One value-free drift finding.

    Attributes:
        change: Stable finding kind, such as ``unmapped_code``.
        transport: The transport concerned.
        code: A transport code identifier.
        operation: An operation id, gRPC method or GraphQL field.
        outcome: An outcome id.
        field: A field path.
    """

    change: str
    transport: Optional[str] = None
    code: Optional[str] = None
    operation: Optional[str] = None
    outcome: Optional[str] = None
    field: Optional[str] = None

    def to_dict(self) -> dict[str, str]:
        """Return only identifiers, codes and field paths."""
        payload = {"change": self.change}
        for name in ("transport", "code", "operation", "outcome", "field"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload

    def describe(self) -> str:
        """Format the finding without any request or response values."""
        parts = [
            f"{name}={value}"
            for name, value in self.to_dict().items()
            if name != "change"
        ]
        return f"{self.change}: {', '.join(parts)}" if parts else self.change


@dataclass(frozen=True)
class ProtocolDriftReport:
    """Deterministic result of a protocol map drift check.

    Attributes:
        contract_version: Version of the checked protocol map.
        issues: Drift findings, sorted.
        counts: Mapped operation, outcome and code counts.
    """

    contract_version: str
    issues: tuple[ProtocolDriftIssue, ...] = ()
    counts: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        unique = {_issue_key(issue): issue for issue in self.issues}
        ordered = tuple(unique[key] for key in sorted(unique))
        object.__setattr__(self, "issues", ordered)
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))

    @property
    def is_clean(self) -> bool:
        """Return whether the map and every transport agree."""
        return not self.issues

    def assert_clean(self) -> "ProtocolDriftReport":
        """Raise :class:`ProtocolDriftError` when any drift was found."""
        if self.issues:
            raise ProtocolDriftError(self)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return the PHI-free machine-readable report."""
        return {
            "schema_version": SCHEMA_VERSION,
            "contract": PROTOCOL_MAP_CONTRACT,
            "contract_version": self.contract_version,
            "counts": dict(sorted(self.counts.items())),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json(self) -> str:
        """Return stable JSON suitable for a CI artifact."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def _issue_key(issue: ProtocolDriftIssue) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(issue.to_dict().items()))


def load_protocol_map(
    path: Union[Path, str] = DEFAULT_PROTOCOL_MAP_PATH,
) -> ProtocolMap:
    """Load and validate the committed protocol map.

    Args:
        path: Location of ``protocol-map.json``.

    Returns:
        The parsed protocol map.

    Raises:
        ProtocolMapError: If the file cannot be read or is not a valid map.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolMapError("The protocol map could not be read") from exc
    return parse_protocol_map(payload)


def parse_protocol_map(payload: Any) -> ProtocolMap:
    """Validate a protocol map document and return its parsed form.

    Args:
        payload: The decoded JSON document.

    Returns:
        The parsed protocol map.

    Raises:
        ProtocolMapError: If any part of the document is malformed.
    """
    document = _mapping(payload, "document")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolMapError("The protocol map schema version is not supported")
    if document.get("contract") != PROTOCOL_MAP_CONTRACT:
        raise ProtocolMapError("The protocol map contract name is not recognized")
    version = document.get("contract_version")
    if not isinstance(version, str) or _SEMVER.fullmatch(version) is None:
        raise ProtocolMapError("The protocol map contract version must be semver")
    categories = tuple(_string_list(document.get("categories"), "categories"))
    if categories != CATEGORIES:
        raise ProtocolMapError("The protocol map categories are not canonical")

    transports = _mapping(document.get("transports"), "transports")
    grpc_transport = _mapping(transports.get("grpc"), "transports")
    status_categories: dict[str, str] = {}
    for status, category in _mapping(
        grpc_transport.get("status_categories"), "transports"
    ).items():
        status_categories[_grpc_status(status)] = _category(category)
    graphql_transport = _mapping(transports.get("graphql"), "transports")
    uncoded_category = _category(graphql_transport.get("uncoded_category"))

    codes = _parse_codes(document.get("codes"))
    operations = _parse_operations(document.get("operations"))
    outcomes = _parse_outcomes(document.get("outcomes"), codes, operations)
    rest_only = _parse_rest_only(document.get("rest_only_codes"))
    unmapped = _parse_unmapped_operations(document.get("unmapped_operations"))
    divergences = _parse_divergences(document.get("known_divergences"))
    return ProtocolMap(
        contract_version=version,
        operations=operations,
        outcomes=outcomes,
        codes=MappingProxyType(codes),
        rest_only_codes=MappingProxyType(rest_only),
        unmapped_operations=MappingProxyType(unmapped),
        divergences=divergences,
        grpc_status_categories=MappingProxyType(status_categories),
        graphql_uncoded_category=uncoded_category,
    )


def discover_transport_surface(
    *,
    service_root: Union[Path, str, None] = None,
    openapi_path: Union[Path, str] = DEFAULT_CONTRACT_PATH,
    graphql_sdl_path: Union[Path, str] = DEFAULT_GRAPHQL_SDL_PATH,
) -> TransportSurface:
    """Read every transport's codes, operations and fields without serving.

    Args:
        service_root: Directory holding the service source files.
        openapi_path: Committed OpenAPI contract.
        graphql_sdl_path: Committed GraphQL SDL contract.

    Returns:
        The discovered transport surface.

    Raises:
        ProtocolMapError: If a source or contract file cannot be read.
    """
    from .auth import ROUTE_POLICIES
    from .graphql_schema import GRAPHQL_FIELD_SCOPES
    from .grpc_policy import GRPC_FAILURES, GRPC_RPC_ROUTES
    from .proto.generated import openmed_pb2

    root = Path(service_root) if service_root is not None else SERVICE_SOURCE_ROOT
    rest_codes = frozenset(discover_service_error_categories(root))
    stream_codes = frozenset(_dict_codes(root / _STREAM_SOURCE))
    graphql_codes: set[str] = set()
    for path in sorted(root.glob(_GRAPHQL_SOURCE_GLOB)):
        graphql_codes.update(_dict_codes(path))

    grpc_failures = {
        reason: (failure.status.name, failure.category)
        for reason, failure in GRPC_FAILURES.items()
    }
    rest_route_scopes = {
        (method, path): tuple(policy.scopes)
        for (method, path), policy in ROUTE_POLICIES.items()
    }
    graphql_field_scopes = {
        name: tuple(scopes) for name, scopes in GRAPHQL_FIELD_SCOPES.items()
    }
    grpc_routes = {
        method: (route[0], route[1]) for method, route in GRPC_RPC_ROUTES.items()
    }

    rest_request_fields = _openapi_request_fields(Path(openapi_path))
    graphql_arguments, graphql_inputs, graphql_responses = _graphql_paths(
        Path(graphql_sdl_path)
    )
    grpc_requests: dict[str, frozenset[str]] = {}
    grpc_responses: dict[str, frozenset[str]] = {}
    service = openmed_pb2.DESCRIPTOR.services_by_name["OpenMedService"]
    for method in service.methods:
        name = f"/{service.full_name}/{method.name}"
        grpc_requests[name] = frozenset(_proto_paths(method.input_type))
        grpc_responses[name] = frozenset(_proto_paths(method.output_type))

    return TransportSurface(
        rest_codes=rest_codes,
        rest_stream_event_codes=stream_codes,
        graphql_codes=frozenset(graphql_codes),
        grpc_failures=MappingProxyType(grpc_failures),
        rest_route_scopes=MappingProxyType(rest_route_scopes),
        graphql_field_scopes=MappingProxyType(graphql_field_scopes),
        grpc_routes=MappingProxyType(grpc_routes),
        rest_request_fields=MappingProxyType(rest_request_fields),
        graphql_arguments=MappingProxyType(graphql_arguments),
        graphql_input_fields=MappingProxyType(graphql_inputs),
        graphql_response_paths=MappingProxyType(graphql_responses),
        grpc_request_paths=MappingProxyType(grpc_requests),
        grpc_response_paths=MappingProxyType(grpc_responses),
    )


def check_protocol_map(
    protocol_map: Union[ProtocolMap, Path, str, None] = None,
    *,
    surface: Optional[TransportSurface] = None,
) -> ProtocolDriftReport:
    """Compare the protocol map with what each transport emits and accepts.

    Args:
        protocol_map: A parsed map, a path to one, or ``None`` for the
            committed ``docs/api/protocol-map.json``.
        surface: The transport surface; discovered from source when omitted.

    Returns:
        A value-free report. It is clean only when every emitted code is
        mapped, no mapped code is stale, gRPC statuses match the failure
        table, every category difference is a declared divergence, and every
        operation, scope and request field agrees with the committed
        contracts.

    Raises:
        ProtocolMapError: If the map document itself is invalid.
    """
    if protocol_map is None:
        parsed = load_protocol_map()
    elif isinstance(protocol_map, ProtocolMap):
        parsed = protocol_map
    else:
        parsed = load_protocol_map(protocol_map)
    discovered = surface if surface is not None else discover_transport_surface()

    issues: list[ProtocolDriftIssue] = []
    issues.extend(_check_codes(parsed, discovered))
    issues.extend(_check_outcomes(parsed))
    issues.extend(_check_divergences(parsed))
    issues.extend(_check_operations(parsed, discovered))
    counts = {
        "operations": len(parsed.operations),
        "outcomes": len(parsed.outcomes),
        "rest_codes": len(parsed.codes["rest"]),
        "rest_only_codes": sum(len(codes) for codes in parsed.rest_only_codes.values()),
        "graphql_codes": len(parsed.codes["graphql"]),
        "grpc_codes": len(parsed.codes["grpc"]),
        "known_divergences": len(parsed.divergences),
    }
    return ProtocolDriftReport(
        contract_version=parsed.contract_version,
        issues=tuple(issues),
        counts=counts,
    )


def assert_protocol_map(
    protocol_map: Union[ProtocolMap, Path, str, None] = None,
    *,
    surface: Optional[TransportSurface] = None,
) -> ProtocolDriftReport:
    """Run :func:`check_protocol_map` and raise when drift is found.

    Raises:
        ProtocolDriftError: If the report has any issue.
        ProtocolMapError: If the map document itself is invalid.
    """
    return check_protocol_map(protocol_map, surface=surface).assert_clean()


def _check_codes(
    protocol_map: ProtocolMap, surface: TransportSurface
) -> list[ProtocolDriftIssue]:
    issues: list[ProtocolDriftIssue] = []
    rest_mapped = set(protocol_map.codes["rest"])
    rest_only = {
        code for codes in protocol_map.rest_only_codes.values() for code in codes
    }
    for code in sorted(rest_mapped & rest_only):
        issues.append(ProtocolDriftIssue("ambiguous_code", "rest", code=code))
    issues.extend(
        _set_drift("rest", surface.rest_codes, rest_mapped | rest_only, "code")
    )

    stream_mapped = {
        variant.code
        for outcome in protocol_map.outcomes
        for variant in outcome.variants.get("rest", ())
        if variant.delivery == "stream_event" and variant.code is not None
    }
    issues.extend(
        _set_drift(
            "rest", surface.rest_stream_event_codes, stream_mapped, "stream_event_code"
        )
    )

    issues.extend(
        _set_drift(
            "graphql",
            surface.graphql_codes,
            set(protocol_map.codes["graphql"]),
            "code",
        )
    )

    grpc_table = protocol_map.codes["grpc"]
    issues.extend(
        _set_drift("grpc", frozenset(surface.grpc_failures), set(grpc_table), "code")
    )
    for reason, (status, category) in sorted(surface.grpc_failures.items()):
        entry = grpc_table.get(reason)
        if entry is None:
            continue
        if entry.status != status:
            issues.append(ProtocolDriftIssue("status_mismatch", "grpc", code=reason))
        if entry.category != category:
            issues.append(ProtocolDriftIssue("category_mismatch", "grpc", code=reason))
    return issues


def _check_outcomes(protocol_map: ProtocolMap) -> list[ProtocolDriftIssue]:
    issues: list[ProtocolDriftIssue] = []
    used: dict[str, set[str]] = {transport: set() for transport in TRANSPORTS}
    for outcome in protocol_map.outcomes:
        for transport in TRANSPORTS:
            variants = outcome.variants.get(transport, ())
            if not variants and transport not in outcome.not_applicable:
                issues.append(
                    ProtocolDriftIssue(
                        "unmapped_outcome", transport, outcome=outcome.id
                    )
                )
            if variants and transport in outcome.not_applicable:
                issues.append(
                    ProtocolDriftIssue(
                        "stale_not_applicable", transport, outcome=outcome.id
                    )
                )
            for variant in variants:
                if variant.code is None:
                    continue
                if variant.transport == "graphql" and variant.envelope == ENVELOPE_REST:
                    used["rest"].add(variant.code)
                else:
                    used[variant.transport].add(variant.code)
    for transport in TRANSPORTS:
        for code in sorted(set(protocol_map.codes[transport]) - used[transport]):
            issues.append(ProtocolDriftIssue("unused_code", transport, code=code))
    return issues


def _check_divergences(protocol_map: ProtocolMap) -> list[ProtocolDriftIssue]:
    issues: list[ProtocolDriftIssue] = []
    declared: set[tuple[str, str]] = set()
    for divergence in protocol_map.divergences:
        if divergence.kind == "category":
            declared.update(
                (divergence.transport, outcome) for outcome in divergence.outcomes
            )
        else:
            for code in divergence.codes:
                if code not in protocol_map.codes[divergence.transport]:
                    issues.append(
                        ProtocolDriftIssue(
                            "unknown_divergence_code",
                            divergence.transport,
                            code=code,
                        )
                    )

    mismatched: set[tuple[str, str]] = set()
    for outcome in protocol_map.outcomes:
        for transport in TRANSPORTS:
            for variant in outcome.variants.get(transport, ()):
                if protocol_map.category_of(variant) != outcome.category:
                    mismatched.add((transport, outcome.id))
    for transport, outcome_id in sorted(mismatched - declared):
        issues.append(
            ProtocolDriftIssue("undeclared_divergence", transport, outcome=outcome_id)
        )
    for transport, outcome_id in sorted(declared - mismatched):
        issues.append(
            ProtocolDriftIssue("stale_divergence", transport, outcome=outcome_id)
        )
    return issues


def _check_operations(
    protocol_map: ProtocolMap, surface: TransportSurface
) -> list[ProtocolDriftIssue]:
    issues: list[ProtocolDriftIssue] = []
    mapped_grpc: set[str] = set()
    mapped_graphql: set[str] = set()
    for operation in protocol_map.operations:
        rest = operation.bindings.get("rest")
        rest_route: Optional[tuple[str, str]] = None
        if rest is not None:
            rest_route = (str(rest["method"]), str(rest["path"]))
            scopes = surface.rest_route_scopes.get(rest_route)
            if scopes is None:
                issues.append(
                    ProtocolDriftIssue(
                        "stale_operation", "rest", operation=operation.id
                    )
                )
            elif scopes != operation.scopes:
                issues.append(
                    ProtocolDriftIssue("scope_mismatch", "rest", operation=operation.id)
                )
            issues.extend(
                _field_drift(
                    operation,
                    "rest",
                    surface.rest_request_fields.get(rest_route),
                    None,
                )
            )

        graphql = operation.bindings.get("graphql")
        if graphql is not None:
            name = str(graphql["field"])
            mapped_graphql.add(name)
            scopes = surface.graphql_field_scopes.get(name)
            if scopes is None or name not in surface.graphql_input_fields:
                issues.append(
                    ProtocolDriftIssue(
                        "stale_operation", "graphql", operation=operation.id
                    )
                )
            elif scopes != operation.scopes:
                issues.append(
                    ProtocolDriftIssue(
                        "scope_mismatch", "graphql", operation=operation.id
                    )
                )
            arguments = surface.graphql_arguments.get(name, {})
            if arguments.get(str(graphql["argument"])) != graphql["input_type"]:
                issues.append(
                    ProtocolDriftIssue(
                        "argument_mismatch", "graphql", operation=operation.id
                    )
                )
            issues.extend(
                _field_drift(
                    operation,
                    "graphql",
                    surface.graphql_input_fields.get(name),
                    surface.graphql_response_paths.get(name),
                )
            )

        grpc_binding = operation.bindings.get("grpc")
        if grpc_binding is not None:
            method = str(grpc_binding["method"])
            mapped_grpc.add(method)
            mirrored = surface.grpc_routes.get(method)
            if mirrored is None or method not in surface.grpc_request_paths:
                issues.append(
                    ProtocolDriftIssue(
                        "stale_operation", "grpc", operation=operation.id
                    )
                )
            elif rest_route is not None and mirrored != rest_route:
                issues.append(
                    ProtocolDriftIssue("route_mismatch", "grpc", operation=operation.id)
                )
            issues.extend(
                _field_drift(
                    operation,
                    "grpc",
                    surface.grpc_request_paths.get(method),
                    surface.grpc_response_paths.get(method),
                )
            )

    unmapped_graphql = set(protocol_map.unmapped_operations.get("graphql", {}))
    for name in sorted(set(surface.graphql_field_scopes) - mapped_graphql):
        if name not in unmapped_graphql:
            issues.append(
                ProtocolDriftIssue("unmapped_operation", "graphql", operation=name)
            )
    for name in sorted(unmapped_graphql - set(surface.graphql_field_scopes)):
        issues.append(ProtocolDriftIssue("stale_operation", "graphql", operation=name))
    unmapped_grpc = set(protocol_map.unmapped_operations.get("grpc", {}))
    for method in sorted(set(surface.grpc_request_paths) - mapped_grpc):
        if method not in unmapped_grpc:
            issues.append(
                ProtocolDriftIssue("unmapped_operation", "grpc", operation=method)
            )
    return issues


def _field_drift(
    operation: ProtocolOperation,
    transport: str,
    request_paths: Optional[frozenset[str]],
    response_paths: Optional[frozenset[str]],
) -> list[ProtocolDriftIssue]:
    issues: list[ProtocolDriftIssue] = []
    if request_paths is not None:
        mapped = {
            path
            for mapping in operation.request_fields
            if (path := mapping.paths.get(transport)) is not None
        }
        for path in sorted(request_paths - mapped):
            issues.append(
                ProtocolDriftIssue(
                    "unmapped_field", transport, operation=operation.id, field=path
                )
            )
        for path in sorted(mapped - request_paths):
            issues.append(
                ProtocolDriftIssue(
                    "stale_field", transport, operation=operation.id, field=path
                )
            )
    if response_paths is not None:
        for mapping in operation.response_fields:
            path = mapping.paths.get(transport)
            if path is not None and path not in response_paths:
                issues.append(
                    ProtocolDriftIssue(
                        "stale_field", transport, operation=operation.id, field=path
                    )
                )
    return issues


def _set_drift(
    transport: str,
    emitted: Iterable[str],
    mapped: Iterable[str],
    kind: str,
) -> list[ProtocolDriftIssue]:
    emitted_set = set(emitted)
    mapped_set = set(mapped)
    issues: list[ProtocolDriftIssue] = []
    for code in sorted(emitted_set - mapped_set):
        if _IDENTIFIER.fullmatch(code) is None:
            issues.append(ProtocolDriftIssue(f"invalid_{kind}", transport))
            continue
        issues.append(ProtocolDriftIssue(f"unmapped_{kind}", transport, code=code))
    for code in sorted(mapped_set - emitted_set):
        issues.append(ProtocolDriftIssue(f"stale_{kind}", transport, code=code))
    return issues


def _parse_codes(raw: Any) -> dict[str, Mapping[str, CodeEntry]]:
    table = _mapping(raw, "codes")
    if set(table) != set(TRANSPORTS):
        raise ProtocolMapError("The protocol map must declare codes for each transport")
    codes: dict[str, Mapping[str, CodeEntry]] = {}
    for transport in TRANSPORTS:
        entries: dict[str, CodeEntry] = {}
        for code, value in _mapping(table[transport], "codes").items():
            entry = _mapping(value, "codes")
            entries[_identifier(code)] = CodeEntry(
                status=_status(transport, entry.get("status")),
                category=_category(entry.get("category")),
            )
        codes[transport] = MappingProxyType(entries)
    return codes


def _parse_operations(raw: Any) -> tuple[ProtocolOperation, ...]:
    operations: list[ProtocolOperation] = []
    seen: set[str] = set()
    for item in _sequence(raw, "operations"):
        entry = _mapping(item, "operations")
        operation_id = _operation_id(entry.get("id"))
        if operation_id in seen:
            raise ProtocolMapError("The protocol map repeats an operation")
        seen.add(operation_id)
        bindings: dict[str, Mapping[str, Any]] = {}
        unsupported: dict[str, str] = {}
        for transport in TRANSPORTS:
            binding = _mapping(entry.get(transport), "operations")
            if binding.get("supported") is False:
                unsupported[transport] = _text(binding.get("reason"))
                continue
            _validate_binding(transport, binding)
            bindings[transport] = MappingProxyType(dict(binding))
        if not bindings:
            raise ProtocolMapError("A protocol operation needs a supporting transport")
        operations.append(
            ProtocolOperation(
                id=operation_id,
                scopes=tuple(_string_list(entry.get("scopes"), "operations")),
                bindings=MappingProxyType(bindings),
                unsupported=MappingProxyType(unsupported),
                request_fields=_parse_fields(entry.get("request_fields"), bindings),
                response_fields=_parse_fields(
                    entry.get("response_fields"), bindings, response=True
                ),
            )
        )
    if not operations:
        raise ProtocolMapError("The protocol map declares no operations")
    return tuple(operations)


def _validate_binding(transport: str, binding: Mapping[str, Any]) -> None:
    required = {
        "rest": ("method", "path", "framing"),
        "graphql": ("field", "argument", "input_type", "framing"),
        "grpc": ("method", "framing"),
    }[transport]
    for key in required:
        if not isinstance(binding.get(key), str) or not binding[key]:
            raise ProtocolMapError("A protocol operation binding is incomplete")


def _parse_fields(
    raw: Any,
    bindings: Mapping[str, Mapping[str, Any]],
    *,
    response: bool = False,
) -> tuple[FieldMapping, ...]:
    fields: list[FieldMapping] = []
    for item in _sequence(raw, "fields"):
        entry = _mapping(item, "fields")
        name = _field_path(entry.get("field"))
        paths: dict[str, Optional[str]] = {}
        for transport in TRANSPORTS:
            if transport not in entry:
                if transport in bindings:
                    raise ProtocolMapError("A protocol field omits a transport path")
                continue
            if transport not in bindings:
                raise ProtocolMapError(
                    "A protocol field names an unsupported transport"
                )
            value = entry[transport]
            paths[transport] = None if value is None else _field_path(value)
        normalize = entry.get("normalize")
        if response:
            if normalize not in NORMALIZERS:
                raise ProtocolMapError("A protocol response field needs a normalizer")
        elif normalize is not None:
            raise ProtocolMapError("Protocol request fields take no normalizer")
        fields.append(
            FieldMapping(
                field=name,
                paths=MappingProxyType(paths),
                normalize=normalize,
            )
        )
    return tuple(fields)


def _parse_outcomes(
    raw: Any,
    codes: Mapping[str, Mapping[str, CodeEntry]],
    operations: Sequence[ProtocolOperation],
) -> tuple[ProtocolOutcome, ...]:
    operation_ids = {operation.id for operation in operations}
    outcomes: list[ProtocolOutcome] = []
    seen: set[str] = set()
    for item in _sequence(raw, "outcomes"):
        entry = _mapping(item, "outcomes")
        outcome_id = _operation_id(entry.get("id"))
        if outcome_id in seen:
            raise ProtocolMapError("The protocol map repeats an outcome")
        seen.add(outcome_id)
        scoped = tuple(_string_list(entry.get("operations", []), "outcomes"))
        if any(operation not in operation_ids for operation in scoped):
            raise ProtocolMapError("A protocol outcome names an unknown operation")
        variants = {
            transport: tuple(
                _parse_variant(transport, value, codes)
                for value in _sequence(entry.get(transport), "outcomes")
            )
            for transport in TRANSPORTS
        }
        not_applicable = {
            _transport(transport): _text(reason)
            for transport, reason in _mapping(
                entry.get("not_applicable", {}), "outcomes"
            ).items()
        }
        outcomes.append(
            ProtocolOutcome(
                id=outcome_id,
                category=_category(entry.get("category")),
                operations=scoped,
                variants=MappingProxyType(variants),
                not_applicable=MappingProxyType(not_applicable),
            )
        )
    if not outcomes:
        raise ProtocolMapError("The protocol map declares no outcomes")
    return tuple(outcomes)


def _parse_variant(
    transport: str,
    raw: Any,
    codes: Mapping[str, Mapping[str, CodeEntry]],
) -> OutcomeVariant:
    entry = _mapping(raw, "outcomes")
    origin = entry.get("origin", "server")
    if origin not in ORIGINS:
        raise ProtocolMapError("A protocol outcome variant has an unknown origin")
    if origin == "client" and transport != "grpc":
        return OutcomeVariant(transport, None, origin="client")

    if transport == "grpc":
        reason = entry.get("reason")
        if reason is not None:
            code = _identifier(reason)
            entry_code = codes["grpc"].get(code)
            if entry_code is None:
                raise ProtocolMapError("A protocol outcome references an unmapped code")
            return OutcomeVariant("grpc", entry_code.status, code=code, origin=origin)
        if origin == "server":
            raise ProtocolMapError("A server gRPC variant needs an openmed-reason")
        return OutcomeVariant("grpc", _grpc_status(entry.get("status")), origin=origin)

    if transport == "graphql" and "rest_code" in entry:
        code = _identifier(entry["rest_code"])
        rest_entry = codes["rest"].get(code)
        if rest_entry is None:
            raise ProtocolMapError("A protocol outcome references an unmapped code")
        return OutcomeVariant(
            "graphql", rest_entry.status, code=code, envelope=ENVELOPE_REST
        )

    if "code" not in entry:
        raise ProtocolMapError("A protocol outcome variant needs a code")
    raw_code = entry["code"]
    if transport == "graphql":
        if raw_code is None:
            return OutcomeVariant("graphql", _HTTP_OK, envelope=ENVELOPE_GRAPHQL)
        code = _identifier(raw_code)
        graphql_entry = codes["graphql"].get(code)
        if graphql_entry is None:
            raise ProtocolMapError("A protocol outcome references an unmapped code")
        return OutcomeVariant(
            "graphql", graphql_entry.status, code=code, envelope=ENVELOPE_GRAPHQL
        )

    code = _identifier(raw_code)
    rest_entry = codes["rest"].get(code)
    if rest_entry is None:
        raise ProtocolMapError("A protocol outcome references an unmapped code")
    delivery = entry.get("delivery", "response")
    if delivery not in DELIVERIES:
        raise ProtocolMapError("A protocol outcome variant has an unknown delivery")
    status = _HTTP_OK if delivery == "stream_event" else rest_entry.status
    return OutcomeVariant("rest", status, code=code, delivery=delivery)


def _parse_rest_only(raw: Any) -> dict[str, tuple[str, ...]]:
    families: dict[str, tuple[str, ...]] = {}
    for family, codes in _mapping(raw, "rest_only_codes").items():
        families[_operation_id(family)] = tuple(
            _identifier(code) for code in _string_list(codes, "rest_only_codes")
        )
    return families


def _parse_unmapped_operations(raw: Any) -> dict[str, Mapping[str, str]]:
    unmapped: dict[str, Mapping[str, str]] = {}
    for transport, entries in _mapping(raw, "unmapped_operations").items():
        unmapped[_transport(transport)] = MappingProxyType(
            {
                str(name): _text(reason)
                for name, reason in _mapping(entries, "unmapped_operations").items()
            }
        )
    return unmapped


def _parse_divergences(raw: Any) -> tuple[KnownDivergence, ...]:
    divergences: list[KnownDivergence] = []
    seen: set[str] = set()
    for item in _sequence(raw, "known_divergences"):
        entry = _mapping(item, "known_divergences")
        divergence_id = _operation_id(entry.get("id"))
        if divergence_id in seen:
            raise ProtocolMapError("The protocol map repeats a divergence")
        seen.add(divergence_id)
        kind = entry.get("kind")
        if kind not in DIVERGENCE_KINDS:
            raise ProtocolMapError("A protocol divergence has an unknown kind")
        _text(entry.get("summary"))
        divergences.append(
            KnownDivergence(
                id=divergence_id,
                kind=str(kind),
                transport=_transport(entry.get("transport")),
                outcomes=tuple(
                    _operation_id(value)
                    for value in _string_list(
                        entry.get("outcomes", []), "known_divergences"
                    )
                ),
                codes=tuple(
                    _identifier(value)
                    for value in _string_list(
                        entry.get("codes", []), "known_divergences"
                    )
                ),
                follow_up=_text(entry.get("follow_up")),
            )
        )
    return tuple(divergences)


def _openapi_request_fields(path: Path) -> dict[tuple[str, str], frozenset[str]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolMapError("The OpenAPI contract could not be read") from exc
    schemas = document.get("components", {}).get("schemas", {})
    fields: dict[tuple[str, str], frozenset[str]] = {}
    for route, item in (document.get("paths") or {}).items():
        for method, operation in (item or {}).items():
            if not isinstance(operation, Mapping):
                continue
            schema = (
                operation.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema", {})
            )
            reference = schema.get("$ref") if isinstance(schema, Mapping) else None
            if not isinstance(reference, str):
                continue
            target = schemas.get(reference.rsplit("/", 1)[-1], {})
            properties = target.get("properties")
            if isinstance(properties, Mapping):
                fields[(str(method).upper(), str(route))] = frozenset(properties)
    return fields


def _graphql_paths(
    path: Path,
) -> tuple[
    dict[str, Mapping[str, str]],
    dict[str, frozenset[str]],
    dict[str, frozenset[str]],
]:
    from graphql import (
        GraphQLInputObjectType,
        GraphQLObjectType,
        build_schema,
        get_named_type,
    )

    try:
        sdl = path.read_text(encoding="utf-8")
        schema = build_schema(sdl)
    except Exception as exc:
        raise ProtocolMapError("The GraphQL SDL contract could not be read") from exc
    query = schema.query_type
    arguments: dict[str, Mapping[str, str]] = {}
    inputs: dict[str, frozenset[str]] = {}
    responses: dict[str, frozenset[str]] = {}
    if query is None:
        return arguments, inputs, responses
    for name, query_field in query.fields.items():
        input_fields: set[str] = set()
        argument_types: dict[str, str] = {}
        for argument_name, argument in query_field.args.items():
            named = get_named_type(argument.type)
            argument_types[argument_name] = named.name
            if isinstance(named, GraphQLInputObjectType):
                input_fields.update(named.fields)
        arguments[name] = MappingProxyType(argument_types)
        inputs[name] = frozenset(input_fields)
        named_output = get_named_type(query_field.type)
        if isinstance(named_output, GraphQLObjectType):
            responses[name] = frozenset(_graphql_output_paths(named_output, "", 0))
        else:
            responses[name] = frozenset()
    return arguments, inputs, responses


def _graphql_output_paths(object_type: Any, prefix: str, depth: int) -> Iterable[str]:
    from graphql import (
        GraphQLObjectType,
        get_named_type,
        is_list_type,
        is_non_null_type,
    )

    for name, output_field in object_type.fields.items():
        field_type = output_field.type
        if is_non_null_type(field_type):
            field_type = field_type.of_type
        listed = is_list_type(field_type)
        named = get_named_type(field_type)
        path = f"{prefix}{name}"
        if isinstance(named, GraphQLObjectType) and depth < _MAX_PATH_DEPTH:
            yield from _graphql_output_paths(
                named, f"{path}[]." if listed else f"{path}.", depth + 1
            )
        else:
            yield path


def _proto_paths(descriptor: Any, prefix: str = "", depth: int = 0) -> Iterable[str]:
    for proto_field in descriptor.fields:
        path = f"{prefix}{proto_field.name}"
        message = proto_field.message_type
        if (
            message is not None
            and not message.full_name.startswith(_WELL_KNOWN_PROTO_PACKAGE)
            and depth < _MAX_PATH_DEPTH
        ):
            repeated = _proto_field_is_repeated(proto_field)
            yield from _proto_paths(
                message, f"{path}[]." if repeated else f"{path}.", depth + 1
            )
        else:
            yield path


def _proto_field_is_repeated(proto_field: Any) -> bool:
    is_repeated = getattr(proto_field, "is_repeated", None)
    if isinstance(is_repeated, bool):
        return is_repeated
    return bool(proto_field.label == proto_field.LABEL_REPEATED)


def _dict_codes(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise ProtocolMapError("A service source file could not be parsed") from exc
    codes: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "code"
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                codes.add(value.value)
    return codes


def _mapping(value: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolMapError(f"The protocol map {section} section is malformed")
    return value


def _sequence(value: Any, section: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ProtocolMapError(f"The protocol map {section} section is malformed")
    return value


def _string_list(value: Any, section: str) -> list[str]:
    items = _sequence(value, section)
    if not all(isinstance(item, str) for item in items):
        raise ProtocolMapError(f"The protocol map {section} section is malformed")
    return list(items)


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ProtocolMapError("The protocol map contains an invalid code")
    return value


def _operation_id(value: Any) -> str:
    if not isinstance(value, str) or _OPERATION_ID.fullmatch(value) is None:
        raise ProtocolMapError("The protocol map contains an invalid identifier")
    return value


def _field_path(value: Any) -> str:
    if not isinstance(value, str) or _FIELD_PATH.fullmatch(value) is None:
        raise ProtocolMapError("The protocol map contains an invalid field path")
    return value


def _category(value: Any) -> str:
    if value not in CATEGORIES:
        raise ProtocolMapError("The protocol map contains an unknown category")
    return str(value)


def _transport(value: Any) -> str:
    if value not in TRANSPORTS:
        raise ProtocolMapError("The protocol map contains an unknown transport")
    return str(value)


def _grpc_status(value: Any) -> str:
    if not isinstance(value, str) or _GRPC_STATUS.fullmatch(value) is None:
        raise ProtocolMapError("The protocol map contains an invalid gRPC status")
    return value


def _status(transport: str, value: Any) -> Status:
    if transport == "grpc":
        return _grpc_status(value)
    if type(value) is not int or not 100 <= value <= 599:
        raise ProtocolMapError("The protocol map contains an invalid HTTP status")
    return value


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolMapError("The protocol map is missing an explanation")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the protocol map drift check as a CI-friendly command.

    Returns:
        ``0`` when clean, ``1`` when drift is found, ``2`` when the map or a
        contract cannot be read.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map",
        type=Path,
        default=DEFAULT_PROTOCOL_MAP_PATH,
        help="Protocol map to check against the local transports.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the deterministic machine-readable report.",
    )
    args = parser.parse_args(argv)
    try:
        report = check_protocol_map(args.map)
    except ProtocolMapError:
        if args.json:
            print(
                json.dumps(
                    {"schema_version": SCHEMA_VERSION, "error": "invalid_protocol_map"}
                )
            )
        else:
            print("Service protocol map is invalid")
        return 2
    if args.json:
        print(report.to_json())
    else:
        status = "in sync" if report.is_clean else "drift found"
        print(f"Service protocol map {report.contract_version}: {status}")
        for issue in report.issues:
            print(issue.describe())
    return 0 if report.is_clean else 1


__all__ = [
    "CATEGORIES",
    "DEFAULT_PROTOCOL_MAP_PATH",
    "PROTOCOL_MAP_CONTRACT",
    "SCHEMA_VERSION",
    "TRANSPORTS",
    "CodeEntry",
    "FieldMapping",
    "KnownDivergence",
    "ObservedOutcome",
    "OutcomeVariant",
    "ProtocolDriftError",
    "ProtocolDriftIssue",
    "ProtocolDriftReport",
    "ProtocolMap",
    "ProtocolMapError",
    "ProtocolOperation",
    "ProtocolOutcome",
    "TransportSurface",
    "assert_protocol_map",
    "check_protocol_map",
    "discover_transport_surface",
    "load_protocol_map",
    "main",
    "parse_protocol_map",
]


if __name__ == "__main__":
    raise SystemExit(main())
