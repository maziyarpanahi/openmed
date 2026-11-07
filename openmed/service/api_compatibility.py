"""Offline compatibility checks for the OpenMed service API contract.

The committed OpenAPI document is the service contract.  This module reduces
that document and the live FastAPI schema to the stable parts that clients
depend on: route/method pairs, required request fields, and machine-readable
error categories.  The reduction deliberately ignores descriptions, examples,
defaults, and response payloads so a report cannot echo request content.

The GraphQL and gRPC surfaces are compared against committed schema baselines
in the same report.  GraphQL SDL is parsed with ``graphql-core`` and protobuf
sources are compiled in-process to a descriptor set with ``grpcio-tools``.
Their findings carry only type, member, argument and field-number identifiers.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from openmed.core.errors import ERROR_CODES

SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH = REPOSITORY_ROOT / "docs" / "api" / "openapi.json"
ERROR_CATEGORIES_EXTENSION = "x-openmed-error-categories"

GRAPHQL_SURFACE = "graphql"
GRPC_SURFACE = "grpc"
SCHEMA_SURFACES = (GRAPHQL_SURFACE, GRPC_SURFACE)
SURFACE_CHECKED = "checked"
SURFACE_SKIPPED = "skipped"
SURFACE_SKIP_REASONS = frozenset(
    {
        "graphql_core_unavailable",
        "graphql_runtime_unavailable",
        "grpc_tools_unavailable",
        "protobuf_unavailable",
    }
)
DEFAULT_BASELINE_DIR = REPOSITORY_ROOT / "docs" / "api" / "baselines"
DEFAULT_GRAPHQL_BASELINE_PATH = DEFAULT_BASELINE_DIR / "graphql-schema.graphql"
DEFAULT_PROTO_BASELINE_PATH = DEFAULT_BASELINE_DIR / "openmed-proto-descriptor.json"
DEFAULT_PROTO_PATH = REPOSITORY_ROOT / "openmed" / "service" / "proto" / "openmed.proto"
PROTO_CONTRACT_SCHEMA_VERSION = 1

_HTTP_METHODS = frozenset(
    {"delete", "get", "head", "options", "patch", "post", "put", "trace"}
)
_CATEGORY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

# These are the stable categories emitted by the service error envelope.  The
# list is intentionally value-only metadata: it never contains a request,
# response, credential, or model payload.
_LEGACY_STABLE_ERROR_CATEGORIES = (
    "auth_rate_limited",
    "authentication_required",
    "backpressure",
    "bad_request",
    "bulk_paths_disabled",
    "circuit_breaker_open",
    "forbidden",
    "grounding_invalid_request",
    "inbound_duplicate_mapping",
    "inbound_duplicate_placeholder",
    "inbound_duplicate_response_key",
    "inbound_malformed_placeholder",
    "inbound_request_scope_error",
    "inbound_restoration_error",
    "inbound_restoration_limit",
    "inbound_unknown_placeholder",
    "inbound_unsupported_response",
    "internal_error",
    "invalid_credentials",
    "model_not_served",
    "not_ready",
    "offline_snapshot_unavailable",
    "payload_too_large",
    "outbound_privacy_error",
    "outbound_redaction_failed",
    "outbound_redactor_required",
    "path_exists",
    "path_invalid",
    "path_outside_root",
    "privacy_gateway_blocked",
    "privacy_gateway_error",
    "privacy_gateway_not_configured",
    "privacy_gateway_reidentification_error",
    "privacy_gateway_transport_error",
    "privacy_proxy_error",
    "privacy_proxy_invalid_request",
    "privacy_proxy_not_configured",
    "privacy_proxy_redaction_failed",
    "privacy_proxy_response_rejected",
    "privacy_proxy_transport_failed",
    "rate_limited",
    "replacement_state_error",
    "replacement_state_limit",
    "replacement_state_not_found",
    "restricted_terminology_unconfigured",
    "service_busy",
    "snapshot_invalid",
    "timeout",
    "unsafe_redaction_output",
    "unsupported_content_type",
    "unsupported_request_body",
    "validation_error",
)
STABLE_ERROR_CATEGORIES = tuple(
    sorted(set(_LEGACY_STABLE_ERROR_CATEGORIES) | set(ERROR_CODES.values()))
)
DEFAULT_ERROR_CATEGORIES = STABLE_ERROR_CATEGORIES


class APIContractError(ValueError):
    """Raised when a checked-in or live API contract is not structurally valid."""


class APICompatibilityError(RuntimeError):
    """Raised when a service API change removes a client-visible contract."""

    def __init__(self, report: "CompatibilityReport") -> None:
        self.report = report
        self.issues = report.failing_issues()
        details = [issue.describe() for issue in self.issues]
        details.extend(
            f"{surface.surface} {SURFACE_SKIPPED}: {surface.reason}"
            for surface in report.surfaces
            if not surface.is_checked
        )
        message = "; ".join(details)
        super().__init__(message or "Service API compatibility check failed")


class APISurfaceUnavailable(RuntimeError):
    """Raised when an optional schema toolchain cannot run in this environment.

    Attributes:
        reason: A controlled skip code from :data:`SURFACE_SKIP_REASONS`.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"API compatibility surface unavailable: {reason}")


ServiceAPICompatibilityError = APICompatibilityError
ApiCompatibilityError = APICompatibilityError


@dataclass(frozen=True)
class RequiredField:
    """One required request field and its safe JSON-schema location."""

    relative_path: str
    schema_path: str


@dataclass(frozen=True)
class ContractOperation:
    """Stable contract data for one route operation."""

    method: str
    route: str
    request_schema: str | None = None
    request_body_required: bool = False
    required_fields: tuple[RequiredField, ...] = ()

    @property
    def identifier(self) -> str:
        """Return a stable route identifier suitable for reports."""

        return f"{self.method} {self.route}"


@dataclass(frozen=True)
class ServiceContract:
    """Reduced, deterministic representation of a service API contract."""

    operations: tuple[ContractOperation, ...] = ()
    error_categories: tuple[str, ...] = STABLE_ERROR_CATEGORIES

    def __post_init__(self) -> None:
        operations = tuple(
            sorted(self.operations, key=lambda operation: operation.identifier)
        )
        categories = _normalize_error_categories(self.error_categories)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "error_categories", categories)

    @property
    def operation_keys(self) -> frozenset[tuple[str, str]]:
        """Return the method/path keys covered by this contract."""

        return frozenset(
            (operation.method, operation.route) for operation in self.operations
        )


@dataclass(frozen=True)
class CompatibilityIssue:
    """One safe compatibility finding.

    REST findings use ``route``, ``schema_path`` and ``error_category``.
    GraphQL and gRPC findings set ``surface`` and identify the changed schema
    element by ``type_name`` (a GraphQL type, ``@directive``, or fully
    qualified protobuf message, enum or service), ``member`` (a field, enum
    value, union member, interface, directive location or RPC), ``argument``
    and, for protobuf, ``field_number``.  No default, description, example or
    other schema value is ever recorded.
    """

    change: str
    route: str | None = None
    schema_path: str | None = None
    error_category: str | None = None
    surface: str | None = None
    type_name: str | None = None
    member: str | None = None
    argument: str | None = None
    field_number: int | None = None

    @property
    def kind(self) -> str:
        """Return the issue kind under the descriptive alias used by callers."""

        return self.change

    def to_dict(self) -> dict[str, str | int]:
        """Return only route, schema-path, stable-category and schema identifiers."""

        payload: dict[str, str | int] = {"change": self.change}
        if self.route is not None:
            payload["route"] = self.route
        if self.schema_path is not None:
            payload["schema_path"] = self.schema_path
        if self.error_category is not None:
            payload["error_category"] = self.error_category
        if self.surface is not None:
            payload["surface"] = self.surface
        if self.type_name is not None:
            payload["type_name"] = self.type_name
        if self.member is not None:
            payload["member"] = self.member
        if self.argument is not None:
            payload["argument"] = self.argument
        if self.field_number is not None:
            payload["field_number"] = self.field_number
        return payload

    def describe(self) -> str:
        """Format a safe human-readable finding without arbitrary values."""

        if self.surface is not None:
            location = self.type_name or "schema"
            if self.member is not None:
                location = f"{location}.{self.member}"
            if self.argument is not None:
                location = f"{location}({self.argument})"
            if self.field_number is not None:
                location = f"{location} #{self.field_number}"
            return f"{self.surface} {self.change}: {location}"

        location = self.route or self.schema_path or self.error_category or "contract"
        if self.schema_path is not None and self.route is not None:
            location = f"{self.route} at {self.schema_path}"
        if self.error_category is not None:
            location = f"{location} ({self.error_category})"
        return f"{self.change}: {location}"


@dataclass(frozen=True)
class SurfaceReport:
    """Deterministic compatibility result for one schema surface.

    Attributes:
        surface: ``graphql`` or ``grpc``.
        status: ``checked`` when both sides were compared, or ``skipped`` when
            the local toolchain for this surface is unavailable.
        reason: Controlled skip code when ``status`` is ``skipped``.
        before_definitions: Named definitions in the baseline schema.
        after_definitions: Named definitions in the candidate schema.
        breaking: Changes that can break an existing client.
        added: Additive changes.
        compatible: Non-additive changes that remain client compatible.
    """

    surface: str
    status: str = SURFACE_CHECKED
    reason: str | None = None
    before_definitions: int = 0
    after_definitions: int = 0
    breaking: tuple[CompatibilityIssue, ...] = ()
    added: tuple[CompatibilityIssue, ...] = ()
    compatible: tuple[CompatibilityIssue, ...] = ()

    def __post_init__(self) -> None:
        if self.surface not in SCHEMA_SURFACES:
            raise APIContractError("Unknown API compatibility surface")
        if self.status not in {SURFACE_CHECKED, SURFACE_SKIPPED}:
            raise APIContractError("Unknown API compatibility surface status")
        if self.status == SURFACE_SKIPPED and self.reason not in SURFACE_SKIP_REASONS:
            raise APIContractError("A skipped surface needs a controlled reason")
        object.__setattr__(self, "breaking", _sort_issues(self.breaking))
        object.__setattr__(self, "added", _sort_issues(self.added))
        object.__setattr__(self, "compatible", _sort_issues(self.compatible))

    @property
    def is_checked(self) -> bool:
        """Return whether the baseline and candidate were actually compared."""

        return self.status == SURFACE_CHECKED

    @property
    def is_compatible(self) -> bool:
        """Return whether no breaking change was found on this surface."""

        return not self.breaking

    def to_dict(self) -> dict[str, Any]:
        """Return the identifier-only machine-readable surface report."""

        payload: dict[str, Any] = {"status": self.status}
        if self.reason is not None:
            payload["reason"] = self.reason
        payload["summary"] = {
            "before_definitions": self.before_definitions,
            "after_definitions": self.after_definitions,
            "added": len(self.added),
            "breaking": len(self.breaking),
        }
        payload["added"] = [issue.to_dict() for issue in self.added]
        payload["breaking"] = [issue.to_dict() for issue in self.breaking]
        payload["compatible"] = [issue.to_dict() for issue in self.compatible]
        return payload


@dataclass(frozen=True)
class CompatibilityReport:
    """Deterministic result of comparing two service contracts.

    The top-level counts and issue lists describe the REST/OpenAPI contract.
    ``surfaces`` holds one :class:`SurfaceReport` per additional schema surface
    that was requested; it is empty for a REST-only comparison.
    """

    before_operations: int
    after_operations: int
    breaking: tuple[CompatibilityIssue, ...] = ()
    added: tuple[CompatibilityIssue, ...] = ()
    compatible: tuple[CompatibilityIssue, ...] = ()
    surfaces: tuple[SurfaceReport, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "breaking", _sort_issues(self.breaking))
        object.__setattr__(self, "added", _sort_issues(self.added))
        object.__setattr__(self, "compatible", _sort_issues(self.compatible))
        surfaces = tuple(sorted(self.surfaces, key=lambda item: item.surface))
        if len({surface.surface for surface in surfaces}) != len(surfaces):
            raise APIContractError("Each API compatibility surface is reported once")
        object.__setattr__(self, "surfaces", surfaces)

    @property
    def is_compatible(self) -> bool:
        """Return whether no breaking contract changes were found."""

        return not self.breaking and all(
            surface.is_compatible for surface in self.surfaces
        )

    @property
    def skipped_surfaces(self) -> tuple[str, ...]:
        """Return requested schema surfaces that could not be compared."""

        return tuple(
            surface.surface for surface in self.surfaces if not surface.is_checked
        )

    @property
    def is_complete(self) -> bool:
        """Return whether every requested surface was compared."""

        return not self.skipped_surfaces

    @property
    def breaking_count(self) -> int:
        """Return the number of breaking changes across all surfaces."""

        return len(self.failing_issues())

    def surface(self, name: str) -> SurfaceReport | None:
        """Return the report for one schema surface, if it was requested."""

        for surface in self.surfaces:
            if surface.surface == name:
                return surface
        return None

    def failing_issues(self) -> tuple[CompatibilityIssue, ...]:
        """Return breaking issues in deterministic order, REST first."""

        issues = list(self.breaking)
        for surface in self.surfaces:
            issues.extend(surface.breaking)
        return tuple(issues)

    def assert_compatible(
        self, *, allow_skipped: bool = False
    ) -> "CompatibilityReport":
        """Raise :class:`APICompatibilityError` when the report fails.

        Args:
            allow_skipped: Accept surfaces skipped because their toolchain is
                unavailable.  By default a skipped surface fails closed.

        Returns:
            This report when it passes.

        Raises:
            APICompatibilityError: On any breaking change, or on a skipped
                surface unless ``allow_skipped`` is set.
        """

        if not self.is_compatible or (self.skipped_surfaces and not allow_skipped):
            raise APICompatibilityError(self)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return the PHI-free machine-readable gate report."""

        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "summary": {
                "before_operations": self.before_operations,
                "after_operations": self.after_operations,
                "added": len(self.added),
                "breaking": len(self.breaking),
            },
            "added": [issue.to_dict() for issue in self.added],
            "breaking": [issue.to_dict() for issue in self.breaking],
            "compatible": [issue.to_dict() for issue in self.compatible],
        }
        if self.surfaces:
            summary = payload["summary"]
            summary["total_added"] = len(self.added) + sum(
                len(surface.added) for surface in self.surfaces
            )
            summary["total_breaking"] = self.breaking_count
            summary["skipped_surfaces"] = list(self.skipped_surfaces)
            payload["surfaces"] = {
                surface.surface: surface.to_dict() for surface in self.surfaces
            }
        return payload

    def to_json(self) -> str:
        """Return stable JSON suitable for a CI artifact."""

        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def load_contract(
    path: Path | str = DEFAULT_CONTRACT_PATH,
    *,
    error_categories: Iterable[str] | None = None,
) -> ServiceContract:
    """Load the checked-in OpenAPI contract without importing the service."""

    contract_path = Path(path)
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise APIContractError("The checked-in API contract could not be read") from exc
    return normalize_contract(payload, error_categories=error_categories)


def normalize_contract(
    payload: Mapping[str, Any] | ServiceContract,
    *,
    error_categories: Iterable[str] | None = None,
) -> ServiceContract:
    """Reduce an OpenAPI mapping to the compatibility contract."""

    if isinstance(payload, ServiceContract):
        if error_categories is None:
            return payload
        return ServiceContract(
            payload.operations, _normalize_error_categories(error_categories)
        )

    if not isinstance(payload, Mapping):
        raise APIContractError("The API contract must be a JSON object")

    paths = payload.get("paths")
    if not isinstance(paths, Mapping):
        raise APIContractError("The API contract paths value must be an object")

    operations: list[ContractOperation] = []
    for route in sorted(paths):
        path_item = paths[route]
        if not isinstance(route, str) or not route.startswith("/"):
            raise APIContractError("The API contract contains an invalid route name")
        if not isinstance(path_item, Mapping):
            raise APIContractError("The API contract contains an invalid route entry")
        for raw_method in sorted(path_item):
            method = str(raw_method).lower()
            if method not in _HTTP_METHODS:
                continue
            operation = path_item[raw_method]
            if not isinstance(operation, Mapping):
                raise APIContractError("The API contract contains an invalid operation")
            operations.append(
                _normalize_operation(
                    payload,
                    route,
                    method,
                    operation,
                )
            )

    extension = payload.get(ERROR_CATEGORIES_EXTENSION)
    if extension is None:
        info = payload.get("info")
        if isinstance(info, Mapping):
            extension = info.get(ERROR_CATEGORIES_EXTENSION)
    categories = (
        _normalize_error_categories(error_categories)
        if error_categories is not None
        else _normalize_error_categories(extension)
    )
    return ServiceContract(tuple(operations), categories)


def build_live_contract(
    app: Any | None = None,
    *,
    error_categories: Iterable[str] | None = None,
) -> ServiceContract:
    """Build a contract from a FastAPI app without making a network call."""

    if app is None:
        from .app import create_app

        app = create_app()
    if isinstance(app, Mapping):
        return normalize_contract(app, error_categories=error_categories)
    openapi = getattr(app, "openapi", None)
    if not callable(openapi):
        raise APIContractError("The live service app does not expose an OpenAPI schema")
    discovered_categories = error_categories
    if discovered_categories is None:
        for attribute in ("service_error_categories", "error_categories"):
            candidate = getattr(app, attribute, None)
            if candidate is not None:
                discovered_categories = candidate
                break
    if discovered_categories is None:
        discovered_categories = discover_service_error_categories()
    return normalize_contract(openapi(), error_categories=discovered_categories)


def discover_service_error_categories(
    service_root: Path | str | None = None,
) -> tuple[str, ...]:
    """Read stable server error codes from local service source files."""

    root = (
        Path(service_root)
        if service_root is not None
        else REPOSITORY_ROOT / "openmed" / "service"
    )
    categories: set[str] = set()
    try:
        paths = sorted(root.rglob("*.py"))
    except OSError as exc:
        raise APIContractError("The service error contract could not be read") from exc

    for path in paths:
        if path.name == Path(__file__).name:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise APIContractError(
                "The service error contract could not be parsed"
            ) from exc
        categories.update(_error_categories_from_ast(tree))

    # The HTTP-exception handler maps all 4xx values to bad_request and all
    # 5xx values to internal_error through a conditional expression, so those
    # two stable categories are not represented by one literal call argument.
    categories.update({"bad_request", "internal_error"})
    # Taxonomy handlers use ``exc.code`` rather than literals so subclasses can
    # retain their public code. Include the canonical registry explicitly.
    categories.update(ERROR_CODES.values())
    return _normalize_error_categories(categories)


def compare_contracts(
    before: Mapping[str, Any] | ServiceContract,
    after: Mapping[str, Any] | ServiceContract,
    *,
    before_error_categories: Iterable[str] | None = None,
    after_error_categories: Iterable[str] | None = None,
) -> CompatibilityReport:
    """Compare two service contracts and classify breaking changes."""

    previous = normalize_contract(before, error_categories=before_error_categories)
    current = normalize_contract(after, error_categories=after_error_categories)
    previous_by_key = {
        (operation.method, operation.route): operation
        for operation in previous.operations
    }
    current_by_key = {
        (operation.method, operation.route): operation
        for operation in current.operations
    }
    breaking: list[CompatibilityIssue] = []
    added: list[CompatibilityIssue] = []
    compatible: list[CompatibilityIssue] = []

    for key in sorted(previous_by_key):
        old_operation = previous_by_key[key]
        new_operation = current_by_key.get(key)
        if new_operation is None:
            breaking.append(
                CompatibilityIssue("operation_removed", old_operation.identifier)
            )
            continue

        if old_operation.request_schema and not new_operation.request_schema:
            breaking.append(
                CompatibilityIssue("request_schema_removed", old_operation.identifier)
            )
        if (
            not old_operation.request_body_required
            and new_operation.request_body_required
        ):
            breaking.append(
                CompatibilityIssue("request_body_required", old_operation.identifier)
            )

        old_fields = {
            field.relative_path: field for field in old_operation.required_fields
        }
        new_fields = {
            field.relative_path: field for field in new_operation.required_fields
        }
        for relative_path in sorted(new_fields.keys() - old_fields.keys()):
            breaking.append(
                CompatibilityIssue(
                    "required_field_added",
                    route=new_operation.identifier,
                    schema_path=new_fields[relative_path].schema_path,
                )
            )
        for relative_path in sorted(old_fields.keys() - new_fields.keys()):
            compatible.append(
                CompatibilityIssue(
                    "required_field_removed",
                    route=old_operation.identifier,
                    schema_path=old_fields[relative_path].schema_path,
                )
            )

    for key in sorted(current_by_key.keys() - previous_by_key.keys()):
        added.append(
            CompatibilityIssue("operation_added", current_by_key[key].identifier)
        )

    previous_categories = set(previous.error_categories)
    current_categories = set(current.error_categories)
    for category in sorted(previous_categories - current_categories):
        breaking.append(
            CompatibilityIssue("error_category_removed", error_category=category)
        )
    for category in sorted(current_categories - previous_categories):
        added.append(
            CompatibilityIssue("error_category_added", error_category=category)
        )

    return CompatibilityReport(
        before_operations=len(previous.operations),
        after_operations=len(current.operations),
        breaking=tuple(breaking),
        added=tuple(added),
        compatible=tuple(compatible),
    )


def check_api_compatibility(
    contract_path: Path | str = DEFAULT_CONTRACT_PATH,
    *,
    app: Any | None = None,
    current_contract: Mapping[str, Any] | ServiceContract | None = None,
    error_categories: Iterable[str] | None = None,
    surfaces: Iterable[str] = (),
    graphql_baseline_path: Path | str = DEFAULT_GRAPHQL_BASELINE_PATH,
    current_graphql_sdl: str | None = None,
    proto_baseline_path: Path | str = DEFAULT_PROTO_BASELINE_PATH,
    proto_path: Path | str = DEFAULT_PROTO_PATH,
    current_proto_contract: Any | None = None,
) -> CompatibilityReport:
    """Compare the checked-in contracts with the live service contracts.

    The REST/OpenAPI comparison always runs.  ``surfaces`` opts into the
    GraphQL and gRPC schema comparisons against their committed baselines.

    Args:
        contract_path: Checked-in OpenAPI contract.
        app: Optional FastAPI app; the default service app is built offline.
        current_contract: Optional candidate OpenAPI mapping or contract.
        error_categories: Optional candidate error categories.
        surfaces: Additional schema surfaces from :data:`SCHEMA_SURFACES`.
        graphql_baseline_path: Committed GraphQL SDL baseline.
        current_graphql_sdl: Candidate SDL; defaults to the live schema.
        proto_baseline_path: Committed protobuf descriptor baseline.
        proto_path: Candidate ``.proto`` source compiled in-process.
        current_proto_contract: Optional candidate descriptor set or contract
            used instead of compiling ``proto_path``.

    Returns:
        A deterministic :class:`CompatibilityReport`.

    Raises:
        APIContractError: If a contract, baseline or schema is invalid.
    """

    requested = _normalize_surfaces(surfaces)
    before = load_contract(contract_path)
    after = (
        normalize_contract(current_contract, error_categories=error_categories)
        if current_contract is not None
        else build_live_contract(app, error_categories=error_categories)
    )
    report = compare_contracts(before, after)
    surface_reports: list[SurfaceReport] = []
    if GRAPHQL_SURFACE in requested:
        surface_reports.append(
            check_graphql_compatibility(
                graphql_baseline_path, current_sdl=current_graphql_sdl
            )
        )
    if GRPC_SURFACE in requested:
        surface_reports.append(
            check_grpc_compatibility(
                proto_baseline_path,
                proto_path=proto_path,
                current_contract=current_proto_contract,
            )
        )
    if not surface_reports:
        return report
    return replace(report, surfaces=tuple(surface_reports))


def assert_api_compatibility(
    contract_path: Path | str = DEFAULT_CONTRACT_PATH,
    *,
    app: Any | None = None,
    current_contract: Mapping[str, Any] | ServiceContract | None = None,
    error_categories: Iterable[str] | None = None,
    surfaces: Iterable[str] = (),
    allow_skipped: bool = False,
    graphql_baseline_path: Path | str = DEFAULT_GRAPHQL_BASELINE_PATH,
    current_graphql_sdl: str | None = None,
    proto_baseline_path: Path | str = DEFAULT_PROTO_BASELINE_PATH,
    proto_path: Path | str = DEFAULT_PROTO_PATH,
    current_proto_contract: Any | None = None,
) -> CompatibilityReport:
    """Run the offline gate and raise if a breaking change is present.

    Accepts the arguments of :func:`check_api_compatibility`; a requested
    surface that was skipped also fails unless ``allow_skipped`` is set.
    """

    return check_api_compatibility(
        contract_path,
        app=app,
        current_contract=current_contract,
        error_categories=error_categories,
        surfaces=surfaces,
        graphql_baseline_path=graphql_baseline_path,
        current_graphql_sdl=current_graphql_sdl,
        proto_baseline_path=proto_baseline_path,
        proto_path=proto_path,
        current_proto_contract=current_proto_contract,
    ).assert_compatible(allow_skipped=allow_skipped)


compare_api_contracts = compare_contracts
run_compatibility_gate = check_api_compatibility


# ---------------------------------------------------------------------------
# GraphQL SDL compatibility
# ---------------------------------------------------------------------------


def build_live_graphql_sdl() -> str:
    """Render the live Strawberry GraphQL schema as SDL without serving it.

    Returns:
        SDL text in the ``scripts/export_graphql_schema.py`` export format.

    Raises:
        ImportError: If the optional GraphQL service runtime is not installed.
    """

    from .graphql_schema import schema

    return f"{str(schema).rstrip()}\n"


def load_graphql_baseline(path: Path | str = DEFAULT_GRAPHQL_BASELINE_PATH) -> str:
    """Read a committed GraphQL SDL baseline.

    Args:
        path: SDL file to read.

    Returns:
        The SDL text.

    Raises:
        APIContractError: If the file cannot be read.
    """

    return _read_schema_text(path)


def compare_graphql_schemas(before: Any, after: Any) -> SurfaceReport:
    """Classify GraphQL schema changes between a baseline and a candidate.

    Removed types, fields, arguments, input fields, enum values, union
    members, interfaces and directives; changed types; tightened input or
    relaxed output nullability; and newly required arguments or input fields
    are breaking.  New definitions and optional inputs are additive.

    Args:
        before: Baseline SDL text or a ``graphql-core`` schema.
        after: Candidate SDL text or a ``graphql-core`` schema.

    Returns:
        A :class:`SurfaceReport`; ``skipped`` with ``graphql_core_unavailable``
        when ``graphql-core`` is not installed.

    Raises:
        APIContractError: If either schema cannot be parsed.
    """

    gql = _import_graphql()
    if gql is None:
        return _skipped_surface(GRAPHQL_SURFACE, "graphql_core_unavailable")
    previous = _build_graphql_schema(gql, before)
    current = _build_graphql_schema(gql, after)
    return _GraphQLComparator(gql).compare(previous, current)


def check_graphql_compatibility(
    baseline_path: Path | str = DEFAULT_GRAPHQL_BASELINE_PATH,
    *,
    current_sdl: str | None = None,
) -> SurfaceReport:
    """Compare the committed GraphQL baseline with the live or given schema.

    Args:
        baseline_path: Committed SDL baseline.
        current_sdl: Candidate SDL; defaults to the live Strawberry schema.

    Returns:
        A :class:`SurfaceReport`, skipped with a controlled reason when
        ``graphql-core`` or the live GraphQL runtime is unavailable.

    Raises:
        APIContractError: If the baseline or candidate is invalid.
    """

    if _import_graphql() is None:
        return _skipped_surface(GRAPHQL_SURFACE, "graphql_core_unavailable")
    baseline = load_graphql_baseline(baseline_path)
    if current_sdl is None:
        try:
            current_sdl = build_live_graphql_sdl()
        except ImportError:
            return _skipped_surface(GRAPHQL_SURFACE, "graphql_runtime_unavailable")
    return compare_graphql_schemas(baseline, current_sdl)


# ---------------------------------------------------------------------------
# Protobuf descriptor compatibility
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtoField:
    """One protobuf field reduced to its wire and generated-code identity."""

    name: str
    number: int
    type: str
    type_name: str | None = None
    cardinality: str = "singular"
    proto3_optional: bool = False
    oneof: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the baseline representation, omitting default attributes."""

        payload: dict[str, Any] = {
            "name": self.name,
            "number": self.number,
            "type": self.type,
        }
        if self.type_name is not None:
            payload["type_name"] = self.type_name
        if self.cardinality != "singular":
            payload["cardinality"] = self.cardinality
        if self.proto3_optional:
            payload["proto3_optional"] = True
        if self.oneof is not None:
            payload["oneof"] = self.oneof
        return payload


@dataclass(frozen=True)
class ProtoMessage:
    """One protobuf message with its fields and reservations.

    ``reserved_numbers`` holds inclusive ``(start, end)`` ranges.
    """

    name: str
    fields: tuple[ProtoField, ...] = ()
    reserved_numbers: tuple[tuple[int, int], ...] = ()
    reserved_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        fields = tuple(sorted(self.fields, key=lambda item: item.number))
        if len({item.number for item in fields}) != len(fields) or len(
            {item.name for item in fields}
        ) != len(fields):
            raise APIContractError("A protobuf message declares a field twice")
        object.__setattr__(self, "fields", fields)
        object.__setattr__(
            self, "reserved_numbers", tuple(sorted(set(self.reserved_numbers)))
        )
        object.__setattr__(
            self, "reserved_names", tuple(sorted(set(self.reserved_names)))
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic baseline representation."""

        payload: dict[str, Any] = {
            "name": self.name,
            "fields": [item.to_dict() for item in self.fields],
        }
        if self.reserved_numbers:
            payload["reserved_numbers"] = [list(item) for item in self.reserved_numbers]
        if self.reserved_names:
            payload["reserved_names"] = list(self.reserved_names)
        return payload


@dataclass(frozen=True)
class ProtoEnum:
    """One protobuf enum with its values and inclusive reserved ranges."""

    name: str
    values: tuple[tuple[str, int], ...] = ()
    reserved_numbers: tuple[tuple[int, int], ...] = ()
    reserved_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = tuple(sorted(self.values, key=lambda item: (item[1], item[0])))
        if len({name for name, _ in values}) != len(values):
            raise APIContractError("A protobuf enum declares a value twice")
        object.__setattr__(self, "values", values)
        object.__setattr__(
            self, "reserved_numbers", tuple(sorted(set(self.reserved_numbers)))
        )
        object.__setattr__(
            self, "reserved_names", tuple(sorted(set(self.reserved_names)))
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic baseline representation."""

        payload: dict[str, Any] = {
            "name": self.name,
            "values": [
                {"name": name, "number": number} for name, number in self.values
            ],
        }
        if self.reserved_numbers:
            payload["reserved_numbers"] = [list(item) for item in self.reserved_numbers]
        if self.reserved_names:
            payload["reserved_names"] = list(self.reserved_names)
        return payload


@dataclass(frozen=True)
class ProtoMethod:
    """One RPC signature."""

    name: str
    request: str
    response: str
    client_streaming: bool = False
    server_streaming: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic baseline representation."""

        return {
            "name": self.name,
            "request": self.request,
            "response": self.response,
            "client_streaming": self.client_streaming,
            "server_streaming": self.server_streaming,
        }


@dataclass(frozen=True)
class ProtoService:
    """One protobuf service and its RPCs."""

    name: str
    methods: tuple[ProtoMethod, ...] = ()

    def __post_init__(self) -> None:
        methods = tuple(sorted(self.methods, key=lambda item: item.name))
        if len({item.name for item in methods}) != len(methods):
            raise APIContractError("A protobuf service declares an RPC twice")
        object.__setattr__(self, "methods", methods)

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic baseline representation."""

        return {"name": self.name, "methods": [item.to_dict() for item in self.methods]}


@dataclass(frozen=True)
class ProtoContract:
    """Reduced, deterministic protobuf descriptor contract.

    Names are fully qualified without the leading dot.  Comments, options,
    default values and JSON names are not part of the contract.
    """

    messages: tuple[ProtoMessage, ...] = ()
    enums: tuple[ProtoEnum, ...] = ()
    services: tuple[ProtoService, ...] = ()

    def __post_init__(self) -> None:
        messages = tuple(sorted(self.messages, key=lambda item: item.name))
        enums = tuple(sorted(self.enums, key=lambda item: item.name))
        services = tuple(sorted(self.services, key=lambda item: item.name))
        names = [item.name for item in (*messages, *enums, *services)]
        if len(set(names)) != len(names):
            raise APIContractError("A protobuf contract declares a name twice")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "enums", enums)
        object.__setattr__(self, "services", services)

    @property
    def definition_count(self) -> int:
        """Return the number of messages, enums and services."""

        return len(self.messages) + len(self.enums) + len(self.services)

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic baseline representation."""

        return {
            "schema_version": PROTO_CONTRACT_SCHEMA_VERSION,
            "surface": GRPC_SURFACE,
            "messages": [item.to_dict() for item in self.messages],
            "enums": [item.to_dict() for item in self.enums],
            "services": [item.to_dict() for item in self.services],
        }

    def to_json(self) -> str:
        """Return stable, reviewable JSON for a committed baseline."""

        return json.dumps(self.to_dict(), indent=2) + "\n"


def normalize_proto_descriptors(descriptors: Any) -> ProtoContract:
    """Reduce protobuf descriptors to a :class:`ProtoContract`.

    Args:
        descriptors: A ``FileDescriptorSet``, serialized descriptor-set bytes,
            one ``FileDescriptorProto``, an iterable of them, a baseline
            mapping, or an existing :class:`ProtoContract`.

    Returns:
        The reduced contract.

    Raises:
        APIContractError: If the descriptors are invalid.
        APISurfaceUnavailable: If bytes are given and ``protobuf`` is missing.
    """

    if isinstance(descriptors, ProtoContract):
        return descriptors
    if isinstance(descriptors, Mapping):
        return _proto_contract_from_mapping(descriptors)
    if isinstance(descriptors, (bytes, bytearray)):
        descriptor_pb2 = _import_descriptor_pb2()
        try:
            descriptors = descriptor_pb2.FileDescriptorSet.FromString(
                bytes(descriptors)
            )
        except Exception:
            raise APIContractError("The protobuf descriptor set is invalid") from None
    if hasattr(descriptors, "message_type"):
        files: Iterable[Any] = (descriptors,)
    elif hasattr(descriptors, "file"):
        files = descriptors.file
    else:
        files = descriptors
    return _proto_contract_from_files(files)


def compile_proto_contract(
    proto_path: Path | str = DEFAULT_PROTO_PATH,
    *,
    include_paths: Iterable[Path | str] = (),
) -> ProtoContract:
    """Compile a ``.proto`` source in-process and reduce its descriptors.

    The source is compiled with ``grpcio-tools`` to a ``FileDescriptorSet`` in
    a private temporary directory; no network or subprocess is used.

    Args:
        proto_path: Source file to compile.
        include_paths: Extra import roots; the source directory and the
            bundled well-known types are always included.

    Returns:
        The reduced contract for the compiled file.

    Raises:
        APISurfaceUnavailable: If ``grpcio-tools`` or ``protobuf`` is missing.
        APIContractError: If the source cannot be read or compiled.
    """

    try:
        import grpc_tools
        from grpc_tools import protoc
    except ImportError:
        raise APISurfaceUnavailable("grpc_tools_unavailable") from None
    descriptor_pb2 = _import_descriptor_pb2()

    source = Path(proto_path).resolve()
    if not source.is_file():
        raise APIContractError("The protobuf contract could not be read")
    well_known = Path(grpc_tools.__file__).resolve().parent / "_proto"
    roots = [source.parent, *(Path(item).resolve() for item in include_paths)]
    with tempfile.TemporaryDirectory(prefix="openmed-proto-contract-") as workdir:
        output = Path(workdir) / "descriptor-set.binpb"
        arguments = [
            "grpc_tools.protoc",
            *(f"-I{root}" for root in roots),
            f"-I{well_known}",
            f"--descriptor_set_out={output}",
            source.name,
        ]
        if protoc.main(arguments) != 0 or not output.is_file():
            raise APIContractError("The protobuf contract could not be compiled")
        payload = output.read_bytes()
    try:
        descriptor_set = descriptor_pb2.FileDescriptorSet.FromString(payload)
    except Exception:
        raise APIContractError("The protobuf descriptor set is invalid") from None
    return normalize_proto_descriptors(descriptor_set)


def load_proto_contract(
    path: Path | str = DEFAULT_PROTO_BASELINE_PATH,
) -> ProtoContract:
    """Load a committed protobuf descriptor baseline.

    Args:
        path: Baseline JSON written by :meth:`ProtoContract.to_json`.

    Returns:
        The validated contract.

    Raises:
        APIContractError: If the baseline cannot be read or is invalid.
    """

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise APIContractError(
            "The protobuf contract baseline could not be read"
        ) from exc
    return _proto_contract_from_mapping(payload)


def compare_proto_contracts(before: Any, after: Any) -> SurfaceReport:
    """Classify protobuf contract changes between a baseline and a candidate.

    Removed, renumbered, renamed or retyped fields; cardinality, presence and
    oneof changes; reuse or removal of reserved numbers and names; removed
    enum values, messages, enums, services and RPCs; and changed RPC
    signatures are breaking.  New definitions, fields and values are additive.

    Args:
        before: Baseline descriptors in any form accepted by
            :func:`normalize_proto_descriptors`.
        after: Candidate descriptors in the same forms.

    Returns:
        A checked :class:`SurfaceReport`.

    Raises:
        APIContractError: If either side is invalid.
    """

    previous = normalize_proto_descriptors(before)
    current = normalize_proto_descriptors(after)
    findings = _Findings(GRPC_SURFACE)
    _compare_proto_messages(findings, previous, current)
    _compare_proto_enums(findings, previous, current)
    _compare_proto_services(findings, previous, current)
    return findings.report(previous.definition_count, current.definition_count)


def check_grpc_compatibility(
    baseline_path: Path | str = DEFAULT_PROTO_BASELINE_PATH,
    *,
    proto_path: Path | str = DEFAULT_PROTO_PATH,
    current_contract: Any | None = None,
) -> SurfaceReport:
    """Compare the committed protobuf baseline with the current proto source.

    Args:
        baseline_path: Committed descriptor baseline JSON.
        proto_path: Candidate ``.proto`` source compiled in-process.
        current_contract: Optional candidate descriptors used instead of
            compiling ``proto_path``.

    Returns:
        A :class:`SurfaceReport`, skipped with ``grpc_tools_unavailable`` or
        ``protobuf_unavailable`` when the local toolchain is missing.

    Raises:
        APIContractError: If the baseline or candidate is invalid.
    """

    baseline = load_proto_contract(baseline_path)
    try:
        current = (
            normalize_proto_descriptors(current_contract)
            if current_contract is not None
            else compile_proto_contract(proto_path)
        )
    except APISurfaceUnavailable as exc:
        return _skipped_surface(GRPC_SURFACE, exc.reason)
    return compare_proto_contracts(baseline, current)


def write_schema_baselines(
    *,
    graphql_baseline_path: Path | str = DEFAULT_GRAPHQL_BASELINE_PATH,
    proto_baseline_path: Path | str = DEFAULT_PROTO_BASELINE_PATH,
    current_graphql_sdl: str | None = None,
    proto_path: Path | str = DEFAULT_PROTO_PATH,
    allow_breaking: bool = False,
) -> tuple[SurfaceReport, ...]:
    """Accept the current GraphQL and protobuf schemas as the new baselines.

    Existing baselines are compared first; the write is refused when that
    comparison finds a breaking change unless ``allow_breaking`` is set, so an
    accepted break is always deliberate and visible in the baseline diff.

    Args:
        graphql_baseline_path: SDL baseline to write.
        proto_baseline_path: Descriptor baseline JSON to write.
        current_graphql_sdl: SDL to accept; defaults to the live schema.
        proto_path: ``.proto`` source to accept.
        allow_breaking: Write even when a breaking change was found.

    Returns:
        The comparisons against the previous baselines that existed.

    Raises:
        APISurfaceUnavailable: If a schema toolchain is unavailable.
        APICompatibilityError: If a breaking change was found and
            ``allow_breaking`` is not set.
        APIContractError: If a schema or existing baseline is invalid.
    """

    gql = _import_graphql()
    if gql is None:
        raise APISurfaceUnavailable("graphql_core_unavailable")
    if current_graphql_sdl is None:
        try:
            current_graphql_sdl = build_live_graphql_sdl()
        except ImportError:
            raise APISurfaceUnavailable("graphql_runtime_unavailable") from None
    sdl = f"{current_graphql_sdl.rstrip()}\n"
    _build_graphql_schema(gql, sdl)
    proto = compile_proto_contract(proto_path)

    graphql_target = Path(graphql_baseline_path)
    proto_target = Path(proto_baseline_path)
    reports: list[SurfaceReport] = []
    if graphql_target.exists():
        reports.append(
            compare_graphql_schemas(load_graphql_baseline(graphql_target), sdl)
        )
    if proto_target.exists():
        reports.append(
            compare_proto_contracts(load_proto_contract(proto_target), proto)
        )
    result = tuple(reports)
    if not allow_breaking and any(report.breaking for report in result):
        raise APICompatibilityError(
            CompatibilityReport(
                before_operations=0, after_operations=0, surfaces=result
            )
        )
    for target, content in ((graphql_target, sdl), (proto_target, proto.to_json())):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return result


def _normalize_operation(
    document: Mapping[str, Any],
    route: str,
    method: str,
    operation: Mapping[str, Any],
) -> ContractOperation:
    request_body = operation.get("requestBody")
    if request_body is None:
        return ContractOperation(method=method.upper(), route=route)
    if not isinstance(request_body, Mapping):
        raise APIContractError("The API contract contains an invalid request body")

    content = request_body.get("content", {})
    if not isinstance(content, Mapping):
        raise APIContractError("The API contract contains invalid request content")
    schema_entry: Mapping[str, Any] | None = None
    media_type = content.get("application/json")
    if media_type is None and content:
        media_type = content[sorted(content)[0]]
    if isinstance(media_type, Mapping):
        candidate = media_type.get("schema")
        if isinstance(candidate, Mapping):
            schema_entry = candidate
    if schema_entry is None:
        return ContractOperation(
            method=method.upper(),
            route=route,
            request_body_required=bool(request_body.get("required", False)),
        )

    schema_path = _request_schema_path(route, method)
    required_fields = _required_fields(
        document,
        schema_entry,
        schema_path=schema_path,
    )
    ref = schema_entry.get("$ref")
    request_schema = ref if isinstance(ref, str) else None
    return ContractOperation(
        method=method.upper(),
        route=route,
        request_schema=request_schema,
        request_body_required=bool(request_body.get("required", False)),
        required_fields=tuple(required_fields),
    )


def _required_fields(
    document: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    schema_path: str,
    relative_prefix: str = "",
    seen_refs: frozenset[str] = frozenset(),
) -> list[RequiredField]:
    ref = schema.get("$ref")
    if isinstance(ref, str):
        if ref in seen_refs:
            return []
        target = _resolve_pointer(document, ref)
        return _required_fields(
            document,
            target,
            schema_path=ref,
            relative_prefix=relative_prefix,
            seen_refs=seen_refs | {ref},
        )

    required = schema.get("required", ())
    if required is None:
        required = ()
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
        raise APIContractError("The API contract contains invalid required fields")
    properties = schema.get("properties", {})
    if properties is None:
        properties = {}
    if not isinstance(properties, Mapping):
        raise APIContractError("The API contract contains invalid schema properties")

    fields: list[RequiredField] = []
    for name in sorted(required):
        if not isinstance(name, str):
            raise APIContractError("The API contract contains an invalid field name")
        relative = _join_pointer(relative_prefix, name)
        absolute = f"{schema_path}/properties/{_escape_pointer(name)}"
        fields.append(RequiredField(relative, absolute))

    for name in sorted(properties):
        child = properties[name]
        if not isinstance(name, str) or not isinstance(child, Mapping):
            continue
        fields.extend(
            _required_fields(
                document,
                child,
                schema_path=(f"{schema_path}/properties/{_escape_pointer(name)}"),
                relative_prefix=_join_pointer(relative_prefix, name),
                seen_refs=seen_refs,
            )
        )

    variants = schema.get("allOf", ())
    if isinstance(variants, Sequence) and not isinstance(variants, (str, bytes)):
        for index, variant in enumerate(variants):
            if isinstance(variant, Mapping):
                fields.extend(
                    _required_fields(
                        document,
                        variant,
                        schema_path=f"{schema_path}/allOf/{index}",
                        relative_prefix=relative_prefix,
                        seen_refs=seen_refs,
                    )
                )

    for keyword in ("anyOf", "oneOf"):
        variants = schema.get(keyword, ())
        if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
            continue
        variant_fields = [
            _required_fields(
                document,
                variant,
                schema_path=f"{schema_path}/{keyword}/{index}",
                relative_prefix=relative_prefix,
                seen_refs=seen_refs,
            )
            for index, variant in enumerate(variants)
            if isinstance(variant, Mapping)
        ]
        if variant_fields:
            common_paths = set(field.relative_path for field in variant_fields[0])
            for candidate_fields in variant_fields[1:]:
                common_paths.intersection_update(
                    field.relative_path for field in candidate_fields
                )
            fields.extend(
                field
                for field in variant_fields[0]
                if field.relative_path in common_paths
            )

    items = schema.get("items")
    if isinstance(items, Mapping):
        fields.extend(
            _required_fields(
                document,
                items,
                schema_path=f"{schema_path}/items",
                relative_prefix=_join_pointer(relative_prefix, "[]"),
                seen_refs=seen_refs,
            )
        )
    return _deduplicate_fields(fields)


def _resolve_pointer(document: Mapping[str, Any], pointer: str) -> Mapping[str, Any]:
    if not pointer.startswith("#/"):
        raise APIContractError("The API contract contains an unsupported schema path")
    value: Any = document
    for token in pointer[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, Mapping) and token in value:
            value = value[token]
        else:
            raise APIContractError("The API contract references a missing schema")
    if not isinstance(value, Mapping):
        raise APIContractError("The API contract references a non-object schema")
    return value


def _error_categories_from_ast(tree: ast.AST) -> set[str]:
    categories: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Name) and target.id == "error_code"
                for target in targets
            ):
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    categories.add(value.value)

        if not isinstance(node, ast.Call):
            continue
        function = node.func
        function_name = (
            function.attr
            if isinstance(function, ast.Attribute)
            else function.id
            if isinstance(function, ast.Name)
            else None
        )
        if function_name == "_error_response" and len(node.args) > 1:
            value = node.args[1]
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                categories.add(value.value)
        if function_name == "BulkPathError" and node.args:
            value = node.args[0]
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                categories.add(value.value)
        if function_name in {"AuthError", "_error_response"}:
            for keyword in node.keywords:
                if keyword.arg != "code":
                    continue
                value = keyword.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    categories.add(value.value)
    return categories


def _normalize_error_categories(
    categories: Iterable[str] | None,
) -> tuple[str, ...]:
    if categories is None:
        return STABLE_ERROR_CATEGORIES
    if isinstance(categories, (str, bytes)):
        raise APIContractError("Error categories must be a sequence")
    try:
        values = list(categories)
    except TypeError as exc:
        raise APIContractError("Error categories must be a sequence") from exc
    normalized: set[str] = set()
    for category in values:
        if (
            not isinstance(category, str)
            or _CATEGORY_PATTERN.fullmatch(category) is None
        ):
            raise APIContractError(
                "The API contract contains an invalid error category"
            )
        normalized.add(category)
    return tuple(sorted(normalized))


def _request_schema_path(route: str, method: str) -> str:
    return (
        "#/paths/"
        f"{_escape_pointer(route)}/{method}/requestBody/content/"
        "application~1json/schema"
    )


def _escape_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _join_pointer(prefix: str, value: str) -> str:
    return f"{prefix}/{_escape_pointer(value)}"


def _deduplicate_fields(fields: Iterable[RequiredField]) -> list[RequiredField]:
    by_relative: dict[str, RequiredField] = {}
    for field in fields:
        by_relative.setdefault(field.relative_path, field)
    return [by_relative[path] for path in sorted(by_relative)]


def _sort_issues(
    issues: Iterable[CompatibilityIssue],
) -> tuple[CompatibilityIssue, ...]:
    return tuple(
        sorted(
            issues,
            key=lambda issue: (
                issue.change,
                issue.route or "",
                issue.schema_path or "",
                issue.error_category or "",
                issue.surface or "",
                issue.type_name or "",
                issue.member or "",
                issue.argument or "",
                -1 if issue.field_number is None else issue.field_number,
            ),
        )
    )


def _normalize_surfaces(surfaces: Iterable[str]) -> frozenset[str]:
    if isinstance(surfaces, (str, bytes)):
        raise APIContractError("Compatibility surfaces must be a sequence")
    try:
        requested = frozenset(surfaces)
    except TypeError as exc:
        raise APIContractError("Compatibility surfaces must be a sequence") from exc
    if not requested <= set(SCHEMA_SURFACES):
        raise APIContractError("Unknown API compatibility surface")
    return requested


def _skipped_surface(surface: str, reason: str) -> SurfaceReport:
    return SurfaceReport(surface=surface, status=SURFACE_SKIPPED, reason=reason)


def _read_schema_text(path: Path | str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise APIContractError("The GraphQL schema could not be read") from exc


class _Findings:
    """Collect identifier-only findings for one schema surface."""

    def __init__(self, surface: str) -> None:
        self.surface = surface
        self.breaking: list[CompatibilityIssue] = []
        self.added: list[CompatibilityIssue] = []
        self.compatible: list[CompatibilityIssue] = []

    def breaks(self, change: str, **location: Any) -> None:
        self.breaking.append(self._issue(change, location))

    def adds(self, change: str, **location: Any) -> None:
        self.added.append(self._issue(change, location))

    def notes(self, change: str, **location: Any) -> None:
        self.compatible.append(self._issue(change, location))

    def _issue(self, change: str, location: Mapping[str, Any]) -> CompatibilityIssue:
        return CompatibilityIssue(change, surface=self.surface, **location)

    def report(self, before: int, after: int) -> SurfaceReport:
        return SurfaceReport(
            surface=self.surface,
            before_definitions=before,
            after_definitions=after,
            breaking=tuple(self.breaking),
            added=tuple(self.added),
            compatible=tuple(self.compatible),
        )


def _import_graphql() -> Any | None:
    try:
        import graphql
    except ImportError:
        return None
    return graphql


def _build_graphql_schema(gql: Any, value: Any) -> Any:
    if isinstance(value, gql.GraphQLSchema):
        return value
    if not isinstance(value, str):
        raise APIContractError("The GraphQL schema must be SDL text")
    try:
        return gql.build_schema(value)
    except Exception:
        # graphql-core errors quote the offending SDL; never chain them.
        raise APIContractError("The GraphQL schema could not be parsed") from None


class _GraphQLComparator:
    """Structural GraphQL schema diff modelled on graphql-js breaking rules."""

    def __init__(self, gql: Any) -> None:
        self.gql = gql
        self.findings = _Findings(GRAPHQL_SURFACE)

    def compare(self, before: Any, after: Any) -> SurfaceReport:
        old_types = self._named_types(before)
        new_types = self._named_types(after)
        for name in sorted(old_types):
            new_type = new_types.get(name)
            if new_type is None:
                self.findings.breaks("type_removed", type_name=name)
                continue
            self._compare_type(name, old_types[name], new_type)
        for name in sorted(new_types.keys() - old_types.keys()):
            self.findings.adds("type_added", type_name=name)
        self._compare_directives(before, after)
        return self.findings.report(len(old_types), len(new_types))

    def _named_types(self, schema: Any) -> dict[str, Any]:
        gql = self.gql
        return {
            name: named_type
            for name, named_type in schema.type_map.items()
            if not gql.is_introspection_type(named_type)
            and not gql.is_specified_scalar_type(named_type)
        }

    def _kind(self, named_type: Any) -> str:
        gql = self.gql
        for kind, predicate in (
            ("scalar", gql.is_scalar_type),
            ("object", gql.is_object_type),
            ("interface", gql.is_interface_type),
            ("union", gql.is_union_type),
            ("enum", gql.is_enum_type),
            ("input_object", gql.is_input_object_type),
        ):
            if predicate(named_type):
                return kind
        return "unknown"

    def _compare_type(self, name: str, old_type: Any, new_type: Any) -> None:
        kind = self._kind(old_type)
        if kind != self._kind(new_type):
            self.findings.breaks("type_kind_changed", type_name=name)
            return
        if kind in {"object", "interface"}:
            self._compare_members(
                "interface",
                name,
                {item.name for item in old_type.interfaces},
                {item.name for item in new_type.interfaces},
            )
            self._compare_fields(name, old_type.fields, new_type.fields)
        elif kind == "union":
            self._compare_members(
                "union_member",
                name,
                {item.name for item in old_type.types},
                {item.name for item in new_type.types},
            )
        elif kind == "enum":
            self._compare_members(
                "enum_value", name, set(old_type.values), set(new_type.values)
            )
        elif kind == "input_object":
            self._compare_inputs(
                "input_field", name, None, old_type.fields, new_type.fields
            )

    def _compare_members(
        self, prefix: str, type_name: str, old: set[str], new: set[str]
    ) -> None:
        for member in sorted(old - new):
            self.findings.breaks(
                f"{prefix}_removed", type_name=type_name, member=member
            )
        for member in sorted(new - old):
            self.findings.adds(f"{prefix}_added", type_name=type_name, member=member)

    def _compare_fields(
        self,
        type_name: str,
        old_fields: Mapping[str, Any],
        new_fields: Mapping[str, Any],
    ) -> None:
        for name in sorted(old_fields):
            new_field = new_fields.get(name)
            if new_field is None:
                self.findings.breaks("field_removed", type_name=type_name, member=name)
                continue
            old_field = old_fields[name]
            change = self._type_change(old_field.type, new_field.type, output=True)
            if change == "changed":
                self.findings.breaks(
                    "field_type_changed", type_name=type_name, member=name
                )
            elif change == "relaxed":
                self.findings.breaks(
                    "field_nullability_relaxed", type_name=type_name, member=name
                )
            elif change == "tightened":
                self.findings.notes(
                    "field_nullability_tightened", type_name=type_name, member=name
                )
            self._compare_inputs(
                "argument", type_name, name, old_field.args, new_field.args
            )
        for name in sorted(new_fields.keys() - old_fields.keys()):
            self.findings.adds("field_added", type_name=type_name, member=name)

    def _compare_inputs(
        self,
        prefix: str,
        type_name: str,
        member: str | None,
        old_values: Mapping[str, Any],
        new_values: Mapping[str, Any],
    ) -> None:
        def location(name: str) -> dict[str, Any]:
            if prefix == "input_field":
                return {"type_name": type_name, "member": name}
            return {"type_name": type_name, "member": member, "argument": name}

        for name in sorted(old_values):
            new_value = new_values.get(name)
            if new_value is None:
                self.findings.breaks(f"{prefix}_removed", **location(name))
                continue
            old_value = old_values[name]
            became_required = self._is_required(new_value) and not self._is_required(
                old_value
            )
            change = self._type_change(old_value.type, new_value.type, output=False)
            if change == "changed":
                self.findings.breaks(f"{prefix}_type_changed", **location(name))
            elif change == "tightened" and not became_required:
                self.findings.breaks(
                    f"{prefix}_nullability_tightened", **location(name)
                )
            elif change == "relaxed":
                self.findings.notes(f"{prefix}_nullability_relaxed", **location(name))
            if became_required:
                self.findings.breaks(f"{prefix}_became_required", **location(name))
            elif not self._same_default(old_value, new_value):
                self.findings.notes(f"{prefix}_default_changed", **location(name))
        for name in sorted(new_values.keys() - old_values.keys()):
            if self._is_required(new_values[name]):
                self.findings.breaks(f"required_{prefix}_added", **location(name))
            else:
                self.findings.adds(f"{prefix}_added", **location(name))

    def _compare_directives(self, before: Any, after: Any) -> None:
        gql = self.gql
        old = {
            item.name: item
            for item in before.directives
            if not gql.is_specified_directive(item)
        }
        new = {
            item.name: item
            for item in after.directives
            if not gql.is_specified_directive(item)
        }
        for name in sorted(old):
            type_name = f"@{name}"
            new_directive = new.get(name)
            if new_directive is None:
                self.findings.breaks("directive_removed", type_name=type_name)
                continue
            old_directive = old[name]
            self._compare_members(
                "directive_location",
                type_name,
                {item.name for item in old_directive.locations},
                {item.name for item in new_directive.locations},
            )
            if old_directive.is_repeatable and not new_directive.is_repeatable:
                self.findings.breaks(
                    "directive_repeatable_removed", type_name=type_name
                )
            self._compare_inputs(
                "directive_argument",
                type_name,
                None,
                old_directive.args,
                new_directive.args,
            )
        for name in sorted(new.keys() - old.keys()):
            self.findings.adds("directive_added", type_name=f"@{name}")

    def _is_required(self, value: Any) -> bool:
        return self.gql.is_non_null_type(value.type) and (
            value.default_value is self.gql.Undefined
        )

    def _same_default(self, old_value: Any, new_value: Any) -> bool:
        undefined = self.gql.Undefined
        old_default = old_value.default_value
        new_default = new_value.default_value
        if old_default is undefined or new_default is undefined:
            return old_default is new_default
        return bool(old_default == new_default)

    def _type_change(self, old_type: Any, new_type: Any, *, output: bool) -> str | None:
        """Return ``None``, ``changed``, ``tightened`` or ``relaxed``."""

        if str(old_type) == str(new_type):
            return None
        if self._shape(old_type) != self._shape(new_type):
            return "changed"
        if output:
            return "tightened" if self._safe_output(old_type, new_type) else "relaxed"
        return "relaxed" if self._safe_input(old_type, new_type) else "tightened"

    def _shape(self, graphql_type: Any) -> tuple[str, ...]:
        gql = self.gql
        parts: list[str] = []
        current = graphql_type
        while not gql.is_named_type(current):
            if gql.is_list_type(current):
                parts.append("[]")
            current = current.of_type
        parts.append(current.name)
        return tuple(parts)

    def _safe_output(self, old: Any, new: Any) -> bool:
        gql = self.gql
        if gql.is_list_type(old):
            return (
                gql.is_list_type(new) and self._safe_output(old.of_type, new.of_type)
            ) or (gql.is_non_null_type(new) and self._safe_output(old, new.of_type))
        if gql.is_non_null_type(old):
            return gql.is_non_null_type(new) and self._safe_output(
                old.of_type, new.of_type
            )
        return (gql.is_named_type(new) and old.name == new.name) or (
            gql.is_non_null_type(new) and self._safe_output(old, new.of_type)
        )

    def _safe_input(self, old: Any, new: Any) -> bool:
        gql = self.gql
        if gql.is_list_type(old):
            return gql.is_list_type(new) and self._safe_input(old.of_type, new.of_type)
        if gql.is_non_null_type(old):
            return (
                gql.is_non_null_type(new) and self._safe_input(old.of_type, new.of_type)
            ) or (not gql.is_non_null_type(new) and self._safe_input(old.of_type, new))
        return gql.is_named_type(new) and old.name == new.name


_PROTO_TYPES = {
    1: "double",
    2: "float",
    3: "int64",
    4: "uint64",
    5: "int32",
    6: "fixed64",
    7: "fixed32",
    8: "bool",
    9: "string",
    10: "group",
    11: "message",
    12: "bytes",
    13: "uint32",
    14: "enum",
    15: "sfixed32",
    16: "sfixed64",
    17: "sint32",
    18: "sint64",
}
_PROTO_REFERENCE_TYPES = frozenset({"enum", "group", "message"})
_PROTO_CARDINALITIES = {1: "singular", 2: "required", 3: "repeated"}
_PROTO_MAX_FIELD_NUMBER = 536_870_911
_PROTO_INT32_MIN = -(2**31)
_PROTO_INT32_MAX = 2**31 - 1
_PROTO_NAME_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]{0,127}(?:\.[A-Za-z_][A-Za-z0-9_]{0,127}){0,31}"
)
_PROTO_BASELINE_ERROR = "The protobuf contract baseline is invalid"


def _import_descriptor_pb2() -> Any:
    try:
        from google.protobuf import descriptor_pb2
    except ImportError:
        raise APISurfaceUnavailable("protobuf_unavailable") from None
    return descriptor_pb2


def _qualify(scope: str, name: str) -> str:
    return f"{scope}.{name}" if scope else name


def _proto_contract_from_files(files: Iterable[Any]) -> ProtoContract:
    messages: list[ProtoMessage] = []
    enums: list[ProtoEnum] = []
    services: list[ProtoService] = []
    try:
        for descriptor in files:
            package = str(descriptor.package)
            for message in descriptor.message_type:
                _collect_proto_message(package, message, messages, enums)
            for enum in descriptor.enum_type:
                enums.append(_proto_enum(_qualify(package, enum.name), enum))
            for service in descriptor.service:
                services.append(
                    ProtoService(
                        name=_qualify(package, service.name),
                        methods=tuple(
                            ProtoMethod(
                                name=str(method.name),
                                request=str(method.input_type).lstrip("."),
                                response=str(method.output_type).lstrip("."),
                                client_streaming=bool(method.client_streaming),
                                server_streaming=bool(method.server_streaming),
                            )
                            for method in service.method
                        ),
                    )
                )
    except (AttributeError, IndexError, KeyError, TypeError):
        raise APIContractError("The protobuf descriptor set is invalid") from None
    return ProtoContract(tuple(messages), tuple(enums), tuple(services))


def _collect_proto_message(
    scope: str,
    descriptor: Any,
    messages: list[ProtoMessage],
    enums: list[ProtoEnum],
) -> None:
    name = _qualify(scope, descriptor.name)
    oneofs = [str(item.name) for item in descriptor.oneof_decl]
    fields: list[ProtoField] = []
    for field in descriptor.field:
        field_type = _PROTO_TYPES[int(field.type)]
        proto3_optional = bool(field.proto3_optional)
        oneof = None
        if field.HasField("oneof_index") and not proto3_optional:
            oneof = oneofs[field.oneof_index]
        fields.append(
            ProtoField(
                name=str(field.name),
                number=int(field.number),
                type=field_type,
                type_name=(
                    str(field.type_name).lstrip(".")
                    if field_type in _PROTO_REFERENCE_TYPES
                    else None
                ),
                cardinality=_PROTO_CARDINALITIES[int(field.label)],
                proto3_optional=proto3_optional,
                oneof=oneof,
            )
        )
    messages.append(
        ProtoMessage(
            name=name,
            fields=tuple(fields),
            # DescriptorProto reserved ranges have an exclusive end.
            reserved_numbers=tuple(
                (int(item.start), int(item.end) - 1)
                for item in descriptor.reserved_range
            ),
            reserved_names=tuple(str(item) for item in descriptor.reserved_name),
        )
    )
    for nested in descriptor.nested_type:
        _collect_proto_message(name, nested, messages, enums)
    for nested_enum in descriptor.enum_type:
        enums.append(_proto_enum(_qualify(name, nested_enum.name), nested_enum))


def _proto_enum(name: str, descriptor: Any) -> ProtoEnum:
    return ProtoEnum(
        name=name,
        values=tuple((str(item.name), int(item.number)) for item in descriptor.value),
        # EnumDescriptorProto reserved ranges have an inclusive end.
        reserved_numbers=tuple(
            (int(item.start), int(item.end)) for item in descriptor.reserved_range
        ),
        reserved_names=tuple(str(item) for item in descriptor.reserved_name),
    )


def _proto_contract_from_mapping(payload: Any) -> ProtoContract:
    try:
        _require_keys(
            payload,
            {"schema_version", "surface"},
            {"messages", "enums", "services"},
        )
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != PROTO_CONTRACT_SCHEMA_VERSION
            or payload["surface"] != GRPC_SURFACE
        ):
            raise ValueError
        return ProtoContract(
            messages=tuple(
                _proto_message_from_mapping(item)
                for item in _proto_list(payload.get("messages", []))
            ),
            enums=tuple(
                _proto_enum_from_mapping(item)
                for item in _proto_list(payload.get("enums", []))
            ),
            services=tuple(
                _proto_service_from_mapping(item)
                for item in _proto_list(payload.get("services", []))
            ),
        )
    except (KeyError, TypeError, ValueError):
        raise APIContractError(_PROTO_BASELINE_ERROR) from None


def _proto_message_from_mapping(payload: Any) -> ProtoMessage:
    _require_keys(payload, {"name", "fields"}, {"reserved_numbers", "reserved_names"})
    return ProtoMessage(
        name=_proto_name(payload["name"]),
        fields=tuple(
            _proto_field_from_mapping(item) for item in _proto_list(payload["fields"])
        ),
        reserved_numbers=_proto_ranges(
            payload.get("reserved_numbers", []), 1, _PROTO_MAX_FIELD_NUMBER
        ),
        reserved_names=_proto_names(payload.get("reserved_names", [])),
    )


def _proto_field_from_mapping(payload: Any) -> ProtoField:
    _require_keys(
        payload,
        {"name", "number", "type"},
        {"type_name", "cardinality", "proto3_optional", "oneof"},
    )
    field_type = payload["type"]
    if field_type not in _PROTO_TYPES.values():
        raise ValueError
    type_name = payload.get("type_name")
    if (field_type in _PROTO_REFERENCE_TYPES) != (type_name is not None):
        raise ValueError
    cardinality = payload.get("cardinality", "singular")
    if cardinality not in _PROTO_CARDINALITIES.values():
        raise ValueError
    proto3_optional = payload.get("proto3_optional", False)
    oneof = payload.get("oneof")
    if type(proto3_optional) is not bool:
        raise ValueError
    return ProtoField(
        name=_proto_name(payload["name"]),
        number=_proto_int(payload["number"], 1, _PROTO_MAX_FIELD_NUMBER),
        type=field_type,
        type_name=None if type_name is None else _proto_name(type_name),
        cardinality=cardinality,
        proto3_optional=proto3_optional,
        oneof=None if oneof is None else _proto_name(oneof),
    )


def _proto_enum_from_mapping(payload: Any) -> ProtoEnum:
    _require_keys(payload, {"name", "values"}, {"reserved_numbers", "reserved_names"})
    values = []
    for item in _proto_list(payload["values"]):
        _require_keys(item, {"name", "number"})
        values.append(
            (
                _proto_name(item["name"]),
                _proto_int(item["number"], _PROTO_INT32_MIN, _PROTO_INT32_MAX),
            )
        )
    return ProtoEnum(
        name=_proto_name(payload["name"]),
        values=tuple(values),
        reserved_numbers=_proto_ranges(
            payload.get("reserved_numbers", []), _PROTO_INT32_MIN, _PROTO_INT32_MAX
        ),
        reserved_names=_proto_names(payload.get("reserved_names", [])),
    )


def _proto_service_from_mapping(payload: Any) -> ProtoService:
    _require_keys(payload, {"name", "methods"})
    methods = []
    for item in _proto_list(payload["methods"]):
        _require_keys(
            item,
            {"name", "request", "response", "client_streaming", "server_streaming"},
        )
        if (
            type(item["client_streaming"]) is not bool
            or type(item["server_streaming"]) is not bool
        ):
            raise ValueError
        methods.append(
            ProtoMethod(
                name=_proto_name(item["name"]),
                request=_proto_name(item["request"]),
                response=_proto_name(item["response"]),
                client_streaming=item["client_streaming"],
                server_streaming=item["server_streaming"],
            )
        )
    return ProtoService(name=_proto_name(payload["name"]), methods=tuple(methods))


def _require_keys(
    payload: Any, required: set[str], optional: set[str] | None = None
) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError
    keys = set(payload)
    if not required <= keys or not keys <= required | (optional or set()):
        raise ValueError


def _proto_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError
    return value


def _proto_name(value: Any) -> str:
    if not isinstance(value, str) or _PROTO_NAME_PATTERN.fullmatch(value) is None:
        raise ValueError
    return value


def _proto_names(value: Any) -> tuple[str, ...]:
    return tuple(_proto_name(item) for item in _proto_list(value))


def _proto_int(value: Any, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError
    return value


def _proto_ranges(
    value: Any, minimum: int, maximum: int
) -> tuple[tuple[int, int], ...]:
    ranges = []
    for item in _proto_list(value):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError
        start = _proto_int(item[0], minimum, maximum)
        end = _proto_int(item[1], minimum, maximum)
        if start > end:
            raise ValueError
        ranges.append((start, end))
    return tuple(ranges)


def _merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _first_uncovered(
    start: int, end: int, ranges: Iterable[tuple[int, int]]
) -> int | None:
    cursor = start
    for range_start, range_end in _merge_ranges(ranges):
        if range_end < cursor:
            continue
        if range_start > cursor:
            return cursor
        cursor = range_end + 1
        if cursor > end:
            return None
    return cursor if cursor <= end else None


def _in_ranges(number: int, ranges: Iterable[tuple[int, int]]) -> bool:
    return any(start <= number <= end for start, end in ranges)


def _compare_proto_reservations(
    findings: _Findings,
    type_name: str,
    old_numbers: tuple[tuple[int, int], ...],
    new_numbers: tuple[tuple[int, int], ...],
    old_names: tuple[str, ...],
    new_names: tuple[str, ...],
    new_members: Iterable[tuple[str, int]],
) -> None:
    for start, end in old_numbers:
        first = _first_uncovered(start, end, new_numbers)
        if first is not None:
            findings.breaks(
                "reserved_range_removed", type_name=type_name, field_number=first
            )
    for start, end in new_numbers:
        first = _first_uncovered(start, end, old_numbers)
        if first is not None:
            findings.adds(
                "reserved_range_added", type_name=type_name, field_number=first
            )
    for name in sorted(set(old_names) - set(new_names)):
        findings.breaks("reserved_name_removed", type_name=type_name, member=name)
    for name in sorted(set(new_names) - set(old_names)):
        findings.adds("reserved_name_added", type_name=type_name, member=name)
    reserved_names = set(old_names)
    for member, number in new_members:
        if _in_ranges(number, old_numbers):
            findings.breaks(
                "reserved_number_reused",
                type_name=type_name,
                member=member,
                field_number=number,
            )
        if member in reserved_names:
            findings.breaks(
                "reserved_name_reused",
                type_name=type_name,
                member=member,
                field_number=number,
            )


def _compare_proto_messages(
    findings: _Findings, before: ProtoContract, after: ProtoContract
) -> None:
    old_messages = {item.name: item for item in before.messages}
    new_messages = {item.name: item for item in after.messages}
    for name in sorted(old_messages):
        new_message = new_messages.get(name)
        if new_message is None:
            findings.breaks("message_removed", type_name=name)
            continue
        _compare_proto_fields(findings, old_messages[name], new_message)
    for name in sorted(new_messages.keys() - old_messages.keys()):
        findings.adds("message_added", type_name=name)


def _compare_proto_fields(
    findings: _Findings, old: ProtoMessage, new: ProtoMessage
) -> None:
    type_name = old.name
    old_by_number = {item.number: item for item in old.fields}
    new_by_number = {item.number: item for item in new.fields}
    old_by_name = {item.name: item for item in old.fields}
    new_by_name = {item.name: item for item in new.fields}

    for number in sorted(old_by_number):
        old_field = old_by_number[number]
        location = {"type_name": type_name, "member": old_field.name}
        new_field = new_by_number.get(number)
        if new_field is None:
            change = (
                "field_number_changed"
                if old_field.name in new_by_name
                else "field_removed"
            )
            findings.breaks(change, field_number=number, **location)
            continue
        if new_field.name != old_field.name:
            findings.breaks("field_renamed", field_number=number, **location)
        if (new_field.type, new_field.type_name) != (
            old_field.type,
            old_field.type_name,
        ):
            findings.breaks("field_type_changed", field_number=number, **location)
        if new_field.cardinality != old_field.cardinality:
            findings.breaks(
                "field_cardinality_changed", field_number=number, **location
            )
        if new_field.proto3_optional != old_field.proto3_optional:
            findings.breaks("field_presence_changed", field_number=number, **location)
        if new_field.oneof != old_field.oneof:
            findings.breaks("field_oneof_changed", field_number=number, **location)

    for number in sorted(new_by_number.keys() - old_by_number.keys()):
        new_field = new_by_number[number]
        previous = old_by_name.get(new_field.name)
        if previous is None:
            findings.adds(
                "field_added",
                type_name=type_name,
                member=new_field.name,
                field_number=number,
            )
        elif previous.number in new_by_number:
            # The old number survives under another name, so the removal
            # branch above did not report this field's move.
            findings.breaks(
                "field_number_changed",
                type_name=type_name,
                member=new_field.name,
                field_number=previous.number,
            )

    _compare_proto_reservations(
        findings,
        type_name,
        old.reserved_numbers,
        new.reserved_numbers,
        old.reserved_names,
        new.reserved_names,
        ((item.name, item.number) for item in new.fields),
    )


def _compare_proto_enums(
    findings: _Findings, before: ProtoContract, after: ProtoContract
) -> None:
    old_enums = {item.name: item for item in before.enums}
    new_enums = {item.name: item for item in after.enums}
    for name in sorted(old_enums):
        new_enum = new_enums.get(name)
        if new_enum is None:
            findings.breaks("enum_removed", type_name=name)
            continue
        old_values = dict(old_enums[name].values)
        new_values = dict(new_enum.values)
        for value_name in sorted(old_values):
            number = old_values[value_name]
            if value_name not in new_values:
                findings.breaks(
                    "enum_value_removed",
                    type_name=name,
                    member=value_name,
                    field_number=number,
                )
            elif new_values[value_name] != number:
                findings.breaks(
                    "enum_value_number_changed",
                    type_name=name,
                    member=value_name,
                    field_number=number,
                )
        for value_name in sorted(new_values.keys() - old_values.keys()):
            findings.adds(
                "enum_value_added",
                type_name=name,
                member=value_name,
                field_number=new_values[value_name],
            )
        old_enum = old_enums[name]
        _compare_proto_reservations(
            findings,
            name,
            old_enum.reserved_numbers,
            new_enum.reserved_numbers,
            old_enum.reserved_names,
            new_enum.reserved_names,
            new_enum.values,
        )
    for name in sorted(new_enums.keys() - old_enums.keys()):
        findings.adds("enum_added", type_name=name)


def _compare_proto_services(
    findings: _Findings, before: ProtoContract, after: ProtoContract
) -> None:
    old_services = {item.name: item for item in before.services}
    new_services = {item.name: item for item in after.services}
    for name in sorted(old_services):
        new_service = new_services.get(name)
        if new_service is None:
            findings.breaks("service_removed", type_name=name)
            continue
        old_methods = {item.name: item for item in old_services[name].methods}
        new_methods = {item.name: item for item in new_service.methods}
        for method_name in sorted(old_methods):
            old_method = old_methods[method_name]
            new_method = new_methods.get(method_name)
            location = {"type_name": name, "member": method_name}
            if new_method is None:
                findings.breaks("rpc_removed", **location)
                continue
            if new_method.request != old_method.request:
                findings.breaks("rpc_request_type_changed", **location)
            if new_method.response != old_method.response:
                findings.breaks("rpc_response_type_changed", **location)
            if (new_method.client_streaming, new_method.server_streaming) != (
                old_method.client_streaming,
                old_method.server_streaming,
            ):
                findings.breaks("rpc_streaming_changed", **location)
        for method_name in sorted(new_methods.keys() - old_methods.keys()):
            findings.adds("rpc_added", type_name=name, member=method_name)
    for name in sorted(new_services.keys() - old_services.keys()):
        findings.adds("service_added", type_name=name)


EXIT_COMPATIBLE = 0
EXIT_BREAKING = 1
EXIT_INVALID_CONTRACT = 2
EXIT_SURFACE_SKIPPED = 3

_SURFACE_LABELS = {GRAPHQL_SURFACE: "GraphQL schema", GRPC_SURFACE: "gRPC contract"}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline compatibility gate as a small CI-friendly command.

    Exit codes: ``0`` compatible, ``1`` breaking changes, ``2`` invalid
    contract or baseline, ``3`` a requested surface was skipped because its
    local toolchain is unavailable (accepted with ``--allow-skipped``).
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
        help="Checked-in OpenAPI contract to compare against the live app.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the deterministic machine-readable report.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--rest-only",
        action="store_true",
        help="Check only the REST/OpenAPI contract.",
    )
    mode.add_argument(
        "--write-baselines",
        action="store_true",
        help="Accept the current GraphQL and protobuf schemas as the baselines.",
    )
    parser.add_argument(
        "--allow-breaking",
        action="store_true",
        help="With --write-baselines, accept breaking schema changes.",
    )
    parser.add_argument(
        "--allow-skipped",
        action="store_true",
        help="Exit 0 when a surface is skipped because its toolchain is missing.",
    )
    parser.add_argument(
        "--graphql-baseline",
        type=Path,
        default=DEFAULT_GRAPHQL_BASELINE_PATH,
        help="Committed GraphQL SDL baseline.",
    )
    parser.add_argument(
        "--graphql-schema",
        type=Path,
        default=None,
        help="Candidate SDL file; defaults to the live Strawberry schema.",
    )
    parser.add_argument(
        "--proto-baseline",
        type=Path,
        default=DEFAULT_PROTO_BASELINE_PATH,
        help="Committed protobuf descriptor baseline.",
    )
    parser.add_argument(
        "--proto",
        type=Path,
        default=DEFAULT_PROTO_PATH,
        help="Candidate .proto source compiled in-process.",
    )
    args = parser.parse_args(argv)

    try:
        candidate_sdl = (
            _read_schema_text(args.graphql_schema)
            if args.graphql_schema is not None
            else None
        )
        if args.write_baselines:
            return _write_baselines_command(args, candidate_sdl)
        report = check_api_compatibility(
            args.contract,
            surfaces=() if args.rest_only else SCHEMA_SURFACES,
            graphql_baseline_path=args.graphql_baseline,
            current_graphql_sdl=candidate_sdl,
            proto_baseline_path=args.proto_baseline,
            proto_path=args.proto,
        )
    except APIContractError:
        if args.json:
            print(
                json.dumps(
                    {"schema_version": SCHEMA_VERSION, "error": "invalid_contract"}
                )
            )
        else:
            print("Service API compatibility contract is invalid")
        return EXIT_INVALID_CONTRACT

    if args.json:
        print(report.to_json())
    else:
        if not report.is_compatible:
            status = "breaking changes found"
        elif report.skipped_surfaces:
            status = "incomplete (surface skipped)"
        else:
            status = "compatible"
        print(f"Service API contract: {status}")
        for surface in report.surfaces:
            print(f"{_SURFACE_LABELS[surface.surface]}: {_surface_status(surface)}")
        for issue in report.failing_issues():
            print(issue.describe())
    if not report.is_compatible:
        return EXIT_BREAKING
    if report.skipped_surfaces and not args.allow_skipped:
        return EXIT_SURFACE_SKIPPED
    return EXIT_COMPATIBLE


def _surface_status(surface: SurfaceReport) -> str:
    if not surface.is_checked:
        return f"{SURFACE_SKIPPED} ({surface.reason})"
    status = "compatible" if surface.is_compatible else "breaking changes found"
    return f"{status} ({len(surface.added)} added, {len(surface.breaking)} breaking)"


def _write_baselines_command(
    args: argparse.Namespace, candidate_sdl: str | None
) -> int:
    try:
        reports = write_schema_baselines(
            graphql_baseline_path=args.graphql_baseline,
            proto_baseline_path=args.proto_baseline,
            current_graphql_sdl=candidate_sdl,
            proto_path=args.proto,
            allow_breaking=args.allow_breaking,
        )
    except APISurfaceUnavailable as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "error": "surface_unavailable",
                        "reason": exc.reason,
                    }
                )
            )
        else:
            print(f"Schema baselines not written: {exc.reason}")
        return EXIT_SURFACE_SKIPPED
    except APICompatibilityError as exc:
        if args.json:
            print(exc.report.to_json())
        else:
            print("Schema baselines not written: breaking changes found")
            for issue in exc.issues:
                print(issue.describe())
        return EXIT_BREAKING

    breaking = sum(len(report.breaking) for report in reports)
    if args.json:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "baselines_written": list(SCHEMA_SURFACES),
                    "accepted_breaking": breaking,
                },
                sort_keys=True,
            )
        )
    else:
        print(f"Schema baselines written ({breaking} accepted breaking changes)")
    return EXIT_COMPATIBLE


__all__ = [
    "APICompatibilityError",
    "APIContractError",
    "APISurfaceUnavailable",
    "ApiCompatibilityError",
    "CompatibilityIssue",
    "CompatibilityReport",
    "ContractOperation",
    "DEFAULT_CONTRACT_PATH",
    "DEFAULT_ERROR_CATEGORIES",
    "DEFAULT_GRAPHQL_BASELINE_PATH",
    "DEFAULT_PROTO_BASELINE_PATH",
    "DEFAULT_PROTO_PATH",
    "GRAPHQL_SURFACE",
    "GRPC_SURFACE",
    "ProtoContract",
    "ProtoEnum",
    "ProtoField",
    "ProtoMessage",
    "ProtoMethod",
    "ProtoService",
    "RequiredField",
    "SCHEMA_SURFACES",
    "STABLE_ERROR_CATEGORIES",
    "SURFACE_SKIP_REASONS",
    "ServiceAPICompatibilityError",
    "ServiceContract",
    "SurfaceReport",
    "assert_api_compatibility",
    "build_live_contract",
    "build_live_graphql_sdl",
    "check_api_compatibility",
    "check_graphql_compatibility",
    "check_grpc_compatibility",
    "compare_api_contracts",
    "compare_contracts",
    "compare_graphql_schemas",
    "compare_proto_contracts",
    "compile_proto_contract",
    "discover_service_error_categories",
    "load_contract",
    "load_graphql_baseline",
    "load_proto_contract",
    "main",
    "normalize_contract",
    "normalize_proto_descriptors",
    "run_compatibility_gate",
    "write_schema_baselines",
]


if __name__ == "__main__":
    raise SystemExit(main())
