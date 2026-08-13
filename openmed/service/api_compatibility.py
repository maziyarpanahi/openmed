"""Offline compatibility checks for the OpenMed service API contract.

The committed OpenAPI document is the service contract.  This module reduces
that document and the live FastAPI schema to the stable parts that clients
depend on: route/method pairs, required request fields, and machine-readable
error categories.  The reduction deliberately ignores descriptions, examples,
defaults, and response payloads so a report cannot echo request content.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH = REPOSITORY_ROOT / "docs" / "api" / "openapi.json"
ERROR_CATEGORIES_EXTENSION = "x-openmed-error-categories"

_HTTP_METHODS = frozenset(
    {"delete", "get", "head", "options", "patch", "post", "put", "trace"}
)
_CATEGORY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

# These are the stable categories emitted by the service error envelope.  The
# list is intentionally value-only metadata: it never contains a request,
# response, credential, or model payload.
STABLE_ERROR_CATEGORIES = (
    "auth_rate_limited",
    "authentication_required",
    "backpressure",
    "bad_request",
    "circuit_breaker_open",
    "forbidden",
    "internal_error",
    "invalid_credentials",
    "not_ready",
    "privacy_gateway_blocked",
    "privacy_gateway_error",
    "privacy_gateway_not_configured",
    "privacy_gateway_reidentification_error",
    "privacy_gateway_transport_error",
    "rate_limited",
    "service_busy",
    "timeout",
    "validation_error",
)
DEFAULT_ERROR_CATEGORIES = STABLE_ERROR_CATEGORIES


class APIContractError(ValueError):
    """Raised when a checked-in or live API contract is not structurally valid."""


class APICompatibilityError(RuntimeError):
    """Raised when a service API change removes a client-visible contract."""

    def __init__(self, report: "CompatibilityReport") -> None:
        self.report = report
        self.issues = report.breaking
        message = "; ".join(issue.describe() for issue in self.issues)
        super().__init__(message or "Service API compatibility check failed")


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
    """One safe compatibility finding."""

    change: str
    route: str | None = None
    schema_path: str | None = None
    error_category: str | None = None

    @property
    def kind(self) -> str:
        """Return the issue kind under the descriptive alias used by callers."""

        return self.change

    def to_dict(self) -> dict[str, str]:
        """Return only route, schema-path, and stable-category metadata."""

        payload = {"change": self.change}
        if self.route is not None:
            payload["route"] = self.route
        if self.schema_path is not None:
            payload["schema_path"] = self.schema_path
        if self.error_category is not None:
            payload["error_category"] = self.error_category
        return payload

    def describe(self) -> str:
        """Format a safe human-readable finding without arbitrary values."""

        location = self.route or self.schema_path or self.error_category or "contract"
        if self.schema_path is not None and self.route is not None:
            location = f"{self.route} at {self.schema_path}"
        if self.error_category is not None:
            location = f"{location} ({self.error_category})"
        return f"{self.change}: {location}"


@dataclass(frozen=True)
class CompatibilityReport:
    """Deterministic result of comparing two service contracts."""

    before_operations: int
    after_operations: int
    breaking: tuple[CompatibilityIssue, ...] = ()
    added: tuple[CompatibilityIssue, ...] = ()
    compatible: tuple[CompatibilityIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "breaking", _sort_issues(self.breaking))
        object.__setattr__(self, "added", _sort_issues(self.added))
        object.__setattr__(self, "compatible", _sort_issues(self.compatible))

    @property
    def is_compatible(self) -> bool:
        """Return whether no breaking contract changes were found."""

        return not self.breaking

    @property
    def breaking_count(self) -> int:
        """Return the number of breaking changes."""

        return len(self.breaking)

    def failing_issues(self) -> tuple[CompatibilityIssue, ...]:
        """Return breaking issues in deterministic order."""

        return self.breaking

    def assert_compatible(self) -> "CompatibilityReport":
        """Raise :class:`APICompatibilityError` when the report fails."""

        if self.breaking:
            raise APICompatibilityError(self)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return the PHI-free machine-readable gate report."""

        return {
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
) -> CompatibilityReport:
    """Compare the checked-in contract with the live service contract."""

    before = load_contract(contract_path)
    after = (
        normalize_contract(current_contract, error_categories=error_categories)
        if current_contract is not None
        else build_live_contract(app, error_categories=error_categories)
    )
    return compare_contracts(before, after)


def assert_api_compatibility(
    contract_path: Path | str = DEFAULT_CONTRACT_PATH,
    *,
    app: Any | None = None,
    current_contract: Mapping[str, Any] | ServiceContract | None = None,
    error_categories: Iterable[str] | None = None,
) -> CompatibilityReport:
    """Run the offline gate and raise if a breaking change is present."""

    return check_api_compatibility(
        contract_path,
        app=app,
        current_contract=current_contract,
        error_categories=error_categories,
    ).assert_compatible()


compare_api_contracts = compare_contracts
run_compatibility_gate = check_api_compatibility


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
            ),
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline compatibility gate as a small CI-friendly command."""

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
    args = parser.parse_args(argv)

    try:
        report = check_api_compatibility(args.contract)
    except APIContractError:
        if args.json:
            print(
                json.dumps(
                    {"schema_version": SCHEMA_VERSION, "error": "invalid_contract"}
                )
            )
        else:
            print("Service API compatibility contract is invalid")
        return 2

    if args.json:
        print(report.to_json())
    else:
        status = "compatible" if report.is_compatible else "breaking changes found"
        print(f"Service API contract: {status}")
        for issue in report.breaking:
            print(issue.describe())
    return 0 if report.is_compatible else 1


__all__ = [
    "APICompatibilityError",
    "APIContractError",
    "ApiCompatibilityError",
    "CompatibilityIssue",
    "CompatibilityReport",
    "ContractOperation",
    "DEFAULT_CONTRACT_PATH",
    "DEFAULT_ERROR_CATEGORIES",
    "RequiredField",
    "STABLE_ERROR_CATEGORIES",
    "ServiceAPICompatibilityError",
    "ServiceContract",
    "assert_api_compatibility",
    "build_live_contract",
    "check_api_compatibility",
    "compare_api_contracts",
    "compare_contracts",
    "discover_service_error_categories",
    "load_contract",
    "main",
    "normalize_contract",
    "run_compatibility_gate",
]


if __name__ == "__main__":
    raise SystemExit(main())
