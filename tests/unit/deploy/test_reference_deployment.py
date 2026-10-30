"""Tests for the v3.4 reference deployment declaration and its docs page."""

from __future__ import annotations

import ast
import copy
import json
import re
from collections.abc import Callable, Iterator, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
DECLARATION_DIR = ROOT / "deploy" / "reference"
DECLARATION_FILE = DECLARATION_DIR / "v3.4-reference-deployment.json"
SCHEMA_FILE = DECLARATION_DIR / "reference-deployment.schema.json"
DOCS_FILE = ROOT / "docs" / "deploy" / "reference-deployment.md"
JOURNEY_MODULE = ROOT / "openmed" / "deploy" / "journey.py"
MCP_MODULE = ROOT / "openmed" / "mcp" / "server.py"
ROOT_MANIFESTS = ("Dockerfile", "Dockerfile.mcp", "docker-compose.yml")

RouteMatcher = Callable[[str, str], bool]

_ROUTE_PARAMETER = re.compile(r"\{[^/{}]+\}")
_FORBIDDEN_PUBLIC_TEXT = {
    "url_scheme": re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://"),
    "ipv4_address": re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    "ipv6_address": re.compile(r"\[[0-9A-Fa-f:]*::[0-9A-Fa-f:]*\]"),
    "loopback_hostname": re.compile(r"\blocalhost\b", re.IGNORECASE),
    "dns_hostname": re.compile(
        r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\."
        r"(?:com|net|org|io|life|ai|dev|local|internal|lan|svc|cluster)\b",
        re.IGNORECASE,
    ),
    "host_port": re.compile(r"\b[a-z][a-z0-9-]*:\d{2,5}\b", re.IGNORECASE),
    "credential_assignment": re.compile(
        r"(?i)\b(?:password|passwd|secret|token|api[_-]?key|dsn)\s*[:=]\s*\S+"
    ),
    "private_key": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    "credential_env_var": re.compile(r"\bOPENMED_[A-Z_]*(?:PASSWORD|DSN|SECRET)\b"),
    "email_address": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "ssn_like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone_like": re.compile(r"\+?\d{1,3}[ -]\d{3}[ -]\d{3,4}[ -]?\d{0,4}\b"),
    "medical_record_number": re.compile(r"\bMRN\b"),
    "calendar_date": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _declaration() -> dict[str, Any]:
    return _load_json(DECLARATION_FILE)


def _schema() -> dict[str, Any]:
    return _load_json(SCHEMA_FILE)


def _validator() -> jsonschema.Draft202012Validator:
    schema = _schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


# --- reference resolution -------------------------------------------------


def _iter_references(
    node: Any, owner: str = "declaration"
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    if isinstance(node, Mapping):
        current = str(node.get("id", owner))
        references = node.get("references")
        if isinstance(references, Mapping):
            yield current, references
        for key, value in node.items():
            if key != "references":
                yield from _iter_references(value, current)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_references(item, owner)


def _module_path(module: str) -> Path | None:
    base = ROOT.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


@lru_cache(maxsize=None)
def _top_level_names(path: Path) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return frozenset(names)


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError("function_not_found")


@lru_cache(maxsize=None)
def _journey_worker_paths() -> frozenset[str]:
    worker = _function(JOURNEY_MODULE, "_serve_worker")
    handlers = {
        node.name
        for node in ast.walk(worker)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("do_")
    }
    assert handlers == {"do_GET"}
    return frozenset(
        node.value
        for node in ast.walk(worker)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("/")
    )


@lru_cache(maxsize=None)
def _mcp_http_path() -> str:
    factory = _function(MCP_MODULE, "create_mcp_server")
    for argument, default in zip(factory.args.kwonlyargs, factory.args.kw_defaults):
        if argument.arg == "streamable_http_path":
            assert isinstance(default, ast.Constant)
            return str(default.value)
    raise AssertionError("mcp_http_path_not_found")


@lru_cache(maxsize=None)
def _grpc_methods() -> frozenset[str]:
    proto = (ROOT / "openmed" / "service" / "proto" / "openmed.proto").read_text(
        encoding="utf-8"
    )
    methods: set[str] = set()
    for service, body in re.findall(r"service\s+(\w+)\s*\{(.*?)\n\}", proto, re.S):
        methods.update(f"{service}/{rpc}" for rpc in re.findall(r"rpc\s+(\w+)", body))
    return frozenset(methods)


@lru_cache(maxsize=1)
def _rest_matcher() -> RouteMatcher:
    pytest.importorskip("fastapi")
    from starlette.routing import Match

    from openmed.service.app import create_app

    routes = tuple(create_app().router.routes)

    def matches(method: str, path: str) -> bool:
        scope = {
            "type": "http",
            "method": method,
            "path": _ROUTE_PARAMETER.sub("synthetic", path),
            "root_path": "",
            "path_params": {},
            "query_string": b"",
            "headers": [],
        }
        return any(route.matches(scope)[0] == Match.FULL for route in routes)

    return matches


def _route_exists(route: Mapping[str, str], rest_matcher: RouteMatcher) -> bool:
    surface = route["surface"]
    method = route["method"]
    path = route["path"]
    if surface == "rest":
        return rest_matcher(method, path)
    if surface == "journey_worker":
        return method == "GET" and path in _journey_worker_paths()
    if surface == "mcp":
        return method == "POST" and path == _mcp_http_path()
    return False


def _nested_key_exists(document: Any, dotted: str) -> bool:
    current = document
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def _manifest_failures(manifest: Mapping[str, Any]) -> list[str]:
    path = ROOT / manifest["path"]
    if not path.is_file():
        return ["missing_manifest"]
    failures: list[str] = []
    services = manifest.get("compose_services", [])
    volumes = manifest.get("compose_volumes", [])
    if services or volumes:
        compose = yaml.safe_load(path.read_text(encoding="utf-8"))
        failures.extend(
            "missing_compose_service"
            for name in services
            if name not in compose.get("services", {})
        )
        failures.extend(
            "missing_compose_volume"
            for name in volumes
            if name not in compose.get("volumes", {})
        )
    helm_values = manifest.get("helm_values", [])
    if helm_values:
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
        failures.extend(
            "missing_helm_value"
            for key in helm_values
            if not _nested_key_exists(values, key)
        )
    return failures


def _reference_failures(
    declaration: Mapping[str, Any],
    *,
    rest_matcher: RouteMatcher | None = None,
) -> list[tuple[str, str]]:
    """Return ``(reason_code, owner_id)`` pairs for unresolved references."""
    matcher = rest_matcher or _rest_matcher()
    failures: list[tuple[str, str]] = []
    for owner, references in _iter_references(declaration):
        for reference in references.get("modules", []):
            module, _, symbol = reference.partition(":")
            path = _module_path(module)
            if path is None:
                failures.append(("missing_module", owner))
            elif symbol and symbol not in _top_level_names(path):
                failures.append(("missing_symbol", owner))
        for route in references.get("routes", []):
            if not _route_exists(route, matcher):
                failures.append(("missing_route", owner))
        for method in references.get("grpc_methods", []):
            if method not in _grpc_methods():
                failures.append(("missing_grpc_method", owner))
        for manifest in references.get("manifests", []):
            failures.extend((code, owner) for code in _manifest_failures(manifest))
        for doc in references.get("docs", []):
            if not (ROOT / doc).is_file():
                failures.append(("missing_doc", owner))
    return failures


def _structural_failures(declaration: Mapping[str, Any]) -> list[tuple[str, str]]:
    failures: list[tuple[str, str]] = []
    sections = (
        "topologies",
        "services",
        "transports",
        "durable_components",
        "unsupported_deployments",
    )
    for section in sections:
        ids = [item["id"] for item in declaration[section]]
        if len(ids) != len(set(ids)):
            failures.append(("duplicate_id", section))
    service_ids = {item["id"] for item in declaration["services"]}
    topology_ids = {item["id"] for item in declaration["topologies"]}
    references = [
        item["id"]
        for item in declaration["topologies"]
        if item["status"] == "reference"
    ]
    if references != [declaration["reference_topology"]]:
        failures.append(("reference_topology_mismatch", "topologies"))
    for topology in declaration["topologies"]:
        for service in topology["services"]:
            if service not in service_ids:
                failures.append(("unknown_service", topology["id"]))
    for transport in declaration["transports"]:
        for service in transport["services"]:
            if service not in service_ids:
                failures.append(("unknown_service", transport["id"]))
    for component in declaration["durable_components"]:
        for service in component["owner_services"]:
            if service not in service_ids:
                failures.append(("unknown_service", component["id"]))
        for topology in component["topologies"]:
            if topology not in topology_ids:
                failures.append(("unknown_topology", component["id"]))
        for objective in ("recovery_point_objective", "recovery_time_objective"):
            status = component.get(objective, {}).get("status")
            if status not in {"declared", "unsupported"}:
                failures.append(("missing_recovery_objective", component["id"]))
    return failures


# --- docs rendering -------------------------------------------------------


def _code(value: str) -> str:
    return f"`{value}`"


def _codes(values: list[str]) -> str:
    return ", ".join(_code(value) for value in values) if values else "none"


def _duration(seconds: int) -> str:
    if seconds == 0:
        return "0 s"
    if seconds % 3600 == 0:
        return f"{seconds // 3600} h"
    if seconds % 60 == 0:
        return f"{seconds // 60} min"
    return f"{seconds} s"


def _objective(objective: Mapping[str, Any]) -> str:
    if objective["status"] == "declared":
        return f"{_duration(objective['target_seconds'])} ({_code(objective['basis'])})"
    return f"unsupported ({_code(objective['reason_code'])})"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _expected_rows(declaration: Mapping[str, Any]) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {
        "topologies": [],
        "services": [],
        "transports": [],
        "durable_components": [],
        "unsupported_deployments": [],
    }
    for topology in declaration["topologies"]:
        manifests = [m["path"] for m in topology["references"]["manifests"]]
        rows["topologies"].append(
            _row(
                [
                    _code(topology["id"]),
                    topology["status"],
                    _codes(manifests),
                    _codes(topology["services"]),
                ]
            )
        )
    for service in declaration["services"]:
        for mode in service["replica_modes"]:
            rows["services"].append(
                _row(
                    [
                        _code(service["id"]),
                        _code(mode["mode"]),
                        mode["status"],
                        _codes(mode["constraint_codes"]),
                    ]
                )
            )
    for transport in declaration["transports"]:
        posture = transport["security_posture"]
        rows["transports"].append(
            _row(
                [
                    _code(transport["id"]),
                    transport["status"],
                    _code(posture["encryption"]),
                    _code(posture["authentication"]),
                    _code(posture["default_exposure"]),
                    _codes(posture["constraint_codes"]),
                ]
            )
        )
    for component in declaration["durable_components"]:
        rows["durable_components"].append(
            _row(
                [
                    _code(component["id"]),
                    _code(component["storage"]),
                    _code(component["replica_mode"]),
                    _objective(component["recovery_point_objective"]),
                    _objective(component["recovery_time_objective"]),
                ]
            )
        )
    for deployment in declaration["unsupported_deployments"]:
        rows["unsupported_deployments"].append(
            _row(
                [
                    _code(deployment["id"]),
                    _code(deployment["reason_code"]),
                    deployment["summary"],
                ]
            )
        )
    return rows


def _page_table_rows(page: str) -> list[str]:
    return [line for line in page.splitlines() if line.startswith("| `")]


def _public_text_findings(text: str) -> list[str]:
    return sorted(
        code for code, pattern in _FORBIDDEN_PUBLIC_TEXT.items() if pattern.search(text)
    )


# --- schema ---------------------------------------------------------------


def test_declaration_validates_against_its_schema() -> None:
    errors = list(_validator().iter_errors(_declaration()))

    assert errors == []


def test_schema_rejects_unknown_fields_and_bad_statuses() -> None:
    validator = _validator()
    declaration = _declaration()

    extra = copy.deepcopy(declaration)
    extra["services"][0]["undeclared_field"] = True
    bad_status = copy.deepcopy(declaration)
    bad_status["transports"][0]["status"] = "experimental"
    bad_version = copy.deepcopy(declaration)
    bad_version["schema_version"] = "openmed.reference_deployment.v0"

    for candidate in (extra, bad_status, bad_version):
        assert list(validator.iter_errors(candidate)) != []


def test_every_durable_component_declares_both_recovery_objectives() -> None:
    declaration = _declaration()

    assert declaration["durable_components"]
    for component in declaration["durable_components"]:
        for key in ("recovery_point_objective", "recovery_time_objective"):
            objective = component[key]
            assert objective["status"] in {"declared", "unsupported"}
            if objective["status"] == "declared":
                assert objective["target_seconds"] >= 0
                assert objective["basis"]
            else:
                assert objective["reason_code"]


@pytest.mark.parametrize(
    "objective", ["recovery_point_objective", "recovery_time_objective"]
)
def test_missing_recovery_objective_is_rejected(objective: str) -> None:
    validator = _validator()
    removed = _declaration()
    del removed["durable_components"][0][objective]
    blank = _declaration()
    blank["durable_components"][0][objective] = {"status": "declared"}
    unsupported_without_reason = _declaration()
    unsupported_without_reason["durable_components"][0][objective] = {
        "status": "unsupported"
    }

    for candidate in (removed, blank, unsupported_without_reason):
        assert list(validator.iter_errors(candidate)) != []
    assert ("missing_recovery_objective", removed["durable_components"][0]["id"]) in (
        _structural_failures(removed)
    )


def test_declared_objectives_are_labelled_as_unmeasured_targets() -> None:
    declaration = _declaration()

    assert declaration["objective_evidence"] == "unmeasured_targets"


# --- references -----------------------------------------------------------


def test_every_referenced_route_module_and_manifest_exists() -> None:
    declaration = _declaration()

    assert _structural_failures(declaration) == []
    assert _reference_failures(declaration) == []


def test_declaration_references_every_reference_compose_service() -> None:
    declaration = _declaration()
    compose = yaml.safe_load(
        (ROOT / "deploy" / "openmed-compose.yaml").read_text(encoding="utf-8")
    )
    declared: set[str] = set()
    for _, references in _iter_references(declaration):
        for manifest in references.get("manifests", []):
            if manifest["path"] == "deploy/openmed-compose.yaml":
                declared.update(manifest.get("compose_services", []))

    assert declared == set(compose["services"])


def _mutated(mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    declaration = _declaration()
    mutate(declaration)
    return declaration


def _first_with(declaration: dict[str, Any], key: str) -> dict[str, Any]:
    for _, references in _iter_references(declaration):
        if references.get(key):
            return references  # type: ignore[return-value]
    raise AssertionError("reference_kind_not_found")


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda d: _first_with(d, "routes")["routes"].append(
                {"surface": "rest", "method": "GET", "path": "/synthetic-missing"}
            ),
            "missing_route",
        ),
        (
            lambda d: _first_with(d, "routes")["routes"].append(
                {"surface": "rest", "method": "DELETE", "path": "/readyz"}
            ),
            "missing_route",
        ),
        (
            lambda d: _first_with(d, "routes")["routes"].append(
                {"surface": "journey_worker", "method": "GET", "path": "/missing"}
            ),
            "missing_route",
        ),
        (
            lambda d: _first_with(d, "routes")["routes"].append(
                {"surface": "mcp", "method": "POST", "path": "/missing"}
            ),
            "missing_route",
        ),
        (
            lambda d: _first_with(d, "modules")["modules"].append(
                "openmed.service.synthetic_missing_module"
            ),
            "missing_module",
        ),
        (
            lambda d: _first_with(d, "modules")["modules"].append(
                "openmed.service.jobs:SyntheticMissingSymbol"
            ),
            "missing_symbol",
        ),
        (
            lambda d: _first_with(d, "grpc_methods")["grpc_methods"].append(
                "OpenMedService/SyntheticMissing"
            ),
            "missing_grpc_method",
        ),
        (
            lambda d: _first_with(d, "manifests")["manifests"].append(
                {"path": "deploy/synthetic-missing.yaml"}
            ),
            "missing_manifest",
        ),
        (
            lambda d: _first_with(d, "manifests")["manifests"].append(
                {
                    "path": "deploy/openmed-compose.yaml",
                    "compose_services": ["synthetic-missing"],
                }
            ),
            "missing_compose_service",
        ),
        (
            lambda d: _first_with(d, "manifests")["manifests"].append(
                {
                    "path": "deploy/openmed-compose.yaml",
                    "compose_volumes": ["synthetic-missing"],
                }
            ),
            "missing_compose_volume",
        ),
        (
            lambda d: _first_with(d, "manifests")["manifests"].append(
                {
                    "path": "deploy/helm/openmed-service/values.yaml",
                    "helm_values": ["journey.syntheticMissing"],
                }
            ),
            "missing_helm_value",
        ),
        (
            lambda d: _first_with(d, "docs")["docs"].append(
                "docs/deploy/synthetic-missing.md"
            ),
            "missing_doc",
        ),
    ],
)
def test_reference_check_fails_on_a_missing_reference(
    mutate: Callable[[dict[str, Any]], None], code: str
) -> None:
    declaration = _mutated(mutate)

    assert list(_validator().iter_errors(declaration)) == []
    assert code in {reason for reason, _ in _reference_failures(declaration)}


def test_structural_check_fails_on_unknown_service_and_topology() -> None:
    unknown_service = _declaration()
    unknown_service["transports"][0]["services"].append("synthetic_missing")
    unknown_topology = _declaration()
    unknown_topology["durable_components"][0]["topologies"].append("synthetic_missing")
    two_references = _declaration()
    for topology in two_references["topologies"]:
        topology["status"] = "reference"

    assert "unknown_service" in {c for c, _ in _structural_failures(unknown_service)}
    assert "unknown_topology" in {c for c, _ in _structural_failures(unknown_topology)}
    assert "reference_topology_mismatch" in {
        c for c, _ in _structural_failures(two_references)
    }


# --- current behavior -----------------------------------------------------


def _by_id(items: list[dict[str, Any]], item_id: str) -> dict[str, Any]:
    matches = [item for item in items if item["id"] == item_id]
    assert len(matches) == 1
    return matches[0]


def test_grpc_is_declared_without_deployment_wiring() -> None:
    declaration = _declaration()
    grpc = _by_id(declaration["transports"], "grpc")
    unsupported = {item["id"] for item in declaration["unsupported_deployments"]}

    assert grpc["status"] == "unsupported"
    assert grpc["security_posture"]["encryption"] == "none"
    assert grpc["security_posture"]["default_exposure"] == "not_wired"
    assert "grpc_network_service" in unsupported
    manifests = [
        path
        for path in (ROOT / "deploy").rglob("*")
        if path.is_file() and path.suffix in {".yaml", ".yml", ".tpl", ""}
    ]
    manifests += [ROOT / name for name in ROOT_MANIFESTS]
    deploy_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in manifests
    )
    assert "grpc" not in deploy_text.lower()


def test_local_job_store_is_declared_single_process_and_unsupported() -> None:
    declaration = _declaration()
    jobs = _by_id(declaration["durable_components"], "service_job_metadata")
    rest = _by_id(declaration["services"], "rest_api")
    unsupported = {item["id"] for item in declaration["unsupported_deployments"]}

    assert jobs["replica_mode"] == "single_process"
    assert jobs["recovery_point_objective"]["status"] == "unsupported"
    assert jobs["recovery_time_objective"]["status"] == "unsupported"
    horizontal = _by_id(
        [dict(mode, id=mode["mode"]) for mode in rest["replica_modes"]], "horizontal"
    )
    assert horizontal["status"] == "restricted"
    assert "multi_replica_async_jobs" in unsupported


def test_reference_topology_is_the_offline_compose_bundle() -> None:
    declaration = _declaration()
    reference = _by_id(declaration["topologies"], declaration["reference_topology"])
    compose = yaml.safe_load(
        (ROOT / "deploy" / "openmed-compose.yaml").read_text(encoding="utf-8")
    )

    assert reference["references"]["manifests"][0]["path"] == (
        "deploy/openmed-compose.yaml"
    )
    assert compose["networks"]["openmed-internal"]["internal"] is True
    published = compose["services"]["openmed"]["ports"]
    assert published == [
        "${OPENMED_BIND_ADDRESS:-127.0.0.1}:${OPENMED_PORT:-8080}:8080"
    ]
    rest = _by_id(declaration["transports"], "rest_http")
    assert rest["security_posture"]["default_exposure"] == "loopback"
    for name, service in compose["services"].items():
        if name != "openmed":
            assert "ports" not in service


# --- docs -----------------------------------------------------------------


def test_docs_page_renders_every_declaration_row_and_nothing_else() -> None:
    page = DOCS_FILE.read_text(encoding="utf-8")
    expected = _expected_rows(_declaration())
    flattened = [row for rows in expected.values() for row in rows]

    missing = [row for row in flattened if row not in page]
    assert missing == []
    assert sorted(_page_table_rows(page)) == sorted(flattened)
    assert "deploy/reference/v3.4-reference-deployment.json" in page
    assert "deploy/reference/reference-deployment.schema.json" in page


def test_docs_page_drifts_are_detected() -> None:
    declaration = _declaration()
    declaration["transports"][0]["status"] = "unsupported"
    page = DOCS_FILE.read_text(encoding="utf-8")
    expected = [row for rows in _expected_rows(declaration).values() for row in rows]

    assert sorted(_page_table_rows(page)) != sorted(expected)


def test_docs_page_and_declaration_contain_no_hosts_credentials_or_patient_data() -> (
    None
):
    page = DOCS_FILE.read_text(encoding="utf-8")
    declaration = DECLARATION_FILE.read_text(encoding="utf-8")

    assert _public_text_findings(page) == []
    assert _public_text_findings(declaration) == []


@pytest.mark.parametrize(
    ("sentinel", "code"),
    [
        ("connect to openmed-db.internal for state", "dns_hostname"),
        ("bind to 10.0.0.12 only", "ipv4_address"),
        ("use the default [::]:50051 listener", "ipv6_address"),
        ("probe localhost first", "loopback_hostname"),
        ("reach postgres:5432 directly", "host_port"),
        ("set password=synthetic-sentinel-value", "credential_assignment"),
        ("export OPENMED_POSTGRES_PASSWORD before startup", "credential_env_var"),
        ("-----BEGIN RSA PRIVATE KEY-----", "private_key"),
        ("patient synthetic.patient@example.test", "email_address"),
        ("SSN 000-12-3456", "ssn_like"),
        ("MRN for the synthetic record", "medical_record_number"),
        ("admitted 2026-01-01", "calendar_date"),
        ("open the service with an explicit scheme://", "url_scheme"),
    ],
)
def test_public_text_scan_flags_sentinels(sentinel: str, code: str) -> None:
    assert code in _public_text_findings(f"Synthetic text: {sentinel}.")


def test_docs_page_is_published_in_nav_and_publication_manifest() -> None:
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    publication = (ROOT / "docs" / "brand" / "system" / "publication.yml").read_text(
        encoding="utf-8"
    )

    assert "deploy/reference-deployment.md" in nav
    assert "- deploy/reference-deployment.md" in publication
