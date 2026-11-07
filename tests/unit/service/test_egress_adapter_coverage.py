"""Static coverage gate: every outbound call site in the service is reviewed.

The scan walks ``openmed/service/**/*.py`` without importing it and finds every
reference that can create an outbound transport: raw HTTP clients and
transports, ``urllib`` openers, gRPC channels, socket connections, and the OTLP
exporter. Each site must appear in :data:`REVIEWED_CALL_SITES` as either
guarded by the egress policy (and the enclosing function must visibly use an
egress guard) or exempt for a stated reason. A new outbound call site, or a
guarded site that drops its guard, fails this test (#3581).
"""

from __future__ import annotations

import ast
from pathlib import Path

from openmed.service.egress import ADAPTER_COVERAGE

SERVICE_ROOT = Path(__file__).resolve().parents[3] / "openmed" / "service"

# Dotted references that construct an outbound-capable transport or socket.
OUTBOUND_REFERENCES = frozenset(
    {
        "httpx.Client",
        "httpx.AsyncClient",
        "httpx.HTTPTransport",
        "httpx.AsyncHTTPTransport",
        "httpx.request",
        "httpx.stream",
        "httpx.get",
        "httpx.post",
        "httpx.put",
        "httpx.patch",
        "httpx.delete",
        "httpx.head",
        "httpx.options",
        "urlopen",
        "urllib.request.urlopen",
        "build_opener",
        "urllib.request.build_opener",
        "grpc.insecure_channel",
        "grpc.secure_channel",
        "socket.create_connection",
        "socket.socket",
        "http.client.HTTPConnection",
        "http.client.HTTPSConnection",
        "OTLPSpanExporter",
        "aiohttp.ClientSession",
        "requests.get",
        "requests.post",
        "requests.put",
        "requests.request",
        "requests.Session",
    }
)

# Names whose presence in the enclosing function shows the egress guard.
GUARD_NAMES = frozenset(
    {
        "egress_scope",
        "egress_urlopen",
        "pinned_grpc_target",
        "EgressOperation",
        "egress_http_client",
        "egress_async_http_client",
    }
)

# Adapter modules may delegate to these module-level guard entry points.
MODULE_GUARD_NAMES = GUARD_NAMES | {"authorize_smart_backend_config"}

GUARDED = "guarded"

# (module, enclosing qualified function) -> "guarded" or "exempt: <reason>".
REVIEWED_CALL_SITES: dict[tuple[str, str], str] = {
    # The guard implementation itself builds the pinned clients.
    ("openmed.service.egress", "egress_http_transport"): "exempt: guard implementation",
    (
        "openmed.service.egress",
        "egress_async_http_transport",
    ): "exempt: guard implementation",
    ("openmed.service.egress", "egress_http_client"): "exempt: guard implementation",
    (
        "openmed.service.egress",
        "egress_async_http_client",
    ): "exempt: guard implementation",
    ("openmed.service.egress", "egress_urlopen"): "exempt: guard implementation",
    (
        "openmed.service.egress",
        "_pinned_create_connection",
    ): "exempt: guard implementation (connects only to checked addresses)",
    (
        "openmed.service.egress",
        "_PinnedHTTPConnection",
    ): "exempt: guard implementation (pinned urllib connection class)",
    (
        "openmed.service.egress",
        "_PinnedHTTPSConnection",
    ): "exempt: guard implementation (pinned urllib connection class)",
    # Remote inference: gRPC channels dial the pinned, checked target.
    (
        "openmed.service.backends.remote_inference",
        "TritonGrpcTransport.__init__",
    ): GUARDED,
    # The OTLP exporter is created only behind an egress decision.
    ("openmed.service.tracing", "guarded_otlp_exporter"): GUARDED,
    # Caller-side SDKs: the caller chooses its own OpenMed server; these are not
    # service-controlled egress and never run inside the service process.
    (
        "openmed.service.client",
        "OpenMedClient.__init__",
    ): "exempt: caller-side REST SDK for a caller-chosen OpenMed server",
}

# Modules that import raw network libraries for reasons other than outbound
# service egress.
EXEMPT_NETWORK_MODULES: dict[str, str] = {
    "openmed.service.client": "caller-side REST SDK",
    "openmed.service.workflow_client": "caller-side workflow SDK over the REST SDK",
    "openmed.service.grpc_server": "inbound gRPC server",
    "openmed.service.proto.generated.openmed_pb2_grpc": "generated inbound stubs",
    "openmed.service.egress": "the egress guard implementation",
}

NETWORK_IMPORT_ROOTS = frozenset(
    {
        "httpx",
        "httpcore",
        "requests",
        "aiohttp",
        "urllib3",
        "grpc",
        "socket",
        "http.client",
        "urllib.request",
    }
)


def _module_name(path: Path) -> str:
    relative = path.relative_to(SERVICE_ROOT.parent.parent).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


class _Scanner(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.sites: dict[str, set[str]] = {}
        self.guards: dict[str, set[str]] = {}

    def _scope(self) -> str:
        return ".".join(self.stack) or "<module>"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.stack.append(node.name)
        # Annotations describe types only; defaults and bodies can connect.
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        for statement in node.body:
            self.visit(statement)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.target)
        if node.value is not None:
            self.visit(node.value)

    def _record(self, node: ast.AST) -> None:
        if not isinstance(getattr(node, "ctx", None), ast.Load):
            return
        dotted = _dotted(node)
        if dotted in OUTBOUND_REFERENCES:
            self.sites.setdefault(self._scope(), set()).add(dotted)
        name = node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", "")
        if name in GUARD_NAMES:
            for depth in range(1, len(self.stack) + 1):
                scope = ".".join(self.stack[:depth])
                self.guards.setdefault(scope, set()).add(name)
            self.guards.setdefault(self._scope(), set()).add(name)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._record(node)
        if _dotted(node) not in OUTBOUND_REFERENCES:
            self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self._record(node)


def _scan() -> tuple[dict[tuple[str, str], set[str]], dict[tuple[str, str], set[str]]]:
    sites: dict[tuple[str, str], set[str]] = {}
    guards: dict[tuple[str, str], set[str]] = {}
    for path in sorted(SERVICE_ROOT.rglob("*.py")):
        module = _module_name(path)
        scanner = _Scanner()
        scanner.visit(ast.parse(path.read_text(encoding="utf-8")))
        for scope, references in scanner.sites.items():
            sites[(module, scope)] = references
        for scope, names in scanner.guards.items():
            guards[(module, scope)] = names
    return sites, guards


def _network_importers() -> set[str]:
    modules = set()
    for path in sorted(SERVICE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                names = [node.module]
            if any(
                name == root or name.startswith(root + ".")
                for name in names
                for root in NETWORK_IMPORT_ROOTS
            ):
                modules.add(_module_name(path))
    return modules


def test_every_outbound_call_site_is_reviewed() -> None:
    sites, _ = _scan()
    unreviewed = sorted(set(sites) - set(REVIEWED_CALL_SITES))
    stale = sorted(set(REVIEWED_CALL_SITES) - set(sites))
    assert unreviewed == [], (
        "New outbound call sites must obtain an egress decision "
        f"(openmed.service.egress) and be reviewed here: {unreviewed!r}"
    )
    assert stale == [], f"Reviewed call sites no longer exist: {stale!r}"


def test_guarded_call_sites_visibly_use_the_egress_guard() -> None:
    _, guards = _scan()
    for site, status in REVIEWED_CALL_SITES.items():
        if status != GUARDED:
            assert status.startswith("exempt: ") and len(status) > len("exempt: ")
            continue
        assert guards.get(site), f"{site!r} creates a transport without a guard"


def test_network_importing_modules_are_covered_or_exempt() -> None:
    covered = {row.module for row in ADAPTER_COVERAGE}
    uncovered = sorted(_network_importers() - covered - set(EXEMPT_NETWORK_MODULES))
    assert uncovered == [], (
        "Service modules importing network libraries must be in "
        f"ADAPTER_COVERAGE or reviewed as exempt: {uncovered!r}"
    )


def test_covered_adapter_modules_import_the_egress_policy() -> None:
    for row in ADAPTER_COVERAGE:
        relative = Path(*row.module.split(".")[2:]).with_suffix(".py")
        source = (SERVICE_ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports_egress = any(
            isinstance(node, ast.ImportFrom)
            and (
                (node.level and node.module == "egress")
                or node.module == "openmed.service.egress"
            )
            for node in ast.walk(tree)
        )
        assert imports_egress, f"{row.module} does not import the egress policy"
        assert any(name in source for name in MODULE_GUARD_NAMES), row.module


def test_raw_http_clients_are_only_built_by_the_guard_or_reviewed_sdks() -> None:
    sites, _ = _scan()
    raw_clients = {
        site
        for site, references in sites.items()
        if references
        & {
            "httpx.Client",
            "httpx.AsyncClient",
            "httpx.post",
            "httpx.get",
            "httpx.request",
            "httpx.stream",
            "urlopen",
            "urllib.request.urlopen",
        }
    }
    allowed_modules = {"openmed.service.egress", "openmed.service.client"}
    assert {module for module, _ in raw_clients} <= allowed_modules


_PLANTED_BYPASS = """
import httpx
import grpc
from urllib.request import urlopen


def helper(client: httpx.Client | None = None) -> httpx.Response:
    return httpx.post("https://example.invalid/x")


class Connector:
    session: httpx.AsyncClient

    def open(self):
        self.channel = grpc.insecure_channel("example.invalid:1")
        return urlopen("https://example.invalid/y")
"""


def test_scanner_flags_planted_bypasses_and_ignores_annotations() -> None:
    scanner = _Scanner()
    scanner.visit(ast.parse(_PLANTED_BYPASS))
    assert scanner.sites == {
        "helper": {"httpx.post"},
        "Connector.open": {"grpc.insecure_channel", "urlopen"},
    }
    assert scanner.guards == {}
