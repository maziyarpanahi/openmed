"""Guard that the openmed package ships no telemetry, analytics, or phone-home.

Roadmap non-goal S8.8 #4 promises "no telemetry-by-default, no license server,
no mandatory network calls post-download", and the marketing site repeats "no
telemetry, no outbound calls at runtime". OM-004 guards raw-PHI logging and
OM-005 guards offline mode; this OM-099 guard asserts the library introduces no
analytics/telemetry client and no non-allowlisted outbound call.

The guard has two layers:

* A static scan of ``openmed/**/*.py`` that fails when a telemetry/analytics SDK
  is imported, a known phone-home host literal appears, a telemetry opt-OUT
  environment variable is read (an opt-out implies telemetry-on-by-default), or a
  brand-new module starts importing a raw-network library without being added to
  the reviewed ``ALLOWED_NETWORK_MODULES`` inventory.
* A runtime check that a representative de-identification path opens no socket,
  reusing the OM-005 socket-blocking pattern from ``test_offline_mode.py``.

Every fixture value below is synthetic.
"""

from __future__ import annotations

import ast
import socket
from pathlib import Path

from openmed.core.config import OpenMedConfig
from openmed.core.offline import OFFLINE_ENV_VAR

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "openmed"

# ---------------------------------------------------------------------------
# Static-scan denylists / allowlist inventory
# ---------------------------------------------------------------------------

# Raw-network modules whose import marks a file as touching the network. Model
# downloads via ``huggingface_hub`` do not appear here because they delegate the
# transport to that library rather than importing a socket/HTTP module directly.
NETWORK_IMPORT_MODULES = frozenset(
    {
        "socket",
        "ssl",
        "requests",
        "httpx",
        "aiohttp",
        "websocket",
        "websockets",
        "http.client",
        "urllib.request",
        "urllib3",
        "ftplib",
        "smtplib",
        "telnetlib",
    }
)

# The reviewed set of openmed modules allowed to touch the network today, each
# for a legitimate, explicitly user-initiated reason (model/dataset download,
# opt-in REST service, opt-in training rendezvous, or the offline guard itself).
# A new module importing a raw-network library must be added here on review --
# that is the "fails on additions" enforcement for outbound calls.
ALLOWED_NETWORK_MODULES: dict[str, str] = {
    "clinical/grounding/vocab.py": "opt-in clinical vocabulary download (offline-gated)",
    "core/offline.py": "the offline network-block guard itself (OM-005)",
    "eval/datasets/drugprot.py": "opt-in evaluation dataset download",
    "eval/suites/shield.py": "opt-in evaluation dataset/rows fetch",
    "service/client.py": "opt-in REST service HTTP client",
    "service/privacy_gateway.py": "opt-in privacy-gateway forwarding (user-configured)",
    "service/smart_backend.py": "opt-in smart-backend routing (user-configured)",
    "service/webhooks.py": "opt-in user-configured webhooks",
    "training/recipe.py": "opt-in training distributed rendezvous",
}

# Import roots of known analytics/telemetry/error-reporting SDKs. Matched against
# the top-level package of every import; none may appear in openmed/.
TELEMETRY_IMPORT_ROOTS = frozenset(
    {
        "segment",
        "posthog",
        "sentry_sdk",
        "sentry",
        "mixpanel",
        "amplitude",
        "amplitude_analytics",
        "statsd",
        "datadog",
        "ddtrace",
        "bugsnag",
        "rollbar",
        "newrelic",
        "opentelemetry",
        "elasticapm",
        "elastic_apm",
        "appdynamics",
        "libhoney",
        "scarf",
        "ga4mp",
    }
)

# Known phone-home / analytics-collector hostnames. Matched (casefolded) against
# string literals in the package.
TELEMETRY_HOST_LITERALS = frozenset(
    {
        "segment.io",
        "sentry.io",
        "google-analytics.com",
        "analytics.google.com",
        "mixpanel.com",
        "amplitude.com",
        "datadoghq.com",
        "bugsnag.com",
        "rollbar.com",
        "newrelic.com",
        "posthog.com",
        "scarf.sh",
        "doubleclick.net",
    }
)

# Environment-variable name fragments that imply a telemetry opt-OUT switch. The
# existence of an opt-out is itself a violation: it means telemetry is on by
# default. Matched (upper-cased) against environment-variable names the package
# actually reads.
TELEMETRY_ENV_FRAGMENTS = (
    "TELEMETRY",
    "ANALYTICS",
    "OPT_OUT",
    "OPTOUT",
    "DO_NOT_TRACK",
    "PHONE_HOME",
    "PHONEHOME",
    "SEND_STATS",
    "USAGE_STATS",
)

# Reviewed exceptions where a telemetry-tooling import is permitted because it is
# opt-in, off by default, and exports only to a user-configured endpoint (not a
# vendor phone-home). Maps ``relative/path.py`` -> allowed import root. The
# permitted properties are additionally pinned by
# ``test_service_tracing_is_opt_in_and_off_by_default``.
TELEMETRY_SDK_ALLOWLIST: dict[str, str] = {
    "service/tracing.py": "opentelemetry",
}

# Attribute chains that read an environment variable by name in ``arg`` 0.
_ENV_READ_ATTRS = {"getenv", "get", "setdefault", "pop"}


# ---------------------------------------------------------------------------
# Pure detectors (operate on source text so the self-tests can plant violations)
# ---------------------------------------------------------------------------


def _iter_import_roots(tree: ast.AST):
    """Yield (top_level_package, dotted_name) for every import in ``tree``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import, never a third-party SDK
                continue
            if node.module:
                yield node.module.split(".")[0], node.module


def _network_modules(source: str) -> set[str]:
    """Return the raw-network modules imported by ``source``."""
    tree = ast.parse(source)
    found: set[str] = set()
    for _root, dotted in _iter_import_roots(tree):
        if dotted in NETWORK_IMPORT_MODULES:
            found.add(dotted)
    return found


def _env_var_names(tree: ast.AST) -> set[str]:
    """Return environment-variable names the source reads by string literal."""
    names: set[str] = set()
    for node in ast.walk(tree):
        # os.environ["NAME"] / os.environ.get("NAME") style subscripts
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if _is_environ(node.value) and isinstance(node.slice.value, str):
                names.add(node.slice.value)
        # os.getenv("NAME"), os.environ.get("NAME"), environ.setdefault("NAME")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in _ENV_READ_ATTRS and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    if node.func.attr == "getenv" or _is_environ(node.func.value):
                        names.add(first.value)
    return names


def _is_environ(node: ast.AST) -> bool:
    """True when ``node`` denotes ``os.environ`` (or a bare ``environ``)."""
    if isinstance(node, ast.Attribute):
        return node.attr == "environ"
    return isinstance(node, ast.Name) and node.id == "environ"


def _string_literals(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value


def _telemetry_sdk_imports(source: str) -> list[str]:
    """Return dotted names of telemetry/analytics SDKs imported by ``source``."""
    tree = ast.parse(source)
    return [
        dotted
        for root, dotted in _iter_import_roots(tree)
        if root in TELEMETRY_IMPORT_ROOTS
    ]


def _telemetry_host_hits(source: str) -> list[str]:
    """Return known phone-home host literals appearing in ``source``."""
    hits: list[str] = []
    for literal in _string_literals(ast.parse(source)):
        lowered = literal.casefold()
        hits.extend(host for host in TELEMETRY_HOST_LITERALS if host in lowered)
    return hits


def _telemetry_optout_envs(source: str) -> list[str]:
    """Return environment variables read by ``source`` that look like opt-outs."""
    hits: list[str] = []
    for env_name in _env_var_names(ast.parse(source)):
        upper = env_name.upper()
        if any(fragment in upper for fragment in TELEMETRY_ENV_FRAGMENTS):
            hits.append(env_name)
    return hits


def _telemetry_violations(
    source: str, *, allowed_sdk_root: str | None = None
) -> list[str]:
    """Return human-readable telemetry indicators found in ``source``.

    ``allowed_sdk_root`` suppresses SDK-import findings for a single reviewed
    import root (e.g. opt-in OpenTelemetry in ``service/tracing.py``); every
    other indicator, including any other SDK, still fails.
    """
    violations: list[str] = []
    for dotted in _telemetry_sdk_imports(source):
        if allowed_sdk_root and (
            dotted == allowed_sdk_root or dotted.startswith(f"{allowed_sdk_root}.")
        ):
            continue
        violations.append(f"telemetry SDK import: {dotted}")
    violations.extend(
        f"telemetry host literal: {host}" for host in _telemetry_host_hits(source)
    )
    violations.extend(
        f"telemetry opt-out env var: {name}" for name in _telemetry_optout_envs(source)
    )
    return violations


def _package_files() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _rel(path: Path) -> str:
    return path.relative_to(PACKAGE_ROOT).as_posix()


# ---------------------------------------------------------------------------
# Static guard: the real package must be clean
# ---------------------------------------------------------------------------


def test_openmed_package_has_no_telemetry_indicators():
    offenders: dict[str, list[str]] = {}
    for path in _package_files():
        rel = _rel(path)
        violations = _telemetry_violations(
            path.read_text(encoding="utf-8"),
            allowed_sdk_root=TELEMETRY_SDK_ALLOWLIST.get(rel),
        )
        if violations:
            offenders[rel] = violations

    assert offenders == {}, f"telemetry/analytics indicators found: {offenders!r}"


def test_service_tracing_is_opt_in_and_off_by_default():
    """The one reviewed telemetry-tooling exception must stay opt-in and safe."""
    tracing = PACKAGE_ROOT / "service" / "tracing.py"
    source = tracing.read_text(encoding="utf-8")

    # OpenTelemetry usage is confined to this single reviewed module.
    users = [
        _rel(path)
        for path in _package_files()
        if any(
            root.startswith("opentelemetry")
            for root, _ in _iter_import_roots(
                ast.parse(path.read_text(encoding="utf-8"))
            )
        )
    ]
    assert users == ["service/tracing.py"], (
        f"opentelemetry imported outside the reviewed carve-out: {users!r}"
    )

    # Off by default and opt-in via an explicit enable flag (never an opt-out).
    assert "enabled: bool = False" in source
    assert 'TRACING_ENABLED_ENV_VAR = "OPENMED_SERVICE_TRACING_ENABLED"' in source

    # The export endpoint is user-supplied; no hardcoded vendor phone-home host.
    assert _telemetry_host_hits(source) == []
    assert _telemetry_optout_envs(source) == []


def test_network_surface_matches_reviewed_allowlist():
    discovered: dict[str, set[str]] = {}
    for path in _package_files():
        modules = _network_modules(path.read_text(encoding="utf-8"))
        if modules:
            discovered[_rel(path)] = modules

    unlisted = sorted(set(discovered) - set(ALLOWED_NETWORK_MODULES))
    assert unlisted == [], (
        "new module(s) import a raw-network library without a reviewed entry in "
        f"ALLOWED_NETWORK_MODULES: {unlisted!r}"
    )

    stale = sorted(set(ALLOWED_NETWORK_MODULES) - set(discovered))
    assert stale == [], (
        "ALLOWED_NETWORK_MODULES lists module(s) that no longer touch the "
        f"network -- prune them: {stale!r}"
    )


# ---------------------------------------------------------------------------
# Self-tests: the detectors must actually fire on planted violations
# ---------------------------------------------------------------------------


def test_guard_detects_planted_telemetry_sdk():
    source = "import sentry_sdk\nsentry_sdk.init('https://example.test')\n"
    assert any("telemetry SDK import" in v for v in _telemetry_violations(source))


def test_guard_detects_planted_phone_home_host():
    source = 'URL = "https://api.segment.io/v1/track"\n'
    assert any("telemetry host literal" in v for v in _telemetry_violations(source))


def test_guard_detects_planted_optout_env_var():
    source = 'import os\nflag = os.getenv("OPENMED_DISABLE_TELEMETRY")\n'
    assert any("telemetry opt-out env var" in v for v in _telemetry_violations(source))


def test_guard_detects_planted_unlisted_network_module():
    source = "import requests\nrequests.post('https://collector.example.test')\n"
    assert _network_modules(source) == {"requests"}


def test_guard_ignores_urllib_parse_and_offline_mentions():
    # urllib.parse is not a network transport; a docstring mentioning telemetry
    # must not be flagged as a telemetry indicator.
    source = (
        '"""Local-first module: no telemetry and no mandatory network access."""\n'
        "from urllib.parse import urljoin\n"
        "value = urljoin('a', 'b')\n"
    )
    assert _network_modules(source) == set()
    assert _telemetry_violations(source) == []


# ---------------------------------------------------------------------------
# Runtime guard: a representative deidentify path opens no socket (OM-005 reuse)
# ---------------------------------------------------------------------------


class _FakeLocalPiiPipeline:
    tokenizer = None

    @staticmethod
    def _entities(text: str):
        return [
            {
                "entity_group": "NAME",
                "score": 0.99,
                "word": "John Doe",
                "start": text.index("John Doe"),
                "end": text.index("John Doe") + len("John Doe"),
            },
            {
                "entity_group": "PHONE",
                "score": 0.98,
                "word": "555-1234",
                "start": text.index("555-1234"),
                "end": text.index("555-1234") + len("555-1234"),
            },
        ]

    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            return [self._entities(item) for item in text]
        return self._entities(text)


class _FakeLocalLoader:
    def __init__(self, config: OpenMedConfig):
        self.config = config
        self.pipeline = _FakeLocalPiiPipeline()

    def create_pipeline(self, *args, **kwargs):
        return self.pipeline

    def get_max_sequence_length(self, *args, **kwargs):
        return None


def test_representative_deidentify_path_opens_no_socket(monkeypatch):
    monkeypatch.delenv(OFFLINE_ENV_VAR, raising=False)
    blocked_attempts: list[tuple] = []

    def fail_socket(*args, **kwargs):
        blocked_attempts.append((args, kwargs))
        raise AssertionError("network egress attempted from deidentify path")

    monkeypatch.setattr(socket.socket, "connect", fail_socket)
    monkeypatch.setattr(socket.socket, "connect_ex", fail_socket)
    monkeypatch.setattr(socket, "create_connection", fail_socket)

    from openmed.core.pii import deidentify, extract_pii

    text = "Patient John Doe called 555-1234."
    config = OpenMedConfig(use_medical_tokenizer=False)
    loader = _FakeLocalLoader(config)

    pii_result = extract_pii(
        text,
        model_name="local-pii",
        config=config,
        loader=loader,
        use_smart_merging=False,
    )
    assert [(e.label, e.text) for e in pii_result.entities] == [
        ("NAME", "John Doe"),
        ("PHONE", "555-1234"),
    ]

    deid_result = deidentify(
        text,
        model_name="local-pii",
        config=config,
        loader=loader,
        use_smart_merging=False,
    )
    assert deid_result.deidentified_text == "Patient [NAME] called [PHONE]."
    assert blocked_attempts == []
