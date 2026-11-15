"""Versioned service security profiles and their fail-closed self-check.

One resolver, :func:`resolve_security_profile`, decides whether the service runs
under the development or the production security profile. Service features ask
it instead of reading ``OPENMED_PROFILE`` themselves.

The self-check is read-only. Each registered check inspects one control of a
configured application and returns a closed status plus a reason code. Reports
carry check ids, statuses, levels, reason codes, versions and digests only:
never secrets, key identifiers, token claims, hosts, origins, connection
strings, paths, request data or clinical values.

Outcome rules:

* ``pass`` and ``not_applicable`` never block.
* ``fail`` blocks when the check is ``required`` in an enforcing profile.
* ``not_implemented`` means the build has no implementation of the control
  while the feature it guards is enabled. With the feature disabled the check
  reports ``not_applicable`` instead, so a not-yet-implemented control blocks a
  production deployment only when that deployment turns the feature on.
* In ``prod`` v1, controls that guard a surface operators cannot switch off on
  this build (GraphQL GET and IDE, API docs, request-supplied egress targets)
  and controls commonly delegated to the platform (listener TLS, admission
  limits, the served-model allowlist) are ``advisory``. Their findings are
  reported but do not block. A later profile version makes them required.

The ``dev`` profile never blocks: every finding is advisory, so development
behavior stays backward compatible and must be selected explicitly.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final
from urllib.parse import parse_qsl, urlsplit

SECURITY_PROFILE_ENV_VAR: Final = "OPENMED_SERVICE_SECURITY_PROFILE"
CONFIG_PROFILE_ENV_VAR: Final = "OPENMED_PROFILE"
LISTEN_HOST_ENV_VAR: Final = "OPENMED_SERVICE_LISTEN_HOST"
LISTEN_TLS_ENV_VAR: Final = "OPENMED_SERVICE_LISTEN_TLS"
JOURNEY_POSTGRES_DSN_ENV_VAR: Final = "OPENMED_JOURNEY_POSTGRES_DSN"
PRIVACY_GATEWAY_ENDPOINT_ENV_VAR: Final = "OPENMED_SERVICE_PRIVACY_GATEWAY_ENDPOINT"
SERVED_MODELS_ENV_VAR: Final = "OPENMED_SERVICE_SERVED_MODELS"

SECURITY_PROFILE_REPORT_SCHEMA_VERSION: Final = (
    "openmed.service.security_profile_report.v1"
)
SECURITY_PROFILE_INVALID_CODE: Final = "security_profile_invalid"
MAX_PRODUCTION_JWT_LIFETIME_SECONDS: Final = 86_400.0

EXIT_PROFILE_PASSED: Final = 0
EXIT_PROFILE_INVALID: Final = 1
EXIT_PROFILE_UNAVAILABLE: Final = 2

# Configuration profiles that select the development security profile. Every
# other value, including an unset variable and custom profile names, selects
# production so an unlabeled deployment fails closed.
DEVELOPMENT_CONFIG_PROFILES: Final = frozenset({"dev", "test", "fast"})

_CHECK_ID: Final = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
_REASON: Final = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_TRUE_VALUES: Final = frozenset({"1", "true", "yes", "on", "enabled"})
_NOT_IMPLEMENTED: Final = "control_not_implemented"
_CHECK_ERROR: Final = "check_error"

_LOGGER = logging.getLogger("openmed.service.security_profile")


class SecurityProfileName(str, Enum):
    """Closed names of the service security profiles."""

    DEV = "dev"
    PROD = "prod"


class ProfileSource(str, Enum):
    """Where the resolved security profile came from."""

    EXPLICIT = "explicit"
    SECURITY_PROFILE_ENV = "security_profile_env"
    CONFIG_PROFILE_ENV = "config_profile_env"
    DEFAULT = "default"


class CheckStatus(str, Enum):
    """Closed outcome of one security check."""

    PASS = "pass"
    FAIL = "fail"
    NOT_IMPLEMENTED = "not_implemented"
    NOT_APPLICABLE = "not_applicable"


class CheckLevel(str, Enum):
    """Whether a finding blocks an enforcing profile or is only reported."""

    REQUIRED = "required"
    ADVISORY = "advisory"


class ReportStatus(str, Enum):
    """Overall report state: pass, advisory findings only, or blocking."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class SecurityControl(str, Enum):
    """Closed control areas used to group checks."""

    LISTENER = "listener"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    HTTP = "http"
    LIMITS = "limits"
    MODELS = "models"
    GRAPHQL = "graphql"
    JOURNEY = "journey"
    EGRESS = "egress"
    GRPC = "grpc"


@dataclass(frozen=True, slots=True)
class SecurityProfile:
    """One versioned security profile.

    Args:
        name: The profile name.
        version: Monotonic profile version; bump it when the set of required
            checks or their thresholds change.
        enforcing: Whether required findings block startup readiness.
    """

    name: SecurityProfileName
    version: int
    enforcing: bool

    @property
    def identifier(self) -> str:
        """Return the stable profile identifier, for example ``...prod.v1``."""
        return f"openmed.service.security_profile.{self.name.value}.v{self.version}"


DEV_PROFILE: Final = SecurityProfile(SecurityProfileName.DEV, 1, enforcing=False)
PROD_PROFILE: Final = SecurityProfile(SecurityProfileName.PROD, 1, enforcing=True)
SECURITY_PROFILES: Final[Mapping[SecurityProfileName, SecurityProfile]] = {
    SecurityProfileName.DEV: DEV_PROFILE,
    SecurityProfileName.PROD: PROD_PROFILE,
}


@dataclass(frozen=True, slots=True)
class ResolvedSecurityProfile:
    """A security profile together with the source that selected it."""

    profile: SecurityProfile
    source: ProfileSource

    @property
    def name(self) -> SecurityProfileName:
        """Return the resolved profile name."""
        return self.profile.name

    @property
    def is_production(self) -> bool:
        """Return whether the production security profile is in effect."""
        return self.profile.name is SecurityProfileName.PROD


def _profile_from_name(value: str) -> SecurityProfile:
    normalized = value.strip().lower()
    try:
        return SECURITY_PROFILES[SecurityProfileName(normalized)]
    except ValueError:
        raise ValueError(
            f"{SECURITY_PROFILE_ENV_VAR} must be one of: dev, prod"
        ) from None


def resolve_security_profile(
    environ: Mapping[str, str] | None = None,
    *,
    override: str | SecurityProfileName | None = None,
) -> ResolvedSecurityProfile:
    """Resolve the service security profile. This is the only resolver.

    Resolution order: an explicit ``override``; then
    ``OPENMED_SERVICE_SECURITY_PROFILE`` (``dev`` or ``prod``); then
    ``OPENMED_PROFILE``, where ``dev``, ``test`` and ``fast`` select ``dev``
    and every other value selects ``prod``; then ``prod`` when nothing is set.

    Args:
        environ: Environment mapping to read. Defaults to ``os.environ``.
        override: Explicit profile name that wins over the environment.

    Returns:
        The resolved profile and its source.

    Raises:
        ValueError: If an explicit profile name is unknown. The message names
            the variable, never the rejected value.
    """
    env = os.environ if environ is None else environ
    if override is not None:
        name = override.value if isinstance(override, SecurityProfileName) else override
        return ResolvedSecurityProfile(_profile_from_name(name), ProfileSource.EXPLICIT)
    explicit = env.get(SECURITY_PROFILE_ENV_VAR, "").strip()
    if explicit:
        return ResolvedSecurityProfile(
            _profile_from_name(explicit), ProfileSource.SECURITY_PROFILE_ENV
        )
    config_profile = env.get(CONFIG_PROFILE_ENV_VAR, "").strip()
    if config_profile:
        profile = (
            DEV_PROFILE
            if config_profile.lower() in DEVELOPMENT_CONFIG_PROFILES
            else PROD_PROFILE
        )
        return ResolvedSecurityProfile(profile, ProfileSource.CONFIG_PROFILE_ENV)
    return ResolvedSecurityProfile(PROD_PROFILE, ProfileSource.DEFAULT)


@dataclass(frozen=True, slots=True)
class ListenerFacts:
    """The address and transport the REST listener binds.

    Args:
        host: Bind host as given to the launcher.
        tls: Whether the listener terminates TLS in-process.
    """

    host: str = field(repr=False)
    tls: bool = False


def listener_facts_from_env(
    environ: Mapping[str, str] | None = None,
) -> ListenerFacts | None:
    """Read the listener facts the reference launcher exports, if any.

    Args:
        environ: Environment mapping to read. Defaults to ``os.environ``.

    Returns:
        Listener facts, or ``None`` when no launcher recorded them.
    """
    env = os.environ if environ is None else environ
    host = env.get(LISTEN_HOST_ENV_VAR, "").strip()
    if not host:
        return None
    tls = env.get(LISTEN_TLS_ENV_VAR, "").strip().lower() in _TRUE_VALUES
    return ListenerFacts(host=host, tls=tls)


@dataclass(frozen=True, slots=True)
class SecurityCheckContext:
    """Read-only inputs one evaluation passes to every check.

    Checks read live application state and the environment. Nothing in the
    context is ever serialized into a report.
    """

    profile: ResolvedSecurityProfile
    app: Any = field(repr=False)
    environ: Mapping[str, str] = field(repr=False)
    listener: ListenerFacts | None = field(default=None, repr=False)
    grpc_address: str | None = field(default=None, repr=False)

    def state(self, name: str) -> Any:
        """Return one ``app.state`` attribute, or ``None`` when absent."""
        return getattr(getattr(self.app, "state", None), name, None)

    def env(self, name: str) -> str:
        """Return one stripped environment value, or an empty string."""
        return str(self.environ.get(name, "") or "").strip()


@dataclass(frozen=True, slots=True)
class CheckOutcome:
    """A check's raw outcome: a closed status and a value-free reason code."""

    status: CheckStatus
    reason: str


CheckEvaluator = Callable[[SecurityCheckContext], CheckOutcome]


@dataclass(frozen=True, slots=True)
class SecurityCheck:
    """One registered security check.

    Args:
        check_id: Stable dotted identifier such as ``auth.enabled``.
        control: Control area the check belongs to.
        prod_level: Level in the production profile. Development is always
            advisory.
        evaluate: Read-only evaluator returning a :class:`CheckOutcome`.
        tracking_issue: Repository issue that implements a missing control.
    """

    check_id: str
    control: SecurityControl
    prod_level: CheckLevel
    evaluate: CheckEvaluator = field(repr=False, compare=False)
    tracking_issue: int | None = None

    def __post_init__(self) -> None:
        if not _CHECK_ID.fullmatch(self.check_id):
            raise ValueError("security check id must be a dotted lowercase name")

    def level_for(self, profile: SecurityProfile) -> CheckLevel:
        """Return this check's level under ``profile``."""
        if not profile.enforcing:
            return CheckLevel.ADVISORY
        return self.prod_level


_REGISTRY: dict[str, SecurityCheck] = {}


def register_security_check(
    check: SecurityCheck, *, replace: bool = False
) -> SecurityCheck:
    """Add a check to the registry evaluated by every self-check.

    Later features plug in their controls here, or replace a built-in
    ``not_implemented`` stub with a real evaluator using ``replace=True``.

    Args:
        check: The check to register.
        replace: Allow replacing an existing check with the same id.

    Returns:
        The registered check.

    Raises:
        ValueError: If the id is already registered and ``replace`` is false.
    """
    if check.check_id in _REGISTRY and not replace:
        raise ValueError("security check id is already registered")
    _REGISTRY[check.check_id] = check
    return check


def unregister_security_check(check_id: str) -> None:
    """Remove a check from the registry if it is present."""
    _REGISTRY.pop(check_id, None)


def registered_security_checks() -> tuple[SecurityCheck, ...]:
    """Return the registered checks in evaluation order."""
    return tuple(_REGISTRY.values())


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One check's result as it appears in a report."""

    check_id: str
    control: SecurityControl
    level: CheckLevel
    status: CheckStatus
    reason: str
    blocking: bool
    tracking_issue: int | None = None

    @property
    def is_finding(self) -> bool:
        """Return whether the control failed or is missing."""
        return self.status in {CheckStatus.FAIL, CheckStatus.NOT_IMPLEMENTED}

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free JSON form of this result."""
        payload: dict[str, Any] = {
            "id": self.check_id,
            "control": self.control.value,
            "level": self.level.value,
            "status": self.status.value,
            "reason": self.reason,
            "blocking": self.blocking,
        }
        if self.tracking_issue is not None:
            payload["tracking_issue"] = self.tracking_issue
        return payload


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SecurityProfileReport:
    """Versioned, value-free result of one security-profile self-check."""

    resolved: ResolvedSecurityProfile
    results: tuple[CheckResult, ...]

    @property
    def blocking_checks(self) -> tuple[str, ...]:
        """Return the ids of checks that block readiness."""
        return tuple(result.check_id for result in self.results if result.blocking)

    @property
    def blocking(self) -> bool:
        """Return whether the profile is invalid and the service must close."""
        return bool(self.blocking_checks)

    @property
    def ready(self) -> bool:
        """Return whether the service may become ready under this profile."""
        return not self.blocking

    @property
    def status(self) -> ReportStatus:
        """Return pass, warn (advisory findings only) or fail."""
        if self.blocking:
            return ReportStatus.FAIL
        if any(result.is_finding for result in self.results):
            return ReportStatus.WARN
        return ReportStatus.PASS

    @property
    def exit_code(self) -> int:
        """Return the stable preflight exit code for this report."""
        return EXIT_PROFILE_INVALID if self.blocking else EXIT_PROFILE_PASSED

    @property
    def profile_digest(self) -> str:
        """Return a digest of the profile id and each check's level."""
        profile = self.resolved.profile
        return _canonical_digest(
            {
                "profile": profile.identifier,
                "checks": [[item.check_id, item.level.value] for item in self.results],
            }
        )

    def _body(self) -> dict[str, Any]:
        profile = self.resolved.profile
        counts = {status.value: 0 for status in CheckStatus}
        for result in self.results:
            counts[result.status.value] += 1
        return {
            "schema_version": SECURITY_PROFILE_REPORT_SCHEMA_VERSION,
            "profile": {
                "name": profile.name.value,
                "version": profile.version,
                "id": profile.identifier,
                "source": self.resolved.source.value,
                "enforcing": profile.enforcing,
                "digest": self.profile_digest,
            },
            "status": self.status.value,
            "ready": self.ready,
            "summary": {
                "checks": len(self.results),
                "passed": counts[CheckStatus.PASS.value],
                "failed": counts[CheckStatus.FAIL.value],
                "not_implemented": counts[CheckStatus.NOT_IMPLEMENTED.value],
                "not_applicable": counts[CheckStatus.NOT_APPLICABLE.value],
                "blocking": len(self.blocking_checks),
                "advisory_findings": sum(
                    1
                    for result in self.results
                    if result.is_finding and not result.blocking
                ),
            },
            "blocking_checks": list(self.blocking_checks),
            "checks": [result.to_dict() for result in self.results],
        }

    @property
    def digest(self) -> str:
        """Return the deterministic digest of the report body."""
        return _canonical_digest(self._body())

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON report, including its digest."""
        body = self._body()
        body["digest"] = _canonical_digest(body)
        return body

    def to_json(self) -> str:
        """Return the report as canonical JSON text."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    def to_text(self) -> str:
        """Return the human report with the same fields as the JSON form."""
        profile = self.resolved.profile
        lines = [
            f"OpenMed service security profile: {profile.name.value} "
            f"v{profile.version} (source: {self.resolved.source.value})",
            f"Status: {self.status.value} "
            f"({len(self.blocking_checks)} blocking, "
            f"{sum(1 for r in self.results if r.is_finding and not r.blocking)} "
            "advisory findings; ! marks a blocking check)",
        ]
        id_width = max((len(result.check_id) for result in self.results), default=0)
        for result in self.results:
            marker = "!" if result.blocking else " "
            line = (
                f"{marker} {result.status.value:<15} {result.check_id:<{id_width}}  "
                f"{result.level.value:<8}  {result.reason}"
            )
            if result.tracking_issue is not None:
                line += f"  (#{result.tracking_issue})"
            lines.append(line)
        lines.append(f"Report digest: {self.digest}")
        return "\n".join(lines)

    def readiness_details(self) -> dict[str, Any]:
        """Return the value-free details exposed by ``/readyz``."""
        profile = self.resolved.profile
        return {
            "profile": profile.name.value,
            "profile_version": profile.version,
            "blocking_checks": list(self.blocking_checks),
            "report_digest": self.digest,
        }


def evaluate_security_checks(
    context: SecurityCheckContext,
    checks: tuple[SecurityCheck, ...] | None = None,
) -> SecurityProfileReport:
    """Run every check against ``context`` and build a report.

    A check that raises, or returns a malformed outcome, fails with the
    ``check_error`` reason so an enforcing profile stays closed.

    Args:
        context: Read-only evaluation inputs.
        checks: Checks to run. Defaults to the registry.

    Returns:
        The value-free report.
    """
    profile = context.profile.profile
    results: list[CheckResult] = []
    for check in registered_security_checks() if checks is None else checks:
        try:
            outcome = check.evaluate(context)
            if not isinstance(outcome, CheckOutcome) or not isinstance(
                outcome.status, CheckStatus
            ):
                raise TypeError("invalid outcome")
            if not isinstance(outcome.reason, str) or not _REASON.fullmatch(
                outcome.reason
            ):
                raise TypeError("invalid reason")
        except Exception:
            outcome = CheckOutcome(CheckStatus.FAIL, _CHECK_ERROR)
        level = check.level_for(profile)
        blocking = (
            profile.enforcing
            and level is CheckLevel.REQUIRED
            and outcome.status in {CheckStatus.FAIL, CheckStatus.NOT_IMPLEMENTED}
        )
        results.append(
            CheckResult(
                check_id=check.check_id,
                control=check.control,
                level=level,
                status=outcome.status,
                reason=outcome.reason,
                blocking=blocking,
                tracking_issue=check.tracking_issue,
            )
        )
    return SecurityProfileReport(resolved=context.profile, results=tuple(results))


def evaluate_service_security_profile(
    app: Any,
    *,
    environ: Mapping[str, str] | None = None,
    profile: str | SecurityProfileName | None = None,
    listener: ListenerFacts | None = None,
    grpc_address: str | None = None,
) -> SecurityProfileReport:
    """Evaluate the security profile of a configured OpenMed REST app.

    Startup, ``/readyz`` and the CLI preflight all call this function.

    Args:
        app: The FastAPI application built by ``create_app``.
        environ: Environment for profile resolution and deployment facts the
            app does not hold (listener, Journey store, egress endpoints).
            Defaults to ``os.environ``.
        profile: Optional explicit profile name.
        listener: Listener facts. Defaults to the launcher-exported values.
        grpc_address: Optional gRPC listener address to include.

    Returns:
        The value-free report.
    """
    env = os.environ if environ is None else environ
    context = SecurityCheckContext(
        profile=resolve_security_profile(env, override=profile),
        app=app,
        environ=env,
        listener=listener if listener is not None else listener_facts_from_env(env),
        grpc_address=grpc_address,
    )
    return evaluate_security_checks(context)


def log_security_profile_report(report: SecurityProfileReport) -> None:
    """Log one value-free line when a report blocks readiness."""
    if not report.blocking:
        return
    profile = report.resolved.profile
    _LOGGER.error(
        "%s profile=%s version=%d blocking=%s digest=%s",
        SECURITY_PROFILE_INVALID_CODE,
        profile.name.value,
        profile.version,
        ",".join(report.blocking_checks),
        report.digest,
    )


@contextmanager
def _profile_environment(profile: str | None) -> Iterator[None]:
    """Expose an explicit profile to configuration parsers for one evaluation."""
    if profile is None:
        yield
        return
    _profile_from_name(profile)
    previous = os.environ.get(SECURITY_PROFILE_ENV_VAR)
    os.environ[SECURITY_PROFILE_ENV_VAR] = profile
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(SECURITY_PROFILE_ENV_VAR, None)
        else:
            os.environ[SECURITY_PROFILE_ENV_VAR] = previous


def run_service_preflight(
    *,
    profile: str | None = None,
    listen_host: str | None = None,
    listen_tls: bool | None = None,
    grpc_address: str | None = None,
) -> SecurityProfileReport:
    """Build the service app from the environment and run the self-check.

    The app is constructed exactly as the server constructs it, but its
    lifespan never starts: nothing is preloaded, bound or contacted.

    Args:
        profile: Optional explicit profile name (``dev`` or ``prod``).
        listen_host: Listener host to check. Defaults to the launcher value.
        listen_tls: Whether that listener terminates TLS in-process.
        grpc_address: Optional gRPC listener address to include.

    Returns:
        The value-free report.

    Raises:
        ValueError: If the service configuration cannot be parsed.
    """
    from .app import create_app

    with _profile_environment(profile):
        app = create_app()
        try:
            listener = listener_facts_from_env()
            if listen_host is not None:
                listener = ListenerFacts(host=listen_host, tls=bool(listen_tls))
            elif listener is not None and listen_tls is not None:
                listener = ListenerFacts(host=listener.host, tls=listen_tls)
            return evaluate_service_security_profile(
                app, profile=profile, listener=listener, grpc_address=grpc_address
            )
        finally:
            tracing = getattr(app.state, "tracing", None)
            if tracing is not None:
                tracing.shutdown()


# ---------------------------------------------------------------------------
# Built-in checks
# ---------------------------------------------------------------------------


def _pass(reason: str) -> CheckOutcome:
    return CheckOutcome(CheckStatus.PASS, reason)


def _fail(reason: str) -> CheckOutcome:
    return CheckOutcome(CheckStatus.FAIL, reason)


def _not_applicable(reason: str) -> CheckOutcome:
    return CheckOutcome(CheckStatus.NOT_APPLICABLE, reason)


def _not_implemented() -> CheckOutcome:
    return CheckOutcome(CheckStatus.NOT_IMPLEMENTED, _NOT_IMPLEMENTED)


def is_loopback_host(host: str) -> bool:
    """Return whether a bind host only accepts local connections.

    Args:
        host: Host name or IP literal, optionally bracketed.

    Returns:
        True for ``localhost`` and loopback addresses; false for wildcard,
        routable and unresolved names.
    """
    value = host.strip().lower()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _grpc_host(address: str) -> str:
    value = address.strip()
    if value.startswith("["):
        return value[: value.find("]") + 1] if "]" in value else value
    return value.rsplit(":", 1)[0] if ":" in value else value


def _auth_config(context: SecurityCheckContext) -> Any:
    return getattr(context.state("auth"), "config", None)


def _mtls_config(context: SecurityCheckContext) -> Any:
    return getattr(context.state("mtls"), "config", None)


def _has_route(app: Any, path: str) -> bool:
    return any(getattr(route, "path", None) == path for route in app.routes)


def _check_listener_transport(context: SecurityCheckContext) -> CheckOutcome:
    listener = context.listener
    if listener is None:
        return _fail("listener_unknown")
    if is_loopback_host(listener.host):
        return _pass("loopback_listener")
    if listener.tls:
        return _pass("tls_listener")
    mtls = _mtls_config(context)
    if getattr(mtls, "enabled", False) and getattr(mtls, "client_cert_header", None):
        return _pass("tls_terminated_by_trusted_proxy")
    return _fail("plaintext_remote_listener")


def _check_auth_enabled(context: SecurityCheckContext) -> CheckOutcome:
    config = _auth_config(context)
    if config is None:
        return _fail("auth_unavailable")
    return _pass("auth_enabled") if config.enabled else _fail("auth_disabled")


def _check_auth_deny_by_default(context: SecurityCheckContext) -> CheckOutcome:
    config = _auth_config(context)
    if config is None:
        return _fail("auth_unavailable")
    if config.deny_by_default:
        return _pass("deny_by_default_enabled")
    return _fail("deny_by_default_disabled")


def _check_auth_credentials(context: SecurityCheckContext) -> CheckOutcome:
    config = _auth_config(context)
    if config is None or not config.enabled:
        return _not_applicable("auth_disabled")
    mtls = _mtls_config(context)
    if config.api_keys or config.jwks or getattr(mtls, "enabled", False):
        return _pass("credentials_configured")
    return _fail("credentials_missing")


def _check_jwt_claims(context: SecurityCheckContext) -> CheckOutcome:
    config = _auth_config(context)
    if config is None or not config.enabled:
        return _not_applicable("auth_disabled")
    if not config.jwks:
        return _not_applicable("jwt_not_configured")
    lifetime = config.jwt_max_lifetime_seconds
    if lifetime is None:
        return _fail("jwt_lifetime_unbounded")
    if lifetime > MAX_PRODUCTION_JWT_LIFETIME_SECONDS:
        return _fail("jwt_lifetime_excessive")
    if not config.jwt_issuer:
        return _fail("jwt_issuer_missing")
    if not config.jwt_audiences:
        return _fail("jwt_audience_missing")
    return _pass("jwt_claims_bounded")


def _check_route_policy(context: SecurityCheckContext) -> CheckOutcome:
    from .auth import validate_route_policies

    try:
        validate_route_policies(context.app)
    except ValueError:
        return _fail("route_policy_undeclared")
    return _pass("route_policies_declared")


def _check_cors(context: SecurityCheckContext) -> CheckOutcome:
    security = context.state("security")
    if security is None:
        return _fail("http_policy_unavailable")
    origins = tuple(security.cors_origins)
    if not origins:
        return _pass("cors_disabled")
    for origin in origins:
        parsed = urlsplit(origin)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or "*" in origin
        ):
            return _fail("cors_origin_not_https")
    return _pass("cors_origins_https")


def _check_trusted_hosts(context: SecurityCheckContext) -> CheckOutcome:
    security = context.state("security")
    if security is None:
        return _fail("http_policy_unavailable")
    hosts = tuple(security.trusted_hosts)
    if not hosts or "*" in hosts:
        return _fail("trusted_hosts_unrestricted")
    return _pass("trusted_hosts_allowlisted")


def _check_api_docs(context: SecurityCheckContext) -> CheckOutcome:
    app = context.app
    exposed = any(
        getattr(app, name, None) for name in ("docs_url", "redoc_url", "openapi_url")
    )
    return _fail("api_docs_exposed") if exposed else _pass("api_docs_hidden")


def _check_request_body(context: SecurityCheckContext) -> CheckOutcome:
    limits = context.state("operational_limits")
    model_limit = context.state("model_body_limit_bytes")
    maximum = getattr(limits, "max_request_bytes", None)
    if (
        type(maximum) is not int
        or maximum <= 0
        or type(model_limit) is not int
        or model_limit <= 0
    ):
        return _fail("request_body_limit_missing")
    return _pass("request_body_bounded")


def _check_admission(context: SecurityCheckContext) -> CheckOutcome:
    runtime = context.state("runtime")
    throttle = getattr(runtime, "throttle", None)
    if throttle is None:
        from .runtime import parse_service_throttle_config

        throttle = parse_service_throttle_config()
    if throttle.rate_limit_enabled or throttle.concurrency_enabled:
        return _pass("admission_bounded")
    return _fail("admission_unbounded")


def _check_served_models(context: SecurityCheckContext) -> CheckOutcome:
    runtime = context.state("runtime")
    if runtime is not None:
        explicit = bool(runtime.model_policy_report().get("explicit"))
    else:
        explicit = SERVED_MODELS_ENV_VAR in context.environ
    if explicit:
        return _pass("served_models_explicit")
    return _fail("served_models_not_explicit")


def _check_graphql_mounted(context: SecurityCheckContext) -> CheckOutcome:
    if not _has_route(context.app, "/graphql"):
        return _not_applicable("graphql_not_mounted")
    return _not_implemented()


def _journey_store_configured(context: SecurityCheckContext) -> bool:
    return bool(context.env(JOURNEY_POSTGRES_DSN_ENV_VAR))


def _check_journey_principal_binding(context: SecurityCheckContext) -> CheckOutcome:
    catalog = context.state("journey_resources")
    records = getattr(catalog, "record_count", 0)
    if not _journey_store_configured(context) and not records:
        return _not_applicable("journey_disabled")
    return _not_implemented()


def _check_journey_encryption(context: SecurityCheckContext) -> CheckOutcome:
    if not _journey_store_configured(context):
        return _not_applicable("journey_store_disabled")
    return _not_implemented()


def _check_journey_store_tls(context: SecurityCheckContext) -> CheckOutcome:
    dsn = context.env(JOURNEY_POSTGRES_DSN_ENV_VAR)
    if not dsn:
        return _not_applicable("journey_store_disabled")
    try:
        parsed = urlsplit(dsn)
        query = parse_qsl(parsed.query, keep_blank_values=True)
    except ValueError:
        return _fail("postgres_dsn_invalid")
    if parsed.scheme not in {"postgres", "postgresql"}:
        return _fail("postgres_dsn_invalid")
    modes = [value for key, value in query if key == "sslmode"]
    if modes == ["verify-full"]:
        return _pass("postgres_tls_verified")
    if not modes:
        return _fail("postgres_tls_not_configured")
    return _fail("postgres_tls_not_verified")


def _check_configured_egress(context: SecurityCheckContext) -> CheckOutcome:
    tracing_config = getattr(context.state("tracing"), "config", None)
    openhim = context.state("openhim_settings")
    configured = (
        bool(context.env(PRIVACY_GATEWAY_ENDPOINT_ENV_VAR))
        or bool(
            getattr(tracing_config, "enabled", False)
            and getattr(tracing_config, "otlp_endpoint", None)
        )
        or bool(getattr(openhim, "enabled", False))
    )
    if not configured:
        return _not_applicable("egress_adapters_disabled")
    return _not_implemented()


_REQUEST_TARGET_ROUTES: Final = (
    "/jobs",
    "/fhir/smart-backend/ingestions",
    "/fhir/bulk/exports",
    "/fhir/bulk/imports",
)


def _check_request_egress(context: SecurityCheckContext) -> CheckOutcome:
    if not any(_has_route(context.app, path) for path in _REQUEST_TARGET_ROUTES):
        return _not_applicable("request_egress_disabled")
    return _not_implemented()


def _check_grpc_transport(context: SecurityCheckContext) -> CheckOutcome:
    if context.grpc_address is None:
        return _not_applicable("grpc_disabled")
    if is_loopback_host(_grpc_host(context.grpc_address)):
        return _pass("grpc_loopback_listener")
    return _not_implemented()


def _check_grpc_auth(context: SecurityCheckContext) -> CheckOutcome:
    if context.grpc_address is None:
        return _not_applicable("grpc_disabled")
    return _not_implemented()


_REQUIRED = CheckLevel.REQUIRED
_ADVISORY = CheckLevel.ADVISORY

for _check in (
    SecurityCheck(
        "listener.transport",
        SecurityControl.LISTENER,
        _ADVISORY,
        _check_listener_transport,
    ),
    SecurityCheck(
        "auth.enabled", SecurityControl.AUTHENTICATION, _REQUIRED, _check_auth_enabled
    ),
    SecurityCheck(
        "auth.credentials",
        SecurityControl.AUTHENTICATION,
        _REQUIRED,
        _check_auth_credentials,
    ),
    SecurityCheck(
        "auth.jwt_claims", SecurityControl.AUTHENTICATION, _REQUIRED, _check_jwt_claims
    ),
    SecurityCheck(
        "auth.deny_by_default",
        SecurityControl.AUTHORIZATION,
        _REQUIRED,
        _check_auth_deny_by_default,
    ),
    SecurityCheck(
        "auth.route_policy",
        SecurityControl.AUTHORIZATION,
        _REQUIRED,
        _check_route_policy,
    ),
    SecurityCheck("http.cors", SecurityControl.HTTP, _REQUIRED, _check_cors),
    SecurityCheck(
        "http.trusted_hosts", SecurityControl.HTTP, _REQUIRED, _check_trusted_hosts
    ),
    SecurityCheck(
        "http.api_docs",
        SecurityControl.HTTP,
        _ADVISORY,
        _check_api_docs,
        tracking_issue=3911,
    ),
    SecurityCheck(
        "limits.request_body", SecurityControl.LIMITS, _REQUIRED, _check_request_body
    ),
    SecurityCheck(
        "limits.admission", SecurityControl.LIMITS, _ADVISORY, _check_admission
    ),
    SecurityCheck(
        "models.served_allowlist",
        SecurityControl.MODELS,
        _ADVISORY,
        _check_served_models,
    ),
    SecurityCheck(
        "graphql.transport",
        SecurityControl.GRAPHQL,
        _ADVISORY,
        _check_graphql_mounted,
        tracking_issue=3544,
    ),
    SecurityCheck(
        "graphql.limits",
        SecurityControl.GRAPHQL,
        _ADVISORY,
        _check_graphql_mounted,
        tracking_issue=3545,
    ),
    SecurityCheck(
        "journey.principal_binding",
        SecurityControl.JOURNEY,
        _REQUIRED,
        _check_journey_principal_binding,
        tracking_issue=3549,
    ),
    SecurityCheck(
        "journey.encryption_at_rest",
        SecurityControl.JOURNEY,
        _REQUIRED,
        _check_journey_encryption,
        tracking_issue=3550,
    ),
    SecurityCheck(
        "journey.store_tls",
        SecurityControl.JOURNEY,
        _REQUIRED,
        _check_journey_store_tls,
    ),
    SecurityCheck(
        "egress.configured_adapters",
        SecurityControl.EGRESS,
        _REQUIRED,
        _check_configured_egress,
        tracking_issue=3581,
    ),
    SecurityCheck(
        "egress.request_targets",
        SecurityControl.EGRESS,
        _ADVISORY,
        _check_request_egress,
        tracking_issue=3581,
    ),
    SecurityCheck(
        "grpc.transport",
        SecurityControl.GRPC,
        _REQUIRED,
        _check_grpc_transport,
        tracking_issue=3546,
    ),
    SecurityCheck(
        "grpc.auth",
        SecurityControl.GRPC,
        _REQUIRED,
        _check_grpc_auth,
        tracking_issue=3547,
    ),
):
    register_security_check(_check)
del _check


__all__ = [
    "CONFIG_PROFILE_ENV_VAR",
    "DEVELOPMENT_CONFIG_PROFILES",
    "DEV_PROFILE",
    "EXIT_PROFILE_INVALID",
    "EXIT_PROFILE_PASSED",
    "EXIT_PROFILE_UNAVAILABLE",
    "LISTEN_HOST_ENV_VAR",
    "LISTEN_TLS_ENV_VAR",
    "MAX_PRODUCTION_JWT_LIFETIME_SECONDS",
    "PROD_PROFILE",
    "SECURITY_PROFILES",
    "SECURITY_PROFILE_ENV_VAR",
    "SECURITY_PROFILE_INVALID_CODE",
    "SECURITY_PROFILE_REPORT_SCHEMA_VERSION",
    "CheckLevel",
    "CheckOutcome",
    "CheckResult",
    "CheckStatus",
    "ListenerFacts",
    "ProfileSource",
    "ReportStatus",
    "ResolvedSecurityProfile",
    "SecurityCheck",
    "SecurityCheckContext",
    "SecurityControl",
    "SecurityProfile",
    "SecurityProfileName",
    "SecurityProfileReport",
    "evaluate_security_checks",
    "evaluate_service_security_profile",
    "is_loopback_host",
    "listener_facts_from_env",
    "log_security_profile_report",
    "register_security_check",
    "registered_security_checks",
    "resolve_security_profile",
    "run_service_preflight",
    "unregister_security_check",
]
